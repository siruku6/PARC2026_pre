"""量子化スイープ・推論時候補 C4: bitsandbytes の 4bit 量子化を、読み込み済みの
π0.5 に後掛けする（policy_server.py 専用。work/quant_sweep/CANDIDATES.md の C4）。

**学習側の候補 C2（solution/train/quant_vision_4bit.py。学習前に量子化する）との
違いは順番だけである。** こちらは policy_server.py の `_build_pi05` が
safetensors から全パラメータを写し終えた「フル精度の完成品」に対して、
サービング直前に量子化する。逆伝播を考えなくてよい分、学習側より制約が緩い
（optimizer の "same dtype" 要求や勾配の量子化誤差を気にしなくてよい）。

対象範囲・危険性は quant_serve_common.py のモジュール docstring と共通なので
そちらを参照。既定 (`QUANT_SERVE_SCOPE=vision_only`) は vision_tower +
multi_modal_projector だけで、modeling_pi05.py の `compute_layer_complete`
（236-305行）が触らない安全な範囲。

**`QUANT_SERVE_SCOPE=full` にしたときの壊れ方（bnb 版）:**
`compute_layer_complete` は `layer.self_attn.o_proj.weight.dtype` /
`layer.mlp.up_proj.weight.dtype` を素の `torch.dtype`（bf16/fp32 のどちらか）
だという前提で `==`/`!=` 比較している。bnb で量子化した `Linear4bit.weight`
（`Params4bit`）は、量子化後は 4bit にパックされた `torch.uint8` の実体を
`.data` に持つため、`.dtype` は `torch.uint8` を返す。つまり
`att_output.dtype != layer.self_attn.o_proj.weight.dtype` は bf16/fp32 の
入力に対して**常に True** になり、`att_output` が `uint8` へ誤ってキャストされた
状態で `o_proj`（= `bnb.matmul_4bit`）に渡る——`bnb.matmul_4bit` は浮動小数点の
入力を要求するため、ここで例外になる見込みが高い（実測は要 GPU。
policy_server.py 側は例外をそのまま伝播させ、隠さない）。

policy_server.py からは `QUANT_SERVE_MODE=bnb_vision4bit` のときだけ import され、
`apply(policy, device)` を呼ぶだけの副作用フリーなモジュールである
（import 自体は bitsandbytes 未インストールでも失敗しない。実際に使うときだけ
`apply()` 内で遅延 import する。理由は quant_optim.py と同じ）。
"""
import torch.nn as nn

import quant_serve_common as _common

# bnb.nn.Linear4bit 自身の既定は "fp4" だが、QLoRA
# （https://arxiv.org/abs/2305.14314）が言語モデルの重みに対して fp4 より精度が
# 落ちにくいと報告している "nf4" を選ぶ。学習ではなく凍結重みの推論用途なので、
# 量子化そのものによる劣化を最小にする側を優先した。
_QUANT_TYPE = "nf4"


def _replace_linears_with_bnb4bit(root: nn.Module, device: str) -> int:
    """root の中の nn.Linear をすべて bnb.nn.Linear4bit に置き換える（in-place）。

    **手順の順番が重要。** bitsandbytes 0.49.2 の `Linear4bit` クラス docstring
    （`bitsandbytes/nn/modules.py`）が例示している手順そのものを踏む。

    1. 空の `Linear4bit` を作る（CPU 上。この時点ではまだ量子化されない）
    2. 元の `nn.Linear` の `state_dict()`（フル精度の重み・bias）をそのまま
       `load_state_dict()` で読ませる
    3. 目的の device へ `.to(device)` した瞬間に、`Params4bit.to()` の中の
       `if device is not None and not self.bnb_quantized: return self._quantize(device)`
       が実際の 4bit 量子化を行う

    `Params4bit.from_prequantized` は「すでに量子化済みの統計量（quant_state）を
    読み込む」側の API であり、ここでは使わない —— 読み込んだのは
    `_build_pi05` が safetensors から写した**フル精度**の重みだからである。

    Parameters
    ----------
    root : nn.Module
        置き換え対象の部分木（例: vision_tower）。
    device : str
        量子化の実行 device。**bnb の 4bit カーネルは CUDA 専用**なので、
        呼び出し側（`apply`）で "cuda" であることを確認してから呼ぶこと。

    Returns
    -------
    count : int
        置き換えた nn.Linear の数。
    """
    import bitsandbytes as bnb  # 遅延 import。理由はモジュール docstring 参照

    count = 0
    for name, child in list(root.named_children()):
        if isinstance(child, nn.Linear):
            quantized = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                quant_type=_QUANT_TYPE,
            )
            quantized.load_state_dict(child.state_dict())
            quantized = quantized.to(device)  # ここで実際に 4bit 量子化される
            setattr(root, name, quantized)
            count += 1
        else:
            count += _replace_linears_with_bnb4bit(child, device)
    return count


def apply(policy, device: str) -> None:
    """読み込み済みの π0.5 方策へ、bnb の 4bit 量子化を後掛けする。

    呼び出し側（policy_server.py の `MyPolicy.__init__`）が
    `self._policy.to(self._device)` を呼ぶ**前**に呼ぶ想定。ここで対象部分木を
    直接 `device` へ移して量子化してしまうので、その後の全体 `.to(device)` は
    対象外の部分（`QUANT_SERVE_SCOPE=vision_only` なら gemma_expert 等）だけを
    動かすことになる。量子化済みの `Params4bit` に対して `.to()` を再度呼んでも、
    `bnb_quantized=True` の分岐（quant_state を付け替えるだけ）に入るので壊れない。

    Parameters
    ----------
    policy : PI05Policy
        `_build_pi05` が読み終えた（フル精度の）方策。
    device : str
        最終的な推論 device。**"cuda" 以外は明示的に落とす**
        （下の Raises 参照）。

    Raises
    ------
    ImportError
        `bitsandbytes` が import できない場合
        （Dockerfile.pi05 が `bitsandbytes==0.49.2` を入れている前提）。
    RuntimeError
        `device != "cuda"`（bnb の 4bit カーネルは CUDA 専用。CPU のまま
        `_quantize()` を呼ぶと `bnb.functional.quantize_4bit` が CUDA カーネルを
        要求して失敗するため、意味の無い遠回りをさせず先に落とす）、
        または対象部分木に nn.Linear が 1 つも無かった場合
        （属性パスは通っても中身が期待と違う可能性がある）。
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "QUANT_SERVE_MODE=bnb_vision4bit が指定されたが bitsandbytes が"
            " import できない。Dockerfile.pi05 が bitsandbytes==0.49.2 を"
            " 入れているはずのイメージ（parc2026-train-pi05 由来）で"
            " 動かしているか確認すること。"
        ) from error

    if device != "cuda":
        raise RuntimeError(
            f"bnb の 4bit 量子化は CUDA 専用の実装である。device={device!r} では"
            " 動かせない（GPU が見えないコンテナ・ホストで起動していないか"
            " 確認すること）。"
        )

    targets = _common.target_modules(policy)
    total = 0
    for label, module in targets.items():
        expected = _common.count_linears(module)
        if expected == 0:
            raise RuntimeError(
                f"{label} の中に nn.Linear が 1 つも無い。属性パスは通っているが"
                " 中身が期待と違う可能性がある（別モデルのチェックポイントを"
                " 読んでいる等）。"
            )
        replaced = _replace_linears_with_bnb4bit(module, device)
        print(
            f"[parc][quant_serve_bnb] {label}: Linear {replaced} 個を"
            f" {_QUANT_TYPE} で 4bit 化した", flush=True,
        )
        total += replaced

    print(
        f"[parc][quant_serve_bnb] scope={_common.scope()}  合計 {total} 個の"
        " Linear を 4bit 化した", flush=True,
    )
