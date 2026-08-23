"""量子化スイープ・推論時候補 C4 (bitsandbytes) / C5 (torchao) が共有する下ごしらえ。

policy_server.py の π0.5 経路（`_build_pi05` がフル精度で読み終えたあとの
`PI05Policy`）から、量子化の対象にする部分木を名前つきで取り出すだけの薄いヘルパー。
実際に「どう量子化するか」（bnb の `Linear4bit` / torchao の `quantize_`）は
呼び出し側の quant_serve_bnb.py / quant_serve_torchao.py がそれぞれ持つ。

**このファイル自身は bitsandbytes も torchao も import しない。** どちらの候補が
選ばれても（あるいは両方とも未選択でも）ここは無条件に import されるので、
量子化ライブラリへの依存はここに持ち込まない。

対象範囲は `QUANT_SERVE_SCOPE`（既定 `vision_only`）で 2 段階に分かれる。

- `vision_only`（既定）: `vision_tower` + `multi_modal_projector` だけ。
  modeling_pi05.py の `embed_image`（446-455行）しか通らない経路で、そこは
  `.weight.dtype` を直接見て分岐するコードを持たない。安全な範囲。
- `full`: 上に加えて `gemma_expert` と paligemma の言語側 Transformer 層
  （`language_model.layers`）も対象にする。**ここは同ファイルの
  `compute_layer_complete`（236-305行）が `layer.self_attn.o_proj.weight.dtype` /
  `layer.mlp.up_proj.weight.dtype` を直接見て分岐するコードに正面から当たる、
  「壊れるかどうかも含めて試す」経路である。** どう壊れうるかは
  quant_serve_bnb.py / quant_serve_torchao.py の docstring を参照。
  `compute_layer_complete` 自体は直さない（要 lerobot 本体の変更で、ここでは対象外。
  work/lerobot-src/ は参照専用で、採点イメージのコンテナ内は git clone した
  素の v0.6.0 のままだから直しても効かない）。

想定パス（2026-08-20、v0.6.0 の modeling_pi05.py / pi_gemma.py で確認済み）:

    policy.model                                        PI05Pytorch
      .paligemma_with_expert                             PaliGemmaWithExpertModel
        .paligemma.model.vision_tower                     SigLIP 画像エンコーダ
        .paligemma.model.multi_modal_projector             画像→言語次元への射影
        .paligemma.model.language_model.layers[*]          paligemma 側 Gemma 層
        .gemma_expert.model.layers[*]                      action expert 側 Gemma 層
"""
import os

import torch.nn as nn

_VALID_SCOPES = ("vision_only", "full")


def scope() -> str:
    """`QUANT_SERVE_SCOPE` を読んで検証する（既定 `vision_only`）。"""
    value = os.environ.get("QUANT_SERVE_SCOPE", "vision_only")
    if value not in _VALID_SCOPES:
        raise ValueError(
            f"QUANT_SERVE_SCOPE の値が不明: {value!r}"
            f"（{' / '.join(_VALID_SCOPES)} のいずれか）"
        )
    return value


def target_modules(policy) -> dict[str, nn.Module]:
    """量子化する部分木を、ログ表示用の名前つきで返す。

    Parameters
    ----------
    policy : PI05Policy
        `_build_pi05`（policy_server.py）がフル精度で読み終えた π0.5 の方策。

    Returns
    -------
    targets : dict[str, nn.Module]
        キーはログ表示用のラベル、値は量子化対象の `nn.Module`。
        `QUANT_SERVE_SCOPE=full` のときだけ `gemma_expert` /
        `paligemma_language_model` が追加される。

    Raises
    ------
    AttributeError
        期待する属性パスが無い場合（lerobot 側の構造がここに書いた版と
        変わっている等）。黙って何もしないより、ここで気づけたほうがよい
        ——このプロジェクトの量子化候補はどれも「fail loudly」を方針にしている
        （solution/train/experiments/quantization/ の各候補と同じ姿勢）。
    """
    try:
        expert_model = policy.model.paligemma_with_expert
        paligemma_model = expert_model.paligemma.model
    except AttributeError as error:
        raise AttributeError(
            "π0.5 の想定パス policy.model.paligemma_with_expert.paligemma.model が"
            "見つからない。modeling_pi05.py（PI05Pytorch / PaliGemmaWithExpertModel）"
            "の構造が変わっていないか確認すること"
            "（quant_serve_common.py のモジュール docstring に想定パスを書いてある）。"
        ) from error

    targets: dict[str, nn.Module] = {
        "vision_tower": paligemma_model.vision_tower,
        "multi_modal_projector": paligemma_model.multi_modal_projector,
    }

    if scope() == "full":
        # **ここから先は compute_layer_complete の .weight.dtype 分岐に当たる、
        # 「壊れるかどうかも含めて試す」経路。** 上のモジュール docstring 参照。
        targets["paligemma_language_model"] = paligemma_model.language_model
        targets["gemma_expert"] = expert_model.gemma_expert

    return targets


def count_linears(module: nn.Module) -> int:
    """module の中（自分自身を含む）にある nn.Linear の数を数える。"""
    return sum(1 for candidate in module.modules() if isinstance(candidate, nn.Linear))
