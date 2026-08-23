#!/usr/bin/env python
"""C2（bnb 4bit vision encoder、work/quant_sweep/CANDIDATES.md）のチェックポイントを、
ふつうの `nn.Linear` 形式へ書き戻す。

## なぜこれが要るか

`quant_vision_4bit.py` は学習中、vision_tower・multi_modal_projector の `nn.Linear` を
`bnb.nn.Linear4bit` に差し替える。保存されたチェックポイントの該当する重みは、平の
`(out_features, in_features)` テンソルではなく、4bit にパックされた `(N, 1) uint8` の
`weight` と、それに付随する `weight.absmax` / `weight.quant_map` / `weight.nested_absmax` /
`weight.nested_quant_map` / `weight.quant_state.bitsandbytes__fp4` という5つの付帯テンソルの
組で保存されている（`bnb.nn.Linear4bit` が `state_dict()` を作るときに自動でこの形へ展開する。
2026-08-22、実機で `bnb.nn.Linear4bit.state_dict()` を直接呼んで確認済み）。

**評価に使う `policy_server.py` の `_build_pi05` は、この形を読めない。** `_build_pi05` は
`load_state_dict()` を経由せず、`model.safetensors` の各テンソルを構築済みモデルの
同名パラメータへ形を突き合わせながら直接 `target.data = tensor` で写す独自の軽量ローダー
（RAM 節約のため。同ファイルの docstring 参照）なので、`(2048, 1152)` を期待している場所に
`(1179648, 1)` が来て「重みの形が合わない」で落ちる（2026-08-21 の回収評価で実際に発生）。

## dtype は vision_tower・multi_modal_projector 全体を bf16 に揃える

**「量子化されていない部分は元の dtype（大半 fp32）のまま残す」を最初に試したが、
実機で3回落ちた。** `patch_embedding`・`q_proj`/`k_proj`/`v_proj` は fp32 のまま
残しても通ったが、`self_attn.out_proj` だけは実行するたびに要求される dtype が
入れ替わるように見えた（fp32 の重みでは「入力が bf16」、bf16 の重みでは
「入力が fp32」という食い違いが両方向で観測された——SigLIP の attention 実装
（SDPA のバックエンド選択、または内部の暗黙キャスト）が絡んでいると見ているが、
正確な原因は特定できていない）。**個々の層ごとに「本来の dtype」を当てにいくのは
もう一つも当たらないやり方だと判断し、この対症療法はやめた。**

代わりに、**vision_tower + multi_modal_projector 全体（量子化されていた層も、
patch_embedding・layer_norm のように量子化されていなかった層も含めて全部）を
bf16 に統一する。** 内部で dtype が揃っていれば、個々の層がどちらの dtype を
「本来」要求するかを当てる必要がなくなる——学習時の `compute_dtype=torch.bfloat16`
（`quant_vision_4bit.py`）と揃えているので、量子化された層にとっては元々の
計算精度と同じである。fp32 のまま残っていた層（patch_embedding・layer_norm 等）
だけ、追加で bf16 への丸めが増える——C2 という候補自体が「凍結側の精度劣化を
受け入れて学習を軽くする」ものなので、この追加の丸めも同じ性質の近似として
許容できると判断した。

## この道具がやること・やらないこと

- **C2 は「学習を軽くする」ための候補**（凍結側の重みメモリだけを 4bit 化。演算量は
  変えていない）で、量子化した重みを**そのまま本番で提供する**ことは元から想定していない
  （`bnb.functional.dequantize_4bit` で戻せば学習中に実際に使われていた値をほぼそのまま
  復元できる——4bit の丸め誤差ぶんの差はあるが、それは C2 の学習自体が最初から
  抱えている誤差そのものである）。
- **したがって「評価用に一度だけ bf16 に戻す」のが最も安全な対応であり、
  `policy_server.py`/`_build_pi05` 側は一切変更しない**（他の全候補が通っている
  実績のあるロード経路を、C2 のためだけに複雑にしないため）。
- vision_tower・multi_modal_projector 以外のキー（gemma_2b・gemma_expert・
  埋め込み・norm など）はそのままコピーする。`config.json` 等の付随ファイル、
  `vlm/` もまとめてコピーする。

## 使い方（GPU が要る。`bnb.functional.dequantize_4bit` が CUDA カーネルのため）

    docker exec quant_train python /opt/pi05/dequantize_bnb4bit_checkpoint.py \\
        /out/pi05_quant_c2/checkpoints/016000/pretrained_model \\
        /out/pi05_quant_c2/checkpoints/016000/pretrained_model_dequant

出力先を `run_docker_eval.sh` の `WEIGHTS=` にそのまま渡せば、他の候補と同じ経路で
評価できる（`vlm/` も一緒にコピーされているので、別途複製する必要は無い）。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import Params4bit
except ImportError as e:
    raise RuntimeError(
        "dequantize_bnb4bit_checkpoint: bitsandbytes が import できません。"
        "bitsandbytes==0.49.2 が入っているコンテナ（quant_train など）の中で実行すること。"
        f"（元の ImportError: {e}）"
    ) from e

from safetensors.torch import load_file, save_file

# `bnb.nn.Linear4bit.state_dict()` が量子化済みの weight パラメータに付ける付帯キーの接尾辞。
# 2026-08-22、実機で `bnb.nn.Linear4bit(64, 32, compute_dtype=torch.bfloat16).state_dict()`
# を直接呼んで確認済み（quant_vision_4bit.py が使う既定設定と同じ呼び方）。
_QUANT_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.bitsandbytes__fp4",
)


def _find_quantized_bases(keys: list[str]) -> list[str]:
    """付帯キーから、4bit 化された「素の」重みキー名を逆算する。"""
    bases: set[str] = set()
    for key in keys:
        for suffix in _QUANT_SUFFIXES:
            if key.endswith(suffix):
                bases.add(key[: -len(suffix)])
    return sorted(bases)


def _dequantize_one(state: dict[str, torch.Tensor], base: str, device: str) -> torch.Tensor:
    """`base`（4bit にパックされた weight キー）を、平のテンソルへ戻す。

    **戻す先の dtype は、同じ Linear の bias の dtype に合わせる（無ければ fp32）。**
    bias は量子化の対象外なので、常に量子化"前"の値のままチェックポイントに
    残っている——つまり「この Linear が実際にどの dtype で動いていたか」を
    weight より確実に教えてくれる。

    2026-08-22、実機の順位一致チェック（`run_quant_rank_check.sh`）で2段階の
    実機確認があってこの結論に至った。

    1回目（fp32 が正しいと考えた根拠）: `quant_vision_4bit.py` の
       `compute_dtype=torch.bfloat16` は量子化後の計算精度の指定であって、
       量子化前の本来の dtype とは無関係。C0（量子化していない baseline）の
       vision_tower の `nn.Linear` はほぼ全部 fp32 で保存されている
       （`patch_embedding`・`layer_norm` も同様）。→ 全部 fp32 に戻したところ、
       `RuntimeError: ... got BFloat16 and Float` が **out_proj** で発生。

    2回目（fp32 だけでは足りないと分かった根拠）: C2 のチェックポイントを
       実際に調べると、**`self_attn.out_proj.bias` だけが全27層で一貫して
       bf16、他（q_proj/k_proj/v_proj/mlp.fc1/fc2/multi_modal_projector.linear
       の bias）は全部 fp32**（C0/C1 ではこの層も fp32 なので、C2 の学習
       固有の現象——SigLIP の attention 実装が out_proj の入力だけ内部で
       bf16 にキャストしている等が疑われるが、正確な原因は未確定）。
       bias は量子化されないのでこの差は量子化処理自体が作ったものではなく、
       実際にその dtype で計算されていたことの記録だと判断した。
       → weight を bias の dtype に揃える方式に変更。
    """
    satellites = {}
    for suffix in _QUANT_SUFFIXES:
        key = base + suffix
        if key not in state:
            raise RuntimeError(
                f"dequantize_bnb4bit_checkpoint: {base} に付帯キー {key} が無い。"
                "quant_vision_4bit.py の量子化設定（blocksize/quant_type/"
                "compress_statistics）が変わっていないか確認すること。"
            )
        satellites[suffix[1:]] = state.pop(key).to(device)  # 先頭の "." を落として渡す

    data = state.pop(base).to(device)
    params = Params4bit.from_prequantized(data=data, quantized_stats=satellites, device=device)
    dequantized = bnb.functional.dequantize_4bit(params.data, quant_state=params.quant_state)
    return dequantized.to(torch.bfloat16).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="C2 の checkpoints/<step>/pretrained_model")
    parser.add_argument("output_dir", type=Path, help="fp32 に戻した内容を書き出す先")
    parser.add_argument("--device", default="cuda", help="既定 cuda（dequantize_4bit は CUDA 専用）")
    args = parser.parse_args()

    if not (args.input_dir / "model.safetensors").is_file():
        raise SystemExit(f"エラー: {args.input_dir}/model.safetensors が無い")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in args.input_dir.iterdir():
        if item.name == "model.safetensors":
            continue  # これは下で書き換えて別に書き出す
        dest = args.output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    state = load_file(str(args.input_dir / "model.safetensors"))
    bases = _find_quantized_bases(list(state.keys()))
    if not bases:
        raise SystemExit(
            "エラー: 4bit 量子化された重みが1つも見つからなかった。"
            "本当に C2（--vision-quant-mode bnb4bit）のチェックポイントか確認すること。"
        )
    print(f"[dequant] {len(bases)} 個の4bit量子化された重みを見つけた。戻す。")
    for base in bases:
        restored = _dequantize_one(state, base, args.device)
        state[base] = restored
        print(f"[dequant]  {base}: -> {tuple(restored.shape)} {restored.dtype}")

    # vision_tower・multi_modal_projector 全体を bf16 に揃える（量子化されて
    # いなかった patch_embedding・layer_norm 等も含む）。理由はモジュール
    # docstring の「dtype は vision_tower・multi_modal_projector 全体を bf16 に
    # 揃える」を参照——個々の層ごとに dtype を当てにいく方式は実機で3回落ちた。
    cast_count = 0
    for key in list(state.keys()):
        if "vision_tower" in key or "multi_modal_projector" in key:
            if torch.is_floating_point(state[key]) and state[key].dtype != torch.bfloat16:
                state[key] = state[key].to(torch.bfloat16)
                cast_count += 1
    print(f"[dequant] vision_tower/multi_modal_projector のうち、"
          f"元々 bf16 でなかった {cast_count} 個のテンソルも bf16 に統一した。")

    save_file(state, str(args.output_dir / "model.safetensors"))
    print(f"[dequant] 完了: {args.output_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
