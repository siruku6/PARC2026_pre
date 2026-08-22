#!/usr/bin/env python
"""C2（bnb 4bit vision encoder、work/quant_sweep/CANDIDATES.md）のチェックポイントを、
ふつうの bf16 の `nn.Linear` 形式へ書き戻す。

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

## この道具がやること・やらないこと

- **C2 は「学習を軽くする」ための候補**（凍結側の重みメモリだけを 4bit 化。演算量は
  変えていない）で、量子化した重みを**そのまま本番で提供する**ことは元から想定していない
  （量子化 4bit の対象は凍結側だけであり、`bnb.functional.dequantize_4bit` で戻せば学習中に
  実際に使われていた値をほぼそのまま復元できる——4bit の丸め誤差ぶんの差はあるが、
  それは C2 の学習自体が最初から抱えている誤差そのものである）。
- **したがって「評価用に一度だけ bf16 に戻す」のが最も安全な対応であり、
  `policy_server.py`/`_build_pi05` 側は一切変更しない**（他の全候補が通っている
  実績のあるロード経路を、C2 のためだけに複雑にしないため）。
- 4bit 化されていないキー（gemma_2b・gemma_expert・埋め込み・norm など）はそのまま
  コピーする。`config.json` 等の付随ファイル、`vlm/` もまとめてコピーする。

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
    """`base`（4bit にパックされた weight キー）を、平の bf16 テンソルへ戻す。"""
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
    parser.add_argument("output_dir", type=Path, help="bf16 に戻した内容を書き出す先")
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

    save_file(state, str(args.output_dir / "model.safetensors"))
    print(f"[dequant] 完了: {args.output_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
