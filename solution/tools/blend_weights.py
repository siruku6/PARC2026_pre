#!/usr/bin/env python3
"""素の重みと追加学習後の重みを λ の割合で混ぜ、提出できる形で書き出す。

**追加学習が本番スコアを下げたときの対処である。** LoRA をマージした重みは
「素の重み ＋ 学習で動いた分」という形をしているので、

    θ_学習後   = θ素 + Δ
    θ_混ぜたもの = θ素 + λ·Δ = (1−λ)·θ素 + λ·θ学習後

**λ を下げると「学習で動いた分」を弱い強さで当てられる。** 学習をやり直す
必要はなく、テンソルを 1 つずつ足すだけで済む。WiSE-FT として論文になっている。

4 回目の提出（λ=1.0 にあたる）は本番 0.016 で、3 回目の素の重み 0.1409 から
8.8 倍悪化した。**手元の 84 件の検証用データでは成功率に差が無かったが、
グリッパの開閉が 20 回を超える回が 13% から 27% へ増えていた。**
λ を下げるとこの荒れも同じ割合で薄まる。

    python3 solution/tools/blend_weights.py --lam 0.25 \\
        --base      work/artifacts/model_weights.bak \\
        --finetuned work/artifacts/model_weights \\
        --output    work/artifacts/model_weights_lam025

**λ=0.0 が素の重みと、λ=1.0 が学習後と 1 バイトも違わないことを必ず確かめる。**
BF16 の丸めを間違えると、静かに壊れた重みが出てくる。

    python3 solution/tools/blend_weights.py --lam 1.0 ... --output /tmp/lam100
    shasum -a 256 /tmp/lam100/model.safetensors <学習後>/model.safetensors

Reads:
  - <base>/model.safetensors
  - <finetuned>/**（`model.safetensors` 以外をそのまま複製する）
Generates:
  - <output>/model.safetensors（混ぜたもの）
  - <output>/**（設定・前処理・`vlm/` は `--finetuned` から複製）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# numpy はこのリポジトリの venv に入っていないことがある。`work/pylibs` へ
# 入れてある場合はそこから読む。**入っていなければ直し方を出して止める。**
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_LIBS = _REPO_ROOT / "work" / "pylibs"
if _VENDORED_LIBS.is_dir():
    sys.path.insert(0, str(_VENDORED_LIBS))

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - 環境依存
    raise SystemExit(
        "numpy が無い。次のどちらかで入れてから実行すること:\n"
        "  python3 -m pip install --target work/pylibs numpy\n"
        "  python3 -m venv /tmp/blend && /tmp/blend/bin/python -m pip install numpy"
    )

_WEIGHTS_NAME = "model.safetensors"

# safetensors の dtype 名 → numpy の型。**この 2 つ以外は扱わない。**
# 実物（SmolVLA + LoRA マージ済み）は BF16 474 個 ＋ F32 26 個だった。
_SUPPORTED_DTYPES = {"F32": np.dtype("<f4"), "BF16": np.dtype("<u2")}

_EXCLUDE_PARTS = frozenset({"__pycache__", ".ipynb_checkpoints", ".DS_Store"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lam", type=float, required=True,
                        help="学習で動いた分を当てる強さ（0.0 で素の重み、1.0 で学習後）")
    parser.add_argument("--base", type=Path, required=True,
                        help="素の重みのディレクトリ")
    parser.add_argument("--finetuned", type=Path, required=True,
                        help="追加学習後（マージ済み）のディレクトリ")
    parser.add_argument("--output", type=Path, required=True,
                        help="書き出し先。既にあれば消してから作る")
    return parser.parse_args()


def _read_header(path: Path) -> tuple[dict, int]:
    """safetensors のヘッダと、本体が始まるバイト位置を返す。

    先頭 8 バイトがヘッダ長（リトルエンディアン）、続く区間が JSON という形式。
    900MB を読み込まずに構成を確認できる。

    Parameters
    ----------
    path : Path
        safetensors ファイル。

    Returns
    -------
    header : dict
        テンソル名 → `dtype` / `shape` / `data_offsets`。`__metadata__` を含む。
    body_start : int
        本体の先頭バイト位置（8 ＋ ヘッダ長）。
    """
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if not 0 < header_size < (1 << 30):
            raise ValueError(f"safetensors のヘッダ長が異常: {path} ({header_size})")
        header = json.loads(stream.read(header_size))
    return header, 8 + header_size


def _check_compatible(base_header: dict, other_header: dict) -> list[str]:
    """混ぜられる構成かを確かめ、テンソル名を並び順つきで返す。

    **1 つでも食い違えば混ぜてはいけない。** 形が違うものを足すと、
    静かに壊れた重みが出てくる。

    Parameters
    ----------
    base_header, other_header : dict
        `_read_header` の結果。

    Returns
    -------
    names : list[str]
        本体での位置が早い順に並べたテンソル名。
    """
    base_names = {name for name in base_header if name != "__metadata__"}
    other_names = {name for name in other_header if name != "__metadata__"}

    if base_names != other_names:
        added = sorted(other_names - base_names)[:3]
        removed = sorted(base_names - other_names)[:3]
        raise ValueError(
            f"テンソルの構成が違う（学習後にだけある {len(other_names - base_names)} 個 "
            f"{added} / 素の重みにだけある {len(base_names - other_names)} 個 {removed}）"
        )

    unsupported = sorted(
        {base_header[name]["dtype"] for name in base_names} - set(_SUPPORTED_DTYPES)
    )
    if unsupported:
        raise ValueError(f"扱えない dtype がある: {unsupported}")

    mismatched = [
        name for name in base_names
        if base_header[name]["dtype"] != other_header[name]["dtype"]
        or base_header[name]["shape"] != other_header[name]["shape"]
    ]
    if mismatched:
        raise ValueError(
            f"dtype か shape が食い違うテンソルが {len(mismatched)} 個: {mismatched[:3]}"
        )

    return sorted(base_names, key=lambda name: base_header[name]["data_offsets"][0])


def _bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    """BF16（16 ビット）を float32 へ広げる。

    **BF16 は float32 の上位 16 ビットそのものである。** だから 16 ビット
    左へずらして float32 として見るだけでよい。numpy に BF16 は無いので
    この経路が必要になる。
    """
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _float32_to_bf16(values: np.ndarray) -> np.ndarray:
    """float32 を BF16（16 ビット）へ戻す。近い側へ丸める。

    **切り捨てではなく丸める。** 下位 16 ビットの最上位（ビット 15）が
    立っていれば 1 を足す。

    λ=0.0 と λ=1.0 のとき元のビット列へ完全に戻ることが要る。そのときの
    値は BF16 を広げたものなので**下位 16 ビットが必ず 0** であり、
    丸めの加算は 0 になる。**つまり往復で 1 ビットも変わらない。**
    """
    bits = values.view(np.uint32)
    rounded = (bits >> 16) + ((bits >> 15) & np.uint32(1))
    # NaN / Inf の境目で 0x10000 へ溢れることがある。uint16 へ落とすと 0 に
    # 化けてしまうので、その前に止める。
    return np.minimum(rounded, np.uint32(0xFFFF)).astype(np.uint16)


def _blend_tensor(dtype_name: str, base_raw: np.ndarray, other_raw: np.ndarray,
                  lam: float) -> np.ndarray:
    """1 つのテンソルを混ぜて、書き出す形のバイト列で返す。"""
    if dtype_name == "F32":
        blended = (1.0 - lam) * base_raw + lam * other_raw
        return blended.astype(np.dtype("<f4"))

    blended = (1.0 - lam) * _bf16_to_float32(base_raw) + lam * _bf16_to_float32(other_raw)
    return _float32_to_bf16(blended)


def _copy_metadata(finetuned: Path, output: Path) -> int:
    """重み以外のファイルを複製し、複製した数を返す。

    **`--finetuned` から複製する。** そちらは 4 回目の提出として本番で動いた
    ディレクトリそのもので、`config.json` の `pretrained_path` が `None` に
    なっている（素の重み側は `lerobot/smolvla_base` を指しており、提出物側で
    解決できないパスになりうる）。`vlm/` もここから引き継ぐ。
    """
    copied = 0
    for path in sorted(finetuned.rglob("*")):
        if not path.is_file() or _EXCLUDE_PARTS & set(path.parts):
            continue
        if path.name == _WEIGHTS_NAME:
            continue
        target = output / path.relative_to(finetuned)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def _write_blended(base_path: Path, other_path: Path, output_path: Path,
                   lam: float) -> tuple[int, float]:
    """混ぜた重みを書き出し、(テンソル数, いちばん大きい相対変化) を返す。

    **素の重みのヘッダをそのまま使う。** バイトの並びを変えないので、
    λ=0.0 のときファイル全体が元と一致する。

    Parameters
    ----------
    base_path, other_path : Path
        それぞれの `model.safetensors`。
    output_path : Path
        書き出す `model.safetensors`。
    lam : float
        学習で動いた分を当てる強さ。

    Returns
    -------
    count : int
        書き出したテンソル数。
    largest_change : float
        素の重みからの相対変化がいちばん大きかった値。**0 なら混ざっていない。**
    """
    base_header, base_body = _read_header(base_path)
    other_header, other_body = _read_header(other_path)
    names = _check_compatible(base_header, other_header)

    # **ヘッダは素の重みのバイト列をそのまま使う。** 自分で JSON を組み直すと
    # 鍵の順や詰め物の長さが変わり、λ=0.0 でもファイルが一致しなくなる。
    with base_path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        header_bytes = stream.read(header_size)

    largest_change = 0.0
    with base_path.open("rb") as base_stream, \
            other_path.open("rb") as other_stream, \
            output_path.open("wb") as out_stream:
        out_stream.write(len(header_bytes).to_bytes(8, "little"))
        out_stream.write(header_bytes)

        for name in names:
            entry = base_header[name]
            start, end = entry["data_offsets"]
            numpy_dtype = _SUPPORTED_DTYPES[entry["dtype"]]

            base_stream.seek(base_body + start)
            base_raw = np.frombuffer(base_stream.read(end - start), dtype=numpy_dtype)

            other_start, other_end = other_header[name]["data_offsets"]
            other_stream.seek(other_body + other_start)
            other_raw = np.frombuffer(
                other_stream.read(other_end - other_start), dtype=numpy_dtype
            )

            blended = _blend_tensor(entry["dtype"], base_raw, other_raw, lam)
            out_stream.write(blended.tobytes())

            # 変化の大きさを見る。**0 のままなら混ざっていない。**
            if entry["dtype"] == "BF16":
                before = _bf16_to_float32(base_raw)
                after = _bf16_to_float32(blended)
            else:
                before, after = base_raw, blended
            scale = float(np.abs(before).max())
            if scale > 0.0:
                change = float(np.abs(after - before).max()) / scale
                largest_change = max(largest_change, change)

    return len(names), largest_change


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.lam <= 1.0:
        raise SystemExit(f"--lam は 0.0 以上 1.0 以下で渡すこと（受け取った値 {args.lam}）")

    base: Path = args.base.resolve()
    finetuned: Path = args.finetuned.resolve()
    output: Path = args.output.resolve()

    for label, directory in (("--base", base), ("--finetuned", finetuned)):
        if not (directory / _WEIGHTS_NAME).is_file():
            raise SystemExit(f"{label} に {_WEIGHTS_NAME} が無い: {directory}")

    print(f"[blend] λ = {args.lam}")
    print(f"[blend] 素の重み   {base}")
    print(f"[blend] 学習後     {finetuned}")
    print(f"[blend] 書き出し先 {output}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    count, largest_change = _write_blended(
        base / _WEIGHTS_NAME, finetuned / _WEIGHTS_NAME, output / _WEIGHTS_NAME, args.lam
    )
    copied = _copy_metadata(finetuned, output)

    total_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(f"[blend] テンソル {count} 個を混ぜ、他の {copied} ファイルを複製した")
    print(f"[blend] 素の重みからの相対変化 いちばん大きいもの {largest_change:.3e}")
    if args.lam > 0.0 and largest_change == 0.0:
        raise SystemExit(
            "**1 つも変わっていない。** 2 つの重みが同じか、混ぜ方が壊れている"
        )
    print(f"[blend] 完了 {total_bytes / 1e6:.0f} MB")
    print("[blend] 次に: solution/tools/install_weights.py で取り込む")


if __name__ == "__main__":
    main()
