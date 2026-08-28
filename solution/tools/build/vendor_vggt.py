#!/usr/bin/env python
"""VGGT のソースを提出物へ同梱する（幾何の接ぎ木・板 F25）。

**なぜ同梱するのか。** 採点環境で `pip install` できるとは限らず、VGGT は PyPI にも
無い（`pip install vggt` は「No matching distribution found」になる。2026-08-28 に確認）。
`vendor_lerobot.py` と同じ考えで、**ソースを `vendor/` に置いて `sys.path` から読む。**

**入れるのは推論に要る分だけである。** `vggt/models/vggt.py` から辿れる 19 ファイルの
外部依存は `torch`・`numpy`・`huggingface_hub` の 3 つだけで、これは採点イメージに
入っている（2026-08-28 に import を辿って確認）。`dependency/`（pycolmap・lightglue
などを使う三次元復元の道具）は推論経路から呼ばれないが、**消すと将来の版で
壊れかねないのでそのまま置く**——全部で 360 KB しかない。

**`__init__.py` を持たない名前空間パッケージである。** だから `vggt.__file__` は
`None` になる。どちらの `vggt` を読んでいるかを確かめたいときは
`vggt.models.vggt.__file__` を見ること。

使い方:

    # 1. 取ってくる（学習コンテナの中に既にあるならそこから写すのが速い）
    pip install --target /tmp/vggt-src "git+https://github.com/facebookresearch/vggt.git"
    python3 solution/tools/build/vendor_vggt.py --source /tmp/vggt-src/vggt

    # 2. 確かめる
    python3 solution/tools/build/vendor_vggt.py --check
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DESTINATION = _REPO_ROOT / "work" / "artifacts" / "vendor" / "vggt"

# これが揃っていなければ推論側が組めない。
_REQUIRED = (
    "models/vggt.py",
    "models/aggregator.py",
    "layers/patch_embed.py",
    "heads/camera_head.py",
)


def vendor(source: Path, destination: Path) -> None:
    """`source`（vggt パッケージのディレクトリ）を `destination` へ写す。"""
    if not (source / "models" / "vggt.py").is_file():
        raise SystemExit(f"vggt パッケージに見えない（models/vggt.py が無い）: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source, destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    size_kb = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file()) // 1024
    print(f"同梱した: {destination}（{size_kb} KB）")


def check(destination: Path) -> None:
    """同梱物が揃っているかを確かめる。**足りなければ 0 点になる前にここで落とす。**"""
    missing = [name for name in _REQUIRED if not (destination / name).is_file()]
    if missing:
        raise SystemExit(f"同梱が足りない: {missing}（{destination}）")
    print(f"揃っている: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=None,
                        help="vggt パッケージのディレクトリ（pip install --target の中の vggt/）")
    parser.add_argument("--destination", type=Path, default=_DEFAULT_DESTINATION,
                        help=f"同梱先（既定: {_DEFAULT_DESTINATION}）")
    parser.add_argument("--check", action="store_true", help="同梱物が揃っているかだけ見る")
    args = parser.parse_args()

    if args.check:
        check(args.destination)
        return
    if args.source is None:
        parser.error("--source か --check のどちらかが要る")
    vendor(args.source, args.destination)
    check(args.destination)


if __name__ == "__main__":
    sys.exit(main())
