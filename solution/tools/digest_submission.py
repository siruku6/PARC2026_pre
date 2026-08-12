#!/usr/bin/env python3
"""提出物の内容ダイジェストを出力する。

同じ内容が複数の場所（ローカル・Docker コンテナ・Colab）に散るため、
片方だけを更新してしまう事故が起きる。ダイジェストを突き合わせれば
それを検出できる。zip とディレクトリのどちらも受け付け、同じ内容なら
同じ値になる。

実行はどのディレクトリからでもよい。

Reads:
  - 指定した zip、または指定したディレクトリ配下の全ファイル
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

# solution/tools/digest_submission.py の 2 階層上がリポジトリルート。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TARGET = _REPO_ROOT / "work" / "submission.zip"

_SKIP_PARTS = frozenset({"__pycache__", ".hf_home", ".ipynb_checkpoints", ".DS_Store"})
_SKIP_SUFFIXES = frozenset({".pyc"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=_DEFAULT_TARGET,
        help="zip またはディレクトリ（既定: work/submission.zip）",
    )
    parser.add_argument(
        "--per-entry",
        action="store_true",
        help="トップレベル項目ごとのダイジェストも出力する",
    )
    return parser.parse_args()


def _is_skipped(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    if _SKIP_PARTS & set(parts):
        return True
    return Path(relative_path).suffix in _SKIP_SUFFIXES


def _read_zip(target: Path) -> dict[str, bytes]:
    """zip の中身を {zip 内パス: 内容} で返す。"""
    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(target) as archive:
        for info in archive.infolist():
            if info.is_dir() or _is_skipped(info.filename):
                continue
            contents[info.filename] = archive.read(info.filename)
    return contents


def _read_directory(target: Path) -> dict[str, bytes]:
    """ディレクトリの中身を {相対パス: 内容} で返す。"""
    contents: dict[str, bytes] = {}
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(target))
        if _is_skipped(relative):
            continue
        contents[relative] = path.read_bytes()
    return contents


def _digest(contents: dict[str, bytes]) -> tuple[str, int, int]:
    """{パス: 内容} から (ダイジェスト, ファイル数, 合計バイト数) を返す。"""
    accumulator = hashlib.sha256()
    total_bytes = 0

    for relative in sorted(contents):
        payload = contents[relative]
        accumulator.update(relative.encode("utf-8"))
        accumulator.update(hashlib.sha256(payload).digest())
        total_bytes += len(payload)

    return accumulator.hexdigest()[:16], len(contents), total_bytes


def _top_level(relative: str) -> str:
    return Path(relative).parts[0]


def main() -> None:
    args = _parse_args()
    target: Path = args.target.resolve()

    if target.is_dir():
        contents = _read_directory(target)
    elif target.is_file():
        contents = _read_zip(target)
    else:
        raise FileNotFoundError(f"対象が見つかりません: {target}")

    if args.per_entry:
        for name in sorted({_top_level(r) for r in contents}):
            subset = {r: c for r, c in contents.items() if _top_level(r) == name}
            digest, count, size = _digest(subset)
            print("  %-18s %s  %4d files  %7.1f MB" % (name, digest, count, size / 1e6))

    digest, count, size = _digest(contents)
    print("TOTAL %s  %d files  %.1f MB" % (digest, count, size / 1e6))


if __name__ == "__main__":
    main()
