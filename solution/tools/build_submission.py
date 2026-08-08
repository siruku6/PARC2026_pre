#!/usr/bin/env python3
"""提出用の zip を組み立てる。

提出物は 2 か所から集める。**自分で書いたコードは solution/ が正**であり、
生成物（重み・同梱した LeRobot）だけが work/artifacts/ にある。両者を
コピーで同期させると必ずどちらかが古くなるため、zip 化のときに直接束ねる。

Omnicampus へ出す zip は `policy_server.py` が**ルート直下**に来る構造でなければ
ならない。フォルダごと固めた zip は自動で 1 階層降ろされるが、採点環境によっては
即エラーになると配布 README が警告しているため、ここで構造を保証する。

重みは圧縮が効かず zip 化が遅くなるだけなので、無圧縮（ZIP_STORED）で固める。

実行はどのディレクトリからでもよい。

Reads:
  - solution/policy_server.py
  - solution/requirements.txt
  - work/artifacts/model_weights/**
  - work/artifacts/vendor/**
Generates:
  - work/submission.zip
  - work/submission.sha256（レポートに記載するハッシュ値）
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

# solution/tools/build_submission.py の 2 階層上がリポジトリルート。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SOLUTION = _REPO_ROOT / "solution"
_DEFAULT_ARTIFACTS = _REPO_ROOT / "work" / "artifacts"
_DEFAULT_OUTPUT = _REPO_ROOT / "work" / "submission.zip"

# solution/ から取るファイル（zip のルート直下に置く）。
_SOLUTION_FILES = ("policy_server.py", "requirements.txt")

# work/artifacts/ から取るディレクトリ（名前をそのまま zip 内のパスにする）。
_ARTIFACT_DIRS = ("model_weights", "vendor")

# 実行時に生成されるもの。含めるとサイズが膨らむだけで意味がない。
_EXCLUDE_PARTS = frozenset({"__pycache__", ".hf_home", ".ipynb_checkpoints", ".DS_Store"})
_EXCLUDE_SUFFIXES = frozenset({".pyc"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--solution",
        type=Path,
        default=_DEFAULT_SOLUTION,
        help="自分で書いたコードの置き場（既定: solution/）",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=_DEFAULT_ARTIFACTS,
        help="重みと同梱ソースの置き場（既定: work/artifacts/）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="出力する zip（既定: work/submission.zip）",
    )
    return parser.parse_args()


def _collect_entries(solution: Path, artifacts: Path) -> list[tuple[Path, str]]:
    """zip に入れる (実ファイル, zip 内のパス) の一覧を作る。

    足りないものがあれば、zip を作る前に例外にする。中身が欠けた zip を
    提出してしまうと、採点 1 回分を無駄にすることになる。
    """
    entries: list[tuple[Path, str]] = []

    for name in _SOLUTION_FILES:
        path = solution / name
        if not path.is_file():
            raise FileNotFoundError(f"必須ファイルがありません: {path}")
        entries.append((path, name))

    for name in _ARTIFACT_DIRS:
        directory = artifacts / name
        if not directory.is_dir():
            raise FileNotFoundError(
                f"生成物がありません: {directory}"
                "（vendor_lerobot.py の実行や重みの配置が済んでいるか確認してください）"
            )
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if _EXCLUDE_PARTS & set(path.parts):
                continue
            if path.suffix in _EXCLUDE_SUFFIXES:
                continue
            entries.append((path, str(path.relative_to(artifacts))))

    return entries


def _write_zip(output: Path, entries: list[tuple[Path, str]]) -> None:
    """ルート直下構造を保ったまま無圧縮で固める。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path, arcname in entries:
            archive.write(path, arcname=arcname)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    solution: Path = args.solution.resolve()
    artifacts: Path = args.artifacts.resolve()
    output: Path = args.output.resolve()

    entries = _collect_entries(solution, artifacts)
    _write_zip(output, entries)

    checksum = _sha256(output)
    checksum_path = output.with_suffix(".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")

    print(f"[build] コード   {solution}")
    print(f"[build] 生成物   {artifacts}")
    print(f"[build] 出力     {output}")
    print(f"[build] {len(entries)} files / {output.stat().st_size / 1e6:.0f} MB")
    print(f"[build] sha256 {checksum}")
    print(f"[build] ハッシュを保存: {checksum_path}")


if __name__ == "__main__":
    main()
