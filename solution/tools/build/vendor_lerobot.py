#!/usr/bin/env python3
"""LeRobot のソースを提出物へ同梱できる形に変換する。

採点環境は Python 3.10 固定である一方、LeRobot v0.6.0 は
`requires-python >= 3.12` を宣言しているため `pip install` できない。
これは配布メタデータによる拒否であり、ソースを直接 `sys.path` に載せる場合は
適用されない。実際に 3.10 で動かない箇所だけを機械的に書き換えることで、
学習側（3.12）と同じコードを採点側（3.10）でも使えるようにする。

書き換えの対象は 3.12/3.11 でしか解釈できない構文・API に限る。
ロジックには一切手を入れない。

実行はどのディレクトリからでもよい。入出力はスクリプト自身の位置から求めた
リポジトリルート基準で解決する。

Reads:
  - work/lerobot-src/src/lerobot/**（GitHub の v0.6.0 タグを展開したもの）
Generates:
  - work/artifacts/vendor/lerobot/**（Python 3.10 で import できる同梱用ソース）
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# solution/tools/build/vendor_lerobot.py の 3 階層上がリポジトリルート。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SOURCE = _REPO_ROOT / "work" / "lerobot-src" / "src" / "lerobot"
_DEFAULT_DESTINATION = _REPO_ROOT / "work" / "artifacts" / "vendor" / "lerobot"

# 3.11 以降でしか typing に存在しない名前。typing_extensions から取り直す。
# typing_extensions は採点イメージにプリインストールされているため追加依存にならない。
_TYPING_EXTENSIONS_NAMES: tuple[str, ...] = ("Unpack",)

_TYPING_IMPORT_PATTERN = re.compile(r"^from typing import (?P<names>.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Backport:
    """3.12/3.11 専用の記述を 3.10 で解釈できる形へ置き換える 1 件分の指示。

    Attributes
    ----------
    relative_path : str
        同梱先ルートからの相対パス。
    before : str
        置換前の文字列。ファイル内に厳密に 1 回だけ現れることを検証する。
    after : str
        置換後の文字列。
    reason : str
        なぜこの書き換えが必要かを 1 行で示す。ログとレポートに使う。
    """

    relative_path: str
    before: str
    after: str
    reason: str


# PEP 695（型引数構文）と 3.11 以降の標準ライブラリ API が対象。
# `str | int` のような PEP 604 の union は 3.10 でそのまま動くため触らない。
_BACKPORTS: tuple[Backport, ...] = (
    Backport(
        relative_path="processor/pipeline.py",
        before="from typing import Any, TypedDict, TypeVar, cast",
        after="from typing import Any, Generic, TypedDict, TypeVar, cast",
        reason="PEP 695 を Generic で置き換えるため Generic を import する",
    ),
    Backport(
        relative_path="processor/pipeline.py",
        before="class DataProcessorPipeline[TInput, TOutput](HubMixin):",
        after="class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):",
        reason="PEP 695: TInput/TOutput は既に TypeVar として定義済み",
    ),
    Backport(
        relative_path="utils/io_utils.py",
        before="from typing import Any",
        after="from typing import Any, TypeVar",
        reason="PEP 695 の型引数を TypeVar で置き換えるため",
    ),
    Backport(
        relative_path="utils/io_utils.py",
        before="def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:",
        after=(
            '_JsonLikeT = TypeVar("_JsonLikeT")\n'
            "\n"
            "\n"
            "def deserialize_json_into_object(fpath: Path, obj: _JsonLikeT) -> _JsonLikeT:"
        ),
        reason="PEP 695: bound は実行時の挙動に影響しないため省略する",
    ),
    Backport(
        relative_path="motors/motors_bus.py",
        before="type NameOrID = str | int\ntype Value = int | float",
        after="NameOrID = str | int\nValue = int | float",
        reason="PEP 695 の type 文。右辺の union 自体は 3.10 で有効",
    ),
    Backport(
        relative_path="datasets/streaming_dataset.py",
        before="from pathlib import Path",
        after="from pathlib import Path\nfrom typing import Generic, TypeVar",
        reason="PEP 695 を Generic/TypeVar で置き換えるため",
    ),
    Backport(
        relative_path="datasets/streaming_dataset.py",
        before="class Backtrackable[T]:",
        after='T = TypeVar("T")\n\n\nclass Backtrackable(Generic[T]):',
        reason="PEP 695: クラス本体が T を参照するため名前は T のまま残す",
    ),
    Backport(
        relative_path="configs/video.py",
        before="from typing import Any, ClassVar, Self",
        after=(
            "from typing import Any, ClassVar\n"
            "\n"
            "try:  # Python 3.11+\n"
            "    from typing import Self\n"
            "except ImportError:  # Python 3.10\n"
            "    from typing import TypeVar\n"
            "\n"
            '    Self = TypeVar("Self")'
        ),
        reason="typing.Self は 3.11 以降。この場に限り TypeVar で代替できる",
    ),
    Backport(
        relative_path="jobs/hf.py",
        before="dt.datetime.now(dt.UTC))",
        after="dt.datetime.now(dt.timezone.utc))",
        reason="datetime.UTC は 3.11 以降。timezone.utc と等価",
    ),
    Backport(
        relative_path="jobs/hf.py",
        before="now = dt.datetime.now(dt.UTC)",
        after="now = dt.datetime.now(dt.timezone.utc)",
        reason="datetime.UTC は 3.11 以降。timezone.utc と等価",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_SOURCE,
        help=f"LeRobot のソースツリー（既定: {_DEFAULT_SOURCE}）",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=_DEFAULT_DESTINATION,
        help=f"同梱先（既定: {_DEFAULT_DESTINATION}）",
    )
    parser.add_argument(
        "--python310",
        type=Path,
        default=None,
        help="構文検証に使う Python 3.10 の実行ファイル。省略すると検証をスキップする",
    )
    return parser.parse_args()


def _copy_source(source: Path, destination: Path) -> None:
    """LeRobot のソースを同梱先へ複製する（既存の同梱先は破棄する）。"""
    if not source.is_dir():
        raise FileNotFoundError(f"LeRobot のソースが見つかりません: {source}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _apply_backport(destination: Path, backport: Backport) -> None:
    """書き換えを 1 件適用する。対象が 1 箇所でなければ例外にする。

    上流のバージョンが変わって対象が消えた場合に黙って素通りすると、
    3.10 で import できないソースを提出してしまう。ここで必ず落とす。
    """
    target = destination / backport.relative_path
    text = target.read_text(encoding="utf-8")
    occurrences = text.count(backport.before)

    if occurrences != 1:
        raise RuntimeError(
            f"{backport.relative_path}: 置換対象が {occurrences} 箇所見つかりました"
            f"（1 箇所を期待）。LeRobot のバージョンを確認してください: {backport.before!r}"
        )

    target.write_text(text.replace(backport.before, backport.after), encoding="utf-8")


def _apply_all_backports(destination: Path) -> None:
    """全ての書き換えを適用し、内容を標準出力に記録する。"""
    for backport in _BACKPORTS:
        _apply_backport(destination, backport)
        print(f"  patched  {backport.relative_path}: {backport.reason}")


def _rewrite_typing_import(match: re.Match[str]) -> str:
    """`from typing import ...` から 3.11 以降の名前を typing_extensions へ振り分ける。"""
    names = [name.strip() for name in match.group("names").split(",")]
    moved = [name for name in names if name in _TYPING_EXTENSIONS_NAMES]

    if not moved:
        return match.group(0)

    remaining = [name for name in names if name not in _TYPING_EXTENSIONS_NAMES]
    lines = []
    if remaining:
        lines.append(f"from typing import {', '.join(remaining)}")
    lines.append(f"from typing_extensions import {', '.join(moved)}")
    return "\n".join(lines)


def _backport_typing_extensions(destination: Path) -> int:
    """3.11 以降の typing 名を typing_extensions からの import に置き換える。

    括弧を使わない 1 行の `from typing import ...` だけを対象にする。
    複数行の import が現れた場合は書き換えられないが、その場合は
    後段の import 検証で必ず失敗するため見逃しにはならない。

    Returns
    -------
    changed : int
        書き換えたファイル数。
    """
    changed = 0

    for path in sorted(destination.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not any(name in text for name in _TYPING_EXTENSIONS_NAMES):
            continue

        rewritten = _TYPING_IMPORT_PATTERN.sub(_rewrite_typing_import, text)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            changed += 1
            print(f"  patched  {path.relative_to(destination)}: typing -> typing_extensions")

    return changed


def _verify_syntax(destination: Path, python310: Path) -> None:
    """同梱先が Python 3.10 の構文で解釈できることを確認する。"""
    result = subprocess.run(
        [str(python310), "-m", "compileall", "-q", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()

    if output:
        raise RuntimeError(f"Python 3.10 で構文エラーが残っています:\n{output}")


def main() -> None:
    args = _parse_args()
    source: Path = args.source.resolve()
    destination: Path = args.destination.resolve()

    print(f"[vendor] 複製 {source} -> {destination}")
    _copy_source(source, destination)

    print(f"[vendor] 3.10 互換への書き換え（構文: {len(_BACKPORTS)} 件）")
    _apply_all_backports(destination)

    changed = _backport_typing_extensions(destination)
    print(f"[vendor] typing_extensions への振り替え: {changed} ファイル")

    if args.python310 is None:
        print("[vendor] 構文検証はスキップしました（--python310 未指定）")
    else:
        print(f"[vendor] 構文検証 {args.python310}")
        _verify_syntax(destination, args.python310.resolve())
        print("[vendor] Python 3.10 の構文で全ファイルが解釈できました")

    file_count = sum(1 for _ in destination.rglob("*.py"))
    print(f"[vendor] 完了: {file_count} ファイル")


if __name__ == "__main__":
    main()
