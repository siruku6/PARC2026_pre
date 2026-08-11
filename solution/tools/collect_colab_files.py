#!/usr/bin/env python3
"""Colab へアップロードするファイルを 1 つのフォルダに集める。

Colab の評価ノートブックは、リポジトリの中の何ファイルかを `/content/` へ
手で置くことを前提にしている。置き場所がばらばら（`solution/` 直下と
`solution/tools/` の下）だと、毎回どれとどれだったかを思い出す羽目になる。

このスクリプトは**その都度コピーを作り直す**ので、リポジトリ側を直したのに
アップロードしたのは古いまま、という取り違えが起きない。
コピーの置き場所 `work/` は git の管理外である。

使い方::

    python3 solution/tools/collect_colab_files.py
    open work/colab_upload          # Finder で開いて、中身を Colab へドラッグ

Reads:
  - solution/policy_server.py
  - solution/tools/make_validation_suite.py
Generates:
  - work/colab_upload/policy_server.py
  - work/colab_upload/make_validation_suite.py
  - work/colab_upload/README.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# (リポジトリ内の場所, Colab で何に使うか)
_UPLOAD_FILES: tuple[tuple[str, str], ...] = (
    (
        "solution/policy_server.py",
        "評価に使う推論サーバ。COLAB_EVAL.md のセル D-2 が /content/ から拾う。",
    ),
    (
        "solution/tools/make_validation_suite.py",
        "検証用スイートを作る。COLAB_EVAL.md のセル H だけで使う。",
    ),
    (
        "solution/tools/check_validation_suite.py",
        "作った CSV が本当に開けるかを、評価を回す前に確かめる。セル H が呼ぶ。",
    ),
    (
        "solution/tools/patch_kit_for_variants.py",
        "配布キットへのパッチ。セル B と セル H が呼ぶ。**これが無いと B で落ちる。**",
    ),
    (
        "solution/tools/rollout_recorder.py",
        "ロールアウトの録画。**patch_kit_for_variants.py が同じ /content/ から"
        "探してキットへ複製する**ので、2 つは必ず揃えて置く。",
    ),
    (
        "solution/tools/summarize_rollout_videos.py",
        "撮れた mp4 を見るべき順に並べ、コマ並べ画像にする。セル F が呼ぶ。",
    ),
)

_READ_ME = """\
このフォルダの中身をまとめて Colab の /content/ へドラッグする。

作り直すには:
    python3 solution/tools/collect_colab_files.py

{descriptions}

置いたあとにやること:
  - policy_server.py … セル D-2 を実行すると提出用ディレクトリへ移る。
    セル E-3 の出力に "[parc] policy_server <版>" が出れば差し替え成功。
  - make_validation_suite.py … セル H を、セル B のあと・セル E-1 の前に実行する。
    このときセル E-1 の N_EPISODES は 1 にすること（タスク数が 4 件から
    56〜84 件へ増えるため、20 のままだと 27 時間かかる）。

手順の本体: docs/solution/COLAB_EVAL.md
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="集める先のディレクトリ（既定: <リポジトリ>/work/colab_upload）",
    )
    # **前の回と比べるなら、検証用スイートの CSV を作り直してはいけない。**
    # 1 件でも違うタスクが混ざると、1 件ずつ対にした比較（符号検定）が
    # 成り立たない。前の回の CSV を一緒に上げれば、セル H がそれを使う。
    parser.add_argument(
        "--val-tasks",
        default=None,
        metavar="PATH",
        help="前の回の VAL_TASKS.csv。渡すと一緒に集める"
             "（例: outputs/0811_101824_base_n5_ep1_max300/VAL_TASKS.csv）",
    )
    return parser.parse_args()


def _resolve_output_dir(argument: str | None) -> Path:
    """コピー先を決める。既定は git 管理外の work/ の下。"""
    if argument is not None:
        return Path(argument).resolve()
    return _REPOSITORY_ROOT / "work" / "colab_upload"


def _copy_upload_files(output_dir: Path) -> list[Path]:
    """アップロード対象を output_dir へ複製し、複製したファイルを返す。"""
    copied: list[Path] = []
    for relative_path, _ in _UPLOAD_FILES:
        source = _REPOSITORY_ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"{relative_path} が見つからない")
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _write_readme(output_dir: Path) -> None:
    """フォルダだけ見ても用途が分かるように説明を置く。"""
    descriptions = "\n".join(
        f"  {Path(relative_path).name}\n    {purpose}"
        for relative_path, purpose in _UPLOAD_FILES
    )
    (output_dir / "README.txt").write_text(
        _READ_ME.format(descriptions=descriptions), encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output)

    # 消してから作り直す。前回の名残が混ざっていると、要らないものまで
    # アップロードしてしまう。
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    try:
        copied = _copy_upload_files(output_dir)
    except FileNotFoundError as error:
        print(f"エラー: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if args.val_tasks:
        source = Path(args.val_tasks).resolve()
        if not source.is_file():
            print(f"エラー: {source} が無い", file=sys.stderr)
            raise SystemExit(1)
        destination = output_dir / "VAL_TASKS.csv"
        shutil.copy2(source, destination)
        copied.append(destination)
        rows = destination.read_text(encoding="utf-8").count("\n")
        print(f"検証用スイートの CSV も集めた（{rows - 1} 行）: {source}")
        print("  **セル H がこれを使う。**同じタスクで前の回と比べられる")

    _write_readme(output_dir)

    print(f"{output_dir} に {len(copied)} ファイルを集めた。")
    for path in copied:
        print(f"  {path.name}  ({path.stat().st_size:,} バイト)")
    print("\nこのフォルダの中身を Colab の /content/ へドラッグする。")
    print(f"  open {output_dir}")


if __name__ == "__main__":
    main()
