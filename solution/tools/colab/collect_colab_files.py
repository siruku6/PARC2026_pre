#!/usr/bin/env python3
"""Colab へアップロードするファイルを 1 つのフォルダに集める。

Colab の評価ノートブックは、リポジトリの中の何ファイルかを `/content/` へ
手で置くことを前提にしている。置き場所がばらばら（`solution/` 直下と
`solution/tools/` の下）だと、毎回どれとどれだったかを思い出す羽目になる。

このスクリプトは**その都度コピーを作り直す**ので、リポジトリ側を直したのに
アップロードしたのは古いまま、という取り違えが起きない。
コピーの置き場所 `work/` は git の管理外である。

使い方::

    python3 solution/tools/colab/collect_colab_files.py
    open work/colab_upload          # Finder で開いて、中身を Colab へドラッグ

Reads:
  - solution/policy_server.py
  - solution/tools/eval/make_validation_suite.py
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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# (リポジトリ内の場所, Colab で何に使うか)
_UPLOAD_FILES: tuple[tuple[str, str], ...] = (
    (
        "solution/tools/colab/colab_eval.py",
        "**ノートブックが呼ぶ本体。**手順 1〜9 が関数になっている。"
        "セルに貼るのは import と 6 行の呼び出しだけ。",
    ),
    (
        "solution/tools/colab/colab_eval_config.py",
        "**編集するのはこのファイルだけ。**重みの場所・塊の数・設定値・録画。"
        "書き換えたらこれだけ上げ直せば効く（colab_eval.py が毎回読み直す）。",
    ),
    (
        "solution/policy_server.py",
        "評価に使う推論サーバ。colab_eval.setup_policy() が /content/ から拾う。",
    ),
    (
        "solution/tools/eval/make_validation_suite.py",
        "検証用スイートを作る。build_suite() だけで使う。",
    ),
    (
        "solution/tools/eval/check_validation_suite.py",
        "作った CSV が本当に開けるかを、評価を回す前に確かめる。build_suite() が呼ぶ。",
    ),
    (
        "solution/tools/eval/patch_kit_for_variants.py",
        "配布キットへのパッチ。setup_kit() と build_suite() が呼ぶ。"
        "**これが無いと setup_kit() で落ちる。**",
    ),
    (
        "solution/tools/eval/rollout_recorder.py",
        "ロールアウトの録画。**patch_kit_for_variants.py が同じ /content/ から"
        "探してキットへ複製する**ので、2 つは必ず揃えて置く。",
    ),
    (
        "solution/tools/eval/summarize_rollout_videos.py",
        "撮れた mp4 を見るべき順に並べ、コマ並べ画像にする。run() が呼ぶ。",
    ),
)

_READ_ME = """\
このフォルダの中身をまとめて Colab の /content/ へドラッグする。

作り直すには:
    python3 solution/tools/colab/collect_colab_files.py

{descriptions}

置いたあと、ランタイムのタイプを GPU にしてから、貼るのはセル 3 つだけ:

    # セル 1 —— 読み込む（ファイルを上げ直したときも、これを流し直すだけでよい）
    import importlib, sys
    sys.path.insert(0, "/content")
    import colab_eval as ce
    importlib.reload(ce)

    # セル 2 —— 準備（上から順に。合わせて 15 分ほど）
    ce.prepare()                  # Drive・7 ファイルが上がっているかの確認・GPU と EGL
    ce.setup_kit()                # clone・setup.sh・パッチ（5〜10 分）
    ce.prepare_weights()          # 重みを用意し、ハッシュで名指しする
    ce.setup_policy()             # 方策サーバーの依存と配置
    ce.build_suite()              # 検証用スイートへ差し替え、塊に割る

    # セル 3 —— 評価（1 つ 28 件・約 51 分。数字を変えて 3 回実行する）
    ce.run(chunk=1)

  - 疎通確認は ce.run(chunk=1, max_tasks=5)（5 分）。**戻し忘れが起きない。**
  - 途中で切れたら ce.status() で、どの塊が終わっているか分かる。
  - colab_eval.py を上げ直したら importlib.reload(ce) を通す。**版が画面に出る。**

手順の説明: docs/solution/colab_eval.md
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
    # 成り立たない。前の回の CSV を一緒に上げれば、手順 5-1 がそれを使う。
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
        print("  **手順 5-1 がこれを使う。**同じタスクで前の回と比べられる")

    _write_readme(output_dir)

    print(f"{output_dir} に {len(copied)} ファイルを集めた。")
    for path in copied:
        print(f"  {path.name}  ({path.stat().st_size:,} バイト)")
    print("\nこのフォルダの中身を Colab の /content/ へドラッグする。")
    print(f"  open {output_dir}")


if __name__ == "__main__":
    main()
