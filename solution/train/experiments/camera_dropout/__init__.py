"""場面カメラを確率で潰す工夫の、train_pi05.py 側の窓口。

**狙い。**Track2 の摂動 4 分類のうち `Camera Viewpoints`（1,599 件・15.9%）は
**机を写す固定カメラの視点をずらす**。手首のカメラは腕について動くのでずれない。
**いまの重みが場面カメラに頼りきっていると、ずらされた時点で崩れる。**
**学習中に場面カメラを確率で潰すと、手首のカメラだけで進める癖が付く。**

**実体は camera_dropout_patch.py の 1 関数だけで、train_pi05.py はここが公開する
6 つの関数を呼ぶ**（quantization / geometry と同じ形にそろえてある）。

    camera_dropout_patch.py                   DatasetReader.get_item を包む本体
    lerobot_train_with_camera_dropout.py      それを import してから lerobot-train を呼ぶ薄いラッパー

**`--scene-camera-dropout` を渡さない限り一切効かない。** 既定は 0.0 で、
渡さなければ工夫なしのときと一字一句同じコマンドが組まれる。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# 潰す工夫を使うときは "lerobot-train"（console script）の代わりに、
# パッチを import してから同じ main() を呼ぶ薄いラッパーを使う。
# **ただし geometry / quantization が選ばれているときは、そちらのラッパーが
# 同じパッチを import する**（ラッパーは 1 つしか選べないため）。
_WRAPPER = Path(__file__).resolve().parent / "lerobot_train_with_camera_dropout.py"

# 配布データ（`lerobot/libero_plus`・配布 tar の `libero_combined_20hz`）の
# どちらも、画像の列は `observation.images.wrist` と `observation.images.front`
# の 2 本である（配布 tar の `meta/info.json` を 9/2 に読んで確認した）。
# **`front` が机を写す固定カメラで、Track2 でずらされるのはこちらである。**
_DEFAULT_SCENE_CAMERA_KEY = "observation.images.front"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """場面カメラを潰す 2 つのオプションを、train_pi05.py の引数に足す。

    Parameters
    ----------
    parser : argparse.ArgumentParser
        train_pi05.py の `_parse_args()` が組み立てているパーサー。
    """
    parser.add_argument("--scene-camera-dropout", type=float, default=0.0,
                        metavar="P",
                        help="机を写す固定カメラ（--scene-camera-key）を、1 コマごとに"
                             "確率 P で真っ黒にする。0.0（既定）なら何もしない。"
                             "**手首のカメラは潰さない**——Track2 でずれるのは場面側だけ"
                             "である。値の選び方は "
                             "docs/plan/20_submission_strategy_to_0917.md 参照")
    parser.add_argument("--scene-camera-key", type=str,
                        default=_DEFAULT_SCENE_CAMERA_KEY, metavar="KEY",
                        help=f"潰す列の名前（既定 {_DEFAULT_SCENE_CAMERA_KEY}）。"
                             "**データの中に無ければ学習を始めた直後に止まる**"
                             "（黙って効かないのを防ぐため）")


def selected(args: argparse.Namespace) -> bool:
    """場面カメラを潰す工夫が選ばれているか。"""
    return args.scene_camera_dropout > 0.0


def validate(args: argparse.Namespace) -> None:
    """渡された確率が使える値かを、学習を始める前に確かめる。

    **列の名前が正しいかどうかは、ここでは確かめられない**——データを読み込む前
    だからである。そちらは camera_dropout_patch.py が最初の 1 コマで見て、
    合わなければその場で止める。

    Raises
    ------
    SystemExit
        確率が [0, 1] の外にあるとき。
    """
    if not selected(args):
        return
    if args.scene_camera_dropout > 1.0:
        raise SystemExit(
            f"[train_pi05] --scene-camera-dropout は 0 以上 1 以下である。"
            f"渡された値: {args.scene_camera_dropout}"
        )
    if not args.scene_camera_key:
        raise SystemExit("[train_pi05] --scene-camera-key が空である。")


def set_environment(args: argparse.Namespace) -> None:
    """潰す確率と列の名前を、子プロセスへ環境変数で伝える。

    ラッパーが読んで camera_dropout_patch.apply() に渡す。ここでは os.environ を
    書き換えるだけで、train_pi05.py の `_run_and_log` の subprocess.Popen は env=
    を渡さず親の環境をそのまま継承するので、これだけで子プロセスに伝わる。
    """
    if not selected(args):
        return
    os.environ["CAMERA_DROPOUT_PROB"] = str(args.scene_camera_dropout)
    os.environ["CAMERA_DROPOUT_KEY"] = args.scene_camera_key


def command_head(args: argparse.Namespace) -> list[str]:
    """学習コマンドの先頭（何を実行するか）を返す。

    Returns
    -------
    head : list[str]
        潰す工夫を選んでいなければ `["lerobot-train"]`（今までと同じ console
        script）。選んでいれば、パッチを import してから同じ main() を呼ぶ
        ラッパーの起動コマンド。
    """
    if selected(args):
        return ["python", str(_WRAPPER)]
    return ["lerobot-train"]


def extra_arguments(args: argparse.Namespace) -> list[str]:
    """lerobot-train への追加引数を返す。

    **この工夫は lerobot-train の引数を 1 つも増やさない**——潰すのは
    DataLoader の中であって、設定で表せる範囲の外である（それがこの
    パッケージが在る理由である）。
    """
    return []


def describe(args: argparse.Namespace) -> str:
    """学習開始時のログに出す 1 行。"""
    if not selected(args):
        return "scene_camera_dropout=なし"
    return (f"scene_camera_dropout={args.scene_camera_dropout}"
            f"（{args.scene_camera_key}）")
