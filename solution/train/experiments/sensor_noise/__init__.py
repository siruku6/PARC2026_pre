"""場面カメラに blur を掛ける工夫の、train_pi05.py 側の窓口。

**狙い。**Track2 の `Sensor Noise` は、机を写す固定カメラの画像だけを
motion / gaussian / zoom blur・fog・glass blur のどれか 1 種類で劣化させる。
public 採点の 1 本目は motion blur である（採点ログの警告から確定。
`docs/talk/tmp_reply/260913_SESSION_next_training.md` の 1-1-1）。
**学習データには劣化した画像が 1 枚も無いので、学習中に同じ劣化を見せる。**

**実体は sensor_noise_patch.py の 1 関数で、train_pi05.py はここが公開する
関数を呼ぶ**（camera_dropout と同じ形にそろえてある）。

    corruptions.py                         劣化 5 種（env_wrapper.py を写したもの）
    sensor_noise_patch.py                  DatasetReader.get_item を包む本体
    lerobot_train_with_sensor_noise.py     それを import してから lerobot-train を呼ぶ薄いラッパー

**`--scene-sensor-noise` を渡さない限り一切効かない。**既定は 0.0 で、
渡さなければ工夫なしのときと一字一句同じコマンドが組まれる。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_WRAPPER = Path(__file__).resolve().parent / "lerobot_train_with_sensor_noise.py"

# 採点の描画の大きさ（配布キット `pipeline/config.py` の `LIBERO_EVAL_CAMERA` の既定）。
_DEFAULT_SIZE = 128


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """blur を掛ける 3 つのオプションを、train_pi05.py の引数に足す。

    掛ける列の名前は、camera_dropout の `--scene-camera-key` を共用する。
    """
    from experiments.sensor_noise.sensor_noise_patch import DEFAULT_WEIGHTS

    parser.add_argument("--scene-sensor-noise", type=float, default=0.0, metavar="P",
                        help="机を写す固定カメラ（--scene-camera-key）に、1 コマごとに"
                             "確率 P で blur を 1 種類掛ける。0.0（既定）なら何もしない。"
                             "**手首のカメラには掛けない**——採点で劣化するのは場面側だけ")
    parser.add_argument("--scene-sensor-noise-weights", type=str, default=DEFAULT_WEIGHTS,
                        metavar="NAME=W,...",
                        help=f"種類ごとの選ばれやすさ（既定 {DEFAULT_WEIGHTS}）")
    parser.add_argument("--scene-sensor-noise-size", type=int, default=_DEFAULT_SIZE,
                        metavar="N",
                        help=f"劣化を掛けるときに縮める一辺（既定 {_DEFAULT_SIZE}＝採点の描画の"
                             "大きさ）。強度の数字を採点とそろえるためにある")


def selected(args: argparse.Namespace) -> bool:
    """blur を掛ける工夫が選ばれているか。"""
    return args.scene_sensor_noise > 0.0


def validate(args: argparse.Namespace) -> None:
    """渡された値が使えるかを、学習を始める前に確かめる。

    Raises
    ------
    SystemExit
        確率が [0, 1] の外・重みの書き方が誤り・一辺が 1 未満のとき。
    """
    if not selected(args):
        return
    if args.scene_sensor_noise > 1.0:
        raise SystemExit(f"[train_pi05] --scene-sensor-noise は 0 以上 1 以下である。"
                         f"渡された値: {args.scene_sensor_noise}")
    if args.scene_sensor_noise_size < 1:
        raise SystemExit(f"[train_pi05] --scene-sensor-noise-size は 1 以上である。"
                         f"渡された値: {args.scene_sensor_noise_size}")
    from experiments.sensor_noise.sensor_noise_patch import parse_weights
    parse_weights(args.scene_sensor_noise_weights)


def set_environment(args: argparse.Namespace) -> None:
    """掛ける確率・列・一辺・重みを、子プロセスへ環境変数で伝える。"""
    if not selected(args):
        return
    os.environ["SENSOR_NOISE_PROB"] = str(args.scene_sensor_noise)
    os.environ["SENSOR_NOISE_KEY"] = args.scene_camera_key
    os.environ["SENSOR_NOISE_SIZE"] = str(args.scene_sensor_noise_size)
    os.environ["SENSOR_NOISE_WEIGHTS"] = args.scene_sensor_noise_weights


def command_head(args: argparse.Namespace) -> list[str]:
    """学習コマンドの先頭。このラッパーは場面カメラを潰すパッチも一緒に当てる。"""
    return ["python", str(_WRAPPER)]


def extra_arguments(args: argparse.Namespace) -> list[str]:
    """lerobot-train への追加引数。**1 つも増やさない**（掛けるのは DataLoader の中）。"""
    return []


def describe(args: argparse.Namespace) -> str:
    """学習開始時のログに出す 1 行。"""
    if not selected(args):
        return "scene_sensor_noise=なし"
    return (f"scene_sensor_noise={args.scene_sensor_noise}"
            f"（{args.scene_camera_key}・一辺 {args.scene_sensor_noise_size}・"
            f"{args.scene_sensor_noise_weights}）")
