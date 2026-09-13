#!/usr/bin/env python3
"""`lerobot-train` の代替。場面カメラに blur を掛けるパッチを当ててから main() を呼ぶ。

**CLI の引数は lerobot-train と完全に同じものをそのまま渡す**——ここでは一切
解釈しない。何を掛けるかは環境変数で受け取る（sensor_noise_patch.py の docstring）。

**場面カメラを潰すパッチも一緒に当てる。**ラッパーは 1 つしか選べないので、
両方が選ばれていても片方が落ちないようにするためである。どちらのパッチも、
自分の環境変数が立っていなければ何もしない。
"""
import sys
from pathlib import Path

# これはスクリプトとして直接起動されるため、sys.path に載るのは自分の
# ディレクトリだけである。パッケージの外側（solution/train、コンテナでは
# /opt/pi05）を載せる。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.camera_dropout import camera_dropout_patch
from experiments.sensor_noise import sensor_noise_patch

camera_dropout_patch.apply()
sensor_noise_patch.apply()

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
