#!/usr/bin/env python3
"""`lerobot-train` の代替。場面カメラを潰すパッチを当ててから main() を呼ぶ。

**CLI の引数は lerobot-train と完全に同じものをそのまま渡す**——ここでは一切
解釈しない。何を潰すかは環境変数で受け取る（camera_dropout_patch.py の docstring）。

**geometry / quantization が選ばれているときは、このラッパーは使われない。**
ラッパーは 1 つしか選べないので、そちらのラッパーが同じパッチを import する。
"""
import sys
from pathlib import Path

# これはスクリプトとして直接起動されるため、sys.path に載るのは自分の
# ディレクトリだけである。**パッチは experiments.camera_dropout パッケージの
# 一員として import する**ので、パッケージの外側（solution/train、コンテナでは
# /opt/pi05）を載せる必要がある。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.camera_dropout import camera_dropout_patch

camera_dropout_patch.apply()

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
