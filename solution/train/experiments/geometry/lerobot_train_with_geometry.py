#!/usr/bin/env python
"""幾何の接ぎ木を有効にしてから lerobot-train を呼ぶ薄いラッパー。

`experiments/quantization/lerobot_train_with_quant.py` と同じ作りである。
**LeRobot v0.6.0 には差し込み口が無い**ので、`lerobot.scripts.lerobot_train.main()`
を呼ぶ前に、副作用目当てで接ぎ木のモジュールを import する。
"""
import os
import sys
from pathlib import Path

# スクリプトとして直接起動されるので、sys.path に載るのは自分のディレクトリだけである。
# `experiments.geometry` パッケージの一員として import するため、外側を載せる。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if os.environ.get("GEO_FUSE_MODE") == "vggt":
    # 副作用: PI05Pytorch.embed_prefix を、幾何を混ぜる版に差し替える
    import experiments.geometry.vggt_fuser  # noqa: F401

# **ラッパーは 1 つしか選べないので、場面カメラを潰すパッチはここでも当てる。**
# CAMERA_DROPOUT_PROB が無いか 0 なら apply() は何もしない
# （experiments/camera_dropout/camera_dropout_patch.py）。
from experiments.camera_dropout import camera_dropout_patch

camera_dropout_patch.apply()

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
