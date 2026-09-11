"""指令のなめらかさを作る側の共有部品。

`replay_video_probe.py`（028）・`contact_phase.py`（026）・`scripted_demo.py` が
同じ通過点の作り方と同じ物差しを使えるように、ここへ寄せてある。

- `envelope`  —— 配布データから測った指令の封筒
- `rotation`  —— 姿勢の回転の約束（world 合成の軸角）
- `waypoints` —— 028 の 5 次多項式・記録の割り直し（そのまま切り出したもの）
- `otg`       —— **Ruckig による、区間をまたいで止まらない通過点**
- `smoothness`—— 配布データと形が揃っているかを測る物差し
"""
from .envelope import (LIBERO_PLUS, MotionEnvelope, envelope_from_environment,
                       measure_envelope)
from .otg import HAS_RUCKIG, RUCKIG_VERSION, MotionState, PoseTrajectoryPlanner
from .waypoints import min_jerk_waypoints, resample_segment

__all__ = ["LIBERO_PLUS", "MotionEnvelope", "envelope_from_environment",
           "measure_envelope", "HAS_RUCKIG", "RUCKIG_VERSION", "MotionState",
           "PoseTrajectoryPlanner", "min_jerk_waypoints", "resample_segment"]
