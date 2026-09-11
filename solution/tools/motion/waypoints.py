"""028 が `replay_video_probe.py` に書いた通過点の作り方を、そのまま切り出したもの。

**なぜ切り出すか。** 同じ 2 つの関数を 026（`contact_clean`）や
`scripted_demo` からも呼びたいが、`replay_video_probe.py` は別のセッションが
持っているので複製すると片方だけ古くなる。**中身は 1 行も変えていない**
（回転の合成だけ `rotation.py` の共有関数に置き換えた。値は同じ）。

**新しい作り方（Ruckig）は `otg.py` にある。**ここは比較の基準として残す枠で、
「同じ入力に同じ通過点を返すか」を `tests/test_motion_matches_028.py` が見ている。
"""
from __future__ import annotations

import numpy as np

from .envelope import LIBERO_PLUS, MotionEnvelope
from .rotation import relative_axisangle, rotate_along

#: 最小ジャーク（5 次多項式）の速度の山は、平均速度のこの倍率になる。
MIN_JERK_PEAK_RATIO = 1.875


def min_jerk_weight(ratio: float) -> float:
    """0 → 1 へ、両端で速度も加速度も 0 になる 5 次多項式の重み。"""
    return 10.0 * ratio ** 3 - 15.0 * ratio ** 4 + 6.0 * ratio ** 5


def min_jerk_waypoints(current_pose: np.ndarray, target_pose: np.ndarray,
                       envelope: MotionEnvelope = LIBERO_PLUS) -> list[np.ndarray]:
    """接近区間の通過点を 5 次多項式の最小ジャークで作る（028 の実装そのもの）。

    **刻み数は「1 コマに出してよい上限」から決まる。**最小ジャークの速度の山は
    平均の `MIN_JERK_PEAK_RATIO` 倍なので、山のところで上限に収まるように取る。
    並進と回転で別々に出して、多いほうを使う。

    Parameters
    ----------
    current_pose, target_pose : np.ndarray
        姿勢 (6,)。前 3 つが位置 [m]、後ろ 3 つが軸角。
    envelope : MotionEnvelope
        1 コマに出してよい指令の上限と、指令 1 単位あたりの実移動量。

    Returns
    -------
    list of np.ndarray
        通過点 (6,) の列。**最後の要素は `target_pose` と同じ姿勢である。**
        現在の姿勢そのものは入らない。
    """
    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)

    delta_pos = target[:3] - current[:3]
    distance_m = float(np.linalg.norm(delta_pos))
    axis_angle_rel = relative_axisangle(current[3:6], target[3:6])
    angle_rad = float(np.linalg.norm(axis_angle_rel))

    steps_pos = np.ceil(MIN_JERK_PEAK_RATIO * distance_m
                        / (envelope.limit_pos * envelope.response_pos_m))
    steps_rot = np.ceil(MIN_JERK_PEAK_RATIO * angle_rad
                        / (envelope.limit_rot * envelope.response_rot_rad))
    num_waypoints = max(int(steps_pos), int(steps_rot), envelope.min_steps)

    waypoints: list[np.ndarray] = []
    for index in range(1, num_waypoints + 1):
        ratio = index / num_waypoints
        weight = min_jerk_weight(ratio)
        position = current[:3] + delta_pos * weight
        if angle_rad * weight < 1e-9:
            orientation = current[3:6].copy()
        else:
            orientation = rotate_along(current[3:6], axis_angle_rel, weight)
        waypoints.append(np.concatenate([position, orientation]))
    return waypoints


def resample_segment(start_pose: np.ndarray, end_pose: np.ndarray,
                     envelope: MotionEnvelope = LIBERO_PLUS) -> list[np.ndarray]:
    """記録された軌跡の 1 区間を、腕が 1 コマで追える間隔に割り直す（028 の実装）。

    **最小ジャークは使わない。**記録の軌跡は自分の速度の形を既に持っているので、
    区間ごとに加速と減速を挟むと記録に無い停止が 1 コマおきに入る。区間の中を
    等分するだけにして、速度の形は記録のものを残す。

    Parameters
    ----------
    start_pose, end_pose : np.ndarray
        記録された連続する 2 コマの姿勢 (6,)。
    envelope : MotionEnvelope
        1 コマに出してよい指令の上限と、指令 1 単位あたりの実移動量。

    Returns
    -------
    list of np.ndarray
        渡す目標 (6,) の列。**最後の要素は `end_pose` である。**
    """
    start = np.asarray(start_pose, dtype=np.float64)
    end = np.asarray(end_pose, dtype=np.float64)

    delta_pos = end[:3] - start[:3]
    distance_m = float(np.linalg.norm(delta_pos))
    axis_angle_rel = relative_axisangle(start[3:6], end[3:6])
    angle_rad = float(np.linalg.norm(axis_angle_rel))

    num_steps = max(
        int(np.ceil(distance_m / (envelope.limit_pos * envelope.response_pos_m))),
        int(np.ceil(angle_rad / (envelope.limit_rot * envelope.response_rot_rad))),
        1,
    )
    if num_steps == 1:
        return [end.copy()]

    targets: list[np.ndarray] = []
    for index in range(1, num_steps + 1):
        ratio = index / num_steps
        position = start[:3] + delta_pos * ratio
        if angle_rad * ratio < 1e-9:
            orientation = end[3:6].copy()
        else:
            orientation = rotate_along(start[3:6], axis_angle_rel, ratio)
        targets.append(np.concatenate([position, orientation]))
    return targets
