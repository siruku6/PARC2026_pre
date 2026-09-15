"""姿勢（位置 3 ＋ 軸角 3）を扱うときの回転の約束を 1 か所に集めたもの。

**なぜ要るか。** 逆向きデータの生成では「今の向き → 目標の向き」を何度も測るが、
**軸角ベクトルの成分ごとの引き算は相対回転ではない。**それをやると回転軸が
毎コマ変わり、腕が目標と違う向きへ回り始める（014 の実測）。正しくは
world 合成 `R_target @ R_now^T` の軸角を取る。同じ式が
`replay_video_probe.py` の `osc_command`・`_approach_waypoints`・
`_resample_segment` に 3 度書かれていたので、ここへ寄せた。

**robosuite があればそれを使う。**シミュレータの中で動くときは
`replay_video_probe.py` と 1 ビットも違わない値を返してほしいからである
（robosuite の `quat2mat`・`mat2quat` は内部で float32 に落とすので、
float64 で書き直すと最後の桁が合わない）。robosuite が無いホストでは
同じ約束の numpy 実装に落ちる——通過点の設計をシムを回さずに検討するため。
"""
from __future__ import annotations

import numpy as np

try:                                                     # シムの中
    from robosuite.utils import transform_utils as _T

    HAS_ROBOSUITE = True
except ImportError:                                      # ホストでの検討用
    _T = None
    HAS_ROBOSUITE = False


def axisangle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """軸角ベクトル (3,) を回転行列 (3, 3) に直す。"""
    if HAS_ROBOSUITE:
        return _T.quat2mat(_T.axisangle2quat(np.asarray(axis_angle, dtype=np.float64)))
    vector = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def matrix_to_axisangle(matrix: np.ndarray) -> np.ndarray:
    """回転行列 (3, 3) を軸角ベクトル (3,) に直す。"""
    if HAS_ROBOSUITE:
        return np.asarray(_T.quat2axisangle(_T.mat2quat(np.asarray(matrix))),
                          dtype=np.float64)
    rotation = np.asarray(matrix, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:                             # 180 度付近は差分式が壊れる
        symmetric = (rotation + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(symmetric), 0.0, None))
        largest = int(np.argmax(axis))
        axis = symmetric[:, largest] / max(axis[largest], 1e-12)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def quaternion_to_axisangle(quaternion: np.ndarray) -> np.ndarray:
    """robosuite 規約の四元数 (x, y, z, w) を軸角ベクトル (3,) に直す。"""
    if HAS_ROBOSUITE:
        return np.asarray(_T.quat2axisangle(np.asarray(quaternion, dtype=np.float64)),
                          dtype=np.float64)
    quat = np.asarray(quaternion, dtype=np.float64)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den < 1e-12:
        return np.zeros(3)
    return quat[:3] * 2.0 * float(np.arccos(w)) / den


def relative_axisangle(from_axis_angle: np.ndarray,
                       to_axis_angle: np.ndarray) -> np.ndarray:
    """「今の向き → 目標の向き」の相対回転を軸角ベクトル (3,) で返す。

    **world 合成 `R_target @ R_now^T` の軸角である。**軸角ベクトルどうしの
    成分ごとの引き算ではない（それは相対回転にならない）。
    """
    rotation_from = axisangle_to_matrix(from_axis_angle)
    rotation_to = axisangle_to_matrix(to_axis_angle)
    return matrix_to_axisangle(rotation_to @ rotation_from.T)


def rotate_along(from_axis_angle: np.ndarray, relative: np.ndarray,
                 ratio: float) -> np.ndarray:
    """`from_axis_angle` から相対回転 `relative` の `ratio` 倍だけ進んだ向きを返す。

    測地線（最短の回り方）に沿って進む。`ratio=0` で元の向き、`ratio=1` で
    `relative` を丸ごと適用した向きになる。
    """
    scaled = np.asarray(relative, dtype=np.float64) * float(ratio)
    if float(np.linalg.norm(scaled)) < 1e-9:
        return np.asarray(from_axis_angle, dtype=np.float64).copy()
    rotated = axisangle_to_matrix(scaled) @ axisangle_to_matrix(from_axis_angle)
    return matrix_to_axisangle(rotated)


def pose_distance(from_pose: np.ndarray, to_pose: np.ndarray) -> tuple[float, float]:
    """2 つの姿勢 (6,) のあいだの並進距離 [m] と回転角 [rad] を返す。"""
    from_pose = np.asarray(from_pose, dtype=np.float64)
    to_pose = np.asarray(to_pose, dtype=np.float64)
    distance = float(np.linalg.norm(to_pose[:3] - from_pose[:3]))
    angle = float(np.linalg.norm(relative_axisangle(from_pose[3:6], to_pose[3:6])))
    return distance, angle
