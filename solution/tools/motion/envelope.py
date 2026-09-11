"""1 コマに出してよい指令の「封筒」——配布データ自身から測った値。

**なぜ要るか。** 自作したデモを配布データと混ぜて学習する以上、**自作側の指令が
配布側と違う形をしていると、そこだけ別の分布になって学習が解けない。**
028 は上限への張り付き（`±1.000` に 20.24%・`±0.375` に 28.81%）を潰したが、
封筒として持っていたのは**上限 2 つだけ**だった。ここでは同じ考え方を
**1 階差分（指令がどれだけ速く変われるか）と 2 階差分**まで広げて持つ。

**単位は「指令」そのものである。**メートルや rad ではない。OSC は
`control_delta=True` なので、`action[0:6]` がそのまま 1 コマの指令になる。
時間の単位も「1 コマ」で、秒ではない。**この単位で持つと、Ruckig の
`max_velocity / max_acceleration / max_jerk` にそのまま渡せる**——
位置を「指令 1 単位ぶんの移動量」で割った座標に取れば、Ruckig の速度が
そのまま指令になるからである（`otg.py`）。

**測り方。** `measure_envelope()` が配布データの `action` 列から百分位を出す。
既定値 `LIBERO_PLUS` は 2026-09-11 に `work/artifacts/libero_plus` の
無作為 400 本・61,373 コマから測ったものである:

    並進 |u|    p50 0.5920  p95 0.9375  p100 0.9375
    回転 |u|    p50 0.0546  p99 0.3439  p100 0.3750
    並進 |Δu|   p50 0.0509  p95 0.1714  p99 0.2518
    回転 |Δu|   p50 0.0086  p95 0.0311  p99 0.0514
    並進 |Δ²u|  p50 0.0321  p95 0.1179  p99 0.1848
    回転 |Δ²u|  p50 0.0064  p95 0.0279  p99 0.0461
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionEnvelope:
    """指令の上限・1 階差分の上限・2 階差分の上限をまとめて持つ。

    Attributes
    ----------
    limit_pos, limit_rot : float
        1 コマに出してよい並進・回転の指令の大きさ（成分ごと）。
    accel_pos, accel_rot : float
        指令が 1 コマで変わってよい量 |Δu|。**立ち上がりの速さを決める。**
    jerk_pos, jerk_rot : float
        |Δu| が 1 コマで変わってよい量 |Δ²u|。
    response_pos_m, response_rot_rad : float
        指令 1 単位あたりに腕が実際に動く量 [m] / [rad]。**実測で当て直す値**
        （028 の 8.5 節。配布データで 12.94 mm ／ 0.1153 rad）。
    initial_accel_ratio : float
        **区間の頭で、いきなり出してよい |Δu| の割合**（`accel_pos` に対する比）。
        **1.0 が配布データの立ち上がり方である。** 5 次多項式の最小ジャークは
        ここが 0 に固定されるため、**先頭 8 コマを捨てて何もしない**
        （実測: 配布は 1 コマ目で |u|>0.2 に達するのに、028 版は 8 コマ目）。
    """

    limit_pos: float = 0.9375
    limit_rot: float = 0.30
    accel_pos: float = 0.25
    accel_rot: float = 0.05
    jerk_pos: float = 0.18
    jerk_rot: float = 0.046
    response_pos_m: float = 0.0128
    response_rot_rad: float = 0.1125
    initial_accel_ratio: float = 1.0


#: 配布データ（`libero_plus`）から測った既定の封筒。
#: 上限は 028 が使っている値をそのまま引き継ぎ（並進は配布データの端そのもの、
#: 回転は端 0.375 の内側 0.30）、差分の上限は上の実測の p99 に置いた。
LIBERO_PLUS = MotionEnvelope()


def envelope_from_environment(base: MotionEnvelope = LIBERO_PLUS) -> MotionEnvelope:
    """環境変数で上書きした封筒を返す。

    **028 が使っている `APPROACH_*` をそのまま読む。**同じ実験を回すときに
    変数名が 2 系統あると、片方だけ効いて結果が静かに嘘になる。
    Ruckig 側だけの摘み（差分の上限・立ち上がり）は `MOTION_*` で足す。
    """
    def value(name: str, default: float) -> float:
        return float(os.environ.get(name, default))

    return MotionEnvelope(
        limit_pos=value("APPROACH_LIMIT_POS", base.limit_pos),
        limit_rot=value("APPROACH_LIMIT_ROT", base.limit_rot),
        accel_pos=value("MOTION_ACCEL_POS", base.accel_pos),
        accel_rot=value("MOTION_ACCEL_ROT", base.accel_rot),
        jerk_pos=value("MOTION_JERK_POS", base.jerk_pos),
        jerk_rot=value("MOTION_JERK_ROT", base.jerk_rot),
        response_pos_m=value("APPROACH_RESPONSE_POS_M", base.response_pos_m),
        response_rot_rad=value("APPROACH_RESPONSE_ROT_RAD", base.response_rot_rad),
        initial_accel_ratio=value("MOTION_INITIAL_ACCEL_RATIO",
                                  base.initial_accel_ratio),
    )


def measure_envelope(episodes: list[np.ndarray], percentile: float = 99.0
                     ) -> dict[str, float]:
    """`action` 列の集まりから、封筒に入れる百分位を測る。

    Parameters
    ----------
    episodes : list of np.ndarray
        エピソードごとの `action` (T, >=6)。**差分はエピソードを跨がずに取る。**
    percentile : float
        上限として採る百分位。既定 99（外れ値 1% は封筒に入れない）。

    Returns
    -------
    dict
        `limit_pos` `limit_rot` `accel_pos` `accel_rot` `jerk_pos` `jerk_rot`。
    """
    stacked = np.concatenate([np.asarray(x, dtype=np.float64)[:, :6] for x in episodes])
    first = np.concatenate([np.diff(np.asarray(x, dtype=np.float64)[:, :6], axis=0)
                            for x in episodes if len(x) >= 2])
    second = np.concatenate([np.diff(np.asarray(x, dtype=np.float64)[:, :6], n=2, axis=0)
                             for x in episodes if len(x) >= 3])

    def top(values: np.ndarray) -> float:
        return float(np.percentile(np.abs(values).max(axis=1), percentile))

    return {"limit_pos": top(stacked[:, :3]), "limit_rot": top(stacked[:, 3:]),
            "accel_pos": top(first[:, :3]), "accel_rot": top(first[:, 3:]),
            "jerk_pos": top(second[:, :3]), "jerk_rot": top(second[:, 3:])}
