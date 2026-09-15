"""Ruckig による、**止まらずにつながる**通過点の生成。

**なぜ 5 次多項式の最小ジャーク（`waypoints.py`）では足りないのか。**
5 次多項式は両端で速度も加速度も 0 に固定される。これは「区間ごとに必ず
止まる」という意味なので、`rollout_phased` のように区間を 7 つ並べる作りでは
**1 本のエピソードに停止が 7 回入る。**配布データを実測すると停止の塊は
**1 本あたり 0.8 個**（掴む所だけ）なのに、028 版の自作データは **6.7 個**ある。

さらに悪いのは**立ち上がりである。**5 次多項式は先頭の加速度も 0 なので、
指令が育つのに何コマもかかる。実測（2026-09-11）:

    |u| が 0.2 に達するまで   配布 1 コマ（中央値）／ 028 版 8 コマ
    先頭 16 コマの |u| 中央   配布 0.15 0.21 0.34 0.48 0.58 0.69 0.76 0.81 …
                              028  0.00 0.00 0.01 0.03 0.05 0.08 0.12 0.16 …

**配布データは 1 コマ目から動いている。**OSC は位置サーボなので、指令を
いきなり出しても腕は自分の応答（時定数 3〜4 コマ）でなめらかに加速する。
**なめらかさは制御器がすでに持っているのに、5 次多項式がその上にもう一段
かけていた。**その結果、教師データの頭が「ほぼ何もしない」列になる——
Track3 の方策が序盤で経路を外れる（記録の 1〜4% しか辿れない）症状と形が合う。

**Ruckig が違うのは 2 点だけである。**

1. **入口と出口の速度・加速度を指定できる。**区間をまたいで速度を渡せるので、
   止まる必要のない継ぎ目で止まらない。
2. **時間最適である。**速度・加速度・躍度の上限を守りながら、いちばん早く着く。
   5 次多項式のように「山が平均の 1.875 倍」という損を払わない。

**単位の取り方。** Ruckig は「位置・速度・加速度・躍度」で話すが、ここでは
**位置を「指令 1 単位ぶんの移動量」で割った座標**に取り、**時間の単位を
1 コマ**に取る。こうすると **Ruckig の速度がそのまま OSC の指令になる**ので、
封筒（`envelope.py`）の値を単位変換なしで `max_velocity` などに渡せる。

**Ruckig が無ければ台形＋S 字の自前実装に落ちる。**`pip install ruckig` が
できない環境でも同じ形の通過点が出る（時間最適ではないぶん数コマ長い）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .envelope import LIBERO_PLUS, MotionEnvelope
from .rotation import relative_axisangle, rotate_along

try:
    import ruckig as _ruckig

    HAS_RUCKIG = True
    RUCKIG_VERSION = getattr(_ruckig, "__version__", "?")
except ImportError:
    _ruckig = None
    HAS_RUCKIG = False
    RUCKIG_VERSION = None

#: 並進 3 軸 ＋ 回転の測地線 1 軸。Ruckig はこの 4 つを時間同期させる。
NUM_DOF = 4


@dataclass
class MotionState:
    """区間の継ぎ目で持ち越す運動の状態。

    Attributes
    ----------
    pose : np.ndarray
        最後に配った通過点 (6,)。**次の呼び出しがここから続いているかの照合に使う。**
    velocity : np.ndarray
        指令の単位で見た速度 (3,)。並進 3 軸ぶん。
    acceleration : np.ndarray
        指令の単位で見た加速度 (3,)。
    angular_velocity : np.ndarray
        回転の速度を**軸角ベクトル (3,) として**持つ。次の区間は回転軸が
        変わるので、そこへ射影して使う。
    angular_acceleration : np.ndarray
        同じく回転の加速度 (3,)。
    """

    pose: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    angular_velocity: np.ndarray
    angular_acceleration: np.ndarray


def _plan_with_ruckig(distance: np.ndarray, entry_velocity: np.ndarray,
                      entry_acceleration: np.ndarray, exit_velocity: np.ndarray,
                      max_velocity: np.ndarray, max_acceleration: np.ndarray,
                      max_jerk: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ruckig で 4 自由度を時間同期して解き、1 コマごとの位置・速度・加速度を返す。

    Parameters
    ----------
    distance : np.ndarray
        各自由度が進む量 (4,)。**指令の単位**（1 コマぶんの指令が 1.0）。
    entry_velocity, entry_acceleration, exit_velocity : np.ndarray
        入口・出口の状態 (4,)。同じ単位・1 コマあたり。
    max_velocity, max_acceleration, max_jerk : np.ndarray
        自由度ごとの上限 (4,)。

    Returns
    -------
    position, velocity, acceleration : np.ndarray
        (N, 4)。**先頭は 1 コマ目（入口そのものは入らない）。**
    """
    otg = _ruckig.Ruckig(NUM_DOF, 1.0)
    inp = _ruckig.InputParameter(NUM_DOF)
    out = _ruckig.OutputParameter(NUM_DOF)
    inp.current_position = [0.0] * NUM_DOF
    inp.current_velocity = list(entry_velocity)
    inp.current_acceleration = list(entry_acceleration)
    inp.target_position = list(distance)
    inp.target_velocity = list(exit_velocity)
    inp.target_acceleration = [0.0] * NUM_DOF
    inp.max_velocity = list(max_velocity)
    inp.max_acceleration = list(max_acceleration)
    inp.max_jerk = list(max_jerk)

    positions, velocities, accelerations = [], [], []
    result = otg.update(inp, out)
    # **解けなかったら空を返す。**呼ぶ側が出口の速度を捨てて引き直す。
    # ここで黙って `out` を 1 点だけ返すと、初期化されていない値が
    # 通過点として渡り、腕がとんでもない所を目標にする。
    if result not in (_ruckig.Result.Working, _ruckig.Result.Finished):
        empty = np.zeros((0, NUM_DOF))
        return empty, empty, empty
    while result == _ruckig.Result.Working:
        positions.append(list(out.new_position))
        velocities.append(list(out.new_velocity))
        accelerations.append(list(out.new_acceleration))
        out.pass_to_input(inp)
        result = otg.update(inp, out)

    # **最後の 1 点は目標そのものに置く。**軌道の所要時間は 1 コマの整数倍とは
    # 限らないので、`Finished` のあとの `update()` は**終端を通り過ぎた点**を
    # 返す（出口に速度を残したときだけ起きる）。それをそのまま通過点にすると、
    # 行き過ぎてから目標へ引き戻す 1 コマが入り、**継ぎ目ごとに小さな逆走**になる。
    positions.append(list(distance))
    velocities.append(list(out.new_velocity))
    accelerations.append(list(out.new_acceleration))
    return (np.asarray(positions), np.asarray(velocities),
            np.asarray(accelerations))


def _s_curve_ramp(from_speed: float, to_speed: float, accel_cap: float,
                  jerk_cap: float) -> np.ndarray:
    """`from_speed` から `to_speed` へ、躍度を刻んで移る速度の列を返す。

    **上げる向きにも下げる向きにも使える。**行き過ぎないように、残りの速度差で
    止まれる加速度に毎コマ抑える。`to_speed` そのものは列に入る（`from_speed` は
    入らない）。差が無ければ空を返す。
    """
    gap = to_speed - from_speed
    if abs(gap) < 1e-12:
        return np.zeros(0)
    sign = 1.0 if gap > 0 else -1.0
    speeds: list[float] = []
    speed, accel = from_speed, 0.0
    for _ in range(1000):
        remaining = (to_speed - speed) * sign
        if remaining <= 1e-9:
            break
        stopping = np.sqrt(max(2.0 * jerk_cap * remaining, 0.0))
        accel = min(accel + jerk_cap, accel_cap, stopping)
        speed = speed + sign * accel
        speed = min(speed, to_speed) if sign > 0 else max(speed, to_speed)
        speeds.append(max(speed, 0.0))
    return np.asarray(speeds) if speeds else np.zeros(0)


def _plan_without_ruckig(distance: np.ndarray, entry_velocity: np.ndarray,
                         _entry_acceleration: np.ndarray, exit_velocity: np.ndarray,
                         max_velocity: np.ndarray, max_acceleration: np.ndarray,
                         max_jerk: np.ndarray,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ruckig が無いときの代わり。躍度を上限に張った形を支配軸で作り、全軸で共有する。

    **時間最適ではない。**所要コマ数がいちばん多い軸（支配軸）の速度の形を作り、
    残りの軸は同じ時間の割合で動かす（＝直線の経路になる）。Ruckig と違って
    軸ごとの上限を使い切らないぶん数コマ長い。**入口と出口の「速度」は守るが、
    入口の「加速度」は守らない**——`_entry_acceleration` を受け取るのは
    `_plan_with_ruckig` と同じ形で呼べるようにするためで、中では読んでいない。

    Parameters
    ----------
    distance, entry_velocity, exit_velocity : np.ndarray
        `_plan_with_ruckig` と同じ (4,)。指令の単位・1 コマあたり。
    _entry_acceleration : np.ndarray
        **使わない**（呼び口を揃えるためだけの引数）。
    max_velocity, max_acceleration, max_jerk : np.ndarray
        自由度ごとの上限 (4,)。

    Returns
    -------
    position, velocity, acceleration : np.ndarray
        (N, 4)。**先頭は 1 コマ目（入口そのものは入らない）。**
    """
    empty = np.zeros((0, NUM_DOF))
    with np.errstate(divide="ignore", invalid="ignore"):
        steps_by_axis = np.where(max_velocity > 0, np.abs(distance) / max_velocity, 0.0)
    lead = int(np.argmax(steps_by_axis))
    total = float(np.abs(distance[lead]))
    if total < 1e-12:
        return empty, empty, empty

    direction = float(np.sign(distance[lead]))
    entry_speed = max(float(entry_velocity[lead] * direction), 0.0)
    exit_speed = max(float(exit_velocity[lead] * direction), 0.0)
    speeds = _speed_profile(total, entry_speed, exit_speed,
                            float(max_velocity[lead]), float(max_acceleration[lead]),
                            float(max_jerk[lead]))
    if len(speeds) == 0:
        return empty, empty, empty

    # 支配軸の進み具合をそのまま全軸へ配る（直線の経路）。
    progress = np.cumsum(speeds) / total
    progress[-1] = 1.0
    position = np.outer(progress, distance)
    velocity = np.outer(speeds * direction / total, distance)
    acceleration = np.diff(velocity, axis=0, prepend=entry_velocity[None, :])
    return position, velocity, acceleration


def _with_cruise(ramp_up: np.ndarray, ramp_down: np.ndarray, total: float,
                 cruise_speed: float) -> np.ndarray:
    """加速と減速のあいだに、残りの距離を等速で埋める区間を挟む。

    **埋め方を 1 つにしておく。**山が収まる場合と、二分探索で山を下げた場合とで
    別々の継ぎ足し方をすると、**山のてっぺんだけ速度が飛ぶ**——なめらかにする
    ために作ったモジュールが、いちばん速い所に段差を作ることになる。
    """
    remainder = total - float(np.sum(ramp_up) + np.sum(ramp_down))
    if remainder <= 1e-12:
        return np.concatenate([ramp_up, ramp_down])
    steps = max(int(np.ceil(remainder / max(cruise_speed, 1e-12))), 1)
    return np.concatenate([ramp_up, np.full(steps, remainder / steps), ramp_down])


def _speed_profile(total: float, entry_speed: float, exit_speed: float,
                   speed_cap: float, accel_cap: float, jerk_cap: float
                   ) -> np.ndarray:
    """1 自由度ぶんの速度の列を作る。**最後の値はちょうど `exit_speed` になる。**

    加速 → 巡航 → 減速の 3 段。加速と減速は加速度を躍度で刻んだ S 字にする
    （台形＝加速度が段で変わる形は使わない）。距離が足りないときは、
    山の高さ（巡航速度）を二分探索で下げて収める。
    """
    def ramps(peak: float) -> tuple[np.ndarray, np.ndarray]:
        up = _s_curve_ramp(entry_speed, peak, accel_cap, jerk_cap)
        down = _s_curve_ramp(exit_speed, peak, accel_cap, jerk_cap)[::-1]
        return up, np.concatenate([down, [exit_speed]])

    up, down = ramps(speed_cap)
    if float(np.sum(up) + np.sum(down)) <= total:
        return _with_cruise(up, down, total, speed_cap)

    # 山が高すぎる。収まる高さを二分探索で探す（20 回で十分に詰まる）。
    low, high = max(entry_speed, exit_speed), speed_cap
    for _ in range(20):
        peak = 0.5 * (low + high)
        up, down = ramps(peak)
        if float(np.sum(up) + np.sum(down)) <= total:
            low = peak
        else:
            high = peak
    up, down = ramps(low)
    return _with_cruise(up, down, total, low)


def _parallel_scaled(distance: np.ndarray, caps: np.ndarray,
                     ratio: float = 1.0) -> np.ndarray:
    """`distance` に**平行**な向きで、どの成分も `caps × ratio` を超えない量を作る。

    **成分ごとに上限を置いてはいけない。**ほとんど動かない軸にも満額が乗り、
    その軸だけ行き過ぎて戻る。**いちばん上限に近づく軸がちょうど上限になる
    大きさ**を選べば、向きを保ったまま大きさだけを決められる。

    **区間の出口の速度と入口の加速度が、どちらもこれである**
    （`_parallel_exit_velocity`・`_parallel_onset_acceleration`）。
    動かない軸（`1e-9` 未満）は上限を見ない——見ると倍率が発散する。
    """
    if ratio <= 0.0:
        return np.zeros(len(distance))
    scales = [caps[i] / abs(distance[i])
              for i in range(len(distance)) if abs(distance[i]) > 1e-9]
    if not scales:
        return np.zeros(len(distance))
    return distance * min(scales) * ratio


def _parallel_exit_velocity(distance: np.ndarray, max_velocity: np.ndarray,
                            ratio: float) -> np.ndarray:
    """区間の出口に残す速度を、**進行方向に平行**に、上限の `ratio` 倍で作る。

    **並進と回転をまとめて 1 つの向きとして扱う**（入口の加速度と違う点である。
    出口では経路の向きを保つことが要り、入口では並進と回転が別々に立ち上がる）。
    """
    return _parallel_scaled(distance, max_velocity, ratio)


def _parallel_onset_acceleration(distance: np.ndarray, onset_pos: float,
                                 onset_rot: float) -> np.ndarray:
    """止まった状態から始める区間の入口加速度を、**進行方向に平行**に作る。

    **成分ごとに `+onset` を置いてはいけない。**符号が無いと Ruckig は
    「いま `+x+y+z` へ加速中である」と受け取り、目標が逆側にあっても
    **まずその加速を打ち消してから向きを変える。**実測（2026-09-13・寄せる区間 160 本）で
    **160 本すべてが頭の 2〜3 コマを行き先と逆向きに出し**、3 → 4 コマ目で
    符号が反転していた（並進の符号反転 7.19%・配布データ 0.14%）。

    **符号だけ付けても足りない**ので `_parallel_scaled` に乗せる。
    **並進と回転は別々に掛ける**——上限が別（`accel_pos` と `accel_rot`）で、
    まとめると回す量の少ない区間で並進の加速度まで削れてしまう。
    """
    return np.concatenate([
        _parallel_scaled(distance[:3], np.full(3, onset_pos)),
        _parallel_scaled(distance[3:], np.array([onset_rot]))])


class PoseTrajectoryPlanner:
    """区間をまたいで速度を持ち越す、姿勢の通過点の生成器。

    **`rollout_phased` の `move_to` が区間ごとに呼ぶことを想定している。**
    直前の区間の出口の速度を覚えていて、**同じ場所から続きが呼ばれたら
    その速度から始める。**離れた場所から呼ばれたら（＝別の場面が始まった）
    止まった状態に戻す。この照合があるので、呼ぶ側は状態を持たなくてよい。

    Parameters
    ----------
    envelope : MotionEnvelope
        指令の封筒。
    chain_tolerance_steps : float
        「続きである」と見なす距離を、1 コマぶんの移動量の何倍で測るか。
        **既定 6 は閉ループの遅れより広く取ってある。**指令は「目標 − 今の姿勢」
        なので、腕は目標より常に 3 コマぶん（全速で約 35 mm）後ろにいる。
        ここを詰めすぎると、続きなのに毎回止まった状態に戻る。
    """

    def __init__(self, envelope: MotionEnvelope = LIBERO_PLUS,
                 chain_tolerance_steps: float = 6.0) -> None:
        self.envelope = envelope
        self.chain_tolerance_steps = float(chain_tolerance_steps)
        self.state: MotionState | None = None
        self.observed: list[np.ndarray] = []
        self.num_planned = 0
        self.num_chained = 0

    def reset(self) -> None:
        """エピソードの切れ目で呼ぶ。持ち越しと観測を捨てる。"""
        self.state = None
        self.observed = []

    def observe(self, pose: np.ndarray) -> None:
        """**実際に動いた腕の姿勢**を 1 コマぶん渡す。

        **これがあると、持ち越した「計画上の速度」ではなく実測の速度から
        次の区間を組める。**区間の切れ目で比例制御が数コマ挟まったり、指を
        閉じるあいだ止まっていたりすると計画上の速度は嘘になるが、実測なら
        そのまま正しい——止まっているときは 0 が入るので、**止まる必要のある
        継ぎ目ではちゃんと止まったところから組み直す。**Ruckig の本来の
        使い方（毎周期、いまの状態から組み直す）に近づく。
        """
        self.observed.append(np.asarray(pose, dtype=np.float64).copy())
        if len(self.observed) > 3:
            self.observed.pop(0)

    def _measured_state(self) -> dict[str, np.ndarray] | None:
        """観測から並進の速度と加速度を出す（指令の単位・1 コマあたり）。"""
        if len(self.observed) < 2:
            return None
        scale = self.envelope.response_pos_m
        velocity = (self.observed[-1][:3] - self.observed[-2][:3]) / scale
        if len(self.observed) < 3:
            acceleration = np.zeros(3)
        else:
            previous = (self.observed[-2][:3] - self.observed[-3][:3]) / scale
            acceleration = velocity - previous
        return {"velocity": velocity, "acceleration": acceleration}

    def waypoints(self, current_pose: np.ndarray, target_pose: np.ndarray,
                  exit_speed_ratio: float = 0.0,
                  next_target: np.ndarray | None = None) -> list[np.ndarray]:
        """`current_pose` から `target_pose` までの通過点を 1 コマ 1 点で返す。

        **出口に速さを残すかどうかは、次にどこへ行くかで決まる。**掴む・置く
        ように「そこで終わる」区間は止まり、通過するだけの区間は流したまま
        次へ渡す。**呼ぶ側はそれを知っている**ので、`next_target` を渡せば
        こちらで向きまで決められる——残った速度が次の行き先を向くので、
        継ぎ目で進行方向が折れない。

        Parameters
        ----------
        current_pose, target_pose : np.ndarray
            姿勢 (6,)。前 3 つが位置 [m]、後ろ 3 つが軸角。
        exit_speed_ratio : float
            出口で残す速さを、上限に対する割合で指定する。**0 で停止**
            （掴む・置くなど、止まる必要のある区間）。
        next_target : np.ndarray or None
            **次の区間の行き先** (6,)。渡すと出口の速度がそこを向く。
            `None` なら今の進行方向のまま出る。**`exit_speed_ratio=0` の
            ときは何を渡しても止まる。**

        Returns
        -------
        list of np.ndarray
            通過点 (6,) の列。最後の要素は `target_pose` と同じ姿勢である。
        """
        envelope = self.envelope
        current = np.asarray(current_pose, dtype=np.float64)
        target = np.asarray(target_pose, dtype=np.float64)

        delta_pos = target[:3] - current[:3]
        axis_angle_rel = relative_axisangle(current[3:6], target[3:6])
        angle_rad = float(np.linalg.norm(axis_angle_rel))
        rotation_axis = (axis_angle_rel / angle_rad if angle_rad > 1e-9
                         else np.zeros(3))

        self.num_planned += 1

        # 指令の単位へ。位置は「指令 1 単位ぶんの移動量」で割る。
        distance = np.concatenate([delta_pos / envelope.response_pos_m,
                                   [angle_rad / envelope.response_rot_rad]])
        entry = self._entry_state(current, rotation_axis, distance)
        if float(np.abs(distance).max()) < 1e-9:
            return [target.copy()]

        # **回転の上限は軸の向きを見て決める。** `osc_command` が見るのは
        # 軸角ベクトルの**成分**なので、軸が斜めなら測地線はそのぶん速く回れる。
        # **回さない区間で上限を膨らませない。** 軸が零ベクトルのときに
        # `1e-6` で割ると、回転の上限だけが 10^5 倍の値になる。進む量は 0 
        # なので害は出ないはずだが、時間同期の解がその桁で荒れる。
        largest_axis_component = (float(np.abs(rotation_axis).max())
                                  if angle_rad > 1e-9 else 1.0)
        max_velocity = np.array([envelope.limit_pos] * 3
                                + [envelope.limit_rot / largest_axis_component])
        max_acceleration = np.array([envelope.accel_pos] * 3
                                    + [envelope.accel_rot / largest_axis_component])
        max_jerk = np.array([envelope.jerk_pos] * 3
                            + [envelope.jerk_rot / largest_axis_component])

        exit_direction = distance
        if next_target is not None:
            onward = np.asarray(next_target, dtype=np.float64)[:3] - target[:3]
            if float(np.linalg.norm(onward)) > 1e-9:
                exit_direction = np.concatenate([onward / envelope.response_pos_m,
                                                 [0.0]])
        exit_velocity = _parallel_exit_velocity(exit_direction, max_velocity,
                                                float(exit_speed_ratio))
        planner = _plan_with_ruckig if HAS_RUCKIG else _plan_without_ruckig
        position, velocity, acceleration = planner(
            distance, entry["velocity"], entry["acceleration"], exit_velocity,
            max_velocity, max_acceleration, max_jerk)
        if len(position) == 0 and float(np.abs(exit_velocity).max()) > 0.0:
            # 出口の速度を残したまま解けない距離だった。止まる形で引き直す。
            position, velocity, acceleration = planner(
                distance, entry["velocity"], entry["acceleration"],
                np.zeros(NUM_DOF), max_velocity, max_acceleration, max_jerk)
        if len(position) == 0:
            return [target.copy()]

        waypoints = self._to_poses(current, axis_angle_rel, distance,
                                   position, target)
        self._remember(waypoints[-1], velocity[-1], acceleration[-1], rotation_axis)
        return waypoints

    def _entry_state(self, current: np.ndarray, rotation_axis: np.ndarray,
                     distance: np.ndarray) -> dict[str, np.ndarray]:
        """持ち越した速度を、この区間の自由度へ移す。続きでなければ止まった状態。"""
        envelope = self.envelope
        # **区間の頭でいきなり出してよい加速度。** 5 次多項式はここが 0 で、
        # そのために先頭 8 コマを捨てていた（モジュールの説明を見よ）。
        onset = envelope.accel_pos * envelope.initial_accel_ratio
        rest = {"velocity": np.zeros(NUM_DOF),
                "acceleration": _parallel_onset_acceleration(
                    distance, onset, envelope.accel_rot * envelope.initial_accel_ratio)}
        translation = self._translation_entry(current)
        if translation is None:
            return rest

        if float(np.linalg.norm(translation["velocity"])) > 1e-6:
            self.num_chained += 1               # 速度を持ったまま始めた区間の数
        angular, angular_accel = self._angular_entry(rotation_axis)
        # **加速度は「入口で出してよい量」以上には持ち込まない。**
        return {
            "velocity": np.concatenate([translation["velocity"], [angular]]),
            "acceleration": np.concatenate([
                np.clip(translation["acceleration"], -onset, onset), [angular_accel]]),
        }

    def _translation_entry(self, current: np.ndarray) -> dict[str, np.ndarray] | None:
        """区間の入口の並進の速度・加速度を出す。止まって始めるなら `None`。

        **実測（`observe()`）が最優先である。**止まっていれば 0 が入り、動いて
        いればその速度が入るので、止まる必要のある継ぎ目とそうでない継ぎ目を
        呼ぶ側が区別しなくてよい。実測が無いときだけ、直前の計画の出口を
        持ち越す——そのときは「続きかどうか」を距離で見分けるしかない。
        """
        measured = self._measured_state()
        if measured is not None:
            return measured
        if self.state is None:
            return None
        gap = float(np.linalg.norm(current[:3] - self.state.pose[:3]))
        allowed = (self.chain_tolerance_steps * self.envelope.limit_pos
                   * self.envelope.response_pos_m)
        if gap > allowed:                                # 別の場面が始まっている
            self.state = None
            return None
        return {"velocity": self.state.velocity,
                "acceleration": self.state.acceleration}

    def _angular_entry(self, rotation_axis: np.ndarray) -> tuple[float, float]:
        """持ち越した回転の速度・加速度を、この区間の軸へ射影する。"""
        if self.state is None:
            return 0.0, 0.0
        return (float(np.dot(self.state.angular_velocity, rotation_axis)),
                float(np.dot(self.state.angular_acceleration, rotation_axis)))

    def _to_poses(self, current: np.ndarray, axis_angle_rel: np.ndarray,
                  distance: np.ndarray, position: np.ndarray,
                  target: np.ndarray) -> list[np.ndarray]:
        """解いた進み具合 (N, 4) を姿勢の列に戻す。

        **並進は軸ごとの解をそのまま使う**（直線に潰さない）。持ち越した速度が
        今の進行方向と違う向きを向いているとき、軸ごとの解は**曲がって**つながる。
        直線に潰すと、その曲がりが「進行方向の急な折れ」に化けて戻ってしまう。
        """
        rotation_span = float(distance[3])
        waypoints: list[np.ndarray] = []
        for row in position:
            ratio_rot = row[3] / rotation_span if abs(rotation_span) > 1e-9 else 1.0
            waypoints.append(np.concatenate([
                current[:3] + row[:3] * self.envelope.response_pos_m,
                rotate_along(current[3:6], axis_angle_rel, ratio_rot),
            ]))
        waypoints[-1] = target.copy()                    # 端数を残さない
        return waypoints

    def _remember(self, last_pose: np.ndarray, velocity: np.ndarray,
                  acceleration: np.ndarray, rotation_axis: np.ndarray) -> None:
        """区間の出口の状態を、次の呼び出しのために持ち越す。"""
        self.state = MotionState(
            pose=np.asarray(last_pose, dtype=np.float64).copy(),
            velocity=np.asarray(velocity[:3], dtype=np.float64).copy(),
            acceleration=np.asarray(acceleration[:3], dtype=np.float64).copy(),
            angular_velocity=rotation_axis * float(velocity[3]),
            angular_acceleration=rotation_axis * float(acceleration[3]),
        )
