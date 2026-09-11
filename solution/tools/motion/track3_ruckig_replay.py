"""028 の逆向きデモ生成を、**通過点の作り方だけ** Ruckig に差し替えて回す。

**`replay_video_probe.py` には 1 行も書き込まない。**あのファイルは 028 の
セッションが持っていて、これまで通り動く必要がある。ここでは実行時に
`replay_video_probe._approach_waypoints` を共有モジュールの生成器に
入れ替えてから、`gripper_shift_sweep.py` をそのまま走らせる。
**差し替えを外せば 028 の作りに戻る**ので、同じコマンドで 1 変数の比較ができる。

**何が変わるか。**5 次多項式（028）は区間の両端で速度も加速度も 0 に固定される。
`rollout_phased` は 1 サイクルを 7 区間に割るので、**1 本に停止が 6〜7 回入る**
（配布データは 1 回）。さらに先頭の加速度が 0 なので、**エピソードの頭 8 コマが
ほぼ何もしない列**になる（配布データは 1 コマ目から動いている）。Ruckig は
入口・出口の速度を指定できるので、この 2 つが消える。

摘み（環境変数）:

    MOTION_PLANNER=ruckig|minjerk   既定 ruckig。minjerk は 028 と同じ通過点を
                                    共有モジュール経由で作る（差し替えの検算用）
    MOTION_EXIT_SPEED=0.25          区間の出口に残す速さ（上限に対する割合）。
                                    0 にすると区間ごとに止まる
    MOTION_ACCEL_POS / _ROT         指令が 1 コマで変わってよい量
    MOTION_JERK_POS / _ROT          その 1 階差分
    MOTION_INITIAL_ACCEL_RATIO      区間の頭でいきなり出してよい加速度の割合
    APPROACH_*                      028 と同じ変数をそのまま読む

使い方（`run_ruckig_replay.sh` が組み立てる）:

    python3 -u /tools/motion/track3_ruckig_replay.py --ep-npz ... --out /out

Reads:
  - `gripper_shift_sweep.py` が読むものと同じ（順方向デモの npz・bddl・init）
Generates:
  - `gripper_shift_sweep.py` が書くものと同じ（`--write-npz` の npz・json）
"""
from __future__ import annotations

import os
import runpy
from dataclasses import dataclass

import numpy as np

import replay_video_probe
from motion.envelope import MotionEnvelope, envelope_from_environment
from motion.otg import HAS_RUCKIG, RUCKIG_VERSION, PoseTrajectoryPlanner
from motion.waypoints import min_jerk_waypoints

#: 区間の出口に残す速さ（上限に対する割合）。オフラインの比較で決めた既定。
DEFAULT_EXIT_SPEED_RATIO = 0.25


@dataclass(frozen=True)
class ReplaySettings:
    """この差し込みの摘み。**取り込み時に 1 か所で読む**（import 時ではない）。

    import 時に `os.environ` を読むと、**読んだ値を後から変えられず、
    `_print_header` が出す値と実際に使う値がずれ得る。**
    """

    planner_name: str
    exit_speed_ratio: float

    @staticmethod
    def from_environment() -> "ReplaySettings":
        """`MOTION_PLANNER` と `MOTION_EXIT_SPEED` から組む。"""
        return ReplaySettings(
            planner_name=os.environ.get("MOTION_PLANNER", "ruckig").lower(),
            exit_speed_ratio=float(os.environ.get("MOTION_EXIT_SPEED",
                                                  DEFAULT_EXIT_SPEED_RATIO)),
        )


def _install(envelope: MotionEnvelope, planner: PoseTrajectoryPlanner,
             settings: ReplaySettings) -> None:
    """`replay_video_probe` の通過点生成を差し替え、エピソードごとに状態を戻す。

    差し替えるのは 3 か所である。

    `_approach_waypoints`
        通過点そのもの。`MOTION_PLANNER=minjerk` なら 028 と同じ列を返す。
    `osc_command`
        **腕が実際にどこに居るかを 1 コマずつ生成器へ渡すため**に包む。
        指令の作り方には触らない。これがあると、区間の切れ目に比例制御が
        挟まったり指を閉じるあいだ止まっていたりしても、**次の区間を
        「いま本当に出ている速度」から組める。**
    `rollout_phased`
        エピソードの頭で持ち越しを捨てる。
    """
    original_rollout = replay_video_probe.rollout_phased
    original_command = replay_video_probe.osc_command

    def approach_waypoints(current_pose: np.ndarray, target_pose: np.ndarray,
                           *_args, **_kwargs) -> list[np.ndarray]:
        """`_approach_waypoints` と同じ呼ばれ方をする差し替え先。"""
        if settings.planner_name == "minjerk":
            return min_jerk_waypoints(current_pose, target_pose, envelope)
        return planner.waypoints(current_pose, target_pose,
                                 settings.exit_speed_ratio)

    def osc_command(obs: dict, target: np.ndarray, grip: float,
                    gain: float = 1.0,
                    rot_limit: float | None = None) -> np.ndarray:
        """指令はそのまま。**撃つ前の腕の姿勢だけ生成器へ渡す。**"""
        planner.observe(np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            replay_video_probe.eef_axisangle(obs)]))
        return original_command(obs, target, grip, gain, rot_limit)

    def rollout_phased(*args, **kwargs):
        """エピソードの頭で持ち越しを捨ててから、元の再生をそのまま呼ぶ。"""
        planner.reset()
        return original_rollout(*args, **kwargs)

    replay_video_probe._approach_waypoints = approach_waypoints
    replay_video_probe.osc_command = osc_command
    replay_video_probe.rollout_phased = rollout_phased


def _print_header(envelope: MotionEnvelope, settings: ReplaySettings) -> None:
    """どの設定で回すかを 1 か所にまとめて出す。"""
    if settings.planner_name == "ruckig" and not HAS_RUCKIG:
        print("[motion] ruckig が入っていない。自前の S 字に落ちる", flush=True)
    print(f"[motion] planner={settings.planner_name} ruckig={RUCKIG_VERSION} "
          f"exit_speed={settings.exit_speed_ratio}", flush=True)
    print(f"[motion] {envelope}", flush=True)


def main() -> None:
    envelope = envelope_from_environment()
    settings = ReplaySettings.from_environment()
    planner = PoseTrajectoryPlanner(envelope)
    _print_header(envelope, settings)
    _install(envelope, planner, settings)
    runpy.run_module("gripper_shift_sweep", run_name="__main__")
    print(f"[motion] 区間 {planner.num_planned} 本を計画し、"
          f"うち {planner.num_chained} 本は速度を持ったまま始めた", flush=True)


if __name__ == "__main__":
    main()
