"""通過点の作り方を、シムを回さずに指令の形で見比べる道具。

**なぜシム無しで比べられるか。** OSC（`control_delta=True`）は位置サーボなので、
1 コマの挙動は次の 2 式でよく近似できる:

    u[t]      = clip((p_ref[t+1] - p_arm[t]) / OUT_MAX, ±1)
    p_arm[t+1] = p_arm[t] + response * u[t]

`response / OUT_MAX = 0.0128 / 0.05 = 0.256` が閉ループの利得である。
自作データの実測（`work/probe_028/L`）で 1 コマの実変位 ÷ 指令 が
**12.4〜13.1 mm/単位**に収まっており、この 1 次モデルと合う。
**あくまで摘みを絞るための近似で、合否はシムで測ること。**

    python3 solution/tools/motion/compare_waypoint_profiles.py \
        --dataset work/artifacts/libero_plus --episodes 200

Reads:
  - work/artifacts/libero_plus/data/chunk-*/file-*.parquet（`--dataset` のとき）
Generates:
  - なし（標準出力に表を出すだけ）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):                            # 直接叩かれたとき
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from motion.envelope import LIBERO_PLUS, MotionEnvelope
    from motion.otg import HAS_RUCKIG, RUCKIG_VERSION, PoseTrajectoryPlanner
    from motion.smoothness import format_table, load_dataset_episodes
    from motion.waypoints import min_jerk_waypoints
else:
    from .envelope import LIBERO_PLUS, MotionEnvelope
    from .otg import HAS_RUCKIG, RUCKIG_VERSION, PoseTrajectoryPlanner
    from .smoothness import format_table, load_dataset_episodes
    from .waypoints import min_jerk_waypoints

#: OSC の `output_max`（並進）。`replay_video_probe.OUT_MAX_POS` と同じ値。
OUT_MAX_POS = 0.05
#: OSC の `output_max`（回転）。
OUT_MAX_ROT = 0.5

#: 逆向きデモの 1 サイクルを、実測に近い距離で並べたもの。
#: `rollout_phased` の A0 → A → B → L → C → D → E に対応する。
DEFAULT_SEGMENTS: list[dict] = [
    {"name": "A0 手前へ", "translation": [0.16, -0.10, 0.05], "rotation_deg": 25.0},
    {"name": "A  掴む姿勢へ", "translation": [0.035, 0.02, -0.04], "rotation_deg": 8.0},
    {"name": "B  閉じる", "translation": [0.0, 0.0, 0.0], "rotation_deg": 0.0,
     "hold_steps": 10},
    {"name": "L  持ち上げ", "translation": [0.0, 0.0, 0.08], "rotation_deg": 0.0},
    {"name": "C  復路", "translation": [-0.14, 0.24, -0.02], "rotation_deg": 35.0},
    {"name": "D  下ろす", "translation": [0.0, 0.0, -0.03], "rotation_deg": 6.0},
    {"name": "E  引く", "translation": [-0.06, -0.05, 0.06], "rotation_deg": 10.0,
     "hold_steps": 10},
]


def _axis_angle(rotation_deg: float, seed: int) -> np.ndarray:
    """再現できる向きの回転軸を作る（大きさは指定の角度）。"""
    generator = np.random.default_rng(seed)
    axis = generator.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis * np.radians(rotation_deg)


def _simulate(waypoints: list[np.ndarray], arm_pose: np.ndarray,
              envelope: MotionEnvelope) -> tuple[np.ndarray, np.ndarray]:
    """通過点の列を 1 次サーボに通し、出る指令と腕の姿勢の列を返す。

    Parameters
    ----------
    waypoints : list of np.ndarray
        1 コマに 1 点ずつ渡す目標 (6,)。
    arm_pose : np.ndarray
        いまの腕の姿勢 (6,)。
    envelope : MotionEnvelope
        指令 1 単位あたりの実移動量を使う。

    Returns
    -------
    commands : np.ndarray
        出た指令 (N, 6)。
    arm_pose : np.ndarray
        最後の腕の姿勢 (6,)。
    """
    commands = []
    pose = np.asarray(arm_pose, dtype=np.float64).copy()
    for waypoint in waypoints:
        translation = np.clip((waypoint[:3] - pose[:3]) / OUT_MAX_POS, -1.0, 1.0)
        rotation = np.clip((waypoint[3:6] - pose[3:6]) / OUT_MAX_ROT, -1.0, 1.0)
        commands.append(np.concatenate([translation, rotation]))
        # 腕は指令 1 単位あたり `response_*` だけ動く（`OUT_MAX_*` は切るときだけ）。
        pose[:3] += translation * envelope.response_pos_m
        pose[3:6] += rotation * envelope.response_rot_rad
    return np.asarray(commands), pose


def _run_episode(make_waypoints, envelope: MotionEnvelope,
                 segments: list[dict], seed: int) -> np.ndarray:
    """1 本ぶんの指令列を作る。`make_waypoints(current, target, last)` を呼ぶ。"""
    arm_pose = np.zeros(6)
    target_pose = np.zeros(6)
    commands: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        target_pose = target_pose.copy()
        target_pose[:3] += np.asarray(segment["translation"], dtype=np.float64)
        target_pose[3:6] += _axis_angle(segment["rotation_deg"], seed * 100 + index)
        moving = (np.linalg.norm(segment["translation"]) > 1e-9
                  or segment["rotation_deg"] > 1e-9)
        if moving:
            waypoints = make_waypoints(arm_pose, target_pose,
                                       index == len(segments) - 1)
            emitted, arm_pose = _simulate(waypoints, arm_pose, envelope)
            commands.append(emitted)
        for _ in range(segment.get("hold_steps", 0)):    # 指を開閉するあいだ保持
            emitted, arm_pose = _simulate([target_pose], arm_pose, envelope)
            commands.append(emitted)
    return np.concatenate(commands)


def _min_jerk_episodes(envelope: MotionEnvelope, segments: list[dict],
                       count: int) -> list[np.ndarray]:
    """5 次多項式（028 の作り方）で作った指令列。"""
    def make(current, target, _is_last):
        return min_jerk_waypoints(current, target, envelope)

    return [_run_episode(make, envelope, segments, seed) for seed in range(count)]


def _ruckig_episodes(envelope: MotionEnvelope, segments: list[dict], count: int,
                     exit_speed_ratio: float) -> list[np.ndarray]:
    """Ruckig（区間をまたいで速度を持ち越す作り方）で作った指令列。"""
    episodes = []
    for seed in range(count):
        planner = PoseTrajectoryPlanner(envelope)

        def make(current, target, is_last, planner=planner):
            return planner.waypoints(current, target,
                                     0.0 if is_last else exit_speed_ratio)

        episodes.append(_run_episode(make, envelope, segments, seed))
    return episodes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="work/artifacts/libero_plus",
                        help="配布データの LeRobot ディレクトリ（比較の基準）")
    parser.add_argument("--episodes", type=int, default=200,
                        help="配布データから取る本数")
    parser.add_argument("--synthetic", type=int, default=40,
                        help="作り方ごとに合成する本数")
    parser.add_argument("--exit-speed", type=float, default=0.5,
                        help="Ruckig が区間の出口に残す速さ（上限に対する割合）")
    parser.add_argument("--no-dataset", action="store_true",
                        help="配布データを読まない（parquet が無いホスト向け）")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    envelope = LIBERO_PLUS
    print(f"ruckig: {'あり ' + str(RUCKIG_VERSION) if HAS_RUCKIG else 'なし（自前の S 字に落ちる）'}")
    print(f"封筒: {envelope}\n")

    named: dict[str, list[np.ndarray]] = {}
    if not args.no_dataset:
        named["配布"] = load_dataset_episodes(args.dataset, args.episodes)
    named["5次多項式"] = _min_jerk_episodes(envelope, DEFAULT_SEGMENTS, args.synthetic)
    named["Ruckig"] = _ruckig_episodes(envelope, DEFAULT_SEGMENTS, args.synthetic,
                                       args.exit_speed)
    print(format_table(named))

    print("\n先頭 16 コマの並進ノルム（中央値）")
    for name, episodes in named.items():
        head = np.stack([np.linalg.norm(x[:16, :3], axis=1) for x in episodes
                         if len(x) >= 16])
        print(f"  {name:>10}: " + " ".join(f"{value:.2f}" for value in np.median(head, axis=0)))


if __name__ == "__main__":
    main()
