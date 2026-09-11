"""共有の通過点生成（`solution/tools/motion/`）の試験。

**確かめるのは 3 つである。**

`TestParityWith028`
    切り出した 5 次多項式・記録の割り直しが、`replay_video_probe.py`（028）の
    `_approach_waypoints` / `_resample_segment` と**同じ通過点を返すか。**
    ここが崩れると、共有化そのものが 028 の結果を静かに変えてしまう。
    **robosuite が要る**ので、この機械の venv では skip になる。常駐コンテナで:

        bash solution/tools/motion/run_motion_tests.sh

`TestRuckigProfile`
    Ruckig 側が**封筒を破らないか・区間の頭で止まらないか・出口の速度を
    持ち越すか。**robosuite が無くても動く（`rotation.py` が numpy に落ちる）。

`TestSmoothness`
    物差しそのもの。**停止の塊と立ち上がりを、作った列で数えられるか。**
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "tools"))

from motion.envelope import LIBERO_PLUS, MotionEnvelope, measure_envelope
from motion.otg import HAS_RUCKIG, PoseTrajectoryPlanner
from motion.smoothness import onset_steps, sign_flip_gated, stop_clusters
from motion.waypoints import min_jerk_waypoints, resample_segment

try:
    import replay_video_probe

    HAS_PROBE = True
except Exception:                                        # robosuite が無い機械
    replay_video_probe = None
    HAS_PROBE = False


def _random_poses(count: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """試験用の姿勢の組を作る。位置は数十 cm、向きは数十度ぶら下がる。"""
    generator = np.random.default_rng(seed)
    pairs = []
    for _ in range(count):
        start = np.concatenate([generator.normal(size=3) * 0.2,
                                generator.normal(size=3) * 0.8])
        end = np.concatenate([generator.normal(size=3) * 0.2,
                              generator.normal(size=3) * 0.8])
        pairs.append((start, end))
    return pairs


@pytest.mark.skipif(not HAS_PROBE,
                    reason="robosuite が無い（常駐コンテナで打つこと）")
class TestParityWith028:
    """028 の実装と同じ通過点を返すこと。"""

    def test_min_jerk_matches(self) -> None:
        for start, end in _random_poses(200, seed=0):
            mine = np.asarray(min_jerk_waypoints(start, end, LIBERO_PLUS))
            theirs = np.asarray(replay_video_probe._approach_waypoints(start, end))
            assert mine.shape == theirs.shape
            # 位置は完全一致。向きは robosuite が内部で float32 に落とすぶんだけずれる
            # （最大 1.5e-6 rad ＝ 8.5e-5 度。到着判定の許容は 5 度）。
            assert np.abs(mine[:, :3] - theirs[:, :3]).max() == 0.0
            assert np.abs(mine[:, 3:] - theirs[:, 3:]).max() < 1e-5

    def test_resample_matches(self) -> None:
        for start, end in _random_poses(200, seed=1):
            mine = np.asarray(resample_segment(start, end, LIBERO_PLUS))
            theirs = np.asarray(replay_video_probe._resample_segment(start, end))
            assert mine.shape == theirs.shape
            assert np.abs(mine[:, :3] - theirs[:, :3]).max() == 0.0
            assert np.abs(mine[:, 3:] - theirs[:, 3:]).max() < 1e-5


class TestRuckigProfile:
    """Ruckig 側の通過点が、封筒を守りつつ止まらないこと。"""

    @staticmethod
    def _step_commands(waypoints: list[np.ndarray], start: np.ndarray,
                       envelope: MotionEnvelope) -> np.ndarray:
        """通過点の間隔を、指令の単位に直した列にする。"""
        poses = np.vstack([start[None, :], np.asarray(waypoints)])
        return np.diff(poses[:, :3], axis=0) / envelope.response_pos_m

    def test_stays_inside_envelope(self) -> None:
        planner = PoseTrajectoryPlanner(LIBERO_PLUS)
        start = np.zeros(6)
        target = np.array([0.20, -0.12, 0.06, 0.0, 0.0, 0.0])
        commands = self._step_commands(planner.waypoints(start, target), start,
                                       LIBERO_PLUS)
        assert np.abs(commands).max() <= LIBERO_PLUS.limit_pos + 1e-6
        assert np.abs(np.diff(commands, axis=0)).max() <= LIBERO_PLUS.accel_pos + 1e-6

    def test_reaches_the_target(self) -> None:
        planner = PoseTrajectoryPlanner(LIBERO_PLUS)
        start = np.zeros(6)
        target = np.array([0.20, -0.12, 0.06, 0.3, -0.2, 0.1])
        assert np.allclose(planner.waypoints(start, target)[-1], target)

    def test_starts_moving_immediately(self) -> None:
        """**5 次多項式より速く立ち上がること。**先頭が空の列だと方策が動かない。"""
        start = np.zeros(6)
        target = np.array([0.20, 0.0, 0.0, 0.0, 0.0, 0.0])
        ruckig = self._step_commands(
            PoseTrajectoryPlanner(LIBERO_PLUS).waypoints(start, target), start,
            LIBERO_PLUS)
        min_jerk = self._step_commands(min_jerk_waypoints(start, target, LIBERO_PLUS),
                                       start, LIBERO_PLUS)
        reach = lambda series: int(np.argmax(np.abs(series[:, 0]) > 0.2))
        assert reach(ruckig) < reach(min_jerk)
        assert reach(ruckig) <= 3

    def test_carries_speed_between_segments(self) -> None:
        """続けて呼ばれた区間の継ぎ目で、指令が 0 に落ちないこと。"""
        planner = PoseTrajectoryPlanner(LIBERO_PLUS)
        first_target = np.array([0.16, 0.0, 0.0, 0.0, 0.0, 0.0])
        second_target = np.array([0.30, 0.02, 0.0, 0.0, 0.0, 0.0])
        first = planner.waypoints(np.zeros(6), first_target, exit_speed_ratio=0.25)
        second = planner.waypoints(first[-1], second_target, exit_speed_ratio=0.0)
        joint = self._step_commands(second, first[-1], LIBERO_PLUS)
        assert planner.num_chained == 1
        assert np.linalg.norm(joint[0]) > 0.15

    def test_zero_exit_speed_stops(self) -> None:
        """出口の速さを 0 にしたら、ちゃんと止まること。"""
        planner = PoseTrajectoryPlanner(LIBERO_PLUS)
        waypoints = planner.waypoints(np.zeros(6),
                                      np.array([0.16, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert np.linalg.norm(planner.state.velocity) < 1e-6

    def test_reset_drops_the_carry(self) -> None:
        planner = PoseTrajectoryPlanner(LIBERO_PLUS)
        planner.waypoints(np.zeros(6), np.array([0.16, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.25)
        planner.reset()
        assert planner.state is None

    def test_distant_call_does_not_chain(self) -> None:
        """遠い場所から呼ばれたら、別の場面として止まった状態に戻すこと。"""
        planner = PoseTrajectoryPlanner(LIBERO_PLUS)
        planner.waypoints(np.zeros(6), np.array([0.16, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.25)
        planner.waypoints(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                          np.array([1.1, 1.0, 1.0, 0.0, 0.0, 0.0]), 0.25)
        assert planner.num_chained == 0

    @pytest.mark.skipif(not HAS_RUCKIG, reason="ruckig が入っていない")
    def test_is_shorter_than_min_jerk(self) -> None:
        """時間最適なので、同じ封筒なら 5 次多項式より短いこと。"""
        start, target = np.zeros(6), np.array([0.20, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert (len(PoseTrajectoryPlanner(LIBERO_PLUS).waypoints(start, target))
                < len(min_jerk_waypoints(start, target, LIBERO_PLUS)))


class TestSmoothness:
    """物差しそのもの。"""

    def test_counts_stop_clusters(self) -> None:
        actions = np.zeros((30, 7))
        actions[5:15, 0] = 0.5                           # 動く区間は 1 つ
        clusters, ratio = stop_clusters([actions])
        assert clusters == 2                             # 頭と尻の 2 つ
        assert ratio == pytest.approx(20 / 30)

    def test_onset_steps(self) -> None:
        actions = np.zeros((20, 7))
        actions[4:, 0] = 0.5
        assert onset_steps([actions]) == 4

    def test_sign_flip_ignores_tiny_values(self) -> None:
        """しきい値より小さい値の揺れを、反転と数えないこと。"""
        actions = np.zeros((10, 7))
        actions[:, 0] = [0.001, -0.001, 0.001, -0.001, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert sign_flip_gated([actions], range(0, 1), threshold=0.02) == 0.0
        assert sign_flip_gated([actions], range(0, 1), threshold=0.0) > 0.0

    def test_measure_envelope_returns_all_keys(self) -> None:
        generator = np.random.default_rng(0)
        episodes = [generator.normal(size=(50, 7)) * 0.3 for _ in range(4)]
        measured = measure_envelope(episodes)
        assert set(measured) == {"limit_pos", "limit_rot", "accel_pos", "accel_rot",
                                 "jerk_pos", "jerk_rot"}
        assert all(value > 0 for value in measured.values())
