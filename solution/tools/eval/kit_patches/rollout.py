# 配布キットの pipeline/rollout.py に、衝突と判定された根拠（どの物体が何 m 動いたか）を
# 1 行 1 ロールアウトの JSONL で書き出す処理を足したもの。
#
# **なぜ要るか。** キットは `object_max_disp` を計算しておきながら、しきい値 1 mm と
# 比べた真偽（`collided`）だけを使い、値そのものを捨てている。**「かすった」のか
# 「倒した」のかが分からないと、衝突対策を推論側の後処理でやるのか学習でやるのかが
# 決まらない。**そこを残すのがこのパッチの目的である。
#
# **なぜ配布キットの中を直さず、こちらに置くのか。** 同じ場所の scorer.py と同じ理由で、
# キットの中を直すとマシンごとに中身がズレる（8/27 に JAIST と会社の RTX6000 で
# 指標の数が食い違った）。`run_final_eval.sh` が `/workspace/pipeline/rollout.py` へ
# 上書きマウントする。
#
# **キットの計算は L1 である。** 元の 212 行目は `np.sum(np.abs(cur - p0))` で、
# 3 軸の絶対値の和を取っている。ユークリッド距離ではないので、実際に動いた距離を
# 最大 √3 倍だけ大きく見せる。**しきい値 1 mm と比べられている値はこの L1 なので、
# `collided` の判定は元のまま変えず、実距離を読むための L2 を別のキーで併記する。**
#
# **環境変数 PARC_OBJECT_DISP_JSONL が書き出し先である。**設定されていなければ
# 何も書かないので、キットの挙動は元のままになる。
#
# 元にした版: b_PARC2026_final @ 2059989 の pipeline/rollout.py
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .config import EvalConfig, PerturbationConfig
from .environment import EnvironmentManager, TaskInfo
from .total_score import load_scoring_config

logger = logging.getLogger(__name__)


def _append_object_disp(record: dict) -> None:
    """衝突の内訳を 1 行の JSON として追記する。

    書き出し先は環境変数 ``PARC_OBJECT_DISP_JSONL`` で、設定されていなければ
    何もしない。**評価は 1 プロセスで 1 エピソードずつ直列に回るので、追記の
    競合は起きない。**書き出しに失敗しても評価を止めない（測定のための付加物で
    あって、評価そのものではない）。
    """
    path = os.environ.get("PARC_OBJECT_DISP_JSONL", "")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as e:
        logger.warning("object_disp の書き出しに失敗: %s", e)


class PolicyInterface(Protocol):

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        ...


@dataclass
class EpisodeResult:
    task_name: str
    episode_id: int
    success: bool
    total_steps: int
    elapsed_time_sec: float
    joint_positions: list[np.ndarray] = field(default_factory=list)
    ee_positions: list[np.ndarray] = field(default_factory=list)
    ee_orientations: list[np.ndarray] = field(default_factory=list)
    gripper_qpos: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    collided: bool = False

    @property
    def trajectory(self) -> list[np.ndarray]:
        return self.joint_positions


@dataclass
class TaskResult:
    task_info: TaskInfo
    episodes: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def avg_steps(self) -> float:
        successful = [e for e in self.episodes if e.success]
        if not successful:
            return 0.0
        return sum(e.total_steps for e in successful) / len(successful)

    @property
    def avg_time(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.elapsed_time_sec for e in self.episodes) / len(self.episodes)


class RolloutExecutor:

    def __init__(
        self,
        env_manager: EnvironmentManager,
        eval_config: EvalConfig,
        scoring_config: dict | None = None,
    ):
        self.env_manager = env_manager
        self.config = eval_config
        self.scoring_config = scoring_config or load_scoring_config()

    def evaluate_task(
        self,
        policy: PolicyInterface,
        task_info: TaskInfo,
        perturbation: PerturbationConfig,
    ) -> TaskResult:
        logger.info(
            "タスク評価開始: %s (%d エピソード)",
            task_info.name, self.config.n_eval_episodes,
        )

        env = self.env_manager.create_env(task_info)

        init_states = self.env_manager.get_perturbed_init_states(
            task_info, perturbation, self.config.n_eval_episodes
        )

        collision_enabled = bool(self.scoring_config.get("collision", {}).get("enabled", True))
        obj_of_interest = (
            self.env_manager.get_obj_of_interest(task_info) if collision_enabled else set()
        )

        episodes: list[EpisodeResult] = []

        try:
            for ep_id in range(self.config.n_eval_episodes):
                result = self._run_episode(
                    env=env,
                    policy=policy,
                    task_info=task_info,
                    init_state=init_states[ep_id],
                    episode_id=ep_id,
                    perturbation=perturbation,
                    obj_of_interest=obj_of_interest,
                )
                episodes.append(result)

                if result.success:
                    logger.debug(
                        "  Episode %d: 成功 (%d steps)", ep_id, result.total_steps
                    )
                else:
                    logger.debug(
                        "  Episode %d: 失敗 (%d steps)", ep_id, result.total_steps
                    )
        finally:
            env.close()

        task_result = TaskResult(task_info=task_info, episodes=episodes)
        logger.info(
            "タスク評価完了: %s — 成功率 %.1f%% (平均 %.1f steps)",
            task_info.name, task_result.success_rate * 100, task_result.avg_steps,
        )
        return task_result

    def _run_episode(
        self,
        env: Any,
        policy: PolicyInterface,
        task_info: TaskInfo,
        init_state: np.ndarray,
        episode_id: int,
        perturbation: PerturbationConfig,
        obj_of_interest: set[str],
    ) -> EpisodeResult:
        start_time = time.time()
        joint_positions: list[np.ndarray] = []
        ee_positions: list[np.ndarray] = []
        ee_orientations: list[np.ndarray] = []
        gripper_qpos_log: list[np.ndarray] = []
        actions_log: list[np.ndarray] = []
        rewards_log: list[float] = []

        cc = self.scoring_config.get("collision", {})
        collision_enabled = bool(cc.get("enabled", True))
        collision_threshold = float(cc.get("threshold_m", 0.001))

        record_video = bool(os.environ.get("T3_VIDEO_DIR")) and episode_id == 0
        video_frames: list[np.ndarray] = []

        env.reset()
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()

        action_dim = env.robots[0].action_dim
        dummy_action = np.zeros(action_dim)
        for _ in range(10):
            obs, _, _, _ = env.step(dummy_action)

        object_init_pos: dict[str, np.ndarray] = {}
        if collision_enabled:
            object_init_pos = {
                k[:-4]: np.asarray(obs[k]).copy()
                for k in obs
                if k.endswith("_pos")
                and not k.startswith("robot0")
                and not k.endswith("_to_robot0_eef_pos")
                and k[:-4] not in obj_of_interest
            }
        object_max_disp: dict[str, float] = {}
        object_max_disp_l2: dict[str, float] = {}

        episode_seed = self.config.seed + episode_id
        policy.reset(instruction=task_info.language, seed=episode_seed)
        done = False
        total_steps = 0

        if record_video:
            self._capture_frame(video_frames, obs)

        for step in range(self.config.max_steps_per_episode):
            obs_for_policy = self.env_manager.apply_observation_noise(
                obs, perturbation
            )

            action = policy.get_action(obs_for_policy)

            action = self.env_manager.apply_action_noise(action, perturbation)

            obs, reward, done, info = env.step(action)

            joint_positions.append(obs.get("robot0_joint_pos", np.zeros(7)).copy())
            ee_positions.append(obs.get("robot0_eef_pos", np.zeros(3)).copy())
            ee_orientations.append(obs.get("robot0_eef_quat", np.array([1, 0, 0, 0], dtype=np.float64)).copy())
            gripper_qpos_log.append(obs.get("robot0_gripper_qpos", np.zeros(2)).copy())
            actions_log.append(action.copy())
            rewards_log.append(float(reward))

            for name, p0 in object_init_pos.items():
                cur = obs.get(name + "_pos")
                if cur is not None:
                    delta = np.asarray(cur) - p0
                    d = float(np.sum(np.abs(delta)))
                    if d > object_max_disp.get(name, 0.0):
                        object_max_disp[name] = d
                    # 実距離。しきい値と比べるのは上の L1 のままで、これは記録専用。
                    d2 = float(np.linalg.norm(delta))
                    if d2 > object_max_disp_l2.get(name, 0.0):
                        object_max_disp_l2[name] = d2

            total_steps = step + 1

            if total_steps % 50 == 0:
                logger.info(
                    "  [進捗] %s: %d/%d steps (%.1fs)",
                    task_info.name, total_steps, self.config.max_steps_per_episode,
                    time.time() - start_time,
                )

            if record_video:
                self._capture_frame(video_frames, obs)

            if done:
                break

        elapsed = time.time() - start_time

        collided = any(d > collision_threshold for d in object_max_disp.values())
        success = bool(done) and not collided

        _append_object_disp({
            "task": task_info.name,
            "episode": episode_id,
            "success": success,
            "done": bool(done),
            "collided": collided,
            "total_steps": total_steps,
            "collision_threshold_m": collision_threshold,
            "obj_of_interest": sorted(obj_of_interest),
            "max_disp_m": {k: round(v, 6) for k, v in object_max_disp.items()},
            "max_disp_l2_m": {k: round(v, 6) for k, v in object_max_disp_l2.items()},
        })

        if record_video:
            self._write_video(task_info.name, video_frames, success)

        return EpisodeResult(
            task_name=task_info.name,
            episode_id=episode_id,
            success=success,
            total_steps=total_steps,
            elapsed_time_sec=elapsed,
            joint_positions=joint_positions,
            ee_positions=ee_positions,
            ee_orientations=ee_orientations,
            gripper_qpos=gripper_qpos_log,
            actions=actions_log,
            rewards=rewards_log,
            collided=collided,
        )

    @staticmethod
    def _capture_frame(frames: list[np.ndarray], obs: dict) -> None:
        try:
            agent = obs.get("agentview_image")
            if agent is None:
                return
            agent = np.asarray(agent)[::-1]
            wrist = obs.get("robot0_eye_in_hand_image")
            if wrist is not None:
                wrist = np.asarray(wrist)[::-1]
                if wrist.shape[0] == agent.shape[0]:
                    agent = np.concatenate([agent, wrist], axis=1)
            frames.append(np.ascontiguousarray(agent, dtype=np.uint8))
        except Exception:
            pass

    def _write_video(self, task_name: str, frames: list[np.ndarray], success: bool) -> None:
        if not frames:
            return
        out_dir = os.environ.get("T3_VIDEO_DIR", "")
        try:
            os.makedirs(out_dir, exist_ok=True)
            fps = int(os.environ.get("T3_VIDEO_FPS", "20"))
            tag = "SUCCESS" if success else "FAIL"
            path = os.path.join(out_dir, f"{task_name}__{tag}.mp4")
            import imageio.v2 as imageio

            imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
            logger.info("動画を保存: %s (%d frames)", path, len(frames))
        except Exception as e:
            logger.warning("動画保存に失敗: %s", e)

    def evaluate_tasks(
        self,
        policy: PolicyInterface,
        task_infos: list[TaskInfo],
        perturbation: PerturbationConfig,
    ) -> list[TaskResult]:
        results = []
        for i, task_info in enumerate(task_infos):
            logger.info(
                "=== タスク %d/%d: %s ===",
                i + 1, len(task_infos), task_info.name,
            )
            result = self.evaluate_task(policy, task_info, perturbation)
            results.append(result)
        return results
