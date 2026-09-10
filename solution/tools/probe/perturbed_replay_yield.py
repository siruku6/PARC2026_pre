"""Can one recorded trajectory be regenerated from a perturbed robot start pose?

The training data covers end-effector offsets of ~1 cm (reference/06); Track2
draws offsets of 2-6.5 cm, where success falls 0.833 -> 0.535 and collision
rises 0.139 -> 0.319 (002). Adding demos in that band needs *new* trajectories,
because the arm starts somewhere the recorded action stream never visits.

Two ways to reuse one recorded trajectory at a new start pose:

  open-loop   replay the recorded 7-D delta actions verbatim
  tracked     drive the arm onto the recorded *absolute* end-effector pose
              sequence with a proportional controller, and record the deltas
              that controller emits as the new action labels

This measures which of the two survives the perturbation, using the grader's own
success and 1 mm collision rules.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/repo/solution/tools/scripted_demo")

TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
LAYOUTS = (0, 1, 2)
INITSTATES = (0, 150, 350, 450)
MAX_STEPS = 420
METHODS = ("open-loop", "ff+fb gain=0.5", "ff+fb gain=1.5")


def variant_name(initstate: int) -> str:
    if initstate == 0:
        return TASK
    return f"{TASK}_view_0_0_100_0_0_initstate_{initstate}"


def axisangle_to_quat(vec: np.ndarray) -> np.ndarray:
    from robosuite.utils.transform_utils import axisangle2quat

    return np.asarray(axisangle2quat(np.asarray(vec, dtype=np.float64)))


def quat_delta_axisangle(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Axis-angle of the rotation that takes `current` onto `target`."""
    from robosuite.utils.transform_utils import (
        quat2axisangle,
        quat_multiply,
        quat_inverse,
    )

    relative = quat_multiply(np.asarray(target, dtype=np.float64),
                             quat_inverse(np.asarray(current, dtype=np.float64)))
    return np.asarray(quat2axisangle(relative))


class Reference:
    """The absolute pose sequence pulled out of one recorded rollout."""

    def __init__(self, record) -> None:  # noqa: ANN001 - EpisodeRecord
        states = np.asarray(record.states, dtype=np.float64)
        actions = np.asarray(record.actions, dtype=np.float64)
        self.positions = states[:, :3]
        self.quats = np.stack([axisangle_to_quat(v) for v in states[:, 3:6]])
        self.grippers = actions[:, 6]
        self.executed = np.asarray(record.executed_actions, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.positions)


def watched_displacement(rollout) -> float:  # noqa: ANN001 - _Rollout
    return float(rollout.record.max_displacement)


def run_open_loop(sd, env, init_state, reference: Reference):  # noqa: ANN001
    env.seed(0)
    rollout = sd._Rollout(env, init_state, MAX_STEPS, record_images=False)
    for action in reference.executed:
        if not rollout._take_step(action.astype(np.float32)):
            break
    rollout.record.goal_reached = bool(env.check_success())
    return rollout.record


def run_tracked(sd, env, init_state, reference: Reference, *,
                pos_scale: float, rot_scale: float, gain: float,
                lag_limit: float = 0.030, max_hold: int = 40,
                lead: int = 2):  # noqa: ANN001
    """Replay the recorded deltas as feedforward, corrected toward the pose.

    The recorded delta stream carries the demo's velocity profile, which is what
    makes the action labels look like the distributed data. On its own it drifts
    when the arm starts somewhere else, so a proportional term pulls the arm back
    onto the recorded absolute pose. The controller only realises about a fifth
    of a commanded delta per step (reference/28 5.1), hence `gain` above one.
    """
    env.seed(0)
    rollout = sd._Rollout(env, init_state, MAX_STEPS, record_images=False)
    index = 0
    holds = 0
    while index < len(reference):
        aim = min(index + lead, len(reference) - 1)
        error = reference.positions[aim] - rollout.eef_position
        distance = float(np.linalg.norm(reference.positions[index]
                                        - rollout.eef_position))

        feedforward = reference.executed[index]
        action = np.zeros(7)
        action[:3] = np.clip(feedforward[:3] + gain * error / pos_scale, -1.0, 1.0)
        rotation = quat_delta_axisangle(
            reference.quats[aim],
            np.asarray(rollout.obs["robot0_eef_quat"], dtype=np.float64))
        action[3:6] = np.clip(feedforward[3:6] + gain * rotation / rot_scale,
                              -1.0, 1.0)
        action[6] = reference.grippers[index]

        if not rollout._take_step(action.astype(np.float32)):
            break

        # Advance along the reference unless the arm is still catching up, so a
        # perturbed start pose converges instead of being dragged out of phase.
        if distance <= lag_limit or holds >= max_hold:
            index += 1
            holds = 0
        else:
            holds += 1

    rollout.record.goal_reached = bool(env.check_success())
    return rollout.record


def label_quality(record) -> tuple[float, float]:  # noqa: ANN001
    actions = np.asarray(record.actions, dtype=np.float64)
    saturated = float(np.mean(np.abs(actions[:, :6]) >= 0.999))
    return saturated, float(np.std(actions[:, :6]))


def main() -> None:
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    plan = sd.TASK_PLANS[TASK]
    init_states = sd._init_states(libero_root, plan.suite, TASK)

    # Controller scaling: the delta action is normalised by these limits.
    probe_env = sd._make_env(sd._bddl_path(libero_root, plan.suite, TASK), False)
    probe_env.reset()
    controller = probe_env.env.robots[0].controller
    output_max = np.asarray(controller.output_max, dtype=np.float64)
    pos_scale = float(np.max(np.abs(output_max[:3])))
    rot_scale = float(np.max(np.abs(output_max[3:6])))
    print(f"controller output_max = {output_max}")
    print(f"pos_scale={pos_scale}  rot_scale={rot_scale}")

    references: dict[int, Reference] = {}
    for layout in LAYOUTS:
        probe_env.seed(0)
        record = sd.run_episode(probe_env, init_states[layout], plan,
                                max_steps=MAX_STEPS, record_images=False)
        references[layout] = Reference(record)
        print(f"reference layout={layout}: steps={record.steps} "
              f"goal={record.goal_reached} disp={1000 * record.max_displacement:.2f} mm "
              f"len={len(references[layout])}")
    probe_env.close()

    header = (f"{'initstate':>10} {'eef off':>8} {'method':>14} "
              f"{'goal':>6} {'disp mm':>9} {'steps':>6} {'sat':>6} {'std':>6}")
    print()
    print(header)
    print("-" * len(header))

    summary: dict[tuple[int, str], list[bool]] = {}
    for initstate in INITSTATES:
        bddl = sd._bddl_path(libero_root, plan.suite, variant_name(initstate))
        env = sd._make_env(bddl, with_images=False)

        # End-effector offset this perturbation produces, measured the way the
        # grader sets up an episode (reset, restore, ten zero actions).
        obs = sd.reset_to(env, init_states[LAYOUTS[0]])
        offset_ref = np.asarray(obs["robot0_eef_pos"])

        for layout in LAYOUTS:
            reference = references[layout]
            offset = float(np.linalg.norm(
                offset_ref - references[LAYOUTS[0]].positions[0])) if initstate else 0.0
            for method in METHODS:
                start = time.perf_counter()
                if method == "open-loop":
                    record = run_open_loop(sd, env, init_states[layout], reference)
                else:
                    record = run_tracked(sd, env, init_states[layout], reference,
                                         pos_scale=pos_scale, rot_scale=rot_scale,
                                         gain=float(method.split("=")[1]))
                saturated, spread = label_quality(record)
                clean = record.max_displacement <= 0.001
                summary.setdefault((initstate, method), []).append(
                    bool(record.goal_reached and clean))
                print(f"{initstate:>10} {100 * offset:7.2f}cm {method:>14} "
                      f"{str(record.goal_reached):>6} "
                      f"{1000 * record.max_displacement:9.2f} {record.steps:>6} "
                      f"{saturated:6.2f} {spread:6.3f}"
                      f"   ({time.perf_counter() - start:.1f}s)")
        env.close()

    print()
    print("usable demos (goal reached AND every non-target object under 1 mm)")
    for initstate in INITSTATES:
        row = [f"initstate {initstate:>3}"]
        for method in METHODS:
            got = summary.get((initstate, method), [])
            row.append(f"{method} {sum(got)}/{len(got)}")
        print("   ".join(row))


if __name__ == "__main__":
    main()
