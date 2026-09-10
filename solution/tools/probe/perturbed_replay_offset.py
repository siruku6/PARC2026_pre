"""Does a replayed episode keep the perturbed start pose, or wash it out?

If replaying a demo's actions inside a robot-init-perturbed environment succeeds
only because the arm snaps back onto the nominal path, the generated episode adds
no coverage of the offset band the policy actually fails in. This measures the
distance between the perturbed replay's end-effector path and the nominal one,
step by step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/repo/solution/tools/scripted_demo")

TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
LAYOUT = 0
MAX_STEPS = 420
INITSTATES = (150, 350, 450)


def main() -> None:
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    plan = sd.TASK_PLANS[TASK]
    init_states = sd._init_states(libero_root, plan.suite, TASK)

    nominal_env = sd._make_env(sd._bddl_path(libero_root, plan.suite, TASK), False)
    nominal_env.seed(0)
    reference = sd.run_episode(nominal_env, init_states[LAYOUT], plan,
                               max_steps=MAX_STEPS)
    nominal_path = np.asarray(reference.ee_positions, dtype=np.float64)
    actions = np.asarray(reference.executed_actions, dtype=np.float64)
    print(f"nominal: steps={reference.steps} goal={reference.goal_reached}")
    nominal_env.close()

    print()
    print(f"{'initstate':>10} {'frame 0':>9} {'25%':>8} {'50%':>8} {'75%':>8} "
          f"{'max':>8} {'mean':>8} {'goal':>6}")
    print("-" * 74)
    for initstate in INITSTATES:
        name = f"{TASK}_view_0_0_100_0_0_initstate_{initstate}"
        env = sd._make_env(sd._bddl_path(libero_root, plan.suite, name), False)
        env.seed(0)
        rollout = sd._Rollout(env, init_states[LAYOUT], MAX_STEPS,
                              record_images=False)
        for action in actions:
            if not rollout._take_step(action.astype(np.float32)):
                break
        goal = bool(env.check_success())
        path = np.asarray(rollout.record.ee_positions, dtype=np.float64)
        count = min(len(path), len(nominal_path))
        gap = np.linalg.norm(path[:count] - nominal_path[:count], axis=1) * 100
        quarters = [gap[int(count * f)] for f in (0.25, 0.50, 0.75)]
        print(f"{initstate:>10} {gap[0]:8.2f}cm {quarters[0]:7.2f}cm "
              f"{quarters[1]:7.2f}cm {quarters[2]:7.2f}cm {gap.max():7.2f}cm "
              f"{gap.mean():7.2f}cm {str(goal):>6}")
        env.close()


if __name__ == "__main__":
    main()
