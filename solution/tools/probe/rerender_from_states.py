"""Check the two claims a regeneration plan rests on.

1. A camera-perturbed variant env (`_view_...`) accepts the *base* task's saved
   MuJoCo state, so recorded demos can be re-rendered from new viewpoints with
   zero physics steps.
2. How much wall clock one env construction costs, since a per-viewpoint pass
   needs one env per viewpoint.
"""

from __future__ import annotations

import os
import time

import numpy as np

CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SIZE = 256
TASK = "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
SUITE = "libero_spatial"
VIEWS = ("", "_view_45_0_100_0_0_initstate_0", "_view_300_0_150_5_5_initstate_0")


def make_env(bddl: str, with_images: bool):
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=with_images,
        has_offscreen_renderer=with_images,
        has_renderer=False,
        control_freq=20,
        horizon=2000,
        ignore_done=True,
        camera_names=list(CAMERAS),
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
    )


def main() -> None:
    from libero.libero import get_libero_path

    bddl_root = get_libero_path("bddl_files")
    init_path = os.path.join(get_libero_path("init_states"), SUITE, f"{TASK}.pruned_init")
    import torch

    init_states = np.asarray(torch.load(init_path, weights_only=False))

    # A short recorded trajectory to replay: 30 steps of the zero action.
    base = make_env(os.path.join(bddl_root, SUITE, f"{TASK}.bddl"), with_images=True)
    base.reset()
    base.env.sim.set_state_from_flattened(init_states[0])
    base.env.sim.forward()
    trace = []
    for _ in range(30):
        base.env.step(np.zeros(7))
        trace.append(base.env.sim.get_state().flatten().copy())
    print(f"recorded {len(trace)} states, dim={trace[0].shape[0]}")
    base.close()

    frames = {}
    for view in VIEWS:
        name = f"{TASK}{view}"
        bddl = os.path.join(bddl_root, SUITE, f"{name}.bddl")
        exists = os.path.exists(bddl)
        start = time.perf_counter()
        env = make_env(bddl, with_images=True)
        build = time.perf_counter() - start
        env.reset()
        sim = env.env.sim

        start = time.perf_counter()
        rendered = []
        for state in trace:
            sim.set_state_from_flattened(state)
            sim.forward()
            rendered.append(sim.render(width=IMAGE_SIZE, height=IMAGE_SIZE,
                                       camera_name="agentview"))
        replay = time.perf_counter() - start

        frames[view or "base"] = np.asarray(rendered[-1])
        print(f"view={view or '(base)':<34} bddl_exists={exists}  "
              f"build={build:6.2f} s  re-render 30 frames={replay:5.3f} s  "
              f"nq={sim.model.nq}")
        env.close()

    ref = frames["base"].astype(np.int16)
    for key, img in frames.items():
        diff = np.abs(img.astype(np.int16) - ref).mean()
        print(f"mean |pixel diff| vs base for {key:<34} {diff:7.2f}")


if __name__ == "__main__":
    main()
