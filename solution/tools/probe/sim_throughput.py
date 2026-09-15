"""Measure LIBERO simulation throughput, split into physics and rendering.

The repository's cost estimates for data regeneration (docs/reference/28) assume
"env.step" costs 16-384 ms, but that number always included offscreen rendering.
This script separates the two so a generation budget can be computed.
"""

from __future__ import annotations

import os
import time

import numpy as np

CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SIZE = 256
TASK = "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
SUITE = "libero_spatial"


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


def timed(label: str, fn, repeats: int) -> float:
    fn()  # warm up once, outside the measurement
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    per_call = (time.perf_counter() - start) / repeats
    print(f"{label:<44} {per_call * 1000:8.2f} ms  (x{repeats})")
    return per_call


def main() -> None:
    from libero.libero import get_libero_path

    root = get_libero_path("bddl_files")
    bddl = os.path.join(root, SUITE, f"{TASK}.bddl")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}  bddl={bddl}")

    init_path = os.path.join(get_libero_path("init_states"), SUITE, f"{TASK}.pruned_init")
    import torch

    init_states = np.asarray(torch.load(init_path, weights_only=False))
    print(f"init states: {init_states.shape}")

    # --- physics only -------------------------------------------------------
    env = make_env(bddl, with_images=False)
    env.reset()
    env.env.sim.set_state_from_flattened(init_states[0])
    env.env.sim.forward()
    zero = np.zeros(7)
    for _ in range(10):
        env.env.step(zero)

    nq = env.env.sim.model.nq
    nv = env.env.sim.model.nv
    print(f"model: nq={nq} nv={nv} nbody={env.env.sim.model.nbody}")

    step_physics = timed("env.step  physics only", lambda: env.env.step(zero), 200)

    def restore() -> None:
        env.reset()
        env.env.sim.set_state_from_flattened(init_states[0])
        env.env.sim.forward()

    restore_cost = timed("reset + set_state + forward", restore, 10)

    def restore_no_reset() -> None:
        env.env.sim.set_state_from_flattened(init_states[0])
        env.env.sim.forward()

    restore_cheap = timed("set_state + forward (no reset)", restore_no_reset, 100)
    env.close()

    # --- physics + rendering ------------------------------------------------
    env_img = make_env(bddl, with_images=True)
    env_img.reset()
    env_img.env.sim.set_state_from_flattened(init_states[0])
    env_img.env.sim.forward()
    for _ in range(10):
        env_img.env.step(zero)

    step_render = timed("env.step  physics + 2 cameras", lambda: env_img.env.step(zero), 100)

    sim = env_img.env.sim

    def render_two() -> None:
        for cam in CAMERAS:
            sim.render(width=IMAGE_SIZE, height=IMAGE_SIZE, camera_name=cam)

    render_cost = timed("render 2 cameras only", render_two, 100)
    env_img.close()

    # --- what the numbers mean ---------------------------------------------
    demo_len = 169  # mean frames per demo in work/artifacts/libero_states
    print()
    print(f"{'physics-only rollout of one demo (169 steps)':<52}"
          f"{step_physics * demo_len:7.2f} s")
    print(f"{'rendered rollout of one demo (169 steps)':<52}"
          f"{step_render * demo_len:7.2f} s")
    print(f"{'render-only pass over one demo (169 frames)':<52}"
          f"{(restore_cheap + render_cost) * demo_len:7.2f} s")
    print(f"{'rendering share of a rendered rollout':<52}"
          f"{100 * (step_render - step_physics) / step_render:7.1f} %")


if __name__ == "__main__":
    main()
