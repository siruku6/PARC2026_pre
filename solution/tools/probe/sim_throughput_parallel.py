"""Measure how physics throughput scales across worker processes.

A regeneration budget is set by aggregate steps/second, not by one process.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SIZE = 256
TASK = "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
SUITE = "libero_spatial"
STEPS = 60


def worker(_index: int, ready: mp.Barrier, out: mp.Queue) -> None:  # type: ignore[valid-type]
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    import torch

    bddl = os.path.join(get_libero_path("bddl_files"), SUITE, f"{TASK}.bddl")
    init_path = os.path.join(get_libero_path("init_states"), SUITE, f"{TASK}.pruned_init")
    init_states = np.asarray(torch.load(init_path, weights_only=False))

    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
        control_freq=20,
        horizon=2000,
        ignore_done=True,
        camera_names=list(CAMERAS),
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
    )
    env.reset()
    env.env.sim.set_state_from_flattened(init_states[0])
    env.env.sim.forward()
    zero = np.zeros(7)
    for _ in range(10):
        env.env.step(zero)

    ready.wait()  # start all workers at the same instant
    start = time.perf_counter()
    for _ in range(STEPS):
        env.env.step(zero)
    out.put((time.perf_counter() - start) / STEPS)
    env.close()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    print(f"cpu count: {os.cpu_count()}   OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    for n in (1, 4, 8, 12, 16):
        ready = mp.Barrier(n)
        out: mp.Queue = mp.Queue()
        procs = [mp.Process(target=worker, args=(i, ready, out)) for i in range(n)]
        for p in procs:
            p.start()
        per_step = [out.get() for _ in range(n)]
        for p in procs:
            p.join()
        mean_ms = 1000 * float(np.mean(per_step))
        aggregate = n / float(np.mean(per_step))
        print(f"workers={n:<3} per-step {mean_ms:7.2f} ms   "
              f"aggregate {aggregate:7.1f} steps/s   "
              f"= {aggregate * 3600 / 169:8.0f} demo-rollouts/hour")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
