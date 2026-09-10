---
license: mit
task_categories:
  - robotics
tags:
  - lerobot
  - libero
  - pi05
  - imitation-learning
  - synthetic
configs:
  - config_name: default
    data_files: data/*/*.parquet
---

# jaist_mix_v2 — LIBERO-plus + synthetic reverse-direction episodes (49.5 %)

A LeRobot v3.0 dataset built to teach a π0.5 policy the *reverse* of what LIBERO demonstrates.

The distributed LIBERO demonstrations put objects **onto** shelves, plates and trays. The tasks we needed
take them **off** again — and only **1 of 2,066** instruction strings in the distributed training data is a
`take … off` / `out of` task. This dataset mixes the distributed data with synthetic episodes that supply
the missing direction.

| Part | Episodes | Frames | Tasks | Share of frames |
|---|---|---|---|---|
| [`lerobot/libero_plus`](https://huggingface.co/datasets/lerobot/libero_plus), thinned to 75 episodes/task | 3,000 | 501,403 | 40 | 50.5 % |
| **Synthetic reverse-direction episodes** (397 unique, replicated ×8) | 3,176 | 491,440 | 3 | **49.5 %** |
| **Total** | **6,176** | **992,843** | **43** | |

## The reverse episodes

| Instruction | Episodes | Frames | Mean length |
|---|---|---|---|
| `take the yellow book off the top of the shelf and put it back on the table on the left` | 1,200 | 181,880 | 152 |
| `take the red mug off the left plate and put it back on the table behind the other two mugs` | 1,176 | 199,344 | 170 |
| `take the alphabet soup out of the tray and place it on the table in front of the ketchup` | 800 | 110,216 | 138 |

**These are not time-reversed playback.** Each episode is an ordinary forward-in-time rollout of a scripted
controller inside LIBERO simulation, in six phases:

1. **approach** — move to a grasp pose with the gripper open
2. **grasp** — close in place
3. **lift** — retreat straight up
4. **return** — follow a reference path
5. **place** — stop on contact
6. **release**

Images, `action` and `observation.state` all advance in time, at 20 fps, exactly like a recorded
demonstration. Time reversal is used in one place only: the *reference path* of phase 4 is read backwards
out of a distributed forward demonstration, which is what makes the resulting task the inverse of it.

The grasp pose is derived from the forward demonstration's own end-effector pose at its grasp frame, then
re-expressed in the current object frame, so it follows the object however it happens to be placed.

**Each episode starts from a distributed initial state** (`pruned_init`) of the corresponding evaluation
task, and only successful rollouts were written. Replication (×8) is per-episode duplication, chosen so
that one pass over the reverse frames takes ~7,800 optimizer steps at batch 16.

### Why "v2"

A first version of these episodes let the scripted controller's **per-step wrist rotation command saturate
at the ±1 action clip**, while the distributed demonstrations stay inside ±0.375. 15.2 % of frames exceeded
10 σ, and **52.6 % of those sat in the first 10 % of each episode** — the approach phase. A policy trained
on that version shrank its own rotation commands toward the mean (95th percentile 0.25 against an envelope
of 0.375).

This version rate-limits per-step rotation to ±0.375. Measured over 40 episodes per task:

| | v1 | **v2** |
|---|---|---|
| 95th percentile of \|rotation\| | 0.98 – 1.00 | **0.375** |
| Frames outside ±0.375 | 8.9 – 26.0 % | **0.0 %** |
| … in the first 10 % of the episode | 25.3 – 97.6 % | **0.0 %** |
| Frames beyond 10 σ | 15.2 % | **0.0 %** |

A policy trained on v2 issues rotation commands at 0.39 — matching the envelope — and reaches a lower
training loss (0.346 against 0.41–0.44 at the same step count, same recipe).

The placement phase was fixed at the same time: v1 released objects up to 40 mm above the surface, v2
releases at 1.00 mm (book) / 2.90 mm (soup).

> **What this dataset does not fix.** A policy trained on it still does not grasp in closed loop — see
> [`siruku6/pi05_t3rev_v2`](https://huggingface.co/siruku6/pi05_t3rev_v2) for the evaluation. The rotation
> defect was real and is gone; it was not the whole cause.

## Format

LeRobot `codebase_version: v3.0`, `robot_type: panda`, 20 fps.

| Feature | dtype | shape |
|---|---|---|
| `observation.images.front` | video | 256 × 256 × 3 |
| `observation.images.wrist` | video | 256 × 256 × 3 |
| `observation.state` | float32 | 8 — `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`, raw scale |
| `action` | float32 | 7 — OSC delta pose `[dx, dy, dz, drx, dry, drz, gripper]` |

Images are stored **rotated 180°** relative to the raw simulator render, matching the convention of the
policy's preprocessing. `action[6]` follows the robosuite convention: **−1 opens, +1 closes.**

```
data/chunk-000/file-000.parquet
meta/info.json  meta/tasks.parquet  meta/episodes/chunk-000/file-000.parquet
meta/stats.json  meta/stats.json.bak
videos/observation.images.front/chunk-000/file-{000..017}.mp4
videos/observation.images.wrist/chunk-000/file-{000..016}.mp4
```

**`meta/stats.json` is pinned, not recomputed.** It carries the normalization statistics of the base
policy's own normalizer (difference 0.000e+00), and `meta/stats.json.bak` is the recomputed file it
replaced. This matters: recomputing statistics over a mixture that is half synthetic shifts the
normalizer, and a fine-tune that starts from a checkpoint whose normalizer was fitted on the distributed
data will then see inputs on a different scale.

## Use

```sh
hf download siruku6/jaist_mix_v2 --repo-type dataset --local-dir ./jaist_mix_v2
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/jaist_mix_v2", root="./jaist_mix_v2")
```

## Intended use and limitations

Research artifact. **Simulation only** — a simulated Franka Panda in LIBERO, with no real-robot
validation. The reverse episodes come from a scripted controller, not a human, so they are smoother and
more stereotyped than human demonstrations, and they cover exactly three scenes. The ×8 replication means
the effective number of distinct reverse trajectories is 397, not 3,176.

## License and attribution

Released under the **MIT License**, inheriting from the LIBERO components it is built on.

| Component | Source | License |
|---|---|---|
| Forward episodes | [`lerobot/libero_plus`](https://huggingface.co/datasets/lerobot/libero_plus), redistributing [`Sylvest/LIBERO-plus`](https://huggingface.co/datasets/Sylvest/LIBERO-plus) | MIT |
| Scenes, meshes, task definitions | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — Copyright (c) 2023 Lifelong Robot Learning | MIT |
| Simulator | [robosuite](https://github.com/ARISE-Initiative/robosuite) / [MuJoCo](https://github.com/google-deepmind/mujoco) | MIT / Apache-2.0 |
| Dataset format and tooling | [LeRobot](https://github.com/huggingface/lerobot) v0.6.0 | Apache-2.0 |
| Reverse episodes | generated by the authors of this repository | MIT |

**No model weights are contained here**, so the Gemma Terms of Use that govern the policy checkpoints do
not apply to this dataset.
