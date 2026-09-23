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

# libero_plus_bowl5

*日本語版: [README.ja.md](README.ja.md)*

`lerobot/libero_plus` (14,347 episodes) with the 5 "black bowl" pick-and-place tasks —
the ones with the highest collision rate against a nearby object during scoring — removed
and replaced by newly-recorded scripted demonstrations that grasp and place more
carefully. Everything else in `libero_plus` is unchanged and re-used via hardlink (videos
were not re-encoded).

## Why these 5 tasks

All 5 involve picking up the black bowl next to or on top of another object, where the
gripper has to straddle the bowl's rim (the bowl is wider than the gripper, so it cannot be
grasped from directly above):

- `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`

In the distributed demonstrations, the human demonstrator's grasp radius (wrist ~4.0 cm
from the bowl's rotation axis) sits inside the rim wall's actual center (~4.71 cm), so one
finger contacts the wall before the other — the bowl gets pushed rather than gripped. This
family accounted for 11.3 of the ~17 percentage points of score lost to unwanted contact
across all LIBERO-plus tasks measured (see composition table below).

## Composition

| Part | Episodes | Note |
|---|---|---|
| `libero_plus` minus the 5 bowl tasks | 12,467 | Hardlinked from `lerobot/libero_plus`, untouched |
| Newly recorded scripted demos, 5 bowl tasks | 1,618 | Replaces the 1,880 removed episodes (see below) |
| **Total** | **14,085** | 40 tasks, 2,264,704 frames, 20 fps |

The new demos were generated with a scripted rim-grasp policy
(`solution/tools/scripted_demo/scripted_demo.py`, class `RimPickPlace`) that grasps at the
rim wall's actual center radius and reduces descent speed before contact, across two
initial-state sources (the evaluator's 50 canonical perturbed poses per task, and 50
poses sampled around each human demonstration's initial pose, for 4 of the 5 tasks) times
a scripted-policy pass. Episodes where the policy did not reach the goal, or touched a
non-target object by more than 1mm, were discarded rather than kept — hence 1,618 kept out
of an attempted ~1,880 (in line with the family's independently-measured contact-free
success rate of 130/150, 86.7%, at grasp offsets up to 9.67 cm — well beyond the ≤1.0 cm
of hand offset present in the original human demonstrations).

Task instruction text was normalized to match the *distributed* dataset's phrasing (not
the raw BDDL text used at scoring time), since the eval harness maps the grader's
instruction to one of the training set's exact 40 strings before it reaches the policy —
a scripted demo recorded with off-vocabulary phrasing would sit outside that lookup and
never be matched at inference time.

## Format

LeRobot v3.0 dataset, same feature schema as `lerobot/libero_plus`:

| Feature | dtype | shape |
|---|---|---|
| `observation.images.front` | video | [256, 256, 3] |
| `observation.images.wrist` | video | [256, 256, 3] |
| `observation.state` | float32 | [8] |
| `action` | float32 | [7] |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | float32 / int64 | [1] |

## Contents

```
data/chunk-000/...
videos/observation.images.front/...
videos/observation.images.wrist/...
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
bowl5_expected.json          # expected episode/frame/task counts, used as a build-time gate
```

## Use

```python
from huggingface_hub import hf_hub_download  # or: hf download siruku6/libero_plus_bowl5 --repo-type dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("siruku6/libero_plus_bowl5")
```

## Intended use and limitations

- Built for fine-tuning π0.5/LIBERO policies as a drop-in replacement for
  `lerobot/libero_plus`; the only intentional difference from the distributed dataset is
  the 5 bowl tasks' demonstrations, so it is meant to be compared against a model trained
  on the unmodified `libero_plus` with everything else held fixed.
- The scripted replacement demos are simulation-only (LIBERO + simulated Franka Panda via
  robosuite/MuJoCo), same as the rest of `libero_plus`.
- Only the 5 named tasks were touched.
  [`libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose)
  extends this by also varying initial robot pose and visual appearance across a larger
  set of re-recorded episodes (16,820 episodes).
- Training-time effect: isolated with an action-expert-only fine-tune, swapping in this
  data (vs. unmodified `libero_plus`) moved the local Track2 composite score from 0.3656
  to 0.3354 (down), concentrated in one held-out camera-perturbation condition. A
  full-parameter fine-tune on this data scored 0.4004 (best of the four weights compared),
  but there is no full-parameter-on-unmodified-data control to isolate how much of that is
  the data vs. the training regime — see `docs/resources/datasets.md` §4 for the full
  comparison table.

## License and attribution

Built on the LIBERO benchmark, distributed via `lerobot/libero_plus`.

| Component | License |
|---|---|
| LIBERO (Copyright (c) 2023 UT-Austin Robot Perception and Learning Lab / Lifelong-Robot-Learning, https://github.com/Lifelong-Robot-Learning/LIBERO) | MIT |
| `lerobot/libero_plus` (unmodified portion, hardlinked here) | MIT (inherits from LIBERO) |
| Newly recorded scripted demonstrations (this repo's own contribution) | MIT |
| robosuite (https://github.com/ARISE-Initiative/robosuite) | MIT |
| MuJoCo (https://github.com/google-deepmind/mujoco) | Apache-2.0 |
