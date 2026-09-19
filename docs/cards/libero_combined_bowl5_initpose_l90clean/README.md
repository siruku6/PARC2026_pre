---
license: mit
task_categories:
  - robotics
language:
  - en
tags:
  - robotics
  - libero
  - lerobot
  - imitation-learning
  - manipulation
size_categories:
  - 10K<n<100K
---

# libero_combined_bowl5_initpose_l90clean

*日本語版: [README.ja.md](README.ja.md)*

A LeRobot-format demonstration dataset for the **LIBERO** manipulation benchmark, rebuilt from
the distributed LIBERO / LIBERO-plus data to remove three things that were teaching the wrong
lesson: **demonstrations that bump into non-target objects**, **a gripper encoding that labelled
"open" backwards**, and **a robot initial pose distribution too narrow to cover the evaluation**.

| | |
|---|---|
| Episodes | **21,642** |
| Frames | **3,431,137** |
| Tasks | **111** |
| Size | 27.8 GB |
| Format | LeRobot v2.1, 20 Hz |
| Cameras | scene (`observation.images.image`) and wrist (`observation.images.wrist_image`), 256×256 |
| State / action | 8-dim state, 7-dim delta end-effector pose |

Built for the **PARC2026** robot-manipulation competition.

## How it was built

The dataset is the last of a chain. Each step is listed with what it changed, so you can pick
an earlier one if a change does not suit you.

```
LIBERO-plus (distributed)
  └─ libero_plus_bowl5            bowl tasks re-recorded          14,085 eps
      └─ libero_plus_bowl5_initpose   contact removed, initial poses varied   16,820 eps
          └─ libero_combined_bowl5_initpose   + libero_90 and standard suites  21,642 eps
              └─ libero_combined_bowl5_initpose_l90clean   ← this dataset      21,642 eps
```

### 1. Bowl tasks re-recorded (`libero_plus_bowl5`)

Five "bowl" tasks had the highest collision rate in the distributed data. All 1,880 episodes
were dropped and replaced with demonstrations from a scripted policy whose grasp radius was
aligned to the centre of the rim and whose descent speed was reduced. The replacement reached
130/150 success under perturbation before recording.

### 2. Contact removed, initial poses varied (`libero_plus_bowl5_initpose`)

- **1,532 episodes were dropped** — those that contacted a non-target object, and those longer
  than 300 frames (the evaluation truncation limit).
- **3,972 new demonstrations** were recorded to refill them, with the **robot initial pose
  varied** and appearance (lighting, table and background textures, camera) layered on top.
  We verified that appearance changes leave the physics bit-identical, so they do not dilute
  the physical distribution.
- **303 more** were re-recorded shorter, for episodes originally dropped for exceeding 300 frames.
- Guards applied throughout: at most 3 variants per source demonstration, episodes cut 10 steps
  after the goal (matching the distributed data), instructions phrased the way the distributed
  data phrases them.

### 3. Standard suites added (`libero_combined_bowl5_initpose`)

The first 5,186 episodes of the distributed `libero_combined_20hz` were appended: the standard
suites (`libero_10`, `libero_spatial`, `libero_object` — 1,265 episodes) and **`libero_90`**
(3,921 episodes), giving 111 tasks in total.

**The gripper encoding was fixed here.** The leading gripper value in these episodes is
`{0, 1}` where the rest of the data uses `±1`. Left uncorrected, "open" carries the opposite
label.

Removal rules differed by source: the standard suites used the same rules as above
(bowl tasks 218, over-length 65, contact 71); `libero_90` dropped only the 10 over-length
episodes, because the original HDF5 needed for contact scoring was not available.

### 4. libero_90 cleaned (this dataset)

**250 `libero_90` episodes judged dirty were replaced with clean demonstrations of the same
tasks, at the same count.** Episode counts per task are unchanged; only the frame count drops
(−5,803 versus the parent).

**The 250 replacements are appended at the end of the dataset**, so episode indices do not line
up with the parent version.

## Known limitations

- **The `libero_90` portion has not been contact-cleaned.** Steps 1–2 cleaned the 40
  LIBERO-plus tasks, but `libero_90` only had over-length episodes removed. **Demonstration
  quality is therefore not uniform across tasks.**
- **Adding data did not reliably help.** A model trained on the parent
  (`libero_combined_bowl5_initpose`) scored clearly *below* one trained on the smaller
  `libero_plus_bowl5` on our held-out set (0.3515 vs 0.4055). Broadening from 40 tasks to 111
  hurt on that measure. Treat "more tasks" as a hypothesis to test, not an improvement.
- **Continued training on this dataset damaged a good model.** Taking a checkpoint that scored
  0.498 on the competition and training 8,000 further steps here produced a model that scored
  well on `libero_90` tasks (0.422 vs 0.156) and was indistinguishable from its parent on the
  target task distribution. Its competition score is not a clean measurement — that submission
  was built with a broken tokenizer — but the trade it made is visible and worth knowing about.
- **Appearance variation is layered, not sampled per frame.** It is applied at recording time,
  so a given episode has one appearance throughout.

## Rebuilding from components

**This dataset can be rebuilt, episode for episode, from its components plus the manifests.**
The recipe, source and manifests live under `rebuild/` in this repository and in
[`libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose/tree/main/rebuild).

**It was done for real on a different machine on 2026-09-18; all 21,642 action sequences matched.**

```
行動列が一致した本: 21642 / 21642      (matching episodes: 21642 / 21642)
```

Three components are needed: [`libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose),
[`libero_90_clean`](https://huggingface.co/datasets/siruku6/libero_90_clean),
and the organizers' `libero_combined_20hz` (LeRobot v3.0).

## License and provenance

Derived from LIBERO and LIBERO-plus, distributed under the **MIT License**
(Copyright (c) 2023 Lifelong Robot Learning). Re-recorded demonstrations in this dataset were
produced by a scripted policy in the same simulator and are released under the same terms.

## Related

- Model trained on this dataset: [`siruku6/molmoact2_lora_v1_40k`](https://huggingface.co/siruku6/molmoact2_lora_v1_40k)
- Parent dataset `libero_combined_bowl5_initpose`: **not published** — build it with the bundle below
- 40-task branch: [`siruku6/libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose)
