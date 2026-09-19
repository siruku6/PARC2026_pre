---
license: mit
task_categories:
- robotics
tags:
- lerobot
- libero
- vision-language-action
- imitation-learning
---

# libero_90_clean_appearance

*日本語版: [README.ja.md](README.ja.md)*

**`libero_90`, cleaned and then varied in scene appearance** (table, wall, lighting).
**2,830 of its 3,705 episodes appear nowhere in the distributed data** — they are re-recorded.

| | |
|---|---|
| Episodes | **3,705** |
| Frames | **521,185** |
| Tasks | **73** |
| Format | LeRobot v3.0, 20 Hz |

## How it differs from [`libero_90_clean`](https://huggingface.co/datasets/siruku6/libero_90_clean)

| | `libero_90_clean` | this dataset |
|---|---|---|
| Episodes | 3,911 | **3,705** |
| Frames | 560,254 | **521,185** |
| Appearance | **not varied** | **varied** |
| Used to refill the mixture? | **yes** (its last 250) | **no** |

**The refill came from `libero_90_clean`.** This dataset explains only 110 of those 250 by
`(task, length)`, so it is not the source — **provenance was settled by sha256 of the action
sequences.**

## What "appearance" means here

**Four axes, 25 % each by default.**

| Axis | What changes |
|---|---|
| `plain` | nothing |
| `light` | lighting (per-task `_light_N` bddl) |
| `background` | **table and wall textures** (per-task `_table_N` and `_tb_N` bddl) |
| `camera` | viewpoint |

**Physics is bit-identical under these changes** (verified across 6 tasks × 4 variants).
**One appearance per whole episode** — it is applied at recording time, so it cannot vary
frame to frame.

**Sensor noise (`_noise_N`) is deliberately not layered**: it only applies in the outer
wrapper's post-processing and would never land on recorded frames, and the training side
already has equivalent blur augmentation.

**The candidate pools are in a manifest**
(`solution/tools/regen/manifests/l90_appearance_pools.json` — `table`, `tb` and `light` for
three scene families). **That manifest has no camera axis**, unlike the 40-task recordings.

## Composition

**Measured, not assumed** — sha256 over the six arm dimensions, 2026-09-19.

| | Episodes |
|---|---|
| Matching the distributed `libero_combined_20hz` | 875 / 3,705 |
| **Appearing nowhere in the distributed data — re-recorded** | **2,830 / 3,705** |
| Matching `libero_90_clean` | 889 / 3,705 |

**Layering appearance alone would leave the arm motion untouched.** It did not, so **these were
rolled out again.** That is why this dataset cannot be deleted.

## Known limitations

- **The procedure that produced it is not documented.** Neither the drop from 3,911 to 3,705
  nor the appearance mix actually used was recorded. **The composition above was measured on
  2026-09-19, after the fact.**
- **It was not used in any mixture.** None of the published models have seen it.
- `libero_90` has no initial-pose-varied demonstrations (appearance only).

## License

Built from LIBERO (MIT, Copyright (c) 2023 Lifelong Robot Learning). The re-recorded
demonstrations are released on the same terms.
