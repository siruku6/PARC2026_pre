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

# libero_90_clean

*日本語版: [README.ja.md](README.ja.md)*

**LIBERO's `libero_90` with demonstrations that bump into non-target objects removed.**
**The last 250 episodes are re-recorded** — that is the part of this dataset that cannot be
reproduced from anything else.

| | |
|---|---|
| Episodes | **3,911** |
| Frames | **560,254** |
| Tasks | **73** |
| Format | LeRobot v3.0, 20 Hz |

## Why it exists

**Contact cleaning had been skipped for the `libero_90` portion.** The 40 LIBERO-plus tasks were
filtered on contact, but `libero_90` had **only the 10 over-300-frame episodes dropped** —
because the original HDF5 needed to score contact (66.7 GB) was not available locally.

**That was finished on 2026-09-16.** The result is this dataset, which serves as the
**replacement pool** for
[`libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean).

## What counted as dirty

**Touching or displacing a non-target object**, scored from the raw MuJoCo states in LIBERO's
own `libero_90` HDF5 (no replay).

| | |
|---|---|
| Reference frame | 10 (chosen from 0, 5, 10, 20) |
| Displacement | **L1 sum over the three axes** |
| Threshold | **1.0 mm** (same as the grader) |
| Non-target | observation `<name>_pos` that is not `robot0*`, not `_to_robot0_eef_pos`, and not in the bddl `:obj_of_interest` |

**Across 4,500 demonstrations:** 296 displaced something by more than 1 mm, 296 touched
something, **363 (8.1 %) did either.** Of the 3,911 `libero_90` episodes in the mixture,
**250 were judged dirty.**

## Composition

**Measured, not assumed** — sha256 over the six arm dimensions of the action sequence,
2026-09-19.

| Part | Episodes | Origin |
|---|---|---|
| Head (indices 0–3660) | **3,661** | **Straight from the distributed data** — all 3,661 match `libero_combined_20hz` down to the arm motion |
| Tail (indices 3661–3910) | **250** | **223 of them appear nowhere in the distributed data — they are re-recorded.** The remaining 27 do match it |

**This is why the dataset cannot be deleted.** Those 223 would require re-running the scripted
policy to reproduce, and would not come out identical.

**Gripper values are normalized to `±1`** — left at the distributed `{0,1}`, "open" carries the
opposite label.

## How it is used

**As the refill pool for the mixture.** The 250 dirty episodes are dropped from the parent's
`libero_90` range (indices 17731–21641) and **this dataset's last 250 are appended at the end.**
Per-task episode counts are unchanged; only the frame count drops (**−5,803**).

**Provenance was established by sha256 of the action sequences**, not by name or length —
`(task, length)` does not identify the 250 uniquely.

**The full rebuild bundle is in
[`libero_plus_bowl5_initpose`'s `rebuild/`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose/tree/main/rebuild).**
It was run on a different machine on 2026-09-18 and **all 21,642 action sequences matched.**

## Known limitations

- **The procedure that produced the 250 re-recorded episodes is not a script.** It was done by
  hand on 2026-09-16. **The ledger of which episode went where survives**
  (`l90clean_replace.json`), **but the steps to generate those 250 again do not.**
- **`libero_90` has no initial-pose-varied and no appearance-varied demonstrations.** Those
  exist only on the LIBERO-plus side, so **demonstration quality is not uniform across tasks.**
- An appearance-layered sibling exists,
  [`libero_90_clean_appearance`](https://huggingface.co/datasets/siruku6/libero_90_clean_appearance),
  **but the refill used this one** (the un-layered side).

## License

Built from LIBERO (MIT, Copyright (c) 2023 Lifelong Robot Learning). The re-recorded
demonstrations are released on the same terms.
