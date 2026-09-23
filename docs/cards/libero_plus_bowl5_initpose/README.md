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

# libero_plus_bowl5_initpose

*日本語版: [README.ja.md](README.ja.md)*

A LeRobot-format imitation-learning dataset for LIBERO-style manipulation tasks,
covering the same 40 evaluated tasks as
[`siruku6/libero_plus_bowl5`](https://huggingface.co/siruku6/libero_plus_bowl5), with
part of it replaced by newly collected demonstrations that vary the robot's initial
pose and scene appearance.

| | |
|---|---|
| Episodes | 16,820 |
| Frames | 2,712,308 |
| Tasks | 40 |
| Size | ~22.6 GB |

## How it was built

Starting from `libero_plus_bowl5` (14,085 episodes: `libero_plus` with the 5
highest-collision "bowl" tasks' demonstrations replaced by re-collected scripted ones):

1. Removed 1,532 episodes that either touched a non-target object (contact at reference
   frame 10, >1mm displacement) or exceeded 300 frames (past the scoring cutoff).
2. Backfilled the same per-task episode counts with 3,972 newly collected
   demonstrations across all 40 tasks, all with a perturbed robot initial pose (except
   two "moka pot" tasks — see below) and blended scene appearance (lighting,
   table/background texture, camera — 25% weight each, physics unaffected).
3. Distributed the remainder to bring every task up to its quota.
4. Added 303 further episodes ("path A"): episodes that had been dropped for exceeding
   300 frames, re-shot with consecutive action commands merged into fewer, larger steps
   so they finish within the frame budget, at the original (unperturbed) initial pose.

| | Episodes |
|---|---|
| Base (`libero_plus_bowl5`) | 14,085 |
| Removed (contact / >300 frames, minus 965 already fixed by the bowl-task replacement) | −1,532 |
| Kept | 12,553 |
| Added — replayed-and-perturbed demonstrations | +3,453 |
| Added — scripted demonstrations | +511 |
| Added — path A (shortened re-shoots) | +303 |
| **Total** | **16,820** |

New demonstrations are produced by first replaying the original recorded actions, and
only invoking an inverse-kinematics solver (Ruckig) to steer toward the original demo's
starting pose when the perturbed initial pose causes the replay to fail a gate. All new
episodes pass three gates: goal reached within 300 frames, no non-target object
displaced beyond 1mm at any point, and (for replay-based ones) the initial end-effector
offset from the original demo is at least 2cm (so it is a meaningfully different start).

## Known gaps

- The two "moka pot" tasks have no perturbed-initial-pose demonstrations: trial runs
  reached the goal in only 1/62 attempts when perturbing pose for those tasks, versus
  50/50 without perturbation, so they were left unperturbed (object placement is still
  varied, drawn from the same 50 layouts as the original human demonstrations).
- The "on the ramekin" bowl task's 61 episodes are skewed toward the initial poses that
  are physically reachable for that layout (572 of 575 failed attempts were collisions
  between the arm and the ramekin while reaching over the rim).
- Not every episode dropped for exceeding 300 frames could be recovered by path A: 10 of
  35 groups (60 episodes) for the seven affected tasks, and 25 of 50 groups for the moka
  pot tasks, did not pass the gates under any tested consolidation level.

## Relationship to other datasets

This dataset is the base for `libero_combined_bowl5_initpose`, which appends a filtered head of
the distributed `libero_combined_20hz` tar (adding 71 tasks outside this dataset's 40) on top of
it unchanged, and in turn for
[`libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean).
**Neither mixture is stored as data** — both are built from components, and the recipe, source
and manifests are in [`rebuild/`](rebuild) here.

**On 2026-09-18 that recipe was run on a different machine and all 21,642 action sequences of
the final mixture matched the published copy**, which is why the mixtures need not be stored.

The episode/frame counts above reflect the 2026-09-14 build.

## License

Built from LIBERO ([Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
MIT) and LIBERO-plus ([Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus),
MIT), plus author-generated scripted and perturbed-initial-pose demonstrations released
under the same MIT terms.

Provided as-is, without warranty of any kind.
