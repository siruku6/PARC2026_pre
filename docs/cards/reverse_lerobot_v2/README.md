---
license: mit
task_categories:
- robotics
tags:
- lerobot
- libero
- vision-language-action
- imitation-learning
- synthetic
---

# reverse_lerobot_v2

*日本語版: [README.ja.md](README.ja.md)*

**LeRobot-format demonstrations for "reverse-direction" tasks** — taking an object *out* rather
than putting it *in*. Every episode was produced by a scripted policy inside the simulator;
none of it is a subset of distributed data.

| | |
|---|---|
| Episodes | **397** |
| Frames | **61,430** |
| Tasks | **3** |
| Format | LeRobot v3.0, 20 Hz |

## What it is for

**PARC2026's Track 3 asked for the reverse of the training tasks** (organizer disclosure,
2026-08-28). Where the forward data says "put the soup can in the tray", Track 3 asks
**"take the can out of the tray and put it on the table"**.

**Forward-trained weights scored 0 of 8 on every Track 3 submission.** This dataset exists to
close that gap.

| Task | What it does |
|---|---|
| Yellow book | take it off the shelf |
| Alphabet-soup can | take it out of the tray |
| Red mug | take it off the left plate |

**All 397 instruction strings are byte-identical to the Track 3 bddl text.**
**Track 3's fourth task — the two mugs — has no episodes here.**

## How it was generated

**These are not forward demonstrations played backwards.** They are episodes recorded by
stepping the simulator normally, forward in time.

**Time reversal is used only to produce the reference path for the carry segment** — after the
object is grasped. **The approach-and-grasp segment is synthesized**, because a forward
demonstration contains only "how to approach before grasping" and says nothing about where to
put the object once it is out.

1. `gripper_shift_sweep.py` sweeps every (forward demo × distributed initial state) pair.
   The best combination was `--mode phased --init-source t3 --grasp-frame clamped --standoff 0.06`
2. **A dry run first, then a min-cost assignment** (`scipy.optimize.linear_sum_assignment`), so
   **each of the 50 distributed initial-state rows is used once, with the best-quality demo**
3. Episodes that reached the reverse goal are written as npz and converted with
   `npz_to_lerobot.py`

## What changed from v1

Three changes from v1 (`reverse_lerobot_389`, 389 episodes, 67,775 frames):

- **Rotation limit applied by scaling the whole rotation vector**, not by clipping per component
  (`ROT_LIMIT=0.375`)
- Dropped the book task's horizontal-levelling motion
- Settle for 80 frames after the lift

| | v1 | v2 |
|---|---|---|
| Frames with \|z\| > 10 | 15.2 % | **0.0 %** |
| Max \|z\| | 26.1 | 9.8 |
| Drop | 40 mm / 37 mm | **1.0 mm / 2.9 mm** |
| Red mug | 139 | 147 |
| Total | 389 | **397** |
| Median length | 173 | 152 |

## Known limitations

- **It damages forward tasks that share objects.** Training with 49.5 % reverse data dropped
  the object-sharing rows by **22 to 25 points**, while the rest stayed flat. **The mixing ratio
  is a trade, not a free addition.**
- **The Track 3 submission score stayed 0.000 throughout.** Local evaluation became non-zero for
  the first time with this lineage (0.000 → 0.0235), but that never reached the leaderboard.
- **No demonstrations for the two-mugs task.**
- **The arm's initial pose is identical across every episode.**
- **Colour-only disambiguation is not learned.**

## Mixing

[`siruku6/jaist_mix_v2`](https://huggingface.co/datasets/siruku6/jaist_mix_v2) (6,176 episodes)
is `libero_plus` thinned to 75 episodes per task (3,000) plus **this dataset duplicated eight
times** (397 × 8 = 3,176). **The reverse share is 49.5 %.**

```bash
COPIES=8 PER_TASK=75 REV_DIR=work/artifacts/reverse_lerobot_v2 \
MIX=jaist_mix_v2 bash solution/train/jaist_dataset_build.sh
```

Normalization statistics are frozen to the forward side's (difference 0.000e+00).

**3,176 of that mixture's episodes are eight copies of these 397.** Stored as the component it
is one-eighth the size.

## License

Built from LIBERO (MIT, Copyright (c) 2023 Lifelong Robot Learning). The scripted demonstrations
here were produced in the same simulator and are released on the same terms.
