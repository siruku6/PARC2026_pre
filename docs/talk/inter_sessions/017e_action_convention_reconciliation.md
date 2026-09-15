# 017e. Settle what the `action` column actually is (014's measurements contradict the prototype's stated convention)

> **Status: open** ／ Assignee: the session working on the 006 prototype (`solution/tools/track3_reversal/`) ／ Updated: 2026-09-05
>
> **English mirror of [017](017_action_convention_reconciliation.md).** Same request, same numbers.
> **017 (Japanese) is authoritative** — if the two ever disagree, trust 017 and fix this file.
> Reply in whichever file you prefer.

## Background

PARC2026 finals. Track1 0.492 / Track2 0.233 / **Track3 0.000** (best other entrant: 0.09603
= 1 successful run out of 8). Deadline is **9/17**.

Track3 is the **reverse** of existing LIBERO tasks ("put A on B" → "return B to A"). The
training data contains zero reverse-direction examples, which reads as the main cause of the
0.000. The leading plan for reverse data is **time reversal** — walk a forward demo backwards
([006](006_track3_reverse_direction.md)).

**This request feeds directly into 006's "forward 0/3" diagnosis.** Section 2 of 006 reports
that replaying a demo forward satisfies task 92's `:goal` only 0 times out of 3, and lists
three candidate causes. Candidate **(b) is "action convention drift (gripper polarity, step
size)"**. While measuring that same `action` column for
014 (**closed 9/7; its findings now live in
[reference/25](../../reference/25_dataset_inventory.md) §8.7–8.8**),
**I found it contradicts the convention the prototype declares in its own docstrings.** 014 was scoped to "is `state` usable for time
reversal", so I cannot settle this from here — **it needs the side that has the simulator.**

What is already established, and what is out of scope:

- The `observation.state` column is sound (8 dims, no NaN, not a static copy). **Settled.**
- The time-reversal procedure itself (reverse the absolute poses, rebuild the deltas) **is
  self-consistent** — a roundtrip test gives median residuals of 0.6–0.9 mm. **`state` is not
  the problem.**
- 014's first report contained three errors. **Every number below is the corrected value.**
- Image/video content and the `IPEC-COMMUNITY` v2.1 datasets were not examined (not needed).

## What I need from you

Decide whether the `action` column is **normalized relative deltas** or **raw OSC commands you
can replay as-is**, by **replaying one forward episode in the simulator**. Deliverable: **a
one-line verdict**, plus the **per-frame `state` comparison** (which frame it starts diverging at)
that backs it.

## Why it's needed

**This single point changes how reverse demos get built**, which changes the schedule to 9/17.

| If `action` is… | How reverse demos get built | 006's forward 0/3 is… |
|---|---|---|
| **Normalized relative deltas** | Reverse the absolute poses from `state`, rebuild deltas, divide by the fitted constant (014 already roundtrip-tested this) | Explained by convention drift. `actions[::-1]` **needs a sign flip**, so the prototype needs fixing |
| **Raw OSC commands** | Dividing by a constant is not enough — you need **controller-aware inversion**, or you re-source the trajectory by replaying inside the kit | Not a convention problem. Cause is (a) physics/geometry or (c) init-state mismatch |

Either way 006's next step changes. Until it's settled, time reversal has to carry two
implementations.

## Established facts

### The two places that disagree

**What the prototype declares** (`solution/tools/track3_reversal/reversal_probe.py`):

- `build_reversed_actions` docstring (lines 50–68): "Data convention:
  **`use_relative_actions=False` (absolute EEF pose, 6 dims + gripper, 1 dim)**. Reversing
  absolute poses is meaningful (it retraces the same spatial path)."
- Line 129, the `--gripper-mode` help text, repeats it: "the data is `use_relative_actions=False`
  (absolute pose)".
- The implementation is `rev = np.ascontiguousarray(actions[::-1].copy())` — **it reorders and
  never negates.** That is correct *if* the values are absolute poses. If they are relative
  deltas, it is **exactly the failure mode 014's request warned about**: "simply reversing frame
  order destroys the meaning of 'offset from the previous pose'."
- `gripper_mode=invert` does `rev[:, 6] = 1.0 - rev[:, 6]`. But **`action[6]` only ever takes the
  values ±1**, so `1 − x` yields {0, 2}. **2 is outside the command range.** The expression looks
  written for a `[0,1]` gripper.

**What 014 measured** (`~/src/train_data/verify014_calib.py`, runs in 1.4 s):

The column I measured is **the same column** `host_extract_actions.py` pulls from
`libero_combined_20hz_extracted/.../data/chunk-000/file-000.parquet` — same file, same column
name. So the two are asserting opposite things about identical numbers.

### 014's measurements (corrected)

Dataset: `libero_combined_20hz` at `~/src/train_data/libero_combined_20hz_extracted/`. There is
exactly **one** data file, `data/chunk-000/file-000.parquet`: 136 MiB, **3,028,708 rows,
19,533 episodes** (indices 0–19532).

`observation.state` is 8 dims — `[eef_pos(3), axis_angle(3), gripper_qpos(2)]` — at **raw scale**
(eef_pos ±0.48 m, axis-angle ±4.8 rad, each finger 0–0.04 m; opening width `d6 − d7` spans
0.004–0.080 m). `action` is 7 dims, **clipped at ±0.9375 (= 60/64)**, and `action[6]` is
**exactly ±1**.

Episodes read: **ep 5195** "put the bowl on the plate" (92 frames, pick & place) and **ep 5225**
"open the top drawer and put the bowl inside" (184 frames, drawer — chosen to check whether
contact other than grasping corrupts the state column).

| Measurement | ep 5195 | ep 5225 | 20 episodes sampled at random from all 19,533 |
|---|---|---|---|
| Frame pairing (decided by correlation) | **A: `action[t]` ↔ `t→t+1`** (0.9525 vs B 0.8891) | **A** (0.9501 vs 0.9271) | — |
| Translation scale | 0.01138 m/unit | 0.01072 m/unit | mean 0.01110, sd 0.00211 (**±19%**), range 0.00580–0.01293 |
| Translation per-component median residual | 0.59 mm | 0.84 mm | mean 0.75 mm, range 0.27–1.64 mm (**3 of 20 exceed 1 mm**) |
| Translation correlation | 0.952 | 0.926 | — |
| Rotation scale (**world**, `R[t+1] @ R[t].T`) | 0.0913 rad/unit | 0.0734 rad/unit | mean 0.10149, sd 0.00977 (±10%), range 0.08220–0.11530 |
| Rotation per-component correlation (world) | **0.814** | **0.802** | — |
| Rotation per-component correlation (body, `R[t].T @ R[t+1]`) | −0.225 | −0.182 | — |

**Build rotations in the world frame.** The body frame drives the correlation negative.
And do not subtract axis-angle vectors componentwise (`np.diff(axis_angle)`) — that is not a
relative rotation, and it drops the correlation to 0.221 / 0.335. **014's first report made
exactly this mistake**, which is why its rotation figures (0.777 / 0.50) should be ignored.

### What 014 could *not* decide — the heart of this request

**Whether `action` is the delta that was *achieved* or the delta that was *commanded* is
unresolved.** Fitting `pos[t+k] − pos[t] ≈ scale × action[t]` across lookaheads `k`, the
**correlation peaks at `k=3`**, not `k=1`:

| k | 1 | 2 | **3** | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| ep 5195 correlation | 0.9518 | 0.9685 | **0.9741** | 0.9693 | 0.9332 | 0.8740 |
| ep 5225 correlation | 0.9262 | 0.9348 | **0.9388** | 0.9384 | 0.9257 | 0.9020 |

(Residual is the wrong statistic here — it grows mechanically with `k` regardless of which
hypothesis holds, because the displacement grows while `action[t]` stays fixed. Correlation is
the discriminating one. 014's first report used the residual and drew the opposite conclusion;
that conclusion is **withdrawn**.)

`k=1` not being the best fit suggests **the controller does not reach its target within one
frame** — i.e. `action` is closer to a *command*. That is consistent with reading `action` as a
raw OSC command whose fitted "scale" is really a **response ratio** (achieved ÷ commanded).

**The rotation axis gives an independent check.**
`solution/tools/scripted_demo/scripted_demo.py:164` documents the yaw (wrist) command as
"**1.0 is the ceiling (`output_max` is 0.5 rad)**". The measured rotation scale is
**0.10149 rad/unit**, so the response ratio is **0.10149 / 0.5 ≈ 0.20**. Doing the same for
translation (taking `output_max` = 0.05 m) gives **0.01110 / 0.05 ≈ 0.22**. **Translation and
rotation independently land on ≈0.2.** One explanation — "at 20 Hz, one step achieves about 20%
of the commanded motion" — accounts for both. **If that reading is right, `action` replays as-is**,
meaning the prototype has the relative/absolute label wrong but the "just feed it in" part right.

Caveat: the translation `output_max` of 0.05 m is **unverified on this host** (robosuite is not
installed). **The 0.5 rad figure also rests on a single line in the repo, and it describes the
yaw axis specifically** — whether all three rotation axes share that value is unconfirmed.
**Please record the actual `output_max` from the controller config and put it in your report** —
whether the translation and rotation response ratios really agree at ≈0.2 is the deciding
evidence for this reading.

**A second piece of evidence:** the literal check 4 from 014's request **fails**. Even after
fitting the normalization constant, the fraction of frames where the state delta matches `action`
**within 1 mm is 23.1% / 15.3%**, far below the 90% the request set as its bar. Quantization
(1/64 of a unit ≈ 0.18 mm) is far too small to explain the gap, so **`action` is not a constant
multiple of the single-frame state delta.**

### The gripper is unresolved

The two state dims are sign-symmetric across the fingers (`d7 ≈ −d6`). Recovering `action[6]`
from the sign of the width change agrees **93.5% / 83.8%** of the time (frames where the width
does not move are undecidable).

But **how to build the gripper channel for a reversed demo was not measured.** Reversing the
forward `action[6]` gives you a valid ±1 sequence, but **time reversal swaps "grasp" and
"release"**, so whether that sequence *means* the right thing is a separate question. The
prototype offering `keep` / `invert` / `skip` is the right posture — **none of them has been
shown correct.**

## What to do

1. **Replay one forward episode and compare `state` frame by frame.** This is the same work as
   item 1 of 006's section 3 — the difference is that **014 now supplies the reference values to
   compare against.** For ep 5195 or 5225, get the reference `state` series with:

   ```
   /home/ubuntu/.pyenv/versions/3.12.14/bin/python ~/src/train_data/verify014_calib.py scale
   ```

   To use the task 92 episodes instead (ep 4061–4063), edit `TARGET_EPISODES` in
   `verify014_calib.py` and rerun (1.4 s).
2. **Decide** from how far the replayed `eef_pos` drifts from the dataset's
   `observation.state[:, :3]`, per frame:
   - **Within a few mm across all frames → `action` is a raw command and replays as-is.** In that
     case reverse demos *cannot* be built by dividing by a constant.
   - **Diverges from the very start → init state or environment mismatch** (not a convention
     problem; 006's candidates (a)/(c)).
   - **Diverges gradually from mid-episode → physics difference, or the step size is off.**
3. **Resolve the `gripper_mode=invert` expression.** Since `action[6]` is ±1, `1.0 - x` returns
   {0, 2}. If the intent is "flip ±1", it should be `-x`; if the intent is "flip a `[0,1]` value",
   then the input range is not what the code assumes. Decide which.
4. **Record the settled convention as one line in `docs/reference/25_dataset_inventory.md`.**
   Section 8.5 already documents `observation.state` as 8 dims and `action` as 7 dims, but
   **says nothing about relative vs absolute, or about normalization** — that silence is where
   this disagreement came from.

## Definition of done

- **A one-line verdict: `action` is a raw command / normalized relative deltas.**
- The **per-frame `eef_pos` drift** backing it (max mm, and the frame divergence starts at).
- A one-line decision on the `gripper_mode=invert` expression.
- The above reflected into `reference/25`.

## Out of scope

- Generating the real reverse dataset (all 100 tasks). That waits on this verdict.
- Re-validating `observation.state` — 014 covered it. **The state column is sound.**
- Widening the scan beyond 014's two episodes. One replay settles the convention.
- Training or submitting. This request closes with a doc edit and one simulator replay.

## How to report

A table (per-frame drift, divergence frame index) + the one-line verdict + the commands you ran.
Keep prose to 5 lines.

## If a judgement call comes up

This request exists to decide **whether reverse demos get built by dividing by a constant or by
going through the controller.** **If you think of a better way to measure that, change it** —
e.g. replay 3 episodes instead of 1, use ep 5195 instead of task 92, grab one video via EGL.
If you change the method, say why in your report.

If the replay comes back genuinely ambiguous, write one closing paragraph on **whether to switch
to the "replay demos inside the kit and keep only the ones that succeed" construction**, with
your reasoning — 006's section 3 already sets up that branch.
