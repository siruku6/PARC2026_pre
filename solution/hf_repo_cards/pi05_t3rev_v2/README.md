---
license: gemma
license_link: https://ai.google.dev/gemma/terms
base_model: siruku6/pi05_full_runpod
datasets:
  - siruku6/jaist_mix_v2
  - lerobot/libero_plus
library_name: lerobot
pipeline_tag: robotics
tags:
  - pi0
  - pi05
  - vla
  - libero
  - lerobot
  - negative-result
---

# π0.5 LIBERO — reverse-direction manipulation, rotation-clipped data (10,000 steps)

An action-expert fine-tune of [`siruku6/pi05_full_runpod`](https://huggingface.co/siruku6/pi05_full_runpod)
on a mixture of [`lerobot/libero_plus`](https://huggingface.co/datasets/lerobot/libero_plus) and
**~49.5 % synthetic reverse-direction episodes** we generated in LIBERO simulation.

The target skill is the *reverse* of what LIBERO demonstrates: the distributed demos put objects **onto**
shelves, plates and trays, and the tasks we needed take them **off** again (`take the red mug off the left
plate and put it on the table`). Only one of 2,066 instruction strings in the distributed training data is
a "take … off / out of" task, so we synthesised the missing direction with a scripted controller.

> **This is a published negative result.** The data fix this run was built to test *did* land — the policy
> now issues wrist-rotation commands of the same magnitude as the data, instead of shrinking them toward
> the mean. **But the policy still does not grasp.** Across 20 scripted evaluation rollouts it lifted the
> target object zero times; the two runs where the object rose more than 30 mm were the arm knocking the
> scene over (other objects displaced by 279 mm and 299 mm). The weights and the training log are here so
> the comparison is reproducible.

## Lineage

| Stage | Weights | Steps | Trainable | Note |
|---|---|---|---|---|
| Base | `lerobot/pi05_libero_base` @ `a217bfd3b14673cf2ce597e69997ab21866438dd` | — | — | π0.5, 4.14 B params |
| +1 | `siruku6/pi05_stage1_24000` | 24,000 | action expert | `lerobot/libero_plus` |
| +2 | [`siruku6/pi05_full_runpod`](https://huggingface.co/siruku6/pi05_full_runpod) | 3,000 | **all 4.14 B** | `lerobot/libero_plus` |
| **+3 (this repo)** | `checkpoints/002000` … `010000` | **10,000** | action expert | **+ reverse-direction data, rotation clipped** |

## Contents

Five checkpoints, saved every 2,000 steps, in LeRobot's on-disk layout, plus the full training log.

```
checkpoints/002000/pretrained_model/   config.json, model.safetensors (9,354,050,752 B),
                                       policy_preprocessor.json + normalizer safetensors,
                                       policy_postprocessor.json + unnormalizer safetensors,
                                       train_config.json
checkpoints/002000/training_state/     optimizer_state.safetensors (2,195,136,504 B),
                                       optimizer_param_groups.json, scheduler_state.json,
                                       rng_state.safetensors, training_step.json
... 004000/ 006000/ 008000/ 010000/
pi05_t3rev_v2.log
```

**The `vlm/` tokenizer directory is not included.** A policy server that constructs the tokenizer offline
needs it; copy it from the base repo (`siruku6/pi05_full_runpod`), 17 MB.

## Training setup

| | |
|---|---|
| Objective / policy | π0.5 (`pi05`), flow-matching action expert, action chunk 50, `n_action_steps` 10 |
| Trainable | **693,422,112 of 4,143,404,816 parameters** (`freeze_vision_encoder=true`, `train_expert_only=true`) |
| Batch / LR | 16 / 5e-6 peak → 2.5e-6, cosine decay, 1,000 warmup steps |
| Precision | bfloat16, gradient checkpointing on |
| Seed | 42 |
| Image augmentation | brightness / contrast / saturation / hue / sharpness / affine / resized-crop / perspective, up to 4 per sample |
| Normalization | `MEAN_STD` for state and action, pinned to the base model's normalizer (difference 0.000e+00) |
| Data | [`siruku6/jaist_mix_v2`](https://huggingface.co/datasets/siruku6/jaist_mix_v2) — 6,176 episodes, 992,843 frames, 43 tasks, 20 fps, Franka Panda |
| Hardware / time | 1× NVIDIA H100-20C (19.8 GB), 3 h 52 min, 1.375 s/step, 16.27 GB VRAM |
| Framework | LeRobot v0.6.0 |

## The data

| Part | Episodes | Unique frames | Share of samples |
|---|---|---|---|
| `lerobot/libero_plus`, thinned to 75 episodes per task | 3,000 | 501,403 | 50.5 % |
| **Reverse-direction episodes, replicated ×8** | 397 → 3,176 | **61,430** | **49.5 %** |

The reverse episodes cover three tasks (yellow book off a shelf, alphabet soup out of a tray, red mug off
a plate). They are **not time-reversed playback**. Each one is an ordinary forward-in-time simulator
rollout in six phases — approach with the gripper open, close in place, lift, follow a return path, place,
release — recorded at 20 fps with images, actions and states all advancing in time. Only the return
path's *reference trajectory* is read backwards out of a distributed forward demo.

### What changed versus the previous attempt

An earlier run used a first version of this data in which the scripted controller's **per-step wrist
rotation command saturated at the ±1 action clip**, while the distributed demos stay inside ±0.375. 15.2 %
of frames exceeded 10 σ, and **52.6 % of those were in the first 10 % of each episode** — exactly the
approach phase. This version rate-limits rotation to ±0.375, and 0.0 % of frames exceed 10 σ.

The effect on the policy is measurable, and it is the one the fix predicted:

| 95th percentile of per-step rotation command | value |
|---|---|
| distributed demos (the envelope) | 0.375 |
| **policy trained on v1 data** | **0.25** — shrunk toward the mean |
| **policy trained on this data** | **0.39** — matches the envelope |

Training loss also comes down: 0.41–0.44 → **0.346** at the same 8,700 steps, same recipe, same mixture
ratio.

## Evaluation — what still fails

Measured in LIBERO simulation on the four reverse-direction evaluation tasks (three of which have
reverse training data), 300 steps per rollout.

| | v1-data policy | **this policy** |
|---|---|---|
| Grasped and lifted the target | 0 / 20 | **0 / 20** |
| Lifted by knocking the scene over | 1 | 2 |
| Distance to a grasp pose the data succeeded from, median | 160.4 mm | **145.4 mm** |
| … within 30 mm | 0 / 12 | **0 / 12** |
| Fraction of the training trajectory tracked within 30 mm | 0.02 | 0.02 |

Object selection is *not* the bottleneck. Measured without rollouts — integrate the 50-step action chunk
into a heading, compare it against every object in the scene — the sibling policy trained on the earlier
version of this data aims at the correct object on two of the three tasks (cosine +0.97 and +0.76, against
the strongest competitor). The failure is downstream: started from a training episode's own initial state,
the policy leaves the recorded path within the first 10–20 % of it and never comes back within 8 cm, where
two valid recorded paths from the same initial state differ by only 1–20 mm.

Raising the number of executed steps per plan from 16 to 50 makes it worse on every measure, so plan
truncation is not the cause either.

## Use

```sh
hf download siruku6/pi05_t3rev_v2 --include 'checkpoints/010000/pretrained_model/*' \
    --local-dir ./t3rev_v2
hf download siruku6/pi05_full_runpod --include 'vlm/*' --local-dir ./t3rev_v2/checkpoints/010000/pretrained_model
```

Then point LeRobot at the local directory:

```sh
lerobot-train --policy.type=pi05 \
    --policy.pretrained_path=./t3rev_v2/checkpoints/010000/pretrained_model ...
```

## Intended use and limitations

Research artifact. Trained and evaluated **only in LIBERO simulation** with a simulated Franka Panda —
there is no real-robot validation, and nothing here should be run on physical hardware without your own
safety review. Performance outside the LIBERO task and camera setup is unknown. **The policy does not
reliably grasp**; see the evaluation section before building on it.

## License

These weights are a Model Derivative of Gemma (via PaliGemma inside π0.5) and are released under the
**[Gemma Terms of Use](https://ai.google.dev/gemma/terms)**. Use is also subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). See [NOTICE](NOTICE) for
the third-party attributions that come with the base model, the data and the training code.
