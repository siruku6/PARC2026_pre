---
license: gemma
license_link: https://ai.google.dev/gemma/terms
base_model: lerobot/pi05_libero_base
datasets:
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

# π0.5 LIBERO — scene-camera-dropout variant (2,500 steps)

A full-parameter fine-tune of [`lerobot/pi05_libero_base`](https://huggingface.co/lerobot/pi05_libero_base)
on [`lerobot/libero_plus`](https://huggingface.co/datasets/lerobot/libero_plus), trained with the
third-person ("scene") camera **blanked to zeros with probability 0.5 per sample**, so that the policy is
pushed to rely on the wrist camera.

> **This is a published negative result.** In our simulated LIBERO evaluation the dropout variant did
> *not* beat its identically-trained non-dropout sibling, and we stopped the line of work. The weights and
> the training log are here so the comparison is reproducible.

## Lineage

| Stage | Weights | Steps | Note |
|---|---|---|---|
| Base | `lerobot/pi05_libero_base` @ `a217bfd3b14673cf2ce597e69997ab21866438dd` | — | π0.5, 4.14 B params |
| +1 | `siruku6/pi05_full_runpod` | 3,000 | full-parameter fine-tune, no dropout |
| +2 | `siruku6/pi05_trial1500` `001500` | 1,500 | no dropout |
| **+3 (this repo)** | `000500` … `002500` | **2,500** | **scene-camera dropout p=0.5** |

The only difference from the non-dropout sibling run (same starting weights, same 2,500 steps, same
hyper-parameters, same seed) is the dropout.

## Contents

Five checkpoints, saved every 500 steps. Each directory is the flattened content of a LeRobot
`pretrained_model/`: `config.json`, `model.safetensors` (9,354,050,752 bytes), `train_config.json`.
`logs/pi05_camdrop2500.log` is the full training log.

```
000500/  001000/  001500/  002000/  002500/  logs/
```

## Training setup

| | |
|---|---|
| Objective / policy | π0.5 (`pi05`), flow-matching action expert, action chunk 50, `n_action_steps` 10 |
| Trainable | **all 4,143,404,816 parameters** (`freeze_vision_encoder=false`, `train_expert_only=false`) |
| Batch / LR | 64 / 5e-6 peak, cosine decay with warmup |
| Precision | bfloat16, gradient checkpointing on |
| Seed | 42 |
| Image augmentation | brightness / contrast / saturation / hue / sharpness / affine / resized-crop / perspective, up to 4 per sample |
| Data | `lerobot/libero_plus` — 14,347 episodes, 2,238,036 frames, 40 tasks, 20 fps, Franka Panda |
| Hardware / time | 1× NVIDIA RTX PRO 6000 Blackwell, 4 h 46 min 37 s, 6.84 s/step, 49.4 GB VRAM |
| Framework | LeRobot v0.6.0 |

## The dropout

For each training sample the scene-camera image is replaced with zeros with probability 0.5; the wrist
camera is never touched. The realised rate, counted every 20,000 frames, was 0.500 / 0.502 / 0.502 / 0.505.

**It costs nothing to run.** Against the non-dropout sibling: step time 6.833–6.835 s vs 6.832–6.834 s,
data time 0.007 s in both, 49.42 GB VRAM in both.

**It does change what the model learns.** Training loss sits above the sibling's throughout, and the gap
stops closing after ~900 steps:

| Steps | 1–100 | 401–500 | 801–900 | 1401–1500 | 2001–2100 | 2401–2500 |
|---|---|---|---|---|---|---|
| dropout | 0.2662 | 0.2566 | 0.2558 | 0.2506 | 0.2500 | 0.2532 |
| no dropout | 0.2382 | 0.2354 | 0.2362 | 0.2304 | 0.2304 | 0.2354 |
| **gap** | **+11.8%** | +9.0% | +8.3% | +8.8% | +8.5% | **+7.6%** |

Read this as: what the scene camera was carrying is genuinely gone, and 2,500 steps of wrist-only practice
did not make it back.

## Use

```sh
hf download siruku6/pi05_camdrop2500 --include '002500/*' --local-dir ./camdrop2500
```

Then point LeRobot at the local directory:

```sh
lerobot-train --policy.type=pi05 --policy.pretrained_path=./camdrop2500/002500 ...
```

Note that the `pretrained_model/` level is flattened away in this repo, so `lerobot-train --resume`
against the Hub path will not find `train_config.json` where it expects it. Downloading first and passing
a local `--policy.pretrained_path` works.

## Intended use and limitations

Research artifact. Trained and evaluated **only in LIBERO simulation** with a simulated Franka Panda —
there is no real-robot validation, and nothing here should be run on physical hardware without your own
safety review. Performance outside the LIBERO task and camera setup is unknown.

## License

These weights are a Model Derivative of Gemma (via PaliGemma inside π0.5) and are released under the
**[Gemma Terms of Use](https://ai.google.dev/gemma/terms)**. Use is also subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). See [NOTICE](NOTICE) for
the third-party attributions that come with the base model, the dataset and the training code.
