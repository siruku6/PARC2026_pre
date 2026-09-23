---
license: apache-2.0
base_model: allenai/MolmoAct2-LIBERO-LeRobot
datasets:
  - siruku6/libero_combined_bowl5_initpose_l90clean
library_name: lerobot
pipeline_tag: robotics
language:
  - en
tags:
  - robotics
  - vla
  - vision-language-action
  - libero
  - lora
  - molmoact2
---

# MolmoAct2-LIBERO LoRA (24,000 steps)

*日本語版: [README.ja.md](README.ja.md)*

A LoRA adapter on [`allenai/MolmoAct2-LIBERO-LeRobot`](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot),
trained on [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean).

**This is the checkpoint with measurements.** Its 40,000-step continuation lives in
[`siruku6/molmoact2_lora_v1_40k`](https://huggingface.co/siruku6/molmoact2_lora_v1_40k) and has
never been evaluated. If you want numbers, use this one.

## What it is

| | |
|---|---|
| Base model | `allenai/MolmoAct2-LIBERO-LeRobot` (MolmoAct2 5B, Apache 2.0) |
| Adaptation | LoRA on the VLM only (`enable_lora_vlm: true`; action expert untouched) |
| Rank / alpha / dropout | 64 / 16 / 0.05 |
| Steps | **24,000** |
| Batch | 4 |
| Optimizer | adamw, lr 1e-5, weight_decay 0.0, betas (0.9, 0.95), eps 1e-6, grad_clip_norm 1.0 |
| Scheduler | cosine_decay_with_warmup, warmup 200, peak 1e-5, decay to 1e-6 |
| Seed | 1000 |
| Image augmentation | **on** (up to 3 of brightness/contrast/saturation/hue/sharpness/affine ±5°) |
| Framework | LeRobot v0.6.0 |

Weights are saved **unmerged with PEFT parameter names**; loading needs `peft` and
`enable_lora_vlm: true`.

## Evaluation

84 perturbed LIBERO rows × 5 episodes = 420 episodes, EGL rendering, `samples=1`, no adaptive
replanning. Collision rate excludes scene artifacts no policy can avoid.

| Model | Success | Goal reached | Collision | Smoothness |
|---|---|---|---|---|
| π0.5 `bowl5_full` (different lineage, full fine-tune) | **0.798** | **0.881** | 0.138 | **0.503** |
| **this model (24,000)** | 0.774 | 0.833 | 0.138 | 0.494 |
| 16,000 steps, no augmentation, batch 2 | 0.695 | 0.752 | 0.150 | 0.482 |

Broken down by perturbation type (the 24,000 row against the π0.5 baseline):

| Rows | Contents | π0.5 `bowl5_full` | this model |
|---|---|---|---|
| 12 | base, unperturbed | 0.917 | **0.917** |
| 60 | sensor noise | **0.790** | 0.763 |
| 12 | robot initial pose | 0.600 | 0.600 |

Honest reading:

- **Comparable to the π0.5 baseline, not better.** Every per-type difference is inside the
  ~0.05 run-to-run noise we measured.
- **The 16,000 → 24,000 improvement is confounded**: augmentation off→on, batch 2→4, and
  16k→24k steps all changed together.
- **This evaluation set has no camera-viewpoint, background, or lighting rows.** It cannot
  speak to appearance robustness — which is what half of the competition's own task
  distribution tested.
- **The unmodified base model beat this lineage** on the competition's scoring runs (0.530 and
  0.424 on two tracks). We never got a valid scored run for any of our LoRA checkpoints.

## Usage and the offline-loading trap

See [`siruku6/molmoact2_lora_v1_40k`](https://huggingface.co/siruku6/molmoact2_lora_v1_40k) —
the loading code is identical, and it documents the failure mode in full. In short: the policy
config names `allenai/MolmoAct2-FAST-Tokenizer` as a Hub repository, so **the policy will not
start in an offline environment** unless `discrete_action_tokenizer` points at a local
directory. **A byte-identical copy is bundled here as `fast_action_tokenizer/`** — point the
config at it.

Also note: **adaptive resampling does not help this model.** With `samples=3` plus
`PARC_ADAPTIVE_CADENCE` it scored 0.846 on a 13-task subset, against 0.877 with `samples=1`.
It does help π0.5. Do not carry the setting across model families.

## License and provenance

| Component | Holder | License |
|---|---|---|
| Base weights `allenai/MolmoAct2-LIBERO-LeRobot` | Allen Institute for AI | Apache 2.0 |
| LoRA adapter (this repository) | the author | Apache 2.0 |
| LeRobot | The HuggingFace team | Apache 2.0 |
| LIBERO / LIBERO-plus demonstration data | Lifelong Robot Learning and others | MIT |

## Related

- 40,000-step continuation: [`siruku6/molmoact2_lora_v1_40k`](https://huggingface.co/siruku6/molmoact2_lora_v1_40k)
- Dataset: [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)
