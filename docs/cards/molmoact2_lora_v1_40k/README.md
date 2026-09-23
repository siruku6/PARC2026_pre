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

# MolmoAct2-LIBERO LoRA (40,000 steps)

*日本語版: [README.ja.md](README.ja.md)*

A LoRA adapter trained on top of [`allenai/MolmoAct2-LIBERO-LeRobot`](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot)
for the **LIBERO** manipulation benchmark, using a demonstration set rebuilt to reduce
two specific failure modes: contact with non-target objects, and a narrow distribution of
robot initial poses.

Trained for the **PARC2026** robot-manipulation competition (finals round 1).

**This checkpoint has no benchmark score.** See [Evaluation](#evaluation) — the numbers below come
from its 24,000-step sibling, and the 40,000-step checkpoint was never scored before the
competition deadline. Treat it as a starting point, not as a validated result.

## What this is

| | |
|---|---|
| Base model | `allenai/MolmoAct2-LIBERO-LeRobot` (MolmoAct2 5B, Apache 2.0) |
| Adaptation | LoRA on the VLM only (`enable_lora_vlm: true`, action expert untouched) |
| Rank / alpha / dropout | 64 / 16 / 0.05 |
| Steps | **40,000** |
| Dataset | [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean) (21,642 episodes, 111 tasks) |
| Framework | LeRobot v0.6.0 |

**Weights are saved unmerged, with PEFT parameter names** (`base_model.model.*`, `.base_layer.*`).
Loading requires `peft` and `enable_lora_vlm: true` in the policy config.

## Training

```
steps          40000
batch_size     4
optimizer      adamw, lr 1e-5, weight_decay 0.0, betas (0.9, 0.95), eps 1e-6
               grad_clip_norm 1.0
scheduler      cosine_decay_with_warmup
               warmup 200, decay 40000, peak_lr 1e-5, decay_lr 1e-6
seed           1000
chunk_size     10      n_action_steps 10
```

Image augmentation was **on** (`image_transforms.enable: true`, up to 3 of: brightness
0.8–1.2, contrast 0.8–1.2, saturation 0.5–1.5, hue ±0.05, sharpness 0.5–1.5,
affine ±5° with ±5% translation). This matches the augmentation the base model itself was
trained with.

Checkpoints were saved every 8,000 steps. This repository holds `040000/`.

## Evaluation

**No score exists for this checkpoint.** What follows is its 24,000-step sibling, measured on
84 perturbed LIBERO task rows × 5 episodes = 420 episodes (EGL rendering, `samples=1`,
no adaptive replanning). "Collision rate" excludes scene artifacts that no policy can avoid.

| Model | Success | Goal reached | Collision | Smoothness |
|---|---|---|---|---|
| π0.5 baseline (full fine-tune, separate lineage) | **0.798** | **0.881** | 0.138 | **0.503** |
| **This lineage @ 24,000 steps** | 0.774 | 0.833 | 0.138 | 0.494 |
| This lineage @ 16,000 steps (no image augmentation, batch 2) | 0.695 | 0.752 | 0.150 | 0.482 |

Read this carefully:

- **The 16,000 → 24,000 gain is real but confounded.** Three things changed at once:
  image augmentation off → on, batch 2 → 4, and 16k → 24k steps. It is not a single-variable
  comparison, so "augmentation helped" is not a supported claim.
- **24,000 steps does not beat the π0.5 baseline** on this evaluation set. The gap (−0.024)
  is inside the run-to-run noise we measured (~0.05), so the honest reading is "comparable".
- **The evaluation set contains no camera-viewpoint, background, or lighting perturbations.**
  It is 12 base rows, 60 sensor-noise rows, and 12 robot-initial-pose rows. Conclusions about
  viewpoint or appearance robustness cannot be drawn from it.
- **The unmodified base model outscored this lineage** on the competition's own scoring runs
  (0.530 and 0.424 on two tracks). Whether 40,000 steps of LoRA improves on the base model is
  **unknown and untested**.

## Usage

```python
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.configs.policies import PreTrainedConfig

config = PreTrainedConfig.from_pretrained("path/to/checkpoint")
config.checkpoint_path = "path/to/checkpoint/molmoact2_base_small"
policy = MolmoAct2Policy.from_pretrained("path/to/checkpoint", config=config, strict=True)
```

`peft` must be installed. `molmoact2_base_small/` (config, tokenizer, normalization stats) is
included in this repository and is required.

### Running without network access

**The policy config names a Hub repository for the discrete action tokenizer**
(`discrete_action_tokenizer: "allenai/MolmoAct2-FAST-Tokenizer"`). At load time LeRobot
resolves that string as a local path first, and falls back to `snapshot_download` if no such
path exists. **In an offline environment the fallback fails and the policy will not start.**

**A copy of that tokenizer is bundled here as `fast_action_tokenizer/`**, byte-identical to
the upstream one. Point the config at it and no network is needed:

```python
config.discrete_action_tokenizer = "path/to/checkpoint/fast_action_tokenizer"
```

We lost a competition submission to exactly this (the server could not start; score 0), which
is why it ships with the weights. **Any config that still names a Hub repository will fail the
same way** — check every string in `policy_preprocessor.json` before relying on an offline run.

## License and provenance

| Component | Holder | License |
|---|---|---|
| Base weights `allenai/MolmoAct2-LIBERO-LeRobot` | Allen Institute for AI | Apache 2.0 |
| LoRA adapter (this repository) | the author | Apache 2.0 |
| LeRobot | The HuggingFace team | Apache 2.0 |
| LIBERO / LIBERO-plus demonstration data | Lifelong Robot Learning and others | MIT |

Training data is not included here; only the resulting weights.

## Related

- Dataset: [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)
- Base model: [`allenai/MolmoAct2-LIBERO-LeRobot`](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot)
