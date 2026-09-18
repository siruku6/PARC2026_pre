---
license: gemma
license_link: https://ai.google.dev/gemma/terms
base_model: siruku6/pi05_bowl5_full
base_model_relation: finetune
datasets:
  - siruku6/libero_combined_bowl5_initpose_l90clean
library_name: lerobot
pipeline_tag: robotics
language:
  - en
tags:
  - pi0
  - pi05
  - vla
  - libero
  - lerobot
  - negative-result
---

# π0.5 LIBERO — continued from `bowl5_full`, on the libero_90-cleaned dataset

*日本語版: [README.ja.md](README.ja.md)*

Full-parameter continued fine-tuning of [`siruku6/pi05_bowl5_full`](https://huggingface.co/siruku6/pi05_bowl5_full)
(`005500`) for 8,000 further steps on
[`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean),
a 111-task dataset that adds `libero_90` and the standard LIBERO suites to the 40-task
LIBERO-plus branch its parent was trained on.

Checkpoints: `002000`, `004000`, `006000`, `008000`.

**Read this before using it.** This checkpoint trades away performance on its parent's task
distribution to gain performance on `libero_90`. Whether that is a good trade depends entirely
on what you are evaluating. For us it was not.

## What it is

| | |
|---|---|
| Starting point | `siruku6/pi05_bowl5_full` `005500` (itself a full fine-tune of `lerobot/pi05_libero_base`) |
| Parameters trained | **all 4.14B** (`freeze_vision_encoder: false`, `train_expert_only: false`) |
| Steps | 8,000 (checkpoints every 2,000) |
| Batch | 64 |
| Optimizer | adamw, lr 5e-5, weight_decay 0.01, betas (0.9, 0.95), eps 1e-8, grad_clip_norm 1.0 |
| Scheduler | cosine_decay_with_warmup, warmup 200, peak 5e-5, decay to 2.5e-6 |
| Seed | 42 |
| Image augmentation | **off** |
| Framework | LeRobot v0.6.0 |

## Evaluation

`008000`, measured against its own parent. Perturbed LIBERO rows × 5 episodes, EGL rendering.

| Evaluation set | What it contains | `bowl5_full` (parent) | **`l90clean` `008000`** |
|---|---|---|---|
| 84 rows × 5 (base / sensor noise / initial pose) | none of the 111 tasks' `libero_90` part | **0.798** | 0.793 |
| 29 rows × 5 (background textures, lighting) | the target task distribution | **0.697** | 0.690 |
| 72 rows × 5 (camera viewpoint, initial pose) | the target task distribution | **0.792** | 0.767 |
| 45 `libero_90` tasks | **in this model's training data, not the parent's** | 0.156 | **0.422** |

**The whole gain is on `libero_90`, which this model trained on and its parent did not.**
On every set drawn from the task distribution we cared about, the two are indistinguishable
(differences of 0.005–0.025, against measured run-to-run noise of about 0.05).

### About the competition score

This checkpoint was submitted to the PARC2026 finals and scored **0.075**, against **0.498**
for its parent on the same track with byte-identical inference settings.

**That score is not a measurement of this checkpoint.** The submission archive was built with
the wrong tokenizer in `model_weights/vlm/` — a SmolVLM2 (`GPT2Tokenizer`) directory instead of
the π0.5 Gemma one — so every instruction was tokenized into the wrong token IDs. We reproduced
it afterwards: same weights, same settings, same two tasks, swapping only the tokenizer
directory gave 0.000 (wrong) versus 0.500 (correct).

**So the honest summary is: this checkpoint has no valid competition score.** What the local
evaluation says stands: comparable to its parent on the target distribution, better on
`libero_90`.

## Usage

```python
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.policies import PreTrainedConfig

config = PreTrainedConfig.from_pretrained("path/to/008000")
config.load_vlm_weights = False        # VLM weights are in this checkpoint
config.vlm_model_name = "path/to/008000/vlm"
policy = PI05Policy.from_pretrained("path/to/008000", config=config)
```

`vlm/` holds the **Gemma** tokenizer — 4 files, `tokenizer_class: GemmaTokenizer`. If you
assemble a deployment directory by hand, check that value. A directory containing
`merges.txt` and `vocab.json` is **not** the π0.5 tokenizer, and the policy will run while
silently failing to follow instructions.

## License and provenance

Derived from `lerobot/pi05_libero_base`, which is published under the **Gemma Terms of Use**.

> Gemma is provided under and subject to the Gemma Terms of Use found at
> [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms)

Section 3.2 (Use Restrictions) and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) apply.
LeRobot is Apache 2.0; the LIBERO / LIBERO-plus demonstration data is MIT.

## Related

- Parent: [`siruku6/pi05_bowl5_full`](https://huggingface.co/siruku6/pi05_bowl5_full)
- Dataset: [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)
- Index of all published checkpoints: [PARC2026 model index](https://github.com/siruku6/PARC2026_pre/blob/submission/v1-base-weights/docs/cards/README.md)
