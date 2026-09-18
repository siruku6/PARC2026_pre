# PARC2026 — index of published models and datasets

*日本語版: [README.ja.md](README.ja.md)*

Models and datasets built for the **PARC2026** robot-manipulation competition (LIBERO tasks).
This page is meant to be enough on its own to decide what to use and what to trust.

## Contents

| Section | One line |
|---|---|
| [1. Which one should I use](#1-which-one-should-i-use) | Pick exactly one, by purpose |
| [2. Model lineage](#2-model-lineage) | What descends from what, and the scores |
| [3. Dataset lineage](#3-dataset-lineage) | Where the re-recorded demonstrations live |
| [4. Retention policy](#4-retention-policy) | **What to keep, what may be deleted, and the rule behind it** |
| [5. Traps we fell into](#5-traps-we-fell-into) | So you do not fall into the same ones |

---

## 1. Which one should I use

| Goal | Use |
|---|---|
| **One π0.5 that works on LIBERO** | [`pi05_bowl5_full`](https://huggingface.co/siruku6/pi05_bowl5_full) `005500` — **the best we verified by scoring** (track 1: 0.498) |
| **Try the MolmoAct2 lineage** | [`molmoact2_lora_v1`](https://huggingface.co/siruku6/molmoact2_lora_v1) `024000` — **the only one with measurements** |
| **Cover `libero_90` as well** | [`pi05_l90clean_from_bowl5`](https://huggingface.co/siruku6/pi05_l90clean_from_bowl5) `008000` — read section 4 first |
| **Train from data yourself** | [`libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose) (40 tasks) or [`libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean) (111 tasks) |

**The unmodified `allenai/MolmoAct2-LIBERO-LeRobot` outscored every model we trained**
(track 1: 0.530, track 2: 0.424). If you are picking a starting point, start there.

## 2. Model lineage

```
lerobot/pi05_libero_base  (Gemma Terms of Use)
  ├─ pi05_full_runpod ─ pi05_trial1500 ─ (ladder2500)     track1 0.492 / 0.475
  │                       └─ pi05_t3rev_v2                 reverse-direction; track3 0.000
  └─ pi05_bowl5_full                                       track1 0.498 ← best π0.5
        └─ pi05_l90clean_from_bowl5                        no valid score (section 5)

allenai/MolmoAct2-LIBERO-LeRobot  (Apache 2.0)
  │     unmodified                                         track1 0.530 / track2 0.424 ← best overall
  └─ molmoact2_lora_v1 (024000) ─ molmoact2_lora_v1_40k (040000)
                                   unscored, unevaluated
```

**Scoring runs used 8 episodes.** One episode moving costs about 0.06, so **differences below
0.05 are not readable.** Most differences above are inside that band.

Per-model cards, in Japanese and English, are in the directories below this one.

## 3. Dataset lineage

**The re-recorded demonstrations are the only thing that cannot be rebuilt.** Datasets are
listed by whether they hold any.

| Dataset | Holds re-recorded demos? | Contents |
|---|---|---|
| [`libero_plus_bowl5`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5) | **Yes** — the bowl-task replacements | 14,085 episodes, 40 tasks |
| [`libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose) | **Yes** — 4,275 initial-pose-varied episodes | 16,820 episodes, 40 tasks |
| [`libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean) | **Yes** — 250 clean `libero_90` episodes | 21,642 episodes, 111 tasks |
| [`jaist_mix_v2`](https://huggingface.co/datasets/siruku6/jaist_mix_v2) | **No, it is a mixture** — 397 reverse episodes duplicated 8× and mixed with distributed data | 6,176 episodes |

**Build manifests and scripts are in [`solution/tools/regen/`](../../solution/tools/regen/)**
(which episodes were kept or dropped, contact scores, how initial poses were varied).
**Any mixture can be rebuilt from the components plus these manifests.**

## 4. Retention policy

**Datasets fall into three classes. Whether one may be deleted follows from its class.**

| Class | If lost | Policy |
|---|---|---|
| **Re-downloadable** (LIBERO / LIBERO-plus originals) | download again | **do not store** |
| **Distributed by the organizers** (`libero_combined_20hz`, 21 GB) | **re-downloadable** — not on the Hub, but distributed via Google Drive | **need not be stored**, though it is slow to fetch, so keep it on machines that use it |
| **Mixtures** (determined by components + manifests + scripts) | rebuildable, given the manifests | **keep only those a published model points to** |
| **Re-recorded demonstrations** | **never reproducible identically** (requires re-running the scripted policy) | **always keep, as separate components** |

**Keeping components separately and documenting the merge is the right choice**, for two reasons.

1. **A mixture destroys your ability to attribute.** We know that broadening from 40 tasks to
   111 *hurt* (0.4055 → 0.3515) **only because both existed separately.** With the mixture
   alone, that comparison is gone forever.
2. **Mixtures duplicate storage.** Of `jaist_mix_v2`'s 6,176 episodes, 3,176 are **eight copies
   of the same 397.** Stored as components it is one-eighth the size and strictly more
   informative.

**One condition on keeping a mixture anyway:** the training data of a published model must
remain retrievable. That is why `libero_combined_bowl5_initpose_l90clean` stays —
`molmoact2_lora_v1` and `molmoact2_lora_v1_40k` point at it.

**And one warning.** The step that replaced 250 `libero_90` episodes **exists in neither a
script nor a manifest** (it was done by hand on 2026-09-16). **Deleting that mixture today
means it cannot be rebuilt.** To make it deletable, write out the replaced episode indices and
the demonstrations used to refill them
([manifests, section 3](../../solution/tools/regen/manifests/README.md)).

### Applied to what currently exists

| Dataset | Verdict | Why |
|---|---|---|
| `libero_plus_bowl5` | **keep** | only copy of the bowl re-recordings |
| `libero_plus_bowl5_initpose` | **keep** | only copy of the initial-pose re-recordings |
| `libero_combined_bowl5_initpose_l90clean` | **keep** (and publish) | a published model points at it, and the `libero_90` cleaning is not reproducible |
| `jaist_mix_v2` (6.3 GB) | **replace with its component** | **the component `reverse_lerobot_v2` (397 episodes) was never uploaded.** Upload it, document the mixing recipe (thin `libero_plus` to 75/task, duplicate the reverse set 8×), and the 6.3 GB mixture becomes unnecessary |
| `lerobot_init_pose_all` (1.4 GB) | **may be deleted** | 1,250 initial-pose episodes from 2026-09-12, plain appearance only, **never used for training**; superseded by the 4,275 in `libero_plus_bowl5_initpose` |

## 5. Traps we fell into

**Every one of these cost us points.**

| Trap | What happened |
|---|---|
| **Put another model's tokenizer in π0.5's `vlm/`** | A submission with SmolVLM2 (`GPT2Tokenizer`) in place of Gemma scored **0.075 instead of 0.498**. **The policy runs and silently ignores instructions.** `vlm/` must be 4 files with `tokenizer_class: GemmaTokenizer`. If `merges.txt` is there, it is wrong |
| **A Hub name in a config, fetched at load time** | MolmoAct2's `discrete_action_tokenizer` points at `allenai/MolmoAct2-FAST-Tokenizer`; **in the offline scoring environment the server failed to start — score 0.** Bundle it and point at a local path |
| **Local rehearsals not exercising the real path** | Both of the above passed locally — **because we mounted the HF cache into the container.** Only an offline rehearsal counts |
| **Continued training is not additive** | 8,000 further steps on a 0.498 checkpoint improved `libero_90` and left the target distribution unchanged. **More steps is not more skill** |
| **"No difference locally" is not permission to swap** | Swapping in a candidate that measures the same **removes your upside and keeps your downside** |

Details are in [`docs/knowledge/`](../knowledge/).
