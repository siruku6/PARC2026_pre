# README.md / NOTICE template

The actual per-repo README.md and NOTICE files are not kept in this repo — they live
only on the Hub, at `https://huggingface.co/siruku6/<name>` (or
`https://huggingface.co/datasets/siruku6/<name>`). Read an already-published sibling's
card there before writing a new one:

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("siruku6/<sibling-name>", "README.md")  # add repo_type="dataset" for a dataset
print(open(p).read())
```

Pick the sibling with the closest lineage (same base checkpoint, same kind of data
change) — reuse its frontmatter fields and section order verbatim, adjust the content.
The template below is the schema those cards converge on.

## Language

**English.** The audience is Hub visitors, not this project's Japanese-language
internal docs.

## Model card (`README.md`) frontmatter

```yaml
---
license: gemma
license_link: https://ai.google.dev/gemma/terms
base_model: <the checkpoint this run started from — lerobot/pi05_libero_base, or
  siruku6/<parent-run> if this is a further fine-tune of a previous run>
datasets:
  - <HF dataset repo id(s) actually used to train this checkpoint>
library_name: lerobot
pipeline_tag: robotics
tags:
  - pi0
  - pi05
  - vla
  - libero
  - lerobot
  - negative-result   # include only if the run was not adopted for submission
---
```

Only list a dataset under `datasets:` if it is (or is being made, in this same
publish pass) public — a private dataset referenced from a public model card is a dead
link for every reader but the repo owner.

## Model card body — section order

1. **Title** — one line: architecture + what changed + step count, e.g. "π0.5 LIBERO —
   action-expert fine-tune on the v2 reverse-direction mixture (24,000 steps)".
2. **One-paragraph description** — what this is a fine-tune of, on what data, in one or
   two sentences, each claim on its own sentence.
3. **Negative-result / outcome callout**, if applicable — a `>` blockquote stating
   plainly whether this run was adopted, rejected, or submitted-but-pending, and why in
   one sentence. This project treats a documented negative result as a legitimate
   publish target, not something to soften or omit. State the outcome before the reader
   has to dig for it.
4. **Lineage** — a table: Stage / Weights / Steps / Trainable / Note (or Data), one row
   per stage back to the base checkpoint. If this run is one of several siblings that
   differ by exactly one variable, add a second small table naming that variable and
   the sibling repos.
5. **Contents** — the literal file tree of what `hf upload`'d (check with
   `list_repo_files`, don't assume), plus the `model.safetensors` sha256 (from
   `weights_catalog.md` or `repo_info(..., files_metadata=True)`).
6. **Training setup** — a table: objective/policy, trainable parameter count (exact
   numbers, both the trained subset and the total), batch/LR/schedule, precision, seed,
   image augmentation (or "none"), normalization (and whether it was pinned to a parent
   checkpoint's normalizer or refit), data (repo id + episode/frame/task counts),
   hardware/wall-clock/step time, framework version, final loss.
7. **Evaluation** — every number needs its control/comparison run in the same table or
   the adjacent one; never report a bare success rate with nothing to compare it to.
   Pull numbers from `weights_catalog.md`'s per-run subsection, not from memory.
8. **Use** — the `hf download` + training/inference invocation a reader would actually
   run.
9. **Intended use and limitations** — simulation-only caveat (LIBERO + simulated Franka
   Panda, no real-robot validation), plus whatever specific weakness the evaluation
   section surfaced (doesn't grasp reliably, accuracy cost on tasks sharing an object
   with new training data, etc.).
10. **License** — the fixed Gemma paragraph (below), pointing to `NOTICE`.

## Fixed license paragraph (model cards)

```markdown
## License

These weights are a Model Derivative of Gemma (via PaliGemma inside π0.5) and are released under the
**[Gemma Terms of Use](https://ai.google.dev/gemma/terms)**. Use is also subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). See [NOTICE](NOTICE) for
the third-party attributions that come with the base model, the data and the training code.
```

## NOTICE — structure

```
NOTICE — siruku6/<name>

Gemma is provided under and subject to the Gemma Terms of Use found at
ai.google.dev/gemma/terms

These weights are a Model Derivative of Gemma: they were produced by fine-tuning
<base checkpoint, and its own parent chain if this is a further fine-tune>, a pi0.5
policy whose vision-language backbone is PaliGemma / Gemma. Distribution and use of
these weights are subject to the Gemma Terms of Use and to the Gemma Prohibited Use
Policy (ai.google.dev/gemma/prohibited_use_policy). Anyone who redistributes these
weights, or a further derivative of them, must pass on the Gemma Terms of Use and
this notice.

Third-party components
----------------------

Base model
  <repo id>
  revision <sha>
  License: Gemma Terms of Use (model card metadata: license: gemma)

Training data
  <repo id or description> — <one line per distinct source, with its own license and,
  for anything built on LIBERO, the chain back to LIBERO's MIT license and
  Lifelong-Robot-Learning/LIBERO copyright line>

Simulator            # only if synthetic/scripted episodes are part of the data
  robosuite — https://github.com/ARISE-Initiative/robosuite (MIT)
  MuJoCo — https://github.com/google-deepmind/mujoco (Apache-2.0)

Training code
  LeRobot v<version> — https://github.com/huggingface/lerobot
  License: Apache License 2.0
```

## Dataset card (`README.md`) frontmatter — for a dataset repo instead of a model

```yaml
---
license: mit
task_categories:
  - robotics
tags:
  - lerobot
  - libero
  - pi05
  - imitation-learning
  - synthetic   # if it contains scripted/synthetic episodes
configs:
  - config_name: default
    data_files: data/*/*.parquet
---
```

Body: composition table (part / episodes / frames / tasks / share), how any synthetic
part was generated (be concrete: phases of the scripted rollout, what's genuinely
time-reversed vs. forward-recorded), format table (LeRobot feature dtypes/shapes),
file tree, `Use` (download + `LeRobotDataset(...)` snippet), `Intended use and
limitations`, and a `License and attribution` table crediting every upstream component
back to LIBERO — datasets built from LIBERO carry no Gemma obligation, only MIT.
