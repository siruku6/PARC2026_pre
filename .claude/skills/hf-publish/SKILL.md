---
name: hf-publish
description: |
  Write a model/dataset card and NOTICE for a Hugging Face repo (siruku6/*) from this
  project's own primary sources, upload it, and flip the repo to public. Use when the
  user asks to publish, open up, or make public an HF repo that currently holds
  training output from this project (pi05_*, jaist_mix_*, etc.), or when freeing up
  the private-storage quota is the actual goal.
---

# Publishing an HF repo (model or dataset) from this project

## Why this comes up

Private repos count against the account's 100 GB Hugging Face storage quota; public
ones don't. Moving a private repo to public — without deleting anything — frees its
full size from the quota immediately. Before training the next run, check whether the
quota is actually tight enough to matter:

```python
from huggingface_hub import HfApi
api = HfApi()
total = 0
for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets)):
    for r in lister(author="siruku6"):
        i = api.repo_info(r.id, repo_type=kind, files_metadata=True)
        if i.private:
            total += sum((f.lfs.size if f.lfs else (f.size or 0)) for f in i.siblings)
print(f"private total: {total/1e9:.2f} GB / 100 GB")
```

## Primary sources — never invent numbers

This project keeps its actual facts in these places, in order of authority:

| Fact | Where |
|---|---|
| What was trained, on what data, for how long, and what happened to it (adopted / rejected / submitted) | `docs/reference/19_weights_catalog.md` |
| Exact hyperparameters, base checkpoint revision, batch/lr/steps | `train_config.json` next to each checkpoint under `work/train/<run>/checkpoints/<step>/pretrained_model/` |
| Evaluation numbers and the reasoning behind adopt/reject calls | `docs/reference/19_weights_catalog.md` (the per-run subsections), `docs/knowledge/*.md` |
| Dataset composition and provenance | `docs/reference/25_dataset_inventory.md` and any existing dataset's own card already on the Hub |

Grep `weights_catalog.md` for the run name first — it already states step count, batch,
LR, hardware, wall-clock time, final loss, which checkpoints exist, which one (if any)
was uploaded to HF, its sha256, and its 1-variable comparison against sibling runs. Do
not restate a number from memory or from a chat summary; open the file.

## Before publishing anything: two checks, and one you cannot make yourself

| Check | Command | Stop if |
|---|---|---|
| **Leaked tokens in the training log** | `hf download siruku6/<name> --include 'logs/*' --local-dir /tmp/ck && grep -rn 'hf_[A-Za-z0-9]\{20,\}' /tmp/ck` | Any match. Revoke that token before doing anything else — don't just scrub the log. |
| **Machine paths in `train_config.json`** | (informational only) | Nothing — these are local filesystem paths like `/opt/dlami/nvme/...`, not secrets. Not a reason to hold back. |

## Drafting the card

The README.md/NOTICE content, structure, and frontmatter schema are documented in
[references/output-format.md](references/output-format.md) — read it before writing.
It also explains how to pull an already-published sibling card down from the Hub to use
as a concrete template (this repo does not keep local copies of published cards; the
Hub is the only copy once a card is uploaded).

Write the draft to a scratch path, not into the project tree — nothing under
`solution/` should hold a copy of a card once it's uploaded (see below).

## Before touching visibility: check what the card links to

A model card's `datasets:` frontmatter, and any inline links to sibling repos, must
resolve for a reader who is not authenticated as the repo owner. Before flipping
anything to public, check the current visibility of everything the new card
references — including repos that an *already-published* sibling card links to; those
links can go stale too:

```python
api.repo_info("siruku6/<name>", repo_type="model").private   # or "dataset"
```

If a referenced repo is still private, surface this to the user with
`AskUserQuestion` before proceeding. Don't silently publish a card with a dead link, and
don't silently expand scope to publish something the user didn't ask about either.

## Uploading

```sh
hf upload siruku6/<name> <scratch-path>/README.md README.md
hf upload siruku6/<name> <scratch-path>/NOTICE NOTICE
```

Add `--repo-type dataset` for a dataset repo.

## Flipping visibility

The web UI (`Settings` → `Change visibility` → confirm by typing the repo name) is the
most reliable path. The API equivalent is one line, but note two failure modes:

```python
api.update_repo_settings(repo_id="siruku6/<name>", private=False)                # model
api.update_repo_settings(repo_id="siruku6/<name>", private=False, repo_type="dataset")
```

- A `403` means the token lacks permission to change repo settings — switch to the web
  UI, don't try to work around it.
- This call has also been observed to return `None` with **no error** while leaving a
  *dataset* repo's `private` flag unchanged. **Always verify** with `repo_info(...)`
  afterward; if it's still `True`, fall back to a direct HTTP PUT, which has worked
  when the SDK call silently no-oped:

```python
import requests
from huggingface_hub import get_token
token = get_token()
requests.put(
    f"https://huggingface.co/api/datasets/siruku6/<name>/settings",  # or /api/models/
    json={"private": False},
    headers={"Authorization": f"Bearer {token}"},
)
```

## Final check

List files and re-confirm visibility on every repo touched (the one asked for, plus any
newly-public dependency) before reporting done:

```python
for repo, rtype in [("siruku6/<name>", "model"), ...]:
    info = api.repo_info(repo, repo_type=rtype)
    print(repo, "private=", info.private)
    print("README present:", "README.md" in api.list_repo_files(repo, repo_type=rtype))
```

If the goal was freeing quota, re-run the quota count from the top of this file and
report the before/after numbers.

## Cleanup

Do not leave a copy of the uploaded README.md/NOTICE anywhere in this repo — the Hub is
the single source of truth for a published card, and a local copy just goes stale. Once
`hf upload` succeeds and visibility is confirmed public, delete the scratch draft.
