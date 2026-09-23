# Rebuilding from components

*日本語版: [README.ja.md](README.ja.md)*

**Everything needed to rebuild the merged dataset, so that the merged copy can be deleted.**

**This bundle is self-contained.** No external repository is required.

| File | What it is |
|---|---|
| `solution_<commit>.tar.gz` | **The source tree used to build it** — scripts, Dockerfile, manifests |
| `manifests/` | **Which episodes were dropped and what refilled them** (a browsable copy; the same files are inside the tarball) |
| `SOURCE_COMMIT.txt` | The revision this came from |

**On 2026-09-18 this bundle alone was used to rebuild the dataset on a different machine
(an Apple Silicon Mac). All 21,642 action sequences matched the published copy.**

```
行動列が一致した本: 21642 / 21642      (matching episodes: 21642 / 21642)
```

## Contents

- [1. What this produces](#1-what-this-produces)
- [2. Materials](#2-materials)
- [3. Setting up the environment](#3-setting-up-the-environment)
- [4. Building](#4-building)
- [5. Verifying](#5-verifying)
- [6. Known snags](#6-known-snags)

## 1. What this produces

**`libero_combined_bowl5_initpose_l90clean`** — 21,642 episodes, 3,431,137 frames,
111 tasks, 27.8 GB.

```
libero_plus_bowl5_initpose (HF, 16,820 eps)  ┐
libero_combined_20hz (organizer, 19,533 eps) ┤→ libero_combined_bowl5_initpose (parent, 21,642)
                                              │
libero_90_clean (HF, 3,911 eps) ─────────────┴→ l90clean (250 replaced, 21,642)
```

**Two halves.** The first builds the parent; the second swaps out 250 dirty `libero_90` episodes.

## 2. Materials

| Needed | Where from |
|---|---|
| `siruku6/libero_plus_bowl5_initpose` (22.6 GB) | Hugging Face |
| `siruku6/libero_90_clean` (3.6 GB) | Hugging Face |
| **`libero_combined_20hz` (21 GB, LeRobot v3.0)** | **Distributed by the PARC2026 organizers** (via the Omnicampus environment, or the Google Drive they announced) |

```bash
hf download siruku6/libero_plus_bowl5_initpose --repo-type dataset \
    --local-dir work/artifacts/libero_plus_bowl5_initpose
hf download siruku6/libero_90_clean --repo-type dataset \
    --local-dir work/artifacts/libero_90_clean
```

**It must be v3.0.** `libero_combined_20hz_v2` is v2.1 and will not work. **There is a reason:**
the build avoids re-encoding video only because **v3.0 packs dozens of episodes into one mp4 and
reads them back by start timestamp.** v2.1 breaks that assumption.

```bash
python3 -c "
import json; i=json.load(open('<extracted>/libero_combined_20hz/meta/info.json'))
print(i['codebase_version'], i['total_episodes'], i['total_tasks'])"
# expect: v3.0 19533 111
```

**Budget 120 GB of disk.** Videos are placed as hard links, so building costs nothing extra —
except the merge step, which copies them.

## 3. Setting up the environment

**Extract the source and work inside it.**

```bash
tar xzf solution_<commit>.tar.gz          # yields solution_src/solution/
cd solution_src
```

**Everything heavy runs inside Docker.** Your host python3 needs nothing installed — it is only
used to read numbers out of `meta/info.json` with the standard library.

```bash
docker build -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train
```

**On arm64 (Apple Silicon), add `--build-arg WITH_QUANT=0`.** The CUDA wheel index has no
aarch64 build of `torchao`, and **`torchao` and `bitsandbytes` are for training, not for
building datasets.**

```bash
docker build --build-arg WITH_QUANT=0 \
    -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train
```

**Check it works.**

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    -c "import lerobot; print(lerobot.__version__)"
# 0.6.0 means you are set
```

**The container sees the tree root as `/workspace/repo`**, so **outputs must stay inside it.**

**Put the manifests where the scripts look** — under `work/regen/`.

```bash
mkdir -p work/regen/combined
cp solution/tools/regen/manifests/head_keep.json       work/regen/combined/head_keep.json
cp solution/tools/regen/manifests/keep_replace_v2.json work/regen/keep_replace_v2.json
cp solution/tools/regen/manifests/grader_texts.txt     work/regen/grader_texts.txt
```

## 4. Building

### First half — the parent

```bash
COMBINED=<extracted>/libero_combined_20hz \
STAGE=c bash solution/tools/regen/build_initpose_datasets.sh
```

**Skip `STAGE=p`** — its output, `libero_plus_bowl5_initpose`, was downloaded from Hugging Face.

**What it does.**

1. Take the **4,822** episodes that `head_keep.json` keeps, out of the first 5,186 distributed ones
2. **Rewrite their gripper values from `{0,1}` to `±1`** — otherwise "open" is labelled backwards
3. Append them to `libero_plus_bowl5_initpose` (16,820) → **21,642 eps / 3,436,940 frames / 111 tasks**
4. Run four gates: counts, gripper values, normalization scale, and **whether the appended
   head still points at the same frames as the source** (pixels and state)

### Second half — replacing 250 `libero_90` episodes

```bash
IMAGE=parc2026-train-pi05 \
PARENT=work/artifacts/libero_combined_bowl5_initpose \
POOL=work/artifacts/libero_90_clean \
bash solution/tools/regen/rebuild_l90clean.sh
```

**It reads only `manifests/l90clean_replace.json`** — the 250 dropped indices and, for each
appended episode, its `source_episode_index`.

1. Derive keep-files from the manifest (keep 21,392 / append 250)
2. Drop the 250 from the parent
3. Extract the 250 from `libero_90_clean`, in manifest order
4. Append → **21,642 eps / 3,431,137 frames / 111 tasks**
5. Compare against the manifest

**The appended 250 land at the end**, so their positions differ from the ones removed.
To map between the two, read `l90clean_replace.json`.

**The source pool is `libero_90_clean`, not `libero_90_clean_appearance`** — the latter explains
only 110 of the 250 by `(task, length)`. **Provenance was established by sha256 of the action
sequences**, not by name or length.

## 5. Verifying

**Step 5 checks the counts.**

```
merged_episodes: 21642 (manifest 21642) OK
merged_frames: 3431137 (manifest 3431137) OK
merged_tasks: 111 (manifest 111) OK
```

**If counts are not enough for you, compare the contents.**

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/regen/compare_action_signatures.py \
    --a <published copy> --b work/artifacts/libero_combined_bowl5_initpose_l90clean_rebuilt
```

```
  A 21642 本 / B 21642 本
行動列が一致した本: 21642 / 21642
```

## 6. Known snags

**Every one of these was hit for real on 2026-09-18.**

| Symptom | What it is |
|---|---|
| **It stops mid-output, saying nothing** | **Docker ran out of memory.** Docker Desktop's default can be small (1.9 GB on one Mac). Check `docker info --format '{{.MemTotal}}'` and give it **8 GB or more** |
| `could not select device driver "" with capabilities: [[gpu]]` | No GPU on this machine. **Building needs no GPU** — prefix the command with `GPUS=` to drop the flag |
| `No matching distribution found for torchao==0.17.0` | Building on arm64. Use `--build-arg WITH_QUANT=0` |
| **Disk drops sharply during the merge** | Videos were copied instead of hard-linked. Check that source and output are **on the same filesystem**, and that it supports hard links (exFAT/FAT do not) |

**Measured** (Apple Silicon, Docker with 8 GB).

| Step | Time |
|---|---|
| First half, 1–2 (extract, gripper fix) | ~1 min |
| **First half, 3 (merging 21,642 episodes)** | **~4 min** |
| First half, gates 1–4 | a few min |
| **Second half, 4 (merge)** | **~2 min 40 s** |

## License

Derived from LIBERO and LIBERO-plus, distributed under the **MIT License**
(Copyright (c) 2023 Lifelong Robot Learning). The bundled source is available on the same terms.
