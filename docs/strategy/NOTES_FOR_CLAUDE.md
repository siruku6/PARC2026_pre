# Notes for Claude — PARC2026_pre repository

Machine-facing notes. Written in English per the project convention (docs for humans → Japanese,
docs for agents → English). Human-facing strategy lives in the numbered files in this directory.

**Purpose.** Record facts about this repository that are non-obvious, that cost real
investigation to establish, or that are easy to get wrong on a later pass. Everything here was
verified against the working tree or an authoritative source on the date noted. Re-verify before
relying on anything marked *unverified*.

Investigation date: **2026-08-08**. Repo state: branch `main`, clean, HEAD `eb8a063`.

---

## 1. What this repository is

Distribution kit for the PARC2026 qualifier (Track 1). A participant trains a VLA policy, wraps it
in an HTTP server, and submits a zip. The kit reproduces the production grading pipeline locally,
minus the private task set and minus the scoring weights.

| Path | Role |
|---|---|
| [pipeline/](../../pipeline/) | Track 1 evaluation pipeline (the grader) |
| [compe/t1/](../../compe/t1/) | Track 1 example task definitions + LIBERO benchmark registration |
| [submission_template/](../../submission_template/) | What the participant edits and submits |
| [evaluate.py](../../evaluate.py) | End-to-end zip evaluation — **Docker-only, see §4.1** |
| [validate_submission.py](../../validate_submission.py) | Pre-submission checker (static + smoke test) |
| [examples/](../../examples/) | Colab notebook: SmolVLA LoRA fine-tune on LIBERO-plus Spatial |
| [setup.sh](../../setup.sh) | Builds the local venv, clones LIBERO-plus + LIBERO, patches, fetches assets |
| [docs/PARC2026開発コンペティション_予選.pdf](../PARC2026開発コンペティション_予選.pdf) | Official participant guide (23 pages, Japanese) |

The PDF classifies as **TEXT-FIRST** (`pua_ratio=0.0`, no rotation, 1-up). Extract it with
`pdf_text.js` from the `pdf-reading` skill; no rendering needed.

---

## 2. Ground truth: where each fact lives

Do not answer questions about grading behavior from the README alone — the README is prose written
over the code, and in two places the code is more specific.

| Question | Authoritative source |
|---|---|
| What obs keys/shapes the server receives | [pipeline/rollout.py](../../pipeline/rollout.py) + [validate_submission.py](../../validate_submission.py) `_make_dummy_obs` |
| How success is decided | `pipeline/rollout.py` `_run_episode` |
| What the scorer computes | [pipeline/scorer.py](../../pipeline/scorer.py) |
| What a submission may contain | `validate_submission.py` constants at the top |
| Grading image contents | [README.md](../../README.md) "採点環境" + appendix pip list |
| Rules, deadlines, prohibitions, report contents | the PDF (not in the code) |

---

## 3. Non-obvious runtime behavior (verified in code)

### 3.1 `seed` is sent but never reaches the policy

`RemotePolicyClient.reset()` in [pipeline/remote_policy.py](../../pipeline/remote_policy.py) posts
`{"instruction": ..., "seed": ...}`, but the `/reset` handler in
[submission_template/policy_server.py](../../submission_template/policy_server.py) forwards only
`instruction`:

```python
instruction = data.get("instruction", "")
_policy.reset(instruction=instruction)     # seed dropped
```

That handler is inside the **"変更不可" (do-not-modify)** region. So a submission cannot use the
episode seed without touching forbidden code. Note also the signature mismatch: `BasePolicy.reset`
takes `(instruction)` while the `PolicyInterface` protocol in `rollout.py` declares
`(instruction, seed)`. Only the pipeline-side implementations (`RandomPolicy`,
`RemotePolicyClient`) take `seed`.

### 3.2 Episode start: 10 zero-action steps before the first observation

`_run_episode` does `env.reset()` → `set_state_from_flattened(init_state)` → `sim.forward()` →
**10 × `env.step(zeros)`**, and the obs from the last of those is what the policy first sees.
Object reference positions for collision detection are captured at that same moment.

### 3.3 Collision detection details

```python
object_init_pos = {
    k[:-4]: np.asarray(obs[k]).copy()
    for k in obs
    if k.endswith("_pos")
    and not k.startswith("robot0")
    and not k.endswith("_to_robot0_eef_pos")
    and k[:-4] not in obj_of_interest
}
...
d = float(np.sum(np.abs(np.asarray(cur) - p0)))     # Manhattan, not Euclidean
...
collided = any(d > collision_threshold for d in object_max_disp.values())
success = bool(done) and not collided
```

- Threshold **0.001 m**, from the builtin default `{"collision": {"enabled": True, "threshold_m": 0.001}}`.
- Distance is the **sum of per-axis absolute displacements**, not the Euclidean norm.
- The **maximum over all steps** is used — restoring a nudged object does not help.
- `obj_of_interest` is parsed from the BDDL `(:obj_of_interest ...)` block by a regex in
  [pipeline/environment.py](../../pipeline/environment.py).

### 3.4 `check_ontop` is monkey-patched to require support contact

[compe/t1/register.py](../../compe/t1/register.py) installs `_install_support_contact_guard()`,
which wraps `SiteObjectState.check_ontop` so the goal also requires the object to be **touching a
static scene geom** (gripper/robot/mount geoms and the object's own geoms are excluded). Holding the
object above the target does not satisfy the goal; it must be released.

This patch is installed by `register_t1()`, which `EnvironmentManager.__init__` calls via
`_ensure_libero_t1_registered()`. It is applied globally to LIBERO once imported.

### 3.5 Scoring configs are absent → `overall_score` is just mean success rate

`total_score_config.json`, `normalization_config.json`, and `scoring_config.json` are **not in the
repo**. `load_scoring_config()` in [pipeline/total_score.py](../../pipeline/total_score.py) falls
back to builtin defaults with `weights: {}`, and `Scorer.score_track` sets
`overall_score = float(overall_success)`.

So the local kit's `overall_score` is **not** the leaderboard score. The real formula (PDF p.11) is
a gate (success ∧ no-collision) multiplied by a weighted sum of five normalized metrics
(time, jerk, SPARC, trajectory, rotation) with `w1+…+w5 = 1`; weights and normalization are secret.
The reference leaderboard score for the distributed training example is **0.0633** (PDF p.8).

### 3.6 Defaults that disagree between CLI and config

| Setting | `pipeline/cli.py` default | `EvalConfig` default | README example |
|---|---|---|---|
| `--max-steps` | **300** | 600 | 600 |
| `--n-episodes` | 20 | 20 | 2 (smoke) |

Always pass `--max-steps 600` explicitly when measuring; otherwise episodes are truncated at 300 and
success rates read low for the wrong reason.

Other env-overridable values: `LIBERO_EVAL_CAMERA` (default 128, sets both height and width),
`LIBERO_CONTROL_FREQ` (default 20), `SERVER_TIMEOUT` (default 120), `EVAL_OUTPUT_DIR`,
`TOTAL_SCORE_CONFIG`, `NORMALIZATION_CONFIG`. Metric `dt` is **hardcoded to 0.05** in
`Scorer._compute_episode_metrics`, consistent with control_freq 20 Hz.

### 3.7 Track 1 injects no noise of its own

`TRACK_PERTURBATIONS[Track.TRACK1]` is a default-constructed `PerturbationConfig` — every field is
`0.0` / `False`. `apply_observation_noise` and `apply_action_noise` are therefore no-ops. All
perturbation comes from the LIBERO-plus task variants themselves (background textures, lighting),
not from the pipeline. Also note `get_perturbed_init_states` logs about applying noise but never
actually perturbs the states — the noise branches only log.

### 3.8 The example task set spans three suites

[compe/t1/T1_TASKS.csv](../../compe/t1/T1_TASKS.csv) has **4 rows**:

| suite | category | difficulty |
|---|---|---|
| libero_spatial | Background Textures | L3 |
| libero_object | Background Textures | L5 |
| libero_object | Light Conditions | L2 |
| libero_goal | Light Conditions | L4 |

The distributed notebook trains on **libero_spatial only**. This mismatch is the single most
actionable finding for strategy. `register.py` strips `_light_*` / `_table_N` suffixes
(`_SUFFIX_RE`) when resolving the `.pruned_init` init-states file, i.e. perturbation variants share
the base task's init states.

**Do not extend `_SUFFIX_RE` to cover more categories — enumeration always misses some.**
Observed suffix forms differ per category: `_table_7`, `_tb_3`, `_light_15`, `_noise_2`,
`_language_25`, `_add_13`, `view_0_0_100_0_0_initstate_1`. Since init-states files exist
**only for base tasks** (`init_files/libero_spatial/` holds 10), resolve by **longest-prefix
match against the filenames that actually exist**. `solution/tools/eval/make_validation_suite.py`
does this and emits the result in the CSV's `init_states_file` column;
`solution/tools/eval/patch_kit_for_variants.py` patches `register.py` to honour that column.

**`task_classification.json` is a dict keyed by suite name**, whose values are lists of
variant records `{"id", "name", "category", "difficulty_level"}`. **Variant records carry no
`suite` key** — the suite exists only as the outer key. Reading it as
`list(raw.values())` yields lists, not records, and silently produces
`{"name": <the whole record>, "category": <suite name>}`, so every category filter matches
nothing. This bug cost one Colab run.

### 3.4 Most variants have NO bddl file of their own — the perturbation is in the name

Verified by counting a local clone of LIBERO-plus: **6,287 of the 10,030 variants have no
`<name>.bddl`.** Camera-viewpoint, robot-initial-state and sensor-noise perturbations are
encoded **as numbers inside the variant name** and parsed at env-construction time by
`libero/libero/envs/env_wrapper.py:207-233`:

```
..._place_it_on_the_plate_view_0_0_100_2_352_initstate_0_noise_3
                          └ horizon, vertical, scale%, endpoint_rot, endpoint_vert
                                              └ init_state (robot suffix)  └ noise
```

`env_wrapper` splits on `_view_` to find the real bddl, so the file only has to exist for
the **base** task. Two consequences, both silent:

- Treating "no `<name>.bddl`" as a skip **wipes out 4 of the 7 categories**. That is what
  produced a 36-task suite (3 categories × 4 suites × 3) instead of 84.
- Rewriting `task_id` to the resolved base name makes every file open **and disables every
  perturbation** — you would measure the unperturbed task and never know. Pass `task_id`
  through untouched; only use the resolved path to check existence and read `:language`.

The split is clean: all 6,287 have both `_view_` and `_initstate_`, and all 3,743 without
`_view_` do have a real bddl. No ambiguous cases.

### 3.4b The distributed harness also opens the variant bddl — and does NOT strip `_view_`

`env_wrapper` strips `_view_` only when **constructing the env**. The harness opens the bddl
a second time, for collision detection, and that path is unstripped:

```python
bddl_path = suite.get_task_bddl_file_path(task_id)   # pipeline/environment.py:73
...
def parse_obj_of_interest(bddl_file):                # pipeline/environment.py:32
    txt = Path(bddl_file).read_text()                # FileNotFoundError for _view_ variants
```

The exception propagates to `Pipeline.run`, which catches it per-track and writes
`_errored_track_score` — so **the run ends with exit code 0 and looks healthy**. On
2026-08-10 this stopped a supposed 84-task run after 3 tasks in 173 s.
`solution/tools/eval/patch_kit_for_variants.py` patches `parse_obj_of_interest` with the same
`_view_` rule.

**Verify before spending an hour on a run:** `solution/tools/eval/check_validation_suite.py`
rebuilds all three paths (env bddl, collision bddl, init states) exactly as the harness
does and names the first row that fails. It **reads `--kit-dir` to decide whether the
patches are applied** rather than taking a flag — a flag would let "I thought I patched it"
pass the very check that exists to catch it. Against an unpatched kit it reports 64 failing
rows; against a patched one, 84/84 open.

### 3.5 Upstream's own init-states resolution is buggy; use longest-prefix match

`Benchmark.get_task_init_states` (LIBERO-plus) does `name.split("_table_")[0]`, which
truncates `pick_up_the_black_bowl_from_table_center_...` to
`pick_up_the_black_bowl_from` — a task that does not exist. 1,621 variants fail this way.
Since init-states files exist **only for base tasks** (10 per suite), resolve by
**longest-prefix match against the filenames that actually exist**, requiring the next
character to be a separator. Verified: 0 of 10,030 unresolved.
`solution/tools/eval/make_validation_suite.py` does this and writes the answer to the CSV's
`init_states_file` column; the H cell patches `register.py` to honour that column.

---

## 4. Traps when running things in this repo

### 4.1 `evaluate.py` only works inside the Docker image

Two independent reasons:

1. Absolute paths `/workspace/LIBERO-plus`, `/workspace`, `/workspace/compe` are module constants,
   and `run_pipeline` passes `cwd="/workspace"`. Outside the container this fails.
2. It does `import sandbox as _sb`, and **`sandbox.py` is not in this repository**. `_sb` is `None`,
   so the venv isolation path is skipped and `install_requirements()` runs
   `pip install -r requirements.txt` **into the current interpreter's environment**.

Consequence: running `python evaluate.py sub.zip` on the host installs the submission's dependencies
into the kit's own venv and can break the evaluation pipeline (e.g. a submission requiring numpy 2.x
would overwrite the pinned 1.26.4). Inside a disposable container this is harmless.

Safe local path: start the policy server separately and run
`python -m pipeline --server-url http://localhost:8000`.

### 4.2 Production venv semantics differ from local

In production, the submission gets a venv created **with `--system-site-packages`**, so anything not
listed in `requirements.txt` resolves to the grading image's version, and anything listed shadows it
for the submission process only. Locally (per §4.1) there is no such isolation. Never reason about
dependency conflicts from local behavior.

### 4.3 The distributed Docker image is CPU-only

`FROM ubuntu:22.04`, `torch 2.11.0+cpu`, `MUJOCO_GL=osmesa`. Production is
`nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04`, `torch 2.11.0+cu130`, `MUJOCO_GL=EGL`.
CUDA-related failures are invisible locally. The README documents the two-line patch to build a
production-equivalent image.

Rendering backend differs (osmesa vs EGL), so rendered pixels are not guaranteed bit-identical
between the local kit and production.

### 4.4 `setup.sh` mutates the user's home directory

It overwrites `~/.libero/config.yaml` (previous file saved to `.bak`). It also wires
`assets` to the **upstream LIBERO** clone while `bddl_files` / `init_states` / `benchmark_root`
point at **LIBERO-plus**. It patches LIBERO-plus in place (`touch __init__.py`, `sed` for
`torch.load(..., weights_only=False)`).

Requires `python3.10` on PATH. Prefer Docker for anything behavioral (also the user's standing
preference).

### 4.5 `requirements.txt` restrictions are enforced, not advisory

`validate_submission.py` rejects, as ERROR: `-i` / `--index-url` / `--extra-index-url` / `-f` /
`--find-links` / `-e` / `--editable` / `-r` / `-c`; any URL scheme among
`file, http, https, ftp, ftps, ssh, git, hg, svn, bzr`; direct references; and local paths.
Allowed options are only `--hash`, `--require-hashes`, `--no-index`, `--pre`, `--no-binary`,
`--only-binary`, `--prefer-binary`. The same check runs in production right after unzip.

Practical effect: **you cannot pin a wheel from the PyTorch CUDA index**. Everything must resolve
from default PyPI, or already be present in the image.

### 4.6 Whether `av` is required depends on whether `datasets` is installed

`policy_server.py` does `from lerobot.policies.factory import make_pre_post_processors`.
That import chain reaches `lerobot/policies/groot/processor_groot.py`, which contains:

```python
if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None
```

`lerobot/datasets/__init__.py` calls `require_package("datasets")` **and**
`require_package("av")` at module level. So:

| `datasets` present? | Branch | `av` needed? |
|---|---|---|
| No | skipped | **No** |
| Yes | taken → imports `lerobot.datasets` | **Yes** |

Our `requirements.txt` declares neither. This works because the submission venv has no
`datasets` — verified in the container, and confirmed by the production run scoring 0.05832.

**Latent fragility.** The production venv is created with `--system-site-packages`. If the
grading image ever gains `datasets` without `av`, server startup fails and the track scores 0.
Adding `av` to `requirements.txt` would remove the risk at the cost of changing an artifact
that is already known-good. Not done as of 2026-08-09; revisit if a startup failure appears.

Colab hits this immediately because Colab ships `datasets` — see
[docs/solution/colab_eval.md](../solution/colab_eval.md) ("依存を2プロセスに分けるのと、
av が要る理由"). `colab_eval.setup_policy()` pins `av>=15.0.0,<16.0.0` for this reason.

---

## 5. Dependency facts (verified 2026-08-08)

The central problem participants hit: the training notebook and the grading image require different
Python versions.

| | Training (distributed notebook) | Grading (submission runtime) |
|---|---|---|
| Python | ≥ 3.12 — notebook cell 2 raises `RuntimeError("Python 3.12以上が必要です。")` | 3.10.12, system `/usr/bin/python3.10` |
| LeRobot | v0.6.0 | must be provided by the submission |
| torch | Colab default | 2.11.0+cu130 |
| numpy | LeRobot 0.6.0 wants ≥2.0.0,<2.3.0 | 1.26.4 |
| CUDA | Colab-dependent | 13.0; only `libnvJitLink.so.13` exists |

LeRobot PyPI metadata:

| version | requires-python | torch constraint |
|---|---|---|
| 0.1.0 – 0.4.4 | `>=3.10` | 0.4.4: `>=2.2.1,<2.11.0` |
| 0.5.0 – 0.6.1 | `>=3.12` | 0.6.0: `>=2.7,<2.12.0` |

`transformers` 5.x has `requires_python >= 3.10.0`, so it is **not** a blocker on 3.10. The only
hard Python gate is LeRobot's own metadata — and it is metadata, i.e. a `pip install` gate, not
necessarily a source-level incompatibility. Vendoring the source into the submission bypasses it.

Key asymmetry driving the recommendation in [02_環境戦略.md](02_環境戦略.md):
LeRobot **0.6.0**'s `torch>=2.7,<2.12.0` is satisfied by the image's 2.11.0+cu130, whereas
**0.4.4**'s `torch<2.11.0` conflicts with it and would drag in a CUDA 12 torch plus the whole
`nvidia-*-cu12` wheel set.

`torchvision` is **not** in the grading image's pip list, and cannot be installed from the PyTorch
CUDA index (§4.5). Avoid import paths that require it.

Notebook pins worth keeping (they are the reproducibility anchors):

| item | value |
|---|---|
| base weights | `lerobot/smolvla_libero_plus` rev `7bb70aa5bc92b82c9239142775d3a173103567ff` |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| dataset | `lerobot/libero_plus` rev `f3f49f426d75030177b18778374005bc12ccd588` |
| LeRobot | tag `v0.6.0` |
| LIBERO-plus (notebook) | commit `4976dc3`; `robosuite==1.4.1`, `bddl==1.0.1`, `gym==0.26.2` |
| LIBERO-plus (kit) | `setup.sh` clones `--depth 1` HEAD; `robosuite==1.4.0`, `bddl==3.6.0`, `gym==0.25.2` |

Note the last two rows: **the notebook and the kit pin different versions of robosuite, bddl, and
gym, and different LIBERO-plus commits.** The `examples/README.md` acknowledges this and says to do
evaluation on the kit side.

---

## 6. Unverified — confirm before relying on it

| Claim | How to settle it |
|---|---|
| LeRobot v0.6.0 source is Python 3.10 syntax-compatible | `python3.10 -m compileall -q <vendored src>` |
| SmolVLA's inference path does not require `torchvision` | import it in a venv without torchvision |
| A v0.6.0-saved checkpoint (with `policy_preprocessor*.json`) loads in v0.4.4 | try it; the processor pipeline API changed |
| torch 2.11.0+cu130 works with numpy 2.x | run in the GPU container |
| Training data resolution / camera feature names / state layout | `LeRobotDatasetMetadata(...).features` |
| Gripper sign convention matches the dataset | inspect `md.stats["action"]`, then one local pipeline run |
| Production `max_steps` and `n_eval_episodes` | not public; kit defaults are 600 / 20 |

---

## 7. Working-environment notes for future sessions

- The Bash sandbox in this session allows network only to `api.github.com`, `aws.amazon.com`,
  `docs.anthropic.com`, `www.anthropic.com`, `claude.com`. `curl` to raw.githubusercontent.com and
  pypi.org is **denied**. The `WebFetch` tool is not subject to that allowlist and was used for the
  dependency facts in §5.
- `pip` / `docker` were not exercised in this session. Nothing in §5 was confirmed by an actual
  install.
- The PDF's page numbers cited throughout the strategy docs refer to PDF pages (1-indexed), which
  match slide order 1:1 (no N-up).

---

## 8. Session handoff — 2026-08-11 (deadline 8/14)

### 8.1 Repository state

Branch **`submission/v1-base-weights`** at `b234fea`. Working tree clean.

| Fact | Detail |
|---|---|
| **Push to `github` is PENDING** | `github/submission/v1-base-weights` is at `2981978`. Fast-forward, **no force needed** |
| Today's 9 commits were squashed to 5 | Contiguous groups only, no reordering. Tree hash verified identical at each step |
| Backup tag | `backup/before-tidy-20260811` → `27de31b` (pre-squash tip) |
| `submission/v2-tuned-weights` | **Fully merged in** (`87e0dfc`, two parents). Obsolete as a working branch |
| `github/gpu/v2-snapshot` | The GPU server's rebased v2 (3 commits). **Fully absorbed** — `28b9eb6` cherry-picked as `b234fea`; the other two were content-identical to `f2eae39`/`e4edc39`. Safe to delete on the remote |
| `docs/reference/03_chunk_params.md` | Renamed to **`07_`** (collided with `03_report_values.md`) |

**The GPU server must `git pull` before any training run.** Its `train.py` defaults were the old
ones (batch 8 / lr 3e-4 / 200 episodes) that produced the 8.8× regression.

### 8.2 In flight on the GPU server

- **lora20k 84-task eval**, started 2026-08-11 05:23 UTC, ~2.7 h. Weights
  `work/artifacts/model_weights_lora20k` (`sha256 2cd9faab…`, identical to the 4th submission's
  checkpoint). `verify_merge.py` confirms 37 changed tensors.
- **Purpose: validate the local suite as a proxy.** Production separated base (0.1409) from
  lora20k (0.016) by 8.8×. If the 84-task suite cannot separate them, no training result can be
  judged locally. This gates everything downstream.
- **The base 84-task result lives on the GPU server** at
  `outputs/0811_101824_base_n5_ep1_max300` — **not yet transferred here.** Needed for item 1.

### 8.3 Priority order agreed with the user

`0` push + GPU sync → `5a` lora20k eval (running) → `1` analyse the base 84-task run →
`6` (answered) → `5b` training run A (base recipe, all data, no augmentation) →
`2` design training run B → `3` (answered) → `4` model write-up, folded into the report (item 16).

Full item list with rationale: [../talk/open_items.md](../talk/open_items.md).

### 8.4 Corrections made today — do not re-derive these

| Claim that was wrong | What is actually true |
|---|---|
| "Unfreezing the vision encoder is the biggest lever" | **No-op under LoRA.** `_get_default_peft_targets` in `policies/smolvla/modeling_smolvla.py` targets only `lm_expert` q/v projections + 5 projection layers (32 + 5 = the 37 tensors `verify_merge` counts). `freeze_vision_encoder` / `train_expert_only` change no trainable weight |
| "Track 1 evaluates exactly 2 perturbation categories" | Organizers said only **"not all 7"** plus the criterion text. They named no category. `Sensor Noise` (1,601, the largest) plausibly qualifies. See [../reference/05_perturbation_categories.md](../reference/05_perturbation_categories.md) |
| "EGL can be fixed on this machine" | **vGPU product is `NVIDIA Virtual Compute Server`** — compute-only, no OpenGL. Not fixable from our side. The "sphere renders in 0.2 ms" observation was not evidence of GPU rendering |
| "EGL would save about an hour" | Measured EGL is **38.6 ms/step** vs osmesa 384 ms. Would have been 2.7 h → 40 min. Moot now |
| "Filtering the training data to visual-perturbation categories gives 4.5× density" | **Impossible and not worth it.** No variant name in any metadata column; ordering is fully mixed (window brightness spread 0.028); only 40 instruction strings so `Language Instructions` variants are absent. And appearance-shifted episodes are already **53.5%**, so filtering buys 1.9× while halving the data. See [../reference/06_training_data_composition.md](../reference/06_training_data_composition.md) |
| "The summariser does not report jerk" | It always did (`mean_avg_cartesian_jerk` etc.). A parallel session added `--compare` instead |

### 8.5 Weak evidence to shore up

**The eval-side appearance spread (`light` 0.287 / `table`,`tb` 0.405, range 0.212–0.689) comes
from a third party's EDA over 15 initial frames, not from our own measurement.** The conclusion
"the augmentation range ×0.8–1.2 does not cover the eval side (×0.71–1.70)" rests on it.

It is now measurable in-house: the finished 84-task runs contain the initial frames. Doing so
requires no policy inference.

### 8.6 Sandbox quirks hit this session

- `mktemp -d` → `Operation not permitted`. Use `mkdir -p "$TMPDIR/<name>"`.
- No `pandas` / `pyarrow` in any host interpreter, and `work/venv310` has no `pip`.
  Workaround used: `python3 -m venv "$TMPDIR/pq" && "$TMPDIR/pq/bin/python" -m pip install pyarrow`
  (pypi.org is reachable). The `local-uv_env:latest` Docker image has pandas 2.3.3 + pyarrow 25,
  but **`docker run` was denied by the permission layer** — prefer the venv.
- `git branch -D` failed with `could not lock config file .git/config`; the branch deletion itself
  still took effect.
- `diff <(...)` process substitution is denied; write to `$TMPDIR` and use `git diff --no-index`.
- **Working directory persists between Bash calls.** A `cd docs` in one call broke later relative
  paths. Use absolute paths or re-`cd` to the repo root.
- Parallel Claude sessions edit the same files. `open_items.md` changed under us twice, and the
  checked-out branch changed once. Re-read before editing; verify `git branch --show-current`
  before any git operation.
