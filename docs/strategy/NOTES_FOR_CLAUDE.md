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
