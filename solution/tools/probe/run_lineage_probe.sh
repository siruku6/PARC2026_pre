#!/usr/bin/env bash
# `lineage_probe.py` を学習イメージで回す（lerobot と torch が要る）。
#
#   bash solution/tools/probe/run_lineage_probe.sh <重み> <観測npz> [出力json]
#   OBS2=work/probe_color/base_obs.npz bash ... で観測を 2 つ渡せる
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
W="${1:?重みのディレクトリを渡すこと}"
OBS="${2:?観測の npz を渡すこと}"
OBS2="${OBS2:-}"
JSON="${3:-}"
IMAGE="${IMAGE:-parc2026-train-pi05}"
case "$W" in /*) ;; *) W="$REPO/$W" ;; esac
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -e HF_HOME=/hf_home -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$REPO:/repo" -v "$W:/weights:ro" -v "$HOME/.cache/huggingface:/hf_home" \
    --entrypoint python "$IMAGE" \
    -u /repo/solution/tools/probe/lineage_probe.py /weights \
    "/repo/$OBS" ${OBS2:+"/repo/$OBS2"} \
    --fallback-norm /repo/work/artifacts/pi05_rp3000 \
    ${JSON:+--json-out "/repo/$JSON"}
