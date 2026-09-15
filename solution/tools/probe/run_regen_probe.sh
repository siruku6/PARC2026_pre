#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-parc2026-final}"
GL="${GL:-egl}"
SCRIPT="${SCRIPT:-perturbed_replay_yield.py}"
docker run --rm --gpus all \
    -e "MUJOCO_GL=${GL}" -e PYTHONUNBUFFERED=1 \
    -v "$PWD:/probe" \
    -v "$HOME/src/PARC2026_pre:/repo:ro" \
    --entrypoint python \
    "$IMAGE" "/probe/${SCRIPT}"
