#!/usr/bin/env bash
# 生成系の道具を評価イメージの中で走らせる包み。
# 物理だけを回すので描画は要らないが、EGL を渡しておけば描画も通る。
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${IMAGE:-parc2026-final}"
GL="${GL:-egl}"

# コンテナは root で走るので、書いたファイルは root のものになる。
# **`CHOWN_PATH` を渡すと、走り終えたあとに持ち主を戻す**（`run_final_eval.sh` と同じ手）。
CHOWN_PATH="${CHOWN_PATH:-}"

status=0
docker run --rm --gpus all \
    -e "MUJOCO_GL=${GL}" -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -v "$REPOSITORY_ROOT:/workspace/repo" \
    -w /workspace/repo \
    --entrypoint python \
    "$IMAGE" "$@" || status=$?

if [ -n "$CHOWN_PATH" ]; then
    docker run --rm -v "$CHOWN_PATH:/out" --entrypoint chown "$IMAGE" \
        -R "$(id -u):$(id -g)" /out
fi
exit "$status"
