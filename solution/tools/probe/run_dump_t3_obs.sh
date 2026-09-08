#!/usr/bin/env bash
# `dump_t3_obs.py` をシムのイメージで回す。
#
#   bash solution/tools/probe/run_dump_t3_obs.sh work/probe_color/t3_obs.npz [init]
#
# **root で走らせて、あとで所有を戻す。**`--user` を付けると LIBERO が
# `/.libero` を作れずに落ちる（`_t3_container.sh` と同じ形にしてある）。
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
OUT="${1:?出力の npz を渡すこと（リポジトリからの相対）}"
INIT="${2:-0}"
CSV="${CSV:-}"          # compe/val の CSV を渡すと T3 ではなくその課題を取る
ONLY="${ONLY:-}"        # CSV のうち task_id がこれに一致するものだけ（空白区切り）
IMAGE="${IMAGE:-parc2026-final}"
mkdir -p "$REPO/$(dirname "$OUT")"
docker run --rm --gpus all \
    -e LIBERO_ROOT=/workspace/LIBERO-plus -e MUJOCO_GL="${MUJOCO_GL:-egl}" \
    -e PYTHONPATH=/workspace/LIBERO-plus:/workspace:/workspace/compe \
    -w /workspace \
    -v "$REPO:/repo" --entrypoint python "$IMAGE" \
    -u "/repo/solution/tools/probe/dump_t3_obs.py" "/repo/$OUT" --init-index "$INIT" \
    ${CSV:+--csv "/repo/$CSV"} ${ONLY:+--only $ONLY} \
    || exit 1
docker run --rm -v "$REPO/$(dirname "$OUT")":/o --entrypoint chown "$IMAGE" \
    -R "$(id -u):$(id -g)" /o >/dev/null 2>&1 || true
