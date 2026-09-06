#!/usr/bin/env bash
# 017: 順再生・逆再生を動画付きで回す（コード修正の前後を見比べるため）。
#
#   bash solution/tools/track3_reversal/run_replay_video.sh <before|after> [gripper-mode]
#
# 出力: work/probe_017/videos/<label>_ep<N>_{forward,reverse_from_fwdfinal,reverse_t3}.mp4
#       work/probe_017/replay_metrics_<label>.json
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
LABEL="${1:-before}"
GRIPPER="${2:-keep}"
OUT="${OUT:-$REPO/work/probe_017}"
T3R="$REPO/solution/tools/track3_reversal"
mkdir -p "$OUT"

QUIET='^\[libero\]|WARNING|warn|is not|^\[Mj|NVIDIA|license|License|Copyright|^=+$|CUDA|^\s*$|docs\.nvidia|Gym has been|Please upgrade|migration_guide|unmaintained'

docker run --rm \
  -v "$T3R:/app:ro" -v "$OUT:/out" \
  -e LIBERO_ROOT=/workspace/LIBERO-plus -e MUJOCO_GL=egl \
  -e PYTHONPATH=/app:/workspace/LIBERO-plus:/workspace:/workspace/compe \
  -w /workspace \
  parc2026-final python -u /app/replay_video_probe.py \
    --ep-npz /out/ep_data.npz \
    --forward-bddl /workspace/LIBERO-plus/libero/libero/bddl_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl \
    --forward-inits /workspace/LIBERO-plus/libero/libero/init_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.pruned_init \
    --t3-bddl /workspace/compe/t3/assets/bddl_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.bddl \
    --t3-inits /workspace/compe/t3/assets/init_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.pruned_init \
    --label "$LABEL" --gripper-mode "$GRIPPER" ${EXTRA:-} \
    --out /out 2>&1 | grep -vE "$QUIET"

docker run --rm -v "$OUT:/out" --entrypoint chown parc2026-final -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
echo "=== 出力: $OUT ==="
