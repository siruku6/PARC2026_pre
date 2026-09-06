#!/usr/bin/env bash
# 017 続き: 逆向き再生の gripper 指令だけを時間シフトして掃く。
#
#   bash solution/tools/track3_reversal/run_gripper_shift_sweep.sh [shifts] [episodes]
#
# 環境変数で選べる軸（既定はすべて現行の挙動）:
#   MODE=shift|phased          軌道の作り方
#   INIT_SOURCE=forward-final|t3   逆向きの初期状態をどこから取るか
#   GRASP_FRAME=release|grasp|clamped  掴む姿勢を順方向のどのフレームから測るか
#   T3_INIT=<行番号>            --init-source t3 のとき使う init
#   STANDOFF=0.06              掴む姿勢の手前に置く待機点の距離[m]
#   RENDER="0,5"               この shift だけ動画も残す
#   REPEATS=4                  繰り返し回数
#
# 例: bash .../run_gripper_shift_sweep.sh "-8,-4,0,4,8" "4061,4062"
# 例: MODE=phased INIT_SOURCE=t3 GRASP_FRAME=grasp bash .../run_gripper_shift_sweep.sh "0" "4061"
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
SHIFTS="${1:--8,-6,-4,-2,0,2,4,6,8}"
EPS="${2:-}"
OUT="${OUT:-$REPO/work/probe_017}"
T3R="$REPO/solution/tools/track3_reversal"

QUIET='^\[libero\]|WARNING|warn|is not|^\[Mj|NVIDIA|license|License|Copyright|^=+$|CUDA|^\s*$|docs\.nvidia|Gym has been|Please upgrade|migration_guide|unmaintained'

docker run --rm \
  -v "$T3R:/app:ro" -v "$OUT:/out" \
  -e LIBERO_ROOT=/workspace/LIBERO-plus -e MUJOCO_GL=egl \
  -e PYTHONPATH=/app:/workspace/LIBERO-plus:/workspace:/workspace/compe \
  -w /workspace \
  parc2026-final python -u /app/gripper_shift_sweep.py \
    --ep-npz /out/ep_data.npz \
    --forward-bddl /workspace/LIBERO-plus/libero/libero/bddl_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl \
    --forward-inits /workspace/LIBERO-plus/libero/libero/init_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.pruned_init \
    --t3-bddl /workspace/compe/t3/assets/bddl_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.bddl \
    --t3-inits /workspace/compe/t3/assets/init_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.pruned_init \
    --standoff "${STANDOFF:-0}" \
    --init-source "${INIT_SOURCE:-forward-final}" \
    --grasp-frame "${GRASP_FRAME:-release}" \
    --t3-init-index "${T3_INIT:-0}" \
    "--shifts=$SHIFTS" ${EPS:+--episodes "$EPS"} ${RENDER:+"--render-shifts=$RENDER"} \
    --repeats "${REPEATS:-1}" --mode "${MODE:-shift}" \
    --out /out 2>&1 | grep -vE "$QUIET"

docker run --rm -v "$OUT:/out" --entrypoint chown parc2026-final -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
echo "=== 出力: $OUT/gripper_shift_sweep.json ==="
