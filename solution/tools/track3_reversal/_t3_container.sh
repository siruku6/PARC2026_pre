#!/usr/bin/env bash
# track3_reversal の各プローブが共有する、コンテナ実行の定型。
#
#   source "$(dirname "$0")/_t3_container.sh"
#   run_in_container my_probe.py --foo bar
#
# **タスク資産のパスをここ 1 か所に置くのが目的である。**同じ 4 つの絶対パスを
# 各ランナーに書くと、資産の場所や PYTHONPATH を直したときに片方だけ古くなり、
# 「比べるために作った 2 つのプローブ」が黙って別条件になる。
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
OUT="${OUT:-$REPO/work/probe_017}"
T3R="$REPO/solution/tools/track3_reversal"

# LIBERO-plus 側（順方向 task 92）と、配布キット側（逆向き T3）の資産。
FWD_BDDL=/workspace/LIBERO-plus/libero/libero/bddl_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl
FWD_INITS=/workspace/LIBERO-plus/libero/libero/init_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.pruned_init
T3_BDDL=/workspace/compe/t3/assets/bddl_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.bddl
T3_INITS=/workspace/compe/t3/assets/init_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.pruned_init

# 起動時の雑音（CUDA/EGL/gym の警告）を落とす。結果の行だけ残したい。
QUIET='^\[libero\]|WARNING|warn|is not|^\[Mj|NVIDIA|license|License|Copyright|^=+$|CUDA|^\s*$|docs\.nvidia|Gym has been|Please upgrade|migration_guide|unmaintained'

run_in_container() {
  local script="$1"; shift
  mkdir -p "$OUT"
  docker run --rm \
    -v "$T3R:/app:ro" -v "$OUT:/out" \
    -e LIBERO_ROOT=/workspace/LIBERO-plus -e MUJOCO_GL=egl \
    -e PYTHONPATH=/app:/workspace/LIBERO-plus:/workspace:/workspace/compe \
    -w /workspace \
    parc2026-final python -u "/app/$script" "$@" 2>&1 | grep -vE "$QUIET"
  # コンテナが root で書いた出力を、ホスト側の所有に戻す
  docker run --rm -v "$OUT:/out" --entrypoint chown parc2026-final \
    -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
  echo "=== 出力: $OUT ==="
}
