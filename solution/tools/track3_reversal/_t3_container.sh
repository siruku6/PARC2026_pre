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

# LIBERO-plus 側（順方向）と、配布キット側（逆向き T3）の資産。
#
# **逆向きは順方向デモの時刻反転で作るので、T3 の 1 件ごとに「対応する順方向」が
# 要る。**その対応をここで持つ。`TASK=soup` のように選ぶ（既定は book）。
# 配布 T3 の 4 件目（two mugs）は順方向 2 本の合成が要るのでまだ無い。
TASK="${TASK:-book}"
case "$TASK" in
  book) FWD_STEM=STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf
        T3_STEM=take_the_yellow_book_off_the_shelf_and_put_it_on_the_table ;;
  soup) FWD_STEM=LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray
        T3_STEM=take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table ;;
  mug)  FWD_STEM=LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate
        T3_STEM=take_the_red_mug_off_the_left_plate_and_put_it_on_the_table ;;
  *) echo "TASK は book|soup|mug のどれか（来た値: $TASK）" >&2; exit 1 ;;
esac
L90=/workspace/LIBERO-plus/libero/libero
T3A=/workspace/compe/t3/assets
FWD_BDDL="$L90/bddl_files/libero_90/$FWD_STEM.bddl"
FWD_INITS="$L90/init_files/libero_90/$FWD_STEM.pruned_init"
T3_BDDL="$T3A/bddl_files/libero_t3/$T3_STEM.bddl"
T3_INITS="$T3A/init_files/libero_t3/$T3_STEM.pruned_init"

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
