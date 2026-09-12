#!/usr/bin/env bash
# 丼 5 課題を、**区間をつなぐ所の通過点を先に作る／作らない**の 2 通りで回し、
# 指令の形と成功率を並べて出す。**この 2 つは同時に読むこと。**
#
#   bash solution/tools/regen/run_glide_ab.sh
#   MODES=on TAG=_g0 MOTION_GLIDE_GAIN=0 bash solution/tools/regen/run_glide_ab.sh
#
# `MODES` で片側だけ回せる。`TAG` は置き場所と log の名前に付く後ろ札で、
# **摘みを振った回どうしが同じ名前で上書きし合わないようにする。**
#
# **なぜ 2 つ同時なのか。**停止コマだけを見て採るのは、028 で `hold_at` を
# 採りかけたのと同じ失敗である（指令の形は直ったが 50 配置 246/250 が
# 10 配置 2/10 に落ちた）。**なめらかさは学習のための性質で、掴めない手本は
# 混ぜる値打ちが無い。**
#
# Reads:
#   - LIBERO-plus の bddl と `.pruned_init`（コンテナの中）
# Generates:
#   - work/regen/glide_ab/{on,off}/*.npz … 行動だけ（描画しない）
#   - work/regen/glide_ab/logs/*.log     … 成功率とステップ数
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"
OUT="${OUT:-work/regen/glide_ab}"
mkdir -p "$OUT/logs"

TASKS=(
  pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
)

TAG="${TAG:-}"
read -r -a MODE_LIST <<< "${MODES:-on off}"

for mode in "${MODE_LIST[@]}"; do
    for task in "${TASKS[@]}"; do
        bash solution/tools/regen/run_in_container.sh \
            solution/tools/scripted_demo/scripted_demo.py verify \
            --task "$task" --max-steps 300 --glide "$mode" \
            --save-actions "$OUT/${mode}${TAG}" \
            > "$OUT/logs/${mode}${TAG}_${task:26:28}.log" 2>&1 &
    done
    wait
    echo "=== glide=$mode$TAG ==="
    for log in "$OUT"/logs/${mode}${TAG}_*.log; do
        echo "  $(basename "$log" .log)  $(grep -h '^成功' "$log")"
    done
done

docker run --rm -v "$REPOSITORY_ROOT/$OUT:/out" --entrypoint chown \
    parc2026-final -R "$(id -u):$(id -g)" /out
