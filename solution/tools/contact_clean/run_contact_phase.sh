#!/usr/bin/env bash
# 接触の局面診断を**課題ごとに並列で**回す。
# 1 課題 50 本で 14 分かかるので、直列に回すと 7 課題で 1.6 時間になる。
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"
DEMOS="${DEMOS:-20}"
OUT_DIR="${OUT_DIR:-work/regen/contact_phase}"
mkdir -p "$OUT_DIR" work/regen/logs

TASKS=(
  pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
  pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
)

for task in "${TASKS[@]}"; do
    bash solution/tools/regen/run_in_container.sh \
        solution/tools/contact_clean/contact_phase.py \
        --states work/artifacts/libero_states \
        --out "$OUT_DIR/$task.json" --demos "$DEMOS" --task "$task" \
        > "work/regen/logs/phase_$task.log" 2>&1 &
done
wait
# コンテナは root で書くので、持ち主を戻す。
docker run --rm -v "$REPOSITORY_ROOT/work/regen:/out" --entrypoint chown \
    parc2026-final -R "$(id -u):$(id -g)" /out
echo "=== $OUT_DIR ==="
