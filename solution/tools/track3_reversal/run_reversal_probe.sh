#!/usr/bin/env bash
# 006 やること1・パス①(時間反転) の go/no-go 実行スクリプト。
# 会社 RTX6000 (eng003sasaki01) 用。CPU 完結(GPU 不要・描画不要)。
#
# 前提:
#   - parc2026-final 画像がローカルにある
#   - tar の展開先(下記 EXTRACT)。無いなら:
#       mkdir -p /home/ubuntu/src/train_data/libero_combined_20hz_extracted
#       tar -xf /home/ubuntu/src/train_data/libero_combined_20hz.tar -C /home/ubuntu/src/train_data/libero_combined_20hz_extracted
#
# 使い方:
#   OUT_DIR=/tmp/t3c1_out bash solution/tools/track3_reversal/run_reversal_probe.sh
#
# 出力: $OUT_DIR に fwd/rev の最終状態 npz と reversal_probe_results.json
set -euo pipefail

REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
EXTRACT="${1:-/home/ubuntu/src/train_data/libero_combined_20hz_extracted}"
OUT_DIR="${OUT_DIR:-/tmp/t3c1_out}"
mkdir -p "$OUT_DIR"

if [ ! -d "$EXTRACT/data" ] || [ ! -d "$EXTRACT/meta" ]; then
  echo "error: $EXTRACT に LeRobot v3.0 展開先(data/ meta/)が見当たらない" >&2
  echo "       展開: tar -xf /home/ubuntu/src/train_data/libero_combined_20hz.tar -C $EXTRACT" >&2
  exit 1
fi

# 0) 事前チェック: 画像内の pyarrow・forward init・t3 assets
docker run --rm --entrypoint bash parc2026-final:latest -c '
L=/workspace/LIBERO-plus/libero/libero
python3 -c "import pyarrow; print(\"pyarrow\", pyarrow.__version__)" || echo "WARN: pyarrow not in image"
ls $L/init_files/libero_90/ | grep top_of_the_shelf || echo "WARN: forward pruned_init not found"
ls /workspace/compe/t3/assets/init_files/libero_t3/ | head
'

docker run --rm \
  -v "$REPO":/host:ro \
  -v "$EXTRACT":/data:ro \
  -v "$OUT_DIR":/out \
  --entrypoint bash \
  parc2026-final:latest -c "
cd /workspace
export PYTHONPATH=/workspace
L=/workspace/LIBERO-plus/libero/libero
python3 /host/solution/tools/track3_reversal/reversal_probe.py \
  --dataset-root /data \
  --task-index 92 \
  --forward-dddl \$L/bddl_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl \
  --forward-inits \$L/init_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.pruned_init \
  --t3-dddl /workspace/compe/t3/assets/bddl_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.bddl \
  --t3-inits /workspace/compe/t3/assets/init_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.pruned_init \
  --out /out
"