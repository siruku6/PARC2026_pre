#!/usr/bin/env bash
# 006 やること1・パス①(時間反転) の go/no-go 実行スクリプト。
# 会社 RTX6000 (eng003sasaki01) 用。CPU 完結(GPU 不要・描画不要)。
#
# 2 段構成:
#   段1(ホスト, pyenv の python3 = pyarrow 入り): tar 展開先から action 列を抽出
#   段2(parc2026-final 画像内): 順再生して最終状態を捕まえ、反転 action を
#       T3 の初期状態から再生して :goal 判定
#
# 前提:
#   - parc2026-final 画像がローカルにある
#   - tar の展開先（無いなら）:
#       mkdir -p /home/ubuntu/src/train_data/libero_combined_20hz_extracted
#       tar -xf /home/ubuntu/src/train_data/libero_combined_20hz.tar -C /home/ubuntu/src/train_data/libero_combined_20hz_extracted
#   - ホストの python3 に pyarrow がある（pyenv の 3.12.14 で確認済み）
#
# 使い方:
#   bash solution/tools/track3_reversal/run_reversal_probe.sh [展開先] [gripper-mode]
#   gripper-mode: keep(既定) | invert
#
# 出力: /tmp/t3c1_out に fwd/rev の最終状態 npz と reversal_probe_results.json
set -euo pipefail

REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
EXTRACT="${1:-/home/ubuntu/src/train_data/libero_combined_20hz_extracted}"
GRIPPER_MODE="${2:-keep}"
OUT_DIR="${OUT_DIR:-/tmp/t3c1_out}"
mkdir -p "$OUT_DIR"

if [ ! -d "$EXTRACT/data" ] || [ ! -d "$EXTRACT/meta" ]; then
  echo "error: $EXTRACT に LeRobot v3.0 展開先(data/ meta/)が見当たらない" >&2
  echo "       展開: tar -xf /home/ubuntu/src/train_data/libero_combined_20hz.tar -C $EXTRACT" >&2
  exit 1
fi

echo "=== 段1: ホスト側で action 列を抽出（task 92 の成功エピソード先頭3本）==="
python3 "$REPO/solution/tools/track3_reversal/host_extract_actions.py" \
  --dataset-root "$EXTRACT" \
  --task-index 92 \
  --out "$OUT_DIR/t3c1_actions.npz"

# ホストのパスは画像内では見えないので、npz を OUT_DIR 直下に置く（画像からは /out で読む）
echo "=== 段2: parc2026-final 内で順再生→反転再生→:goal 判定（gripper=$GRIPPER_MODE）==="
docker run --rm \
  -v "$REPO":/host:ro \
  -v "$OUT_DIR":/out \
  --entrypoint bash \
  parc2026-final:latest -c "
cd /workspace
export PYTHONPATH=/workspace
L=/workspace/LIBERO-plus/libero/libero
python3 /host/solution/tools/track3_reversal/reversal_probe.py \
  --actions-npz /out/t3c1_actions.npz \
  --forward-dddl \$L/bddl_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl \
  --forward-inits \$L/init_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.pruned_init \
  --t3-dddl /workspace/compe/t3/assets/bddl_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.bddl \
  --t3-inits /workspace/compe/t3/assets/init_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.pruned_init \
  --gripper-mode $GRIPPER_MODE \
  --out /out
"
echo "=== 結果は $OUT_DIR/reversal_probe_results.json ==="