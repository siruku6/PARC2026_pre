#!/usr/bin/env bash
# 重みの系譜を順に流して、各点の json を貯める。
#
#   bash solution/tools/probe/run_lineage_all.sh
#
# **1 点あたり 8〜12 分。**重みの読み込み（8.8 GB を 1 テンソルずつ）が支配的で、
# 推論そのものは数秒である。**同時に走らせない**——1 点で 10 GB 使う。
#
# 順番は系譜の順にしてある（base → stage1 → rp3000 → trial1500、
# そこから分岐する proto20 と t3rev）。**先に base を見るのが要である**
# ——私たちが 1 歩も学習していない状態で何ができていたかが基準になる。
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
cd "$REPO"
OUT="${OUT:-work/probe_color}"
T3="${T3:-work/probe_color/t3_obs.npz}"
BASE="${BASE:-work/probe_color/base_obs.npz}"

# 名前 → 重みの場所。**順番を保つので配列で持つ。**
NAMES=(base stage1_24000 rp3000 trial1500_001500 proto20_002000 t3rev_008700)
PATHS=(
  work/artifacts/pi05_libero_base
  work/artifacts/pi05_stage1_24000
  work/artifacts/pi05_rp3000
  work/artifacts/trial1500/001500
  work/artifacts/pi05_proto20_002000
  work/artifacts/pi05_t3rev_008700
)

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; w="${PATHS[$i]}"
    json="$OUT/lin_$name.json"
    if [ -f "$json" ]; then
        echo "=== $name はもうある — 飛ばす ==="; continue
    fi
    if [ ! -f "$w/model.safetensors" ]; then
        echo "=== $name は重みが無い（${w}）— 飛ばす ==="; continue
    fi
    echo; echo "########## $name ##########"
    OBS2="$BASE" bash solution/tools/probe/run_lineage_probe.sh "$w" "$T3" "$json" \
        2>&1 | grep -viE 'svt|warning|^$'
done
echo; echo "=== 揃った json ==="; ls -la "$OUT"/lin_*.json 2>/dev/null
