#!/usr/bin/env bash
# 017: 順再生・逆再生を動画付きで回す（コード修正の前後を見比べるため）。
#
#   bash solution/tools/track3_reversal/run_replay_video.sh <before|after> [gripper-mode]
#
# 出力: work/probe_017/videos/<label>_ep<N>_*.mp4 と replay_metrics_<label>.json
# EXTRA=--legacy で修正前の作り方（gripper 復号なし・符号反転なし）を再現する。
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_t3_container.sh"

run_in_container replay_video_probe.py \
  --ep-npz /out/ep_data.npz \
  --forward-bddl "$FWD_BDDL" --forward-inits "$FWD_INITS" \
  --t3-bddl "$T3_BDDL" --t3-inits "$T3_INITS" \
  --label "${1:-before}" --gripper-mode "${2:-keep}" ${EXTRA:-} \
  --out /out
