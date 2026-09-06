#!/usr/bin/env bash
# 017 続き: 逆向き再生の作り方を掃いて、配布 T3 の goal に通るかを測る。
#
#   bash solution/tools/track3_reversal/run_gripper_shift_sweep.sh [shifts] [episodes]
#
# 環境変数で選べる軸（既定はすべて従来の挙動）:
#   MODE=shift|phased                  軌道の作り方
#   INIT_SOURCE=forward-final|t3|t3-lying
#                                      逆向きの初期状態をどこから取るか
#                                      （t3-lying = 腕は home・本は順再生が残した寝姿勢）
#   GRASP_FRAME=release|grasp|clamped  掴む姿勢を順方向のどのフレームから測るか
#   STANDOFF=0.06                      掴む姿勢の手前に置く待機点の距離[m]
#   T3_INIT=0|0,2,5|0-49|all           --init-source t3 のとき使う init の行
#                                      （配布 T3 は 1 タスクに 50 行入っている）
#   RENDER="0,5"                       この shift だけ動画も残す
#
# **実測でいちばん良かった組み合わせ**（3 本すべてで逆向き goal が立つ）:
#   MODE=phased INIT_SOURCE=t3 GRASP_FRAME=clamped STANDOFF=0.06 \
#     bash .../run_gripper_shift_sweep.sh "0" "4061,4062,4063"
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_t3_container.sh"

run_in_container gripper_shift_sweep.py \
  --ep-npz /out/ep_data.npz \
  --forward-bddl "$FWD_BDDL" --forward-inits "$FWD_INITS" \
  --t3-bddl "$T3_BDDL" --t3-inits "$T3_INITS" \
  "--shifts=${1:--8,-6,-4,-2,0,2,4,6,8}" \
  ${2:+--episodes "$2"} \
  --mode "${MODE:-shift}" \
  --init-source "${INIT_SOURCE:-forward-final}" \
  --grasp-frame "${GRASP_FRAME:-release}" \
  --standoff "${STANDOFF:-0}" \
  --t3-init-index "${T3_INIT:-0}" \
  ${RENDER:+"--render-shifts=$RENDER"} \
  --out /out
