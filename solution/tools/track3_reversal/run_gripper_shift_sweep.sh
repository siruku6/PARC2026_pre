#!/usr/bin/env bash
# 017 続き: 逆向き再生の作り方を掃いて、配布 T3 の goal に通るかを測る。
#
#   bash solution/tools/track3_reversal/run_gripper_shift_sweep.sh [shifts] [episodes]
#
# 環境変数で選べる軸（既定はすべて従来の挙動）:
#   TASK=book|soup|mug                 どの T3 タスクか（資産の対応は _t3_container.sh）
#   EP_NPZ=ep_soup.npz                 順方向デモを抜いた npz（$OUT からの相対）
#   MODE=shift|phased                  軌道の作り方
#   INIT_SOURCE=forward-final|t3|t3-lying
#                                      逆向きの初期状態をどこから取るか
#                                      （t3-lying = 腕は home・本は順再生が残した寝姿勢）
#   GRASP_FRAME=release|grasp|clamped  掴む姿勢を順方向のどのフレームから測るか
#   STANDOFF=0.06                      掴む姿勢の手前に置く待機点の距離[m]
#   LIFT=0.08                          掴んだ直後に真上へ上がる高さ[m]
#                                      （皿やトレイを擦らないため）
#   RELEASE=schedule|contact           放す時刻。contact=触れた所で止めて開く
#   SETTLE=20                          contact のとき下ろすコマ数の上限
#   MAX_TAIL=0                         放したあと手を引く区間の上限コマ数
#                                      （-1=全部辿る・0=落とす。学習には不要）
#   MAX_STEPS=300                      これを超えた回は npz に書かない
#                                      （本番の打ち切りが 300 コマ）
#   LEVEL=1                            放す前に荷物を水平へ直す（傾いたまま置くと
#                                      放したあとに自分で起き上がる）
#   MAX_APPROACH=60                    接近の各段に許すコマ数（遠い場面では上げる）
#   SYMMETRY=1|2|0                     掴み方として等価な回転（0=缶のような回転体）
#   GRASP_STYLE=transfer|top-down      掴む姿勢を順方向から移すか、真上からの握りを作るか
#   T3_INIT=0|0,2,5|0-49|all           --init-source t3 のとき使う init の行
#                                      （配布 T3 は 1 タスクに 50 行入っている）
#   RENDER="0,5"                       この shift だけ動画も残す
#   WRITE_NPZ=reverse_npz              通った回を学習データの中継物として書く
#                                      （$OUT からの相対。MODE=phased のときだけ）
#
# **実測でいちばん良かった組み合わせ**（3 本すべてで逆向き goal が立つ）:
#   MODE=phased INIT_SOURCE=t3 GRASP_FRAME=clamped STANDOFF=0.06 \
#     bash .../run_gripper_shift_sweep.sh "0" "4061,4062,4063"
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_t3_container.sh"

run_in_container gripper_shift_sweep.py \
  --ep-npz "/out/${EP_NPZ:-ep_data.npz}" \
  --forward-bddl "$FWD_BDDL" --forward-inits "$FWD_INITS" \
  --t3-bddl "$T3_BDDL" --t3-inits "$T3_INITS" \
  "--shifts=${1:--8,-6,-4,-2,0,2,4,6,8}" \
  ${2:+--episodes "$2"} \
  --mode "${MODE:-shift}" \
  --init-source "${INIT_SOURCE:-forward-final}" \
  --grasp-frame "${GRASP_FRAME:-release}" \
  --standoff "${STANDOFF:-0}" \
  --lift "${LIFT:-0}" \
  --release "${RELEASE:-schedule}" \
  --settle "${SETTLE:-20}" \
  --max-tail "${MAX_TAIL:--1}" \
  --max-steps "${MAX_STEPS:-300}" \
  --trace-release "${TRACE:-0}" \
  ${LEVEL:+--level-before-release} \
  --max-approach "${MAX_APPROACH:-60}" \
  --object-symmetry "${SYMMETRY:-1}" \
  --grasp-style "${GRASP_STYLE:-transfer}" \
  --t3-init-index "${T3_INIT:-0}" \
  ${RENDER:+"--render-shifts=$RENDER"} \
  ${WRITE_NPZ:+"--write-npz=/out/$WRITE_NPZ"} \
  --out /out
