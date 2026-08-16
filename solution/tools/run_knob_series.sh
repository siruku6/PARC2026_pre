#!/usr/bin/env bash
# 設定値の候補を 1 つずつ順番に評価する。
#
# **同時には回せない。** `run_docker_eval.sh` は `parc-policy` という固定名の
# 容器を消してから作るため、2 本目が 1 本目のサーバーを殺す。
# このため、ここでは `--detach` を付けずに 1 本ずつ終わるまで待つ。
#
# 既定の一覧（`compe/val/EVAL_TASKS_FAST12.csv`）は 12 件で、
# **n=16 で衝突して失格した 4 件と、成功していた 8 件**を選んである。
# 滑らかさ（ジャーク・SPARC）は 1 本 130 ステップほどの平均なので 12 件でも読めるが、
# **成功件数の差を 12 件で判断してはいけない**（最後に 36 件で確かめる）。
#
# 使い方:
#   nohup bash solution/tools/run_knob_series.sh > work/knob_series.log 2>&1 &
#   tail -f work/knob_series.log
#
# 各行は「ラベル<TAB>環境変数の並び」である。上から順に回す。
set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"

EVAL_CSV_PATH="${EVAL_CSV:-compe/val/EVAL_TASKS_FAST12.csv}"
WEIGHTS_PATH="${WEIGHTS:-work/artifacts/model_weights_pi05_024000}"

# 比べる候補。**1 行 1 つの設定値だけ動かす**のを基本にし、
# 最後に「効いたものを重ねた回」を置く。
# **8/14 18:40 に並べ替えた。** blend_steps=4 が 36 件で当たった
# （成功 0.694 → 0.778・ジャーク −10%・方策の衝突率 0.111 → 0.083）ので、
# **8 を先頭に上げる。** 4 より良くなるか、行き過ぎて応答が遅れるかの
# どちらかで、どちらに出てもそのまま読める。
# **既定が 4 になった（版 3.10.0）ので、4 と比べたい回は 0 を明示的に渡す。**
KNOB_SERIES=(
    "blend8	PARC_N_ACTION_STEPS=16 PARC_BLEND_STEPS=8"
    "ema05	PARC_N_ACTION_STEPS=16 PARC_BLEND_STEPS=4 PARC_EMA_ALPHA=0.5"
    "infer20	PARC_N_ACTION_STEPS=16 PARC_BLEND_STEPS=4 PARC_NUM_INFERENCE_STEPS=20"
    "n20	PARC_N_ACTION_STEPS=20 PARC_BLEND_STEPS=4"
)

echo "===== 設定値を順番に回す（${#KNOB_SERIES[@]} 本）====="
echo "  一覧   $EVAL_CSV_PATH"
echo "  重み   $WEIGHTS_PATH"
echo

for entry in "${KNOB_SERIES[@]}"; do
    label="${entry%%	*}"
    knobs="${entry#*	}"

    echo "===== $label  ($knobs) ====="
    date '+  開始 %H:%M:%S'

    env $knobs \
        POLICY_IMAGE=parc2026-train-pi05 \
        WEIGHTS="$WEIGHTS_PATH" \
        EVAL_CSV="$EVAL_CSV_PATH" \
        LABEL="pi05_24k_$label" \
        MAX_STEPS=300 \
        N_EPISODES=1 \
        bash solution/tools/run_docker_eval.sh

    status=$?
    date '+  終了 %H:%M:%S'
    if [ "$status" -ne 0 ]; then
        echo "  **失敗（終了コード $status）。次へ進む。**"
    fi
    echo
done

echo "===== 全部終わった。並べて比べる ====="
echo "  python3 solution/tools/summarize_gpu_eval.py --output-dir <各出力先> \\"
echo "      --csv $EVAL_CSV_PATH --compare <基準の回>"
