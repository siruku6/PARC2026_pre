#!/usr/bin/env bash
# 量子化スイープ（run_quant_sweep.sh）の docker exec 終了コード誤判定バグにより、
# 評価がスキップされた候補をまとめて評価し直す回収用スクリプト。
# work/quant_sweep/PROGRESS.md 2026-08-21 の記録を参照。
#
# **run_quant_sweep.sh（本体のスイープ）が完全に終了してから実行すること。**
# GPU を共有するため、学習が動いている間に走らせると学習側が落ちるリスクがある。
# 確認方法: `ps aux | grep run_quant_sweep` に何も出ない、または
# work/quant_sweep/sweep.log の末尾に「===== 量子化スイープ終了 =====」がある。
#
# 使い方（学習側スイープが終わったことを確認してから）:
#   nohup bash solution/tools/run_quant_eval_recovery.sh \
#       > work/quant_sweep/eval_recovery.log 2>&1 &
#
# C0〜C3 は各自のチェックポイントを、C4/C5 は C0 のチェックポイントを
# 推論時量子化して評価する（work/quant_sweep/CANDIDATES.md 参照）。
# 重みファイルが無い候補（学習がまだ終わっていない等）は自動的に飛ばす。
set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"

SWEEP_DIR="work/quant_sweep"
RESULTS_CSV="$SWEEP_DIR/results.csv"
PROGRESS_MD="$SWEEP_DIR/PROGRESS.md"
STEPS="${QUANT_SWEEP_STEPS:-16000}"
POLICY_IMAGE="${QUANT_SWEEP_POLICY_IMAGE:-parc2026-train-pi05}"

log() { echo "[recovery] $(date '+%F %T') $*"; }

record_csv() {
    printf '%s,%s,%s,%s,"%s"\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$1" "eval" "$2" "$3" >> "$RESULTS_CSV"
}

checkpoint_path() {
    local run_name="pi05_quant_${1,,}"
    printf '%s/work/train/%s/checkpoints/%06d/pretrained_model' \
        "$REPOSITORY_ROOT" "$run_name" "$STEPS"
}

C0_WEIGHTS="$(checkpoint_path C0)"

run_eval() {
    local name="$1" weights="$2" serve_mode="${3:-}"
    log "評価: $name (weights=$weights, serve_mode=${serve_mode:-なし})"

    if [ ! -f "$weights/model.safetensors" ]; then
        log "  重みが無いため見送る: $weights"
        record_csv "$name" skip weights_missing
        return
    fi

    WEIGHTS="$weights" EVAL_CSV="$REPOSITORY_ROOT/compe/val/EVAL_TASKS.csv" \
        LABEL="quant_${name}_recovery" MAX_STEPS=300 N_EPISODES=1 \
        POLICY_IMAGE="$POLICY_IMAGE" \
        QUANT_SERVE_MODE="$serve_mode" \
        bash solution/tools/run_docker_eval.sh
    local status=$?

    if [ "$status" -ne 0 ]; then
        log "  評価失敗（exit $status）。次へ進む。"
        record_csv "$name" fail "exit=$status"
    else
        record_csv "$name" ok ""
    fi
}

log "===== 回収用の評価をまとめて実行する ====="
{
    echo
    echo "## $(date '+%Y-%m-%d %H:%M %Z') — 評価の回収実行を開始（run_quant_eval_recovery.sh 自動記録）"
} >> "$PROGRESS_MD"

run_eval C0 "$C0_WEIGHTS"
run_eval C1 "$(checkpoint_path C1)"
run_eval C2 "$(checkpoint_path C2)"
run_eval C3 "$(checkpoint_path C3)"
run_eval C4 "$C0_WEIGHTS" bnb_vision4bit
run_eval C5 "$C0_WEIGHTS" torchao_int8

log "===== 回収完了 ====="
{
    echo
    echo "## $(date '+%Y-%m-%d %H:%M %Z') — 評価の回収実行が完了（run_quant_eval_recovery.sh 自動記録）"
    echo
    echo "$(tail -n 10 "$RESULTS_CSV" | sed 's/^/    /')"
} >> "$PROGRESS_MD"
