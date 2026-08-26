#!/usr/bin/env bash
# 量子化スイープを「実験を加速する道具」として使ってよいかを判定するための
# 追加評価。統括セッションからの指摘（2026-08-22）に基づく:
#
#   量子化ありのスコア・ステップ数の絶対値は見てはいけない
#   （量子化すると方策の出力そのものが変わるため、docs/knowledge/07_local_eval_cannot_rank.md
#   の「手元評価は本番の順位を付けられない」とまったく同じ問題が立つ）。
#   見るべきは「量子化なしでの候補の順位」と「量子化ありでの候補の順位」が
#   一致しているかだけである。一致するなら、以降の探査は量子化して回せる
#   （本数を増やせる）。一致しないなら、量子化は提出物の軽量化にしか使えない。
#
# **候補は学習側の4本（C0 baseline・C1 bnb8bit optimizer・C2 bnb4bit vision
# encoder・C3 torchao float8 expert）。** C4/C5 は元から「推論時に量子化を
# 掛けるかどうか」自体が候補の軸であり、この4本に「同じ量子化方式（bnb 4bit の
# vision-only）を後掛けしたときの評価」を追加で走らせることで、
#
#   量子化なし: 4本をそれぞれ素の（推論時量子化なしの）状態で評価した順位
#   量子化あり: 同じ4本に QUANT_SERVE_MODE=bnb_vision4bit を後掛けして評価した順位
#
# という対にする。C2 はチェックポイント自体が学習時に4bit量子化されているため、
# 素の評価には dequantize_bnb4bit_checkpoint.py で bf16 に戻したものを使う
# （work/quant_sweep/PROGRESS.md 2026-08-22 参照。他候補と土台を揃えるため）。
#
# 使い方（GPU が空いていることを確認してから。docker build/run を伴うため
# ユーザー自身が実行すること）:
#
#   docker build -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train
#   nohup bash solution/tools/quant/run_quant_rank_check.sh \
#       > work/quant_sweep/rank_check.log 2>&1 &
#
# **C0・C1 の「量子化なし」は回収済み（2026-08-21 の run_quant_eval_recovery.sh
# で C0 mean_success_rate 0.7222・C1 0.6667 を確認済み）なのでここでは
# 再実行しない。** 新しく要るのは6本（C2・C3の量子化なし、C0〜C3の量子化あり）
# × 約70分 = 約7時間の見込み。結果は results.csv に phase=rank_check として
# 追記され、PROGRESS.md にも追記する。
set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"

SWEEP_DIR="work/quant_sweep"
RESULTS_CSV="$SWEEP_DIR/results.csv"
PROGRESS_MD="$SWEEP_DIR/PROGRESS.md"
STEPS="${QUANT_SWEEP_STEPS:-16000}"
POLICY_IMAGE="${QUANT_SWEEP_POLICY_IMAGE:-parc2026-train-pi05}"

log() { echo "[rank_check] $(date '+%F %T') $*"; }

record_csv() {
    printf '%s,%s,%s,%s,"%s"\n' \
        "$(date +%Y-%m-%dT%H:%M:%S)" "$1" "rank_check" "$2" "$3" >> "$RESULTS_CSV"
}

# checkpoints/<step>/pretrained_model の場所（C2 だけ dequant 版を指す）。
C0_WEIGHTS="$REPOSITORY_ROOT/work/train/pi05_quant_c0/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
C1_WEIGHTS="$REPOSITORY_ROOT/work/train/pi05_quant_c1/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
C2_WEIGHTS="$REPOSITORY_ROOT/work/train/pi05_quant_c2/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model_dequant"
C3_WEIGHTS="$REPOSITORY_ROOT/work/train/pi05_quant_c3/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"

run_eval() {
    local name="$1" weights="$2" serve_mode="${3:-}"
    log "評価: $name (weights=$weights, serve_mode=${serve_mode:-なし})"

    if [ ! -f "$weights/model.safetensors" ]; then
        log "  重みが無いため見送る: $weights"
        record_csv "$name" skip weights_missing
        return
    fi

    WEIGHTS="$weights" EVAL_CSV="$REPOSITORY_ROOT/compe/val/EVAL_TASKS.csv" \
        LABEL="rankcheck_${name}" MAX_STEPS=300 N_EPISODES=1 \
        POLICY_IMAGE="$POLICY_IMAGE" \
        QUANT_SERVE_MODE="$serve_mode" \
        bash solution/tools/eval/run_docker_eval.sh
    local status=$?

    if [ "$status" -ne 0 ]; then
        log "  評価失敗（exit $status）。次へ進む。"
        record_csv "$name" fail "exit=$status"
    else
        record_csv "$name" ok ""
    fi
}

log "===== 順位一致チェック用の評価をまとめて実行する（6本、約7時間の見込み） ====="
{
    echo
    echo "## $(date '+%Y-%m-%d %H:%M %Z') — 順位一致チェックの評価を開始（run_quant_rank_check.sh 自動記録）"
} >> "$PROGRESS_MD"

# 量子化なし（素の評価。C2 は bf16 に戻した版）。
# C0・C1 は 2026-08-21 の回収評価で確認済み（0.7222・0.6667）のためここでは省く。
run_eval C2_plain "$C2_WEIGHTS"
run_eval C3_plain "$C3_WEIGHTS"

# 量子化あり（同じ4本に bnb 4bit vision-only を後掛け）
run_eval C0_quant "$C0_WEIGHTS" bnb_vision4bit
run_eval C1_quant "$C1_WEIGHTS" bnb_vision4bit
run_eval C2_quant "$C2_WEIGHTS" bnb_vision4bit
run_eval C3_quant "$C3_WEIGHTS" bnb_vision4bit

log "===== 順位一致チェック完了 ====="
{
    echo
    echo "## $(date '+%Y-%m-%d %H:%M %Z') — 順位一致チェックの評価が完了（run_quant_rank_check.sh 自動記録）"
    echo
    echo "$(tail -n 8 "$RESULTS_CSV" | sed 's/^/    /')"
    echo
    echo "**各回の mean_success_rate は work/gpu_eval/\$(date)_rankcheck_*/ 配下の"
    echo "server_8000.json を見ること。順位表はここに手で追記する。**"
} >> "$PROGRESS_MD"
