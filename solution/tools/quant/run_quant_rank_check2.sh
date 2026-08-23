#!/usr/bin/env bash
# 順位一致チェックの第2弾。2026-08-22、ユーザーからの2つの指摘に基づく組み直し:
#
#   1. 「本番スコア0.28のモデルでも評価したい」→ 調べたところ、本番スコア表
#      （docs/knowledge/05_score_history.md）に0.28ちょうどの行は無く、最も近い
#      3.9.0=0.264は3.8.0=0.177と**同じ重み**（model_weights_pi05_024000）で、
#      推論設定（n_action_steps）が違うだけだった。重みが別物である3本目として、
#      本番0.485（4.2.0。02_build_submission.mdで4.2.0は「pi05_rp3000と同上」と
#      明記されている）の `pi05_rp3000` を使う。
#   2. 「どうせやるなら、明確に速度が速くなった量子化の方法で」→ bnb_vision4bit
#      後掛け（C4/C5系）は、vision encoderという「推論1回・21計算のうち1回」にしか
#      効かず、速度改善そのものは実測していない（PROGRESS.md 2026-08-22 07:12節）。
#      一方 `compile_model=true`（max-autotune）は sample_actions 全体に効き、
#      実測で7.16倍（-86%）という明確な高速化を確認済み（同日09:06節）。
#      今回はこちらを「量子化あり」条件の代わりに使う。
#
# 3本の評価（弱い方＝model_weights_pi05_024000は量子化なし版が実行済みのため対象外）:
#
#   weak_compile   : model_weights_pi05_024000 に compile_model=true を掛けたもの
#   strong_plain   : pi05_rp3000（compile無し）
#   strong_compile : pi05_rp3000 に compile_model=true を掛けたもの
#
# 判定: 弱い方(0.177)と強い方(0.485)は本番スコアで既に0.308の差があり、
# ノイズ（n=36で標準偏差およそ0.07〜0.08）よりはっきり大きい。compile前後で
# この差の順序・おおよその大きさが保たれるかを見る。
#
# おまけ（4本目、任意）: C0_compile。PROGRESS.md 2026-08-22 09:06節で「36件評価での
# 正しさ確認が未確認」と指摘されていたのはこれ。要らなければ末尾の1行をコメント
# アウトすること。
#
# **2026-08-23の容量整理で work/train/pi05_compile_infer_eval（26GB）を消したので、
# このまま流すと weak_compile・strong_compile・C0_compile の3本が重みを見つけられない。**
# 流す前に3本を作り直す（重み本体は消す前と sha256 が一致することを確認済み）:
#
#   cd "$(git rev-parse --show-toplevel)"
#   mkdir -p work/train/pi05_compile_infer_eval
#   cp -r work/train/pi05_quant_c0/checkpoints/016000/pretrained_model \
#         work/train/pi05_compile_infer_eval/pretrained_model          # sha256 89fc796d
#   cp -r work/artifacts/pi05_rp3000 \
#         work/train/pi05_compile_infer_eval/strong_pretrained_model   # sha256 0c5d542e
#   cp -r work/artifacts/model_weights_pi05_024000 \
#         work/train/pi05_compile_infer_eval/weak_pretrained_model     # sha256 d46bffcf
#   # 消す前の config.json（3本とも "compile_model": true）をそのまま戻す
#   for name in pretrained_model strong_pretrained_model weak_pretrained_model; do
#       cp work/train/_deleted_configs/pi05_compile_infer_eval/$name/config.json \
#          work/train/pi05_compile_infer_eval/$name/config.json
#   done
#
# コピー元3本はどれも vlm/ を持っているので、評価が5秒で落ちる問題は起きない。
# ただし26GBを書き戻すので、先に df で空きを見ること（docs/howto/09_free_disk_space.md）。
#
# 使い方（GPUが空いていることを確認してから。docker run を伴うためユーザー自身が
# 実行すること）:
#
#   nohup bash solution/tools/quant/run_quant_rank_check2.sh \
#       > work/quant_sweep/rank_check2.log 2>&1 &
#
# 3本（+おまけ1本）×約65〜70分 ＝ 約3.5〜5時間の見込み。
set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"

SWEEP_DIR="work/quant_sweep"
RESULTS_CSV="$SWEEP_DIR/results.csv"
PROGRESS_MD="$SWEEP_DIR/PROGRESS.md"
POLICY_IMAGE="${QUANT_SWEEP_POLICY_IMAGE:-parc2026-train-pi05}"

log() { echo "[rank_check2] $(date '+%F %T') $*"; }

record_csv() {
    printf '%s,%s,%s,%s,"%s"\n' \
        "$(date +%Y-%m-%dT%H:%M:%S)" "$1" "rank_check2" "$2" "$3" >> "$RESULTS_CSV"
}

WEAK_COMPILE_WEIGHTS="$REPOSITORY_ROOT/work/train/pi05_compile_infer_eval/weak_pretrained_model"
STRONG_PLAIN_WEIGHTS="$REPOSITORY_ROOT/work/artifacts/pi05_rp3000"
STRONG_COMPILE_WEIGHTS="$REPOSITORY_ROOT/work/train/pi05_compile_infer_eval/strong_pretrained_model"
C0_COMPILE_WEIGHTS="$REPOSITORY_ROOT/work/train/pi05_compile_infer_eval/pretrained_model"

run_eval() {
    local name="$1" weights="$2"
    log "評価: $name (weights=$weights)"

    if [ ! -f "$weights/model.safetensors" ]; then
        log "  重みが無いため見送る: $weights"
        record_csv "$name" skip weights_missing
        return
    fi

    WEIGHTS="$weights" EVAL_CSV="$REPOSITORY_ROOT/compe/val/EVAL_TASKS.csv" \
        LABEL="rankcheck2_${name}" MAX_STEPS=300 N_EPISODES=1 \
        POLICY_IMAGE="$POLICY_IMAGE" \
        bash solution/tools/run_docker_eval.sh
    local status=$?

    if [ "$status" -ne 0 ]; then
        log "  評価失敗（exit $status）。次へ進む。"
        record_csv "$name" fail "exit=$status"
    else
        log "  完了。"
        record_csv "$name" ok ""
    fi
}

log "開始"
run_eval weak_compile "$WEAK_COMPILE_WEIGHTS"
run_eval strong_plain "$STRONG_PLAIN_WEIGHTS"
run_eval strong_compile "$STRONG_COMPILE_WEIGHTS"
run_eval C0_compile "$C0_COMPILE_WEIGHTS"   # おまけ。要らなければこの行をコメントアウト
log "全部終了"

{
    echo ""
    echo "## $(date '+%Y-%m-%d %H:%M') JST — run_quant_rank_check2.sh 完了"
    echo ""
    echo "weak_compile / strong_plain / strong_compile / C0_compile の4本を実行した。"
    echo "結果は work/gpu_eval/*_rankcheck2_* の summary.txt、集計は results.csv 参照。"
} >> "$PROGRESS_MD"
