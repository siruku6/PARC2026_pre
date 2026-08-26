#!/usr/bin/env bash
# 量子化スイープの常駐ドライバ。tmux/nohup 配下で起動し、SSH 切断後も動き続ける。
#
# **土台はこの bash スクリプトだけ。** 「claude セッション自体を常駐させる」形は
# 採らない —— claude プロセスが tmux デタッチ＋SSH切断を超えて自律的に動き続けるかは
# 公式ドキュメントに記載が無い未確認事項だったため（2026-08-20、
# work/quant_sweep/PROGRESS.md 参照）。判断が要る場面だけ、このスクリプトが
# `claude -p --tools ""`（ツール権限ゼロ、テキストの JSON 判断だけを返す、
# すぐ終わる短命呼び出し）を呼ぶ。呼び出しが壊れても後述のホワイトリスト・
# 固定順フォールバックで安全に進む。
#
# 学習は常駐コンテナへの `docker exec` で回す（毎回 docker build/run し直さない —
# work/quant_sweep/CANDIDATES.md 参照）。評価は既存の run_docker_eval.sh を
# そのまま使う（run_knob_series.sh（削除済み）と同じ、実証済みの docker run/rm サイクル。
# 出力先・重みのマウント先が回ごとに変わる設計なので、無理に常駐コンテナへ
# 作り替えない）。
#
# 使い方（ユーザーが1回だけ、コンテナを常駐させた後に）:
#   tmux new -d -s quant_sweep \
#       'bash solution/tools/quant/run_quant_sweep.sh 2>&1 | tee -a work/quant_sweep/sweep.log'
#   tmux attach -t quant_sweep   # 後で様子を見るとき
#
# 環境変数（省略時の既定）:
#   TRAIN_CONTAINER   常駐させた学習コンテナの名前（既定 quant_train）
#   QUANT_SWEEP_STEPS 各候補の学習ステップ数（既定 16000。work/quant_sweep/PROGRESS.md
#                     のステップ数見積りを参照。較正 run の実測値で見直すこと）
#   QUANT_SWEEP_BATCH batch_size（既定 16。work/train/pi05_full_local.log の実測値）
#   MAX_SWEEP_HOURS   壁時計の強制終了までの時間（既定 24）
set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"

SWEEP_DIR="work/quant_sweep"
RESULTS_CSV="$SWEEP_DIR/results.csv"
PROGRESS_MD="$SWEEP_DIR/PROGRESS.md"
mkdir -p "$SWEEP_DIR"
[ -f "$RESULTS_CSV" ] || echo "timestamp,candidate,phase,status,detail" > "$RESULTS_CSV"

TRAIN_CONTAINER="${TRAIN_CONTAINER:-quant_train}"
# **run_docker_eval.sh の既定 POLICY_IMAGE は parc2026-train（SmolVLA 用）である。**
# このスイープは π0.5 専用なので、明示的に上書きしないと間違ったイメージ
# （bitsandbytes/torchao が入っていない、pi05 も読めない）で評価してしまう。
POLICY_IMAGE="${QUANT_SWEEP_POLICY_IMAGE:-parc2026-train-pi05}"
STEPS="${QUANT_SWEEP_STEPS:-16000}"
BATCH_SIZE="${QUANT_SWEEP_BATCH:-16}"
MAX_SWEEP_HOURS="${MAX_SWEEP_HOURS:-24}"
MIN_FREE_GB=40
START_EPOCH=$(date +%s)
MAX_SECONDS=$((MAX_SWEEP_HOURS * 3600))

# **ホワイトリスト。** claude -p の判断であっても、ここに無い名前は実行しない。
CANDIDATES_ALL=(C0 C1 C2 C3 C4 C5)
declare -A RESULT_OF=()   # 候補名 -> ok/fail/skip（試行済みかどうかもこれで判定）

log() {
    local line="[sweep] $(date '+%F %T') $*"
    echo "$line"
}

progress_note() {
    # PROGRESS.md に時刻付きで追記する（人間向け）。
    {
        echo
        echo "## $(date '+%Y-%m-%d %H:%M %Z') — $1（run_quant_sweep.sh 自動記録）"
        echo
        echo "$2"
    } >> "$PROGRESS_MD"
}

record_csv() {
    # $1=candidate $2=phase(train/eval) $3=status(ok/fail/skip) $4=detail
    printf '%s,%s,%s,%s,"%s"\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$1" "$2" "$3" "$4" >> "$RESULTS_CSV"
}

check_disk() {
    local free_gb
    free_gb=$(df --output=avail -BG "$REPOSITORY_ROOT" | tail -1 | tr -dc '0-9')
    if (( free_gb < MIN_FREE_GB )); then
        log "**空き ${free_gb}GB < ${MIN_FREE_GB}GB。危険域につき見送る。**"
        return 1
    fi
    return 0
}

time_left_seconds() {
    local elapsed=$(( $(date +%s) - START_EPOCH ))
    echo $(( MAX_SECONDS - elapsed ))
}

remaining_candidates() {
    local out=()
    for c in "${CANDIDATES_ALL[@]}"; do
        [ -z "${RESULT_OF[$c]:-}" ] && out+=("$c")
    done
    echo "${out[@]}"
}

checkpoint_path() {
    # $1=candidate名 → ホスト側から見たチェックポイントのパス
    # （学習コンテナの /out は work/train を bind mount している前提。
    #   Dockerfile.pi05 の使い方コメント参照）
    local run_name="pi05_quant_${1,,}"
    printf '%s/work/train/%s/checkpoints/%06d/pretrained_model' \
        "$REPOSITORY_ROOT" "$run_name" "$STEPS"
}

# ---------------------------------------------------------------- 学習(docker exec)

train_candidate() {
    local name="$1"; shift
    local run_name="pi05_quant_${name,,}"
    local extra_args=("$@")
    log "学習開始: $name (run-name=$run_name, steps=$STEPS, batch=$BATCH_SIZE) ${extra_args[*]:-}"

    if ! check_disk; then
        record_csv "$name" train skip disk_low
        return 1
    fi

    docker exec "$TRAIN_CONTAINER" python /opt/pi05/train_pi05.py \
        --run-name "$run_name" --steps "$STEPS" --batch-size "$BATCH_SIZE" \
        --save-freq "$STEPS" "${extra_args[@]}"
    local status=$?

    # **`docker exec` 自身の終了コードを鵜呑みにしない。** 2026-08-21、実機で
    # C0・C1 とも `docker exec` は exit=120 を返したが、コンテナ内の
    # train_pi05.py 自身のログには「終了コード 0」「End of training」が記録され、
    # チェックポイントも実際に書けていた（6時間規模の docker exec 特有の現象と
    # みられるが、正確な原因は未確定）。**チェックポイントが実在するかどうかを
    # 真の判定基準にする** —— docker exec の終了コードが非ゼロでも、期待する
    # safetensors ファイルがあれば学習は成功したとみなす。
    local checkpoint_dir
    checkpoint_dir="$(checkpoint_path "$name")"
    if [ "$status" -ne 0 ] && [ ! -f "$checkpoint_dir/model.safetensors" ]; then
        log "**学習失敗（$name, exit $status, チェックポイントも無い）。次の候補へ進む。**"
        record_csv "$name" train fail "exit=$status"
        return 1
    elif [ "$status" -ne 0 ]; then
        log "**$name: docker exec は exit=$status だったが、チェックポイントは"
        log "  実在する（$checkpoint_dir/model.safetensors）。学習成功として扱う。**"
    fi

    # steps/sec・peak mem をログから抜く（lerobot-train 自身の指標。新規計装不要）。
    local log_file="work/train/${run_name}.log"
    local steps_per_sec="NA" peak_mem="NA"
    if [ -f "$log_file" ]; then
        steps_per_sec=$(grep -oP 'updt_s:\K[0-9.]+' "$log_file" | tail -n +3 | \
            python3 -c "
import sys
v = [float(x) for x in sys.stdin]
print(round(1/(sum(v)/len(v)), 3) if v else 'NA')
" 2>/dev/null || echo NA)
        peak_mem=$(grep -oP 'mem_gb:\K[0-9.]+' "$log_file" | sort -g | tail -1 || echo NA)
    fi
    record_csv "$name" train ok "steps_per_sec=$steps_per_sec,peak_mem_gb=$peak_mem"

    # 学習が終わった直後（もう書き込みプロセスは無い）に間引く。最終チェックポイント
    # 1個だけ残す。KEEP=1 の警告（保存中を消す競合）はここでは当たらない。
    docker exec "$TRAIN_CONTAINER" bash /opt/pi05/prune_checkpoints.sh \
        "/out/${run_name}" 1 --once || true
    return 0
}

# ---------------------------------------------------------------- 評価(run_docker_eval.sh)

eval_candidate() {
    # $1=候補名 $2=重みのホストパス $3=QUANT_SERVE_MODE(空可) $4=QUANT_SERVE_SCOPE(空可)
    local name="$1" weights="$2" serve_mode="${3:-}" serve_scope="${4:-}"
    log "評価開始: $name (weights=$weights, serve_mode=${serve_mode:-なし})"

    if ! check_disk; then
        record_csv "$name" eval skip disk_low
        return 1
    fi
    if [ ! -f "$weights/model.safetensors" ]; then
        log "**評価見送り（$name）: 重みが無い $weights**"
        record_csv "$name" eval skip weights_missing
        return 1
    fi

    WEIGHTS="$weights" EVAL_CSV="$REPOSITORY_ROOT/compe/val/EVAL_TASKS.csv" \
        LABEL="quant_${name}" MAX_STEPS=300 N_EPISODES=1 \
        POLICY_IMAGE="$POLICY_IMAGE" \
        QUANT_SERVE_MODE="$serve_mode" QUANT_SERVE_SCOPE="$serve_scope" \
        bash solution/tools/eval/run_docker_eval.sh
    local status=$?

    if [ "$status" -ne 0 ]; then
        log "**評価失敗（$name, exit $status）。次の候補へ進む。**"
        record_csv "$name" eval fail "exit=$status"
        return 1
    fi
    record_csv "$name" eval ok ""
    return 0
}

# ---------------------------------------------------------------- 1候補の実行

run_one_candidate() {
    local name="$1"
    local ok=1

    case "$name" in
        C0)
            train_candidate C0 && ok=0 || ok=1
            [ "$ok" -eq 0 ] && eval_candidate C0 "$(checkpoint_path C0)" || true
            ;;
        C1)
            train_candidate C1 --optimizer-mode bnb8bit && ok=0 || ok=1
            [ "$ok" -eq 0 ] && eval_candidate C1 "$(checkpoint_path C1)" || true
            ;;
        C2)
            train_candidate C2 --vision-quant-mode bnb4bit && ok=0 || ok=1
            [ "$ok" -eq 0 ] && eval_candidate C2 "$(checkpoint_path C2)" || true
            ;;
        C3)
            train_candidate C3 --expert-quant-mode float8 && ok=0 || ok=1
            [ "$ok" -eq 0 ] && eval_candidate C3 "$(checkpoint_path C3)" || true
            ;;
        C4)
            # 学習不要。C0 のチェックポイントを推論時にだけ bnb で 4bit 化する。
            eval_candidate C4 "$(checkpoint_path C0)" bnb_vision4bit
            ok=$?
            ;;
        C5)
            # 学習不要。C0 のチェックポイントを推論時にだけ torchao で量子化する。
            eval_candidate C5 "$(checkpoint_path C0)" torchao_int8
            ok=$?
            ;;
        *)
            log "**未知の候補名（無視する）: $name**"
            return 1
            ;;
    esac

    if [ "$ok" -eq 0 ]; then
        RESULT_OF["$name"]="ok"
    else
        RESULT_OF["$name"]="fail"
    fi
    progress_note "候補 $name 完了" "$(tail -n 4 "$RESULTS_CSV" | sed 's/^/    /')"
}

# ---------------------------------------------------------------- 判断(claude -p)

# claude -p は判断だけを返す短命呼び出し。--tools "" で全ツールを無効化しているので、
# 何が返ってきても実行系には触れられない（安全策の要）。返ってきた候補名は
# ホワイトリスト（CANDIDATES_ALL かつ未試行）と突き合わせてからしか使わない。
JUDGE_SCHEMA='{"type":"object","properties":{"next":{"type":["string","null"]},"stop":{"type":"boolean"},"reason":{"type":"string"}},"required":["stop","reason"]}'

judge_next_candidate() {
    local remaining="$1"
    local history candidates_doc prompt raw next_val stop_val reason_val

    history="$(tail -n 30 "$RESULTS_CSV" 2>/dev/null)"
    candidates_doc="$(cat "$SWEEP_DIR/CANDIDATES.md" 2>/dev/null)"

    prompt="あなたはπ0.5量子化スイープの進行判断だけを行う。実行は一切しない。
残っている候補: ${remaining}
候補の説明:
${candidates_doc}

これまでの結果台帳（末尾30行）:
${history}

残り時間（秒）: $(time_left_seconds)

次にどの候補を試すべきか、優先順位を判断せよ。C0（baseline）が未試行なら必ず C0 を選ぶこと
（対照群が無いと他は比較できない）。もう試す価値が無い、または残り時間が足りないと判断したら
stop を true にせよ。"

    raw=$(claude -p "$prompt" --tools "" --output-format json --json-schema "$JUDGE_SCHEMA" 2>/dev/null)
    if [ -z "$raw" ]; then
        echo "FALLBACK||judge_call_empty"
        return
    fi

    next_val=$(echo "$raw" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    so = d.get('structured_output') or {}
    print(so.get('next') or '')
except Exception:
    print('')
" 2>/dev/null)
    stop_val=$(echo "$raw" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    so = d.get('structured_output') or {}
    print('1' if so.get('stop') else '0')
except Exception:
    print('0')
" 2>/dev/null)
    reason_val=$(echo "$raw" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    so = d.get('structured_output') or {}
    print(so.get('reason', ''))
except Exception:
    print('')
" 2>/dev/null)

    echo "${next_val}|${stop_val}|${reason_val}"
}

pick_next_candidate() {
    # judgment を試し、失敗・不正な返答・ホワイトリスト外なら固定順フォールバック
    # （CANDIDATES_ALL の並び順で、未試行の最初のもの）。
    local remaining
    remaining="$(remaining_candidates)"
    if [ -z "$remaining" ]; then
        echo "|1|全候補を試し終えた"
        return
    fi

    local judged next stop reason
    judged="$(judge_next_candidate "$remaining")"
    IFS='|' read -r next stop reason <<< "$judged"

    # ホワイトリスト＋未試行チェック。外れていたらフォールバック。
    local valid=0
    for c in $remaining; do
        [ "$c" = "$next" ] && valid=1
    done
    if [ "$valid" -ne 1 ]; then
        next="$(echo "$remaining" | awk '{print $1}')"
        reason="判断が無効/取得失敗のため固定順フォールバック（${reason:-理由なし}）"
        stop=0
    fi
    echo "${next}|${stop}|${reason}"
}

# ---------------------------------------------------------------- メインループ

# ---------------------------------------------------------------- 前提の確認

for bin in docker claude python3; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        log "**エラー: $bin が PATH に無い。tmux/nohup 環境の PATH を確認すること。**"
        exit 1
    fi
done
if [ "$(docker inspect -f '{{.State.Running}}' "$TRAIN_CONTAINER" 2>/dev/null)" != "true" ]; then
    log "**エラー: 学習コンテナ ($TRAIN_CONTAINER) が動いていない。**"
    log "  先に: docker run -d --gpus all --name $TRAIN_CONTAINER ... parc2026-train-pi05 sleep infinity"
    exit 1
fi
if ! docker image inspect "$POLICY_IMAGE" >/dev/null 2>&1; then
    log "**エラー: 評価用イメージが無い: $POLICY_IMAGE**"
    log "  docker build -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train"
    exit 1
fi

log "===== 量子化スイープ開始（上限 ${MAX_SWEEP_HOURS} 時間） ====="
progress_note "スイープ開始" "候補: ${CANDIDATES_ALL[*]} / steps=$STEPS / batch=$BATCH_SIZE / 上限=${MAX_SWEEP_HOURS}h"

while true; do
    left=$(time_left_seconds)
    if (( left <= 0 )); then
        log "**壁時計の上限（${MAX_SWEEP_HOURS}時間）に到達。強制終了する。**"
        progress_note "壁時計上限で終了" "この時点の結果は $RESULTS_CSV を参照"
        break
    fi

    decision="$(pick_next_candidate)"
    IFS='|' read -r next stop reason <<< "$decision"

    if [ "$stop" = "1" ] || [ -z "$next" ]; then
        log "終了判断: ${reason:-理由なし}"
        progress_note "判断により終了" "${reason:-理由なし}"
        break
    fi

    log "次の候補: $next（理由: ${reason:-なし}）"
    run_one_candidate "$next"
done

log "===== 量子化スイープ終了 ====="
log "結果: $RESULTS_CSV"
progress_note "スイープ終了" "$(cat "$RESULTS_CSV" | sed 's/^/    /')"
