#!/usr/bin/env bash
# 学習中のチェックポイントを、新しい方から KEEP 個だけ残して間引き続ける。
#
# LeRobot の save_checkpoint は古いチェックポイントを消さない（GitHub上の
# train_utils.py で確認済み）。pi05 は1個 11GB（train_expert_only）〜
# 25GB（フルパラメータ。2026-08-14 に RunPod で実測）あるため、save_freq=500 の
# ままだと数十個で数百GBに達し、ディスクが尽きて学習が落ちる。
#
# 使い方は3通り。**どれも同じスクリプトで、--once を付けるかどうかだけが違う。**
#
#   # 1. ずっと居座らせる（既定。ターミナルを占有する）
#   solution/train/prune_checkpoints.sh work/train/<run-name> 2
#
#   # 2. 端末から切り離して放っておく（学習と一緒に流すならこれ）
#   nohup solution/train/prune_checkpoints.sh work/train/<run-name> 2 \
#       --log work/train/<run-name>/prune.log >/dev/null 2>&1 &
#
#   # 3. cron から1分ごとに1回だけ点検させる（常駐プロセスを持たない形）
#   * * * * * /home/ubuntu/src/PARC2026_pre/solution/train/prune_checkpoints.sh \
#       /home/ubuntu/src/PARC2026_pre/work/train/<run-name> 2 --once \
#       --log /home/ubuntu/src/PARC2026_pre/work/train/<run-name>/prune.log
#
# 引数:
#   $1              run ディレクトリ（この下の checkpoints/ を見る）
#   $2              KEEP（既定 2）。**2 未満にはしないこと** —— 保存中の1個を
#                   消してしまう競合と、LeRobot が張る checkpoints/last の
#                   リンク切れが起きうる
#   --once          1回だけ点検して終わる（cron 向け）
#   --interval SEC  点検の間隔（既定 30。--once では無視される）
#   --log FILE      ログの追記先（既定は標準出力）
#   --dry-run       消さずに、消す対象だけ出す
#
# 同じ run ディレクトリに対する二重起動は flock で弾く（cron の実行が
# 前回の削除中に重なっても安全）。止めたいときは常駐版なら Ctrl+C か
# `pkill -f prune_checkpoints.sh`、cron 版なら crontab の行を消す。
# どちらの止め方でも、その時点の最新チェックポイントは残ったままになる。
set -euo pipefail

RUN_DIR=""
KEEP=2
ONCE=false
INTERVAL=30
LOG_FILE=""
DRY_RUN=false

while (( $# > 0 )); do
    case "$1" in
        --once)     ONCE=true; shift ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --interval) INTERVAL="${2:?--interval に秒数が要る}"; shift 2 ;;
        --log)      LOG_FILE="${2:?--log にファイル名が要る}"; shift 2 ;;
        -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
        -*)         echo "不明な引数: $1" >&2; exit 2 ;;
        *)
            if [ -z "$RUN_DIR" ]; then RUN_DIR="$1"; else KEEP="$1"; fi
            shift ;;
    esac
done

if [ -z "$RUN_DIR" ]; then
    echo "使い方: prune_checkpoints.sh <run-dir> [KEEP] [--once|--interval SEC|--log FILE|--dry-run]" >&2
    exit 2
fi
if ! [[ "$KEEP" =~ ^[0-9]+$ ]] || (( KEEP < 1 )); then
    echo "KEEP は 1 以上の整数で指定すること（推奨は 2 以上）: $KEEP" >&2
    exit 2
fi
if (( KEEP < 2 )); then
    echo "[prune] **警告: KEEP=1 は保存中のチェックポイントを消す危険がある**" >&2
fi

# ログ先を決める。--log を付けたときは、以降の出力を丸ごとそこへ追記する
# （nohup / cron から呼ぶとき、標準出力の行き先を気にしなくて済むように）。
if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec >>"$LOG_FILE" 2>&1
fi

log() { echo "[prune] $(date '+%F %T') $*"; }

# 絶対パスにしてからロック名を作る。相対パスと絶対パスで別ロックになると
# 二重起動を弾けないため（cron は絶対パス、手打ちは相対パスになりがち）。
RUN_DIR_ABS="$(cd "$(dirname "$RUN_DIR")" 2>/dev/null && pwd)/$(basename "$RUN_DIR")" \
    || RUN_DIR_ABS="$RUN_DIR"
CKPT_DIR="$RUN_DIR_ABS/checkpoints"

if command -v flock >/dev/null 2>&1; then
    LOCK_FILE="${TMPDIR:-/tmp}/prune_checkpoints$(echo "$RUN_DIR_ABS" | tr -c 'A-Za-z0-9' '_').lock"
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        # cron 版が1分ごとに来るので、ここは黙って抜ける（ログを埋めない）。
        exit 0
    fi
fi

prune_once() {
    [ -d "$CKPT_DIR" ] || return 0

    # 6桁ゼロ埋めの数字ディレクトリだけを対象にする。pretrained_model や、
    # LeRobot が張る checkpoints/last（シンボリックリンク。-type d では
    # 拾われない）は対象外になる。
    local dirs=() count=0 i
    mapfile -t dirs < <(find "$CKPT_DIR" -maxdepth 1 -type d -regextype posix-extended \
        -regex '.*/[0-9]+' | sort)
    count=${#dirs[@]}
    (( count > KEEP )) || return 0

    for ((i = 0; i < count - KEEP; i++)); do
        local size
        size="$(du -sh "${dirs[$i]}" 2>/dev/null | cut -f1)"
        if [ "$DRY_RUN" = true ]; then
            log "（dry-run。消さない）削除対象: ${dirs[$i]}（${size:-?}）"
        else
            log "削除: ${dirs[$i]}（${size:-?}）"
            rm -rf "${dirs[$i]}"
        fi
    done
    if [ "$DRY_RUN" = true ]; then return 0; fi
    log "残り $KEEP 個 / ディスク空き $(df -h --output=avail "$CKPT_DIR" 2>/dev/null | tail -1 | tr -d ' ')"
}

if [ "$ONCE" = true ]; then
    prune_once
    exit 0
fi

log "監視開始: $CKPT_DIR（最新 $KEEP 個を残す・${INTERVAL}秒ごと）"
while true; do
    prune_once
    sleep "$INTERVAL"
done
