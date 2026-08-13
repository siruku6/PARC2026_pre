#!/usr/bin/env bash
# 学習中のチェックポイントを、新しい方から KEEP 個だけ残して間引き続ける。
#
# LeRobot の save_checkpoint は古いチェックポイントを消さない（GitHub上の
# train_utils.py で確認済み）。pi05 は1個 11GB あるため、save_freq=500 の
# ままだと数十個で数百GBに達し、ディスクが尽きて学習が落ちる。
#
# 学習コンテナとは別のターミナルで、これを並行して回すこと。
#
#   solution/train/prune_checkpoints.sh work/train/<run-name> [KEEP=2]
#
# KEEP は既定 2。1 にすると「保存中の1個」を消してしまう競合が起きうるので、
# 2 未満にはしないこと。Ctrl+C で止めれば、その時点の最新チェックポイントは
# 残ったままになる。
set -euo pipefail

RUN_DIR="${1:?使い方: prune_checkpoints.sh work/train/<run-name> [KEEP]}"
KEEP="${2:-2}"
CKPT_DIR="$RUN_DIR/checkpoints"

echo "[prune] 監視対象: $CKPT_DIR（最新 $KEEP 個を残す）"

while true; do
    if [ -d "$CKPT_DIR" ]; then
        # 6桁ゼロ埋めの数字ディレクトリだけを対象にする（pretrained_model 等は無視）。
        mapfile -t dirs < <(find "$CKPT_DIR" -maxdepth 1 -type d -regextype posix-extended \
            -regex '.*/[0-9]+' | sort)
        count=${#dirs[@]}
        if (( count > KEEP )); then
            for ((i = 0; i < count - KEEP; i++)); do
                echo "[prune] $(date '+%H:%M:%S') 削除: ${dirs[$i]}（$(du -sh "${dirs[$i]}" 2>/dev/null | cut -f1)）"
                rm -rf "${dirs[$i]}"
            done
        fi
    fi
    sleep 30
done
