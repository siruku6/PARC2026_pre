#!/usr/bin/env bash
# 逆向きプローブが書いた npz を LeRobot 形式へ変換して、読めることまで確かめる。
#
#   bash solution/tools/track3_reversal/run_npz_to_lerobot.sh [npz-dir] [出力] [repo-id]
#
# **シムの画像（parc2026-final）と lerobot（parc2026-train-pi05）は別の画像に
# 入っている。**前者に lerobot は無く、後者に LIBERO は無い。だから
# 「npz を書く」と「npz を変換する」は必ず 2 回に分かれる。
#
# 変換で 67 分の 1 に縮む（npz は画素そのまま・変換後は動画）。運ぶのは変換後。
#
# 要るものの一覧は docs/reference/29_training_data_requirements.md。
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
OUT="${OUT:-$REPO/work/probe_017}"
NPZ_DIR="${1:-reverse_npz}"
DEST="${2:-reverse_lerobot}"
REPO_ID="${3:-local/t3_reverse_book}"
IMAGE="${IMAGE:-parc2026-train-pi05}"

run() {
  docker run --rm \
    -v "$REPO/solution/tools:/host/tools:ro" -v "$OUT:/data" \
    --entrypoint python "$IMAGE" \
    /host/tools/scripted_demo/npz_to_lerobot.py "$@"
}

run build --npz-dir "/data/$NPZ_DIR" --output "/data/$DEST" --repo-id "$REPO_ID"
# **揃っていることと、学習が読めることは別である。**動画の符号化が壊れていても
# build は通る。読み出しはここで初めて落ちる。
run inspect --root "/data/$DEST" --repo-id "$REPO_ID"

docker run --rm -v "$OUT:/data" --entrypoint chown "$IMAGE" \
  -R "$(id -u):$(id -g)" /data >/dev/null 2>&1 || true
echo "=== 出力: $OUT/$DEST ==="
