#!/usr/bin/env bash
# 配布データセットを課題ごとに均等に間引く（`npz_to_lerobot.py subset` をコンテナで回す）。
#
#   ROOT=work/artifacts/libero_plus OUT_DIR=work/artifacts \
#   bash solution/tools/track3_reversal/run_lerobot_subset.sh libero_plus_sub 75
#
# **$OUT_DIR からの相対で指す。**第 1 引数が出来上がりの名前、第 2 引数が 1 課題あたりの本数。
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
OUT_DIR="${OUT_DIR:-$REPO/work/artifacts}"
ROOT="${ROOT:?ROOT に元のデータセットを渡すこと}"
IMAGE="${IMAGE:-parc2026-train-pi05}"
NAME="${1:?出来上がりの名前を渡すこと}"
PER_TASK="${2:-75}"
SRC_REPO_ID="${SRC_REPO_ID:-lerobot/libero_plus}"

case "$ROOT" in /*) ;; *) ROOT="$REPO/$ROOT" ;; esac

docker run --rm \
    -v "$REPO/solution/tools:/host/tools:ro" \
    -v "$ROOT:/src:ro" -v "$OUT_DIR:/out" \
    --entrypoint python "$IMAGE" \
    /host/tools/scripted_demo/npz_to_lerobot.py subset \
        --root /src --output "/out/$NAME" --repo-id "$SRC_REPO_ID" --per-task "$PER_TASK" \
    || exit 1

# **所有を戻す。**コンテナが root で書くので、戻さないと消せない
docker run --rm -v "$OUT_DIR:/out" --entrypoint chown "$IMAGE" \
    -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
echo "=== 出来上がり: $OUT_DIR/$NAME ==="
