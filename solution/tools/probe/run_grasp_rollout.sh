#!/usr/bin/env bash
# `grasp_rollout.py` を 2 コンテナ構成で回す（`run_final_eval.sh` と同じ理由で 2 つ要る）。
#
#   parc2026-train-pi05 … GPU torch と LeRobot        → 方策サーバー
#   parc2026-final      … LIBERO とキットの資産       → シム
#
#   bash solution/tools/probe/run_grasp_rollout.sh <重み> [出力ディレクトリ]
#
# 環境変数
#   EPISODES=2   1 条件あたりの本数
#   STEPS=300    打ち切り
#   TASKS=""     bddl の stem を空白区切りで（省略すると 4 課題すべて）
#   MODES="full stripped"   full=配布のまま／stripped=邪魔な物を消す
#   MUJOCO_GL=egl           このマシンは EGL が通る（描画が 16 倍速い）
set -euo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
W="${1:?重みのディレクトリを渡すこと}"
OUT="${2:-work/probe_grasp/$(date +%m%d_%H%M%S)}"
case "$W" in /*) ;; *) W="$REPO/$W" ;; esac
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
mkdir -p "$OUT"

KIT_IMAGE="${KIT_IMAGE:-parc2026-final}"
POLICY_IMAGE="${POLICY_IMAGE:-parc2026-train-pi05}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

# **HTTP サーバーの依存 3 つを足した派生イメージ**を使う（`run_final_eval.sh` と同じ）。
# タグに元イメージの ID を含めるので、元を作り直したら別タグになる。
POLICY_IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$POLICY_IMAGE" | cut -d: -f2 | cut -c1-12)"
SERVER_IMAGE="${POLICY_IMAGE}-server-${POLICY_IMAGE_ID}"
if ! docker image inspect "$SERVER_IMAGE" >/dev/null 2>&1; then
    echo "-- $SERVER_IMAGE を作る"
    printf 'FROM %s\nRUN pip install --no-cache-dir "fastapi>=0.68" "uvicorn>=0.15" "msgpack>=1.0"\n' \
        "$POLICY_IMAGE" | docker build -t "$SERVER_IMAGE" -f - . >/dev/null
fi

NETWORK="parc_grasp_$$"
POLICY_HOST="parc_grasp_policy_$$"
docker network create "$NETWORK" >/dev/null
cleanup() {
    docker logs "$POLICY_HOST" > "$OUT/server.log" 2>&1 || true
    docker rm -f "$POLICY_HOST" >/dev/null 2>&1 || true
    docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "-- 方策サーバーを起動する（重みの読み込みに 1〜2 分）: $W"
docker run -d --name "$POLICY_HOST" --network-alias policy --gpus all --network "$NETWORK" \
    -v "$REPO/solution/policy_server.py:/app/policy_server.py:ro" \
    -v "$W:/app/model_weights:ro" \
    -v "$HOME/.cache/huggingface:/hf_home" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$SERVER_IMAGE" python /app/policy_server.py --port 8000 >/dev/null

for _ in $(seq 1 300); do
    if docker exec "$POLICY_HOST" python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" \
        2>/dev/null; then echo "   /health OK"; break; fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$POLICY_HOST" 2>/dev/null)" != "true" ]; then
        echo "   方策サーバーが落ちた。ログを $OUT/server.log に出す" >&2
        docker logs "$POLICY_HOST" > "$OUT/server.log" 2>&1 || true
        exit 1
    fi
    sleep 2
done

echo "-- シムを回す（MUJOCO_GL=$MUJOCO_GL）"
docker run --rm --gpus all --network "$NETWORK" --ulimit core=0 \
    -e "MUJOCO_GL=$MUJOCO_GL" -e LIBERO_ROOT=/workspace/LIBERO-plus \
    -e PYTHONPATH=/workspace/LIBERO-plus:/workspace:/workspace/compe \
    -v "$REPO/solution/tools/probe:/app:ro" -v "$OUT:/out" \
    -w /workspace "$KIT_IMAGE" \
    python -u /app/grasp_rollout.py --server http://policy:8000 \
        --out /out/grasp_rollout.json \
        --episodes "${EPISODES:-2}" --steps "${STEPS:-300}" \
        ${TASKS:+--tasks $TASKS} --modes ${MODES:-full stripped} \
    2>&1 | tee "$OUT/rollout.log"

docker run --rm -v "$OUT:/out" --entrypoint chown "$KIT_IMAGE" \
    -R "$(id -u):$(id -g)" /out || echo "   持ち主を戻せなかった"
echo "== 出力: $OUT"
