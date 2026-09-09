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
#   MODES="full stripped"   full=配布のまま／stripped=競合する物を消す
#   KEEP_LANDMARKS=1        指示文が名指しする物を残す（既定は消す）
#   RECORD=1                mp4 を <出力>/videos へ残す（**既定は録らない**）
#   PARC_*                  推論の設定値。**評価と同じ設定で測るために渡す**
#   MUJOCO_GL=egl           このマシンは EGL が通る（描画が 16 倍速い）
set -euo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
W="${1:?重みのディレクトリを渡すこと}"
OUT="${2:-work/probe_grasp/$(date +%m%d_%H%M%S)}"
case "$W" in /*) ;; *) W="$REPO/$W" ;; esac
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
mkdir -p "$OUT"

# **重みに vlm/ が無いと方策サーバーは 5 秒で落ちる**（`run_final_eval.sh` の教訓）。
# 読み込みに 1〜2 分かけてから落ちるのを避けるため、起動より前に確かめる。
if [ ! -d "$W/vlm" ]; then
    echo "$W に vlm/ が無い。方策サーバーは 5 秒で落ちる" >&2
    echo "（pi05_rp3000 から vlm/ を symlink ではなく copy で持ってくること）" >&2
    exit 1
fi

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
        "$POLICY_IMAGE" | docker build -t "$SERVER_IMAGE" - >/dev/null
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

# **推論の設定値を評価と同じ形で渡せるようにする**（`run_final_eval.sh` と同じ一覧）。
# これが無いと、説明しようとしている評価とは別の設定の方策を測ることになる。
policy_environment=()
for name in PARC_N_ACTION_STEPS PARC_NUM_INFERENCE_STEPS PARC_SAMPLES \
            PARC_EMA_ALPHA PARC_BLEND_STEPS PARC_SEED PARC_NORMALIZE_INSTRUCTION \
            PARC_GAIN_TRANSLATION PARC_GAIN_ROTATION PARC_GEO_IMAGE_SIZE; do
    if [ -n "${!name:-}" ]; then policy_environment+=(-e "$name=${!name}"); fi
done

echo "-- 方策サーバーを起動する（重みの読み込みに 1〜2 分）: $W"
docker run -d --name "$POLICY_HOST" --network-alias policy --gpus all --network "$NETWORK" \
    -v "$REPO/solution/policy_server.py:/app/policy_server.py:ro" \
    -v "$W:/app/model_weights:ro" \
    -v "$HOME/.cache/huggingface:/hf_home" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "${policy_environment[@]}" \
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
    python -u /app/grasp_rollout.py \
        ${EPISODES:+--episodes "$EPISODES"} ${STEPS:+--steps "$STEPS"} \
        ${TASKS:+--tasks $TASKS} ${MODES:+--modes $MODES} \
        ${KEEP_LANDMARKS:+--keep-landmarks} \
        ${RECORD:+--video-dir /out/videos} \
    2>&1 | tee "$OUT/rollout.log"

docker run --rm -v "$OUT:/out" --entrypoint chown "$KIT_IMAGE" \
    -R "$(id -u):$(id -g)" /out || echo "   持ち主を戻せなかった"
echo "== 出力: $OUT"
