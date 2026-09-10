#!/usr/bin/env bash
# `divergence.py` を 2 コンテナ構成で回す（`run_grasp_rollout.sh` と同じ理由で 2 つ要る）。
#
#   bash solution/tools/probe/run_divergence.sh <重み> [出力ディレクトリ]
#
# 環境変数
#   EPISODES=2   1 課題あたりに並べる本数（初期状態の番号ごとに 1 本）
#   PAIRS=""     "stem=ホスト側の npz ディレクトリ" を空白区切りで。省略すると
#                **v1 の 3 課題**（proto20 が学習に使ったデータ）を並べる
#   （指示文は bddl の :language 1 つだけ。記録も同じ文である）
#   TAIL_STEPS=50  記録の長さに足して回す歩数
#   PARC_*       推論の設定値。**評価と同じ設定で測るために渡す**
#   MUJOCO_GL=egl
set -euo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
W="${1:?重みのディレクトリを渡すこと}"
OUT="${2:-work/probe_diverge/$(date +%m%d_%H%M%S)}"
case "$W" in /*) ;; *) W="$REPO/$W" ;; esac
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
mkdir -p "$OUT"

# **既定は v2 の 3 課題**（`work/probe_rot2/*`）。
# **v1（`work/probe_017*`）は 9/11 に消した**——v1 で学習した重みはもう測らないと決めた
# （[knowledge/22](../../../docs/knowledge/22_reverse_data_interferes_with_same_objects.md)）。
# **v1 を並べ直すなら生成からやり直しになる。**
: "${PAIRS:=take_the_yellow_book_off_the_shelf_and_put_it_on_the_table=work/probe_rot2/book/reverse_npz_book
take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table=work/probe_rot2/soup/reverse_npz_soup
take_the_red_mug_off_the_left_plate_and_put_it_on_the_table=work/probe_rot2/mug/reverse_npz_mug}"

if [ ! -d "$W/vlm" ]; then
    echo "$W に vlm/ が無い。方策サーバーは 5 秒で落ちる" >&2
    exit 1
fi

KIT_IMAGE="${KIT_IMAGE:-parc2026-final}"
POLICY_IMAGE="${POLICY_IMAGE:-parc2026-train-pi05}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

POLICY_IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$POLICY_IMAGE" | cut -d: -f2 | cut -c1-12)"
SERVER_IMAGE="${POLICY_IMAGE}-server-${POLICY_IMAGE_ID}"
if ! docker image inspect "$SERVER_IMAGE" >/dev/null 2>&1; then
    echo "-- $SERVER_IMAGE を作る"
    printf 'FROM %s\nRUN pip install --no-cache-dir "fastapi>=0.68" "uvicorn>=0.15" "msgpack>=1.0"\n' \
        "$POLICY_IMAGE" | docker build -t "$SERVER_IMAGE" - >/dev/null
fi

NETWORK="parc_div_$$"
POLICY_HOST="parc_div_policy_$$"
docker network create "$NETWORK" >/dev/null
cleanup() {
    docker logs "$POLICY_HOST" > "$OUT/server.log" 2>&1 || true
    docker rm -f "$POLICY_HOST" >/dev/null 2>&1 || true
    docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

policy_environment=()
for name in PARC_N_ACTION_STEPS PARC_NUM_INFERENCE_STEPS PARC_SAMPLES PARC_RESET_SEED \
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

# **記録した npz をコンテナへ見せる。**ディレクトリごとに read-only で mount し、
# `--pair` にはコンテナ側の道を渡す。
mounts=() pairs=()
index=0
for pair in $PAIRS; do
    stem="${pair%%=*}"; host="${pair#*=}"
    case "$host" in /*) ;; *) host="$REPO/$host" ;; esac
    [ -d "$host" ] || { echo "**$host が無い**" >&2; exit 1; }
    mounts+=(-v "$host:/rec/$index:ro")
    pairs+=(--pair "$stem=/rec/$index")
    index=$((index + 1))
done

echo "-- シムを回す（MUJOCO_GL=$MUJOCO_GL・$index 課題）"
docker run --rm --gpus all --network "$NETWORK" --ulimit core=0 \
    -e "MUJOCO_GL=$MUJOCO_GL" -e LIBERO_ROOT=/workspace/LIBERO-plus \
    -e PYTHONPATH=/workspace/LIBERO-plus:/workspace:/workspace/compe \
    -v "$REPO/solution/tools/probe:/app:ro" -v "$OUT:/out" "${mounts[@]}" \
    -w /workspace "$KIT_IMAGE" \
    python -u /app/divergence.py "${pairs[@]}" \
        ${EPISODES:+--episodes "$EPISODES"} \
        ${TAIL_STEPS:+--tail-steps "$TAIL_STEPS"} \
    2>&1 | tee "$OUT/divergence.log"

docker run --rm -v "$OUT:/out" --entrypoint chown "$KIT_IMAGE" \
    -R "$(id -u):$(id -g)" /out || echo "   持ち主を戻せなかった"
echo "== 出力: $OUT"
