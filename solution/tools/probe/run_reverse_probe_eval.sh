#!/usr/bin/env bash
# 016 推奨方式の評価: 順方向と逆向きのタスクを、同じ重み・同じ設定で測る。
#
# run_final_eval.sh は track3 に CSV の差し替え口を持たない（compe/t3 は assets から
# 組む）。本スクリプトはそれの薄い派生で、**自分の t3 assets（順＋逆）を評価コンテナの
# `compe/t3/assets/{bddl_files,init_files}/libero_t3` へ bind-mount して差し替える**
# 点だけが違う。イメージ再ビルドは不要（キットは assets をそのパスから読む）。
#
# 前置き（ホスト側で先に済ませること）:
#   1. t3 assets を組み立てる
#        python3 solution/tools/probe/_016_assemble_t3assets.py \
#            --forward-bddl work/probe_016/l10 --forward-init work/probe_016/l10 \
#            --reverse work/probe_016/reverse --out work/probe_016/t3assets \
#            --stem <s1> --stem <s2> ...
#   2. 本スクリプトを T3_ASSETS 指定で回す
#
# 使い方:
#   T3_ASSETS=$PWD/work/probe_016/t3assets \
#   WEIGHTS=$PWD/work/artifacts/trial1500/001500 \
#   MUJOCO_GL=egl TRACK_EPISODES=2 PARC_N_ACTION_STEPS=16 PARC_EMA_ALPHA=0.3 \
#   LABEL=t3_reverse_probe RECORD=1 \
#     bash solution/tools/probe/run_reverse_probe_eval.sh
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T3_ASSETS="${T3_ASSETS:?T3_ASSETS（組み立てた t3 assets ディレクトリ）が必要}"
[ -d "$T3_ASSETS/bddl_files/libero_t3" ] || { echo "T3_ASSETS に bddl_files/libero_t3 が無い: $T3_ASSETS" >&2; exit 2; }
[ -d "$T3_ASSETS/init_files/libero_t3" ] || { echo "T3_ASSETS に init_files/libero_t3 が無い: $T3_ASSETS" >&2; exit 2; }

TRACK="${1:-track3}"
case "$TRACK" in track1|track2|track3) ;; *) echo "トラックは track1/track2/track3" >&2; exit 2;; esac
[ "$TRACK" = "track3" ] || { echo "この wrapper は track3 専用（順／逆 assets の差し替え）" >&2; exit 2; }

TRACK_EPISODES="${TRACK_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-300}"
WEIGHTS="${WEIGHTS:-$REPOSITORY_ROOT/work/artifacts/trial1500/001500}"
KIT_IMAGE="${KIT_IMAGE:-parc2026-final}"
POLICY_IMAGE="${POLICY_IMAGE:-parc2026-train-pi05}"
MUJOCO_GL="${MUJOCO_GL:-egl}"   # このマシンは RTX6000（Pass-Through）なので egl

KIT_ROOT="${KIT_ROOT:-$(cd "$REPOSITORY_ROOT/../b_PARC2026_final" 2>/dev/null && pwd)}"
[ -n "$KIT_ROOT" ] || { echo "KIT_ROOT を渡すか、リポジトリの隣に b_PARC2026_final が要る" >&2; exit 1; }

for image in "$KIT_IMAGE" "$POLICY_IMAGE"; do
    docker image inspect "$image" >/dev/null 2>&1 || { echo "イメージ $image が無い" >&2; exit 1; }
done
[ -d "$WEIGHTS/vlm" ] || { echo "$WEIGHTS に vlm/ が無い（pi05_rp3000 から copy すること）" >&2; exit 1; }

POLICY_IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$POLICY_IMAGE" | cut -d: -f2 | cut -c1-12)"
SERVER_IMAGE="${POLICY_IMAGE}-server-${POLICY_IMAGE_ID}"
if ! docker image inspect "$SERVER_IMAGE" >/dev/null 2>&1; then
    printf 'FROM %s\nRUN pip install --no-cache-dir "fastapi>=0.68" "uvicorn>=0.15" "msgpack>=1.0"\n' \
        "$POLICY_IMAGE" | docker build -t "$SERVER_IMAGE" -f - "$REPOSITORY_ROOT" >/dev/null
fi

label="$TRACK"
[ -n "${LABEL:-}" ] && label="${label}_${LABEL}"
OUTPUT_DIR="$REPOSITORY_ROOT/work/final_eval/$(date +%m%d_%H%M%S)_${label}"
mkdir -p "$OUTPUT_DIR/results"
exec > >(tee -a "$OUTPUT_DIR/run.log") 2>&1

echo "== 逆向きプロブ評価  本数=$TRACK_EPISODES  打ち切り=$MAX_STEPS"
echo "   重み: $WEIGHTS"
echo "   t3 assets（差し替え）: $T3_ASSETS"
echo "   出力: $OUTPUT_DIR"

# t3 assets の差し替えマウント。キットの symlink が指すパスに bind する。
t3_bddl_in=$T3_ASSETS/bddl_files/libero_t3
t3_init_in=$T3_ASSETS/init_files/libero_t3

# ---------------------------------------------------------- 方策側
NETWORK="parc_revprobe_$$"
POLICY_HOST="parc_revprobe_policy_$$"
docker network create "$NETWORK" >/dev/null
cleanup() {
    docker logs "$POLICY_HOST" > "$OUTPUT_DIR/server.log" 2>&1 || true
    docker rm -f "$POLICY_HOST" >/dev/null 2>&1 || true
    docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

policy_environment=()
for name in PARC_N_ACTION_STEPS PARC_NUM_INFERENCE_STEPS PARC_SAMPLES \
            PARC_EMA_ALPHA PARC_BLEND_STEPS PARC_SEED PARC_NORMALIZE_INSTRUCTION \
            PARC_GAIN_TRANSLATION PARC_GAIN_ROTATION PARC_GEO_IMAGE_SIZE \
            PARC_BLOCK_SCENE_CAMERA; do
    [ -n "${!name:-}" ] && policy_environment+=(-e "$name=${!name}")
done

echo "-- 方策サーバー起動"
docker run -d --name "$POLICY_HOST" --gpus all --network "$NETWORK" \
    -v "$REPOSITORY_ROOT/solution/policy_server.py:/app/policy_server.py:ro" \
    -v "$REPOSITORY_ROOT/solution/geometry_graft.py:/app/geometry_graft.py:ro" \
    -v "$REPOSITORY_ROOT/work/artifacts/vendor/vggt:/app/vendor/vggt:ro" \
    -v "$WEIGHTS:/app/model_weights:ro" \
    -v "$HOME/.cache/huggingface:/hf_home" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "${policy_environment[@]}" \
    "$SERVER_IMAGE" python /app/policy_server.py --port 8000 >/dev/null

for _ in $(seq 1 300); do
    if docker exec "$POLICY_HOST" python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" 2>/dev/null; then
        echo "   /health OK"; break
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$POLICY_HOST" 2>/dev/null)" != "true" ]; then
        docker logs "$POLICY_HOST" > "$OUTPUT_DIR/server.log" 2>&1 || true
        echo "   方策サーバーが落ちた" >&2; exit 1
    fi
    sleep 2
done
docker logs "$POLICY_HOST" > "$OUTPUT_DIR/server.log" 2>&1 || true
grep -m1 "\[parc\] policy_server" "$OUTPUT_DIR/server.log" || true

# ---------------------------------------------------------- 評価側
record_environment=()
if [ "${RECORD:-1}" != "0" ]; then
    mkdir -p "$OUTPUT_DIR/videos"
    record_environment=(-e T3_VIDEO_DIR=/workspace/videos -v "$OUTPUT_DIR/videos:/workspace/videos")
fi
SCORER_PATCH="$REPOSITORY_ROOT/solution/tools/eval/kit_patches/scorer.py"
scorer_mount=(); [ -f "$SCORER_PATCH" ] && scorer_mount=(-v "$SCORER_PATCH:/workspace/pipeline/scorer.py:ro")
ROLLOUT_PATCH="$REPOSITORY_ROOT/solution/tools/eval/kit_patches/rollout.py"
rollout_mount=()
[ -f "$ROLLOUT_PATCH" ] && rollout_mount=(-v "$ROLLOUT_PATCH:/workspace/pipeline/rollout.py:ro"
    -e PARC_OBJECT_DISP_JSONL=/workspace/results/object_disp.jsonl)
# 自分の t3 assets をイメージ内の assets へ差し替え
t3_mount=(-v "$t3_bddl_in:/workspace/compe/t3/assets/bddl_files/libero_t3:ro"
          -v "$t3_init_in:/workspace/compe/t3/assets/init_files/libero_t3:ro")

echo "-- 評価を回す（t3 assets を差し替え済み）"
docker run --rm --gpus all --network "$NETWORK" --ulimit core=0 \
    -e "MUJOCO_GL=$MUJOCO_GL" \
    "${scorer_mount[@]}" "${rollout_mount[@]}" "${t3_mount[@]}" "${record_environment[@]}" \
    -v "$OUTPUT_DIR/results:/workspace/results" \
    "$KIT_IMAGE" \
    python -m pipeline --server-url "http://$POLICY_HOST:8000" \
        --track "$TRACK" --n-episodes "$TRACK_EPISODES" --max-steps "$MAX_STEPS" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

docker run --rm -v "$OUTPUT_DIR:/out" --entrypoint chown "$KIT_IMAGE" \
    -R "$(id -u):$(id -g)" /out || echo "   持ち主を戻せなかった"

echo
echo "== 結果"
docker logs "$POLICY_HOST" 2>&1 | grep -c "POST /act" | sed 's/^/   合計ステップ数（POST \/act）: /' || true
echo "   json: $OUTPUT_DIR/results/"
python3 "$REPOSITORY_ROOT/solution/tools/eval/score_composite.py" --by-category "$OUTPUT_DIR" \
    || echo "   合計点を付け直せなかった（あとから打てる）"