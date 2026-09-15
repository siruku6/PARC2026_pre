#!/usr/bin/env bash
# 段階0（032番）専用: MolmoAct2-LIBERO-LeRobot を track2 で測る。
#
# **`run_final_eval.sh` を複製した理由は 2 つ**——共有中の本体は
# 稼働中の評価コンテナ 2 本から直にマウントされているため触らない、
# かつ molmoact2 には次の 2 点で本体と違う扱いが要るため。
#
#   1. `WEIGHTS/vlm` チェックを飛ばす（molmoact2 は vlm/ 同梱の仕組みを使わない）
#   2. `parc2026-train-pi05` イメージに焼き込み済みの
#      `/opt/lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py`
#      （重みを2回読む素の実装）を、`solution/tools/eval/lerobot_patches/` の
#      直した版で上書きマウントする
#      ——`work/artifacts/vendor/lerobot/...` は最終提出 zip 用の別コピーで、
#      ローカル評価のイメージには一切効かない（2026-09-15 に確認）。
#      **`work/` は git の管理外なので、直した版は `solution/` 側に置いてある**
#      （9/15。vendor 側だけに置いていた前回の直しは、実際には入っていなかった）。
#
# 使い方
#
#   WEIGHTS=$PWD/work/artifacts/molmoact2_libero_lerobot \
#     bash solution/tools/eval/run_molmoact2_stage0.sh track2
#
# 環境変数は run_final_eval.sh と同じ（TRACK_EPISODES・MAX_STEPS・LABEL・
# MUJOCO_GL・TASKS_CSV・PARC_* など）。SERVER_PY の既定はこの worktree の
# policy_server.py（molmoact2 の分岐入り）。
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$REPOSITORY_ROOT/../b_PARC2026_final" && pwd)}"
cd "$REPOSITORY_ROOT"

TRACK="${1:-track2}"
case "$TRACK" in
    track1 | track2 | track3) ;;
    *) echo "トラックは track1 / track2 / track3 のいずれか（渡された値: $TRACK）" >&2; exit 2 ;;
esac

TRACK_EPISODES="${TRACK_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-300}"
WEIGHTS="${WEIGHTS:?WEIGHTS を渡すこと（例: work/artifacts/molmoact2_libero_lerobot）}"
SERVER_PY="${SERVER_PY:-$REPOSITORY_ROOT/solution/policy_server.py}"
KIT_IMAGE="${KIT_IMAGE:-parc2026-final}"
POLICY_IMAGE="${POLICY_IMAGE:-parc2026-train-pi05}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
MOLMOACT2_MODELING_PATCH="${MOLMOACT2_MODELING_PATCH:-$REPOSITORY_ROOT/solution/tools/eval/lerobot_patches/modeling_molmoact2.py}"

if [ ! -f "$MOLMOACT2_MODELING_PATCH" ]; then
    echo "直した modeling_molmoact2.py が無い: $MOLMOACT2_MODELING_PATCH" >&2
    exit 2
fi
if [ ! -d "$WEIGHTS/molmoact2_base_small" ]; then
    echo "$WEIGHTS に molmoact2_base_small/ が無い（config・トークナイザ・norm_stats.json）。" >&2
    echo "  032 の 0-2（huggingface-cli download allenai/MolmoAct2-LIBERO ...）を先に打つこと。" >&2
    exit 2
fi

if [ -n "${TASKS_CSV:-}" ]; then
    case "$TASKS_CSV" in
        /*) ;;
        *) TASKS_CSV="$REPOSITORY_ROOT/$TASKS_CSV" ;;
    esac
    if [ ! -f "$TASKS_CSV" ]; then
        echo "TASKS_CSV が見つからない: $TASKS_CSV" >&2
        exit 2
    fi
    TASKS_CSV="$(cd "$(dirname "$TASKS_CSV")" && pwd)/$(basename "$TASKS_CSV")"
fi

label="${TRACK}_molmoact2_stage0"
[ -n "${LABEL:-}" ] && label="${label}_${LABEL}"
OUTPUT_DIR="$REPOSITORY_ROOT/work/final_eval/$(date +%m%d_%H%M%S)_${label}"
mkdir -p "$OUTPUT_DIR/results"
printf '{"track": "%s", "episodes": %s, "max_steps": %s, "mujoco_gl": "%s", "tasks_csv": "%s", "weights": "%s"}\n' \
    "$TRACK" "$TRACK_EPISODES" "$MAX_STEPS" "$MUJOCO_GL" "${TASKS_CSV:-}" "$WEIGHTS" > "$OUTPUT_DIR/eval_conditions.json"

exec > >(tee -a "$OUTPUT_DIR/run.log") 2>&1

for image in "$KIT_IMAGE" "$POLICY_IMAGE"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "イメージ $image が無い" >&2
        exit 1
    fi
done

POLICY_IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$POLICY_IMAGE" | cut -d: -f2 | cut -c1-12)"
SERVER_IMAGE="${POLICY_IMAGE}-server-${POLICY_IMAGE_ID}"
if ! docker image inspect "$SERVER_IMAGE" >/dev/null 2>&1; then
    echo "-- $SERVER_IMAGE を作る（HTTP サーバーの依存 3 つを足すだけ）"
    printf 'FROM %s\nRUN pip install --no-cache-dir "fastapi>=0.68" "uvicorn>=0.15" "msgpack>=1.0"\n' \
        "$POLICY_IMAGE" | docker build -t "$SERVER_IMAGE" -f - . >/dev/null
fi

echo "== 段階0評価  トラック=$TRACK  本数=$TRACK_EPISODES  打ち切り=$MAX_STEPS  描画=$MUJOCO_GL"
echo "   重み: $WEIGHTS"
echo "   modeling_molmoact2.py の差し替え: $MOLMOACT2_MODELING_PATCH"
echo "   出力: $OUTPUT_DIR"

NETWORK="parc_final_$$"
POLICY_HOST="parc_final_policy_$$"
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
            PARC_GAIN_TRANSLATION PARC_GAIN_ROTATION PARC_GEO_IMAGE_SIZE; do
    if [ -n "${!name:-}" ]; then
        policy_environment+=(-e "$name=${!name}")
    fi
done

echo "-- 方策サーバーを起動する"
docker run -d --name "$POLICY_HOST" --gpus all --network "$NETWORK" \
    -v "$SERVER_PY:/app/policy_server.py:ro" \
    -v "$REPOSITORY_ROOT/solution/geometry_graft.py:/app/geometry_graft.py:ro" \
    -v "$REPOSITORY_ROOT/work/artifacts/vendor/vggt:/app/vendor/vggt:ro" \
    -v "$WEIGHTS:/app/model_weights:ro" \
    -v "$MOLMOACT2_MODELING_PATCH:/opt/lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py:ro" \
    -v "$HOME/.cache/huggingface:/hf_home" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "${policy_environment[@]}" \
    "$SERVER_IMAGE" python /app/policy_server.py --port 8000 >/dev/null

for _ in $(seq 1 300); do
    if docker exec "$POLICY_HOST" python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" \
        2>/dev/null; then
        echo "   /health OK"
        break
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$POLICY_HOST" 2>/dev/null)" != "true" ]; then
        echo "   方策サーバーが落ちた。ログを $OUTPUT_DIR/server.log に出す" >&2
        docker logs "$POLICY_HOST" > "$OUTPUT_DIR/server.log" 2>&1 || true
        exit 1
    fi
    sleep 2
done

docker logs "$POLICY_HOST" > "$OUTPUT_DIR/server.log" 2>&1 || true
grep -m1 "\[parc\] policy_server" "$OUTPUT_DIR/server.log" || true

record_environment=()
if [ "${RECORD:-0}" != "0" ]; then
    mkdir -p "$OUTPUT_DIR/videos"
    record_environment=(-e "T3_VIDEO_DIR=/workspace/videos"
                        -v "$OUTPUT_DIR/videos:/workspace/videos")
fi

SCORER_PATCH="$REPOSITORY_ROOT/solution/tools/eval/kit_patches/scorer.py"
scorer_mount=()
[ -f "$SCORER_PATCH" ] && scorer_mount=(-v "$SCORER_PATCH:/workspace/pipeline/scorer.py:ro")

ROLLOUT_PATCH="$REPOSITORY_ROOT/solution/tools/eval/kit_patches/rollout.py"
rollout_mount=()
if [ -f "$ROLLOUT_PATCH" ]; then
    rollout_mount=(-v "$ROLLOUT_PATCH:/workspace/pipeline/rollout.py:ro"
                   -e "PARC_OBJECT_DISP_JSONL=/workspace/results/object_disp.jsonl")
fi

tasks_mount=()
if [ -n "${TASKS_CSV:-}" ]; then
    case "$TRACK" in
        track1) csv_in_kit=/workspace/compe/t1/T1_TASKS.csv ;;
        track2) csv_in_kit=/workspace/compe/t2/T2_TASKS.csv ;;
        track3) echo "track3 には CSV が無い" >&2; exit 1 ;;
    esac
    tasks_mount=(-v "$TASKS_CSV:$csv_in_kit:ro")
    cp "$TASKS_CSV" "$OUTPUT_DIR/TASKS_CSV.csv"
fi

echo "-- 評価を回す"
docker run --rm --gpus all --network "$NETWORK" \
    --ulimit core=0 \
    -e "MUJOCO_GL=$MUJOCO_GL" \
    "${scorer_mount[@]}" \
    "${rollout_mount[@]}" \
    "${tasks_mount[@]}" \
    "${record_environment[@]}" \
    -v "$OUTPUT_DIR/results:/workspace/results" \
    "$KIT_IMAGE" \
    python -m pipeline \
        --server-url "http://$POLICY_HOST:8000" \
        --track "$TRACK" \
        --n-episodes "$TRACK_EPISODES" \
        --max-steps "$MAX_STEPS" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

docker run --rm -v "$OUTPUT_DIR:/out" --entrypoint chown "$KIT_IMAGE" \
    -R "$(id -u):$(id -g)" /out || echo "   持ち主を戻せなかった"

echo
echo "== 結果"
docker logs "$POLICY_HOST" 2>&1 | grep -c "POST /act" \
    | sed 's/^/   合計ステップ数: /' || true
echo "   json: $OUTPUT_DIR/results/"

python3 "$REPOSITORY_ROOT/solution/tools/eval/score_composite.py" \
    --by-category "$OUTPUT_DIR" \
    || echo "   合計点を付け直せなかった"
