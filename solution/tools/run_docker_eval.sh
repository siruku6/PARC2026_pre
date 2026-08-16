#!/usr/bin/env bash
# GPU マシンで検証用スイートを流す。**既にある 2 つのイメージをそのまま使う。**
#
# 学習は Docker の中でやったので、ホストの `.venv` には torch も LeRobot も
# 無い。**イメージを 2 つ並べれば、どちらも作り直さずに評価できる。**
#
#   parc2026-train … 学習に使ったイメージ。GPU torch と LeRobot が入っている → 方策側
#   parc2026       … 配布キットのイメージ。python3.10 と LIBERO が入っている → 評価側
#
# **本番と同じ 2 プロセス構成**である。依存が正面衝突するので 1 つに混ぜられない。
#
#   bash solution/tools/run_docker_eval.sh --check    # 5 件。5 分で配線を見る
#   bash solution/tools/run_docker_eval.sh            # 84 件。90 分
#   bash solution/tools/run_docker_eval.sh --detach   # 同じものを裏で流して抜ける
#
# 事前に 2 つのイメージが要る。
#   docker build -t parc2026-train solution/train      # 学習で作ってあるはず
#   docker build -t parc2026 .                         # 配布キット
#
# 環境変数（省略時の既定）
#   MAX_STEPS=300  本番と同じ打ち切り
#   N_EPISODES=1   **タスクあたり**の本数。件数が多いので 1 のままにする
#   EVAL_CSV       **測るタスクの一覧。既定は compe/val/EVAL_TASKS.csv（36 件）。**
#                  学習が引かなかった実演がある指示文だけで作った集合である
#                  （solution/tools/make_eval_split.py）。
#                  `EVAL_CSV=` と空にすると、代わりに 84 件を生成して測る
#   PER_CELL=3     84 件を生成するときの、スイート × カテゴリあたりの本数。
#                  **EVAL_CSV を空にしたときだけ効く**
#   RECORD=1       ロールアウトを mp4 に撮る（失敗した回だけ）
#   WEIGHTS        測る重み。既定 work/artifacts/model_weights
#                  **素の重み（0.1409 の構成）と学習後を比べるときはここを変える**
#   LABEL          出力先の名前に足す語。どの重みの回か分かるようにする
#   PARC_*         推論の設定値。空なら policy_server.py の既定が効く
#
# Reads:
#   - solution/policy_server.py, solution/tools/*.py
#   - work/artifacts/model_weights/**
# Generates:
#   - work/gpu_eval/<日時>_<設定>/eval.log・server.log・*.json・videos/
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"

POLICY_IMAGE="${POLICY_IMAGE:-parc2026-train}"
KIT_IMAGE="${KIT_IMAGE:-parc2026}"
MAX_STEPS="${MAX_STEPS:-300}"
N_EPISODES="${N_EPISODES:-1}"
PER_CELL="${PER_CELL:-3}"
# **既定は 36 件の集合である。** 84 件は学習に使ったタスクを検証にも使っており、
# 手元の順序が採点環境と逆になった（docs/howto/04_proxy_eval.md）。
# 空にすると 84 件を生成する道に戻る。
EVAL_CSV="${EVAL_CSV-$REPOSITORY_ROOT/compe/val/EVAL_TASKS.csv}"
RECORD="${RECORD:-1}"
NETWORK="${NETWORK:-parc-eval-net}"
POLICY_HOST="parc-policy"
CHECK_ONLY=""
DETACH=""

for argument in "$@"; do
    case "$argument" in
        --check)  CHECK_ONLY=1 ;;
        --detach) DETACH=1 ;;
        *) echo "知らない引数: $argument" >&2; exit 2 ;;
    esac
done

if [ -n "$DETACH" ] && [ -z "${PARC_EVAL_CHILD:-}" ]; then
    mkdir -p "$REPOSITORY_ROOT/work/gpu_eval"
    launcher_log="$REPOSITORY_ROOT/work/gpu_eval/launcher.log"
    PARC_EVAL_CHILD=1 nohup bash "$0" > "$launcher_log" 2>&1 &
    echo "[eval] 裏で流し始めた (pid=$!)"
    echo "[eval] 進み具合:  tail -f $launcher_log"
    echo "[eval] **端末を閉じても続く。** 止めるには kill $! && docker rm -f parc-policy"
    exit 0
fi

# ---------------------------------------------------------------- 0. 前提

echo "===== 0/5 前提を確かめる ====="

for image in "$POLICY_IMAGE" "$KIT_IMAGE"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "エラー: イメージが無い: $image" >&2
        echo "  docker build -t parc2026-train solution/train" >&2
        echo "  docker build -t parc2026 ." >&2
        exit 1
    fi
    echo "  イメージ $image あり"
done

# **学習イメージには HTTP サーバーの依存が入っていない。**
# LeRobot が `fastapi` を宣言しているのは `phone` extra（遠隔操作用）だけで、
# `[training,smolvla,peft]` では入らない。`msgpack` と `uvicorn` も同様である。
# 提出物の requirements.txt が 3 つを「必須」に挙げているのはこのためで、
# **採点環境では venv に入るが、学習イメージには無い。**
#
# 1 度だけ薄い派生イメージを作る。2 回目以降はキャッシュが効く。
SERVER_IMAGE="${POLICY_IMAGE}-server"
if ! docker image inspect "$SERVER_IMAGE" >/dev/null 2>&1; then
    echo "  $SERVER_IMAGE を作る（HTTP サーバーの依存 3 つを足すだけ）"
    printf 'FROM %s\nRUN pip install --no-cache-dir "fastapi>=0.68" "uvicorn>=0.15" "msgpack>=1.0"\n' \
        "$POLICY_IMAGE" | docker build -t "$SERVER_IMAGE" -f - . >/dev/null
fi
echo "  イメージ $SERVER_IMAGE あり"

WEIGHTS="${WEIGHTS:-$REPOSITORY_ROOT/work/artifacts/model_weights}"
WEIGHTS="$(cd "$WEIGHTS" 2>/dev/null && pwd || echo "$WEIGHTS")"
if [ ! -f "$WEIGHTS/model.safetensors" ]; then
    echo "エラー: 重みが無い: $WEIGHTS/model.safetensors" >&2
    echo "  素の重み（0.1409 の構成）を測るなら、先に取り寄せること:" >&2
    echo "    bash solution/tools/fetch_base_weights.sh" >&2
    exit 1
fi
if [ ! -d "$WEIGHTS/vlm" ]; then
    echo "エラー: $WEIGHTS/vlm が無い（SmolVLM の設定とトークナイザ）" >&2
    exit 1
fi
echo "  重み $WEIGHTS"
echo "       $(du -h "$WEIGHTS/model.safetensors" | cut -f1)  sha256 $(sha256sum "$WEIGHTS/model.safetensors" | cut -c1-16)"

# 方策側のイメージから GPU が見えるか。**ここで落ちれば 90 分を無駄にしない。**
docker run --rm --gpus all "$SERVER_IMAGE" python -c "
import torch
print(f'  方策側 torch {torch.__version__}  CUDA {torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA が見えない'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# ---------------------------------------------------------------- 1. 出力先

label="n${PARC_N_ACTION_STEPS:-5}_ep${N_EPISODES}_max${MAX_STEPS}"
[ -n "${LABEL:-}" ] && label="${LABEL}_${label}"
[ -n "${PARC_EMA_ALPHA:-}" ] && label="${label}_ema${PARC_EMA_ALPHA//./}"
[ -n "${PARC_BLEND_STEPS:-}" ] && label="${label}_blend${PARC_BLEND_STEPS}"
[ -n "${PARC_SAMPLES:-}" ] && label="${label}_s${PARC_SAMPLES}"
[ -n "$CHECK_ONLY" ] && label="check_${label}"

OUTPUT_DIR="$REPOSITORY_ROOT/work/gpu_eval/$(date +%m%d_%H%M%S)_${label}"
mkdir -p "$OUTPUT_DIR"
echo "  出力先 $OUTPUT_DIR"

# **測るタスクの一覧を、コンテナから見える場所へ先に置く。**
# コンテナへ渡しているのは `/host/tools` と `/out` の 2 つだけなので、
# `$OUTPUT_DIR`（＝コンテナの `/out`）へ写すのがいちばん短い。
# **置いてあれば `docker_eval_inside.sh` は生成せずにこれを使う。**
if [ -n "$EVAL_CSV" ]; then
    if [ ! -f "$EVAL_CSV" ]; then
        echo "  **測るタスクの一覧がない: $EVAL_CSV**" >&2
        echo "  作る: python3 solution/tools/make_eval_split.py --record <episodes.json>" >&2
        echo "  84 件を生成する道に戻す: EVAL_CSV= bash $0" >&2
        exit 1
    fi
    cp "$EVAL_CSV" "$OUTPUT_DIR/VAL_TASKS.csv"
    echo "  測るタスク $(( $(wc -l < "$EVAL_CSV") - 1 )) 件（$EVAL_CSV）"
fi

# 方策側へ渡す並びを作る。**`policy_server.py` は重みを自分の隣から読む**
# （`Path(__file__).parent / "model_weights"`）ので、この形にする必要がある。
SERVE_DIR="$OUTPUT_DIR/serve"
mkdir -p "$SERVE_DIR"
cp "$REPOSITORY_ROOT/solution/policy_server.py" "$SERVE_DIR/"

# ---------------------------------------------------------------- 2. 後片付け

cleanup() {
    docker rm -f "$POLICY_HOST" >/dev/null 2>&1 || true
    docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup            # 前回の残骸があれば先に消す

docker network create "$NETWORK" >/dev/null

# ---------------------------------------------------------------- 3. 方策側

echo
echo "===== 1/5 方策サーバーを起動する（$SERVER_IMAGE）====="

policy_environment=()
for name in PARC_N_ACTION_STEPS PARC_SAMPLES PARC_BLEND_STEPS PARC_EMA_ALPHA \
            PARC_NUM_INFERENCE_STEPS; do
    value="$(eval "printf '%s' \"\${$name:-}\"")"
    if [ -n "$value" ]; then
        policy_environment+=(-e "$name=$value")
        echo "  設定値 $name=$value"
    fi
done

docker run -d --name "$POLICY_HOST" --gpus all --network "$NETWORK" \
    -v "$SERVE_DIR/policy_server.py:/app/policy_server.py:ro" \
    -v "$WEIGHTS:/app/model_weights:ro" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "${policy_environment[@]}" \
    "$SERVER_IMAGE" python /app/policy_server.py --port 8000 >/dev/null

echo "  起動を待つ（重みの読み込みに 1〜2 分）"
for _ in $(seq 1 300); do
    if docker exec "$POLICY_HOST" python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" \
        2>/dev/null; then
        echo "  /health OK"
        break
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$POLICY_HOST" 2>/dev/null)" != "true" ]; then
        echo "  **方策サーバーが落ちた。** ログの末尾:" >&2
        docker logs --tail 40 "$POLICY_HOST" >&2
        exit 1
    fi
    sleep 1
done

# **版と実効設定を残す。** 差し替え忘れにその場で気づける。
docker logs "$POLICY_HOST" > "$OUTPUT_DIR/server.log" 2>&1
grep '^\[parc\]' "$OUTPUT_DIR/server.log" | sed 's/^/  /' || \
    echo "  **[parc] 行が無い。** 古い policy_server.py を見ている恐れがある"

# ---------------------------------------------------------------- 4. 評価側

echo
echo "===== 2/5 評価側を起動する（$KIT_IMAGE）====="

kit_environment=(
    # **描画のやり方。既定は `osmesa`（CPU 描画）である。**
    # このマシンの vGPU は `NVIDIA Virtual Compute Server`（計算専用の製品）で、
    # **OpenGL に対応しない。** `egl` にすると「空の場面 1 枚 0.1 ms」という
    # 速さの測定は通るのに（実質何も描いていない）、**本当の場面で 50〜99 ステップ目に
    # `read_pixels` で異常終了する**（2026-08-14 に 3 回とも再現）。
    # 詳しくは docs/howto/03_gpu_eval.md「描画は osmesa を使う」。
    # **既定を egl にしていたせいで、この罠に 2 度はまっている。**
    # 描画のできる GPU（vWS 系や普通の GPU）で回すときだけ MUJOCO_GL=egl を渡す。
    -e "MUJOCO_GL=${MUJOCO_GL:-osmesa}"
    # 止まった場所を後から知れるようにする。`docker exec <名前> kill -ABRT 1`
    # で python の居場所が出る。**これが無いと当て推量になる。**
    -e "PYTHONFAULTHANDLER=1"
    # **これを渡さないと EGL が使えない。** nvidia-container-runtime は
    # 既定で compute,utility しか通さず、描画（graphics）を落とす。
    # そのままだと Mesa のソフトウェア実装に落ち、**1 ステップ 300 ms になる**
    # （84 件で 16 時間。EGL なら 39 ms で 90 分）。
    -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"
    -e "SERVER_URL=http://$POLICY_HOST:8000"
    -e "TASKS=$([ -n "$CHECK_ONLY" ] && echo 5 || echo 0)"
    -e "MAX_STEPS=$MAX_STEPS"
    -e "N_EPISODES=$N_EPISODES"
    -e "PER_CELL=$PER_CELL"
    -e "RECORD=$RECORD"
)

# **`--ulimit core=0` と `PYTHONFAULTHANDLER=1` は、落ちたときに気づくためにある。**
# 2026-08-14、評価側が 50 ステップ目で segfault した。既定のままだとカーネルが
# コアダンプを書きに行き（`core_pattern` は apport へのパイプ）、**GPU の
# メモリを抱えたプロセスのダンプは終わらない。** 結果、`docker ps` では
# 動いたまま、進捗だけが止まる——落ちたことに 20 分気づけなかった。
# **`PYTHONFAULTHANDLER` は元から kit_environment に入っていた。** それでも
# 位置が出なかったのは、ダンプが終わらず出力を流し切れなかったからである。
docker run --rm --gpus all --network "$NETWORK" \
    --ulimit core=0 \
    "${kit_environment[@]}" \
    -v "$REPOSITORY_ROOT/solution/tools:/host/tools:ro" \
    -v "$OUTPUT_DIR:/out" \
    "$KIT_IMAGE" bash /host/tools/docker_eval_inside.sh

# **評価側のコンテナだけは root のまま走らせている。**
# 中にある配布キット（`/workspace` に置かれた `pipeline/` や `compe/`）へ
# パッチを当てる必要があり、そこは root のものだからである。
#
# その結果、結果の置き場（ホストの `work/gpu_eval/<回>/` を、コンテナには
# `/out` という名前で見せている）に書かれたファイルも root のものになる。
# **そのままだと `ubuntu` では消せない 900 MB が残る。** 持ち主を戻しておく。
docker run --rm -v "$OUTPUT_DIR:/out" --entrypoint chown "$KIT_IMAGE" \
    -R "$(id -u):$(id -g)" /out || \
    echo "  持ち主を戻せなかった。消すときは sudo が要る"

# ---------------------------------------------------------------- 5. 判定

# **評価が終わってから取り直す。** 起動直後に取ると [parc] 行だけで、
# 推論中の警告が残らない。
docker logs "$POLICY_HOST" > "$OUTPUT_DIR/server.log" 2>&1 || true

echo
echo "===== 判定 ====="
docker run --rm -v "$OUTPUT_DIR:/out" \
    -v "$REPOSITORY_ROOT/solution/tools:/host/tools:ro" \
    "$KIT_IMAGE" python /host/tools/summarize_gpu_eval.py \
        --output-dir /out --csv /out/VAL_TASKS.csv
