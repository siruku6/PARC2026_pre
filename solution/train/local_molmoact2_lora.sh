#!/usr/bin/env bash
# このマシン（H100-20C・20 GB・1 枚）で MolmoAct2 の LoRA 学習を起こす。
#
#   # 載る batch を探して、載った最大でそのまま本番へ進む
#   setsid nohup env RUN=molmoact2_lora_v1 LADDER="4 2" STEPS=4000 \
#       bash solution/train/local_molmoact2_lora.sh > ~/molmoact2_lora_v1.log 2>&1 &
#
#   # 1 つずつ打つなら
#   TRIAL=1 BATCH=4 bash solution/train/local_molmoact2_lora.sh   # 試し走り
#   RUN=molmoact2_lora_v1 BATCH=4 bash solution/train/local_molmoact2_lora.sh
#
# **中身は [jaist_molmoact2_lora.sh](jaist_molmoact2_lora.sh) をそのまま呼ぶ。**
# 学習の引数・学習率・直しの内訳はあちらにあり、ここが足すのは Docker の口だけである
# ——このマシンには学習用の venv が無く、[Dockerfile.molmoact2](Dockerfile.molmoact2) で
# 焼いた `parc2026-train-molmoact2` の中に lerobot と peft が入っているためである。
#
# | Docker のためにやること | なぜ |
# |---|---|
# | `VENV=none` | イメージの中には venv が無い（lerobot は最初から入っている） |
# | 直し 2 ファイルの bind-mount | `--user` で非 root にすると `/opt/lerobot` を上書きできない |
# | `FOREGROUND=1` | 切り離すとスクリプトが終わった時点でコンテナごと落ちる |
# | `--shm-size` | 読み手が使う共有メモリ。既定の 64 MB では 1 歩も進まずに落ちる |
#
# **batch は 4 が上限である**（9/16 の実測。8 は OOM。`docs/reference/19` の 7.10）。
# 本番はコンテナを切り離して起こすので、**この shell を閉じても学習は続く。**
#
# Reads:
#   - $DS（データセット。q01/q99 が無ければこのスクリプトが足す）
#   - $INIT_DIR（出発点の重み）と $INIT_DIR/molmoact2_base_small（前処理用の小さいファイル）
# Generates:
#   - $OUT/$RUN/checkpoints/**   チェックポイント
#   - $OUT/$RUN.nohup            学習のログ
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-parc2026-train-molmoact2}"
DS="${DS:-$REPO/work/artifacts/libero_combined_bowl5_initpose_l90clean}"
INIT_DIR="${INIT_DIR:-$REPO/work/artifacts/molmoact2_libero_lerobot}"
OUT="${OUT:-$REPO/work/train}"
BATCH="${BATCH:-4}"
STEPS="${STEPS:-4000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHM_SIZE="${SHM_SIZE:-16g}"
TRIAL="${TRIAL:-0}"
TRIAL_STEPS="${TRIAL_STEPS:-20}"
RUN="${RUN:-molmoact2_lora}"
LADDER="${LADDER:-}"
# **出発点は 1 か所を `/init` に挿すので、小さいほうはその下と決まっている**（口にしない）
SMALL_DIR="$INIT_DIR/molmoact2_base_small"
PATCH="$REPO/solution/tools/eval/lerobot_patches/modeling_molmoact2.py"
TRAIN_PATCH="$REPO/solution/tools/eval/lerobot_patches/lerobot_train.py"
LEROBOT_ROOT=/opt/lerobot/src/lerobot

die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }
step() { echo; echo "=== $* ==="; }

# 1 回分の走り。$1=試し走りなら 1 ／ $2=batch。試し走りは終了コードで載ったかを返す
run_once() {
    local trial="$1" batch="$2"
    local run steps save_freq container peak_file peak reached=0

    run="$RUN"; steps="$STEPS"; save_freq="$SAVE_FREQ"
    if [ "$trial" = 1 ]; then
        run="${RUN}_trial_b${batch}"
        # 保存が起きない長さにする（チェックポイント 1 個で 11 GB ある）
        steps=$(( TRIAL_STEPS + 1000 )); save_freq=$steps
    fi
    container="molmoact2_${run}"
    peak_file="$OUT/$run.peak"

    docker ps -a --format '{{.Names}}' | grep -qx "$container" \
        && die "同じ名前のコンテナが在る: $container" "docker rm -f $container で消すか RUN を変える"
    [ "${RESUME:-0}" = 1 ] || [ ! -e "$OUT/$run" ] \
        || die "出力先が既にある: $OUT/$run" "RUN を変えるか消す。続きからなら RESUME=1"

    if [ "${RESUME:-0}" = 1 ]; then
        step "続きから起こす（$IMAGE・batch と歩数は保存済みの設定のまま）"
    else
        step "学習を起こす（$IMAGE・batch $batch・$steps 歩）"
    fi
    docker run -d --name "$container" --gpus all --shm-size="$SHM_SIZE" \
        --user "$(id -u):$(id -g)" \
        -e RUN="$run" -e VENV=none -e FOREGROUND=1 \
        -e DS=/data -e DATASET_ID="local/$(basename "$DS")" \
        -e INIT_DIR=/init -e SMALL_DIR=/init/molmoact2_base_small -e OUT=/out \
        -e BATCH="$batch" -e STEPS="$steps" -e SAVE_FREQ="$save_freq" -e NUM_WORKERS="$NUM_WORKERS" \
        -e ACTION_MODE="${ACTION_MODE:-both}" -e LOG_FREQ="${LOG_FREQ:-20}" -e SCOPE="${SCOPE:-lora}" \
        -e RESUME="${RESUME:-0}" -e IMAGE_TFS="${IMAGE_TFS:-true}" \
        -v "$REPO:/repo:ro" -v "$DS:/data:ro" -v "$INIT_DIR:/init:ro" -v "$OUT:/out" \
        -v "$HOME/.cache/huggingface:/hf_home" \
        -v "$PATCH:$LEROBOT_ROOT/policies/molmoact2/modeling_molmoact2.py:ro" \
        -v "$TRAIN_PATCH:$LEROBOT_ROOT/scripts/lerobot_train.py:ro" \
        "$IMAGE" bash /repo/solution/train/jaist_molmoact2_lora.sh >/dev/null \
        || die "コンテナを起こせない"
    echo "  コンテナ $container"

    step "$TRIAL_STEPS 歩まで待つ（メモリの山も測る）"
    # **山は 0.5 秒ごとに取る。**確保の山は数百ミリ秒で過ぎるので、秒より粗いと取りこぼす
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$peak_file" 2>/dev/null &
    local sampler=$!
    for _ in $(seq 1 360); do
        # **末尾だけ読む。**`docker logs` は既定でログの先頭から全部を吐き、待つほど重くなる
        docker logs --tail 200 "$container" 2>&1 | tr '\r' '\n' | grep -q "step:$TRIAL_STEPS " \
            && { reached=1; break; }
        docker ps --format '{{.Names}}' | grep -qx "$container" || break
        sleep 10
    done
    kill "$sampler" 2>/dev/null
    peak=$(sort -rn "$peak_file" | head -1); rm -f "$peak_file"

    if [ "$reached" = 1 ]; then
        # 1 歩の秒数と「何時間で何歩か」は、RunPod 側と同じ道具で出す
        python3 "$REPO/solution/train/summarize_speed.py" "$OUT/$run.nohup" \
            --batch "$batch" --peak-mib "${peak:-0}"
    else
        echo "  **$TRIAL_STEPS 歩に届かずに終わった（batch $batch）。**"
        docker logs --tail 200 "$container" 2>&1 | tr '\r' '\n' \
            | grep -m3 -iE "out of memory|OutOfMemoryError|Error" | sed 's/^/    /'
    fi

    if [ "$trial" = 1 ]; then
        step "試し走りを片付ける"
        docker rm -f "$container" >/dev/null
        rm -rf "${OUT:?}/$run" "$OUT/$run.nohup"
        return $(( 1 - reached ))
    fi
    # **本番で届かなかったときに黙って「走っている」と書かない。**
    # まだ生きているなら読み込みが長引いているだけかもしれないので、消さずに残す
    if [ "$reached" != 1 ]; then
        docker ps --format '{{.Names}}' | grep -qx "$container" \
            && echo "  **$TRIAL_STEPS 歩へ届かなかったが、コンテナは生きている。**残す" \
            || die "本番が立ち上がらなかった（batch $batch）" \
                   "docker logs $container | tr '\\r' '\\n' | tail -40 を読む"
    fi
    echo
    echo "=== 走っている ==="
    echo "  ログ: docker logs -f $container | tr '\\r' '\\n'（$OUT/$run.nohup にも書いている）"
    echo "  止める: docker rm -f $container"
}

step "前提"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "イメージが無い: $IMAGE" \
       "docker build -t parc2026-train-molmoact2 -f solution/train/Dockerfile.molmoact2 solution/train"
[ -f "$DS/meta/info.json" ] || die "データが無い: $DS" \
    "hf download siruku6/libero_combined_bowl5_initpose_l90clean --repo-type dataset --local-dir $DS"
[ -f "$INIT_DIR/model.safetensors" ] || die "出発点の重みが無い: $INIT_DIR" \
    "hf download allenai/MolmoAct2-LIBERO-LeRobot --revision f0c5a9567c2b72faadec901e16e055c8b098c2f5 --local-dir $INIT_DIR"
[ -f "$SMALL_DIR/config.json" ] || die "前処理用の小さいファイルが無い: $SMALL_DIR" \
    "依頼 032 の 0-2 を打つ（allenai/MolmoAct2-LIBERO の config・tokenizer・norm_stats）"
grep -q has_base_weights "$PATCH" || die "直しの中身が違う: $PATCH"
grep -q is_molmoact2 "$TRAIN_PATCH" || die "直しの中身が違う: $TRAIN_PATCH"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /'

step "分位点（q01/q99）を足す（既にあれば何もしない）"
# **MolmoAct2 は state と action を QUANTILES で正規化する。**π0.5 向けに組んだデータには
# 入っていない。判定は道具の側が持っているので、ここでは条件を付けずに 1 回打つ
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$DS:/data" -v "$REPO/solution:/solution:ro" \
    "$IMAGE" python /solution/tools/train/add_quantile_stats.py --root /data \
    || die "分位点を足せない" "上の出力を読む"

if [ -n "$LADDER" ]; then
    step "載る batch を探す（$LADDER の大きいほうから）"
    for b in $LADDER; do
        echo; echo "--- batch $b を試す ---"
        if run_once 1 "$b"; then
            step "載った最大は batch $b。この batch で本番を起こす"
            run_once 0 "$b"
            exit $?
        fi
        echo "  batch $b は載らなかった"
    done
    die "どの batch も載らなかった（$LADDER）" "TRIAL=1 BATCH=1 で下を確かめる"
fi

run_once "$TRIAL" "$BATCH"
