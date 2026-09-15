#!/usr/bin/env bash
# 本選の配布キット（PARC2026_final）で、手元の重みを 3 トラックのどれかで評価する。
#
# **予選の run_docker_eval.sh と同じ 2 プロセス構成**である。依存が正面衝突するので
# 1 つのコンテナに混ぜられない。
#
#   parc2026-final      … 本選の配布キット。python3.10・LIBERO-plus・pipeline → 評価側
#   parc2026-train-pi05 … 学習に使ったイメージ。GPU torch と LeRobot     → 方策側
#
# 先に配布キットのイメージを 1 度だけ作る（30〜60 分・30 GB 前後）。
#
#   cd /home/ubuntu/src/b_PARC2026_final && docker build -t parc2026-final .
#
# 使い方
#
#   bash solution/tools/eval/run_final_eval.sh track3            # 4 タスク × 2 本
#   TRACK_EPISODES=1 bash solution/tools/eval/run_final_eval.sh track2
#
# 環境変数（省略時の既定）
#   TRACK_EPISODES=2  タスクあたりの本数。**比べる実行どうしは必ず同じ本数にする。**
#                     1 行の中で episode 番号ごとに初期状態が決まっていて（`init_states[ep_id]`）、
#                     番号によって難しさが違う。本数が違うと別の初期状態の混ざりを比べることになる
#                     （9/15 に 1・3・5 本の実行を並べて重みを選びかけた。docs/howto/03 の「比べる評価の揃え方」）
#   MAX_STEPS=300     本番と同じ打ち切り
#   RECORD=1          各タスクの 1 本目を mp4 に残す（成功／失敗が名前に入る）
#   WEIGHTS           測る重み。既定は 4.2.0 の重み（work/artifacts/pi05_rp3000）
#   LABEL             出力先の名前に足す語
#   SERVER_PY         起こす policy_server.py。既定は solution/ のもの。
#                     **提出物を検証するときは zip から取り出した方を渡す**——
#                     版としきい値は zip の中身が正であり、solution/ は
#                     組んだ後に戻していることがある（[howto/02](../../../docs/howto/02_build_submission.md)）。
#   PARC_*            推論の設定値。空なら policy_server.py の既定が効く
#   MUJOCO_GL=osmesa  描画のやり方。**JAIST サーバの vGPU は OpenGL に対応しないので
#                     osmesa（CPU 描画）のままにする。** 会社の RTX6000（Pass-Through の
#                     実機）では `MUJOCO_GL=egl` を渡す。箱 1 個の描画で 9.7 ms → 0.6 ms。
#   KIT_ROOT          配布キットの場所。既定は「このリポジトリの隣の b_PARC2026_final」。
#                     **存在確認にしか使わない**（pipeline は評価イメージの中にある）
#
# Reads:
#   - $SERVER_PY（既定 solution/policy_server.py）
#   - work/artifacts/<重み>/**
# Generates:
#   - work/final_eval/<日時>_<トラック>_<LABEL>/results/*.json・server.log
#   - 同じところに score_composite.md（採点側と同じ形の合計点＋その計算式）
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_ROOT="${KIT_ROOT:-$(cd "$REPOSITORY_ROOT/../b_PARC2026_final" && pwd)}"
cd "$REPOSITORY_ROOT"

TRACK="${1:-track3}"
case "$TRACK" in
    track1 | track2 | track3) ;;
    *) echo "トラックは track1 / track2 / track3 のいずれか（渡された値: $TRACK）" >&2; exit 2 ;;
esac

TRACK_EPISODES="${TRACK_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-300}"
WEIGHTS="${WEIGHTS:-$REPOSITORY_ROOT/work/artifacts/pi05_rp3000}"
SERVER_PY="${SERVER_PY:-$REPOSITORY_ROOT/solution/policy_server.py}"
KIT_IMAGE="${KIT_IMAGE:-parc2026-final}"
POLICY_IMAGE="${POLICY_IMAGE:-parc2026-train-pi05}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"

# **`TASKS_CSV` はここで解決して存在を確かめる。**方策サーバーの起動（重みの読み込みに
# 1〜2 分）より前に落とさないと、綴りを 1 文字間違えただけで 2 分待たされる
# （2026-08-28 に実際に起きた）。**相対パスはリポジトリの根から見る**——
# `compe/val/...` と書いて別のディレクトリから叩けるようにするためである。
if [ -n "${TASKS_CSV:-}" ]; then
    case "$TASKS_CSV" in
        /*) ;;
        *) TASKS_CSV="$REPOSITORY_ROOT/$TASKS_CSV" ;;
    esac
    if [ ! -f "$TASKS_CSV" ]; then
        echo "TASKS_CSV が見つからない: $TASKS_CSV" >&2
        echo "  リポジトリの根は $REPOSITORY_ROOT である。" >&2
        echo "  ここに置いてあるもの:" >&2
        ls "$REPOSITORY_ROOT/compe/val/"*.csv 2>/dev/null | sed 's/^/    /' >&2
        echo "  新しく作った CSV は git pull で降りてくる。" >&2
        exit 2
    fi
    TASKS_CSV="$(cd "$(dirname "$TASKS_CSV")" && pwd)/$(basename "$TASKS_CSV")"
fi

label="$TRACK"
[ -n "${LABEL:-}" ] && label="${label}_${LABEL}"
OUTPUT_DIR="$REPOSITORY_ROOT/work/final_eval/$(date +%m%d_%H%M%S)_${label}"
mkdir -p "$OUTPUT_DIR/results"
# **比べるための条件を実行ディレクトリへ残す**（9/15 に踏んだ：本数と描き方が違う実行を並べて重みを選びかけた）。
# `decision_score.py` がこれを読み、候補どうしで本数・描き方・行の一覧が違えば止まる。
printf '{"track": "%s", "episodes": %s, "max_steps": %s, "mujoco_gl": "%s", "tasks_csv": "%s", "weights": "%s"}\n' \
    "$TRACK" "$TRACK_EPISODES" "$MAX_STEPS" "$MUJOCO_GL" "${TASKS_CSV:-}" "$WEIGHTS" > "$OUTPUT_DIR/eval_conditions.json"

# **端末が消えても読めるように、この先の出力を run.log にも落とす。**
# バックグラウンドへ回したあと SSH を切ると、端末への書き込みが失敗して
# `set -euo pipefail` でスクリプトが死ぬ。ファイルにも書いておけば結果は残る。
exec > >(tee -a "$OUTPUT_DIR/run.log") 2>&1

# ---------------------------------------------------------------- 0. 前提の確認

for image in "$KIT_IMAGE" "$POLICY_IMAGE"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "イメージ $image が無い。上のコメントの docker build を先に実行する" >&2
        exit 1
    fi
done

# **HF キャッシュを 2 か所へ載せているのは、イメージによって置き場が違うため。**
# π0.5 学習イメージは `ENV HF_HOME=/hf_home` を持つので `/root/.cache/huggingface` に
# 載せても読まれない（2026-08-26 に学習側で踏んだ）。両方に載せればどちらでも読まれる。

# **学習イメージには HTTP サーバーの依存が入っていない。**
# LeRobot が `fastapi` を宣言しているのは遠隔操作用の extra だけで、学習用の
# extra には入らない。`msgpack` と `uvicorn` も同じである。**入れずに起動すると
# `ModuleNotFoundError: No module named 'msgpack'` で落ちる**（2026-08-25 に踏んだ）。
#
# 予選の run_docker_eval.sh と同じやり方で派生イメージを作る。**タグに元イメージの
# ID を含めるので、元が作り直されたら別タグになり、古い中身が生き残らない。**
POLICY_IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$POLICY_IMAGE" | cut -d: -f2 | cut -c1-12)"
SERVER_IMAGE="${POLICY_IMAGE}-server-${POLICY_IMAGE_ID}"
if ! docker image inspect "$SERVER_IMAGE" >/dev/null 2>&1; then
    echo "-- $SERVER_IMAGE を作る（HTTP サーバーの依存 3 つを足すだけ）"
    printf 'FROM %s\nRUN pip install --no-cache-dir "fastapi>=0.68" "uvicorn>=0.15" "msgpack>=1.0"\n' \
        "$POLICY_IMAGE" | docker build -t "$SERVER_IMAGE" -f - . >/dev/null
fi

if [ ! -d "$WEIGHTS/vlm" ]; then
    echo "$WEIGHTS に vlm/ が無い。評価は 5 秒で落ちる" >&2
    echo "（pi05_rp3000 から vlm/ を symlink ではなく copy で持ってくること）" >&2
    exit 1
fi

echo "== 本選評価  トラック=$TRACK  本数=$TRACK_EPISODES  打ち切り=$MAX_STEPS  描画=$MUJOCO_GL"
echo "   重み: $WEIGHTS"
echo "   出力: $OUTPUT_DIR"

# ---------------------------------------------------------------- 1. 方策側

NETWORK="parc_final_$$"
POLICY_HOST="parc_final_policy_$$"
docker network create "$NETWORK" >/dev/null

cleanup() {
    # **消す前に全ログを保存する。**消したあとでは `docker logs` が引けず、
    # 合計ステップ数（`POST /act` の回数）を数え直せなくなる。
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

echo "-- 方策サーバーを起動する（重みの読み込みに 1〜2 分）"
docker run -d --name "$POLICY_HOST" --gpus all --network "$NETWORK" \
    -v "$SERVER_PY:/app/policy_server.py:ro" \
    -v "$REPOSITORY_ROOT/solution/geometry_graft.py:/app/geometry_graft.py:ro" \
    -v "$REPOSITORY_ROOT/work/artifacts/vendor/vggt:/app/vendor/vggt:ro" \
    -v "$WEIGHTS:/app/model_weights:ro" \
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

# 採用された設定値は server.log の [parc] 行にある。渡した値ではなくこちらを信じる。
docker logs "$POLICY_HOST" > "$OUTPUT_DIR/server.log" 2>&1 || true
grep -m1 "\[parc\] policy_server" "$OUTPUT_DIR/server.log" || true

# ---------------------------------------------------------------- 2. 評価側

# **動画を残す。**本選キットは `T3_VIDEO_DIR` が設定されているときだけ、
# 各タスクの 1 本目（episode_id = 0）を mp4 に書く（`pipeline/rollout.py`）。
# ファイル名は `<タスク名>__SUCCESS.mp4` / `__FAIL.mp4` になる。
record_environment=()
if [ "${RECORD:-0}" != "0" ]; then
    mkdir -p "$OUTPUT_DIR/videos"
    record_environment=(-e "T3_VIDEO_DIR=/workspace/videos"
                        -v "$OUTPUT_DIR/videos:/workspace/videos")
    echo "-- 動画を残す: $OUTPUT_DIR/videos"
fi

# **配布キットの採点器を、衝突率と SPARC を出す版で上書きする。**
# 本選の配布版はこの 2 つを出さない。キットの中を直すとマシンごとにズレるので
# （8/27 に JAIST と会社の RTX6000 で指標の数が 16 個と 14 個に食い違った）、
# こちらのリポジトリに置いたものをマウントする。
SCORER_PATCH="$REPOSITORY_ROOT/solution/tools/eval/kit_patches/scorer.py"
scorer_mount=()
if [ -f "$SCORER_PATCH" ]; then
    scorer_mount=(-v "$SCORER_PATCH:/workspace/pipeline/scorer.py:ro")
    echo "-- 採点器を差し替える（衝突率と SPARC を出す版）"
fi

# **衝突と判定された根拠を残す。**キットの `pipeline/rollout.py` は、操作対象でない
# 物体が初期位置からどれだけ動いたか（`object_max_disp`）を計算しておきながら、
# しきい値 1 mm と比べた真偽だけを使って値を捨てている。**「かすった」のか「倒した」の
# かが分からないと、衝突対策を推論側の後処理でやるのか学習でやるのかが決まらない。**
# 差し替え版は 1 行 1 ロールアウトの JSONL を `PARC_OBJECT_DISP_JSONL` へ書く。
# **書き出し先はコンテナの中なので、results のマウントの下に置く**（外から見えるのは
# `$OUTPUT_DIR/results/object_disp.jsonl`）。
ROLLOUT_PATCH="$REPOSITORY_ROOT/solution/tools/eval/kit_patches/rollout.py"
rollout_mount=()
if [ -f "$ROLLOUT_PATCH" ]; then
    rollout_mount=(-v "$ROLLOUT_PATCH:/workspace/pipeline/rollout.py:ro"
                   -e "PARC_OBJECT_DISP_JSONL=/workspace/results/object_disp.jsonl")
    echo "-- 衝突の内訳を残す: $OUTPUT_DIR/results/object_disp.jsonl"
fi

# **評価する行を、こちらの CSV に差し替える。**
# 配布キットは `compe/t2/T2_TASKS.csv` の 8 行を読むだけで、行を選び直す口が無い。
# **その 8 行は Track1 の摂動（背景・照明）しか含まないので、Track2 の摂動で
# 候補を比べたいときは行ごと入れ替える必要がある**（docs/plan/16 の 2 節）。
# variant 名の数値は一覧を参照せず文字列から読まれるので、一覧に無い組み合わせも走る
# （docs/reference/20 の 2.3 節）。
tasks_mount=()
if [ -n "${TASKS_CSV:-}" ]; then
    # 存在確認と絶対パス化は、方策サーバーを起動する前に済ませてある。
    case "$TRACK" in
        track1) csv_in_kit=/workspace/compe/t1/T1_TASKS.csv ;;
        track2) csv_in_kit=/workspace/compe/t2/T2_TASKS.csv ;;
        track3) echo "track3 には CSV が無い（compe/t3 は paths.py と assets から組む）" >&2; exit 1 ;;
        *) echo "TRACK が読めない: $TRACK" >&2; exit 1 ;;
    esac
    tasks_mount=(-v "$TASKS_CSV:$csv_in_kit:ro")
    # **入力の一覧を実行ディレクトリへ写す。**次の週に同じ CSV があるとは限らない。
    cp "$TASKS_CSV" "$OUTPUT_DIR/TASKS_CSV.csv"
    echo "-- 評価する行を差し替える: $TASKS_CSV（$(( $(wc -l < "$TASKS_CSV") - 1 )) 行）→ $csv_in_kit"
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

# 評価側は root で走るので、結果の持ち主を戻しておく（戻さないと ubuntu では消せない）。
docker run --rm -v "$OUTPUT_DIR:/out" --entrypoint chown "$KIT_IMAGE" \
    -R "$(id -u):$(id -g)" /out || echo "   持ち主を戻せなかった。消すときは sudo が要る"

# ---------------------------------------------------------------- 3. 読み方

echo
echo "== 結果"
docker logs "$POLICY_HOST" 2>&1 | grep -c "POST /act" \
    | sed 's/^/   合計ステップ数（POST \/act の回数）: /' || true
echo "   json: $OUTPUT_DIR/results/"
echo "   成功・衝突・軌道の総距離は json の中にある"

# **採点側と同じ形の合計点を付け直して、実行ディレクトリへ残す。**
# 配布キットが出す `overall_score` は成功率そのもので、なめらかさが入っていない
# （docs/reference/17_scoring_formula.md）。**あとから「この点はどの式か」を
# 引けるように、式も一緒に score_composite.md へ書く。**
# 失敗しても評価の結果は残っているので、ここでは止めない。
python3 "$REPOSITORY_ROOT/solution/tools/eval/score_composite.py" \
    --by-category "$OUTPUT_DIR" \
    || echo "   合計点を付け直せなかった。あとから score_composite.py を打てばよい"
