#!/usr/bin/env bash
# 評価側コンテナの中でやること。**ホストからは run_docker_eval.sh が呼ぶ。**
#
# 方策サーバーは別のコンテナで動いている（$SERVER_URL）。ここは
# 配布キットの評価パイプラインを回すだけである。
#
# 環境変数（run_docker_eval.sh が渡す）
#   SERVER_URL   方策サーバーの URL
#   TASKS        0 なら全部。5 なら先頭 5 件だけ（配線の確認用）
#   MAX_STEPS    1 エピソードの打ち切り
#   N_EPISODES   タスクあたりのエピソード数
#   PER_CELL     スイート × カテゴリあたりの本数
#   RECORD       空でなければ mp4 を撮る
set -euo pipefail

KIT="${KIT:-/workspace}"
PY="${PY:-$KIT/venv/bin/python}"
TOOLS_DIR="${TOOLS_DIR:-/host/tools}"
OUT_DIR="${OUT_DIR:-/out}"
SERVER_URL="${SERVER_URL:?SERVER_URL が要る}"
TASKS="${TASKS:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
N_EPISODES="${N_EPISODES:-1}"
PER_CELL="${PER_CELL:-3}"
RECORD="${RECORD:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$KIT/LIBERO-plus:$KIT:$KIT/compe"

# ---------------------------------------------------------------- 描画

echo "[3/5] GPU 描画（EGL）が本当に効いているかを測る"
# **当てにしてはいけない。** libglvnd は egl_vendor.d の JSON でベンダを選ぶ。
# NVIDIA の登録が無いと Mesa のソフトウェア実装を掴み、**例外も警告も
# 出さずに 8 倍近く遅くなる**（Colab で実際に踏んだ）。
# ここで測っておけば、90 分を無駄にしてから気づくことがない。
ICD=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [ ! -f "$ICD" ]; then
    mkdir -p "$(dirname "$ICD")"
    printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$ICD"
    echo "  $ICD を作った（NVIDIA の EGL を使わせる）"
fi

"$PY" - <<'PY'
import os, time
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
model = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>"
)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 128, 128)
renderer.update_scene(data); renderer.render()      # 1 枚目は初期化を含む
start = time.time()
for _ in range(50):
    renderer.update_scene(data); renderer.render()
per_frame_ms = (time.time() - start) / 50 * 1000
print(f"  1 枚 {per_frame_ms:.1f} ms")
if per_frame_ms > 8.0:
    print("  **遅い。ソフトウェア描画に落ちている。**")
    print("  --gpus all と NVIDIA_DRIVER_CAPABILITIES に graphics が要る。")
    raise SystemExit(1)
print("  GPU 描画で動いている")
PY

# ---------------------------------------------------------------- 準備

echo
echo "[4/5] 検証用スイートを作り、キットにパッチを当てる"

CSV="$OUT_DIR/VAL_TASKS.csv"
"$PY" "$TOOLS_DIR/make_validation_suite.py" \
    --libero-plus "$KIT/LIBERO-plus" \
    --exclude "$KIT/compe/t1/T1_TASKS.csv" \
    --output "$CSV" --per-cell "$PER_CELL" 2>&1 | tail -10

"$PY" "$TOOLS_DIR/patch_kit_for_variants.py" --kit-dir "$KIT"

# 録画の本体をキットから読める場所へ置く（パッチが import する）。
cp "$TOOLS_DIR/rollout_recorder.py" "$KIT/pipeline/rollout_recorder.py"

if [ ! -f "$KIT/compe/t1/T1_TASKS.csv.example" ]; then
    cp "$KIT/compe/t1/T1_TASKS.csv" "$KIT/compe/t1/T1_TASKS.csv.example"
fi
cp "$CSV" "$KIT/compe/t1/T1_TASKS.csv"
echo "  CSV を差し替えた（$(( $(wc -l < "$CSV") - 1 )) 件）"

echo
echo "  評価の前に、全部開けるかを確かめる"
"$PY" "$TOOLS_DIR/check_validation_suite.py" \
    --csv "$CSV" --libero-plus "$KIT/LIBERO-plus" --kit-dir "$KIT"

# ---------------------------------------------------------------- 評価

echo
echo "[5/5] 評価を回す"

if [ -n "$RECORD" ]; then
    export PARC_RECORD_DIR="$OUT_DIR/videos"
    export PARC_RECORD_WHICH="${PARC_RECORD_WHICH:-failures}"
    export PARC_RECORD_MAX="${PARC_RECORD_MAX:-40}"
    echo "  録画あり（which=$PARC_RECORD_WHICH）"
fi

extra_arguments=()
if [ "$TASKS" != "0" ]; then
    extra_arguments+=(--max-tasks "$TASKS")
    echo "  **先頭 $TASKS 件だけ。** 配線の確認用"
else
    echo "  全件・$N_EPISODES エピソード・$MAX_STEPS 歩"
fi

cd "$KIT"
set +e
"$PY" -m pipeline \
    --server-url "$SERVER_URL" --track track1 \
    --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
    --output-dir "$OUT_DIR" "${extra_arguments[@]}" \
    > "$OUT_DIR/eval.log" 2>&1
echo "  終了コード=$?"
set -e
echo "  [結果] 行: $(grep -c '\[結果\]' "$OUT_DIR/eval.log" || true)"
