#!/usr/bin/env bash
# GPU マシンで検証用スイートを流す。**放っておける形で回す。**
#
# Colab には 3 つの困りごとがあった。
#   - 見ていないと切れる（学習は 9,100 ステップで切れた）
#   - `/content` は消えるので Drive へ逃がす仕掛けが要る
#   - GPU が当たるかどうかが運次第
# GPU マシンなら 3 つとも消える。**しかも Colab より速い**（下の「なぜ速いか」）。
#
# ホスト（GPU マシン）で直接実行する。Docker は使わない。
# 配布 Dockerfile は CPU 描画（osmesa）で固定されており、**1 ステップ 300 ms** で
# 84 件が 16 時間になる。EGL で GPU 描画にすれば 39 ms で済む。
#
#   bash solution/tools/eval/run_gpu_eval.sh --check     # 5 分。配線だけ確かめる
#   bash solution/tools/eval/run_gpu_eval.sh            # 90 分。84 件を流す
#   bash solution/tools/eval/run_gpu_eval.sh --detach   # 同じものを裏で流して抜ける
#
# 環境変数（省略時の既定）
#   POLICY_PYTHON  学習に使った venv の python。**GPU torch と LeRobot が要る**
#   KIT_DIR        配布キットの置き場（既定 work/eval-kit。無ければ作る）
#   MAX_STEPS=300  本番と同じ打ち切り
#   N_EPISODES=1   **タスクあたり**の本数。84 件あるので 1 のままにする
#   PER_CELL=3     スイート × カテゴリあたりの本数（CSV 全体で 84 件）
#   RECORD=1       ロールアウトを mp4 で録画する（既定で有効。失敗した回だけ）
#   PARC_*         推論の設定値。空なら policy_server.py の既定が効く
#   QUANT_SERVE_MODE   推論時量子化スイープ（π0.5 専用。work/quant_sweep/
#                  CANDIDATES.md の C4/C5）。空なら量子化なし（既定）。
#                  bnb_vision4bit（bitsandbytes）/ torchao_int8（torchao）。
#                  **$POLICY_PYTHON に bitsandbytes/torchao が入っていないと
#                  ImportError で落ちる**（意図した fail loudly。
#                  Dockerfile.pi05 の venv には入っているが、host 直の venv には
#                  自分で入れないと無い）
#   QUANT_SERVE_SCOPE  量子化する範囲。既定 vision_only。詳細は
#                  solution/tools/quant/quant_serve_common.py
#
# Reads:
#   - solution/policy_server.py, solution/tools/*.py
#   - work/artifacts/model_weights/**（提出物と同じ重み）
# Generates:
#   - work/gpu_eval/<日時>_<設定>/eval.log・server.log・*.json・videos/
#     run_config.json … このスクリプトが渡した設定値（渡した値のみ。実際に
#     採用された値は server.log の [parc] 行を見ること）
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"

KIT_DIR="${KIT_DIR:-$REPOSITORY_ROOT/work/eval-kit}"
POLICY_PYTHON="${POLICY_PYTHON:-$REPOSITORY_ROOT/.venv/bin/python}"
MAX_STEPS="${MAX_STEPS:-300}"
N_EPISODES="${N_EPISODES:-1}"
PER_CELL="${PER_CELL:-3}"
RECORD="${RECORD:-1}"
PORT="${PORT:-8000}"
CHECK_ONLY=""
DETACH=""

for argument in "$@"; do
    case "$argument" in
        --check)  CHECK_ONLY=1 ;;
        --detach) DETACH=1 ;;
        *) echo "知らない引数: $argument" >&2; exit 2 ;;
    esac
done

# --detach は自分自身を nohup で呼び直す。**ここで再帰を止める。**
if [ -n "$DETACH" ] && [ -z "${PARC_GPU_EVAL_CHILD:-}" ]; then
    mkdir -p "$REPOSITORY_ROOT/work/gpu_eval"
    launcher_log="$REPOSITORY_ROOT/work/gpu_eval/launcher.log"
    PARC_GPU_EVAL_CHILD=1 nohup bash "$0" > "$launcher_log" 2>&1 &
    echo "[gpu-eval] 裏で流し始めた (pid=$!)"
    echo "[gpu-eval] 進み具合:  tail -f $launcher_log"
    echo "[gpu-eval] **端末を閉じても続く。** 止めるには kill $!"
    exit 0
fi

# ---------------------------------------------------------------- 0. 前提

echo "===== 0/6 前提を確かめる ====="

if [ ! -x "$POLICY_PYTHON" ]; then
    echo "エラー: 方策側の python が無い: $POLICY_PYTHON" >&2
    echo "  学習に使った venv を POLICY_PYTHON で指すこと。" >&2
    echo "  例: POLICY_PYTHON=~/src/PARC2026_pre/.venv/bin/python bash $0" >&2
    exit 1
fi

if ! "$POLICY_PYTHON" -c 'import torch, lerobot' 2>/dev/null; then
    echo "エラー: $POLICY_PYTHON に torch か lerobot が入っていない" >&2
    exit 1
fi

"$POLICY_PYTHON" - <<'PY'
import torch
print(f"  方策側 torch {torch.__version__}  CUDA {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("エラー: CUDA が見えない。GPU で推論できないと 84 件は終わらない")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
PY

WEIGHTS="$REPOSITORY_ROOT/work/artifacts/model_weights"
if [ ! -f "$WEIGHTS/model.safetensors" ]; then
    echo "エラー: 重みが無い: $WEIGHTS/model.safetensors" >&2
    echo "  install_weights.py まで済ませること（docs/howto/02_build_submission.md）。" >&2
    exit 1
fi
echo "  重み: $(du -h "$WEIGHTS/model.safetensors" | cut -f1)"

# ---------------------------------------------------------------- 1. EGL

echo
echo "===== 1/6 GPU 描画（EGL）を確かめる ====="
# **ここが最大の罠である。** libglvnd は「どのベンダ実装を使うか」を
# egl_vendor.d の JSON で決める。NVIDIA の登録が無いと Mesa の
# ソフトウェア実装を掴み、**例外も警告も出さずに 11 倍遅くなる。**
# Colab で実際に踏んだ（1 ステップ 300 ms のまま 84 件を流すと 16 時間になる）。
export MUJOCO_GL=egl
ICD=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [ ! -f "$ICD" ]; then
    echo "  $ICD が無い。**Mesa のソフトウェア描画に落ちる恐れがある。**"
    echo "  sudo が使えるなら次を実行すること:"
    echo "    sudo mkdir -p $(dirname $ICD) && sudo tee $ICD >/dev/null <<'EOF'"
    echo '    {"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'
    echo "    EOF"
else
    echo "  $ICD あり"
fi

# ---------------------------------------------------------------- 2. キット

echo
echo "===== 2/6 配布キットを用意する ====="
# **このリポジトリ自身が配布キットの fork である。** だから clone 元は
# GitHub ではなく手元を使う。上流から取り直すと、いま読んでいる
# `pipeline/` と版が食い違いうる（`patch_kit_for_variants.py` の
# アンカーが外れる形で気づくが、90 分を無駄にしてからになる）。
#
# **別の場所に複製するのは、パッチが要るからである。** `compe/` と
# `pipeline/` は提出物では未改変のまま保つ必要があり、
# `patch_kit_for_variants.py` はリポジトリ自身への適用を拒む。
if [ ! -d "$KIT_DIR" ]; then
    echo "  手元から複製する: $KIT_DIR"
    git clone --quiet --shared "$REPOSITORY_ROOT" "$KIT_DIR"
fi

KIT_PYTHON="$KIT_DIR/venv/bin/python"
if [ ! -x "$KIT_PYTHON" ]; then
    echo "  setup.sh を回す（10〜20 分。アセットの取得を含む）"
    setup_python="$(command -v python3.10 || command -v python3)"
    echo "  キット側の python: $setup_python"
    ( cd "$KIT_DIR" && PYTHON="$setup_python" bash setup.sh )
fi
echo "  キット: $KIT_DIR"

# EGL が本当に GPU を掴んだかを、**測って**確かめる。
# ソフトウェア描画だと 1 ステップあたり 10 倍近く遅い。ここで分かれば 90 分を無駄にしない。
echo "  描画の速さを測る（ソフトウェア描画の取りこぼしを防ぐ）"
MUJOCO_GL=egl PYTHONPATH="$KIT_DIR/LIBERO-plus:$KIT_DIR:$KIT_DIR/compe" \
    "$KIT_PYTHON" - <<'PY'
import os, time
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
model = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>"
)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 128, 128)
renderer.update_scene(data)
renderer.render()                                   # 1 枚目は初期化を含む
start = time.time()
for _ in range(50):
    renderer.update_scene(data)
    renderer.render()
per_frame_ms = (time.time() - start) / 50 * 1000
print(f"  1 枚 {per_frame_ms:.1f} ms")
if per_frame_ms > 8.0:
    print("  **遅い。ソフトウェア描画に落ちている可能性がある。**")
    print("  10_nvidia.json を置いてから出し直すこと（上の 1/6 を参照）。")
else:
    print("  GPU 描画で動いている")
PY

# ---------------------------------------------------------------- 3. CSV

echo
echo "===== 3/6 検証用スイートを作る ====="
RUN_TOOLS="$REPOSITORY_ROOT/solution/tools/eval"
CSV="$KIT_DIR/compe/val/VAL_TASKS.csv"
mkdir -p "$(dirname "$CSV")"

"$KIT_PYTHON" "$RUN_TOOLS/make_validation_suite.py" \
    --libero-plus "$KIT_DIR/LIBERO-plus" \
    --exclude "$KIT_DIR/compe/t1/T1_TASKS.csv" \
    --output "$CSV" --per-cell "$PER_CELL" 2>&1 | tail -12

echo
echo "===== 4/6 キットにパッチを当て、CSV を差し替える ====="
"$KIT_PYTHON" "$RUN_TOOLS/patch_kit_for_variants.py" --kit-dir "$KIT_DIR"

# 録画の本体をキットから読める場所へ置く（パッチが import する）。
cp "$RUN_TOOLS/rollout_recorder.py" "$KIT_DIR/pipeline/rollout_recorder.py"

if [ ! -f "$KIT_DIR/compe/t1/T1_TASKS.csv.example" ]; then
    cp "$KIT_DIR/compe/t1/T1_TASKS.csv" "$KIT_DIR/compe/t1/T1_TASKS.csv.example"
fi
cp "$CSV" "$KIT_DIR/compe/t1/T1_TASKS.csv"
echo "  CSV を差し替えた（元は T1_TASKS.csv.example）"

echo
echo "  評価の前に、全部開けるかを確かめる"
"$KIT_PYTHON" "$RUN_TOOLS/check_validation_suite.py" \
    --csv "$CSV" --libero-plus "$KIT_DIR/LIBERO-plus" --kit-dir "$KIT_DIR"

# ---------------------------------------------------------------- 5. 出力先

# 設定から名前を作る。**時計だけだと、どの設定の回か分からなくなる。**
label="n${PARC_N_ACTION_STEPS:-5}_ep${N_EPISODES}_max${MAX_STEPS}"
[ -n "${PARC_EMA_ALPHA:-}" ] && label="${label}_ema${PARC_EMA_ALPHA//./}"
[ -n "${PARC_BLEND_STEPS:-}" ] && label="${label}_blend${PARC_BLEND_STEPS}"
[ -n "${PARC_SAMPLES:-}" ] && label="${label}_s${PARC_SAMPLES}"
[ -n "${QUANT_SERVE_MODE:-}" ] && label="${label}_${QUANT_SERVE_MODE}"
[ -n "$CHECK_ONLY" ] && label="check_${label}"

OUTPUT_DIR="$REPOSITORY_ROOT/work/gpu_eval/$(date +%m%d_%H%M%S)_${label}"
mkdir -p "$OUTPUT_DIR"
echo
echo "===== 5/6 方策サーバーを起動する ====="
echo "  出力先: $OUTPUT_DIR"

if [ -n "$RECORD" ]; then
    export PARC_RECORD_DIR="$OUTPUT_DIR/videos"
    export PARC_RECORD_WHICH="${PARC_RECORD_WHICH:-failures}"
    export PARC_RECORD_MAX="${PARC_RECORD_MAX:-40}"
    echo "  録画あり: which=${PARC_RECORD_WHICH}（既定は失敗した回だけ）"
fi

for name in PARC_N_ACTION_STEPS PARC_SAMPLES PARC_BLEND_STEPS PARC_EMA_ALPHA; do
    if [ -n "$(eval "printf '%s' \"\${$name:-}\"")" ]; then
        export "$name"
        echo "  設定値: $name=$(eval "printf '%s' \"\$$name\"")"
    fi
done

# **このスクリプトが渡した設定値を、実行直後に記録として残す。**
# ここに出るのは「渡した値」であり、空欄は policy_server.py 側の既定が効く。
# 「実際に採用された値」は $SERVER_LOG の [parc] 行（起動後に別途出る）で見る。
CONFIG_JSON="$OUTPUT_DIR/run_config.json"
python3 - "$CONFIG_JSON" <<'PY'
import json
import os
import sys

keys = [
    "KIT_DIR", "POLICY_PYTHON", "MAX_STEPS", "N_EPISODES", "PER_CELL", "RECORD", "PORT",
    "PARC_N_ACTION_STEPS", "PARC_NUM_INFERENCE_STEPS", "PARC_BLEND_STEPS", "PARC_EMA_ALPHA",
    "PARC_SAMPLES", "PARC_RTC", "PARC_RTC_HORIZON", "PARC_RTC_GUIDANCE",
    "PARC_NORMALIZE_INSTRUCTION", "PARC_NORM_EMBED",
    "QUANT_SERVE_MODE", "QUANT_SERVE_SCOPE",
    "PARC_RECORD_DIR", "PARC_RECORD_WHICH", "PARC_RECORD_MAX", "PARC_RECORD_SCALE", "PARC_RECORD_FPS",
]
config = {key: os.environ.get(key) for key in keys}
with open(sys.argv[1], "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
    f.write("\n")
PY
echo "  渡した設定値を書き出した: $CONFIG_JSON"

# **`policy_server.py` は重みを自分の隣から読む**（`Path(__file__).parent /
# "model_weights"`）。環境変数では動かせないので、提出物と同じ並びを作る。
# `vendor/` は置かない — 入れると同梱版の LeRobot を読み、GPU torch の
# 入った venv より古くなる。**測りたいのは推論の中身であって 3.10 互換ではない。**
SERVE_DIR="$OUTPUT_DIR/serve"
mkdir -p "$SERVE_DIR"
cp "$REPOSITORY_ROOT/solution/policy_server.py" "$SERVE_DIR/"
ln -sfn "$WEIGHTS" "$SERVE_DIR/model_weights"

# **推論時量子化スイープ（C4: bnb / C5: torchao。QUANT_SERVE_MODE）の追加ファイル。**
# run_docker_eval.sh と同じ理由で、policy_server.py の隣へ常にコピーしておく
# （未設定なら import されないので無害）。$POLICY_PYTHON に bitsandbytes/torchao が
# 入っていない venv を指すと、実際に QUANT_SERVE_MODE を使ったときに import で
# 落ちる——それは意図した「fail loudly」であり、ここでは黙って揃えたりしない。
for _quant_tool in quant_serve_common.py quant_serve_bnb.py quant_serve_torchao.py; do
    cp "$REPOSITORY_ROOT/solution/tools/quant/$_quant_tool" "$SERVE_DIR/"
done

SERVER_LOG="$OUTPUT_DIR/server.log"
( cd "$SERVE_DIR" && \
  "$POLICY_PYTHON" policy_server.py --port "$PORT" > "$SERVER_LOG" 2>&1 ) &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
    if "$KIT_PYTHON" -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health', timeout=2)" \
        2>/dev/null; then
        echo "  /health OK"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  **サーバーが落ちた。** ログの末尾:" >&2
        tail -40 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done

# **版と実効設定を残す。** 差し替え忘れにその場で気づける。
grep '^\[parc\]' "$SERVER_LOG" | sed 's/^/  /' || \
    echo "  **[parc] 行が無い。** 古い policy_server.py を見ている恐れがある"

# ---------------------------------------------------------------- 6. 評価

echo
echo "===== 6/6 評価を回す ====="
extra_arguments=()
if [ -n "$CHECK_ONLY" ]; then
    extra_arguments+=(--max-tasks 5)
    echo "  **--check なので先頭 5 件だけ。** 5 分ほどで終わる"
else
    echo "  84 件・$N_EPISODES エピソード・$MAX_STEPS 歩（90 分ほど）"
fi

# **全部をログへ流し込むと、画面が何時間も無言になる。**
# キットは 50 ステップごとに `[進捗]` を出しているので、
# **それを画面へ、全文はログへ**という形にする（`-u` と `--line-buffered`
# が無いと数分ぶん溜まってから出る）。終了コードは python のものを使う。
set +e
( cd "$KIT_DIR" && \
  MUJOCO_GL=egl \
  PYTHONPATH="$KIT_DIR/LIBERO-plus:$KIT_DIR:$KIT_DIR/compe" \
  "$KIT_PYTHON" -u -m pipeline \
      --server-url "http://localhost:$PORT" --track track1 \
      --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
      --output-dir "$OUTPUT_DIR" "${extra_arguments[@]}" 2>&1 ) \
  | tee "$OUTPUT_DIR/eval.log" \
  | grep --line-buffered -E '=== タスク|\[進捗\]|\[結果\]|\[録画\]|ERROR|Traceback' \
  | sed -u 's/^[0-9-]* [0-9:]* \[[A-Z]*\] [a-z.]*: *//'
echo "  終了コード=${PIPESTATUS[0]}"
set -e

# ---------------------------------------------------------------- 判定

echo
echo "===== 判定 ====="
# **CSV を明示して渡す。** 摂動カテゴリは分類表（CSV の category 列）でしか
# 分からない。視点・指示文・初期姿勢の摂動は variant 名がどれも同じ形なので、
# CSV を見失うと 3 カテゴリが 1 行に束ねられ、**学習の的を決める表が静かに
# 意味を失う。** 探させずに渡す。
"$KIT_PYTHON" "$RUN_TOOLS/summarize_gpu_eval.py" \
    --output-dir "$OUTPUT_DIR" --csv "$CSV"
