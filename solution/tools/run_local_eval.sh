#!/usr/bin/env bash
# 配布キットの評価パイプラインを、提出物のポリシーサーバーに対して回す。
#
# 本番と同じ 2 プロセス構成を再現する。
#   - ポリシーサーバー: 提出物専用 venv（--system-site-packages）で起動する
#   - 評価パイプライン: 配布キットの venv で起動する
# 採点環境に合わせて外部通信を遮断した状態（HF_HUB_OFFLINE=1）で回すため、
# Hub へ取りに行く実装が混ざっていればここで必ず落ちる。
#
# コンテナ内で実行すること。ホストでは LIBERO の依存が揃わない。
#
#   N_EPISODES=1 MAX_STEPS=100 bash run_local_eval.sh --max-tasks 1   # 疎通確認
#   N_EPISODES=5 MAX_STEPS=600 bash run_local_eval.sh                 # 本測定
#
# 追加の引数は python -m pipeline へそのまま渡す。
set -euo pipefail

SUBMISSION_DIR="${SUBMISSION_DIR:-/sub}"
SUBMISSION_PYTHON="${SUBMISSION_PYTHON:-/opt/subvenv2/bin/python}"
KIT_DIR="${KIT_DIR:-/workspace}"
KIT_PYTHON="${KIT_PYTHON:-/workspace/venv/bin/python}"
PORT="${PORT:-8000}"
N_EPISODES="${N_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-600}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results}"
SERVER_LOG="${SERVER_LOG:-$SUBMISSION_DIR/server.log}"

# /act 1 リクエストの上限。本番は 10 秒で固定であり、ここを緩めても
# 本番が緩むわけではない。CPU のコンテナでは推論が 20 秒を超えるため、
# 「観測変換が正しいか」を確かめる目的のときだけ大きくする。
TIMEOUT="${TIMEOUT:-10}"

# 推論時の後処理。いずれも空なら無効で、素の挙動になる。
# 詳細は solution/README.md の「推論時の後処理」を参照。
#   PARC_N_ACTION_STEPS=25 bash run_local_eval.sh   # 閉ループ寄りにする
#   PARC_SAMPLES=3         bash run_local_eval.sh   # chunk を 3 回引いて典型を選ぶ
#   PARC_BLEND_STEPS=3     bash run_local_eval.sh   # chunk 境界の段差を消す
#   PARC_EMA_ALPHA=0.3     bash run_local_eval.sh   # 常時の指数平滑
#
# 呼び出し側で VAR=値 の形で渡せば子プロセスへ引き継がれるが、
# 明示的に export して意図を残す。
for _name in PARC_N_ACTION_STEPS PARC_SAMPLES PARC_BLEND_STEPS PARC_EMA_ALPHA; do
    if [ -n "$(eval "printf '%s' \"\${$_name:-}\"")" ]; then
        export "$_name"
    fi
done

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTHONPATH="$KIT_DIR/LIBERO-plus:$KIT_DIR:$KIT_DIR/compe"
export HF_HOME="${HF_HOME:-$SUBMISSION_DIR/empty_hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "$SUBMISSION_DIR"
"$SUBMISSION_PYTHON" policy_server.py --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[eval] ポリシーサーバーを起動 (pid=$SERVER_PID, log=$SERVER_LOG)"

for elapsed in $(seq 1 300); do
    if "$KIT_PYTHON" -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health', timeout=2)" \
        2>/dev/null; then
        echo "[eval] /health が応答しました (${elapsed}秒)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[eval] サーバーが起動に失敗しました。ログの末尾:"
        tail -40 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

echo "[eval] 評価を開始 (n_episodes=$N_EPISODES, max_steps=$MAX_STEPS, timeout=${TIMEOUT}s)"
if [ "$TIMEOUT" != "10" ]; then
    echo "[eval] 注意: /act のタイムアウトを本番（10秒）から緩めている。スコアの参考にはならない。"
fi
if [ -n "${PARC_N_ACTION_STEPS:-}" ]; then
    echo "[eval] n_action_steps=${PARC_N_ACTION_STEPS}（既定の 50 から変更している）"
fi
if [ -n "${PARC_SAMPLES:-}" ]; then
    echo "[eval] samples=${PARC_SAMPLES}（chunk を複数回引いて medoid を選ぶ。推論時間はこの倍数）"
fi
if [ -n "${PARC_BLEND_STEPS:-}" ]; then
    echo "[eval] blend_steps=${PARC_BLEND_STEPS}（chunk 境界の平滑化を有効化している）"
fi
if [ -n "${PARC_EMA_ALPHA:-}" ]; then
    echo "[eval] ema_alpha=${PARC_EMA_ALPHA}（常時の指数平滑を有効化している）"
fi
cd "$KIT_DIR"
"$KIT_PYTHON" -m pipeline \
    --server-url "http://localhost:$PORT" \
    --track track1 \
    --n-episodes "$N_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --timeout "$TIMEOUT" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
