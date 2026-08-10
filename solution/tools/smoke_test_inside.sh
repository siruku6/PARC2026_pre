#!/usr/bin/env bash
# 疎通確認の本体。**コンテナの中で走らせる。**
#
# ホストからは smoke_test_suite.sh が呼ぶ。すでに起動しているコンテナで
# 試すこともできる。
#
#   docker exec -e TASKS=5 -e MAX_STEPS=20 \
#       -v ... （実行中のコンテナには mount を足せないので、
#              tools を /host/tools に見せた状態で起動しておくこと）
#       <container> bash /host/tools/smoke_test_inside.sh
#
# 環境変数（省略時の既定）
#   TASKS=5        回すタスク数。CSV の先頭からこの件数
#   MAX_STEPS=20   1 エピソードの上限歩数。**成功させる必要はない**
#   PER_CELL=3     スイート × カテゴリあたりの本数（CSV 全体は 84 件になる）
#   TOOLS_DIR=/host/tools   スクリプトの置き場
#   OUT_DIR=/out            出力先
set -euo pipefail

KIT="${KIT:-/workspace}"
PY="${PY:-$KIT/venv/bin/python}"
TOOLS_DIR="${TOOLS_DIR:-/host/tools}"
OUT_DIR="${OUT_DIR:-/out}"
TASKS="${TASKS:-5}"
MAX_STEPS="${MAX_STEPS:-20}"
PER_CELL="${PER_CELL:-3}"

mkdir -p "$OUT_DIR"

echo "[1/6] 検証用スイートの CSV を作る"
"$PY" "$TOOLS_DIR/make_validation_suite.py" \
    --libero-plus "$KIT/LIBERO-plus" \
    --exclude "$KIT/compe/t1/T1_TASKS.csv" \
    --output "$OUT_DIR/VAL_TASKS.csv" --per-cell "$PER_CELL" 2>&1 | tail -12

echo
echo "[2/6] キットにパッチを当てる"
"$PY" "$TOOLS_DIR/patch_kit_for_variants.py" --kit-dir "$KIT"

echo
echo "[3/6] CSV を差し替える"
if [ ! -f "$KIT/compe/t1/T1_TASKS.csv.example" ]; then
    cp "$KIT/compe/t1/T1_TASKS.csv" "$KIT/compe/t1/T1_TASKS.csv.example"
fi
cp "$OUT_DIR/VAL_TASKS.csv" "$KIT/compe/t1/T1_TASKS.csv"
echo "  差し替えた（元は T1_TASKS.csv.example）"

echo
echo "[4/6] 全部開けるか、評価の前に確かめる"
"$PY" "$TOOLS_DIR/check_validation_suite.py" \
    --csv "$OUT_DIR/VAL_TASKS.csv" --libero-plus "$KIT/LIBERO-plus" \
    --kit-dir "$KIT" --max-tasks "$TASKS"

echo
echo "[5/6] ランダム方策のサーバーを起動する"
# **学習した重みは使わない。** 確かめたいのはタスクを読み込めるかであり、
# 方策の質とは無関係である。配布テンプレートは一様乱数を返すだけなので、
# 起動が速く CPU 推論も要らない。
"$PY" "$KIT/submission_template/policy_server.py" --port 8000 \
    > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    if "$PY" -c 'import urllib.request; urllib.request.urlopen("http://localhost:8000/health", timeout=2)' 2>/dev/null; then
        echo "  /health OK"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  **サーバーが落ちた。** ログの末尾:"
        tail -20 "$OUT_DIR/server.log"
        exit 1
    fi
    sleep 1
done

echo
echo "[6/6] 評価を回す（$TASKS タスク・1 エピソード・$MAX_STEPS 歩）"
cd "$KIT"
set +e
"$PY" -m pipeline --server-url http://localhost:8000 --track track1 \
    --n-episodes 1 --max-steps "$MAX_STEPS" --max-tasks "$TASKS" \
    --output-dir "$OUT_DIR" > "$OUT_DIR/eval.log" 2>&1
echo "  終了コード=$?"
set -e
echo "  [結果] 行: $(grep -c '\[結果\]' "$OUT_DIR/eval.log" || true)"
