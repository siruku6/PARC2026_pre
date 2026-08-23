#!/usr/bin/env bash
# 検証用スイートが「最後まで走るか」だけを、コンテナ内で短時間に確かめる。
#
# **成功率は見ない。** 見るのは次の 3 点である。
#   - タスクの読み込みで例外が出ないか（`_view_` を含む変種で落ちていた）
#   - `[結果]` 行が指定した本数そろうか
#   - JSON に `n_total_episodes` が入るか（例外時は入らない）
#
# **方策は配布テンプレートのランダム行動を使う。** 学習した重みは要らない。
# 落ちていたのはタスクを読み込む時点であり、方策とは無関係である。
# 重み 869MB の読み込みと CPU 推論（1 手 20 秒超）を省けるので、
# 同じ確認が数分で終わる。
#
# ホストから実行する。イメージは `docker build -t parc2026 .` で作ったもの。
#
#   bash solution/tools/eval/smoke_test_suite.sh              # 5 タスク・20 ステップ
#   TASKS=7 MAX_STEPS=40 bash solution/tools/eval/smoke_test_suite.sh
#   RECORD=1 bash solution/tools/eval/smoke_test_suite.sh     # 録画の配線も確かめる
#
# Reads:
#   - solution/tools/ の 4 スクリプト（コンテナへ読み取り専用で渡す）
#   - コンテナ内の /workspace（配布キットと LIBERO-plus）
# Generates:
#   - work/smoke/VAL_TASKS.csv・eval.log・server.log・server_8000.json
#   - work/smoke/videos/*.mp4（RECORD=1 のとき）
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${IMAGE:-parc2026}"
TASKS="${TASKS:-5}"
MAX_STEPS="${MAX_STEPS:-20}"
PER_CELL="${PER_CELL:-3}"
RECORD="${RECORD:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPOSITORY_ROOT/work/smoke}"

mkdir -p "$OUTPUT_DIR"
echo "[smoke] イメージ $IMAGE / $TASKS タスク / $MAX_STEPS 歩 / 出力 $OUTPUT_DIR"

# コンテナ内の手順は 1 つのスクリプトに置いてある。
# **ここでヒアドキュメントを入れ子にしない。** 引用の解釈が二重になって
# 壊れやすく、壊れても静かに間違った文字列が書かれるため。
docker run --rm \
    -e TASKS="$TASKS" -e MAX_STEPS="$MAX_STEPS" -e PER_CELL="$PER_CELL" \
    -e RECORD="$RECORD" \
    -v "$REPOSITORY_ROOT/solution/tools:/host/tools:ro" \
    -v "$OUTPUT_DIR:/out" \
    "$IMAGE" bash /host/tools/eval/smoke_test_inside.sh

echo
echo "===== 判定 ====="
python3 "$REPOSITORY_ROOT/solution/tools/eval/summarize_smoke_test.py" \
    --output-dir "$OUTPUT_DIR" --expected-tasks "$TASKS"
