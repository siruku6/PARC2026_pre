#!/usr/bin/env bash
# tests/test_camera_dropout.py を、常駐コンテナ quant_train の本物の torch と
# lerobot で走らせる。
#
# **なぜスクリプトが要るのか。** quant_train が見えているのは
# `work/train`（コンテナの `/out`）だけで、`solution/train` も `tests/` も
# マウントされていない。だから走らせるには、そのつど必要なものを `/out` の下へ
# 写す必要がある——**2 回以上打つものなので 1 行にした**（`.claude/CLAUDE.md`）。
#
# 使い方（リポジトリ直下で）:
#     bash solution/tools/train/run_camera_dropout_test.sh
#
# この機械の venv には torch も lerobot も入っていないため、
# `pytest tests/test_camera_dropout.py` を直接打つと全部 skip になる。
set -euo pipefail

CONTAINER=quant_train
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
STAGE="$ROOT/work/train/_camera_dropout_check"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "常駐コンテナ $CONTAINER が起動していない。" >&2
    echo "  docker ps -a で確認し、起動をユーザーに依頼すること" >&2
    echo "  （このマシンでは Claude から docker run はできない）。" >&2
    exit 1
fi

# **毎回作り直す。**前回の写しが残っていると、直したはずのコードで通ったのか
# 古い写しで通ったのかが区別できない。
python3 -c "import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$STAGE"
mkdir -p "$STAGE/tests"
cp -r "$ROOT/solution" "$STAGE/solution"
cp "$ROOT/tests/test_camera_dropout.py" "$STAGE/tests/"

# **pytest はイメージに入っていない。**入れても `docker commit` しない限り
# コンテナを作り直した時点で消えるので、無ければそのつど入れる。
if ! docker exec "$CONTAINER" python -m pytest --version >/dev/null 2>&1; then
    echo "コンテナに pytest が無いので入れる"
    docker exec "$CONTAINER" pip install --quiet pytest
fi

docker exec -w /out/_camera_dropout_check "$CONTAINER" \
    python -m pytest tests/test_camera_dropout.py -v
