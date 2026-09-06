#!/usr/bin/env bash
# 016 下調べ: フォークの libero_10 bddl/init をホストへ取り出す（docker run を script 経由で）。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUT:-$REPO/work/probe_016/l10}"
mkdir -p "$OUT"
cp "$REPO/solution/tools/probe/_016_extract_l10.py" /tmp/_016_extract_l10.py

docker run --rm \
  -v "$REPO/solution/tools/probe/_016_extract_l10.py:/app/_016_extract_l10.py:ro" \
  -v "$OUT:/out" \
  -e OUT_DIR=/out \
  parc2026-final \
  python /app/_016_extract_l10.py

echo "ホスト出力: $OUT"