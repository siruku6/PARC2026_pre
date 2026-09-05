#!/usr/bin/env bash
# 016: 1 タスクの逆向き bddl ＋ 逆向き初期状態を作る（docker run 1 本のみ）。
# 引数: $1 = タスクの stem
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
S="${1:?stem が必要}"
L10="$REPO/work/probe_016/l10"
R="$REPO/work/probe_016/reverse"
mkdir -p "$R"
STAGE=$(mktemp -d)
cp "$L10/bddl_files/$S.bddl" "$L10/init_files/$S.pruned_init" "$STAGE/"
docker run --rm \
  -v "$STAGE:/in:ro" \
  -v "$R:/out" \
  -e LIBERO_ROOT=/workspace/LIBERO-plus \
  -e MUJOCO_GL=egl \
  -v "$REPO/solution/tools/probe:/app:ro" \
  parc2026-final python -u /app/_016_make_reverse.py --in /in --out /out --stem "$S"
rm -rf "$STAGE"