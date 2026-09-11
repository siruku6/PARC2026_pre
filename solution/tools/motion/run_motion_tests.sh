#!/usr/bin/env bash
# 共有の通過点生成の試験を、robosuite と ruckig が居るコンテナで打つ。
#
#   bash solution/tools/motion/run_motion_tests.sh
#
# **ホストの venv には robosuite が無いので、そのまま pytest を打つと
# 028 との一致（`TestParityWith028`）が skip になる。**そこが本丸なので、
# 共有化に手を入れたらこちらで打つこと。
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
VENDOR="${VENDOR:-$REPO/work/vendor_py310}"

docker run --rm \
  -v "$REPO/solution/tools/track3_reversal:/app:ro" \
  -v "$REPO/solution/tools:/tools:ro" \
  -v "$REPO/tests:/tests:ro" \
  -v "$VENDOR:/vendor:ro" \
  -e PYTHONPATH=/app:/tools:/vendor:/workspace/LIBERO-plus:/workspace:/workspace/compe \
  -e LIBERO_ROOT=/workspace/LIBERO-plus -e MUJOCO_GL=egl \
  -w /workspace \
  parc2026-final python -m pytest /tests/test_motion_waypoints.py -q 2>&1 \
  | grep -vE '^\[libero\]|WARNING|robosuite|CUDA|NVIDIA|Copyright|licen|^=+$|^$|Toolkit|docs\.nvidia|container image'
