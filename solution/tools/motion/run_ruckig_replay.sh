#!/usr/bin/env bash
# 028 の逆向きデモ生成を、通過点だけ Ruckig に差し替えて回す。
#
#   bash solution/tools/motion/run_ruckig_replay.sh [shifts] [episodes]
#
# **`run_gripper_shift_sweep.sh` と同じ環境変数がそのまま効く。**
# 違うのは 3 つだけ:
#   MOTION_PLANNER=ruckig|minjerk  通過点の作り方（既定 ruckig）
#   MOTION_EXIT_SPEED=0.25         区間の出口に残す速さ
#   OUT=work/probe_ruckig          出力先（既定はここ）
#
# **`_t3_container.sh` を書き換えずに済ませるため、コンテナの起動はここに持つ。**
# 028 の側は今までどおり `run_gripper_shift_sweep.sh` で動く。
#
# ruckig は 69 GB のイメージに焼かず、ホストへ入れたものを読み取り専用で挿す:
#   pip install --target work/vendor_py310 --python-version 310 \
#       --only-binary=:all: --platform manylinux_2_28_x86_64 ruckig
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT="${OUT:-$REPO/work/probe_ruckig}"
VENDOR="${VENDOR:-$REPO/work/vendor_py310}"
export REPO OUT

# タスク資産の対応表と QUIET は 028 のものをそのまま使う（重複させない）。
source "$REPO/solution/tools/track3_reversal/_t3_container.sh"

if [[ ! -d "$VENDOR" ]]; then
  echo "ruckig が $VENDOR に無い。上のコメントの pip install を先に打つこと" >&2
  exit 1
fi

mkdir -p "$OUT"
motion_env=()
for name in MOTION_PLANNER MOTION_EXIT_SPEED MOTION_ACCEL_POS MOTION_ACCEL_ROT \
            MOTION_JERK_POS MOTION_JERK_ROT MOTION_INITIAL_ACCEL_RATIO \
            APPROACH_LIMIT_POS APPROACH_LIMIT_ROT APPROACH_RESPONSE_POS_M \
            APPROACH_RESPONSE_ROT_RAD APPROACH_MIN_WAYPOINTS; do
  [[ -n "${!name:-}" ]] && motion_env+=(-e "$name=${!name}")
done

docker run --rm \
  -v "$REPO/solution/tools/track3_reversal:/app:ro" \
  -v "$REPO/solution/tools:/tools:ro" \
  -v "$VENDOR:/vendor:ro" \
  -v "$OUT:/out" \
  -e LIBERO_ROOT=/workspace/LIBERO-plus -e MUJOCO_GL=egl \
  -e PYTHONPATH=/app:/tools:/vendor:/workspace/LIBERO-plus:/workspace:/workspace/compe \
  ${motion_env[@]+"${motion_env[@]}"} \
  -w /workspace \
  parc2026-final python -u /tools/motion/track3_ruckig_replay.py \
    --ep-npz "/out/${EP_NPZ:-ep_data.npz}" \
    --forward-bddl "$FWD_BDDL" --forward-inits "$FWD_INITS" \
    --t3-bddl "$T3_BDDL" --t3-inits "$T3_INITS" \
    "--shifts=${1:-0}" \
    ${2:+--episodes "$2"} \
    --mode "${MODE:-phased}" \
    --init-source "${INIT_SOURCE:-t3}" \
    --grasp-frame "${GRASP_FRAME:-clamped}" \
    --standoff "${STANDOFF:-0.06}" \
    --lift "${LIFT:-0.08}" \
    --release "${RELEASE:-contact}" \
    --settle "${SETTLE:-80}" \
    --max-tail "${MAX_TAIL:-5}" \
    --max-steps "${MAX_STEPS:-300}" \
    --trace-release "${TRACE:-0}" \
    ${LEVEL:+--level-before-release} \
    --max-approach "${MAX_APPROACH:-200}" \
    ${ROT_LIMIT:+--rot-limit "$ROT_LIMIT"} \
    --rot-tol-deg "${ROT_TOL_DEG:-5}" \
    --object-symmetry "${SYMMETRY:-1}" \
    --grasp-style "${GRASP_STYLE:-transfer}" \
    --t3-init-index "${T3_INIT:-0}" \
    ${RENDER:+"--render-shifts=$RENDER"} \
    ${WRITE_NPZ:+"--write-npz=/out/$WRITE_NPZ"} \
    --out /out 2>&1 | grep -vE "$QUIET"

docker run --rm -v "$OUT:/out" --entrypoint chown parc2026-final \
  -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
echo "=== 出力: $OUT ==="
