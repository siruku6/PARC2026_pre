#!/usr/bin/env bash
# 生成系の道具を評価イメージの中で走らせる包み。
# 物理だけを回すので描画は要らないが、EGL を渡しておけば描画も通る。
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${IMAGE:-parc2026-final}"
GL="${GL:-egl}"

# **GPU の無い台では `GPUS=` を渡して外す。**付けたまま GPU が無いと
# `could not select device driver "" with capabilities: [[gpu]]` で起動しない。
# データの組み上げ（`merge_many.py`・`rebuild_dataset.py`）は GPU を使わないので、
# 外しても結果は変わらない。物理を回す道具は GPU が要る。
#
# **渡されなければ自分で見る。**`GPUS` を明示したときはそれに従い（空なら付けない）、
# 何も言われていないときだけ nvidia ランタイムの有無で決める。
# **`:-` ではなく `+set` で見ているのは、`:-` が「空」も「未設定」と同じ扱いに
# するからである**（`GPUS=` を渡しても既定へ戻ってしまう。2026-09-18 に実際に踏んだ）。
if [ -z "${GPUS+set}" ]; then
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        GPUS="--gpus all"
    else
        GPUS=""
        echo "run_in_container.sh: nvidia ランタイムが見えないので --gpus all を付けない" >&2
    fi
fi

# コンテナは root で走るので、書いたファイルは root のものになる。
# **`CHOWN_PATH` を渡すと、走り終えたあとに持ち主を戻す**（`run_final_eval.sh` と同じ手）。
CHOWN_PATH="${CHOWN_PATH:-}"

# **摘みは環境変数で渡す。**`GRASP_CLEARANCE_ENOUGH`（掴む方位の余裕の頭打ち）と、
# `motion/envelope.py` が読む `APPROACH_*`・`MOTION_*` を素通しにする。
# 名前を挙げた変数だけを渡すので、余計なものは入らない。
PASS_THROUGH=()
for name in GRASP_CLEARANCE_ENOUGH \
            APPROACH_LIMIT_POS APPROACH_LIMIT_ROT \
            APPROACH_RESPONSE_POS_M APPROACH_RESPONSE_ROT_RAD \
            MOTION_ACCEL_POS MOTION_ACCEL_ROT MOTION_JERK_POS MOTION_JERK_ROT \
            MOTION_INITIAL_ACCEL_RATIO \
            MOTION_GLIDE MOTION_GLIDE_EXIT_SPEED MOTION_GLIDE_GAIN \
            MOTION_GLIDE_LAG ; do
    if [ -n "${!name:-}" ]; then
        PASS_THROUGH+=(-e "$name=${!name}")
    fi
done

# **ruckig は 69 GB のイメージに焼かない。**ホストへ展開したものを読み取り専用で
# 挿し、`PYTHONPATH` の先頭へ足す（`run_ruckig_replay.sh` と同じ手）。
# 入れ方:
#   pip install --target work/vendor_py310 --python-version 310 \
#       --only-binary=:all: --platform manylinux_2_28_x86_64 ruckig
# **無くても落ちない。**`motion/otg.py` が自前の台形＋S 字に切り替わる
# （時間最適ではないぶん数コマ長いだけで、指令の形は同じ）。
VENDOR="${VENDOR:-$REPOSITORY_ROOT/work/vendor_py310}"
VENDOR_MOUNT=()
if [ -d "$VENDOR" ]; then
    VENDOR_MOUNT=(-v "$VENDOR:/vendor:ro"
                  -e "PYTHONPATH=/vendor:/workspace/LIBERO-plus:/workspace:/workspace/compe")
fi

status=0
# **`-i` を付けてある。**`python - <<PY` の形で渡す使い方があるので、
# 標準入力を繋いでおく（`-t` は付けない。走らせる先は端末ではない）。
docker run --rm -i ${GPUS} \
    -e "MUJOCO_GL=${GL}" -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    "${PASS_THROUGH[@]+"${PASS_THROUGH[@]}"}" \
    "${VENDOR_MOUNT[@]+"${VENDOR_MOUNT[@]}"}" \
    -v "$REPOSITORY_ROOT:/workspace/repo" \
    -w /workspace/repo \
    --entrypoint python \
    "$IMAGE" "$@" || status=$?

if [ -n "$CHOWN_PATH" ]; then
    docker run --rm -v "$CHOWN_PATH:/out" --entrypoint chown "$IMAGE" \
        -R "$(id -u):$(id -g)" /out
fi
exit "$status"
