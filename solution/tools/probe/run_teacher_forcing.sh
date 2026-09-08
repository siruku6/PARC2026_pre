#!/usr/bin/env bash
# `teacher_forcing.py` を学習イメージで回す（lerobot と torch が要る）。
#
#   bash solution/tools/probe/run_teacher_forcing.sh <重み> <データセット> [出力json]
#
# **データセットは学習に使ったものを渡す。**逆向きデータそのものを渡せば、
# 「35 周見たデータを再現できるか」が測れる。
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
W="${1:?重みのディレクトリを渡すこと}"
DS="${2:?データセットのディレクトリを渡すこと}"
JSON="${3:-}"
IMAGE="${IMAGE:-parc2026-train-pi05}"
case "$W" in /*) ;; *) W="$REPO/$W" ;; esac
case "$DS" in /*) ;; *) DS="$REPO/$DS" ;; esac
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -e HF_HOME=/hf_home -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$REPO:/repo" -v "$W:/weights:ro" -v "$DS:/dataset:ro" \
    -v "$HOME/.cache/huggingface:/hf_home" \
    --entrypoint python "$IMAGE" \
    -u /repo/solution/tools/probe/teacher_forcing.py /weights /dataset \
    ${EPISODES_PER_TASK:+--episodes-per-task "$EPISODES_PER_TASK"} \
    ${OFFSETS:+--offsets "$OFFSETS"} \
    ${JSON:+--json-out "/repo/$JSON"}
