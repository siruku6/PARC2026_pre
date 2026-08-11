#!/usr/bin/env bash
# 素の重み（追加学習なし）を HuggingFace から取り寄せる。
#
# **本番スコア 0.1409 を出したのはこの重みである。** 3 回目までの提出は
# すべて `lerobot/smolvla_libero_plus@7bb70aa5` をそのまま使っており、
# こちらで学習した部分は 1 つも無い。**したがって失われていない。**
# revision を固定して取り直せば、いつでも同じものが手に入る。
#
# 学習後の重みと比べるときに使う。
#
#   bash solution/tools/fetch_base_weights.sh
#   WEIGHTS=work/artifacts/model_weights_base LABEL=base \
#       bash solution/tools/run_docker_eval.sh --detach
#
# 環境変数
#   OUTPUT  置き場所（既定 work/artifacts/model_weights_base）
#   IMAGE   使うイメージ（既定 parc2026-train。huggingface_hub が入っている）
#
# Generates:
#   - work/artifacts/model_weights_base/**（約 869 MB）
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${OUTPUT:-$REPOSITORY_ROOT/work/artifacts/model_weights_base}"
IMAGE="${IMAGE:-parc2026-train}"

# **配布ノートブックが指している revision である。** ここを変えると
# 3 回目までの提出とは別の重みになる。
REVISION="7bb70aa5bc92b82c9239142775d3a173103567ff"

if [ -f "$OUTPUT/model.safetensors" ] && [ -d "$OUTPUT/vlm" ]; then
    echo "[fetch] すでにある: $OUTPUT"
    echo "        $(du -h "$OUTPUT/model.safetensors" | cut -f1)"
    exit 0
fi

mkdir -p "$OUTPUT"
echo "[fetch] lerobot/smolvla_libero_plus@${REVISION:0:8} を取り寄せる（869 MB）"

docker run --rm \
    -v "$OUTPUT:/out" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e "REVISION=$REVISION" \
    "$IMAGE" python - <<'PY'
import os
from huggingface_hub import snapshot_download

revision = os.environ["REVISION"]

# `eval/`（評価動画 402 ファイル・25.6 MB）は推論に使わないので除く。
snapshot_download(
    "lerobot/smolvla_libero_plus",
    revision=revision,
    local_dir="/out",
    ignore_patterns=["eval/*", "eval/**"],
)

# **VLM の本体重みは要らない。** ポリシー側のチェックポイントに入っている。
# 設定とトークナイザだけを置く（`policy_server.py` がここを見る）。
snapshot_download(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    local_dir="/out/vlm",
    allow_patterns=["*.json", "*.txt", "*.model", "tokenizer*"],
)
print("取り寄せた")
PY

echo
echo "[fetch] 置いた: $OUTPUT"
echo "        model.safetensors  $(du -h "$OUTPUT/model.safetensors" | cut -f1)"
echo "        sha256 $(sha256sum "$OUTPUT/model.safetensors" | cut -c1-32)"
echo
echo "これで測る:"
echo "  WEIGHTS=$OUTPUT LABEL=base \\"
echo "      bash solution/tools/run_docker_eval.sh --detach"
