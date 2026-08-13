#!/usr/bin/env bash
# RunPod のポッド上で、solution/train/Dockerfile.pi05 と同じ依存関係を直接入れる。
# Docker のビルド・レジストリへの push は不要—— RunPod の公式テンプレート
# （例: "RunPod PyTorch 2.x"）に SSH で入り、このスクリプトを1回実行するだけでよい。
#
# 前提: RunPod のポッドは既定で root として動くので、solution/train/Dockerfile.pi05
# で行った「非rootで動かすための回避（/hf_home・HOME=/tmp・USER=trainer）」は不要。
#
# 使い方（ポッドに SSH で入ったあと）:
#   bash runpod_setup.sh
#
# 実行後、次のように学習を始める（80GB GPU 前提のフルパラメータ学習）。
#
#   export HF_HOME=/workspace/hf_home   # 後述。永続ボリュームを使うなら変える
#   hf auth login
#   cd /workspace
#   python train_pi05.py --run-name pi05_full --steps 30000 --batch-size 64 \
#       --num-workers 8 --save-freq 5000 \
#       --freeze-vision-encoder false --train-expert-only false
#
# **チェックポイントは1個 11GB 前後、間引きが無いと積み上がる**（ローカルと同じ問題）。
# 別ターミナル（同じポッドに2つ目の SSH セッションを開く）で
#   bash prune_checkpoints.sh /out/pi05_full 2
# を並行して回すこと（RunPod のボリューム容量に応じて KEEP を調整）。
set -euo pipefail

TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_VERSION="2.11.0"
TORCHCODEC_VERSION="0.11.1"
TORCHVISION_VERSION="0.26.0"
LEROBOT_TAG="v0.6.0"
LEROBOT_DIR="/opt/lerobot"

echo "[setup] apt 依存関係"
apt-get update -qq
apt-get install -y --no-install-recommends git build-essential ffmpeg >/dev/null

echo "[setup] torch / torchcodec / torchvision（版を固定。RunPod テンプレートの"
echo "        プリインストール版と食い違うと import 時に落ちるため、必ず上書きする）"
pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
pip install --no-cache-dir "torchcodec==${TORCHCODEC_VERSION}" --index-url "${TORCH_INDEX}"

echo "[setup] NPP（torchcodec のデコードに要る CUDA ライブラリ）"
pip install --no-cache-dir "nvidia-npp-cu12"
SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NPP_SO="$(find "${SITE_PACKAGES}/nvidia" -name 'libnppicc.so*' -print -quit 2>/dev/null || true)"
if [ -n "${NPP_SO}" ]; then
    dirname "${NPP_SO}" > /etc/ld.so.conf.d/nvidia-npp.conf
    ldconfig
    ldconfig -p | grep -q libnppicc.so.12 && echo "[setup] NPP 登録 OK" \
        || { echo "[setup] **NPP の登録に失敗**。ldconfig -p | grep nppicc で確認すること"; exit 1; }
else
    echo "[setup] **libnppicc.so が見つからない**。site-packages: ${SITE_PACKAGES}"
    exit 1
fi

pip install --no-cache-dir "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX}"

echo "[setup] lerobot ${LEROBOT_TAG}（extras: training, pi）"
if [ ! -d "${LEROBOT_DIR}" ]; then
    git clone --quiet --depth 1 --branch "${LEROBOT_TAG}" \
        https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
fi
pip install --no-cache-dir -e "${LEROBOT_DIR}[training,pi]"
pip uninstall -y -q torchao || true

echo "[setup] 版の対を確認"
python3 -c "
import torch, torchcodec, torchvision
from torchcodec.decoders import VideoDecoder
torchvision.ops.nms
print(f'torch {torch.__version__} / torchcodec {torchcodec.__version__} / torchvision {torchvision.__version__} 整合')
"

echo "[setup] 完了。次は:"
echo "  1. hf auth login  （gemma ライセンス同意済みのアカウントで）"
echo "  2. train.py / train_pi05.py / prune_checkpoints.sh を /workspace へ転送"
echo "     （scp か、リポジトリを git clone/pull できるなら git で）"
echo "  3. python /workspace/train_pi05.py --steps 30000 --batch-size 64 \\"
echo "         --num-workers 8 --save-freq 5000 \\"
echo "         --freeze-vision-encoder false --train-expert-only false"
