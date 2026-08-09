# Ubuntu GPU マシンで LoRA 学習を回すためのイメージ。
#
#   docker build -t parc2026-train solution/train
#   docker run --gpus all --rm -it \
#       -v "$PWD/work/train:/out" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#       parc2026-train
#
# 採点環境（Python 3.10 / CUDA 13）とは別物である。**学習側は LeRobot が要求する
# Python 3.12 をそのまま使う。** 提出物の 3.10 互換は vendor_lerobot.py が別途担保しており、
# 学習と推論で同じ重みが読めればよい。
#
# CUDA は torch の wheel に同梱されるものを使うため、CUDA 入りのベースイメージは要らない。
# ホスト側に NVIDIA ドライバと nvidia-container-toolkit があればよい。
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# torch の wheel をどの index から取るか。ホストのドライバに合わせて変える。
#   cu128 … 既定。LeRobot v0.6.0 の uv.lock が torch 2.11.0+cu128 を解決する
#   cu130 … より新しいドライバ（採点イメージはこちら）
#   cpu   … GPU 無しで配線だけ確認するとき
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# **torch と torchcodec の版は対で決まる。片方だけ動かしてはいけない。**
#
# LeRobot は torch>=2.7,<2.12 と torchcodec>=0.3.0,<0.12.0 としか宣言していない。
# 一方 **torchcodec は依存を一切宣言しない**（uv.lock で確認済み）。torch の ABI に
# 直接リンクしているにもかかわらず、である。つまり pip はこの組み合わせの
# 不整合を検出できず、**インストールは成功して実行時に落ちる。**
#
# 版は LeRobot v0.6.0 の uv.lock が解決した組み合わせに合わせる。学習
# ノートブックが Colab のプリインストール版で動いていたのも同じ対である。
ARG TORCH_VERSION=2.11.0
ARG TORCHCODEC_VERSION=0.11.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX}

# lerobot より先に入れる。先に固定しておけば、後続の解決は制約を満たすので
# 触られない。あとから入れると <0.12.0 の最新が選ばれ、対が崩れる。
RUN pip install --no-cache-dir "torchcodec==${TORCHCODEC_VERSION}"

# 学習用ノートブックと同じタグ・同じ extras。ここを変えると重みの互換性が崩れる。
# training は dataset を含むため、dataset を重ねて書く必要はない。
ARG LEROBOT_TAG=v0.6.0
RUN git clone --quiet --depth 1 --branch ${LEROBOT_TAG} \
        https://github.com/huggingface/lerobot.git /opt/lerobot \
    && pip install --no-cache-dir -e "/opt/lerobot[training,smolvla,peft]" \
    && pip uninstall -y -q torchao || true

# 版の対が崩れていないことを、ビルド時に確かめる。実行時まで持ち越さない。
RUN python -c "\
import torch, torchcodec; \
from torchcodec.decoders import VideoDecoder; \
print(f'torch {torch.__version__} / torchcodec {torchcodec.__version__} 整合')"

WORKDIR /workspace
COPY train.py /workspace/train.py

# 既定の出力先。ホストのディレクトリをここへマウントする。
VOLUME ["/out"]
ENV OUTPUT_ROOT=/out

CMD ["python", "/workspace/train.py"]
