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
#   cu124 … ドライバ 550 以降。互換性が広く既定にしている
#   cu128 … より新しいドライバ
#   cpu   … GPU 無しで配線だけ確認するとき
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# LeRobot v0.6.0 は torch>=2.7,<2.12 を要求する。
RUN pip install --no-cache-dir "torch==2.8.0" --index-url ${TORCH_INDEX}

# 学習用ノートブックと同じタグ。ここを変えると重みの互換性が崩れる。
ARG LEROBOT_TAG=v0.6.0
RUN git clone --quiet --depth 1 --branch ${LEROBOT_TAG} \
        https://github.com/huggingface/lerobot.git /opt/lerobot \
    && pip install --no-cache-dir -e "/opt/lerobot[training,smolvla,peft,dataset]" \
    && pip uninstall -y -q torchao || true

WORKDIR /workspace
COPY train.py /workspace/train.py

# 既定の出力先。ホストのディレクトリをここへマウントする。
VOLUME ["/out"]
ENV OUTPUT_ROOT=/out

CMD ["python", "/workspace/train.py"]
