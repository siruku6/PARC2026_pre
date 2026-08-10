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

# torch と torchcodec の wheel をどの index から取るか。ホストのドライバに合わせる。
#
#   cu128 … 既定。LeRobot v0.6.0 の uv.lock が torch 2.11.0+cu128 を解決する
#   cu130 … より新しいドライバ（採点イメージはこちら）
#   cpu   … **x86_64 のホストでしか使えない。** 下の注記を読むこと
#
# **`cpu` は arm64（Apple Silicon など）では通らない。**
# torchcodec 0.11.1 の aarch64 向け wheel が、index ごとに揃っていないためである。
#
#   | index | 0.11.1 x86_64 | 0.11.1 aarch64 |
#   |-------|---------------|----------------|
#   | cpu   | あり          | **無い（0.13.0 から）** |
#   | cu128 | あり          | あり           |
#   | cu130 | あり          | あり           |
#
# 0.13.0 へ上げる逃げ道は無い。LeRobot が `torchcodec<0.12.0` と宣言しているため、
# 後続の `pip install lerobot[training]` で 0.11.x へ引き戻され、同じ壁に当たる。
# **arm64 で配線だけ見たいときは `--build-arg TORCH_INDEX=.../cu128` を使う**
# （GPU が無くても import は通る）。
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# **torch と torchcodec は、版だけでなく CUDA の系統も対で決まる。**
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
ARG TORCHVISION_VERSION=0.26.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX}

# **torch と同じ index から取ること。`--index-url` を省くと PyPI から落ちてくる。**
# torchcodec の README いわく「Linux では CUDA 入りの wheel が既定で入り、GPU が
# 無くても動く」。ただしそれは `libnvrtc` があればの話で、**cpu 版の torch には
# CUDA ライブラリが 1 つも入っていない。** 省略して 2026-08-10 に実際に落ちた。
#
#   OSError: libnvrtc.so.13: cannot open shared object file
#
# PyPI の 0.11.1 が要求するのは `.so.13`（CUDA 13）なので、cu128 の torch
# （`.so.12` を持ってくる）と組んでも同じ失敗になる。**index を揃えるのが正しい。**
#
# lerobot より先に入れる。先に固定しておけば、後続の解決は制約を満たすので
# 触られない。あとから入れると <0.12.0 の最新が選ばれ、対が崩れる。
RUN pip install --no-cache-dir "torchcodec==${TORCHCODEC_VERSION}" --index-url ${TORCH_INDEX}

# torchcodec 0.11.x はコアの .so が CUDA の NPP（画像・色空間変換）ライブラリに
# 直接リンクされている。GPU デコードを使わない呼び出しでも import 時に読み込まれる
# ため、無いと "libnppicc.so.12: cannot open shared object file" で落ちる。
# https://github.com/meta-pytorch/torchcodec/issues/1323
#
# torch の cu128 wheel が連れてくる nvidia-*-cu12 群には NPP が含まれない
# （torch 自体は NPP を使わないため）。torchcodec もこれを依存として宣言しない
# （上の注記のとおり）ので、明示的に入れる。
# TORCH_INDEX（pytorch のミラー）には無いパッケージなので、素の PyPI から取る
# ＝ --index-url は付けない。
#
# **pip install するだけでは足りない。** libnvrtc など torch が使う CUDA ライブラリは
# torch 自身が import 時に site-packages/nvidia/*/lib を明示的に preload する
# （torch/cuda/__init__.py の固定リストに載っている）ため動的リンカが見つけられるが、
# NPP はそのリストに無い。site-packages 配下に .so を置くだけでは動的リンカの探索先に
# 入らないので、ldconfig に directory を教えて明示的に登録する。
# ここで `ldconfig -p | grep` に失敗させておけば、登録漏れをビルド時に検出できる。
RUN pip install --no-cache-dir "nvidia-npp-cu12" \
    && NPP_SO="$(find /usr/local/lib/python3.12/site-packages/nvidia \
            -name 'libnppicc.so*' -print -quit)" \
    && test -n "$NPP_SO" \
    && dirname "$NPP_SO" > /etc/ld.so.conf.d/nvidia-npp.conf \
    && ldconfig \
    && ldconfig -p | grep -q libnppicc.so.12

# **torchvision も torch と対で決まる**（LeRobot の transforms が
# `torchvision.transforms.v2` を使うため必須）。ここで固定せずに lerobot の
# editable install に任せると、torchvision は `--index-url` の付かない素の
# PyPI から解決される。PyPI の torchvision は torch 2.11.0+cu128 とは別の
# ビルドのため、ABI が合わず import 時点で落ちる。2026-08-10 に実際に落ちた。
#
#   RuntimeError: operator torchvision::nms does not exist
#
# 対応表（https://github.com/pytorch/vision#readme）で torch 2.11 は
# torchvision 0.26 と対になっている。torch/torchcodec と同じ理由で、
# 同じ index から先に入れておく。
RUN pip install --no-cache-dir "torchvision==${TORCHVISION_VERSION}" --index-url ${TORCH_INDEX}

# 学習用ノートブックと同じタグ・同じ extras。ここを変えると重みの互換性が崩れる。
# training は dataset を含むため、dataset を重ねて書く必要はない。
ARG LEROBOT_TAG=v0.6.0
RUN git clone --quiet --depth 1 --branch ${LEROBOT_TAG} \
        https://github.com/huggingface/lerobot.git /opt/lerobot \
    && pip install --no-cache-dir -e "/opt/lerobot[training,smolvla,peft]" \
    && pip uninstall -y -q torchao || true

# 版の対が崩れていないことを、ビルド時に確かめる。実行時まで持ち越さない。
#
# **このチェックは消してはいけない。** LeRobot の自動フォールバックは
# `importlib.util.find_spec("torchcodec")` で判定しており（import_utils.py）、
# **入ってさえいれば壊れていても「使える」と答える。** ここで止めないと、
# 学習データ 6.4 GB を落としきったあとにデコードで落ちる。
RUN python -c "\
import torch, torchcodec, torchvision; \
from torchcodec.decoders import VideoDecoder; \
torchvision.ops.nms; \
print(f'torch {torch.__version__} / torchcodec {torchcodec.__version__} '\
      f'/ torchvision {torchvision.__version__} 整合')"

WORKDIR /workspace
# **学習とマージの両方を入れる。** マージ（LoRA を初期重みへ畳み込む）は同じ
# LeRobot / peft / torch の組み合わせを要求するので、イメージを分けない。
# 片方だけ COPY すると、学習が終わったあとに
# "can't open file '/workspace/merge.py'" で止まる。
COPY train.py merge.py /workspace/

# 既定の出力先。ホストのディレクトリをここへマウントする。
VOLUME ["/out"]
ENV OUTPUT_ROOT=/out

CMD ["python", "/workspace/train.py"]
