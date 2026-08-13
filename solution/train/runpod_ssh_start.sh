#!/usr/bin/env bash
# RunPod の Pod で、sshd を立ててからコンテナを寝かせ続ける起動用スクリプト。
# Dockerfile.pi05 で作ったカスタムイメージを RunPod で使うときに、
# Pod 作成画面の「Container Start Command」へこれを指定する:
#
#   bash /opt/pi05/runpod_ssh_start.sh
#
# **なぜ要るのか。** RunPod の「Basic SSH」（RunPod のサーバを経由する簡易接続）は
# scp / sftp に対応していない。11GB 級のチェックポイントを持ち帰るには
# 公開 IP + Pod 内の sshd による「Full SSH」が必要で、公式テンプレート
# （RunPod PyTorch 等）には最初から入っているが、**カスタムイメージには
# 自分で入れて起動しないと使えない**
# （https://docs.runpod.io/pods/configuration/use-ssh 2026-08-13 確認）。
#
# 公式が案内している起動コマンドは apt で openssh-server をその場で入れる形だが、
# ここでは **イメージのビルド時に入れてある**（Dockerfile.pi05）ので、
# 起動のたびに apt を叩かない。Pod の起動が速く、ネットワークにも依存しない。
#
# TCP 22 番を Pod のテンプレートで公開しておくこと（Dockerfile 側の EXPOSE 22
# だけでは RunPod のポート割り当てには足りない）。
set -euo pipefail

# sshd は特権分離用の空ディレクトリが無いと起動を拒む
# （"Missing privilege separation directory: /run/sshd"）。
mkdir -p /run/sshd

# ホスト鍵はイメージに焼かず、この Pod で作る（Dockerfile 側でビルド時の鍵を
# 消してある）。同じイメージを使う全 Pod が同一のホスト鍵を持つのを避けるため。
ssh-keygen -A

# **`~/.ssh` と書かない。** このイメージは非root実行のために `HOME=/tmp` を
# 設定してあるので、`~` は `/tmp` になる。一方 sshd は authorized_keys を
# `/etc/passwd` に書かれた home（root なら `/root`）から読むため、`~/.ssh` に
# 置くと鍵が読まれず公開鍵認証が通らない。パスを明示する。
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
else
    echo "[runpod_ssh_start] 警告: 環境変数 PUBLIC_KEY が空です。" >&2
    echo "[runpod_ssh_start]   RunPod の Settings → SSH Public Keys に公開鍵を" >&2
    echo "[runpod_ssh_start]   登録してから Pod を作り直すこと。今回は SSH で" >&2
    echo "[runpod_ssh_start]   入れないので、Web Terminal を使うことになる。" >&2
fi

# `-e` でログを標準エラーへ出す（Pod 管理画面の Logs から見える）。
# sshd は既定で自分をバックグラウンドへ落とすので、このまま次の行へ進む。
/usr/sbin/sshd -e

echo "[runpod_ssh_start] sshd を起動した。Pod の Connect タブに出る"
echo "[runpod_ssh_start] ssh root@<IP> -p <port> で入れる（scp も使える）。"

# コンテナを生かし続ける。学習は SSH か Web Terminal から tmux の中で始める。
exec sleep infinity
