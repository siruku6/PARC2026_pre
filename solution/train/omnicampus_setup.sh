#!/usr/bin/env bash
# Omnicampus 配布 GPU 環境（Blackwell / RTX PRO 6000・96 GB）に、**Docker を使わずに**
# π0.5 のフルパラメータ学習環境を作る。solution/train/Dockerfile.pi05 と同じ依存構成を、
# 素の venv に作り直したものである（runpod_setup.sh に続く 3 つ目の変種）。
#
# **なぜ Docker を使わないのか。** 本選資料 v1.1 の 14 ページに次の 2 行がある。
#   「配布しているOmnicampus提出の事前チェック用Docker以外のDockerの使⽤はサポートしていません
#     （…参加者が独⾃で使⽤するカスタムDockerを導⼊することを前提としたストレージ構造＆容量に
#     なっていないため、使⽤不可であることを想定しており…）」
#   「Omnicampus提出の事前チェック⽤DockerのDocker run のみをサポートしています」
# つまり配布イメージ（評価・提出前チェック用）は `docker run` してよいが、
# **学習用の parc2026-train-pi05 を持ち込む道は無い。**
#
# **なぜ python3.12 を自分で用意するのか。** 配布環境の python は 3.10.12
# （`/opt/jupyter/bin/python`）だが、**lerobot v0.6.0 の `requires-python` は `>=3.12`**
# である（コンテナ内の pyproject.toml で確認）。3.10 のままでは
# `pip install -e lerobot[training,pi]` が「Package requires a different Python」で止まる。
# 学習で使う lerobot の版は下げられない——`001500`（採点側で Track1 0.492 を出した重み）が
# v0.6.0 で作られており、梯子はその続きだからである。
#
# **なぜ `pip install --user` を使わないのか。** 配布環境の python は virtualenv であり、
# user site が無効になっている（8/31 の実測: `ERROR: Can not perform a '--user' install.
# User site-packages are not visible in this virtualenv.`）。自分の venv を作る。
#
# 使い方（Notebook のセルで。`!` も `%%bash` も付けず、bash セルとして流す）:
#
#   bash ~/pi05_code/solution/train/omnicampus_setup.sh
#   source ~/env_pi05.sh
#
# **2 回目以降のセッションでも、そのまま流してよい。**下の置き場のとおり venv は
# `$HOME` にあり、10 分ごとの自動保存で次のセッションに残っている。残っていれば
# 作り直さず、健全性の確認だけで終わる（1 分）。
#
# 置き場（本選資料 v1.1 p14 のストレージ構成に合わせた）:
#
#   $HOME            40 GiB・**10 分ごとと終了時に自動保存**   → venv・lerobot・python3.12
#   ~/data           100 GiB・保存は手動（parc-home-sync data-push）
#   ~/dataset        読み出し専用・配布データセット（tar）
#   /opt/dlami/nvme  約 1.7 TB・**セッション終了で消える**     → HF キャッシュ・学習の出力
#
# **$HOME に何を置くかは足し算で決まる。** venv 約 8.2 GB（学習イメージの site-packages の
# 実測 8.4 GB から、ここでは入れない bitsandbytes 177 MB と torchao 22 MB を引いた値）
# ＋ lerobot の clone 24 MB ＝ **約 8.5 GB**。40 GiB のうち残り 31 GiB を、
# 重み（1 個 9.4 GB）とチェックポイント（1 個 11 GB）にどう割るかは、このスクリプトでは決めない
# （docs/howto/14_blackwell_first_session.md の「2 回目以降のセッション」を見ること）。
set -euo pipefail

VENV="${VENV:-$HOME/venv-pi05}"
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/lerobot}"
LEROBOT_TAG="v0.6.0"
# Blackwell の compute capability は 12.0（8/31 の実測）。sm_120 のカーネルを含むのは
# cu128 以降の wheel だけで、PyPI の既定 wheel では `no kernel image is available` になる。
# **--index-url を 3 つとも付けること。**1 つでも外すと別の CUDA 版が来て対が壊れる。
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="2.11.0"
TORCHCODEC_VERSION="0.11.1"
TORCHVISION_VERSION="0.26.0"
# 消える領域。HF のキャッシュ（データ 15.83 GB）と学習の出力（1 個 11 GB）を置く。
SCRATCH="${SCRATCH:-/opt/dlami/nvme}"

t0=$(date +%s)
elapsed() { echo "$(( ($(date +%s) - t0) / 60 )) 分経過"; }

echo "=============================================================="
echo "[0/6] 下見（この 6 行は報告にそのまま貼れる）"
echo "=============================================================="
grep -m1 PRETTY_NAME /etc/os-release || true
echo "python: $(python3 -V 2>&1) / $(command -v python3)"
echo "python3.12: $(command -v python3.12 || echo 'なし')"
echo "uv: $(command -v uv || echo 'なし')"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
echo "CPU: $(nproc) コア / RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
df -h "$HOME" "$SCRATCH" 2>&1 | tail -n +2

echo "=============================================================="
echo "[1/6] python3.12 を用意する"
echo "=============================================================="
PY312=""
if command -v python3.12 >/dev/null 2>&1; then
    PY312="$(command -v python3.12)"
    echo "  既に入っている: $PY312"
else
    # uv は python-build-standalone のビルド済み CPython を取ってくる（root 不要）。
    # 置き場は ~/.local/share/uv/python で、**$HOME なので次のセッションにも残る。**
    if ! command -v uv >/dev/null 2>&1; then
        echo "  uv を入れる（~/.local/bin へ。root 不要）"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || { echo "  uv が入らなかった。分岐 5 を見ること" >&2; exit 1; }
    uv python install 3.12
    PY312="$(uv python find 3.12)"
    echo "  用意した: $PY312"
fi
"$PY312" -V

echo "=============================================================="
echo "[2/6] venv（$VENV）"
echo "=============================================================="
# --system-site-packages は付けない。配布環境の python は 3.10 で、
# そちらの site-packages（torch 2.11.0+cu128 の cp310 版）は 3.12 からは読めない。
if [ ! -x "$VENV/bin/python" ]; then
    "$PY312" -m venv "$VENV"
    echo "  作った"
else
    echo "  既にある（作り直さない）"
fi
# activate は未定義の変数（PS1 など）を触ることがあるので、-u を一時的に外す。
set +u
# shellcheck disable=SC1091
source "$VENV/bin/activate"
set -u
python -V
pip install --quiet --upgrade pip setuptools wheel

echo "=============================================================="
echo "[3/6] torch / torchcodec / torchvision（$TORCH_INDEX）"
echo "=============================================================="
# **順番を変えないこと。** lerobot を先に入れると、lerobot の解決器が自分の許す範囲で
# いちばん新しい torch を選び直し、対が壊れる（Dockerfile.pi05 と同じ理由）。
if python -c "import torch,torchcodec,torchvision" 2>/dev/null; then
    echo "  3 つとも入っている（入れ直さない）"
else
    t_dl=$(date +%s)
    pip install --no-cache-dir "torch==${TORCH_VERSION}"               --index-url "$TORCH_INDEX"
    pip install --no-cache-dir "torchcodec==${TORCHCODEC_VERSION}"     --index-url "$TORCH_INDEX"
    pip install --no-cache-dir "nvidia-npp-cu12"
    pip install --no-cache-dir "torchvision==${TORCHVISION_VERSION}"   --index-url "$TORCH_INDEX"
    dl_min=$(( ($(date +%s) - t_dl) / 60 ))
    dl_mb=$(du -sm "$VENV" | cut -f1)
    echo "  取得と展開に ${dl_min} 分（venv は ${dl_mb} MB）"
    echo "  **この数字を報告に残すこと。**2 回目以降のセッションで venv を作り直す"
    echo "  ことになった場合の、そのまま払う代金である。"
fi

# torchcodec は動画のデコードで NPP（CUDA の画像処理ライブラリ）を dlopen する。
# root なら ldconfig に登録し、そうでなければ LD_LIBRARY_PATH で通す。
NPP_DIR="$(python - <<'PY'
import pathlib, sysconfig
base = pathlib.Path(sysconfig.get_paths()["purelib"])
hits = sorted(base.glob("nvidia/**/libnppicc.so*"))
print(hits[0].parent if hits else "")
PY
)"
if [ -z "$NPP_DIR" ]; then
    echo "  **libnppicc.so が見つからない。**動画は PI05_VIDEO_BACKEND=pyav 相当に落ちる" >&2
elif [ "$(id -u)" -eq 0 ]; then
    echo "$NPP_DIR" > /etc/ld.so.conf.d/nvidia-npp.conf
    ldconfig
    ldconfig -p | grep -q libnppicc.so.12 && echo "  NPP を ldconfig に登録した"
else
    export LD_LIBRARY_PATH="$NPP_DIR:${LD_LIBRARY_PATH:-}"
    echo "  NPP を LD_LIBRARY_PATH で通した（root でないため）"
fi

echo "=============================================================="
echo "[4/6] lerobot $LEROBOT_TAG（extras: training, pi）"
echo "=============================================================="
if [ ! -d "$LEROBOT_DIR/.git" ]; then
    git clone --quiet --depth 1 --branch "$LEROBOT_TAG" \
        https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
# **入っているかを `import lerobot` で判定してはいけない**（8/31 に実機で踏んだ）。
# clone した `$HOME/lerobot` はソースを `src/lerobot` に持つので直下に `__init__.py` が無く、
# Notebook の作業ディレクトリが `$HOME` だと **PEP 420 の名前空間パッケージとして
# `import lerobot` が成功してしまう**（中身は空で `__version__` も無い）。
# その結果 pip install が丸ごと飛ばされ、最後の確認まで気づけなかった。
# **入っているかは pip に聞く。**
if pip show lerobot >/dev/null 2>&1; then
    echo "  既に入っている: lerobot $(pip show lerobot | awk '/^Version:/{print $2}')"
else
    pip install --no-cache-dir -e "${LEROBOT_DIR}[training,pi]"
fi

echo "=============================================================="
echo "[5/6] 学習スクリプトの置き場を確かめる"
echo "=============================================================="
# train_pi05.py は同じディレクトリの train.py と experiments/ を import する
# （`from train import ...` と `from experiments import geometry, quantization`）。
# **3 つが同じ並びで揃っていないと動かない。**
CODE_DIR="${CODE_DIR:-$HOME/pi05_code/solution/train}"
if [ -f "$CODE_DIR/train_pi05.py" ] && [ -d "$CODE_DIR/experiments" ]; then
    echo "  ある: $CODE_DIR"
else
    echo "  **無い: $CODE_DIR**"
    echo "  取り方は docs/howto/14_blackwell_first_session.md の A3 を見ること。"
fi

# **`~/.hf_token` の形をここで見る。**非公開リポジトリは未認証だと「無い」と言われるだけで、
# 認証の問題だと分からない。そして**置き換え用の文字列をそのまま貼ると**（8/31 に実際に起きた）
# `hf` が `'ascii' codec can't encode characters` で落ち、原因が token だと読めない。
if [ -f "$HOME/.hf_token" ]; then
    if LC_ALL=C grep -qE '^hf_[A-Za-z0-9]{20,}$' "$HOME/.hf_token"; then
        echo "  ~/.hf_token: 形は正しい（$(wc -c < "$HOME/.hf_token") バイト）"
    else
        echo "  **~/.hf_token が token の形をしていない。**置き換え用の文字列がそのまま"
        echo "  入っていないか確かめること（`hf_` で始まる英数字だけの 1 行である）。"
        echo "  非 ASCII が入っていると hf は \"'ascii' codec can't encode characters\" で落ちる。"
    fi
else
    echo "  **~/.hf_token が無い。**重みの置き場は 3 つとも非公開なので、"
    echo "  このままでは A4 の hf download が \"not found\" と言う（名前の間違いではない）。"
fi

echo "=============================================================="
echo "[6/6] 通ったことを、壊れていたら落ちる形で確かめる"
echo "=============================================================="
python - <<'PY'
import torch, torchvision
print(f"  torch {torch.__version__} / torchvision {torchvision.__version__}")
torchvision.ops.nms  # ABI が食い違っていればここで undefined symbol になる
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    import torchcodec
    print(f"  torchcodec {torchcodec.__version__}: ok")
except Exception as e:
    print(f"  **torchcodec が使えない（{type(e).__name__}: {e}）**")
assert torch.cuda.is_available(), "CUDA が見えていない"
print(f"  GPU {torch.cuda.get_device_name(0)} / capability {torch.cuda.get_device_capability(0)}")
# **import が通っても、sm_120 のカーネルが入っていなければここで初めて落ちる**
# （`no kernel image is available for execution on the device`）。
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print(f"  bf16 の行列積 ok（{float((a @ a).float().abs().mean()):.3f}）")
import importlib.metadata as md
import lerobot
# **`lerobot.__file__` を必ず見る。**空の名前空間パッケージ（上の [4/6] の注記）を
# 掴んでいると、ここが None になる。
assert lerobot.__file__, f"lerobot が空の名前空間パッケージである（{lerobot.__path__}）"
print(f"  lerobot {md.version('lerobot')}（{lerobot.__file__}）")
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # 学習経路が import できるか
print(f"  PI05Config: ok")
PY

cat > "$HOME/env_pi05.sh" <<EOF
# π0.5 の学習を回すシェルで毎回読み込む。**セルごとに読み直すこと。**
#
#   . ~/env_pi05.sh      # **ドットで読む。**Notebook のセルが /bin/sh（dash）で動くことが
#                        # あり、そこには source が無い（2026-08-31 に踏んだ
#                        # \`/bin/sh: 29: source: not found\`）。ドットは sh にも bash にもある。
#
# venv の activate も読まない。**やることは PATH を前に足すだけ**で、それは
# どのシェルでも同じに書ける。activate は bash 前提の書き方を含む。
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:\$HOME/.local/bin:\$PATH"
unset PYTHONHOME 2>/dev/null || true
export HF_HOME="$SCRATCH/hf_home"          # データと重みのキャッシュ。消える領域
export OUTPUT_ROOT="$SCRATCH/out"           # 学習の出力。消える領域
export LEROBOT_ROOT="$LEROBOT_DIR"
export CODE_DIR="$CODE_DIR"
# **HF のトークンは \$HF_HOME/token に入る。**その HF_HOME は消える領域なので、
# 何もしないと**セッションごとに hf auth login をやり直す**ことになる。
# \$HOME/.hf_token（自動保存に乗る）に書いておけば、HF_TOKEN が優先されるので省ける。
if [ -f "\$HOME/.hf_token" ]; then export HF_TOKEN="\$(cat "\$HOME/.hf_token")"; fi
${NPP_DIR:+export LD_LIBRARY_PATH="$NPP_DIR:\${LD_LIBRARY_PATH:-}"}
mkdir -p "\$HF_HOME" "\$OUTPUT_ROOT"
EOF
echo "  $HOME/env_pi05.sh を書いた（$HOME なので次のセッションにも残る）"

echo "=============================================================="
echo "完了。$(elapsed)"
echo "=============================================================="
cat <<'EOF'
次にやること（**source ではなくドットで読む。**セルが /bin/sh のとき source は無い）:

  . ~/env_pi05.sh
  hf auth whoami                # 名前が出れば通っている。出なければ ~/.hf_token を確かめる

そのあとは docs/howto/14_blackwell_first_session.md の A4（重みを取る）へ進む。
EOF
