# Blackwell サーバの 1 日目（2026-08-30）——学習を回せるかを確定させ、対照の 1 本を流す

**このファイルだけで 1 日が終わるように書いた。**別のファイルを開く必要はない。

対象は運営が配布する GPU 環境（**Blackwell サーバ**・NVIDIA RTX PRO 6000 Blackwell 1 台）。
**本選 1 の期間全体で 60 時間しか使えず、起動してから自分で止めるまでずっと減る。**
**この日に使うのは約 8 時間、多くても 12 時間である**（下の「時間の見積もり」）。

**この日の目的は 1 つ。**夜が終わるときに、次のどちらかが確定していること。

| 転び方 | その日のうちに決まること |
|---|---|
| **学習を回せる** | 対照の重みが 3 つ HF に載り、翌日から 1 日 1 つ提出できる |
| **回せない** | 理由が 1 行で書ける（外に出られない・VRAM が足りない・環境が作れない）。**RunPod に $50 積む判断をその日に出す** |

作成 2026-08-26、**2026-08-30 の計画として全面的に書き直した**（2026-08-29）。
出典は [本選の配布資料 v1.1](../materials/PARC2026開発コンペティション_本選_v1.1.pdf)。

## 項目一覧

| 節 | 一行要約 |
|---|---|
| [時間の見積もり](#時間の見積もり60-時間のうち-8-時間を使う) | 8 時間で済ませ、最悪でも 12 時間で自動的に止まる |
| [時間割](#時間割午後夕方夜) | 午後 150 分・夕方 90 分・夜は放置 |
| [確かめる順を決めた理由](#確かめる順は安いものからではないこれが駄目なら全部消えるものから) | 外に出られるかが最初 |
| [午後 A](#午後-a-機械が使えるかを確かめる150-分) | 入る → 外へ出られるか → 環境 → 重み → VRAM と速さ |
| [午後 D（並行）](#午後-d並行配布データセットの棚卸し40-分cpu-だけ) | pip が走っている裏で、tar を展開せずに数える |
| [夕方 B](#夕方-b-対照の学習を始める90-分) | 1,500 ステップ・カメラ拡張なしを nohup で始める |
| [夜 C](#夜-c-放置する) | 監視セルが待機による自動停止を防ぎ、終わったら止める |
| [翌朝](#翌朝-15-分) | 上がった重みを数え、提出の順番を決める |
| [想定と違ったとき](#想定と違ったときの分岐一覧) | VRAM・接続・セッションの寿命・HF の空きの 4 つ |
| [落とし穴](#落とし穴踏むと-1-日が消えるもの) | 過去に実際に踏んだもの |
| [確認したい仮定](#確認したい仮定) | 現物を見ていない前提 |

---

## 時間の見積もり（60 時間のうち 8 時間を使う）

| 区分 | 中身 | 見積もり |
|---|---|---|
| 午後 A | 下見・環境作り・重みの取得・VRAM と速さの実測 | **150 分** |
| 午後 D | 配布データセットの棚卸し（**A と並行。GPU を使わない**） | 40 分（A に隠れる） |
| 夕方 B | 学習の開始と、最初の保存が HF に上がるまでの見届け | **90 分** |
| 夜 C | 学習が回っているだけ（1,500 ステップ・実測 10.24 秒/step なら 4 時間 16 分） | **255 分** |
| 止まるまでの余り | 監視セルが終わってから、待機による自動停止まで | 最大 60 分 |
| **合計** | | **約 8 時間 15 分。残り約 52 時間** |

**なぜ 1,500 ステップで止めるのか。** 比べる相手（`pi05_trial1500`）が
**1,500 ステップ・500 ごとに 3 つ保存**だからである。**同じステップ数で揃えないと、
拡張の有無を比べたことにならない。**それより長く回しても、
**判定できる物差しが 1 日 1 トラック 1 回の提出しか無い**ので測れない
（[../knowledge/18_local_eval_missed_the_gain.md](../knowledge/18_local_eval_missed_the_gain.md)）。
**測れないステップに GPU 時間を払うより、60 時間を残すほうが価値が高い。**

**時間の減り方。**

| 止まり方 | 条件 |
|---|---|
| **自分で止める** | `File > Hub Control Panel > Stop Server`。**押した瞬間に消費が止まる** |
| 待機による自動停止 | セルの実行が何も無い状態が **1 時間**続いたとき |
| 起動時間による自動停止 | 起動から **12 時間**。**本選 1 では延長ボタンが使えない** |

**ブラウザのタブを閉じても止まらない。**画面の緑枠が今回の 12 時間に対する残り、
青枠が 60 時間に対する残りで、**青枠は起動時点の値のまま更新されない。**

---

## 時間割（午後・夕方・夜）

**`T0` は Notebook の画面が出た時刻である。**下の例は `T0 = 13:00` として書いた。
**T0 を紙に書き留めること。**すべての締め切りがここから決まる。

| 時刻（例） | T0 から | やること | 在席 |
|---|---|---|---|
| 13:00 | +0 分 | [A0](#a0-入る5-分) 入る。T0 を記録 | 要 |
| 13:05 | +5 分 | [A1](#a1-外へ出られるかを最初に確かめる10-分) **外へ出られるか** | 要 |
| 13:15 | +15 分 | [A2](#a2-機械を数える10-分) 機械を数える（VRAM・CPU・ディスク・python） | 要 |
| 13:25 | +25 分 | [A3](#a3-学習の環境を作る35-分) 環境を作る（torch 一式 ＋ lerobot）／裏で [D](#午後-d並行配布データセットの棚卸し40-分cpu-だけ) を流す | 要 |
| 14:00 | +60 分 | [A4](#a4-出発点の重みを取り往復の速さを測る15-分) 重みを取る・往復の速さを測る | 要 |
| 14:15 | +75 分 | [A5](#a5-batch-64-が載るかと1-ステップの時間を60-ステップで測る35-分) **VRAM と速さを 60 ステップで測る** | 要 |
| 14:50 | +110 分 | [A6](#a6-自分で止められるかを確かめる5-分) 自分で Stop できるか（API）を確かめる | 要 |
| 14:55 | +115 分 | [A7](#a7-ステップ数を決める10-分) ステップ数を決める | 要 |
| 15:05 | +125 分 | [B1](#b1-上げ先を用意する10-分) HF の空きを見て、上げ先を作る | 要 |
| 15:15 | +135 分 | [B2](#b2-学習を-nohup-で始める10-分) **学習を nohup で始める** | 要 |
| 15:25 | +145 分 | [B3](#b3-開始-2-分後に-1-行だけ確かめる2-分) `num_learnable_params` を確かめる | 要 |
| 15:30 | +150 分 | [B4](#b4-チェックポイントを-hf-へ上げるループを始める10-分) 上げるループを始める | 要 |
| 16:10 | +190 分 | [B5](#b5-最初の-500-ステップが-hf-に載るまで見届ける40-分) **最初の 500 ステップが HF に載るのを見届ける** | 要 |
| 16:20 | +200 分 | [B6](#b6-監視セルを実行状態のまま残して離席する5-分) 監視セルを残して離席 | 最後 |
| 〜19:50 | +410 分 | [C](#夜-c-放置する) 学習が回る（放置） | 不要 |
| 〜20:30 | +450 分 | 最後の上げが終わり、監視セルが自分で Stop を押す | 不要 |

**夜に置いたものは、切れても壊れない。**学習は `nohup` で走り、
**500 ステップごとの保存が出るたびに HF へ上がる**ので、
セッションが落ちても**上がったところまでは残る。**

---

## 確かめる順は「安いものから」ではない。「これが駄目なら全部消えるもの」から

**外へ出られなければ、重みも学習データも運べず、学習の話が全部消える。**だから
**A1 を最初に置いた。**次が環境（A3）、次が VRAM（A5）である。
**配布データセットの棚卸し（D）は GPU も外への接続も要らない**ので、
**pip が走っている裏で流す。**A が全部倒れても D の答えだけは持ち帰れる。

---

## 午後 A 機械が使えるかを確かめる（150 分）

**実行場所は Notebook のセルである。**`%%bash` のセルにするか、各行の先頭に `!` を付ける。
**出力は必ず `~/` に落とす。**`$HOME` は 10 分ごとと終了時に自動保存され、次回起動時に
復元されるので、貼り忘れても残る。

### A0 入る（5 分）

1. `https://parc-playground.omnicamp.us/hub/login` を開く
2. `Sign in with Auth0` → Omnicampus のアカウントでログイン
3. GPU の割り当てを待つ（**リージョンは複数あり `us-east-1` とは限らない**）
4. Notebook の画面が出たら、**その時刻を `T0` として書き留める**

**この時点から 60 時間が減り始める。**

### A1 外へ出られるかを最初に確かめる（10 分）

**これが分かると何が決まるか: 学習の話が続くか、その場で終わるか。**

```bash
{
echo "### 時刻(T0)"; date -u; date
echo "### PyPI"; timeout 30 pip download --no-deps -d /tmp/pipprobe six 2>&1 | tail -3
echo "### HF の API"; timeout 30 python3 -c "
from huggingface_hub import HfApi
print(HfApi().model_info('lerobot/pi05_libero_base').sha[:8])" 2>&1 | tail -3
echo "### 素の HTTPS"; timeout 30 curl -sS -o /dev/null -w '%{http_code} %{speed_download} B/s\n' \
  https://huggingface.co/api/models/lerobot/pi05_libero_base
echo "### git"; timeout 30 git ls-remote https://github.com/huggingface/lerobot.git v0.6.0 2>&1 | tail -2
} 2>&1 | tee ~/recon_net.txt
```

**分岐。**

| 結果 | どうするか |
|---|---|
| 4 つとも通る | そのまま A2 へ |
| **PyPI が通らない** | 環境を作れない。**この日の Blackwell は D（データの棚卸し）だけに使い、終わったら止める。**RunPod に $50 積む |
| **HF だけ通らない** | 重み 9.37 GB を `$HOME`（40 GiB）へ手で運ぶ道が残る。ただしこのマシンからの `scp` は 1.7 MB/s で 1 時間半かかる。**その日は諦めて D だけにする** |
| 401 や 403 が返る | 認証の問題である。A3 の後の `hf auth login` で直る見込み。**先に進んでよい** |

### A2 機械を数える（10 分）

**これが分かると何が決まるか: batch 64 が載るか（VRAM）、`--num-workers` をいくつにするか
（CPU）、チェックポイントをどこに置くか（ディスク）、pip の入れ方（python）。**

```bash
{
echo "### GPU"; nvidia-smi
echo "### GPU(要約)"; nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap,driver_version --format=csv
echo "### 他に誰か使っているか"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "### CPU/RAM"; nproc; free -g | head -2
echo "### disk"; df -h $HOME ~/data /opt/dlami/nvme /tmp 2>&1
echo "### python"; python -V; which python pip
echo "### pip の入り先"; python -m site --user-base; python -c "import site;print('user site 有効:',site.ENABLE_USER_SITE)"
echo "### 主要ライブラリ"; pip list 2>/dev/null | grep -iE "^(torch|torchvision|torchaudio|torchcodec|transformers|lerobot|accelerate|safetensors|numpy|huggingface) "
echo "### ffmpeg"; ffmpeg -version 2>&1 | head -1
echo "### root になれるか"; id; sudo -n true 2>&1 | head -1
echo "### 永続化の道具"; which parc-home-sync tmux nohup hf huggingface-cli
} 2>&1 | tee ~/recon.txt
```

**読み方。**

| 見るもの | 期待値 | 外れたときの意味 |
|---|---|---|
| `memory.total` | **97,887 MiB 前後（96 GB）** | 下の[分岐 1](#分岐-1-vram-が-96-gb-でなかった) |
| `compute_cap` | 12.0（Blackwell） | cu128 以上の wheel が要る |
| `nproc` | 不明 | `--num-workers` は `min(nproc/2, 16)` にする |
| `/opt/dlami/nvme` の空き | 約 1.7 TB | **無ければチェックポイントの置き場が `$HOME` の 40 GiB しか無い**（[分岐 3](#分岐-3-optdlaminvme-が-無い)） |
| `user site 有効` | True | False なら venv を作る（A3 の分岐） |

**置き場はこう決める。**

| 何を | どこへ | なぜ |
|---|---|---|
| pip で入れるもの | `~/.local`（`$HOME` の下） | **自動保存されるので、次のセッションで作り直さなくて済む** |
| 学習データと HF のキャッシュ | `/opt/dlami/nvme/hf_home` | 15.83 GB ある。`$HOME` の 40 GiB を食い潰さない |
| 学習の出力（1 個 25 GB） | `/opt/dlami/nvme/out` | 同上。**セッション終了で消えるので、HF へ上げるまでが勝負** |
| 数えた結果のテキスト | `~/`（`$HOME`） | 自動保存で残る |

### A3 学習の環境を作る（35 分）

**これが分かると何が決まるか: Notebook の python で π0.5 の学習が動くか。
ここが通らなければ、この日の学習は無い。**

**A2 の `pip list` を見てから流す。**`torch` が既に入っていて版が 2.8 以上なら、
**torch を入れ直さない**（数 GB の取得が無駄になる）。入れ直す必要があるのは
`torchcodec` か `torchvision` が無いか、版が対で合っていないときだけである。

```bash
# 0. 置き場を決める（このセッションの間ずっと使う。毎セルの先頭で export し直すこと）
export HF_HOME=/opt/dlami/nvme/hf_home
export OUTPUT_ROOT=/opt/dlami/nvme/out
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT"

# 1. いま入っている torch の版と CUDA の版を見る
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" 2>&1

# 2. 入っていない／対が合わないときだけ、同じ index から 3 つ入れる
#    **--index-url を 3 行とも付けること。**1 行でも外すと PyPI の別 CUDA 版が来て、
#    最初の動画デコードで libnvrtc.so が無いと言って落ちる。
TORCH_INDEX=https://download.pytorch.org/whl/cu128   # driver が CUDA 13 なら cu130 にする
pip install --user --no-cache-dir "torch==2.11.0"       --index-url $TORCH_INDEX
pip install --user --no-cache-dir "torchcodec==0.11.1"  --index-url $TORCH_INDEX
pip install --user --no-cache-dir "torchvision==0.26.0" --index-url $TORCH_INDEX

# 3. torchcodec が動画を読むのに要る CUDA のライブラリ（root が無くても通る形）
pip install --user --no-cache-dir nvidia-npp-cu12
export LD_LIBRARY_PATH="$(python -c "
import pathlib, sysconfig
base = pathlib.Path(sysconfig.get_paths()['purelib'])
print(next(iter(base.glob('nvidia/**/libnppicc.so*'))).parent)"):${LD_LIBRARY_PATH:-}"

# 4. lerobot を $HOME に置く（自動保存されるので次回も残る）
[ -d ~/lerobot ] || git clone --quiet --depth 1 --branch v0.6.0 \
    https://github.com/huggingface/lerobot.git ~/lerobot
pip install --user --no-cache-dir -e ~/lerobot[training,pi]
pip uninstall -y -q torchao || true

# 5. 通ったことを、対が壊れていたら落ちる形で確かめる
python -c "
import torch, torchcodec, torchvision
from torchcodec.decoders import VideoDecoder
torchvision.ops.nms
import lerobot
print(f'torch {torch.__version__} / torchcodec {torchcodec.__version__} / torchvision {torchvision.__version__} 整合')
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**`pip install` の順番を変えないこと。**lerobot を先に入れると、lerobot の解決器が
自分の許す範囲でいちばん新しい torch を選び直し、**対が壊れる。**

**分岐。**

| つまずき | どうするか |
|---|---|
| `user site 有効: False`（A2） | `python -m venv --system-site-packages ~/venv && source ~/venv/bin/activate` を先に流し、以降 `--user` を外す。**`~/venv` も `$HOME` なので残る** |
| `undefined symbol: _ZN...` | 版の対が合っていない。3 つとも同じ `--index-url` で入れ直す |
| `libnvrtc.so.NN: cannot open` | CUDA の版が食い違っている。`TORCH_INDEX` を `cu130` にして入れ直す |
| `ffmpeg` が無く、root にもなれない | `--dataset.video_backend=torchcodec` は ffmpeg を要る。**`pip install --user imageio[ffmpeg]` で入る ffmpeg を `PATH` に足す** |
| **どうやっても 40 分で通らない** | 打ち切る。**この日は D の結果だけ持ち帰り、RunPod に積む。**深追いすると 60 時間が溶ける |

**このあと `hf auth login` を 1 回だけ流す**（write 権限のトークンを手元に用意しておく）。
**忘れると、非公開リポジトリが「無い」と言われる。**名前の間違いではない。

### A4 出発点の重みを取り、往復の速さを測る（15 分）

**これが分かると何が決まるか: 9.4 GB を上げるのに何分かかるか。
それが分かると、A7 で「上げ切る時間」を差し引いてステップ数を決められる。**

**取るのは `siruku6/pi05_full_runpod`（＝ `rp3000`・9.37 GB）である。**
これが `pi05_trial1500` の出発点なので、**対照も同じところから始める。**

```bash
export HF_HOME=/opt/dlami/nvme/hf_home
mkdir -p /opt/dlami/nvme/weights
time hf download siruku6/pi05_full_runpod --local-dir /opt/dlami/nvme/weights/rp3000
ls -la /opt/dlami/nvme/weights/rp3000
du -sh /opt/dlami/nvme/weights/rp3000
# `config.json` と `model.safetensors` が直下にあること。これが --init-from に渡す形である。
```

**降りてきた秒数から MB/s を出す**（9,370 MB ÷ 秒数）。
**参考値: RunPod では 9.4 GB の上げが 16 分だった**（`docs/howto/06_runpod_pi05.md`）。
上げは降りより遅いことが多いので、**上げは「降りの 2 倍の時間」と見積もる。**

### A5 batch 64 が載るかと、1 ステップの時間を、60 ステップで測る（35 分）

**これが分かると何が決まるか: batch を 64 のままにできるか（VRAM）と、
1,500 ステップが今日のうちに終わるか（速さ）。チェックポイントが書けることも同時に分かる。**

**`train_pi05.py` をこの機械へ運ぶ。**`experiments/` も一緒に運ぶこと
（`train_pi05.py` が `from experiments import geometry, quantization` を実行する）。

```bash
mkdir -p ~/src && cd ~/src
git clone --quiet https://github.com/siruku6/PARC2026_pre.git   # 非公開。ユーザ名と PAT を訊かれる
ls ~/src/PARC2026_pre/solution/train/train_pi05.py
```

**GitHub に届かないときは HF を経由する**（このマシンで tar.gz を作って私有リポジトリへ
上げ、Blackwell で落とす。**116 KB なので 10 秒で終わる**）。

```bash
export HF_HOME=/opt/dlami/nvme/hf_home
export OUTPUT_ROOT=/opt/dlami/nvme/out
export PATH="$HOME/.local/bin:$PATH"
cd ~/src/PARC2026_pre/solution/train

NW=$(python -c "import os;print(min(os.cpu_count()//2, 16))")
time python train_pi05.py --run-name probe60 \
    --init-from /opt/dlami/nvme/weights/rp3000 \
    --steps 60 --batch-size 64 --num-workers "$NW" \
    --learning-rate 5e-6 --save-freq 60 \
    --freeze-vision-encoder false --train-expert-only false \
    2>&1 | tee ~/probe60.log
```

**読むのは 3 つ。**

```bash
grep -m1 num_learnable_params ~/probe60.log     # 4143404816 であること
grep -oE "step:[0-9]+ .*" ~/probe60.log | tail -2
# **1 ステップの秒数は `updt_s` ＋ `data_s` である。**lerobot 自身が同じ足し方で
# 残り時間を出している（`src/lerobot/scripts/lerobot_train.py` 603 行）。
python3 - <<'PY'
import re, pathlib
lines = [l for l in pathlib.Path("~/probe60.log").expanduser().read_text().splitlines()
         if "updt_s" in l]
last = lines[-1]
get = lambda k: float(re.search(rf"{k}:([0-9.]+)", last).group(1))
print(last)
print(f"1 ステップ = {get('updt_s') + get('data_s'):.2f} 秒 / mem_gb = {get('mem_gb'):.2f}")
PY
du -sh $OUTPUT_ROOT/probe60/checkpoints/000060 $OUTPUT_ROOT/probe60/checkpoints/000060/pretrained_model
```

| 見るもの | 期待値（出典） | 外れたら |
|---|---|---|
| `num_learnable_params` | **4143404816** | 693422112 なら行動側しか学習していない。**その場で止める** |
| `mem_gb` | **49.37 前後**（A100 80GB での実測・`19_weights_catalog.md`） | [分岐 1](#分岐-1-vram-が-96-gb-でなかった) |
| 1 ステップ | **10.05〜10.24 秒**（A100 80GB・batch 64 の実測） | A7 の式に実測値を入れる |
| `pretrained_model` | **8.8 GB**、チェックポイント全体で 25 GB | 大きく違えば保存の設定が違う |

**測り終えたら `probe60` を消す**（25 GB がそのまま残る）。

```bash
rm -rf /opt/dlami/nvme/out/probe60
```

### A6 自分で止められるかを確かめる（5 分）

**これが分かると何が決まるか: 夜中に学習が終わったあと、誰も押さないまま
1 時間（待機による自動停止）または最大 12 時間ぶん消費し続けるかどうか。
自分で止められるなら、その差ぶんの GPU 時間が 60 時間に戻る。**

```bash
python3 - <<'PY'
import os, urllib.request
url = os.environ.get("JUPYTERHUB_API_URL")
tok = os.environ.get("JUPYTERHUB_API_TOKEN")
user = os.environ.get("JUPYTERHUB_USER")
print("API_URL =", url, "/ USER =", user, "/ TOKEN =", "あり" if tok else "無し")
if url and tok and user:
    req = urllib.request.Request(f"{url}/users/{user}",
                                 headers={"Authorization": f"token {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            print("GET /users/<user> ->", r.status)
    except Exception as e:
        print("読めない:", e)
PY
```

**ここでは読むだけにする。**止める側（`DELETE /users/<user>/server`）を試すと
その場でセッションが終わる。**押すのは夜の監視セルの最後だけである。**

| 結果 | 夜の終わり方 |
|---|---|
| `GET` が 200 | 監視セルの最後で `DELETE` を試みる。**通れば即座に消費が止まる** |
| トークンが無い・403 | **待機による自動停止（1 時間）に任せる。**それでも 12 時間の上限より早い |

### A7 ステップ数を決める（10 分）

**これが分かると何が決まるか: 今夜の学習を 1,500・1,000・500 のどれで回すか。**

**1,500 が第 1 候補である。**比べる相手 `pi05_trial1500` が
**1,500 ステップ・500 ごとに 3 つ保存**なので、**同じ刻みで揃えると 3 対の比較になり、
提出枠を 3 日ぶん使える。**

```bash
python3 - <<'PY'
import datetime, math
# ---- 実測値を 3 つ入れる ----
sec_per_step = 10.24          # A5 の実測
upload_mb_s  = 10.0           # A4 の実測（降り）の半分を上げの速さとみなす
t0           = "13:00"        # A0 で書き留めた時刻
# ---------------------------------
now = datetime.datetime.now()
start = now.replace(hour=int(t0[:2]), minute=int(t0[3:]), second=0, microsecond=0)
deadline = start + datetime.timedelta(hours=11)      # 12 時間の自動停止に 1 時間の余裕
upload_sec = 3 * 9400 / upload_mb_s                  # 9.4 GB を 3 回上げる
budget = (deadline - now).total_seconds() - upload_sec
for cand in (1500, 1000, 500):
    if cand * sec_per_step <= budget:
        print(f"ステップ数 = {cand}（学習 {cand*sec_per_step/3600:.1f} 時間 ＋ "
              f"上げ {upload_sec/3600:.1f} 時間、締切まで {budget/3600:.1f} 時間）")
        break
else:
    print("500 も入らない。今日は学習を回さず、環境を残して止める")
PY
```

**500 も入らないときは回さない。**環境（`~/.local` と `~/lerobot`）は `$HOME` に残るので、
**翌日は A3 を飛ばして 30 分早く始められる。**

---

## 午後 D（並行）配布データセットの棚卸し（40 分・CPU だけ）

**A3 の pip が走っている間に、別のセルで流す。**GPU も外への接続も要らない。
**A が全部倒れてもこの答えは持ち帰れる。**

**なぜ今これを数えるのか。** 8/29 にこのマシンで数えたところ、
**いま学習に使っている `Sylvest/libero_plus` の 40 タスクは
標準スイート 4 つ（spatial・object・goal・libero_10）ちょうどで、
`libero_90` の 90 タスクを 1 つも学習していない**ことが分かった
（[../reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md)）。
**そして Track3 の元タスクは SCENE 名のものに限られ、その 9 割が `libero_90` にある**
（運営の `check_no_leaks.py` の禁止語 `(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+` に
「元タスク（順方向 BDDL）の識別子」という説明が付いている）。
**つまり `libero_90` が配布物の中にあるかどうかで、Track3 に手が届くかが決まる。**

### D1 何が配られているかを見る（5 分）

**これが分かると何が決まるか: 下の D2〜D4 で開く tar の名前。**

```bash
{
echo "### 並び"; find ~/dataset -maxdepth 3 | head -80
echo "### 容量"; du -sh ~/dataset/* 2>/dev/null
echo "### tar の一覧"; find ~/dataset -name '*.tar*' -printf '%12s  %p\n' | sort -k2
echo "### tar 以外"; find ~/dataset -maxdepth 2 -name '*.zip' -o -maxdepth 2 -name 'meta' -type d
} 2>&1 | tee ~/recon_dataset.txt
```

**`~/dataset` は読み出し専用である。**展開するなら `~/data`（100 GiB）か
`/opt/dlami/nvme` へ出す。**この節では 1 つも展開しない。**

### D2 展開せずに、中の `meta/info.json` だけを読む（10 分）

**これが分かると何が決まるか: 4 つ。**
**(a) `libero_90` が配布されているか**（Track3 の学習データを作れるか）、
**(b) `libero_combined_20hz` は何の統合か**（自分で混ぜずに済むか）、
**(c) 配布の LIBERO-plus が手元と同じものか**（15.83 GB の取得を省けるか）、
**(d) LeRobot の形式が v3.0 か**（`--dataset-root` でそのまま読めるか）。

```bash
for t in $(find ~/dataset -name '*.tar*' | sort); do
  echo "=== $t"
  p=$(tar tf "$t" 2>/dev/null | grep -m1 'meta/info\.json') || { echo "  meta/info.json 無し"; continue; }
  tar xf "$t" -O "$p" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
keys = ('codebase_version','fps','total_episodes','total_frames','total_tasks','total_videos')
print('  ' + ', '.join(f'{k}={d.get(k)}' for k in keys))
"
done 2>&1 | tee ~/recon_dataset_info.txt
```

**読み方（この表と突き合わせる）。**数値の出典は公開版の `meta/info.json`
（[../reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md)）。

| 見えた `total_tasks` / `total_episodes` | 中身の読み | 決まること |
|---|---|---|
| 40 / 14,347 | `libero_plus` そのもの | 手元と同じ。**HF から 15.83 GB を落とさずに済む** |
| **73 / 3,921** | **`libero_90_no_noops`** | **Track3 と Track2 に向けてタスクを 73 種類足せる** |
| 10 / 379 | `libero_10_no_noops` | 既に `libero_plus` に入っている。足しても種類は増えない |
| 40 / 16,040 | `libero_plus` ＋ 標準スイート 4 つ | **種類は増えない。**エピソード数だけ 1,693 本増える |
| **113 / 19,961** | **`libero_plus` ＋ 標準 4 つ ＋ `libero_90`** | **これ 1 つを `--dataset-root` に渡せば、混ぜる作業が要らない** |

| `codebase_version` | 意味 |
|---|---|
| `v3.0` | `train_pi05.py` がそのまま読める（`meta/episodes/*.parquet` を見に行く） |
| `v2.1` | **そのままでは読めない。**`~/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py` で変換が要る。**15 GB の変換に時間がかかるので、今日は使わない** |

### D3 指示文の一覧を書き出す（10 分）

**これが分かると何が決まるか: `libero_90` を足すと指示文が何種類増えるか。
LeRobot 形式ではタスク＝指示文なので、名前の集合を突き合わせれば重複が出る。**

```bash
for t in $(find ~/dataset -name '*.tar*' | sort); do
  n=$(basename "$t" | sed 's/\.tar.*//')
  p=$(tar tf "$t" 2>/dev/null | grep -m1 -E 'meta/tasks\.(jsonl|parquet)') || continue
  case "$p" in
    *.jsonl) tar xf "$t" -O "$p" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line:
        print(json.loads(line)['task'])" ;;
    *.parquet) tar xf "$t" -O "$p" > /tmp/t.parquet && python3 -c "
import pandas as pd
df = pd.read_parquet('/tmp/t.parquet')
# **v3.0 の meta/tasks.parquet は指示文を列ではなく index に持つ**
# （lerobot 0.6.0 の datasets/io_utils.py 184-187 行 load_tasks が
#  読んだあとで index.name を 'task' に付け直している）。
s = df['task'] if 'task' in df.columns else df.index.to_series()
print('\n'.join(s.astype(str)))" ;;
  esac | sort -u > ~/tasks_"$n".txt
  echo "$n: $(wc -l < ~/tasks_"$n".txt) 種類"
done 2>&1 | tee ~/recon_tasks.txt
```

### D4 集合を突き合わせる（5 分）

**これが分かると何が決まるか: `libero_plus` に足して意味があるのはどれか。**

```bash
python3 - <<'PY'
import pathlib
files = sorted(pathlib.Path.home().glob("tasks_*.txt"))
sets = {f.stem.replace("tasks_", ""): set(f.read_text().splitlines()) for f in files}
names = list(sets)
print(f"{'':28s}" + "".join(f"{n[:12]:>14s}" for n in names))
for a in names:
    row = "".join(f"{len(sets[a] & sets[b]):>14d}" for b in names)
    print(f"{a[:28]:28s}{row}")
print()
for a in names:
    for b in names:
        if a != b:
            only = len(sets[b] - sets[a])
            if only:
                print(f"{b} にしかない指示文: {only} 種類（{a} と比べて）")
PY
```

**期待する形。**`libero_plus`（40）と `libero_90`（73）の重なりは **0** のはずである
（8/29 にこのマシンで数えた）。**0 でなければ、この読みが間違っている。**

### D5 配布ベースラインの推論設定を読む（5 分）

**これが分かると何が決まるか: 運営が既定にしている chunk 長・ノイズ除去の回数・
後処理の有無。これはリーダーボードで Track1 0.286 を出した設定そのものである。**

```bash
{
unzip -l ~/dataset/pi05_step005000_submission_py310.zip | head -30
unzip -p ~/dataset/pi05_step005000_submission_py310.zip requirements.txt
unzip -p ~/dataset/pi05_step005000_submission_py310.zip policy_server.py | head -120
} 2>&1 | tee ~/recon_baseline.txt
```

**zip の中の名前が違えば `unzip -l` の一覧を見て直す。**

### D6 数えた結果を `$HOME` に残す（5 分）

```bash
ls -la ~/recon*.txt ~/tasks_*.txt
# 貼って持ち帰るのは recon_dataset_info.txt・recon_tasks.txt・D4 の出力の 3 つでよい
```

**`~/data` を使ったなら `parc-home-sync data-push` を流すこと。**
`$HOME` は自動保存されるが、**`~/data` は手で保存しないと消える。**

---

## 夕方 B 対照の学習を始める（90 分）

**何を回すか。`pi05_trial1500` と 1 つだけ違う走りである。**

| | `pi05_trial1500`（8/28・RunPod） | **今夜（Blackwell）** |
|---|---|---|
| 出発点 | `rp3000` | 同じ |
| ステップ数 | 1,500 | 同じ |
| batch | 64 | 同じ |
| 学習率 | 5e-6 | 同じ |
| 種 | 42 | 同じ |
| 画像側 | 開けた | 同じ |
| **カメラ拡張** | **あり** | **なし** |

**変えるのはここだけである。**8/29 に採点側で上がった（Track1 0.370 → 0.492 /
Track2 0.154 → 0.233）のは「画像側を開けたフルパラメータ学習」と「カメラ拡張」の
**どちらなのかが切り分けられていない。**この 1 本でそれが決まる。

**loss では比べられない。**カメラ拡張が入ると画像がゆがむので loss の水準そのものが
上がる（`19_weights_catalog.md`）。**比べられるのは採点側の点だけである。**

### B1 上げ先を用意する（10 分）

**これが分かると何が決まるか: 3 つのチェックポイントを HF に置けるか。
置けないなら `~/data` へ回す。**

```bash
export PATH="$HOME/.local/bin:$PATH"
python3 - <<'PY'
from huggingface_hub import HfApi, create_repo
api = HfApi()
total = 0
for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets)):
    for r in lister(author="siruku6"):
        info = api.repo_info(r.id, repo_type=kind, files_metadata=True)
        size = sum((f.lfs.size if f.lfs else (f.size or 0)) for f in info.siblings)
        total += size
        print(f"{kind:8s} {r.id:42s} {size/1e9:7.2f} GB private={info.private}")
print(f"合計 {total/1e9:.2f} GB / 非公開の枠 100 GB")
create_repo("siruku6/pi05_ctrl1500", private=True, exist_ok=True)
print("上げ先を作った: siruku6/pi05_ctrl1500")
PY
```

**8/29 にこのマシンから数えた時点で合計 67.03 GB、空きは約 33 GB だった。**
**要るのは `pretrained_model` 3 つ ＝ 26.4 GB である。**

| 空き | どうするか |
|---|---|
| 27 GB 以上 | そのまま進む |
| **27 GB 未満** | [分岐 4](#分岐-4-hf-の空きが-27-gb-に足りない) |

### B2 学習を nohup で始める（10 分）

```bash
export HF_HOME=/opt/dlami/nvme/hf_home
export OUTPUT_ROOT=/opt/dlami/nvme/out
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$(python -c "
import pathlib, sysconfig
base = pathlib.Path(sysconfig.get_paths()['purelib'])
print(next(iter(base.glob('nvidia/**/libnppicc.so*'))).parent)"):${LD_LIBRARY_PATH:-}"
cd ~/src/PARC2026_pre/solution/train

NW=$(python -c "import os;print(min(os.cpu_count()//2, 16))")
nohup python train_pi05.py --run-name pi05_ctrl1500 \
    --init-from /opt/dlami/nvme/weights/rp3000 \
    --steps 1500 --batch-size 64 --num-workers "$NW" \
    --learning-rate 5e-6 --save-freq 500 \
    --freeze-vision-encoder false --train-expert-only false \
    > $OUTPUT_ROOT/pi05_ctrl1500.nohup 2>&1 &
echo "PID $!"
```

**`--camera-viewpoint-augment` を付けないこと。**これが今回の変数である。

**`--learning-rate 5e-6` を省かないこと。**省くと頂点 2.5e-5 に戻り、
`rp3000` が 2.5e-6 で終わっているぶんの **10 倍**から再スタートする。
**`--init-from` は `pretrained_model` しか読まないので、学習率の刻みは引き継がれない**
（`load_training_state` が読む `training_state/` は `--resume` のときだけ使われる。
`lerobot/scripts/lerobot_train.py` の 385〜389 行と
`lerobot/common/train_utils.py` の 201 行）。

**`--steps` は A7 で決めた数に置き換える。**1,000 なら `--save-freq 500`（2 つ）、
500 なら `--save-freq 250`（2 つ）にして、**途中の重みが 2 つ以上残るようにする。**

### B3 開始 2 分後に 1 行だけ確かめる（2 分）

```bash
grep -m1 num_learnable_params /opt/dlami/nvme/out/pi05_ctrl1500.log
tail -5 /opt/dlami/nvme/out/pi05_ctrl1500.log
# 出発点の設定に torch.compile が入っていないこと（B2 を流す前に見てもよい）
python3 -c "
import json; print('compile_model =',
    json.load(open('/opt/dlami/nvme/weights/rp3000/config.json'))['compile_model'])"
```

| 見るもの | 期待値 | 違ったら |
|---|---|---|
| `num_learnable_params` | **4143404816** | **693422112 ならその場で止める。**行動側しか学習していない |
| `compile_model` | **False**（8/29 にこのマシンで `rp3000/config.json` を読んだ。`train_pi05.py` はこの値を上書きしない） | **True なら止める。**チェックポイントに焼き付き、評価サーバの起動に 1,177.9 秒かかって落ちる |
| `tail` の最後の行 | `step:20 smpl:1K ... loss:... updt_s:... mem_gb:...` | 例外が出ていれば A3 の環境に戻る |

### B4 チェックポイントを HF へ上げるループを始める（10 分）

**これが分かると何が決まるか: セッションが消えたときに何が残るか。
`/opt/dlami/nvme` はインスタンス終了で消えるので、上がっていないものは失われる。**

```bash
cat > ~/upload_ckpt.sh <<'EOF'
#!/usr/bin/env bash
# 出来上がったチェックポイントの pretrained_model だけを HF へ上げる。
# 上げ終えた印は $HOME に置く（チェックポイントの中に置くと、次の上げに混ざる）。
set -u
OUT=/opt/dlami/nvme/out/pi05_ctrl1500
REPO=siruku6/pi05_ctrl1500
mkdir -p ~/uploaded
while true; do
  for d in "$OUT"/checkpoints/*/pretrained_model; do
    [ -d "$d" ] || continue
    step=$(basename "$(dirname "$d")")
    [ -f ~/uploaded/"$step" ] && continue
    # 書き終える前に上げないよう、30 秒間サイズが変わらないことを確かめる
    a=$(du -sb "$d" | cut -f1); sleep 30; b=$(du -sb "$d" | cut -f1)
    [ "$a" = "$b" ] || continue
    if hf upload "$REPO" "$d" "$step" --repo-type model; then
      date -Iseconds > ~/uploaded/"$step"
      echo "$(date -Iseconds) 上げた $step" >> ~/upload_ckpt.log
      rm -rf "$(dirname "$d")/training_state"   # 16 GB。上げないので消して空きを戻す
    fi
  done
  pgrep -f "train_pi05.py --run-name pi05_ctrl1500" >/dev/null || break
  sleep 60
done
echo "$(date -Iseconds) 上げるループを終える" >> ~/upload_ckpt.log
EOF
chmod +x ~/upload_ckpt.sh
export PATH="$HOME/.local/bin:$PATH"
nohup ~/upload_ckpt.sh > ~/upload_ckpt.nohup 2>&1 &
echo "PID $!"
```

**学習が終わってからもう一巡してから抜けるので、最後の 001500 も上がる。**

### B5 最初の 500 ステップが HF に載るまで見届ける（40 分）

**ここまでが在席である。**500 ステップは 10.24 秒/step なら 85 分後に出る。
**待ちきれないときは、A5 で作った `probe60` の `pretrained_model` を先に 1 回上げて、
上げの道が通ることだけ確かめてから消す。**

```bash
watch -n 60 'ls -la /opt/dlami/nvme/out/pi05_ctrl1500/checkpoints/ 2>/dev/null; \
  cat ~/upload_ckpt.log 2>/dev/null; \
  tail -2 /opt/dlami/nvme/out/pi05_ctrl1500.log'
```

**確かめるのは 2 つ。**(a) `checkpoints/000500/` が出来ること、
(b) `~/upload_ckpt.log` に「上げた 000500」が出ること。
**(b) が出ないうちは離席しない。**上げの道が通っていなければ、
夜通し回しても朝には何も残らない。

### B6 監視セルを実行状態のまま残して離席する（5 分）

**これが分かると何が決まるか: 待機による自動停止（1 時間）で学習が殺されないか。**

**`nohup` で走らせた学習は「セルの実行」ではない。**セルが 1 つも動いていない状態が
1 時間続くと自動停止する規定なので、**カーネルを動かし続けるセルを 1 つ残す。**

```python
# **このセルは Notebook のセルとして実行する（`!` も `%%bash` も付けない）。**
import os, subprocess, time, urllib.request

LOG = "/opt/dlami/nvme/out/pi05_ctrl1500.log"
alive = lambda: bool(subprocess.run(
    ["pgrep", "-f", "train_pi05.py --run-name pi05_ctrl1500"],
    capture_output=True).stdout)

while alive():
    tail = subprocess.run(["tail", "-n", "1", LOG], capture_output=True, text=True).stdout
    print(time.strftime("%H:%M"), tail.strip()[-160:], flush=True)
    time.sleep(300)

print("学習が終わった。上げ残しを待つ", flush=True)
while subprocess.run(["pgrep", "-f", "upload_ckpt.sh"], capture_output=True).stdout:
    time.sleep(60)
print(open(os.path.expanduser("~/upload_ckpt.log")).read(), flush=True)

# A6 で GET が 200 だったときだけ、自分でサーバを止める
url, tok, user = (os.environ.get(k) for k in
                  ("JUPYTERHUB_API_URL", "JUPYTERHUB_API_TOKEN", "JUPYTERHUB_USER"))
if url and tok and user:
    req = urllib.request.Request(f"{url}/users/{user}/server", method="DELETE",
                                 headers={"Authorization": f"token {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print("Stop Server ->", r.status)
    except Exception as e:
        print("止められなかった。待機による自動停止に任せる:", e)
```

**このセルを実行したまま、ブラウザのタブは閉じてよい。**
**カーネルはサーバ側で動き続ける。**

---

## 夜 C 放置する

**やることは無い。**下が同時に走っている。

| 走っているもの | 終わる時刻（例・10.24 秒/step） | 落ちたときに残るもの |
|---|---|---|
| 学習（1,500 ステップ） | 19:50 | `checkpoints/` に出来たところまで |
| 上げるループ | 20:20 | **HF に上がったところまで**（これだけが本当に残る） |
| 監視セル | 20:20 → Stop Server | — |

**最悪でも T0 ＋ 12 時間（例 01:00）に自動停止する。**消費が 12 時間を超えることはない。

---

## 翌朝 15 分

```bash
# このマシンから。Blackwell に入り直す必要は無い
python3 -c "
from huggingface_hub import HfApi
for f in HfApi().repo_info('siruku6/pi05_ctrl1500', repo_type='model').siblings:
    print(f.rfilename)"
```

| 上がっていたもの | 次の 1 手 |
|---|---|
| `000500` / `001000` / `001500` の 3 つ | **`001500` を次の提出枠で出す。**`pi05_trial1500` の同じステップ数と 1 対 1 で比べられる |
| 途中まで | 上がっている中でいちばん大きいステップ数を出す。**対になる相手が `pi05_trial1500` にある**（000500 / 001000 / 001500） |
| 何も無い | 上げの道が通っていなかった。**B5 を飛ばしたのが原因である。**翌日は B5 を必ず待つ |

**評価はこの重みに対して回さない。**手元の 72 行は
**採点側で上がった量（衝突となめらかさ）を見ていない**ので、
**回しても判断が変わらないうえ 1 時間かかる。**
使うのは**大崩れの検出**だけである。

---

## 想定と違ったときの分岐一覧

### 分岐 1 VRAM が 96 GB でなかった

**判定は A2 の `memory.total` と A5 の `mem_gb` で行う。**

| 見えた VRAM | どうするか |
|---|---|
| 80 GB 以上 | **そのまま batch 64。**A100 80GB で 49.37 GB だったので入る |
| 48〜80 GB | batch 64 のまま試す。**`mem_gb` が 49.37 前後で収まれば問題ない** |
| **OOM した** | `--batch-size 32 --steps 3000` にする。**サンプル数を 96,000 で揃える。**ただし**比較の変数が 2 つになる**ので、報告に「batch が違う」と 1 行残す |
| 24 GB 以下 | **フルパラメータ学習は載らない。**この日の学習は無し。D の結果だけ持ち帰る |

**`--gradient-checkpointing` は既定で true である。**外さないこと。

### 分岐 2 外へ出られない

**A1 で判定する。**PyPI も HF も通らなければ、**この日の Blackwell は
D（配布データセットの棚卸し）専用にする。**D は外への接続を 1 度も使わない。
**終わったら止める。**その日のうちに RunPod へ $50 積む判断を出す。

### 分岐 3 `/opt/dlami/nvme` が 無い

**チェックポイント 1 個が 25 GB で、`$HOME` は 40 GiB しかない。**

```bash
# 置き場を ~/data（100 GiB）に変える
export HF_HOME=~/data/hf_home
export OUTPUT_ROOT=~/data/out
```

**`~/data` は手で保存しないと消える**（`parc-home-sync data-push`）。
**学習中は push しない**（書きかけを写す）。**HF へ上げる道だけを頼りにする。**

**`~/data` も足りないなら、`--save-freq` を 750 にして保存を 2 つに減らす。**
学習データ 15.83 GB ＋ チェックポイント 25 GB × 2 ＝ 66 GB で、100 GiB に収まる。

### 分岐 4 HF の空きが 27 GB に足りない

**非公開の枠は 100 GB で、8/29 の時点で 67.03 GB 使っている。**
**RunPod 側（依頼 008）も同じ枠へ上げているので、夕方には減っている可能性がある。**

| 手 | 戻る量 | 決めるのは |
|---|---|---|
| `siruku6/pi05_stage1_24000` を消す | **9.37 GB** | **親の判断。**行動側だけの 24,000 ステップで、依頼 008 が「打ち止め」と結論した重みである |
| `siruku6/parc2026-submission` の古い版を消して履歴を潰す（`super_squash_history`） | 8/29 に 76.4 → 57.7 GB になった実績あり | 親の判断 |
| **上げるのを `001500` だけにする** | 要るのが 8.8 GB で済む | **その場で決めてよい。**比較の要は `001500` である |

**どれも取れないなら `~/data` へ置き、翌朝このマシンへ引き取る。**

### 分岐 5 待機による自動停止で学習が死んだ

**B6 の監視セルを実行し忘れると起きる。**気づくのは翌朝である。
**HF に上がったところまでは残っている。**翌日は `--init-from` に
上がっている中でいちばん大きいステップ数を渡し、**残りのステップ数だけ回す。**
**学習率は 5e-6 のまま**（刻みは引き継がれないので、続きではなく新しい走りになる）。

---

## 落とし穴（踏むと 1 日が消えるもの）

| 落とし穴 | 対処 |
|---|---|
| **`docker` は使えない**（資料 v1.1 でサポート外） | Notebook の python へ直に pip |
| **`hf auth login` を忘れると非公開リポジトリが `not found` に見える** | 先にログイン。**名前の間違いではない** |
| **`hf` コマンドが無い**（古い `huggingface_hub` では `huggingface-cli`） | A2 の `which hf huggingface-cli` で見る。無いほうを使うか、`pip install --user -U huggingface_hub` で新しくする。引数の並びは `hf upload <repo> <ローカル> <リポジトリ内のパス>` で同じ |
| **学習のチェックポイントに `vlm/` は付かない** | 評価も zip も落ちる。`rp3000` から `cp -r` する（`ln -s` は駄目）。**今日は要らない**（提出物は重みのある機械で組む） |
| **`df` が答えない共有ストレージがある** | 使用量は `du -sh`、枠は画面で見る |
| **ディスクが埋まると書いている途中のファイルが壊れる** | 書く前に空きを見る。**`policy_server.py` を実際に 0 バイトにした** |
| **`--learning-rate` を省くと 10 倍から再スタートする** | 必ず `5e-6` を書く |
| **`--camera-viewpoint-augment` を惰性で付ける** | **今日は付けない。**これが変数である |
| **長い `docker exec` は成功しても終了コード 120 を返す**（他の機械の話） | ここでは `nohup` なので、**ログの最終行と成果物で判定する** |

---

## 確認したい仮定

- **`~/dataset` の中身は 2026-08-26 の資料の記述から書いています。現物を見ていません。**
  tar の名前と中の階層が違えば、D2・D3 の `find` と `grep` の当たりが変わります
  （まず `tar tf … | head` で並びを見てください）。
- **Notebook の python で `pip install --user` が使えるものとしています。**
  使えなければ A3 の分岐（`python -m venv --system-site-packages ~/venv`）に移ります。
- **`nohup` で始めたプロセスがセルの終了後も生き続けるものとしています。**
  死ぬなら、学習そのものを B6 の監視セルの中で `subprocess.run` として動かす形に
  変える必要があります（ただしセルが切れると学習も死ぬので、`~/data` への保存間隔を
  詰めることになります）。
- **`JUPYTERHUB_API_TOKEN` で自分のサーバを止められるかは未確認です**（A6 で読むだけ）。
  止められなければ、待機による自動停止で 1 時間ぶん余計に消費します。
- **RTX PRO 6000 Blackwell の VRAM 96 GB は NVIDIA の製品仕様で、配布資料には
  書かれていません。**外れると分岐 1 に移ります。
- **HF の非公開の枠の空きは 8/29 にこのマシンから数えた 67.03 GB を基準にしています。**
  依頼 008（RunPod）が同じ枠へ上げるので、**明日の夕方には減っています。**B1 で数え直します。

---

## あとで見る資料

| 知りたいこと | ファイル |
|---|---|
| どのマシンに何があるか・60 時間の減り方 | [../resources/infra_and_docker.md](../resources/infra_and_docker.md) |
| 配布データセットのタスク構成（8/29 に数えたもの） | [../reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md) |
| 流した学習・重み・版・出した点 | [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) |
| 手元の 72 行が重みの判定に使えない理由 | [../knowledge/18_local_eval_missed_the_gain.md](../knowledge/18_local_eval_missed_the_gain.md) |
| RunPod で同じ学習を回す手順 | [06_runpod_pi05.md](06_runpod_pi05.md) |
| 会社の RTX6000 へ評価環境を持ち込む | [12_bring_eval_to_rtx6000.md](12_bring_eval_to_rtx6000.md) |
