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

**学習を Docker で回す道は無い。**配布資料 v1.1 p14 が
「配布しているOmnicampus提出の事前チェック⽤Docker以外のDockerの使⽤はサポートしていません」
「Omnicampus提出の事前チェック⽤DockerのDocker run のみをサポートしています」と書いている。
**`solution/train/Dockerfile.pi05` と同じ依存構成を素の venv に作る**（下の A3）。

作成 2026-08-26、**2026-08-30 の計画として全面的に書き直した**（2026-08-29）。
**2026-08-31、実機の結果を受けて A3（環境構築）を書き直し、
「2 回目以降のセッション」の節を足した。**
出典は [本選の配布資料 v1.1](../materials/PARC2026開発コンペティション_本選_v1.1.pdf)。

## 項目一覧

| 節 | 一行要約 |
|---|---|
| [セッションの寿命](#セッションの寿命--何をしても-12-時間で切れる) | 起動から 12 時間で切れる。セルが動いていても切れる |
| [済んだ確認](#済んだ確認打ち直さない結果だけ残す) | **A0〜A2 は通した。打ち直さない。**結果の値だけ置いてある |
| [置き場](#置き場毎セッション効く読み飛ばさない) | `$HOME` は自動保存・`~/data` は手動・`/opt/dlami/nvme` は消える・`~/dataset` は読み出し専用 |
| [A3 環境](#a3-学習の環境を作る毎セッション-1-行) | **毎セッション 3 行で戻る**（実測 1 分）。土台の事実と、まだ起こりうる分岐 |
| [A4〜A7](#a4-出発点の重みを取り往復の速さを測る15-分) | **ここから未実行。**重みを取る → 速さを測る → 止め方 → ステップ数を決める |
| [午後 D（並行）](#午後-d並行配布データセットの棚卸し40-分cpu-だけ) | 未実行。tar を展開せずに数える（GPU も接続も使わない） |
| [夕方 B](#夕方-b-対照の学習を始める90-分) | 未実行。nohup で始め、保存のたびに HF へ上げる |
| [夜 C](#夜-c-放置する) | 監視セルが待機による自動停止を防ぎ、終わったら止める |
| [翌朝](#翌朝-15-分) | 上がった重みを数え、提出の順番を決める |
| [2 回目以降のセッション](#2-回目以降のセッション--60-時間は12-時間--5-回である) | **60 時間は連続していない。**12 時間 × 最大 5 回で、境目ごとに払うものがある |
| [想定と違ったとき](#想定と違ったときの分岐一覧) | batch・HF の空き・待機による停止の 3 つ |
| [落とし穴](#落とし穴踏むと-1-日が消えるもの) | 過去に実際に踏んだもの |
| [確認したい仮定](#確認したい仮定まだ現物を見ていないもの) | 現物を見ていない前提 |

---

## セッションの寿命 —— 何をしても 12 時間で切れる

| 止まり方 | 条件 |
|---|---|
| **自分で止める** | `File > Hub Control Panel > Stop Server`。**押した瞬間に消費が止まる** |
| 待機による自動停止 | セルの実行が何も無い状態が **1 時間**続いたとき |
| 起動時間による自動停止 | **起動から 12 時間。セルが動いていても切れる。本選 1 では延長ボタンが使えない** |

**ブラウザのタブを閉じても止まらない。**画面の緑枠が今回の 12 時間に対する残り、
青枠が 60 時間に対する残りで、**青枠は起動時点の値のまま更新されない。**

**`T0`（Notebook の画面が出た時刻）を紙に書き留める。**締め切りはすべてここから決まる。
**60 時間の割り方は下の[2 回目以降のセッション](#2-回目以降のセッション--60-時間は12-時間--5-回である)にある。**

---

## 済んだ確認（**打ち直さない。結果だけ残す**）

**A0〜A3 は 2026-08-31 の 1 回目のセッションで通した。**打っていたコマンドは消した。
**環境は `$HOME` にあるので、2 回目以降のセッションでも A3 の 1 行だけで戻る。**

| 確認 | 結果 |
|---|---|
| **A1 外へ出られるか** | **PyPI・HF の API・素の HTTPS・git の 4 つとも通った。**帯域は **133〜566 MB/s**（下の A3 の実測） |
| **A2 機械** | RTX PRO 6000 Blackwell **97,887 MiB**・**他に誰も使っていない**・compute capability **12.0**・**CPU 16 コア**・**RAM 124 GB**・`/opt/dlami/nvme` **1.7 TB**・**root になれる**・Ubuntu 22.04.5 |
| **A3 環境** | 下の[A3](#a3-学習の環境を作る毎セッション-1-行) |

**`--num-workers` は 8 にする**（CPU 16 コアの半分）。

### 置き場（**毎セッション効く。読み飛ばさない**）

**置き場はこう決める。**配布環境のストレージ構成は本選資料 v1.1 の 14 ページにある。

| 領域 | 容量 | 保存のされ方 |
|---|---|---|
| `$HOME` | **40 GiB**（資料 v1.1 p14）。**ただし 8/31 の実測は 98 GB だった**（`open_items.md` の `F53`） | **10 分ごとと終了時に自動保存**。起動時に最後の状態が戻る |
| `~/data` | **100 GiB** | **保存は手動**（`parc-home-sync data-push` / `data-pull`） |
| `~/dataset` | — | **読み出し専用。**配布データセットとベースライン重み。**消えない** |
| `/opt/dlami/nvme` | 約 1.7 TB | **永続化されない。**停止・終了で消える |

**`$HOME` の中の 3 つは symlink である**（8/31 の `ls -lka ~` で確認）。**どれも `$HOME` の
自動保存には乗らない**——保存されるのは `$HOME` の実体だけで、symlink の先は別の領域である。

| symlink | 先 | 意味 |
|---|---|---|
| `~/data` | `/opt/dlami/nvme/data/<アカウント名>` | **消える領域の上にある。**`parc-home-sync data-push` を打った分だけが残る |
| `~/.cache` | `/opt/dlami/nvme/cache/<アカウント名>` | **消える。****だから `hf auth login` のトークン（`~/.cache/huggingface/token`）も毎回消える。**トークンは `~/.hf_token`（実体が `$HOME`）に置く（A3） |
| `~/dataset` | `/opt/parc/dataset` | **配布データセットの実体。**読み出し専用で消えない |

**`$HOME` の中身は root 所有である**（同じ `ls`）。書けているので実害は無いが、
**`chown` を前提にした手順は書かないこと。**

| 何を | どこへ | なぜ |
|---|---|---|
| venv（約 8.2 GB）・lerobot の clone（24 MB）・python3.12 | **`$HOME`** | **自動保存されるので、2 回目以降のセッションで作り直さなくて済む。**`~/.local` に `--user` で入れる道は無い（下の A3） |
| 学習データと HF のキャッシュ | `/opt/dlami/nvme/hf_home` | 15.83 GB ある。`$HOME` の 40 GiB を食い潰さない |
| 学習の出力（1 個 11 GB） | `/opt/dlami/nvme/out` | 同上。**セッション終了で消えるので、HF へ上げるまでが勝負** |
| 数えた結果のテキスト | `~/`（`$HOME`） | 自動保存で残る |

**配布データセットは `~/dataset` の tar を `~/data` へ展開してから使う**（資料 v1.1 p19。
`~/dataset` は読み出し専用で、展開用のシェルスクリプトが配布されている）。
**`~/dataset` は消えないので、HF から 15.83 GB を落とし直す必要は無い**——
2 回目以降のセッションでも、そこから展開し直すだけで済む。

### A3 学習の環境を作る（**毎セッション 1 行**）

**これが分かると何が決まるか: 配布環境で π0.5 の学習が動くか。**
**1 回目（2026-08-31）で通した。**環境は `$HOME` にあるので、
**2 回目以降のセッションはこの 3 行を打ち直すだけで戻る**（実測 1 分）。

```sh
export HF_TOKEN="$(cat ~/.hf_token)"
bash ~/pi05_code/solution/train/omnicampus_setup.sh 2>&1 | tee ~/recon_setup.txt
. ~/env_pi05.sh
```

**`source` ではなくドット（`.`）で読む。**Notebook のセルは `/bin/sh`（dash）で動くことがあり、
そこに `source` は無い（8/31 に `/bin/sh: source: not found` を踏んだ）。

**通ったことの見分け方は最後の 3 行である。**

```
  lerobot 0.6.0（/home/<account>/lerobot/src/lerobot/__init__.py）
  PI05Config: ok
  完了。
```

**`~/.hf_token` を作るのは初回だけ**（`$HOME` なので残る）。
`hf_` で始まる英数字だけの 1 行にする——**置き換え用の文字列をそのまま貼ると**
`hf` が `'ascii' codec can't encode characters` で落ちる（8/31 に踏んだ）。

```sh
printf '%s' 'hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' > ~/.hf_token && chmod 600 ~/.hf_token
```

**スクリプトが何をするかは
[../../solution/train/omnicampus_setup.sh](../../solution/train/omnicampus_setup.sh)
の冒頭に書いてある。**土台になっている事実は 3 つ。

| 事実 | 出どころ |
|---|---|
| **カスタム Docker は使えない**（配布の事前チェック用イメージの `docker run` だけがサポート対象）。**あのイメージは提出前の最終確認に使う** | 本選資料 v1.1 p14。採点環境の Docker で学習しない理由は `open_items.md` の「決めた」8/31 の行 |
| 配布環境の python は **3.10.12**（virtualenv・user site 無効）なので、**`--user` は使えず python3.12 を自分で用意する**（`uv`。実測 457 ミリ秒） | 8/31 の実測 |
| 学習に使う **lerobot は v0.6.0 から下げられない**（`requires-python >= 3.12`） | `001500`（採点側で Track1 0.492）が v0.6.0 製で、梯子はその続きである。**版を変えると「歩数だけを変える」が崩れる** |

**分岐（まだ起こりうるものだけ）。**

| つまずき | どうするか |
|---|---|
| `not found` と言われる（`pi05_code`・`pi05_full_runpod` など） | **名前の間違いではない。**3 つとも非公開なので、**認証していないだけである。**`~/.hf_token` を置いて `. ~/env_pi05.sh` → `hf auth whoami` |
| `source: not found` | セルが `/bin/sh` である。**ドット（`.`）で読む** |
| `[4/6]` が `既に入っている: lerobot`（版が空） | **古い版のスクリプトである。**`hf download siruku6/pi05_code --local-dir ~/pi05_code` で取り直す |
| `undefined symbol: _ZN...` / `no kernel image is available` | torch 一式の版の対が壊れている。`rm -rf ~/venv-pi05` して作り直す（実測 1 分） |

#### A3 の実測（2026-08-31・1 回目のセッション）

| 測ったもの | 値 |
|---|---|
| **帯域** | **wheel 1 個あたり 133〜566 MB/s**（820 MB の torch 本体が 1 秒）。**torch 一式 4 つで合わせて 1 分** |
| venv の大きさ | torch 一式で **7,256 MB**、lerobot 一式を入れて完成 |
| uv で python3.12 | cpython 3.12.14（32.5 MiB）を **457 ミリ秒** |
| NPP | **root なので `ldconfig` に登録できた** |
| GPU | capability **(12, 0)**・**bf16 の行列積が通った**（sm_120 のカーネルが入っている） |
| lerobot | **0.6.0（editable）・`PI05Config` の import まで通った** |
| **`$HOME` の容量** | **98 GB（`/dev/loop5`）。**同じ日の先の測定では **40 GB（`/dev/loop4`）**で、資料 v1.1 p14 も 40 GiB と書いている。**食い違っている。**`df -h ~ ~/data` を両方指定して確かめる（`open_items.md` の `F53`） |

**帯域が速いと分かったので、次の 3 つはもう考えなくてよい**（`open_items.md` の「決めた」8/31）。
**①4.5 GB の取得を避ける工夫 ②python 3.10 のまま配布環境の torch を使う道
③採点環境の Docker で学習する道。**

### A4 出発点の重みを取り、往復の速さを測る（15 分）

**これが分かると何が決まるか: 9.4 GB を上げるのに何分かかるか。
それが分かると、A7 で「上げ切る時間」を差し引いてステップ数を決められる。**

**どちらの走りをするかで、取る重みが違う。**

| 走り | 出発点 | 中身 |
|---|---|---|
| ✅ **梯子（既定）** | **`siruku6/pi05_trial1500` の `001500`（8.73 GiB）** | **採点側で Track1 0.492・Track2 0.233 を出した重みそのもの。**カメラ拡張あり・lr 5e-6・batch 64 のまま**歩数だけ延ばす**（依頼 011・[../talk/260831_today.md](../talk/260831_today.md) 1 節） |
| 対照 | `siruku6/pi05_full_runpod`（＝ `rp3000`・9.37 GB） | **カメラ拡張なしで 1,500 歩。**拡張が効いたかを切り分ける走りで、**判定には提出枠が 1 日要る**（`open_items.md` `#5` の推しは「RunPod の結果が出たあと」） |

```sh
. ~/env_pi05.sh          # HF_HOME・OUTPUT_ROOT・venv・HF_TOKEN がこれで入る
mkdir -p /opt/dlami/nvme/weights

# 梯子（既定）。--include で 001500 だけ取る。repo 全体は 001000 も含み 17.5 GiB ある
time hf download siruku6/pi05_trial1500 --include '001500/*' \
    --local-dir /opt/dlami/nvme/weights/trial1500
ls -la /opt/dlami/nvme/weights/trial1500/001500
du -sh /opt/dlami/nvme/weights/trial1500/001500

# 対照をやる場合はこちら
# time hf download siruku6/pi05_full_runpod --local-dir /opt/dlami/nvme/weights/rp3000
```

**`--init-from` に渡すのは `config.json` と `model.safetensors` が直下にあるディレクトリである。**
`001500` は直下に 7 ファイル、**`vlm/` も 4 ファイル同梱されている**（8/31 に HF 上で確認）。
**だから `rp3000` から `vlm/` をコピーする手間は要らない**（`pi05_full_runpod` から始める場合も同梱済み）。

**`Error: Model 'siruku6/pi05_full_runpod' not found.` は名前の間違いではない。**
**認証していないだけである。**`pi05_full_runpod`・`pi05_trial1500`・`pi05_code` の
**3 つとも非公開**で（8/31 に確認）、HF は非公開リポジトリを未認証の相手には
「無い」と答える。**先に `~/.hf_token` を置いて `. ~/env_pi05.sh` し、
`hf auth whoami` で名前が出ることを確かめる**（A3 のスクリプトの最後に手順がある）。

**`Error: Invalid value. 'ascii' codec can't encode characters in position 10-17` が出たら、
`~/.hf_token` に置き換え用の文字列がそのまま入っている**（8/31 に実際に起きた。
説明文の `hf_あなたのトークン` を貼ってしまい、**そのせいで修正版のスクリプトも降りて来ず、
直したはずの不具合がもう一度出た**）。**`hf_` で始まる英数字だけの 1 行に置き換える。**
A3 のスクリプトは `[5/6]` でこの形を検査して警告する。

**降りてきた秒数から MB/s を出す**（9,370 MB ÷ 秒数）。
**参考値: RunPod では 9.4 GB の上げが 16 分だった**（`docs/howto/06_runpod_pi05.md`）。
上げは降りより遅いことが多いので、**上げは「降りの 2 倍の時間」と見積もる。**

### A5 batch 64 が載るかと、1 ステップの時間を、60 ステップで測る（35 分）

**これが分かると何が決まるか: batch を 64 のままにできるか（VRAM）と、
1,500 ステップが今日のうちに終わるか（速さ）。チェックポイントが書けることも同時に分かる。**

**`train_pi05.py` は `$CODE_DIR`（= `~/pi05_code/solution/train`）にある。**A3 で HF から
降ろしてある（`experiments/` と `train.py` も同じ並びに入っている——`train_pi05.py` が
`from train import ...` と `from experiments import geometry, quantization` を実行するため）。

**`git clone` は使わない。**GitHub の HTTPS はパスワード認証が廃止されており、
個人アクセストークンを作らないと通らない。**運び口は HF 1 本にした**（`open_items.md` の「決めた」8/31）。
古い版を掴んでいないか不安なときは取り直す——`hf download siruku6/pi05_code --local-dir ~/pi05_code`。

```sh
. ~/env_pi05.sh          # HF_HOME・OUTPUT_ROOT・venv・HF_TOKEN・CODE_DIR がこれで入る
cd "$CODE_DIR"

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
| `mem_gb` | **49.40**（8/31 の Blackwell 実測。A100 80GB は 49.37） | [分岐 1](#分岐-1-batch-64-が載らなかった) |
| 1 ステップ | **6.85 秒**（8/31 の Blackwell 実測。A100 80GB は 10.05〜10.24 秒） | A7 の式に実測値を入れる |
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

### A7 ステップ数を決める

**これが分かると何が決まるか: 今回のセッションで何歩回せるか。**

**8/31 の 1 回目の実測。**

| 項目 | 値 |
|---|---|
| 1 歩 | **6.85 秒**（`updt_s` 6.840 ＋ `data_s` 0.007。A100 80GB の 10.05〜10.24 秒より **33% 速い**） |
| `mem_gb` | **49.40**（95.0 GB のうち。**batch 64 は余裕で載る**） |
| モデルの読み込み | **約 2 分**（`Creating policy` から `Start offline training` まで） |
| 学習データの取得 | **83 ファイルを 50 秒**（`HF_HOME` に残るので同じセッション内で 2 度目は無い） |
| チェックポイントの保存 | **21 秒**／個 |

```sh
python3 - <<'CALC'
import datetime
# ---- 実測値（8/31 の 1 回目。測り直したら差し替える）----
sec_per_step  = 6.85
load_sec      = 120      # モデルの読み込み
save_sec      = 21       # チェックポイント 1 個の保存
reserve_min   = 40       # 最後の上げと、自分で止めるための余裕
t0            = "05:42"  # A0 で書き留めた起動時刻（UTC）
# ------------------------------------------------------
now = datetime.datetime.utcnow()
start = now.replace(hour=int(t0[:2]), minute=int(t0[3:]), second=0, microsecond=0)
deadline = start + datetime.timedelta(hours=12)
budget = (deadline - now).total_seconds() - load_sec - reserve_min * 60
print(f"締切 {deadline:%H:%M} UTC / 使えるのは {budget/3600:.2f} 時間")
for cand in (4000, 3500, 3000, 2500, 2000, 1500, 1000, 500):
    need = cand * sec_per_step + (cand // 500) * save_sec
    if need <= budget:
        print(f"--steps {cand}（学習 {need/3600:.2f} 時間・500 歩ごとに {cand//500} 個保存）")
        break
else:
    print("500 も入らない。学習は回さず、環境を残して止める")
CALC
```

**通算の歩数と、出力の名前は違う。**`--init-from` は歩数のカウンタを引き継がないので、
**出力は `000500`・`001000`… から始まる。**`001500` から続けた場合、
**出力の `000500` は通算 `002000` である。**HF へ上げるときに通算の名前へ直すこと
（直さないと `pi05_trial1500` の `000500` と区別できなくなる）。

**上げる本数は HF の空きで決まる。**チェックポイント 1 個が **8.8 GB**、非公開の枠は
100 GB で 8/29 の実測が 67.03 GB（空き約 33 GB）だった。**500 歩ごとに全部上げると入らない。**
[分岐 2](#分岐-2-hf-の空きが-27-gb-に足りない)と
[../talk/260831_today.md](../talk/260831_today.md) の「決めてほしいこと 1」に従い、
**1,000 歩ごと・最新のいくつかだけにする。**

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
| **27 GB 未満** | [分岐 2](#分岐-2-hf-の空きが-27-gb-に足りない) |

### B2 学習を nohup で始める（10 分）

```bash
. ~/env_pi05.sh          # HF_HOME・OUTPUT_ROOT・venv・HF_TOKEN がこれで入る
cd "$CODE_DIR"          # NPP は A3 で ldconfig に登録済み。LD_LIBRARY_PATH は要らない

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

## 2 回目以降のセッション —— 60 時間は「12 時間 × 5 回」である

**これが分かると何が決まるか: 1 本の長い学習は取れない。梯子を何段に割るか。**

**起動から 12 時間で自動的に止まり、本選 1 では延長ボタンが使えない**
（本選資料 v1.1 p15）。**60 時間は連続した 60 時間ではなく、最大 5 回の
12 時間である。**セッションをまたぐたびに、次の 3 つを払う。

| 払うもの | 1 回目 | 2 回目以降 | なぜ |
|---|---|---|---|
| 環境（venv 約 8.2 GB） | **A3 の実測（60 分見込み）** | **1 分** | `$HOME` にあり自動保存される。スクリプトは有無を見て作り直さない |
| 学習データ | `~/dataset` の tar を `~/data` へ展開 | **同じだけ払う** | `~/data` の実体は消える領域である（上の「置き場」）。**ただし `~/dataset` は消えないので、HF から落とし直す必要は無い** |
| **続きの重み（11 GB）** | 不要（`001500` を HF から取る） | **HF から降ろす分** | `/opt/dlami/nvme` の出力は消える |

**1 セッションの正味の学習時間は、12 時間からこの 3 つと「上げ切る時間」を引いた残りである。**
**A3 と A4 の実測が出るまで、この残りは決めない。**

**セッションの終わりで気をつけること 2 つ。**

| こと | 中身 |
|---|---|
| **12 時間の打ち切りは待たずに、自分で止める** | 押した瞬間に消費が止まる。**放置して自動停止に任せると、学習が終わってからの分まで払う** |
| **最後のチェックポイントが HF に上がってから止める** | `/opt/dlami/nvme` は消える。**上がっていない歩数は存在しないのと同じ**（依頼 011） |

**`$HOME` の 40 GiB に何を置くかは足し算で決まる。**

| 置くもの | 大きさ | 置くか |
|---|---|---|
| venv ＋ lerobot ＋ python3.12 | **約 8.5 GB** | **置く**（毎回作り直す代金が消える） |
| 続きの重み 1 個 | **9.4〜11 GB** | **判断が要る。**置けば次のセッションの HF 往復が消えるが、40 GiB のうち残りが 20 GB を切る |
| チェックポイントの控え 1 個 | 11 GB | **置かない。**HF が控えである |

**ここに 11 GB を置く案には、まだ測っていない risk が 1 つある**——
**`$HOME` は 10 分ごとに自動保存され、起動時に復元される。**30 GB 近い `$HOME` の
復元に何分かかるかは分からない。**復元が学習時間を食うなら、置かないほうが速い。**
**2 回目のセッションの起動にかかった分数を測れば答えが出る。**

---

## 想定と違ったときの分岐一覧

### 分岐 1 batch 64 が載らなかった

**VRAM は 97,887 MiB を実測済みなので、載らないのは A5 で OOM したときだけである。**
そのときは `--batch-size 32 --steps 3000` にして**サンプル数を 96,000 で揃える。**
ただし**比較の変数が 2 つになる**ので、報告に「batch が違う」と 1 行残す。
**`--gradient-checkpointing` は既定で true である。外さないこと。**

### 分岐 2 HF の空きが 27 GB に足りない

**非公開の枠は 100 GB で、8/29 の時点で 67.03 GB 使っている。**
**RunPod 側（依頼 008）も同じ枠へ上げているので、夕方には減っている可能性がある。**

| 手 | 戻る量 | 決めるのは |
|---|---|---|
| `siruku6/pi05_stage1_24000` を消す | **9.37 GB** | **親の判断。**行動側だけの 24,000 ステップで、依頼 008 が「打ち止め」と結論した重みである |
| `siruku6/parc2026-submission` の古い版を消して履歴を潰す（`super_squash_history`） | 8/29 に 76.4 → 57.7 GB になった実績あり | 親の判断 |
| **上げるのを `001500` だけにする** | 要るのが 8.8 GB で済む | **その場で決めてよい。**比較の要は `001500` である |

**どれも取れないなら `~/data` へ置き、翌朝このマシンへ引き取る。**

### 分岐 3 待機による自動停止で学習が死んだ

**B6 の監視セルを実行し忘れると起きる。**気づくのは翌朝である。
**HF に上がったところまでは残っている。**翌日は `--init-from` に
上がっている中でいちばん大きいステップ数を渡し、**残りのステップ数だけ回す。**
**学習率は 5e-6 のまま**（刻みは引き継がれないので、続きではなく新しい走りになる）。

---

## 落とし穴（踏むと 1 日が消えるもの）

| 落とし穴 | 対処 |
|---|---|
| **カスタム Docker は使えない**（資料 v1.1 p14） | `$HOME` の venv に作る（A3）。**配布イメージの `docker run` は提出前の最終確認に使う** |
| **認証していないと非公開リポジトリが `not found` に見える** | `~/.hf_token` を置く（A3）。**名前の間違いではない** |
| **Notebook のセルが `/bin/sh` で動くことがある** | `source` が無い。**`.` で読む**（8/31 に踏んだ） |
| **学習のチェックポイントに `vlm/` は付かない** | 評価も zip も落ちる。`rp3000` から `cp -r` する（`ln -s` は駄目）。**今日は要らない**（提出物は重みのある機械で組む） |
| **`df` が答えない共有ストレージがある** | 使用量は `du -sh`、枠は画面で見る |
| **ディスクが埋まると書いている途中のファイルが壊れる** | 書く前に空きを見る。**`policy_server.py` を実際に 0 バイトにした** |
| **`--learning-rate` を省くと 10 倍から再スタートする** | 必ず `5e-6` を書く |
| **`--camera-viewpoint-augment` を惰性で付ける** | **今日は付けない。**これが変数である |
| **`train_pi05.py` は成功しても終了コード 120 を返す** | 8/31 の 60 歩で実測。`[train_pi05] === 終了（終了コード 0）===` を出したあと、`_Tee.flush` が閉じた stdout を flush しようとして `ValueError` になり、python が 120 で終わる。**`&&` で後続を繋がないこと。**判定は**ログの `終了（終了コード 0）` の行と、成果物の `du -sh` で行う** |

---

## 確認したい仮定（**まだ現物を見ていないもの**）

- **`~/dataset` の中身は資料の記述から書いています。現物を見ていません。**
  tar の名前と中の階層が違えば、D2・D3 の `find` と `grep` の当たりが変わります
  （まず `tar tf … | head` で並びを見てください）。
- **`nohup` で始めたプロセスがセルの終了後も生き続けるものとしています。**
  死ぬなら、学習を B6 の監視セルの中で `subprocess.run` として動かす形に変えます
  （ただしセルが切れると学習も死ぬので、保存間隔を詰めることになります）。
- **`JUPYTERHUB_API_TOKEN` で自分のサーバを止められるかは未確認です**（A6 で読むだけ）。
  止められなければ、待機による自動停止で 1 時間ぶん余計に消費します。
- **`$HOME` の容量が 40 GiB なのか 98 GB なのかが確定していません**（A3 の実測。`open_items.md` の `F53`）。
  40 GiB のほうが正なら、venv 一式（約 8.5 GB）以外を `$HOME` に置く余地は小さくなります。
- **`$HOME` の復元にかかる時間を測っていません。**2 回目のセッションの起動が長ければ、
  `$HOME` に置くものを減らすほうが速くなります。
- **HF の非公開の枠の空きは 8/29 に数えた 67.03 GB を基準にしています。**B1 で数え直します。

## あとで見る資料

| 知りたいこと | ファイル |
|---|---|
| どのマシンに何があるか・60 時間の減り方 | [../resources/infra_and_docker.md](../resources/infra_and_docker.md) |
| 配布データセットのタスク構成（8/29 に数えたもの） | [../reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md) |
| 流した学習・重み・版・出した点 | [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) |
| 手元の 72 行が重みの判定に使えない理由 | [../knowledge/18_local_eval_missed_the_gain.md](../knowledge/18_local_eval_missed_the_gain.md) |
| RunPod で同じ学習を回す手順 | [06_runpod_pi05.md](06_runpod_pi05.md) |
| 会社の RTX6000 で評価を回すとき | [03_gpu_eval.md](03_gpu_eval.md) の「会社の RTX6000 では EGL が効く」 |
