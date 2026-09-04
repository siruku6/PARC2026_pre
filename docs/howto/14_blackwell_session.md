# Blackwell サーバで 1 本回す（毎セッションの手順書）

**セッションを開くたびに、この順で打つ。**別のファイルを開く必要はない。

対象は運営が配布する GPU 環境（**Blackwell サーバ**・NVIDIA RTX PRO 6000 Blackwell 1 台）。
**本選 1 の期間全体で 60 時間しか使えず、起動してから自分で止めるまでずっと減る。**

> **2026-09-03 に、この文書から「1 回目だけの確認」を落とした。**
> 落としたのは A1・A2（機械を見る）・A4 の速さの測定・A5（batch 64 が載るか）・
> A6（自分で止められるか）・午後 D（配布データセットの棚卸し）である。
> **測って分かった値は [../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md) に集めた**
> ——**この手順書には二度書かない。**
> データセットの棚卸しの結果は [../reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md) 8 節にある。

**学習を Docker で回す道は無い。**配布資料 v1.1 p14 が
「配布しているOmnicampus提出の事前チェック⽤Docker以外のDockerの使⽤はサポートしていません」
と書いている。**`solution/train/Dockerfile.pi05` と同じ依存構成を素の venv に作る**（下の A3）。

## 項目一覧

| 節 | 一行要約 |
|---|---|
| [セッションの寿命](#セッションの寿命--何をしても-12-時間で切れる) | 起動から 12 時間で切れる。セルが動いていても切れる |
| **[毎セッション打つ 3 つ](#毎セッション打つ-3-つ環境重みデータ)** | **環境（1 分）・出発点の重み（HF から降ろす）・配布データの展開。****`/opt/dlami/nvme` は消えるので、重みは毎回降ろし直す** |
| [A3 環境](#a3-学習の環境を作る毎セッション-1-行) | **毎セッション 3 行で戻る**（実測 1 分）。土台の事実と、まだ起こりうる分岐 |
| [A7 ステップ数を決める](#a7-ステップ数を決める) | 残り時間から今回の `--steps` を計算する |
| [B 学習を始める](#b-学習を始める90-分) | nohup で始め、保存のたびに HF へ上げる |
| [C 放置する](#c-放置する) | 監視セルが待機による自動停止を防ぎ、終わったら止める |
| [翌朝](#翌朝-15-分) | 上がった重みを数え、提出の順番を決める |
| **[場面カメラを潰して 1 本](#場面カメラを確率で潰して-1-本回す92-の夜に足した)** | **いま回すのはこれである**（`open_items.md` の `F50`①） |
| **[111 タスクのデータで 1 本](#配布の-111-タスクのデータで-1-本回す94-に足したf48)** | **次に回すのはこれである**（`open_items.md` の `F48`）。**データの展開が 1 つ増える** |
| [2 回目以降のセッション](#2-回目以降のセッション--60-時間は12-時間--5-回である) | **60 時間は連続していない。**12 時間 × 最大 5 回で、境目ごとに払うものがある |
| [想定と違ったとき](#想定と違ったときの分岐一覧) | batch・HF の空き・待機による停止の 3 つ |
| [落とし穴](#落とし穴踏むと-1-日が消えるもの) | 過去に実際に踏んだもの |

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

## 毎セッション打つ 3 つ（環境・重み・データ）

**`/opt/dlami/nvme` はインスタンス終了で消える。**だから**セッションを開いたら、
学習を始める前にこの 3 つを毎回払う。****`$HOME` に残るのは venv だけである**
（パスの実体は [../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md) 3 節）。

| # | 払うもの | 1 回目 | 2 回目以降 |
|---|---|---|---|
| **1** | 環境（venv 約 8.5 GB） | 60 分 | **1 分**（`$HOME` にあるので作り直さない。下の A3） |
| **2** | **出発点の重み（8.8 GB）** | 15 分 | **同じだけ払う**（下の「重みを降ろす」） |
| **3** | 配布データの展開 | — | **同じだけ払う**（`~/dataset` の tar を `~/data` へ。**`~/dataset` は消えないので HF から落とし直す必要は無い**） |

### 重みを降ろす（**毎セッション。消えているのが普通である**）

**`/opt/dlami/nvme/weights` は前のセッションの終了時に消えている。**
**`ls` で確かめてから、無ければ降ろす。**

```sh
. ~/env_pi05.sh          # HF_HOME・OUTPUT_ROOT・venv・HF_TOKEN・CODE_DIR がこれで入る
hf auth whoami           # 名前が出ること。出なければ ~/.hf_token を先に置く（A3）

# 残っているかを先に見る（残っていれば降ろさない）
ls -la /opt/dlami/nvme/weights/trial1500/001500 2>/dev/null || echo "消えている。降ろす"

mkdir -p /opt/dlami/nvme/weights
time hf download siruku6/pi05_trial1500 --include '001500/*' \
    --local-dir /opt/dlami/nvme/weights/trial1500

# 降りたことを 2 つで確かめる
du -sh /opt/dlami/nvme/weights/trial1500/001500       # 8.7 GiB 前後
ls /opt/dlami/nvme/weights/trial1500/001500/vlm | wc -l   # 4（無いと評価と zip が落ちる）
```

**これがいまの既定の出発点である**——`siruku6/pi05_trial1500` の **`001500`**（8.73 GiB）。
**採点側で Track1 0.492・Track2 0.233 を出した重みそのもの**で、
[../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) の版 5.0.1 にあたる。
**`--init-from` に渡すのは `config.json` と `model.safetensors` が直下にあるディレクトリである。**

**別の重みから始めるとき。**

| 出発点 | 打つもの | 何のため |
|---|---|---|
| **`trial1500/001500`（既定）** | 上のとおり | **対照が 8.0.0 になる**（同じ出発点・同じ 2,500 歩で測れている） |
| `ladder2500/002500`（＝累計 4,000 歩・8.0.0 の重み） | `--include 'ladder2500/002500/*'` に変える | ここからさらに延ばす場合。**ただし 9/2 に「歩数では決まらない」が確定したので、いまは選ばない** |
| `rp3000`（＝`pi05_full_runpod`・9.37 GB） | `hf download siruku6/pi05_full_runpod --local-dir /opt/dlami/nvme/weights/rp3000` | **いまは選ばない。**カメラ拡張を付けずに始める場合の出発点で、**その対照は 9/2 に取り下げた**（`open_items.md` の判断 5） |

> **`Error: Model '…' not found.` は名前の間違いではない。認証していないだけである。**
> `pi05_full_runpod`・`pi05_trial1500`・`pi05_code` は**3 つとも非公開**で、
> HF は非公開リポジトリを未認証の相手に「無い」と答える。
> **他の罠は [../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md) 5 節にまとめてある。**

### 学習コードを降ろす（**9/3 以降は 1 度必ず取り直す**）

**`~/pi05_code` は `$HOME` にあるので前のセッションのものが残っている。**
**だが 9/2 に場面カメラを潰す実装を足したので、それが入っていない。**
**9/3 に HF へ上げたので、1 度取り直せば入る。**

> **取り直さないと、`--scene-camera-dropout` が「知らない引数」として弾かれる。**
> **黙って無視されるのではなく、その場で止まる**（argparse がエラーにする）。
> だから踏んでも 4 時間 46 分は失わないが、**起動した時間は減る。**

```sh
hf download siruku6/pi05_code --local-dir ~/pi05_code
ls ~/pi05_code/solution/train/experiments/camera_dropout   # 3 ファイル出れば新しい
```

---

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

### A7 ステップ数を決める

**これが分かると何が決まるか: 今回のセッションで何歩回せるか。**

**使う実測値は [../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md) 2 節にある**（1 歩 6.85 秒・読み込み 120 秒・保存 21 秒／個）。**下の計算はその 3 つを使う。**

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
[分岐 2](#分岐-2-hf-の空きが-27-gb-に足りない)のとおり、
**1,000 歩ごと・最新のいくつかだけにする**（**測れる粒度が 1 日 1 トラック 1 回なので、
それより細かく残しても使えない**。8/31 に決めた）。

**500 も入らないときは回さない。**環境（`~/.local` と `~/lerobot`）は `$HOME` に残るので、
**翌日は A3 を飛ばして 30 分早く始められる。**

---

## B 学習を始める（90 分）

**何を回すか。8.0.0 を作った学習と 1 つだけ違う走りである。**

| | 8.0.0（`ladder2500/002500`・8/31） | **今回** |
|---|---|---|
| 出発点 | `trial1500/001500` | 同じ |
| ステップ数 | 2,500 | 同じ |
| batch | 64 | 同じ |
| 学習率 | 5e-6 | 同じ |
| 種 | 42 | 同じ |
| 画像側 | 開けた | 同じ |
| **カメラ拡張**（`--camera-viewpoint-augment`） | **あり** | **あり（揃える）** |
| **場面カメラを確率で潰す** | **なし** | **`--scene-camera-dropout 0.5`** |

**変えるのは最後の 1 行だけである。**理由と設計は
[場面カメラを確率で潰して 1 本回す](#場面カメラを確率で潰して-1-本回す92-の夜に足した)節にある。

> **【9/3 に直した】この節には 8/31 の別の計画が残っていた**——**`--camera-viewpoint-augment`
> を外して「拡張のありなし」を測る対照の 1 本**である。**その計画は 9/2 に取り下げた**
> （`open_items.md` の判断 5。**拡張は Track2 の 4 分類のうち `Camera Viewpoints` だけに
> 当たるので、「届く」と「質」のどちらに効いたのかを分けられない**）。
> **だから拡張は付けたまま揃え、変数は「場面カメラを潰すか否か」の 1 つだけにする。**

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
create_repo("siruku6/pi05_camdrop2500", private=True, exist_ok=True)
print("上げ先を作った: siruku6/pi05_camdrop2500")
PY
```

**8/29 にこのマシンから数えた時点で合計 67.03 GB、空きは約 33 GB だった。**
**2,500 歩・`--save-freq 500` なら `pretrained_model` は 5 個（44 GB）できるが、空きに入らない。**
**上げるのは 1,000 歩ごと・最新のいくつかだけである**（A7 の最後）。

| 空き | どうするか |
|---|---|
| 27 GB 以上 | そのまま進む |
| **27 GB 未満** | [分岐 2](#分岐-2-hf-の空きが-27-gb-に足りない) |

### B2 学習を nohup で始める（10 分）

```bash
. ~/env_pi05.sh          # HF_HOME・OUTPUT_ROOT・venv・HF_TOKEN がこれで入る
cd "$CODE_DIR"          # NPP は A3 で ldconfig に登録済み。LD_LIBRARY_PATH は要らない

NW=$(python -c "import os;print(min(os.cpu_count()//2, 16))")
nohup python train_pi05.py --run-name pi05_camdrop2500 \
    --init-from /opt/dlami/nvme/weights/trial1500/001500 \
    --steps 2500 --batch-size 64 --num-workers "$NW" \
    --learning-rate 5e-6 --save-freq 500 \
    --freeze-vision-encoder false --train-expert-only false \
    --camera-viewpoint-augment --scene-camera-dropout 0.5 \
    > $OUTPUT_ROOT/pi05_camdrop2500.nohup 2>&1 &
echo "PID $!"
```

**`--camera-viewpoint-augment` は付けること。**8.0.0 が付けているので、
**外すと変数が 2 つになる。**（**9/3 まで、ここには「付けないこと」と書いてあった。
それは 9/2 に取り下げた 8/31 の計画の残りである**——上の B の導入を見ること。）

**`--scene-camera-dropout 0.5` が今回の変数である。**これを付けるには
**`~/pi05_code` を 1 度取り直しておく必要がある**（上の「学習コードを降ろす」）。
**取り直していないと、argparse が「知らない引数」として弾いてその場で止まる。**

**`--learning-rate 5e-6` を省かないこと。**省くと頂点 2.5e-5 に戻り、
**出発点の学習が 5e-6 の頂点から下り切ったところで終わっているのに、その 5 倍から
再スタートする。**
**`--init-from` は `pretrained_model` しか読まないので、学習率の刻みは引き継がれない**
（`load_training_state` が読む `training_state/` は `--resume` のときだけ使われる。
`lerobot/scripts/lerobot_train.py` の 385〜389 行と
`lerobot/common/train_utils.py` の 201 行）。

**`--steps` は A7 で決めた数に置き換える。**1,000 なら `--save-freq 500`（2 つ）、
500 なら `--save-freq 250`（2 つ）にして、**途中の重みが 2 つ以上残るようにする。**

### B3 開始 2 分後に 1 行だけ確かめる（2 分）

```bash
grep -m1 num_learnable_params /opt/dlami/nvme/out/pi05_camdrop2500.log
grep -m2 "\[camera_dropout\]" /opt/dlami/nvme/out/pi05_camdrop2500.log
tail -5 /opt/dlami/nvme/out/pi05_camdrop2500.log
# 出発点の設定に torch.compile が入っていないこと（B2 を流す前に見てもよい）
python3 -c "
import json; print('compile_model =', json.load(open(
    '/opt/dlami/nvme/weights/trial1500/001500/config.json'))['compile_model'])"
```

| 見るもの | 期待値 | 違ったら |
|---|---|---|
| `num_learnable_params` | **4143404816** | **693422112 ならその場で止める。**行動側しか学習していない |
| **`[camera_dropout]` の行** | **読み込みの worker ごとに 1 行出る** | **1 つも出なければ潰す処理が動いていない。**その場で止める（読み方は下の「始めた直後に確かめる 2 行」） |
| `compile_model` | **False**（**9/3 に HF の `pi05_trial1500` の `001500/config.json` を降ろして読んだ。**`train_pi05.py` はこの値を上書きしない。`rp3000` も 8/29 に False を確認済み） | **True なら止める。**チェックポイントに焼き付き、評価サーバの起動に 1,177.9 秒かかって落ちる |
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
OUT=/opt/dlami/nvme/out/pi05_camdrop2500
REPO=siruku6/pi05_camdrop2500
mkdir -p ~/uploaded
while true; do
  for d in "$OUT"/checkpoints/*/pretrained_model; do
    [ -d "$d" ] || continue
    # LeRobot は保存のたびに checkpoints/last という symlink を貼り直す。
    # これを拾うと HF に「last」という歩数のふりをした複製が 1 個できる（9/3 に踏んだ）。
    [ -L "${d%/pretrained_model}" ] && continue
    step=$(basename "$(dirname "$d")")
    [ -f ~/uploaded/"$step" ] && continue
    # 書き終える前に上げないよう、30 秒間サイズが変わらないことを確かめる
    a=$(du -sb "$d" | cut -f1); sleep 30; b=$(du -sb "$d" | cut -f1)
    [ "$a" = "$b" ] || continue
    # timeout: 1 個 9.35 GB なので正常なら 15 分。1 時間で諦めて次の巡へ回す。
    if timeout 3600 hf upload "$REPO" "$d" "$step" --repo-type model; then
      date -Iseconds > ~/uploaded/"$step"
      echo "$(date -Iseconds) 上げた $step" >> ~/upload_ckpt.log
      rm -rf "$(dirname "$d")/training_state"   # 16 GB。上げないので消して空きを戻す
    else
      # 成功しか書かないと、失敗と「1 度も試していない」を後から区別できない（9/3 に踏んだ）
      echo "$(date -Iseconds) 失敗 $step（終了コード $?）" >> ~/upload_ckpt.log
    fi
  done
  # 1 巡ごとの足跡。これが無いと、止まったときにどこで止まったかを言えない（9/3 に踏んだ）
  echo "$(date -Iseconds) 見回り $(ls -d "$OUT"/checkpoints/*/ 2>/dev/null | wc -l) 個" \
      >> ~/upload_ckpt.log
  pgrep -f "train_pi05.py --run-name pi05_camdrop2500" >/dev/null || break
  sleep 60
done
echo "$(date -Iseconds) 上げるループを終える" >> ~/upload_ckpt.log
EOF
chmod +x ~/upload_ckpt.sh
export PATH="$HOME/.local/bin:$PATH"
nohup ~/upload_ckpt.sh > ~/upload_ckpt.nohup 2>&1 &
echo "PID $!"
```

> **【9/3 に直した】「もう一巡してから抜けるので最後の 1 個も上がる」と書いていたが、外れていた。**
> **`pgrep` を見るのは `for` を回り終えた後なので、上げている 10 分の間に学習が終わると
> その回で `break` し、最後のチェックポイントが上がらない。**
> **1 個 9.35 GB あるので、この 10 分は普通に起きる。**
> **だから最後の 1 個は、[分岐 5](#分岐-5-hf-に-000500-しか上がっていない9-3-に実際に起きた) の
> 4 番のように手で上げて、上がったことを見てから Stop Server を押す。**

**`train_config.json` はこのループで一緒に上がる。**上げているのが
`pretrained_model` ディレクトリまるごとだからである。**中身の在り処はこうなる。**

| | 場所 |
|---|---|
| 機械の上 | `$OUTPUT_ROOT/pi05_camdrop2500/checkpoints/<歩数>/pretrained_model/train_config.json` |
| **HF** | **`<歩数>/train_config.json`**（`pretrained_model/` の階層は落ちる） |

**`pretrained_model/` の外には書かれない**（LeRobot v0.6.0 の
`configs/train.py` 280 行が `save_directory`＝`pretrained_model` に書く。
階層は `common/train_utils.py` 107〜121 行の図のとおり）。
**上の 2 行は 8/31 の `ladder2500` の実物で確かめた**——
HF の `pi05_trial1500` に `ladder2500/002500/train_config.json` が在る。

> **1 つだけ注意。**階層が落ちるので、**HF から LeRobot の `--resume` で直に続けることはできない**
> （`configs/train.py` 329 行は Hub 上の `<歩数>/pretrained_model/train_config.json` を探す）。
> **`--init-from` は落ちた形のままで読めるので、いつもの続け方には影響しない。**
> **そして `train_config.json` は設定しか持たない**——**loss の推移は B4b のログにしか無い。**

### B4b 学習ログを `$HOME` と HF へ写すループを始める（3 分）

**これが分かると何が決まるか: loss の推移が後から読めるか。**
**`$OUTPUT_ROOT`（`/opt/dlami/nvme/out`）はセッション終了で消える。**
**そしてログは、やり直しでは作れない唯一のものである**——
チェックポイントの `train_config.json` は設定を持つが、**loss の推移は持たない。**
**9/1 に `ladder2500` の「loss は 2,000 歩かけて −2.2%」を判定できたのは、
ログを機械が消える前に降ろしてあったからである**（`open_items.md` の `F61`）。

**置き先は 2 か所にする。**

| 置き先 | 残り方 | 落ちる筋 |
|---|---|---|
| **`$HOME/train_logs/`** | **セッションをまたいで残る**（自動保存の対象） | **`~/data`・`~/.cache`・`~/dataset` に置くと消える**——あの 3 つは一時領域への symlink である |
| **HF の `pi05_camdrop2500`** | **どの機械からでも降ろせる** | 認証が切れていると黙って失敗する（だから `$HOME` 側を先に写す） |

```bash
cat > ~/sync_log.sh <<'EOF'
#!/usr/bin/env bash
# 学習ログを $HOME と HF へ写す。1 分ごとに $HOME、10 分ごとに HF。
set -u
RUN=pi05_camdrop2500
SRC=/opt/dlami/nvme/out
DST=$HOME/train_logs          # symlink でない $HOME の実体に置く（~/data は消える）
REPO=siruku6/pi05_camdrop2500
mkdir -p "$DST"
copy() {
  for f in "$RUN.log" "$RUN.nohup"; do
    [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$f"
  done
}
push() {
  if hf upload "$REPO" "$DST/$RUN.log" "logs/$RUN.log" --repo-type model >/dev/null 2>&1; then
    echo "$(date -Iseconds) HF へ上げた"
  else
    echo "$(date -Iseconds) HF は失敗（~/train_logs への写しは済んでいる）"
  fi
}
i=0
while pgrep -f "train_pi05.py --run-name $RUN" >/dev/null; do
  copy
  i=$((i + 1))
  [ $((i % 10)) -eq 0 ] && push
  sleep 60
done
copy; push                      # 学習が終わったあと、最後に 1 回
echo "$(date -Iseconds) ログの写しを終える"
EOF
chmod +x ~/sync_log.sh
export PATH="$HOME/.local/bin:$PATH"
nohup ~/sync_log.sh > ~/sync_log.nohup 2>&1 &
echo "PID $!"
```

**1 分後に 2 行で確かめる。**

```bash
sleep 65; ls -la ~/train_logs/
tail -3 ~/sync_log.nohup
```

**`~/train_logs/` に 2 つ並べば、最悪の全損は防げている。**
**10 分たっても `HF へ上げた` が出ないときは、`hf auth whoami` を打ち直す**
（`~/.cache` が消える領域なので、トークンはセッションごとに入れ直している）。

### B5 最初の 500 ステップが HF に載るまで見届ける（40 分）

**ここまでが在席である。**500 ステップは **6.85 秒/step なら 57 分後**に出る（8/31 の実測）。
**待ちきれないときは、60 歩だけ回した `pretrained_model` を先に 1 回上げて、
上げの道が通ることだけ確かめてから消す。**

```bash
watch -n 60 'ls -la /opt/dlami/nvme/out/pi05_camdrop2500/checkpoints/ 2>/dev/null; \
  cat ~/upload_ckpt.log 2>/dev/null; \
  tail -2 /opt/dlami/nvme/out/pi05_camdrop2500.log'
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

LOG = "/opt/dlami/nvme/out/pi05_camdrop2500.log"
alive = lambda: bool(subprocess.run(
    ["pgrep", "-f", "train_pi05.py --run-name pi05_camdrop2500"],
    capture_output=True).stdout)

while alive():
    tail = subprocess.run(["tail", "-n", "1", LOG], capture_output=True, text=True).stdout
    print(time.strftime("%H:%M"), tail.strip()[-160:], flush=True)
    time.sleep(300)

print("学習が終わった。上げ残しを待つ", flush=True)
while subprocess.run(["pgrep", "-f", "upload_ckpt.sh"], capture_output=True).stdout:
    time.sleep(60)
print(open(os.path.expanduser("~/upload_ckpt.log")).read(), flush=True)

# JUPYTERHUB_API の GET が 200 だったときだけ、自分でサーバを止める
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

## C 放置する

**やることは無い。**下が同時に走っている。

| 走っているもの | 終わる時刻 | 落ちたときに残るもの |
|---|---|---|
| 学習（2,500 ステップ・6.85 秒/step なら **4 時間 48 分**） | 起動 ＋ 約 5 時間半 | `checkpoints/` に出来たところまで |
| 上げるループ | 20:20 | **HF に上がったところまで**（これだけが本当に残る） |
| 監視セル | 20:20 → Stop Server | — |

**最悪でも T0 ＋ 12 時間（例 01:00）に自動停止する。**消費が 12 時間を超えることはない。

---

## 翌朝 15 分

```bash
# このマシンから。Blackwell に入り直す必要は無い
python3 -c "
from huggingface_hub import HfApi
for f in HfApi().repo_info('siruku6/pi05_camdrop2500', repo_type='model').siblings:
    print(f.rfilename)"
```

**ログも一緒に降ろす。**`logs/` に上がっている（B4b）。

```bash
hf download siruku6/pi05_camdrop2500 --include 'logs/*' \
    --local-dir work/train/pi05_camdrop2500
```

**上がっていなければ、`$HOME` のほうに残っている**（`~/train_logs/`）。
**次のセッションで開いたときに、まずそれを降ろすこと**——`$HOME` は残るが、
**同じ名前で次の走りを回すと上書きされる。**

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

## 場面カメラを確率で潰して 1 本回す（**9/2 の夜に足した**）

**これが `open_items.md` の `F50`①・賭け A である。**狙いは Track2 の
`Camera Viewpoints`（1,599 件・15.9%）——**机を写す固定カメラの視点をずらされる摂動**で、
手首のカメラは腕について動くのでずれない。**学習中に場面カメラを確率で真っ黒にすると、
手首のカメラだけで進める癖が付く。**

**設定では書けないので、コードを足した**（`solution/train/experiments/camera_dropout/`）。
同梱の LeRobot は画像の変換を**カメラの列ごとに呼ぶが、どの列かを変換関数に渡さない**
（`lerobot/datasets/dataset_reader.py` の `get_item`）。

| | |
|---|---|
| 新しい引数 | **`--scene-camera-dropout P`**（既定 0.0＝何もしない）・`--scene-camera-key KEY`（既定 `observation.images.front`） |
| 潰し方 | **その列を 0 で埋める。**π0.5 は画像を `img*2-1` で [-1,1] に写し、足りないカメラを `-1` で埋めるので、**0 を渡すと正規化後にその埋め草と一致する**（`modeling_pi05.py` 1207〜1211 行） |
| **マスクは 1 のまま** | **π0.5 の「カメラが無い」（mask 0）は真似しない。****採点では場面カメラは必ず在って、視点だけがずれる**——つまり mask は 1 である。学習で mask 0 を使うと「0 なら無視」を覚えるが、**採点でそのマスクが立たないので発火しない** |
| 手首のカメラ | **触らない**（試験で確かめている） |
| lerobot-train への引数 | **1 つも増えない。**潰すのは DataLoader の中である |

### なぜ 1 変数なのか（打つのは [B2](#b2-学習を-nohup-で始める10-分) である）

**コマンドは B2 に 1 つだけ置いてある。**ここにも書くと、片方が必ず古くなる。

**`--scene-camera-dropout 0.5` 以外は、8.0.0 を作った学習と 1 文字も違わない**
（`work/train/pi05_ladder2500.log` の 8 行目の `lerobot-train` の行と突き合わせた。
batch 64・学習率 5e-6・種 42・`save_freq` 500・`gradient_checkpointing=true`・bfloat16・
`empty_cameras=1`・`n_action_steps=10`・データ `lerobot/libero_plus` rev `f3f49f42`・
カメラ拡張は 8 種から最大 4 つ）。**だから 1 変数である。**

> **拡張と、潰すことは別のものである。**拡張（`--camera-viewpoint-augment`）は
> **両方のカメラ**に色や歪みをかけ、潰すほうは**場面カメラだけ**を消す。
> **だから拡張は 8.0.0 と揃えて付けたままにし、外さない**——外すと変数が 2 つになる。

**確率 0.5 を選んだ理由と、失うもの。**modality dropout の慣例値で、
**手首だけで進む癖を付けるには半分は要る**という見立てである。
**失うのは、場面カメラを使った学習量が半分になること**——Track1 のように視点がずれない
トラックでは下がりうる。**0.3 なら安全側だが、効きが 1 本ぶんの幅（Track1 0.061・
Track2 0.039）に埋もれる恐れがある。**引数なので next の 1 本で振り直せる。

### 始めた直後に確かめる 2 行（**黙って効かないのを防ぐ**）

```bash
grep -m1 num_learnable_params $OUTPUT_ROOT/pi05_camdrop2500.log   # 4143404816
grep -m2 "\[camera_dropout\]" $OUTPUT_ROOT/pi05_camdrop2500.log
```

**2 行目が 1 つも出なければ、潰す処理は動いていない**——その場で止めて直すこと。
出るのは読み込みの worker ごとに 1 行で、こういう形である::

    [camera_dropout] worker0: observation.images.front を確率 0.5 で潰す（画像の列は [...]）

**列の名前が合わなければ、学習は始まらずにその場で止まる**（在る列の名前と直し方を出す）。
**20,000 コマごとに実測の割合も出る**ので、0.5 に寄っていることを後から確かめられる::

    [camera_dropout] 20000 コマ中 10043 コマを潰した（実測 0.502・指定 0.5）

### 手元で試験を走らせる（1 行・6 秒）

```bash
bash solution/tools/train/run_train_tests.sh
```

**常駐コンテナ `quant_train` の本物の torch 2.11 と lerobot 0.6.0 で 11 件走る**
（`tests/test_camera_dropout.py`）。**この機械の venv には torch も lerobot も
入っていないので、`pytest tests/` を直接打つと skip になる。**

---

## 配布の 111 タスクのデータで 1 本回す（**9/4 に足した。`F48`**）

**これが `open_items.md` の `F48` である。****狙いは Track3 である。**
Track3 の 4 タスクは**4 つとも `libero_90` の場面から作られていて、
そこに出る棚・トレイ・黄色い本は、いまの学習データ（`lerobot/libero_plus` の 40 タスク）に
1 件も無い**（[../knowledge/17_track3_picks_the_wrong_object.md](../knowledge/17_track3_picks_the_wrong_object.md)）。
**配布 tar `libero_combined_20hz` には `libero_90` の 90 件が入っている。**

| | いまのデータ | **配布 tar** |
|---|---|---|
| 名前 | `lerobot/libero_plus` | **`libero_combined_20hz`**（HF には無い。**配布物の tar だけ**） |
| エピソード | 14,347 | **19,533** |
| コマ | 2,238,036 | **3,028,708** |
| タスク | 40 | **111** |
| 版・fps・カメラ | v3.0・20 Hz・front と wrist | **同じ**（変換は要らない） |

**変えるのはデータだけである。****比べる相手は 5.0.1（Track1 0.492・Track2 0.233）**——
**こちらが持っている中で Track1 が最高の重みである。**
だから**出発点は `rp3000`、歩数は 1,500** にし、batch・学習率・カメラ拡張も 5.0.1 に揃える。

> **【9/4 に決めた】`trial1500/001500` から足す案はやめた。**
> **`rp3000` を 0 歩として点を並べ直すと、Track1 は +1,500 歩が頂点で、その先は下りである**
> （+0 が 0.370・**+1,500 が 0.492**・+3,000 が 0.478・+4,000 が 0.475）。
> **`trial1500/001500` から足すと、下っている区間の上に積むことになり、
> データを替えた効果と混ざる。**
> **あわせて `--init-from` は学習率の刻みを引き継がない**ので、続きを足すたびに
> 頂点 5e-6 からの立ち上がりを踏み直す。**`rp3000` から始めれば、その山が 1 つ減る。**
> **1,500 歩なら 2 時間 51 分で、2,500 歩の案より約 2 時間短い。**
> 読みは [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md)
> の「【9/4】出発点そのもの（`rp3000`）を 0 歩として並べ直した」。

### 打つ前に —— 「毎セッション打つ 3 つ」に、データの展開が 1 つ増える

**コードの取り直しは必須である**（**9/4 に `--dataset-repo` を足したので、
取り直さないと「知らない引数」で止まる**）。

```sh
. ~/env_pi05.sh
hf download siruku6/pi05_code --local-dir ~/pi05_code
```

#### 出発点の重みは `rp3000` である（**いつもの `trial1500/001500` ではない**）

```sh
hf download siruku6/pi05_full_runpod --local-dir /opt/dlami/nvme/weights/rp3000
du -sh /opt/dlami/nvme/weights/rp3000
ls /opt/dlami/nvme/weights/rp3000/vlm | wc -l    # 4
```

**`vlm/` が同梱されているので、あとで zip を組むときの `cp -r` は要らない。**

#### 配布 tar を展開する（**このセッションで 1 回。消える領域に置く**）

```sh
. ~/env_pi05.sh
ls -la ~/dataset/lerobot/                       # tar の名前と大きさを見る
tar -tf ~/dataset/lerobot/libero_combined_20hz.tar | head -3   # 中の階層を見る
```

**`libero_combined_20hz/meta/...` のように 1 段でくるまれていれば `--strip-components=1`
を付ける。`meta/...` から始まっていれば付けない。**上の 3 行で決めること。

```sh
DATA=/opt/dlami/nvme/data/libero_combined_20hz
mkdir -p "$DATA"
time tar -xf ~/dataset/lerobot/libero_combined_20hz.tar -C "$DATA" --strip-components=1
ls "$DATA"                                      # meta / data / videos が並ぶこと
python3 -c "
import json; i = json.load(open('$DATA/meta/info.json'))
print(i['total_episodes'], i['total_frames'], i['total_tasks'], i['fps'])"
# 19533 3028708 111 20 と出ること
```

**この 4 つの数が違ったら、そこで止めること。**別の tar である。

### 打つ 1 行（5.0.1 との 1 変数）

```bash
. ~/env_pi05.sh
cd "$CODE_DIR"
DATA=/opt/dlami/nvme/data/libero_combined_20hz

NW=$(python -c "import os;print(min(os.cpu_count()//2, 16))")
nohup python train_pi05.py --run-name pi05_all111 \
    --init-from /opt/dlami/nvme/weights/rp3000 \
    --dataset-repo lerobot/libero_combined_20hz --dataset-root "$DATA" \
    --steps 1500 --batch-size 64 --num-workers "$NW" \
    --learning-rate 5e-6 --save-freq 500 \
    --freeze-vision-encoder false --train-expert-only false \
    --camera-viewpoint-augment \
    > $OUTPUT_ROOT/pi05_all111.nohup 2>&1 &
echo "PID $!"
```

**`--dataset-repo` と `--dataset-root` の 2 つ以外は、5.0.1 を作った学習と 1 文字も違わない**
（`rp3000` から 1,500 歩・batch 64・学習率 5e-6・カメラ拡張あり・画像側を開ける）。

> **`lerobot/libero_combined_20hz` は Hugging Face に無いが、それで構わない。**
> **`--dataset-root` を渡すと、LeRobot は Hub を 1 度も見ない**——`datasets/dataset_metadata.py` の `LeRobotDatasetMetadata.__init__` は
> **渡された root をそのまま使い、そこにメタデータが無かったときだけ Hub へ落ちる**（`except FileNotFoundError` の枝）。
> **だから名前は札にすぎず、実体は展開したディレクトリである。**
> **版（`--dataset.revision`）は付けない**——`train_pi05.py` が、既定のデータのときだけ既定の版を付けるようにしてある。

> **場面カメラを潰す指定（`--scene-camera-dropout`）は付けない。**
> **9/4 に 12.0.0 が 8.0.0 に両トラックとも負けたので、その道は打ち切った。**

### 始めた直後に確かめる 3 行（**ここが要である**）

```bash
grep -m1 num_learnable_params $OUTPUT_ROOT/pi05_all111.log
grep -m1 'dataset.num_episodes' $OUTPUT_ROOT/pi05_all111.log
grep -m1 '学習開始' $OUTPUT_ROOT/pi05_all111.log
```

| 見るもの | 期待値 | 違ったら |
|---|---|---|
| `num_learnable_params` | **4143404816** | 693422112 ならその場で止める（行動側しか学習していない） |
| **`dataset.num_episodes`** | **19533** | **14347 なら乗り換えていない。****その場で止める**——気付かないと 2 時間 51 分を古いデータで使う |
| `学習開始（…）` の行の末尾 | **`データ lerobot/libero_combined_20hz（/opt/dlami/nvme/data/…）`** | データの名前と置き場が出る（9/4 に足した） |

**`--dataset-root` を忘れたときは、`train_pi05.py` がデータを読む前に止める**
（9/4 に検査を足した。**止めないと、番号選びが Hub の 14,347 本を読み、
足した 5,186 本が 1 本も選ばれないまま学習が走る**）。

### この 1 本で気にすること

| 気にすること | 中身 |
|---|---|
| **Track1・Track2 が下がりうる** | **1 タスクあたりの学習量が 40 → 111 に薄まる。**だから**3 トラックとも出して確かめる**——8.0.0 と 12.0.0 が基準になるので、下がればすぐ分かる |
| 正規化の統計が変わる | データが変われば `MEAN_STD` の統計も変わる。**チェックポイントの中に一緒に入るので、zip を組み直せばそれで足りる** |
| 1 エポックの意味 | コマ数が 2,238,036 → 3,028,708 なので、同じ 1,500 歩でも**見る回数の割合は 3/4 に減る**（1,500 歩 × batch 64 ＝ 96,000 コマ＝ 0.032 エポック） |
| 展開先が消える | `/opt/dlami/nvme` はセッション終了で消える。**毎セッション展開し直す**（`~/dataset` の tar は消えない） |

### 手元で試験を走らせる（1 行）

```bash
bash solution/tools/train/run_train_tests.sh
```

**常駐コンテナ `quant_train` の本物の torch 2.11 と lerobot 0.6.0 で 17 件走る**
（`tests/test_camera_dropout.py` 11 件・`tests/test_dataset_choice.py` 6 件）。
**乗り換えの 6 件は、①渡さなければ 1 文字も変わらない ②別のデータに
`lerobot/libero_plus` の版を付けない ③`--dataset-root` を忘れたら止まる、を見ている。**

---

## 2 回目以降のセッション —— 60 時間は「12 時間 × 5 回」である

**これが分かると何が決まるか: 1 本の長い学習は取れない。梯子を何段に割るか。**

**起動から 12 時間で自動的に止まり、本選 1 では延長ボタンが使えない**
（本選資料 v1.1 p15）。**60 時間は連続した 60 時間ではなく、最大 5 回の
12 時間である。**セッションをまたぐたびに、次の 3 つを払う。

| 払うもの | 1 回目 | 2 回目以降 | なぜ |
|---|---|---|---|
| 環境（venv 約 8.2 GB） | **A3 の実測（60 分見込み）** | **1 分** | `$HOME` にあり自動保存される。スクリプトは有無を見て作り直さない |
| 学習データ | `~/dataset` の tar を `~/data` へ展開 | **同じだけ払う** | `~/data` の実体は消える領域である（[../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md) 3 節）。**ただし `~/dataset` は消えないので、HF から落とし直す必要は無い** |
| **続きの重み（11 GB）** | 不要（`001500` を HF から取る） | **HF から降ろす分** | `/opt/dlami/nvme` の出力は消える |

**1 セッションの正味の学習時間は、12 時間からこの 3 つと「上げ切る時間」を引いた残りである。**
**残りの出し方は下の A7 にある。**

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

**VRAM 97,887 MiB に batch 64 が 49.40 GB で載ることは 8/31 に実測済みである**（[../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md) 2 節）。
そのときは `--batch-size 32 --steps 3000` にして**サンプル数を 96,000 で揃える。**
ただし**比較の変数が 2 つになる**ので、報告に「batch が違う」と 1 行残す。
**`--gradient-checkpointing` は既定で true である。外さないこと。**

### 分岐 2 HF の空きが 27 GB に足りない

> **【9/3 に測って分かった】枠に効くのは非公開リポジトリだけである。公開のぶんは数えない。**
> **9/3 に `pi05_camdrop2500` へ 5 個（46.77 GB）が全部上がった。**
> **全リポジトリを足す数え方だと合計 113.8 GB で、100 GB を超えていたのに通っている。**
> **非公開だけ足すと 93.64 GB** で枠の内側である（`pi05_trial1500` 37.49 ＋
> `parc2026-submission` 9.38 ＋ `pi05_camdrop2500` 46.77）。
> **公開の 20.16 GB**（`pi05_stage1_24000` 9.37 ＋ `pi05_full_runpod` 9.37 ＋ 残り）
> **を引き算に入れていたので、8/29 以来ずっと空きを 20 GB 少なく見ていた。**
> **いまの空きは 6.36 GB である。**
>
> **これで直るもの 2 つ。**①**`pi05_stage1_24000` を消しても 1 バイトも戻らない**（公開である）。
> ②**「500 歩ごとに全部上げると入らない」は今回は外れだった**——5 個入った。
> **次に足りなくなったら、消す相手は非公開の 3 つの中から選ぶ。**

**非公開の枠は 100 GB で、9/3 の時点で非公開だけで 93.64 GB 使っている（空き 6.36 GB）。**
**RunPod 側（依頼 008）も同じ枠へ上げているので、夕方には減っている可能性がある。**

| 手 | 戻る量 | 決めるのは |
|---|---|---|
| ~~`siruku6/pi05_stage1_24000` を消す~~ | **0 GB。公開なので枠を食っていない**（9/3 に分かった） | — |
| `siruku6/parc2026-submission` の古い版を消して履歴を潰す（`super_squash_history`） | 8/29 に 76.4 → 57.7 GB になった実績あり | 親の判断 |
| **上げるのを `001500` だけにする** | 要るのが 8.8 GB で済む | **その場で決めてよい。**比較の要は `001500` である |

**どれも取れないなら `~/data` へ置き、翌朝このマシンへ引き取る。**

### 分岐 3 待機による自動停止で学習が死んだ

**B6 の監視セルを実行し忘れると起きる。**気づくのは翌朝である。
**HF に上がったところまでは残っている。**翌日は `--init-from` に
上がっている中でいちばん大きいステップ数を渡し、**残りのステップ数だけ回す。**
**学習率は 5e-6 のまま**（刻みは引き継がれないので、続きではなく新しい走りになる）。

### 分岐 4 打ち間違えて回し始めた —— 途中で止めて打ち直す

**9/3 に実際に起きた**（`--camera-viewpoint-augment` を付けずに回し始めた）。
**順番を守ること。**下の 1 が最後になると、**セッションそのものが落ちる。**

#### 1. 監視セル（B6）を先に止める（**これを飛ばさない**）

**Notebook の停止（■ / Interrupt Kernel）で止める。**

**理由。**B6 のセルは `train_pi05.py` が消えたのを見ると、
上げ残しを待ってから**自分でサーバへ `DELETE` を投げる。**
**学習を先に殺すと、その場でセッションが終わって 12 時間の枠を返せなくなる。**

#### 2. 上げるループを止め、HF に何が上がったかを見る

```bash
pkill -f upload_ckpt.sh
cat ~/upload_ckpt.log
```

**上がっていたら、その分は消す。**打ち間違えた走りの重みなので、
**残すと `pi05_camdrop2500` の中で正しい走りのものと区別できなくなる。**

```bash
python3 - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for f in api.repo_info("siruku6/pi05_camdrop2500", repo_type="model").siblings:
    print(f.rfilename)
# 消すときだけ下を有効にする（パスは上の出力から選ぶ）
# api.delete_folder("000500", repo_id="siruku6/pi05_camdrop2500", repo_type="model")
PY
```

#### 3. 子の側から落とす（**親だけ殺すと GPU が空かない**）

**`train_pi05.py` は学習の本体を別のプロセスとして起こす**
（`subprocess.Popen`。`train_pi05.py` の 467 行）。
**だから `pkill -f train_pi05.py` だけでは、子が VRAM を握ったまま残る。**

> **【9/3 に踏んだ】子の名前は走りによって違う。**
> **`--scene-camera-dropout` を付けると、子は `lerobot-train` ではなく
> `lerobot_train_with_camera_dropout.py` になる**（`experiments/camera_dropout/__init__.py`
> 111 行の `command_head` が `["python", <そのパス>]` を返す。量子化なら
> `lerobot_train_with_quant.py`、幾何なら `lerobot_train_with_geometry.py`）。
> **`pgrep -af 'train_pi05.py|lerobot-train'` は下線の側に当たらないので、
> 親だけ見えて「落ちた」と誤読する**——**9/3 にこれで VRAM が 60,871 MiB 残った。**

**当たっているかを名前で判定しない。GPU に聞く。**

```bash
# 誰が VRAM を握っているかを PID で出す（これが唯一の答えである）
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pgrep -af 'train_pi05|lerobot[-_]train'      # 下線もハイフンも拾う形
```

```bash
pkill -f 'lerobot[-_]train'                   # 子（ラッパーでも lerobot-train でも当たる）
pkill -f train_pi05.py                        # 親
sleep 5
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=memory.used --format=csv   # 数百 MiB まで戻ること
```

**それでも残っていたら、上の PID を直接落とす**（`kill -9 <PID>`）。
**`nvidia-smi` に PID が 1 つも出ないのに VRAM が減らないときだけ、数十秒待つ。**

#### 4. 出力とログを片付ける（**片付けないと打ち直せない／ログが混ざる**）

```bash
. ~/env_pi05.sh
rm -rf $OUTPUT_ROOT/pi05_camdrop2500
mv $OUTPUT_ROOT/pi05_camdrop2500.log $OUTPUT_ROOT/pi05_camdrop2500_noaug_aborted.log
```

| 片付けるもの | 片付けないとどうなるか |
|---|---|
| **出力先のディレクトリ** | **打ち直した瞬間に終わる**——`train_pi05.py` の 497〜501 行が `出力先が既にある` で `SystemExit` する（LeRobot v0.6.0 も同じ理由で落ちる） |
| **ログ** | **ログは追記で開かれる**（`log_file = log_path.open("ab")`・511 行）。**混ざると B3 の `grep -m1 num_learnable_params` が打ち間違えた走りの行を拾う** |

**消さずに残す道もある。**`--run-name` を変えれば両方そのまま残るが、
**HF の上げ先も B6 のセルの `LOG` も名前を持っているので、3 か所直すことになる。**
**打ち間違えた走りに用は無いので、消すほうが速い。**

#### 5. 使った時間は戻らない。**歩数を決め直す**

**60 時間からも、このセッションの 12 時間からも、回した分は引かれている。**
**A7 をもう一度流して `--steps` を決め直すこと**——
**2,500 歩は 4 時間 48 分あるので、締切に入らないことがある。**

そのうえで [B2](#b2-学習を-nohup-で始める10-分) を打ち、
**B3 で `[camera_dropout]` の行が出ることまで確かめる。**

### 分岐 5 HF に `000500` しか上がっていない（**9/3 に実際に起きた**）

**結論を先に。学習は正常で、保存も 4 個できている。壊れているのは上げるループだけである。**
**そして 38 分後に、B6 の監視セルが自分でサーバを止めて、上がっていない 4 個を消す。**

#### いまどうなっているか（9/3 10:20 JST に、この機械から HF を読んで確かめた）

| 見たもの | 実際 |
|---|---|
| 学習 | **2,160 / 2,500 歩・6.84 秒/step・loss 0.250。**正常に走っている（`01:03:18` UTC の行） |
| **`--save-freq 500` は効いている** | ログに `Checkpoint policy after step` が **500・1000・1500・2000 の 4 回**出ている。**機械の上には 4 個ある** |
| **`002500` が出来る時刻** | **`01:42` UTC ＝ 10:42 JST**（残り 340 歩 × 6.84 秒） |
| **HF に在るもの** | **`000500/`（9.35 GB）と `last/`（9.35 GB）と `logs/pi05_camdrop2500.log` だけ。**`001000` 以降が 1 個も無い |
| **`last/` は `000500` の複製である** | 中身は同じ重みで、**空きを 9.35 GB 食っているだけである**（理由は下の表） |
| ログを写すループ（B4b） | **生きている。**HF 上のログが `01:03` UTC まで伸びている。**つまり HF の認証は切れていない** |
| HF の空き | **14.26 GB**（全リポジトリで 85.74 GB / 100 GB）。**チェックポイント 1 個が 9.35 GB なので、いまは 1 個しか入らない** |

**`last/` が上がった理由。**B4 の `for d in "$OUT"/checkpoints/*/pretrained_model` は、
**LeRobot が保存のたびに貼り直す `checkpoints/last` という symlink にも当たる**
（`common/train_utils.py` 85〜90 行の `update_last_checkpoint`）。
`basename` が `last` を返すので、**HF に `last/` という歩数のふりをした複製が 1 個できる。**

#### 上げるループは死んでいなかった。**生きたまま 3 時間、何も試していなかった**

**9/3 10:2x JST に Blackwell で見た出力がこれである。**

| 見たもの | 出た値 | 読み方 |
|---|---|---|
| `pgrep -af upload_ckpt.sh` | **`14925 bash /home/sirukufarios/upload_ckpt.sh`** | **生きている。**死んだのでも `break` したのでもない（`break` すれば `上げるループを終える` が `~/upload_ckpt.log` に出る） |
| `~/upload_ckpt.log` | `21:54:18 上げた 000500` / **`21:59:03 上げた last`** の 2 行だけ | **`last` の複製もここで上がった。**以降 3 時間、**1 行も増えていない** |
| `~/upload_ckpt.nohup` | **`Found 7 files to upload` が 2 回だけ。**2 回目の `✓ Uploaded` で終わり、**例外も失敗も出ていない** | **`hf upload` は 3 回目を始めてすらいない。**始めれば `Found 7 files to upload` が出る。**つまり `hf` の失敗ではない**——ループが `hf` を呼ぶ手前で止まっている |
| `checkpoints/` の中身 | `000500`（リンク数 3）・`001000`・`001500`・`002000`（リンク数 4）・`last -> 002000` | **4 個そろっている。**`000500` だけリンク数が 1 少ないのは、**`last` を上げた回の `rm -rf` が `last` の先の `000500/training_state` を消したからである**（狙いどおり） |

**なぜ `hf` の手前で止まったのかは、まだ分かっていない。**
**上のとおり `hf upload` の失敗でも、プロセスの死でも、`break` でもない**ので、
残るのは `du -sb` か `sleep` か `pgrep` のどれかで止まっている筋だが、**跡が無い。**
**跡が無いこと自体が script の落ち度である**——だから次の 3 つを B4 の script に足した。

| 足したもの | なぜ |
|---|---|
| **1 巡ごとに `見回り` の 1 行をログへ書く** | **止まった位置が分かる。**今回はこれが無かったので、3 時間どこで止まっていたかを永久に言えない |
| **`hf upload` の失敗もログへ書く**（`else` の側） | 成功しか書かない作りだったので、**失敗と「試していない」を区別できなかった** |
| **`timeout 3600` を `hf upload` に掛ける** | 1 個 9.35 GB なので、正常なら 15 分で終わる。**1 時間を超えたら諦めて次の巡へ回す** |

#### 私が書いていて、外れていたこと

| 私の主張（B4 の節） | 実際 | なぜ間違えたか |
|---|---|---|
| **「学習が終わってからもう一巡してから抜けるので、最後のチェックポイントも上がる」** | **抜けるのは `for` を回り終えた後に `pgrep` を見るからで、上げている 10 分の間に学習が終わると、その回で `break` する。****最後の 1 個が上がらない** | 「もう一巡」を、上げに時間がかからない前提で書いた。**1 個 9.35 GB である** |
| 上げるループは放っておけば全部拾う | **`000500` の次から 1 個も上がっていない** | 上げの失敗が `~/upload_ckpt.log` に出ない作りにしたので、失敗を見張る手が無かった |

#### 決めてほしいこと（番号で答えられる）

| # | 決めること | 選ぶと何が起きるか | こちらの推し |
|---|---|---|---|
| **1** | ~~**`last/` を HF から消してよいか**~~ | **済（9/3 10:2x JST・ユーザが「消して良い」と言った）。**`delete_folder` で消し、**リポジトリは 18.71 → 9.35 GB、空きは 14.26 → 23.6 GB になった**（歩数のフォルダを数える手で測った値）。**`002000` と `002500` の 2 個が入る。****`super_squash_history` は打てなかった**（この機械の権限で弾かれた）。**打てていないので、HF 側の実際の課金が履歴も数えているなら 9.35 GB は戻っていない**——ただし**8/29 以来この手で数えた値で運用できているので、まず 2 個入る前提で進む** | — |
| **2** | ~~**`002000` も上げるか**~~ **（済。それどころか 5 個とも上がった**——`01:58:00` UTC までに `000500`・`001000`・`001500`・`002000`・`002500` が HF に在る。**枠に効くのは非公開だけだったので、捨てる必要が無かった。**[分岐 2](#分岐-2-hf-の空きが-27-gb-に足りない) の冒頭に書いた） | 上げると 2 個で 18.7 GB を使い、**空きが 4.9 GB になる。**上げないと**この走りの重みは `000500` と `002500` の 2 点だけになる。**`001000`・`001500` は**どちらの案でも捨てる**——採点は 1 トラック 1 日 1 回しか測れないので、細かい梯子は使えない | **上げる**（`002500` が壊れていたときの控えになる） |
| **3** | **手で上げるか、ループに任せるか** | **案A（手で上げる・推し）**: 確実。**10:42 JST まで在席が要る。**②の待ち時間 15 分 ＋ ③の 15 分。**案B（ループを直して再開）**: 在席が要らない。**ただし上の「外れていたこと」のとおり、`002500` を上げている最中に学習が終わると `break` して `002500` を落とす** | **案A** |

#### やってほしい作業（**案A。番号の順に。1 を飛ばさない**）

**実行場所はすべて Blackwell の Notebook である。**
**この 4 つを私は代われない**——Blackwell には私が入れない。

**1. B6 の監視セル（`while alive():` のセル）を ■ / Interrupt Kernel で止める。所要 10 秒。**
**いま止める。**あのセルは `train_pi05.py` が消えるのを見ると、**上げるループが居ないので
すぐサーバへ `DELETE` を投げる。****`/opt/dlami/nvme` は消えるので、上がっていない
`001000`・`001500`・`002000`・`002500` が同時に消える。**
**止めた後は「1 時間セルが動かないと自動停止」だけが残る**が、下の 2〜4 はセルで打つので当たらない。

**止まったかどうかは、セルの出力で見る。**
**B6 のセルは 300 秒ごとに `HH:MM <ログの末尾>` を 1 行足す作りである。**

| 見るもの | 止まっている | 動いている |
|---|---|---|
| セルの出力の最後の時刻 | **6 分以上前のまま増えない** | 5 分ごとに増える |
| セルの出力の末尾 | **`KeyboardInterrupt` が出ている** | 何も出ていない |
| セルの左の印 | `[ ]` か番号 | **`[*]`** |

**見分けが付かないときは、止まっていても困らない形にする（1 行・10 秒）。**
**B6 は「上げるループが居なくなったら `DELETE` を投げる」作りなので、
`pgrep -f upload_ckpt.sh` に当たるだけのプロセスを 1 つ置けば、B6 は待ち続けて何もしない。**

```bash
nohup sh -c 'sleep 36000  # guard for upload_ckpt.sh' >/dev/null 2>&1 &
pgrep -af upload_ckpt.sh          # 1 行出れば見張りが立った
```

> **これを置いたら、サーバは自分では止まらない。**
> **`002500` を上げ切ったら、下の 5 番で自分で Stop Server を押すこと**——
> 押し忘れると 12 時間の打ち切りまで 60 時間の枠が減り続ける。
> **カーネルの Restart では代えられない**——`nohup` の学習まで一緒に落ちる作りの環境もあり、
> **`002500` を書き終える前に落ちたら 4 時間 48 分が消える。**

**2. 何が起きたかを見て、残っているループを止める（1 分）。**

```bash
. ~/env_pi05.sh
pgrep -af upload_ckpt.sh
cat ~/upload_ckpt.log
tail -20 ~/upload_ckpt.nohup
ls -la $OUTPUT_ROOT/pi05_camdrop2500/checkpoints/
pkill -f upload_ckpt.sh     # 居ても居なくても打つ。手で上げるものと二重にならないように
```

**この 5 行の出力をそのまま貼って報告してほしい。**`upload_ckpt.nohup` の中身が、
次のセッションで同じことを起こさないための唯一の材料である。
**`ls` に `000500`〜`002000` の 4 つと `last` が並ぶことも見ておくこと。**

**3. `002000` を手で上げる（15 分ほど。②で「上げる」を選んだときだけ）。**

```bash
. ~/env_pi05.sh
export PATH="$HOME/.local/bin:$PATH"
hf upload siruku6/pi05_camdrop2500 \
    $OUTPUT_ROOT/pi05_camdrop2500/checkpoints/001500/pretrained_model 001500 \
    --repo-type model
```

**4. 10:42 JST を過ぎたら `002500` を上げる（15 分ほど）。****これが本命である。**

```bash
. ~/env_pi05.sh
export PATH="$HOME/.local/bin:$PATH"
grep -c "終了（終了コード 0）" $OUTPUT_ROOT/pi05_camdrop2500.log   # 1 が出れば学習は終わっている
hf upload siruku6/pi05_camdrop2500 \
    $OUTPUT_ROOT/pi05_camdrop2500/checkpoints/002500/pretrained_model 002500 \
    --repo-type model
```

**5. 上がったことを確かめてから、自分で Stop Server を押す（1 分）。**
**押した瞬間に 60 時間の消費が止まる。**

```bash
python3 -c "
from huggingface_hub import HfApi
for f in HfApi().repo_info('siruku6/pi05_camdrop2500', repo_type='model').siblings:
    print(f.rfilename)" | grep model.safetensors
```

**`002500/model.safetensors` が出ていれば終わりである。**
**出ていなければ止めない**——`/opt/dlami/nvme` は止めた瞬間に消える。

#### 6. 上げが `0.00B/s` で止まったとき（**9/3 10:3x JST に踏んだ。これが 3 時間の正体でもある**）

**見えた形。**`hf upload` が `Found 7 files` → `Preparing 7/7 ✓` → `Uploading 3/3 files ✓` まで進み、
**`Committing 0 / 7` で止まる。****その手前の network の数値が 6.51GB → 2.21GB と減っていき、
最後に `0.00B/s` になる。**この機械から HF を読むと、**コミットは 1 つも増えていない。**

**【9/3 に測った】この形は詰まりではない。待てば通る。**
**`002000` は `Committing 0 / 7` で `0.00B/s` のまま 7 分止まり、
`01:30:32` UTC にコミットが通った**（この機械から HF を読んで確かめた。9.35 GB）。
**だから `0.00B/s` を見ても Ctrl-C を押さないこと。**
**送り終えた後のコミットに 5〜8 分かかり、その間 network は 0 で、進捗も動かない。**

> **9/2 の 3 時間の停止は、これでは説明できない。**あちらは `Found 7 files to upload` が
> **3 回目に出ていない**ので、コミット待ちですらなかった。**原因は依然として分かっていない。**

**7 分待っても通らないときは、下を順に打つ。**
**(b) を先にやること**——**ネットワークを 1 バイトも使わずに、消えない場所へ移せる。**

| # | やること | これで何が変わるか |
|---|---|---|
| **(a)** | 止まっている `hf upload` を **■ / Ctrl-C** で抜ける | 送った分は HF 側に残るので、**打ち直しは最初からにならない** |
| **(b)** | **`$HOME` に控えを取る**（下の 1 つ目のブロック） | **HF が今日 1 バイトも通らなくても重みが残る。**`$HOME` はセッションをまたいで残るので、**次のセッションから上げ直せる** |
| **(c)** | **Xet を切って上げ直す**（2 つ目） | 上げの経路が `hf_xet` から昔の LFS に替わる。**`0.00B/s` が Xet 側の詰まりなら通る** |
| **(d)** | **多コミットで再開できる道具に替える**（3 つ目） | `hf upload-large-folder` は**ファイルごとに小さくコミットして、打ち直すと続きから行く。**1 個 9.35 GB を 1 コミットで通そうとするのをやめる |

```bash
# (b) $HOME に控えを取る。ネットワークを使わない。2 分ほど
df -h ~                     # 空きを見る。1 個で 9.4 GB 要る
mkdir -p ~/ckpt_keep
time cp -a $OUTPUT_ROOT/pi05_camdrop2500/checkpoints/002000/pretrained_model ~/ckpt_keep/002000
du -sh ~/ckpt_keep/002000   # 8.7 GiB 前後
```

```bash
# (c) Xet を切って上げ直す。控え（~/ckpt_keep）から上げれば、送り元は 1 つで済む
export PATH="$HOME/.local/bin:$PATH"
HF_HUB_DISABLE_XET=1 hf upload siruku6/pi05_camdrop2500 \
    ~/ckpt_keep/002000 002000 --repo-type model
```

```bash
# (d) それでも 0.00B/s なら、多コミットの道具に替える。
# upload-large-folder は「フォルダの中身 → リポジトリの直下」に写すので、
# 002000/ の階層を作るために、控えを 002000 という名前でもう 1 段くるむ（cp -al はハードリンクで 0 バイト）
mkdir -p ~/up && cp -al ~/ckpt_keep/002000 ~/up/002000
hf upload-large-folder siruku6/pi05_camdrop2500 ~/up --repo-type=model
```

> **`$HOME` に 9.4 GB を置く risk は測っていない。**`$HOME` は 10 分ごとに自動保存され、
> 起動時に復元される（`open_items.md` の `F53`）。**復元が何分延びるかは分からない。**
> **それでも控えを取る**——**上がっていない重みは、セッションが終わった時点で存在しないのと同じである。**

#### 次の走りで踏まないための直し（**B4 の script は 9/3 に直した**）

**`last` を飛ばす 1 行を足した。**下の B4 の script にもう入っているので、
**次のセッションでは `hf download siruku6/pi05_code` の後、B4 をそのまま打てばよい。**

---

## 落とし穴（踏むと 1 日が消えるもの）

| 落とし穴 | 対処 |
|---|---|
| **カスタム Docker は使えない**（資料 v1.1 p14） | `$HOME` の venv に作る（A3）。**配布イメージの `docker run` は提出前の最終確認に使う** |
| **認証していないと非公開リポジトリが `not found` に見える** | `~/.hf_token` を置く（A3）。**名前の間違いではない** |
| **Notebook のセルが `/bin/sh` で動くことがある** | `source` が無い。**`.` で読む**（8/31 に踏んだ） |
| **`df` が答えない共有ストレージがある** | 使用量は `du -sh`、枠は画面で見る |
| **学習ログを `$OUTPUT_ROOT` にしか置かない** | **セッション終了で消える。****loss の推移はチェックポイントに残らないので、やり直しでは作れない。**B4b のループで `$HOME/train_logs/` と HF の 2 か所へ写す（**`~/data` は symlink で消えるので置き先にしない**） |
| **ディスクが埋まると書いている途中のファイルが壊れる** | 書く前に空きを見る。**`policy_server.py` を実際に 0 バイトにした** |
| **`--learning-rate` を省くと 10 倍から再スタートする** | 必ず `5e-6` を書く |
| **`--camera-viewpoint-augment` を惰性で外す** | **いまは付ける。**8.0.0 が付けているので、外すと変数が 2 つになる（下の「場面カメラ」節） |
| **打ち間違えて回し始めた** | **[分岐 4](#分岐-4-打ち間違えて回し始めた--途中で止めて打ち直す)。****監視セル（B6）を先に止めること**——学習を先に殺すと、B6 が自分でサーバを止めてセッションが終わる |
| **上げるループを仕掛けたまま朝まで放置する** | **上がったかを見張る手が無い。**`~/upload_ckpt.log` は成功のときだけ書くので、**失敗は 1 行も残らない。**9/3 は `000500` の次から 1 個も上がらないまま 4 時間走った（[分岐 5](#分岐-5-hf-に-000500-しか上がっていない9-3-に実際に起きた)）。**寝る前に 1 度、HF の一覧に歩数が 2 つ以上並ぶことを見ること** |
| **`checkpoints/last` の symlink を上げてしまう** | **HF に `last/` という歩数のふりをした複製ができ、空きを 9.35 GB 食う。**B4 の script に `[ -L … ] && continue` を入れた（9/3） |
| **最後のチェックポイントを上げ切る前にサーバが止まる** | **B6 の監視セルは、上げるループが居ないと学習の終了と同時に `DELETE` を投げる。**上げるループが死んでいることに気づいたら、**まず B6 を止める**（分岐 5 の 1 番） |
| **`train_pi05.py` は成功しても終了コード 120 を返す** | 8/31 の 60 歩で実測。`[train_pi05] === 終了（終了コード 0）===` を出したあと、`_Tee.flush` が閉じた stdout を flush しようとして `ValueError` になり、python が 120 で終わる。**`&&` で後続を繋がないこと。**判定は**ログの `終了（終了コード 0）` の行と、成果物の `du -sh` で行う** |

---

## あとで見る資料

| 知りたいこと | ファイル |
|---|---|
| **Blackwell で測った速さ・容量・パスの実体・踏んだ罠** | **[../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md)** |
| どのマシンに何があるか・60 時間の減り方 | [../resources/infra_and_docker.md](../resources/infra_and_docker.md) |
| 配布データセットのタスク構成（8/29 に数えたもの） | [../reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md) |
| 流した学習・重み・版・出した点 | [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) |
| 手元の 72 行が重みの判定に使えない理由 | [../knowledge/18_local_eval_missed_the_gain.md](../knowledge/18_local_eval_missed_the_gain.md) |
| RunPod で同じ学習を回す手順 | [06_runpod_pi05.md](06_runpod_pi05.md) |
| 会社の RTX6000 で評価を回すとき | [03_gpu_eval.md](03_gpu_eval.md) の「会社の RTX6000 では EGL が効く」 |
