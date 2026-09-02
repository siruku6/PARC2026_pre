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
| **[場面カメラを潰して 1 本](#場面カメラを確率で潰して-1-本回す9-2-の夜に足した)** | **いま回すのはこれである**（`open_items.md` の `F50`①） |
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
| `rp3000`（＝`pi05_full_runpod`・9.37 GB） | `hf download siruku6/pi05_full_runpod --local-dir /opt/dlami/nvme/weights/rp3000` | カメラ拡張なしの 1,500 歩から始める場合 |

> **`Error: Model '…' not found.` は名前の間違いではない。認証していないだけである。**
> `pi05_full_runpod`・`pi05_trial1500`・`pi05_code` は**3 つとも非公開**で、
> HF は非公開リポジトリを未認証の相手に「無い」と答える。
> **他の罠は [../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md) 5 節にまとめてある。**

### 学習コードを降ろす（**古い版を掴んでいないか不安なときだけ**）

**`~/pi05_code` は `$HOME` にあるので残っている。**ただし**手元で直した分は入っていない**——
**9/2 に場面カメラを潰す実装を足したので、その日以降は 1 度取り直す。**

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
[分岐 2](#分岐-2-hf-の空きが-27-gb-に足りない)と
[../talk/260831_today.md](../talk/260831_today.md) の「決めてほしいこと 1」に従い、
**1,000 歩ごと・最新のいくつかだけにする。**

**500 も入らないときは回さない。**環境（`~/.local` と `~/lerobot`）は `$HOME` に残るので、
**翌日は A3 を飛ばして 30 分早く始められる。**

---

## B 学習を始める（90 分）

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

**変えるのはここだけである。**8/29 に採点側で両トラックとも上がった
（[../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) の「8/29 の 5.0.1」）のは
「画像側を開けたフルパラメータ学習」と「カメラ拡張」の
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
create_repo("siruku6/pi05_camdrop2500", private=True, exist_ok=True)
print("上げ先を作った: siruku6/pi05_camdrop2500")
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
nohup python train_pi05.py --run-name pi05_camdrop2500 \
    --init-from /opt/dlami/nvme/weights/rp3000 \
    --steps 1500 --batch-size 64 --num-workers "$NW" \
    --learning-rate 5e-6 --save-freq 500 \
    --freeze-vision-encoder false --train-expert-only false \
    > $OUTPUT_ROOT/pi05_camdrop2500.nohup 2>&1 &
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
grep -m1 num_learnable_params /opt/dlami/nvme/out/pi05_camdrop2500.log
tail -5 /opt/dlami/nvme/out/pi05_camdrop2500.log
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
OUT=/opt/dlami/nvme/out/pi05_camdrop2500
REPO=siruku6/pi05_camdrop2500
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

**学習が終わってからもう一巡してから抜けるので、最後の 001500 も上がる。**

### B5 最初の 500 ステップが HF に載るまで見届ける（40 分）

**ここまでが在席である。**500 ステップは 10.24 秒/step なら 85 分後に出る。
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
for f in HfApi().repo_info('siruku6/pi05_camdrop2500', repo_type='model').siblings:
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

### 打つ 1 行（8.0.0 との 1 変数）

```bash
. ~/env_pi05.sh
cd "$CODE_DIR"

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

**最後の `--scene-camera-dropout 0.5` 以外は、8.0.0 を作った学習と 1 文字も違わない**
（`work/train/pi05_ladder2500.log` の 8 行目の `lerobot-train` の行と突き合わせた。
batch 64・学習率 5e-6・種 42・`save_freq` 500・`gradient_checkpointing=true`・bfloat16・
`empty_cameras=1`・`n_action_steps=10`・データ `lerobot/libero_plus` rev `f3f49f42`・
カメラ拡張は 8 種から最大 4 つ）。**だから 1 変数である。**

> **`--camera-viewpoint-augment` は付ける。**8.0.0 が付けているので、
> 外すと変数が 2 つになる。**この 2 つは別のものである**——拡張は
> **両方のカメラ**に色や歪みをかけ、潰すほうは**場面カメラだけ**を消す。

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
bash solution/tools/train/run_camera_dropout_test.sh
```

**常駐コンテナ `quant_train` の本物の torch 2.11 と lerobot 0.6.0 で 11 件走る**
（`tests/test_camera_dropout.py`）。**この機械の venv には torch も lerobot も
入っていないので、`pytest tests/` を直接打つと skip になる。**

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
| **`df` が答えない共有ストレージがある** | 使用量は `du -sh`、枠は画面で見る |
| **ディスクが埋まると書いている途中のファイルが壊れる** | 書く前に空きを見る。**`policy_server.py` を実際に 0 バイトにした** |
| **`--learning-rate` を省くと 10 倍から再スタートする** | 必ず `5e-6` を書く |
| **`--camera-viewpoint-augment` を惰性で外す** | **いまは付ける。**8.0.0 が付けているので、外すと変数が 2 つになる（下の「場面カメラ」節） |
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
