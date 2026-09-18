# 20. `libero_combined_bowl5_initpose_l90clean` を別のマシンで組み直す

TOC:
- [0. このファイルの役割](#0-このファイルの役割)
- [1. 先に確かめること](#1-先に確かめること)
- [2. 材料を揃える](#2-材料を揃える)
- [3. 組む](#3-組む)
- [4. 一致したか確かめる](#4-一致したか確かめる)
- [5. 通っていないところ](#5-通っていないところ)

## 0. このファイルの役割

**`siruku6/libero_combined_bowl5_initpose_l90clean`（27.8 GB）を消せるようにするための手順である。**

**消してよいのは、部品と台帳から同じものが作れると確かめたときだけである**
（[cards の 4 節](../cards/README.ja.md#4-保管の方針)）。
**このファイルは、その確認を会社の RTX6000 以外の台で 1 回通すための手順を持つ。**

**なぜ別の台か。**RTX6000 は空きディスクが足りず（2026-09-18 に 26 GB）、
**親 27.6 GB ＋ 出来上がり 27.8 GB ＋ 落とすもの 50 GB 超**が入らない。
回線も細いので、**落とす量の多い作業は別の台に出す。**

## 1. 先に確かめること

| 確かめること | 打つもの | 要る値 |
|---|---|---|
| **ディスクの空き** | `df -h .` | **120 GB 以上**（下の内訳） |
| git が最新か | `git log --oneline -1` | `solution/tools/regen/rebuild_l90clean.sh` があること |
| 素の python | `python3 -c "import pyarrow, numpy, pandas; print('ok')'"` | `rebuild_dataset.py` はこれだけで動く |
| **lerobot の入った環境** | `docker image inspect parc2026-train-pi05` | **`merge_many.py` と `npz_to_lerobot.py` が `lerobot` を import する。**無ければ 3 節で立てる（`NATIVE=1` で素の python にもできる） |
| **配布データの版** | `meta/info.json` の `codebase_version` | **`v3.0`。**v2.1 の配布物もあるので取り違えない（2 節） |

**空きの内訳。**

| 何 | 大きさ |
|---|---|
| `libero_plus_bowl5_initpose`（落とす） | 22.6 GB |
| 配布 `libero_combined_20hz`（落とすか運ぶ・展開する） | tar 21 GB ＋ 展開 20 GB |
| 親 `libero_combined_bowl5_initpose`（組む） | 27.6 GB |
| `libero_90_clean`（落とす） | 3.6 GB |
| 出来上がり（組む） | 27.8 GB |
| 突き合わせる相手 `l90clean`（落とす。**4 節を数だけで済ませるなら要らない**） | 27.8 GB |

**動画はハードリンクで置くので、組む段でディスクは増えない**
（`rebuild_dataset.py` の冒頭に理由がある）。上の「組む」ぶんは、
メタデータと parquet だけの目安である。

## 2. 材料を揃える

```bash
# 部品（撮り直した実演。これが無いと組めない）
hf download siruku6/libero_plus_bowl5_initpose --repo-type dataset \
    --local-dir work/artifacts/libero_plus_bowl5_initpose
hf download siruku6/libero_90_clean --repo-type dataset \
    --local-dir work/artifacts/libero_90_clean

```

**配布データ `libero_combined_20hz` は Hugging Face には無いが、配布はされている。**
**運営が Google Drive でも配っている**
（[materials/notifications/260914_1000](../materials/notifications/260914_1000_location_of_libero_combined_20hz.txt)）。
手元やほかの台から運んでもよい。

| 取り方 | 置き場 |
|---|---|
| **Google Drive から落とす** | 上の通知にリンクがある（Omnicampus の配布環境経由でも取れる） |
| 会社の RTX6000 から運ぶ | `/home/ubuntu/src/train_data/libero_combined_20hz.tar`（21 GB）／展開済み `libero_combined_20hz_extracted/libero_combined_20hz` |
| Blackwell サーバから運ぶ | `~/dataset/lerobot/` |

**配布は 2 種類あるので、取り違えないこと。**

| 名前 | LeRobot の版 | 使うか |
|---|---|---|
| **`libero_combined_20hz`** | **v3.0** | **こちらを使う** |
| `libero_combined_20hz_v2` | v2.1 | **使わない** |

**v3.0 でなければならない理由がある。**`rebuild_dataset.py` が動画を作り直さずに済むのは、
**v3.0 が「1 本の mp4 に何十本ものエピソードを詰め、開始時刻で切り出して読む」形だから**である。
v2.1 を渡すと、この前提が崩れる。**手元のデータセットはすべて v3.0 で作ってある**
（`meta/info.json` の `codebase_version` で確かめられる）。

**取ったものが同じかは数で確かめる。**

```bash
python3 -c "
import json; i=json.load(open('<展開先>/libero_combined_20hz/meta/info.json'))
print(i['codebase_version'], i['total_episodes'], i['total_tasks'])"
# v3.0 19533 111 になること
```

**なお、配布の統合データを使わずに `libero_plus` などを個別に落として組み上げることもできる。**
手間は増えるが、Google Drive を通さずに済む。

**台帳は git に入っているので落とす必要はない。**ただし `build_initpose_datasets.sh` は
`work/regen/` の下を見るので、**そこへ置き直す。**

```bash
mkdir -p work/regen/combined
cp solution/tools/regen/manifests/head_keep.json      work/regen/combined/head_keep.json
cp solution/tools/regen/manifests/keep_replace_v2.json work/regen/keep_replace_v2.json
cp solution/tools/regen/manifests/grader_texts.txt     work/regen/grader_texts.txt
```

## 3. 組む

**コードは git の `solution/` 以下だけで足りるが、実行環境は別に要る。**

| 要るもの | なぜ | 無いときの代わり |
|---|---|---|
| `solution/` 以下（git） | 台帳もスクリプトもここに入っている | — |
| **Docker イメージ `parc2026-train-pi05`** | **`merge_many.py` と `npz_to_lerobot.py align` が `lerobot` を import する** | **lerobot v0.6.0 を入れて `NATIVE=1`**（`rebuild_l90clean.sh` のみ対応） |
| 配布データ `libero_combined_20hz` | 標準スイートと `libero_90` の元 | 2 節 |
| HF の部品 2 つ | 撮り直した実演 | 2 節 |

**`rebuild_dataset.py` は `numpy`・`pyarrow`・`pandas` だけで動く**ので、
外す・抜くの段はコンテナが無くても通る。**繋ぐ段だけが lerobot を要る。**

### コンテナを立てる

**イメージは `solution/train/Dockerfile.pi05` から作る**（git に入っている・出来上がりは 5.1 GB）。

```bash
docker build -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train
```

**Docker Hub にも上げてある**（`sirukufarios/parc2026-train-pi05:v2`）。
落とすほうが速ければそちらでよい。

```bash
docker pull sirukufarios/parc2026-train-pi05:v2
docker tag  sirukufarios/parc2026-train-pi05:v2 parc2026-train-pi05
```

**ただし上げてあるのは x86_64 の像だけである。**Apple Silicon の Mac では
`no matching manifest` で落とせない（2026-09-18 に確認）。**組むほうに回ること。**

#### arm64（Apple Silicon）で組むとき

**`--build-arg WITH_QUANT=0` を付ける。**

```bash
docker build --build-arg WITH_QUANT=0 \
    -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train
```

**理由。**`Dockerfile.pi05` は学習用なので、量子化スイープ用の
`bitsandbytes` と `torchao` を入れる層を持っている。
**`download.pytorch.org/whl/cu128` には `torchao` の aarch64 wheel が無く、
そこだけが落ちる。**

```
ERROR: Could not find a version that satisfies the requirement torchao==0.17.0
       (from versions: none)
Dockerfile.pi05:116
```

**この 2 つは組み上げには要らない。**`merge_many.py` が import するのは
`lerobot.datasets` だけである。**`WITH_QUANT=0` で飛ばした像は学習には使えないが、
繋ぐ段には使える。**

**それでも駄目なら x86_64 を翻訳して回す。**時間はかかるが、
上げてある像がそのまま使える。

```bash
docker pull --platform linux/amd64 sirukufarios/parc2026-train-pi05:v2
docker tag  sirukufarios/parc2026-train-pi05:v2 parc2026-train-pi05
```

**立てられたか確かめる。**

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    -c "import lerobot; print(lerobot.__version__)"
# 0.6.0 と出れば通っている
```

**コンテナの中で何が起きるか**（`run_in_container.sh` が組み立てる）。

| | |
|---|---|
| リポジトリ | `-v <リポジトリ>:/workspace/repo -w /workspace/repo` で丸ごと挿す |
| **だから出力はリポジトリの下に置く** | `work/artifacts/...` のままでよい。**リポジトリの外を指すと中から見えない** |
| 入り口 | `--entrypoint python`。**渡すのは python スクリプトの引数**である |
| 持ち主 | **コンテナは root で走るので、書いたファイルは root のものになる。**`CHOWN_PATH` を渡すと走り終えたあとに戻す |

**詰まりやすいのは 2 つである。**

| 症状 | 理由と手当て |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | `run_in_container.sh` が **`--gpus all` を付けている。**繋ぐ段に GPU は要らないので、**GPU の無い台では `--gpus all` を外した写しを使う**（`sed 's/--gpus all //' solution/tools/regen/run_in_container.sh > /tmp/run_nogpu.sh`） |
| 出来たファイルが root のもので消せない | `CHOWN_PATH=<出力先> IMAGE=... bash solution/tools/regen/run_in_container.sh ...` で戻す。**`rebuild_l90clean.sh` は渡していない**ので、走り終えたあとに `sudo chown -R "$(id -u):$(id -g)" work/` が要ることがある |



**2 段に分かれる。**前段（親を作る）は既存のスクリプト、後段（250 本の入れ替え）は
2026-09-18 に書いたスクリプトである。

```bash
# 前段: 親 libero_combined_bowl5_initpose を作る
#   STAGE=p は飛ばす——その出力 libero_plus_bowl5_initpose は HF から落としてある
COMBINED=<2 節で展開した先>/libero_combined_20hz \
STAGE=c bash solution/tools/regen/build_initpose_datasets.sh

# 後段: 汚れた 250 本を外し、綺麗な 250 本を末尾に足す
PARENT=work/artifacts/libero_combined_bowl5_initpose \
POOL=work/artifacts/libero_90_clean \
bash solution/tools/regen/rebuild_l90clean.sh
```

**後段が読むのは [`manifests/l90clean_replace.json`](../../solution/tools/regen/manifests/l90clean_replace.json) だけである。**
外す 250 本の番号と、足す 250 本の出どころ（`source_episode_index`）が入っている。

## 4. 一致したか確かめる

**`rebuild_l90clean.sh` の 5 段目が、本数・コマ数・課題数を台帳と突き合わせる。**

```
merged_episodes: 21642（台帳 21642）OK
merged_frames: 3431137（台帳 3431137）OK
merged_tasks: 111（台帳 111）OK
```

**数が合うだけでは足りないと考えるなら、中身も突き合わせる。**
行動列の sha256 で並べれば、1 本ずつ同じかが分かる（台帳を作ったときと同じ方法である）。

```bash
hf download siruku6/libero_combined_bowl5_initpose_l90clean --repo-type dataset \
    --local-dir work/artifacts/l90clean_published
python3 - <<'PY'
import glob, hashlib
import numpy as np, pandas as pd

def frames(root):
    return pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"{root}/data/**/*.parquet", recursive=True))], ignore_index=True)

def signature(df, index):
    actions = np.stack(df.loc[df["episode_index"] == index, "action"].to_numpy())
    return hashlib.sha256(np.ascontiguousarray(actions, dtype=np.float32).tobytes()).hexdigest()

a = frames("work/artifacts/l90clean_published")
b = frames("work/artifacts/libero_combined_bowl5_initpose_l90clean_rebuilt")
same = sum(signature(a, i) == signature(b, i) for i in range(21642))
print(f"行動列が一致した本: {same} / 21642")
PY
```

**21,642 / 21,642 になったら、公開してある混合データセットを消してよい。**

## 5. 通っていないところ

**`rebuild_l90clean.sh` は 1 度も実行していない**（2026-09-18・この台のディスクが足りない）。
**通す人が最初になる。**詰まりそうなところを挙げる。

| 詰まりうる所 | 兆候 | 見るもの |
|---|---|---|
| 前段の門 | `P が門を通っていない` | HF から落とした `libero_plus_bowl5_initpose` に `bowl5_expected.json` があるか |
| 列の書き方の違い | `Same features is expected` | `merge_many.py` の冒頭（`align` を先に打つ必要がある場合がある） |
| 親の本数 | `親の本数が 21,642 でない` | 前段が途中で止まっていないか。`work/regen/logs/` |
| 動画のハードリンク | ディスクが急に減る | 別のファイルシステムを跨いでいないか（跨ぐと写しになる） |

**通ったら、このファイルの 5 節を消して、代わりに実測した所要時間を書くこと。**
