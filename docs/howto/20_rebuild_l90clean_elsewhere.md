# 20. `libero_combined_bowl5_initpose_l90clean` を別のマシンで組み直す

TOC:
- [0. このファイルの役割](#0-このファイルの役割)
- [1. 先に確かめること](#1-先に確かめること)
- [2. 落とすもの・運ぶもの](#2-落とすもの運ぶもの)
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
| python の環境 | `python3 -c "import pyarrow, numpy, pandas; print('ok')"` | LeRobot v0.6.0 の入った環境 |
| **配布 tar が運べるか** | — | **`libero_combined_20hz.tar` は HF に無い。**2 節を先に読む |

**空きの内訳。**

| 何 | 大きさ |
|---|---|
| `libero_plus_bowl5_initpose`（落とす） | 22.6 GB |
| 配布 `libero_combined_20hz`（**運ぶ**・展開する） | tar 21 GB ＋ 展開 20 GB |
| 親 `libero_combined_bowl5_initpose`（組む） | 27.6 GB |
| `libero_90_clean`（落とす） | 3.6 GB |
| 出来上がり（組む） | 27.8 GB |
| 突き合わせる相手 `l90clean`（落とす。**4 節を数だけで済ませるなら要らない**） | 27.8 GB |

**動画はハードリンクで置くので、組む段でディスクは増えない**
（`rebuild_dataset.py` の冒頭に理由がある）。上の「組む」ぶんは、
メタデータと parquet だけの目安である。

## 2. 落とすもの・運ぶもの

```bash
# 部品（撮り直した実演。これが無いと組めない）
hf download siruku6/libero_plus_bowl5_initpose --repo-type dataset \
    --local-dir work/artifacts/libero_plus_bowl5_initpose
hf download siruku6/libero_90_clean --repo-type dataset \
    --local-dir work/artifacts/libero_90_clean

```

**配布データ `libero_combined_20hz` だけは落とせない。**
**`lerobot/libero_combined_20hz` は Hugging Face に無く、運営が本選で配布した tar が唯一の出どころである**
（[reference/25 の 31 行](../reference/25_dataset_inventory.md)・[howto/14](14_blackwell_session.md)）。
**置いてあるのは 2 か所しかない。**

| 置き場 | 形 |
|---|---|
| 会社の RTX6000（`eng003sasaki01`） | `/home/ubuntu/src/train_data/libero_combined_20hz.tar`（21 GB）／展開済み `libero_combined_20hz_extracted/libero_combined_20hz`（20 GB） |
| Blackwell サーバ | `~/dataset/lerobot/` |

**組む台へは、このどちらかから運ぶ**（`scp` か、外付けに載せる）。
**運ぶのは tar 1 つで済む**——21 GB である。

```bash
# 例: RTX6000 から組む台へ
scp eng003sasaki01:/home/ubuntu/src/train_data/libero_combined_20hz.tar .
tar xf libero_combined_20hz.tar -C <展開先>
# $COMBINED は、展開先の中の libero_combined_20hz ディレクトリを指す
```

**台帳は git に入っているので落とす必要はない。**ただし `build_initpose_datasets.sh` は
`work/regen/` の下を見るので、**そこへ置き直す。**

```bash
mkdir -p work/regen/combined
cp solution/tools/regen/manifests/head_keep.json      work/regen/combined/head_keep.json
cp solution/tools/regen/manifests/keep_replace_v2.json work/regen/keep_replace_v2.json
cp solution/tools/regen/manifests/grader_texts.txt     work/regen/grader_texts.txt
```

## 3. 組む

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
