# 部品から組み直す

*English: [README.md](README.md)*

**混ぜたデータセットを消しても作り直せるようにするための一式である。**

**ここに入っているものだけで足りる。**外部のリポジトリは要らない。

| 入っているもの | 何か |
|---|---|
| `solution_<commit>.tar.gz` | **組み上げに使ったソース一式**（スクリプト・Dockerfile・台帳） |
| `manifests/` | **どの本を外し、どの本で埋め戻したかの判断**（中身を見るための写し。tar の中にも同じものが入っている） |
| `SOURCE_COMMIT.txt` | 元にした版 |

**2026-09-18 に、この一式だけで別のマシン（Apple Silicon の Mac）から組み直し、
21,642 本すべての行動列が公開版と一致することを確かめた。**

```
行動列が一致した本: 21642 / 21642
```

## 目次

- [1. 何が出来るか](#1-何が出来るか)
- [2. 材料](#2-材料)
- [3. 実行環境を立てる](#3-実行環境を立てる)
- [4. 組む](#4-組む)
- [5. 一致したか確かめる](#5-一致したか確かめる)
- [6. 詰まりやすいところ](#6-詰まりやすいところ)

## 1. 何が出来るか

**`libero_combined_bowl5_initpose_l90clean`（21,642 本・3,431,137 コマ・111 課題・27.8 GB）**
を、部品と台帳から組み直せる。

```
libero_plus_bowl5_initpose（HF・16,820 本）  ┐
libero_combined_20hz（運営配布・19,533 本）  ┤→ libero_combined_bowl5_initpose（親・21,642 本）
                                             │
libero_90_clean（HF・3,911 本）  ────────────┴→ l90clean（250 本を入れ替え・21,642 本）
```

**2 段に分かれている。**前段が親を作り、後段が `libero_90` の汚れた 250 本を入れ替える。

## 2. 材料

| 要るもの | どこから |
|---|---|
| `siruku6/libero_plus_bowl5_initpose`（22.6 GB） | Hugging Face |
| `siruku6/libero_90_clean`（3.6 GB） | Hugging Face |
| **`libero_combined_20hz`（21 GB・LeRobot v3.0）** | **PARC2026 の運営が配布したもの**（Omnicampus の配布環境、または運営が案内した Google Drive） |

```bash
hf download siruku6/libero_plus_bowl5_initpose --repo-type dataset \
    --local-dir work/artifacts/libero_plus_bowl5_initpose
hf download siruku6/libero_90_clean --repo-type dataset \
    --local-dir work/artifacts/libero_90_clean
```

**配布データは v3.0 でなければならない。**`libero_combined_20hz_v2` は v2.1 なので使わない。
**理由がある**——組み上げが動画を作り直さずに済むのは、**v3.0 が「1 本の mp4 に何十本もの
エピソードを詰め、開始時刻で切り出して読む」形だから**である。v2.1 では前提が崩れる。

```bash
python3 -c "
import json; i=json.load(open('<展開先>/libero_combined_20hz/meta/info.json'))
print(i['codebase_version'], i['total_episodes'], i['total_tasks'])"
# v3.0 19533 111 になること
```

**ディスクは 120 GB 見ておく。****動画はハードリンクで置くので組む段では増えない**が、
繋ぐ段だけは実体で写す。

## 3. 実行環境を立てる

**ソースを展開し、その中で作業する。**

```bash
tar xzf solution_<commit>.tar.gz          # solution_src/solution/ が出てくる
cd solution_src
```

**重い処理はすべて Docker の中で回る。**手元の python3 には何も入れなくてよい
（`meta/info.json` の数を読むのに標準ライブラリだけ使う）。

```bash
docker build -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train
```

**arm64（Apple Silicon）では `--build-arg WITH_QUANT=0` を足す。**
CUDA 用の索引に `torchao` の aarch64 wheel が無いためで、
**`torchao` と `bitsandbytes` は学習用であって組み上げには要らない。**

```bash
docker build --build-arg WITH_QUANT=0 \
    -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train
```

**立ったか確かめる。**

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    -c "import lerobot; print(lerobot.__version__)"
# 0.6.0 と出れば通っている
```

**コンテナはリポジトリ直下を `/workspace/repo` として見る。**だから**出力は展開した木の下に置く。**

**台帳を置き直す**——スクリプトは `work/regen/` の下を見る。

```bash
mkdir -p work/regen/combined
cp solution/tools/regen/manifests/head_keep.json       work/regen/combined/head_keep.json
cp solution/tools/regen/manifests/keep_replace_v2.json work/regen/keep_replace_v2.json
cp solution/tools/regen/manifests/grader_texts.txt     work/regen/grader_texts.txt
```

## 4. 組む

### 前段——親を作る

```bash
COMBINED=<展開した配布データ>/libero_combined_20hz \
STAGE=c bash solution/tools/regen/build_initpose_datasets.sh
```

**`STAGE=p` は飛ばしてよい。**その出力 `libero_plus_bowl5_initpose` は HF から落としてあるからである。

**何をするか。**

1. 配布データの先頭 5,186 本から、台帳（`head_keep.json`）が残すと決めた **4,822 本**を抜く
2. その 4,822 本の**指の値を `{0,1}` から `±1` に直す**——直さないと「開く」に逆の正解が付く
3. `libero_plus_bowl5_initpose`（16,820 本）の末尾に繋ぐ → **21,642 本・3,436,940 コマ・111 課題**
4. 門を 4 つ通す（本数、指の値、正規化の尺度、**繋いだ先頭が元と同じコマを指しているか**）

### 後段——`libero_90` の 250 本を入れ替える

```bash
IMAGE=parc2026-train-pi05 \
PARENT=work/artifacts/libero_combined_bowl5_initpose \
POOL=work/artifacts/libero_90_clean \
bash solution/tools/regen/rebuild_l90clean.sh
```

**読むのは `manifests/l90clean_replace.json` だけである。**外す 250 本の番号と、
足す 250 本の出どころ（`source_episode_index`）が入っている。

1. 台帳から keep-file を作る（残す 21,392 本 / 足す 250 本）
2. 親から 250 本を外す
3. `libero_90_clean` から 250 本を台帳の順に抜く
4. 末尾に繋ぐ → **21,642 本・3,431,137 コマ・111 課題**
5. 台帳と突き合わせる

**足した 250 本は末尾に付く**ので、**外した位置とは並びが違う。**
番号で対応を取るなら `l90clean_replace.json` を引くこと。

**出どころは `libero_90_clean` であって `libero_90_clean_appearance` ではない。**
後者は `(課題, 長さ)` で 110/250 しか説明できない。
**対応は名前や長さではなく、行動列の sha256 で決めてある。**

## 5. 一致したか確かめる

**5 段目が数を突き合わせる。**

```
merged_episodes: 21642（台帳 21642）OK
merged_frames: 3431137（台帳 3431137）OK
merged_tasks: 111（台帳 111）OK
```

**数だけでは足りないと考えるなら、中身も突き合わせる。**

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/regen/compare_action_signatures.py \
    --a <公開版> --b work/artifacts/libero_combined_bowl5_initpose_l90clean_rebuilt
```

```
  A 21642 本 / B 21642 本
行動列が一致した本: 21642 / 21642
```

## 6. 詰まりやすいところ

**どれも 2026-09-18 に実際に踏んだものである。**

| 症状 | 中身 |
|---|---|
| **印字の途中で、何も言わずに止まる** | **Docker に割り当てたメモリ不足。**Docker Desktop の既定は小さい（ある Mac で 1.9 GB だった）。`docker info --format '{{.MemTotal}}'` で見て **8 GB 以上**にする |
| `could not select device driver "" with capabilities: [[gpu]]` | GPU の無い台。**組み上げに GPU は要らない**ので、`GPUS=` を頭に付けて外す |
| `No matching distribution found for torchao==0.17.0` | arm64 で組んだとき。`--build-arg WITH_QUANT=0` |
| **繋ぐ段でディスクが急に減る** | 動画がハードリンクにならず実体で写っている。**元と出力が別のファイルシステムに跨っていないか**、ファイルシステムがハードリンクに対応しているか（exFAT/FAT は不可） |

**実測**（Apple Silicon・Docker 8 GB）。

| 段 | 時間 |
|---|---|
| 前段 1〜2（抜く・指の直し） | 約 1 分 |
| **前段 3（21,642 本を繋ぐ）** | **約 4 分** |
| 前段の門 1〜4 | 数分 |
| **後段 4（繋ぐ）** | **約 2 分 40 秒** |

## ライセンス

LIBERO と LIBERO-plus から派生したもので、配布元は **MIT License**
（Copyright (c) 2023 Lifelong Robot Learning）である。
同梱のソースも同じ条件で使える。
