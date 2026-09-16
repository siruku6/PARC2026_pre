# 20. MolmoAct2 をこのマシン（H100 20GB・1 枚）で LoRA 学習する

**RunPod を借りずに、手元の 20 GB 1 枚で MolmoAct2 の追加学習を回すための手順である。**
[19_train_l90clean_runpod.md](19_train_l90clean_runpod.md) が RunPod（80 GB 級・全パラメータ学習）
を持つのに対し、**こちらは Docker のイメージと 20 GB 1 枚に絞る。**
学習の引数・学習率・直しの内訳は 19 と共通で、
[jaist_molmoact2_lora.sh](../../solution/train/jaist_molmoact2_lora.sh) に 1 か所ある。

| 節 | 中身 |
|---|---|
| [1 このマシンでできるのは LoRA だけ](#1-このマシンでできるのは-lora-だけである) | 20 GB に載る範囲と batch |
| [2 イメージを作る](#2-イメージを作る) | 足りないのは `peft` 1 つなので薄い層で済む |
| [3 打つもの](#3-打つもの) | **1 行で試し走り → 本番まで進む。**session が切れても続く |
| [3.5 `loss:nan` は直っている](#35-lossnan-は直っている916) | 原因は LoRA と重み読み込みの順。直し済み |
| [4 Docker のために足していること](#4-docker-のために足していること) | venv・直しの当て方・前面実行の 3 つ |
| [5 この手順に入っていないこと](#5-この手順に入っていないこと) | 提出物への組み込み |

## 1. このマシンでできるのは LoRA だけである

**全パラメータ学習は batch 8 でも 48.3 GiB 要る**（同梱 LeRobot の
`policies/molmoact2/README.md` の実測表）ので、20 GB には載らない。
**LoRA（VLM に LoRA・行動を出す小さな模型はフル学習）だけが選べる。**

**9/16 に RTX 6000（24 GB）で測った山**（[jaist_molmoact2_lora.sh](../../solution/train/jaist_molmoact2_lora.sh) の冒頭）。

| batch | RTX 6000（24 GB）の山 | **このマシン（20 GB）で 9/16 に測った結果** |
|---|---:|---|
| 2 | 18.7 GB | 未測（4 が通ったので測っていない） |
| **4** | 19 GiB | **通った。1 歩 4.157 秒・山 18.7 GB（19,212 MiB）** |
| 8 | 21 GiB | **OOM**（19.80 GiB 中 36.5 MiB しか空かない） |
| 16 | OOM | 論外 |

**だから `LADDER` は `"4 2"` から始める。**8 を入れると、落ちると分かっている段で
重みの読み込みぶんを捨てる。

**歩数と実時間は [reference/19 の 7.10](../reference/19_weights_catalog.md) にある**
（4,000 歩で 4 時間 37 分・16,000 サンプル）。**20 歩の時点で `loss:nan` が出ているので、
歩数を決める前に切り分けること**（`ACTION_MODE=continuous` と `LOG_FREQ=1`）。

**3 節の試し走りが、この 3 つ（載るか・1 歩の秒数・山）をそのまま出す。**
**勾配累積は lerobot v0.6.0 に無い**ので、実効 batch は batch そのものである
（公式の出発点は 256）。

## 2. イメージを作る

同梱 LeRobot v0.6.0 の `pyproject.toml`（215〜216 行）を見比べると、差は 1 つだけである。

| 追加指定 | 中身 |
|---|---|
| `pi`（`parc2026-train-pi05` に入っている） | transformers + scipy |
| `molmoact2` | transformers + scipy + **peft** |

だから [Dockerfile.molmoact2](../../solution/train/Dockerfile.molmoact2) は
`parc2026-train-pi05` を土台にした薄い層にしてある——torch・torchcodec・lerobot の重い層
（15〜30 分）を焼き直さない。**直した lerobot のファイルは焼かない**——評価も学習も
`solution/tools/eval/lerobot_patches/` から読み取り専用で被せるので、焼くと版が 2 系統になる。
おかげでビルドコンテキストは `Dockerfile.pi05` と同じ `solution/train` で済む。

```bash
docker build -t parc2026-train-molmoact2 -f solution/train/Dockerfile.molmoact2 solution/train
```

## 3. 打つもの

**データは先に降ろしておく**（`siruku6/libero_combined_bowl5_initpose_l90clean`・
21,642 本・111 課題・27.8 GB・非公開なので `hf auth login` が要る）。

```bash
hf download siruku6/libero_combined_bowl5_initpose_l90clean --repo-type dataset \
    --local-dir work/artifacts/libero_combined_bowl5_initpose_l90clean
```

**試し走りから本番までを 1 行で終わらせる。**`LADDER` に書いた batch を大きいほうから
試し、**載った最大でそのまま本番を起こす。**

```bash
setsid nohup env RUN=molmoact2_lora_v1 LADDER="4 2" STEPS=4000 SAVE_FREQ=1000 \
    bash solution/train/local_molmoact2_lora.sh > ~/molmoact2_lora_v1.log 2>&1 &
```

**この shell を閉じても学習は続く。**本番はコンテナを切り離して起こすためで、
`setsid nohup` は梯子の途中（試し走りの間）に shell が死んでも続けるためのものである。

| 見るもの | 打つもの |
|---|---|
| 梯子と起動の様子 | `tail -f ~/molmoact2_lora_v1.log` |
| 学習のログ | `docker logs -f molmoact2_molmoact2_lora_v1 \| tr '\r' '\n'` |
| 止める | `docker rm -f molmoact2_molmoact2_lora_v1` |

**1 つずつ打つなら次の 2 行である。**

```bash
# 試し走り（載るか・1 歩の秒数・メモリの山。出力は残さない）
TRIAL=1 BATCH=2 bash solution/train/local_molmoact2_lora.sh

# 本番
RUN=molmoact2_lora_v1 BATCH=2 STEPS=4000 SAVE_FREQ=1000 \
    bash solution/train/local_molmoact2_lora.sh
```

**分位点（q01/q99）は最初の 1 回だけ自動で足される。**MolmoAct2 は状態と行動を
QUANTILES で正規化するが、π0.5 向けに組んだデータには入っていない。
公式の `augment_dataset_quantile_stats.py` は**全コマの動画を復号する**ので
21,642 本では日をまたぐ——[add_quantile_stats.py](../../solution/tools/train/add_quantile_stats.py)
は parquet の数値列だけを読む（**画像は MolmoAct2 では IDENTITY なので分位点を使わない**）。

## 3.5. `loss:nan` は直っている（9/16）

**9/16 の最初の試し走りは、batch 4 で通ったが 20 歩の時点で `loss:nan grdn:nan` だった。**
**原因は LoRA と重みの読み込みの順である**——LoRA は `__init__` の中、つまり
`--policy.path` の重みを読む前に当たる。PEFT が全パラメータ名へ `base_model.model.` と
`.base_layer.` を挟むため、**素の名前で保存された safetensors 1,295 個がすべて名前不一致となり、
`strict=False` のまま黙って捨てられていた。**未初期化の重みで学習が始まるので 1 歩目から nan になる。

**直し済みである**（`solution/tools/eval/lerobot_patches/modeling_molmoact2.py` の
`_load_as_safetensor`。素の名前を PEFT の名前へ対応付けてから読み、1 個でも対応付かなければ
例外で止める）。**RTX 6000・batch 2 で 1,295 個すべて読み込み、`both` 10 歩・`continuous` 5 歩とも
有限であることを確かめてある。**

## 4. Docker のために足していること

**学習そのものは [jaist_molmoact2_lora.sh](../../solution/train/jaist_molmoact2_lora.sh) を呼んでいる。**
[local_molmoact2_lora.sh](../../solution/train/local_molmoact2_lora.sh) が足すのは Docker の口だけである。

| 足していること | なぜ |
|---|---|
| `VENV=none` | イメージの中に venv は無い（lerobot は最初から入っている） |
| 直し 2 ファイルの bind-mount | `--user` で非 root にすると `/opt/lerobot` を上書きできない。読み取り専用で被せると、呼ばれた側は目印を見て「当たっている」と判断して当てない |
| `FOREGROUND=1` | 切り離すと、スクリプトが終わった時点でコンテナごと落ちる |
| `--shm-size=16g` | 読み手の共有メモリ。既定の 64 MB では 1 歩も進まずに落ちる |

**直しが 2 つ要る理由は 19 の 4.1 節と同じである**（素の重み 21.8 GB の二重読み込みと、
前処理ステップ名の決め打ち）。**列名の違いは `--rename_map` で吸収する**
（データは `front`/`wrist`、出発点は `image`/`wrist_image`）。
`--policy.image_keys` では直らない——`make_policy` がその前に
`validate_visual_features_consistency` で落とすためである（9/16 に踏んだ）。

## 5. この手順に入っていないこと

- **提出物への組み込み。**LoRA の差分を元の重みへ溶かし込んで出すか、差分のまま出して
  `requirements.txt` に `peft` を足すかは、まだ決めていない（依頼 032 の 4.2 節）。
- **歩数の決め方と評価。**19 の 4.3 節（2,000〜4,000 歩の見積もり）と同じである。
- **π0.5 側のイメージの差し替え。**`parc2026-train-pi05` はそのまま残る。
