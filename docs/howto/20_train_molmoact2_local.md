# 20. MolmoAct2 をこのマシン（H100 20GB・1 枚）で LoRA 学習する

**RunPod を借りずに、手元の 20 GB 1 枚で MolmoAct2 の追加学習を回すための手順である。**
[19_train_l90clean_runpod.md](19_train_l90clean_runpod.md) が RunPod（80 GB 級・全パラメータ学習）
を持つのに対し、**こちらは Docker のイメージと 20 GB 1 枚に絞る。**
学習の引数・学習率・直しの内訳は 19 と共通で、
[jaist_molmoact2_lora.sh](../../solution/train/jaist_molmoact2_lora.sh) に 1 か所ある。

| 節 | 中身 |
|---|---|
| [0 現在地（9/17）](#0-現在地917) | いま動いている走りと、対照として残す走り |
| [1 このマシンでできるのは LoRA だけ](#1-このマシンでできるのは-lora-だけである) | 20 GB に載る範囲と batch |
| [2 イメージを作る](#2-イメージを作る) | 足りないのは `peft` 1 つなので薄い層で済む |
| [3 打つもの](#3-打つもの) | **1 行で試し走り → 本番まで進む。**session が切れても続く |
| [3.5 `loss:nan` は直っている](#35-lossnan-は直っている916) | 原因は LoRA と重み読み込みの順。直し済み |
| [3.5.1 設定の確かめ方](#351-設定の確かめ方走り出しの-1-画面で全部見える) | 走り出しのログで見る 7 項目 |
| [3.7 評価のために HF へ上げる](#37-評価のために-hf-へ上げる上げるのは-pretrained_model-だけ) | 上げるのは 11 GB ぶんだけ |
| [3.6 止める・続ける](#36-止める続ける) | `docker rm -f` で止め、`RESUME=1` で続ける |
| [4 Docker のために足していること](#4-docker-のために足していること) | venv・直しの当て方・前面実行の 3 つ |
| [5 この手順に入っていないこと](#5-この手順に入っていないこと) | 提出物への組み込み |

## 0. 現在地（9/17）

| 走り | 中身 | 使い道 |
|---|---|---|
| `work/train/molmoact2_lora_v1_1` | batch 2・16,000 歩まで保存・**画像を崩す細工なし** | **対照。**拡張の有無を比べる相手として残す |
| `work/train/molmoact2_lora_v1` | **batch 4・40,000 歩・拡張あり**（9/17 04:57 に起こした） | 本命。約 14.5 時間 |

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

**探しものは終わっている——batch は 4 を使う。**本走りの実測で **1 歩 1.30 秒**であり、
batch 2 のとき（1.28 秒）とほとんど変わらない。**同じ時間で 2 倍の枚数を見られる**ので、
batch 2 を選ぶ理由は無い。`LADDER` を使うのは、板が変わったときだけでよい。

**歩数と実時間は [reference/19 の 7.10](../reference/19_weights_catalog.md) にある**
（batch 4・40,000 歩で 16 万枚・約 14.5 時間）。

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

**本番はこの 1 行である。**batch と歩数は 1 節のとおり確定している。

```bash
setsid nohup env RUN=molmoact2_lora_v1 BATCH=4 STEPS=40000 SAVE_FREQ=8000 \
    bash solution/train/local_molmoact2_lora.sh > ~/molmoact2_lora_v1.log 2>&1 &
```

**画像を崩す細工は既定で入る**（`IMAGE_TFS=true`。9/17 に既定を変えた）。LeRobot の既定の 6 種
——明るさ・コントラスト（0.8〜1.2 倍）・彩度（0.5〜1.5 倍）・色相（±0.05）・くっきりさ・
**回転 ±5°／平行移動 5%**、1 枚に最大 3 つ——が、出発点の重みの `train_config.json` と同じ中身である。
**切って回すと、Track2 の弱点（カメラ視点・初期姿勢）を鍛えないまま学習することになる**
（依頼 032 の 3.4②・4.6）。切りたいときだけ `IMAGE_TFS=false` を渡す。

**歩数を決める前に 2 つ。**

| 見るもの | 中身 |
|---|---|
| **歩数と時間** | batch 4 で **1 歩 1.30 秒**。40,000 歩なら約 14.5 時間・16 万枚である。板が変わったら `TRIAL_STEPS=60 TRIAL=1 BATCH=4` を 1 回打つと、`summarize_speed.py` が 40 歩目以降の中央値で表を出す（約 5 分） |
| **保存の間隔** | **チェックポイントは 1 個 14 GB**（`pretrained_model` 11 GB ＋ `training_state` 3.3 GB）。**`SAVE_FREQ=8000` なら 5 個で 70 GB。**空きが足りないときは [prune_checkpoints.sh](../../solution/train/prune_checkpoints.sh) を並べて 2 個だけ残す |

**歩数は始める前に決める。**`scheduler_decay_steps` に `STEPS` をそのまま渡しているので、
**途中で止めると学習率が下がりきらないところで終わる。**

**ssh や session を切っても学習は続く。**本番は `docker run -d` で切り離したコンテナが
持っているので、**起こした後は、起こした側の shell が死んでも関係ない。**
`setsid nohup` が要るのは**起こすまでの間**である——梯子（試し走り）の途中で shell が死ぬと、
本番を起こす前に終わってしまう。

| 切れたとき | どうなるか |
|---|---|
| ssh / session が切れる | **学習は続く。**ログは `docker logs` と `$OUT/$RUN.nohup` の両方に残る |
| ホストを再起動する・docker を止める | **止まる。**続きは `--resume` が要る（歩数・optimizer を引き継ぐ） |
| 間引き（`prune_checkpoints.sh`）を前面で動かしていた | **間引きだけ止まる。**`nohup` か cron で並べておくこと |

| 見るもの | 打つもの |
|---|---|
| 梯子と起動の様子 | `tail -f ~/molmoact2_lora_v1.log` |
| 学習のログ | `docker logs -f molmoact2_molmoact2_lora_v1 \| tr '\r' '\n'` |
| 止める | `docker rm -f molmoact2_molmoact2_lora_v1` |

**1 つずつ打つなら次の 2 行である。**2 行目は、コンテナを起こして 20 歩を見届けるまで前面に居るだけなので、**起きたあとに ssh が切れても学習は続く。**

```bash
# 試し走り（載るか・1 歩の秒数・メモリの山。出力は残さない）
TRIAL=1 BATCH=2 bash solution/train/local_molmoact2_lora.sh

# 本番
RUN=molmoact2_lora_v1 BATCH=4 STEPS=40000 SAVE_FREQ=8000 \
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

### 3.5.1 設定の確かめ方（走り出しの 1 画面で全部見える）

**`lerobot-train` は起動時に設定を全部書き出す。**次の 7 つを見れば、狙った走りかどうかが分かる。

| 見るもの | 正しい値 | どこに出るか |
|---|---|---|
| 学習データ | `local/libero_combined_bowl5_initpose_l90clean`・`root: /data`・21,642 本・3,431,137 コマ | `'repo_id'` と `dataset.num_episodes` |
| 列名の読み替え | `front → image`・`wrist → wrist_image` | `'rename_map'` |
| 出発点 | `pretrained_path: /init`・`checkpoint_path: /init/molmoact2_base_small` | `'policy'` の中 |
| **動かす範囲** | `enable_lora_vlm: True`・`enable_lora_action_expert: False`・`train_action_expert_only: False` | 同上 |
| **学習するパラメータ数** | **727,296,544（全体 5,591,928,304 の 13%）**＝ action expert 全部 ＋ VLM の LoRA | `num_learnable_params` |
| **画像を崩す細工** | **`image_transforms.enable: True`**（既定は False。9/17 にスクリプトの既定を True にした） | `'dataset'` の中 |
| 学習率 | 本体 1e-5 ／ ViT 5e-6 ／ つなぎ 5e-6 ／ action expert 5e-5 | `'policy'` の `optimizer_*` |

**`Effective batch size` と `steps` も同じ画面に出る。**batch 4・40,000 歩なら 16 万枚を見る。

## 3.6. 止める・続ける

**止め方。**コンテナを消せば止まる。**保存の直後（`step:8K` のような区切りのすぐ後）を選ぶこと**
——保存の最中に消すと、書きかけのチェックポイントが `checkpoints/last` になる。

```bash
docker logs --tail 3 molmoact2_molmoact2_lora_v1 | tr '\r' '\n'   # いまの歩数を見る
docker rm -f molmoact2_molmoact2_lora_v1
```

**続け方。**同じ `RUN` に `RESUME=1` を付けて打つ。`checkpoints/last` から、歩数・optimizer・
学習率の刻みを引き継いで続く。

```bash
RESUME=1 RUN=molmoact2_lora_v1 bash solution/train/local_molmoact2_lora.sh
```

**`RESUME=1` を付けると、スクリプトが渡す引数そのものが変わる。**
`--policy.path`（出発点の重み）や `--steps`・学習率の代わりに、
`--config_path=<RUN>/checkpoints/last/pretrained_model/train_config.json --resume=true` の
2 つだけを渡す。**だから `BATCH`・`STEPS`・`LR`・`SCOPE` を渡しても効かない**——
設定は保存済みの `train_config.json` が全部持っている。
**歩数や batch を変えたいなら、続きではなく新しい `RUN` で始めること。**

| 覚えておくこと | 中身 |
|---|---|
| **戻る先は最後の保存** | `SAVE_FREQ` ごとにしか保存されない。8,000 歩ごとなら、最悪 8,000 歩（約 2.8 時間）巻き戻る |
| チェックポイント 1 個 | **14 GB**（9/16 実測。11 GB ではない） |
| 保存が 1 個も無いうちは続けられない | `RESUME=1` は `checkpoints/last` が要る |

## 3.7. 評価のために HF へ上げる（上げるのは `pretrained_model/` だけ）

**チェックポイント 1 個は 14 GB あるが、評価に要るのは 11 GB の `pretrained_model/` だけである。**

| 中身 | 大きさ | 上げるか |
|---|---:|---|
| `pretrained_model/`（7 ファイル） | **11 GB** | **上げる。**これだけで評価できる |
| `training_state/`（optimizer・rng・歩数） | 3.3 GB | **要らない。**`RESUME=1` で続けるときにしか使わない |

**別に `molmoact2_base_small/`（約 12 MB）が評価側で要る。**前処理（トークナイザ・
`norm_stats.json`）を組み立てるのに読むためで、`run_molmoact2_stage0.sh` は
`$WEIGHTS/molmoact2_base_small/` があることを確かめてから動く。同じリポジトリに入れておくと、
評価する機械では 1 回降ろすだけで済む。

```bash
CKPT=work/train/molmoact2_lora_v1_1/checkpoints/016000
hf upload siruku6/molmoact2_lora_v1_1 "$CKPT/pretrained_model" 016000 --repo-type model
hf upload siruku6/molmoact2_lora_v1_1 \
    work/artifacts/molmoact2_libero_lerobot/molmoact2_base_small molmoact2_base_small --repo-type model

# **上がったことを確かめる**（終了コード 0 は「上がった」と同じではない）
python3 solution/train/verify_upload.py --repo siruku6/molmoact2_lora_v1_1 --step 016000 \
    --local "$CKPT/pretrained_model"
```

**評価する側で要るもの。**

- **`peft`。**重みは PEFT の名前（`…lora_A.default.weight` など 1,899 鍵のうち 604 個）で
  保存されていて、`config.json` の `enable_lora_vlm` が `True` のまま読まれる。
  `parc2026-train-molmoact2` には入っている。
- **直した `modeling_molmoact2.py`。**素の重み 21.8 GB を読みに行かせないため
  （`run_molmoact2_stage0.sh` が読み取り専用で被せる）。
- `config.json` の `checkpoint_path` は学習したときのコンテナ内の道（`/init/molmoact2_base_small`）
  のまま残るが、**`policy_server.py` が読み込み時に上書きする**ので直さなくてよい。

**手元で評価するなら上げなくてよい。**`WEIGHTS` にこの `pretrained_model/` を指し、
その中に `molmoact2_base_small/` を置けばそのまま回る。

## 4. Docker のために足していること

**学習そのものは [jaist_molmoact2_lora.sh](../../solution/train/jaist_molmoact2_lora.sh) を呼んでいる。**
[local_molmoact2_lora.sh](../../solution/train/local_molmoact2_lora.sh) が足すのは Docker の口だけである。

| 足していること | なぜ |
|---|---|
| `VENV=none` | イメージの中に venv は無い（lerobot は最初から入っている） |
| 直し 2 ファイルの bind-mount | `--user` で非 root にすると `/opt/lerobot` を上書きできない。読み取り専用で被せると、呼ばれた側は目印を見て「当たっている」と判断して当てない |
| `FOREGROUND=1` | 切り離すと、スクリプトが終わった時点でコンテナごと落ちる |
| `--shm-size=16g` | 読み手の共有メモリ。既定の 64 MB では 1 歩も進まずに落ちる |

**ログは `log_freq` ごとの 1 行だけになる。**tqdm の進捗バーは `\r` で上書きする前提で書かれていて、
端末でない先（`docker run -d` の中）では**上書きが効かず 1 歩 1 行として溜まる**——
40,000 歩なら 4 万行である。`lerobot_patches/lerobot_train.py` が、端末でないときは
進捗バーを出さないようにしてある（9/17）。**進捗は `step:… loss:…` の行で見る。**

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
