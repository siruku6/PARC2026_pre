# 025 逆向きデータの現在地と次の一手（9/8）

> **状態: 判断待ち** ／ 更新: 2026-09-08
>
> **この 1 枚で足りるように書く。**9/7〜9/8 に測ったこと、決まったこと、
> いま動いているもの、次に打つものを並べる。
> 実験そのものの詳細は [reference/19](../../reference/19_weights_catalog.md) の
> 「9/7 の `pi05_proto20`」にある（依頼 024 は役目を終えたので 9/8 に消した）。

## 1. 測って分かったこと（すべて実測）

### 1.1 逆向きデータを 2 割混ぜた学習（`proto20`）は、**t3 を動かさず元の性能を落とした**

| | track3（4 タスク × 2 本） | track2 BASE12（12 本） | **track2 VAL72（72 本）** |
|---|---|---|---|
| **`rp3000`**（出発点） | 0.000 | **0.9167** | **0.8056** |
| `all111`（**逆向きなし**で同じ 2,000 歩） | — | 0.7500 | — |
| **`proto20`**（逆向き 2 割） | **0.000** | 0.5833 | **0.5833** |
| `trial1500/001500`（本選最良の重み） | — | — | 0.7500 |

**12 本と 72 本で同じ 0.5833 が出た。**標本の少なさによる見かけではない。

### 1.2 犯人は 2 つで、ほぼ折半

| 区間 | 落ちた分 | 原因 |
|---|---|---|
| `rp3000` → `all111` | **2 タスク** | **逆向きデータと無関係。**「2,000 歩の追加学習」自体 |
| `all111` → `proto20` | **2 タスク** | **逆向きデータ** |

**落ちるタスクは重ならない。**`all111` は `open_the_middle_drawer` と `tomato_sauce`、
`proto20` は `turn_on_the_stove`・`black_bowl_between`・`chocolate_pudding`・`tomato_sauce`。
**特定の物の取り違えというより全般的なドリフトである。**

**`all111` は歩数も激増した**——`put_the_bowl_on_the_plate` が 79 → **295 歩**（打ち切り 300 の寸前）。

### 1.3 学習中は**全 2,000 歩が毎歩クリップ**されていた

| | `all111`（逆向きなし） | **`proto20`（逆向き 2 割）** |
|---|---|---|
| loss 最初 → 最後 | 0.382 → 0.301 | 0.973 → 0.427 |
| **`grdn` 平均** | **0.891** | **1.971** |
| **clip 1.0 を超えた歩** | **4%** | **100%** |

### 1.4 原因は**こちらのデータの回転指令が配布の 4〜5 倍広いこと**

| `action` の次元 | 配布の std | **こちらの std** | 倍率 |
|---|---|---|---|
| 0〜2（並進） | 0.329 / 0.379 / 0.444 | 0.340 / 0.570 / 0.490 | 1.03〜1.50 |
| **3〜5（回転）** | **0.040 / 0.062 / 0.078** | **0.152 / 0.150 / 0.380** | **3.8 / 2.4 / 4.9** |

配布の統計で正規化すると、**\|z\|>10 のコマが 15.1%**（最大 24.8）。
**その 52.6% がエピソードの先頭 10% に集中**している——**復路ではなく、
こちらが合成した「掴み姿勢へ手首を回す接近区間」が原因**である
（[reference/30 の 8 節](../../reference/30_reverse_grasp_pose.md)）。

### 1.5 学習する範囲を絞っても、勾配は収まらなかった

`proto21`（**action expert だけ**・他は `proto20` と同一）を 120 歩だけ回した。

| 20 歩目 | loss | `grdn` | 1 歩の秒数 |
|---|---|---|---|
| `proto20`（全パラメータ） | 1.102 | **3.093** | 6.85 秒 |
| **`proto21`（action expert だけ）** | 1.113 | **2.558** | **5.768 秒** |

**2 つ分かった。**

1. **勾配を暴れさせているのは「学習する範囲」ではなくデータである。**693M に絞っても clip の 2.5 倍。
2. **action expert 限定は 16% しか速くない。**VLM の順伝播が支配的で、
   「軽いから速い」は成り立たない。**20,000 歩 ＝ 32 時間**だった。

### 1.6 t3 の挙動は変わっている——**暴れなくなった**

操作対象でない物が動いた量（`object_disp.jsonl`）。

| タスク | `rp3000` | `proto20` |
|---|---|---|
| 缶 ep0 | cream_cheese **466 mm** | tomato_sauce **13 mm** |
| 本 ep0 | yellow_book **391 mm** | yellow_book **4 mm** |
| 赤マグ ep0 | white_yellow_mug 190 mm | porcelain **18 mm** |

**動画でも確認した。**`rp3000` は棚やマグに突っ込んで薙ぎ倒す。`proto20` は
高い位置でホバリングして、ほとんど触らない。**どちらも掴みに行けていない。**
**「そっと動く」ことは学んだ。方向は間違っていない。**

## 2. 決まったこと

| # | 決めたこと | 理由 |
|---|---|---|
| 1 | **学習は Blackwell では回さない** | 残り 35〜40 時間を 1 本で使い切る規模になった |
| 2 | **JAIST で回す** | 時間無制限。**20 GB には action expert 限定しか載らない**が、いま欲しいのはそれ |
| 3 | **このマシン（RTX6000）では学習しない** | cc 7.5 で **bf16 非対応**。fp32 は約 25 GB 要って 24 GB に入らない。**EGL が効く唯一の評価機なので温存する** |
| 4 | **こちらのデータは 1 コマ 1 回見せれば足りる** | `chunk_size` 50 なので、**ある行動 1 歩が 1 度も教わらない確率は 10⁻²²** |
| 6 | **回転の外れ値と置き方は直した**（9/8） | `ROT_LIMIT=0.375` ＋ 本の `LEVEL` を外す ＋ `SETTLE=80`。**\|z\|>10 が 15.2% → 0.0%**・落下 40 mm → 1.0 mm・**コマ数はむしろ 173 → 152 に短くなった。**データは `work/artifacts/jaist_mix_v2_x8`（検査 7 項目通過） |
| 5 | **配布側は `libero_plus` に戻す**（配布 tar ではなく） | **`rp3000` はこれで学習された**ので、配布側が「すでに学習済み」＝ action expert をほとんど押さない（`stage4_b16_24k` の実測）。**据え置く統計も `rp3000` 同梱の値そのものになる。**配布 tar の 71 個の未学習の指示文が `all111` の退化の筋である |

## 2.5 【9/8】系譜 6 点を測って分かったこと —— **逆向きデータは効いていた**

**ロールアウトを回さずに「どの物へ手を伸ばすか」を測った**
（やり方は [knowledge/20](../../knowledge/20_probe_object_choice_without_rollout.md)）。
**測りの妥当性は、100% 通る 2 課題で正解が cos +0.99 の 1 位になることで確かめてある。**

| T3 課題（正解の順位） | `base` | `stage1` | `rp3000` | `trial1500` | **`proto20`** | **`t3rev`** |
|---|---|---|---|---|---|---|
| 黄本 | 3/3 | 3/3 | 3/3 | 3/3 | **1/3** | **1/3** |
| 缶 | **2/6** | 5/6 | 5/6 | 3/6 | **1/6** | **1/6** |
| 赤マグ | **2/5** | 3/5 | 3/5 | 3/5 | 5/5 | 3/5 |

| # | 分かったこと | 次にどう効くか |
|---|---|---|
| 1 | **逆向きデータは対象物の選択を実際に直した**（黄本・缶が 1 位に） | **データの向きは正しい。**`success_rate` 0.000 がこの獲得を隠していた |
| 2 | **私たちの最初の学習が T3 の選択を潰していた**（缶が `base` の 2 位 → 5 位） | **`base` から弱かったのではない。**回復の余地がある |
| 3 | **`t3rev` は行き過ぎ**——掴む挙動を失い（閉じる 0〜1/50）、**100% 通る課題で 1 位を失った** | **「expert だけで長く」より「全パラメータで短く」。**`proto20` は対照を 1 つも壊していない |
| 4 | **色をほとんど見ていない**（行動列の変化 0.03〜0.35 対 名前 0.06〜1.11） | **`base` から弱く、どの学習でも増えていない。**赤マグ（色だけが手がかり）はこれが理由で解けない |

## 3. まだ決まっていないこと

| # | 判断 | 材料 |
|---|---|---|
| **A'** | **次の 1 本を「全パラメータで短く」にするか** | 上の 2.5 の 3。**Blackwell の残りは 34 時間**で、v2 を全パラメータ 1,900 歩なら **3 時間 37 分**。`t3rev` の失敗は歩数と範囲の取り方に見える |
| **D** | **赤マグ（色だけが手がかり）をどうするか** | 色への反応は `base` から弱い。**行動データを足しても直らない見込み。**指示文の与え方を変えるか、色で見分ける学習データを作るか |
| **B** | 本番の出発点を `rp3000` にするか `trial1500/001500` にするか | 本選最良は `trial1500`（Track1 0.492）だが、**手元の VAL72 では `rp3000` が上**（0.8056 対 0.7500）。代理と本番がずれている |

**C（`all111` の扱い）は 9/8 に片付けた。**手元の指標で `rp3000` に劣ることを
[reference/19](../../reference/19_weights_catalog.md) の「9/7 の `pi05_proto20`」に書いた。

## 4. JAIST で 1 本回す手順

**打つのは 5 つだけである。**データを組むところは
`solution/train/jaist_dataset_build.sh` に、起こすところは
`solution/train/jaist_launch.sh` に畳んである。**どちらも何度打ってもよく、
数が 1 つでも違えば原因と次の一手を出して止まる。**

### 4.0 何を作るのか

| | 本数 | コマ数 | 課題 |
|---|---|---|---|
| `libero_plus` を課題ごとに 75 本へ間引いたもの | **3,000** | **501,403** | 40 |
| 逆向き 389 本 × 7 | **2,723** | **474,425** | 3 |
| **混ぜたもの** | **5,723** | **975,828** | **43** |

**逆向きの割合 48.6%。**正規化統計は **`rp3000` 同梱の normalizer に据え置く**
（＝`rp3000` が学習した尺度そのもの。配布側の見え方が 1 ビットも変わらない）。

**歩数は 8,700。**割合 48.6% で 8,700 歩 × batch 16 ＝ 139,200 サンプル、
うち逆向きに落ちるのが 67,700 回で、**こちらの 67,775 コマを 1 コマ 1 回**見る量である
（根拠は [reference/29 の 7 節](../../reference/29_training_data_requirements.md)）。

### 4.1 このマシン（RTX6000）で組む —— 約 40 分

```bash
cd /home/ubuntu/src/PARC2026_pre

# ① 元データ（15 GB・取得済みなら飛ばされる）
hf download lerobot/libero_plus --repo-type dataset \
    --local-dir work/artifacts/libero_plus

# ② 間引く → 7 倍にする → 混ぜる → 据え置く（1 本で通る）
bash solution/train/jaist_dataset_build.sh
```

**最後に「HF へ上げる 1 行」が出る。**確かめてから手で打つ。

```bash
hf upload siruku6/jaist_mix work/artifacts/jaist_mix_75_x7 . \
    --repo-type dataset --private
```

| 見るもの | 通す条件 |
|---|---|
| 間引いた本数 | **3000 501403 40** |
| 7 倍した本数 | **2723 474425 3** |
| 混ぜた本数 | **5723 975828 43** ／ **割合 48.6%** |
| 据え置き | `meta/stats.json.bak` が出来ていること |

### 4.2 JAIST で回す —— 約 3.4 時間

```bash
# ③ 落とす（データ 6.6 GB ＋ 出発点の重み 8.8 GB）
hf download siruku6/jaist_mix --repo-type dataset --local-dir ~/data/jaist_mix
hf download siruku6/pi05_full_runpod --local-dir ~/weights/rp3000

# ④ 起こす（action expert 限定は script が外せないようにしてある）
cd ~/src/PARC2026_pre
RUN=pi05_t3rev DATASET=~/data/jaist_mix WEIGHTS=~/weights/rp3000 \
    STEPS=8700 SAVE_FREQ=2900 \
    bash solution/train/jaist_launch.sh
```

**`jaist_launch.sh` は最初の 20 歩まで待って、次の 2 つを突き合わせてから離席させる。**

| 見るもの | 通す条件 | 違ったら |
|---|---|---|
| `num_learnable_params` | **693422112** | 全パラメータになっている。20 GB には載らない |
| `dataset.num_episodes` | **5723** | 混ざっていない／落とし切れていない |
| `loss`・`grdn`（20 歩目） | **NaN でない**・`grdn` が **100 未満** | 発散。`stats.json.bak` を戻して打ち直す |

**`grdn` は 20 歩目で必ず見る。**`proto20` はここが 3.093 で、
**全 2,000 歩がクリップに当たり続けた。**速度制限が効いていれば 1 前後に下がるはずで、
**下がらなければ原因は回転以外にある**（＝この 1 本の主目的である）。

```bash
# ⑤ 見張る
docker logs --tail 2 parc-pi05_t3rev 2>&1 | tr '\r' '\n' | grep step:
```

### 4.3 終わったら測る

**このマシンへ持ち帰って評価する**（EGL が効く唯一の機械である）。

```bash
bash solution/tools/eval/run_final_eval.sh track3   # 4 課題 × 2 本
bash solution/tools/eval/run_final_eval.sh track2   # 退化していないかの確認
```

比べる相手は **`rp3000`（track3 0.000・VAL72 0.8056）**と
**`proto20`（track3 0.000・VAL72 0.5833）**である。

| 出た結果 | 読み |
|---|---|
| track3 が動き、track2 が落ちない | **データの向きも量も正しい。**本番の規模へ進む |
| track3 は 0.000 だが **track2 が落ちない** | **退化の原因は回転の外れ値だった。**次は量を増やす |
| track3 は 0.000 で **track2 も落ちる** | **原因は回転以外。**出発点か、追加学習そのものを疑う |

## 4.4 【9/8】2 本目——**逆向きデータを直した版**（v2）

**1 本目との違いは逆向きデータの質だけである。**配布側・出発点・据え置き・
「1 コマ 1 回」の設計はすべて同じにしてある。

| | 1 本目（`pi05_t3rev`） | **2 本目（v2）** |
|---|---|---|
| 逆向きデータ | 9/6 の版・**389 本 / 67,775 コマ** | **修正版・397 本 / 61,430 コマ** |
| **\|z\|>10 のコマ** | **15.2%** | **0.0%** |
| \|z\| の最大 | 26.1 | **9.8**（＝配布データ自身の裾と同値） |
| 落下（本 / 缶） | 40 mm / 37 mm | **1.0 mm / 2.9 mm** |
| 配布側 | `libero_plus` 3,000 本 | **同じ** |
| 逆向きの割合 | 48.6%（×7） | **49.5%（×8）** |
| 歩数 | 8,700 | **7,800** |
| 出発点・据え置き | `rp3000`・差 0.000e+00 | **同じ** |
| データセット | `jaist_mix_75_x7`（5.9 GB） | **`jaist_mix_v2_x8`（6.0 GB）** |

**何を直したのか**（実装は `replay_video_probe.py` と
[reference/30 の 8 節](../../reference/30_reverse_grasp_pose.md)・
[31 の 8.1 節](../../reference/31_reverse_place_without_contact.md)）。

| 直したもの | 効いたこと |
|---|---|
| **`ROT_LIMIT=0.375`**（成分ごとに `clip` せず、ベクトル全体を縮めて回転軸を保つ） | \|z\| 26.1 → 9.8。**缶のコマ数 155 → 138・接近誤差 4.11 → 3.12 mm**（遅くしたのに速くなった） |
| **本の `LEVEL` を外す** | 本の落下 40 → 1.0 mm。**薄い物を水平へ直す動きが持ち上げになっていた** |
| **`SETTLE=80`** | 缶の落下 37 → 2.9 mm。**既定 20 では下ろし切る前にコマを使い果たし、空中で放していた** |

**副産物**——赤マグが 139 → **147 本**（合計 389 → **397 本**）、コマ数の中央 173 → **152**。

### 4.4.1 データはもう出来ている

`work/artifacts/jaist_mix_v2_x8`（**6176 本 / 992,843 コマ / 43 課題**・6.0 GB）。
**`verify_dataset.py` の 7 項目すべて通過済み**（据え置きは `rp3000` と差 **0.000e+00**）。
作り直すなら:

```bash
COPIES=8 PRE=revv2_x REV_DIR=work/artifacts/reverse_lerobot_v2 MIX=jaist_mix_v2_x8 \
    bash solution/train/jaist_dataset_build.sh
```

### 4.4.2 打つのは 4 つ

```bash
# ① このマシンから上げる（6.0 GB）
hf upload siruku6/jaist_mix_v2_x8 work/artifacts/jaist_mix_v2_x8 . \
    --repo-type dataset --private

# ② JAIST で落とす
hf download siruku6/jaist_mix_v2_x8 --repo-type dataset \
    --local-dir ~/data/jaist_mix_v2_x8

# ③ 直した script を取る（verify_dataset.py と、上書きを防いだ build）
cd ~/src/PARC2026_pre && git pull

# ④ 起こす（約 3 時間）
RUN=pi05_mix_v2_x8 DATASET=~/data/jaist_mix_v2_x8 STEPS=7800 SAVE_FREQ=2600 \
    bash solution/train/jaist_launch.sh
```

**`SAVE_FREQ=2600` を必ず渡すこと。**既定の 2000 だとチェックポイントが 4 個＋最終で
**45 GB** になり、JAIST の空き（22 GB）に入らない。2600 なら 2 個＋最終で 27 GB。
**1 本目は渡しておらず、既定の 2000 が効いている**——空きを空けるには、
`008700` を上げ切ってから `pi05_t3rev` のチェックポイントを消すこと。

### 4.4.3 20 歩目に見るもの——**`grdn` が答えである**

| 見るもの | 通す条件 | 1 本目の実測 |
|---|---|---|
| `num_learnable_params` | **693422112** | 693422112 ✓ |
| `dataset.num_episodes` | **6176** | 5723 ✓ |
| `mem_gb` | 20 GB 未満 | **16.27** ✓ |
| **`grdn`** | **1 前後へ落ちること** | **2.5〜7.8（クリップに当たり続けた）** |

**`grdn` がこの一連の作業の答えである。**1 本目は最後まで 4〜7 で、
**勾配クリップ（上限 1.0）に全歩当たっていた。**回転の外れ値を消したのだから、
**v2 で 1 前後に落ちなければ原因は回転以外にある。**

### 4.4.4 終わったら持ち帰るもの——**2 つだけ・8.8 GB**

| # | 何 | 大きさ | 要る理由 |
|---|---|---|---|
| 1 | `checkpoints/007800/pretrained_model/` | **8.8 GB** | 評価に必要。正規化の 4 ファイルも入っている |
| 2 | **学習ログ** | 数 MB | **loss と `grdn` の推移はチェックポイントに残らず、やり直しでも作れない** |

**持ち帰らないもの。**`training_state/`（optimizer・約 17 GB。再開のときだけ要る）と
**`vlm/`**（17 MB。**チェックポイントには付かないが `rp3000` から手元でコピーできる**）。

```bash
# JAIST 側
docker logs parc-pi05_mix_v2_x8 > ~/pi05_mix_v2_x8.log 2>&1   # コンテナを消す前に
CK=~/src/PARC2026_pre/work/train/pi05_mix_v2_x8/checkpoints/007800
hf upload siruku6/pi05_mix_v2_x8 "$CK/pretrained_model" 007800 --repo-type model --private
hf upload siruku6/pi05_mix_v2_x8 ~/pi05_mix_v2_x8.log logs/train.log --repo-type model --private

# このマシン側
hf download siruku6/pi05_mix_v2_x8 --include '007800/*' --local-dir work/artifacts/_v2_dl
W=work/artifacts/pi05_v2_007800
mkdir -p $W && cp -r work/artifacts/_v2_dl/007800/* $W/
cp -r work/artifacts/pi05_rp3000/vlm $W/      # **無いと評価は 5 秒で落ちる**

WEIGHTS=$PWD/$W LABEL=v2_7800 MUJOCO_GL=egl RECORD=1 \
    bash solution/tools/eval/run_final_eval.sh track3
WEIGHTS=$PWD/$W LABEL=v2_7800 MUJOCO_GL=egl TASKS_CSV=compe/val/TRACK2_BASE12.csv \
    TRACK_EPISODES=1 bash solution/tools/eval/run_final_eval.sh track2
```

**`RECORD=1` を必ず渡すこと。**既定は 0 で、**渡さないと動画が 1 本も残らない**
（1 本目で渡し忘れて測り直した）。

## 5. 道具（全部このリポジトリに在る）

| 何 | どこ |
|---|---|
| Blackwell の段取り（何度打ってもよい） | `solution/train/blackwell_setup.sh` |
| Blackwell の起動（前提 7 つを確かめる） | `solution/train/blackwell_launch.sh` |
| **JAIST の起動（docker 経由・action expert 固定）** | **`solution/train/jaist_launch.sh`** |
| チェックポイントを上げて**確かめてから印を付ける** | `solution/train/upload_ckpt.sh` ＋ `verify_upload.py` |
| **正規化統計の検算と据え置き** | `solution/tools/track3_reversal/check_merge_stats.py` |
| **課題ごとに均等に間引く** | `npz_to_lerobot.py subset` ＋ `run_lerobot_subset.sh` |
| 配布 tar の動画を 3 桁に見せる | `solution/tools/track3_reversal/link_video_index_width.py` |

**HF に上げてある**（`siruku6/pi05_code` と `siruku6/reverse_lerobot_389`）。

## 6. 物の置き場

| 何 | どこ |
|---|---|
| 逆向きデータ 389 本 | `work/probe_017/reverse_lerobot_all` ／ HF `siruku6/reverse_lerobot_389` |
| **配布 `libero_combined_20hz`** | **このマシンの `/home/ubuntu/src/train_data/`**（tar 21 GB ＋ 展開済み）・Blackwell の `~/dataset/lerobot/` |
| `libero_plus` | HF `lerobot/libero_plus`（15.8 GB・公開） |
| `rp3000` | `work/artifacts/pi05_rp3000` ／ HF `siruku6/pi05_full_runpod` |
| `proto20` の重み | `work/artifacts/pi05_proto20_002000` ／ HF `siruku6/pi05_proto20`（4 個） |
| `all111` の重み | `work/artifacts/pi05_all111_002000` ／ HF `siruku6/pi05_all111` |
| 評価の結果 | `work/final_eval/0908_*` |

## 7. 踏んだ穴（機械の癖）

**Blackwell と道具まわりの穴は
[reference/27 の 5 節](../../reference/27_blackwell_environment.md#5-踏んだ罠実際に-1-度は踏んだもの)へ移した**
（サーバは UI からしか止まらない・`/opt/dlami/nvme` は消える・6 桁の動画・`align` の軸名など）。
**ここには機械の当たり外れだけを残す。**

| 症状 | 対処 |
|---|---|
| **JAIST は 20 GB。**全パラメータ学習は載らない | action expert 限定のみ |
| **このマシンは cc 7.5。**bf16 非対応 | 学習には使えない。**EGL が効く唯一の評価機** |
| **Blackwell は残り 35〜40 時間** | 1 本 3 時間 49 分。使い切る前に本番の 1 本を決める |

## 8. 消してよい一時物（このマシン・合計 3.5 GB）

**空きは 188 GB あるので急がない。**`rm -rf` は私からは通らないのでユーザが消す。

| どこ | 大きさ | いつの何か |
|---|---|---|
| `work/probe_024/proto_x2` ／ `proto_x4` | **2.4 GB** | 9/7 のコピー検査（`proto_x1` は残す） |
| `work/probe_018/cost_npz` | 613 MB | 018 の所要計測 |
| `work/probe_017/reverse_lerobot_all_tmp1` ／ `reverse_lerobot_bs` | 510 MB | 017 の中間物 |
