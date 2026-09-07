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
| 5 | **配布側は `libero_plus` に戻す**（配布 tar ではなく） | **`rp3000` はこれで学習された**ので、配布側が「すでに学習済み」＝ action expert をほとんど押さない（`stage4_b16_24k` の実測）。**据え置く統計も `rp3000` 同梱の値そのものになる。**配布 tar の 71 個の未学習の指示文が `all111` の退化の筋である |

## 3. まだ決まっていないこと

| # | 判断 | 材料 |
|---|---|---|
| **A** | **回転の外れ値を直してから回すか、いまのデータで 1 本回すか** | JAIST は時間無制限で **3.3 時間**なので、先に 1 本回して確かめる道もある。直すなら接近の回転に速度制限をかける（**コマ数の上限 300 と正面衝突する**） |
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
