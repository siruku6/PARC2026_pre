# 29. 学習データ 1 本に何が要るか（自分で作るとき）

**逆向きデータのように自分でエピソードを作るとき、何を揃えれば学習に載るかを
1 か所に置く。**中身の規約は [25 の 8 節](25_dataset_inventory.md) にあるが、
あちらは「配布物を数えた結果」で、**こちらは「作る側が満たすべき条件」**である。

| 節 | 中身 |
|---|---|
| 1 | **1 本に必要なもの（5 つ）** |
| 2 | 出来上がりの形（LeRobot v3.0 のファイル構成） |
| 3 | **中継物の npz と、既にある変換器** |
| 4 | **指示文は bddl から取れる**（書かなくてよい） |
| 5 | 作るときに踏む穴 |
| **6** | **混ぜると正規化統計が変わる**——検算と据え置きのしかた |
| 5 | 作るときに踏む穴 |

## 1. 1 本に必要なもの

**5 つある。**このうち 1 つでも欠けると学習に載らない。

| # | 中身 | 形 | どこから取るか |
|---|---|---|---|
| 1 | `action` | **(T, 7)** float32 | シムへ送った指令そのもの。`[並進(3), 回転(3), gripper(1)]` |
| 2 | `observation.state` | **(T, 8)** float32 | `[eef_pos(3), 軸角(3), 指 qpos(2)]`・**生スケール**。`obs["robot0_eef_pos"]`・`robot0_eef_quat` を軸角に直したもの・`robot0_gripper_qpos` |
| 3 | `observation.images.front` | (T, 256, 256, 3) uint8 | **`agentview`** カメラ |
| 4 | `observation.images.wrist` | (T, 256, 256, 3) uint8 | **`robot0_eye_in_hand`** カメラ |
| 5 | 指示文 | 文字列 1 つ | bddl の `:language`（4 節） |

**間違えやすいのは 2 つ。**

- **「軌跡」は 2 種類あって別物である。**`action`（送った指令）と
  `observation.state`（そのときのロボットの自己受容）の**両方**が要る。
  片方から他方は作れない。
- **カメラは 2 本である。**俯瞰だけでは足りない。提出物の `policy_server.py` も
  `robot0_eye_in_hand_image` を観測に使っている。

正規化統計（`meta/stats.json`）は `LeRobotDataset` が自動で作るので、
自分で用意する必要は無い。

## 2. 出来上がりの形

`lerobot/libero_plus` と同じ **LeRobot v3.0**。fps 20・`robot_type` は `panda`。

```
<出力>/
├── data/chunk-000/file-000.parquet        # 1 コマ 1 行
├── videos/observation.images.front/chunk-000/file-000000.mp4
├── videos/observation.images.wrist/chunk-000/file-000000.mp4
└── meta/
    ├── info.json          # features の宣言。**混ぜる相手と一致していること**
    ├── tasks.parquet      # 指示文の一覧（1 行 1 文字列）
    ├── episodes/…parquet  # 本ごとの長さ・動画の位置・統計
    └── stats.json         # 正規化統計
```

parquet の 1 行は `observation.state`・`action`・`timestamp`・`frame_index`・
`episode_index`・`index`・`task_index`。**画像は parquet に入らず動画になる。**

## 3. 中継物の npz と、既にある変換器

**形式を手で書き起こさないこと。**列の並びや `meta/` の中身がずれると、
混ぜたときに読めなくなる。既に道具がある。

| 道具 | 役目 |
|---|---|
| [`solution/tools/scripted_demo/scripted_demo.py`](../../solution/tools/scripted_demo/scripted_demo.py) の `SavedEpisode` | **npz の取り決めの唯一の置き場**。写しを作らない |
| [`solution/tools/scripted_demo/npz_to_lerobot.py`](../../solution/tools/scripted_demo/npz_to_lerobot.py) | `build`（npz → データセット）・`inspect`（学習が読めるか）・`align`（列を揃える）・`merge`（既存データへ足す） |

npz の鍵は次のとおり。**この形で書けば、変換から先は既存のコードがそのまま動く。**

| 鍵 | 中身 |
|---|---|
| `state` | (T, 8) float32 |
| `action` | (T, 7) float32 |
| `front` / `wrist` | (T, 256, 256, 3) uint8 |
| `instruction` | 文字列 |
| `success` | bool。**`build` は失敗した回を既定で捨てる**（`--keep-failures` で残せる） |
| `steps` | int |

`build` は**画像に触る前に `success` を見る**ので、失敗した回のために
100 MB 級の伸長を払わずに済む。

**シムを動かす側に lerobot を入れる必要は無い。**`scripted_demo` は lerobot を
import しない。npz を書くところまでをコンテナで行い、変換は lerobot がある側で行う。

## 4. 指示文は bddl から取れる——**書かなくてよい**

配布 T3 の bddl に `:language` が入っている。`env.language_instruction` で取れる。

| タスク | `:language` |
|---|---|
| yellow book | take the yellow book off the top of the shelf and put it back on the table on the left |
| alphabet soup | take the alphabet soup out of the tray and place it on the table in front of the ketchup |
| red mug | take the red mug off the left plate and put it back on the table behind the other two mugs |
| two mugs | take the white mug off the left plate and the yellow and white mug off the right plate, then place the white mug to the left of the red mug and the yellow and white mug to the right of the red mug |

**ファイル名より長い。**ファイル名から起こすと本物と違う文になるので、
必ず bddl から読むこと。

自分で逆向き bddl を作る場合の指示文の型は
[`solution/tools/probe/_016_make_reverse.py`](../../solution/tools/probe/_016_make_reverse.py)
が持っている（`take A off B and put it back` の形）。

## 5. 作るときに踏む穴

| 穴 | 中身 |
|---|---|
| **`state` と `action` の長さを合わせる** | `state[t]` は **`action[t]` を撃つ前**の姿勢である。素直に記録すると state が 1 本長くなる（初期姿勢＋各ステップ後）ので、**撃つ前に記録する**か末尾を落とす |
| **gripper の符号化が 2 系統ある** | `{0,1}` 系と `{-1,+1}` 系が混在している。自分で作るぶんは robosuite 規約（`-1`=開・`+1`=閉）に揃える。詳細と復号は [25 の 8.7 節](25_dataset_inventory.md) |
| **混ぜる前に `align` する** | lerobot の版が違うと列の書き方が食い違う。`merge` の前に `align` を通す |
| **1 タスクだけで学習しない** | 手本が 1 タスクぶんしか無い追加学習で、本番 0.1409 → 0.016 に崩れた実績がある。`build` は確認用、**提出候補を作るのは `merge`** |
| **シム側の落とし穴** | 状態復元で復らないもの・fixture の配置・predicate の呼び方は [28_mppi_data_regeneration_survey.md](28_mppi_data_regeneration_survey.md) の 5.1〜5.2 節。env をまたぐ姿勢の写し替えは [28_track3_contents.md](28_track3_contents.md) の 5.6 節 |

## 6. 配布データに混ぜると、**配布データの正規化まで変わる**

**2026-09-07 に、逆向き 389 本を配布データへ混ぜる前に確かめた事実である。**

### 6.1 学習が読むのは「データセットの統計」で、重みの中の値ではない

| 事実 | どこで確かめたか |
|---|---|
| pi05 は `observation.state` と `action` を **mean/std で正規化する** | `rp3000` と `all111` の `config.json` の `normalization_mapping` が `{ACTION: MEAN_STD, STATE: MEAN_STD, VISUAL: IDENTITY}`。**`configuration_pi05.py` の既定は `QUANTILES` だが、重みの `config.json` が勝つ** |
| **学習は `dataset.meta.stats` を使う** | `lerobot_train.py` 315〜332 行が `preprocessor_overrides["normalizer_processor"]["stats"]` を `dataset.meta.stats` で**上書きする**（`--resume` でないとき） |
| 推論は**チェックポイント同梱の preprocessor** を読む | 学習が書き出した値がそのまま使われるので、**学習側で決めた統計と食い違わない** |
| 画像は正規化されない | `VISUAL: IDENTITY`。`stats.json` に値はあるが使われない |

**帰結。**混ぜたデータセットで学習すると、**配布データ側の入力・出力の尺度まで変わる。**

### 6.2 統計の合成は mean/std だけが厳密で、**分位点は近似**

lerobot の `aggregate_feature_stats`（`datasets/compute_stats.py`）は
**平均をコマ数で重み付き平均し、分散に群間の散らばりを足す並列アルゴリズム**を使う。
これは数学的に厳密である。**一方で分位点はコマ数による単純な重み付き平均**で、
集合全体の分位点にはならない。

**3 通りで検算した**（道具は `solution/tools/track3_reversal/check_merge_stats.py`）。

| 検算 | 結果 |
|---|---|
| 同じデータセットを自分自身に混ぜる | **mean・std は最大差 0.000e+00**（count だけ 2 倍） |
| 3 つの実物を混ぜた結果と、式で合成した値 | **最大差 3×10⁻¹⁸** |
| `meta/stats.json` と生 parquet からの直接計算 | **mean 5×10⁻⁸・std 3×10⁻⁷・min/max 完全一致。ただし分位点は `q50` が最大 1.44 ずれる** |

**`MEAN_STD` を使っているかぎり分位点のずれは無害である。`QUANTILES` に切り替えるなら
統計を作り直すこと。**

### 6.3 据え置きたいときは `meta/stats.json` を書き換える

```bash
python3 solution/tools/track3_reversal/check_merge_stats.py \
    predict --provided <配布の stats.json か重みの normalizer> --ours <こちらの stats.json>
python3 solution/tools/track3_reversal/check_merge_stats.py \
    freeze  --merged <混ぜたデータセット> --source <配布のデータセット>
```

**耐えられた実績は ±8% しか無い。**`libero_plus`（2,238,036 コマ）から
`libero_combined_20hz`（3,028,708 コマ）へ乗り換えた 9/5 の `pi05_all111` では、
**std が動いた幅は action 0.95〜1.08 倍・state 0.95〜1.04 倍**だった
（`rp3000` と `all111` の normalizer を突き合わせた）。
**それより大きく動かすなら、据え置くか、統計を持たない
`lerobot/pi05_libero_base` から通し直すかを選ぶ**（base は
`normalizer_processor` の `stats` が空で `.safetensors` も同梱されていないので、
**最初の学習が尺度を決める**）。

**`freeze` が書き換えるのは `observation.state` と `action` だけ**で、元の値は
`meta/stats.json.bak` に残る。**混ぜた割合 20% で配布データの回転チャンネルが
0.43 倍に縮む**ことを実測してある——縮む原因（こちらの接近の回転が配布の 4〜5 倍広い）は
[30 の 8 節](30_reverse_grasp_pose.md) にある。

