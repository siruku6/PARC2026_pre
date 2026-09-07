# 024 prototype 学習——いまある 389 本を配布データに混ぜて、学習が通ることを確かめる

> **状態: 手順確定・未実行** ／ 更新: 2026-09-07
>
> **これは「点を取る学習」ではない。**目的は 2 つだけ——
> **①混ぜた学習が最後まで通ること ②逆向きデータが Track3 を 0.000 から動かすかの初報**。
> **明日の手元評価用データを作る（[023](023_local_eval_dataset.md)）前に、これを先に片付ける。**

## 1. 背景

| | |
|---|---|
| どのトラックか | **Track3**（逆向きの指示。順方向の Track1・Track2 とは別立て） |
| いまの点 | **Track3 は 0.000**（本選の提出全 7 版すべて）。Track1 は 0.492 が最高 |
| 何で測るか | **`compe/t3` の 4 タスク × 2 本**（`run_final_eval.sh track3`）。本番は非公開タスク |
| すでに分かっていること | **配布ベースラインの Track3 は 0.000**（[reference/19](../../reference/19_weights_catalog.md)）。**そのベースラインが `libero_combined_20hz`（111 指示文）で学習されたというのは配布資料の記述で、裏は取っていない**（[reference/25](../../reference/25_dataset_inventory.md) 8.3 節）。**順方向の場面を見せるだけでは逆向きは動かない、という読みはここから来ている** |
| すでに外したこと | 歩数を伸ばす道（loss が寝た・[reference/19](../../reference/19_weights_catalog.md)）、場面カメラ潰し（9/4 に負けた） |

**だから次に試す価値があるのは「逆向きのデモそのものを見せる」1 手だけである。**
その 1 手が**学習として成立するのか**を、いちばん安く確かめるのがこの依頼である。

## 2. 何をしてほしいか

**`work/probe_017/reverse_lerobot_all`（389 本）を配布データに 2 割の割合で混ぜ、
`rp3000` から 2,000 歩・batch 64 で 1 本回して、Track3 と Track1 を測る。**

成果物は**表 1 枚**——学習が通ったか、`dataset.num_episodes` が期待値だったか、
loss の水準、Track3 と Track1 の点。

## 3. 割合は「間引く」のではなく「コピーを足す」で作る

**これがこの依頼の要である。**

**エピソード番号を重複させて重みを付ける道は無い。** LeRobot は `--dataset.episodes` を
**集合として**扱う（`episode_index.isin(episodes)`・`train.py` の `_select_emphasized_episodes` の
注意書き）。**だから重みは「番号の指定」では作れない。**

**しかし、データセットそのものに同じエピソードを何本も入れるのは通る。**
`merge` はエピソード番号を振り直して連結するだけなので、**同じデータセットを自分自身に
混ぜられる。**9/7 に手元で実測した。

```
proto_x1 (389 本 / 67,775 コマ) + proto_x1 → 778 本 / 135,550 コマ / 3 タスク
778 + 778 → 1,556 本 / 271,100 コマ / 3 タスク   （40 秒。指示文は 3 のまま重複しない）
```

**1 コマ目を実際に読む検査（`inspect`）も通った。**`action(7)` / `observation.state(8)` /
`observation.images.front` / `.wrist` が揃い、**`stats/observation.images.front/mean` の列も残る。**

### なぜ「配布側を間引く」より良いのか

もう一方の道は `--emphasize-tasks` に 3 つの指示文を渡して
`--emphasis-multiplier 9.1` を付けることである（**実装済み。フレーム数で割合を解く**）。
これは **2 割にするために配布データの 91% を捨てる**（残るのは 8.95%・約 1,749 本）。

**引ける回数はどちらでも同じである。**2,000 歩 × batch 64 = 128,000 回のうち、
配布側に落ちるのは 102,720 回。**違うのは、その 102,720 回が何コマの上に散るかである。**

| 道 | 配布側の母数 | 配布側 1 コマあたりの回数 | 111 指示文の覆い |
|---|---|---|---|
| **コピーを足す** | **3,028,708 コマ（丸ごと）** | **0.034 回** | **全部** |
| 間引く | 271,100 コマ | 0.379 回 | **タスクごとに 9% だけ** |

**間引く道は、同じ回数を 11 分の 1 の狭い集合に打ち込む。**多様性を捨てて繰り返しを増やす
だけなので、**この依頼では使わない。**（`--emphasize-tasks` を消すという話ではない。
**配布データを丸ごと持っていない機械で回すときには要る。**）

## 4. 前提となる事実

### 4.1 混ぜる 2 つ

| | 配布データ | **こちらの 389 本** |
|---|---|---|
| 名前 | `libero_combined_20hz`（Blackwell の `~/dataset/lerobot/*.tar`・19.6 GiB） | `work/probe_017/reverse_lerobot_all`（**391 MB**） |
| エピソード | **19,533** | **389** |
| コマ | **3,028,708** | **67,775** |
| 指示文 | **111** | **3** |
| 形式 | v3.0・fps 20・カメラ 2 つ・state 8 次元・action 7 次元 | **同じ**（だから混ざる） |

**指示文は `compe/t3` の bddl の `:language` と 1 文字も違わない。**9/7 に突き合わせた。

| 課題 | 指示文 | 本数 |
|---|---|---|
| 本 | `take the yellow book off the top of the shelf and put it back on the table on the left` | 150 |
| 赤マグ | `take the red mug off the left plate and put it back on the table behind the other two mugs` | 139 |
| 缶 | `take the alphabet soup out of the tray and place it on the table in front of the ketchup` | 100 |

**two mugs は入っていない。**4 タスク目は [023](023_local_eval_dataset.md) で作る。

### 4.2 コピーの本数と割合

`share = N × 67,775 / (3,028,708 + N × 67,775)`。引ける回数は 2,000 歩 × batch 64 = **128,000 回**。

| N | 割合 | こちらの容量 | 混ぜた後のコマ数 | **こちらの 1 コマが引かれる回数（期待値）** |
|---|---|---|---|---|
| 1（そのまま足す） | 2.19% | 0.4 GB | 3,096,483 | **0.041**（96% が 1 度も見られない） |
| 5 | 10.1% | 2.0 GB | 3,367,583 | 0.190 |
| **11** | **19.8%** | **4.3 GB** | **3,774,233** | **0.373** |
| 22 | 33.0% | 8.6 GB | 4,519,758 | 0.623 |
| 45 | 50.2% | 17.6 GB | 6,078,583 | 0.948 |

**2 割は N = 11 である。**

### 4.3 「同じコマを何度も学習してしまう」は起きない

**サンプラは `torch.randperm` で、1 周（epoch）のあいだ同じコマを 2 度引かない**
（`EpisodeAwareSampler`・`shuffle=True`。**学習イメージ
`parc2026-train-pi05` の `/opt/lerobot` で 9/7 に確認**。pi05 は `drop_n_last_frames` を
持たないので、落ちるコマも無い）。

**1 周は 3,774,233 コマ ＝ batch 64 で 58,973 歩である。**2,000 歩は**その 3.4%** にすぎない。

| 起きること | 数 |
|---|---|
| 同じコマが 2 度引かれる（1 周を超える） | **起きない。**1 周の 3.4% しか回さない |
| **こちらの実体 1 コマが 2 回以上当たる**（11 本のコピーの別々が引かれる） | **5.4%** |
| **こちらの実体 1 コマが 1 度も当たらない** | **68.9%** |

### 4.3.1 その 68.9% は、学習の穴にはなっていない —— **1 コマが 50 歩ぶんの答えを持つ**

**pi05 は `chunk_size = 50` である**（`action_delta_indices = range(50)`。
`rp3000` と `all111` の `config.json` はどちらも `chunk_size: 50` / `n_action_steps: 10`）。
**つまり 1 サンプルは「観測 1 枚 → 続く 50 歩の行動」を教える。**

だから**ある 1 歩の行動 `a[t]` は、`t-49` から `t` までの 50 個のコマのどれが引かれても教わる。**

| 覆い | 割合 20%・2,000 歩でどうなるか |
|---|---|
| **観測（入力）として使われるコマ** | **31.1%**（残りの 68.9% は入力にならない） |
| **行動（答え）として教わる歩** | **ほぼ 100%**（1 歩が漏れる確率は 6×10⁻⁹） |

**漏れているのは「観測の見た目の多様性」だけである。**20 fps なので隣のコマの絵は
ほとんど同じで、**軌道そのものは 1 歩も漏れていない。**

### 4.3.2 それでも入力側の覆いを上げたいときの手立て

覆いは `1 − exp(−引いた回数 ÷ こちらのコマ数)` で決まる。**引いた回数 ＝ 128,000 × 割合**。

| 手立て | 割合 | こちらに落ちる回数 | **入力の覆い** | 配布に落ちる回数 | 代償 |
|---|---|---|---|---|---|
| いまの案 | 20% | 25,600 | **31%** | 102,400 | — |
| **割合を上げる**（N=22） | 33% | 42,240 | **46%** | 85,760 | 配布の補強が 3 分の 2 に |
| 割合を上げる（N=45） | 50% | 64,000 | **61%** | 64,000 | 配布の補強が半分に |
| **歩数を倍にする** | 20% | 51,200 | **53%** | 204,800 | **GPU 3 時間 49 分 → 7 時間 38 分** |
| **こちらを 1 init 1 デモに間引く**（389 → 150 本・26,100 コマ） | 20% | 25,600 | **62%** | 102,400 | デモの掴み方の違いが消える（**init 50 行の覆いは保つ**） |
| 間引く ＋ 割合 33% | 33% | 42,240 | **80%** | 85,760 | 同上 |

**割合を上げる道は、5 節（正規化）を読んでから決めること。**
**据え置きを併せて打てば、割合を上げても配布側の正規化は動かない。**

**この依頼では 20%（N=11）のままで進める。**行動側の覆いは元から 100% なので、
**入力の覆いを 31% から上げても得るものは「観測の見た目の多様性」だけ**である。
**上げたくなったら N=22（33%）が手頃**——コピーを増やすだけで、
据え置き（5 節）を併せて打てば配布側の正規化も動かない。
**間引く道（389 → 150 本）は、init 50 行の覆いは保つが掴み方の違いを捨てるので、
覆いの数字のために選ぶ価値は無い。**

### 4.4 回す機械と条件（**9/5 の `pi05_all111` と 1 変数だけ違う形にする**）

| | `pi05_all111`（9/5・既にある） | **prototype 学習** |
|---|---|---|
| 機械 | Blackwell（RTX PRO 6000・95 GB） | **同じ** |
| 出発点 | `siruku6/pi05_full_runpod`（＝`rp3000`） | **同じ** |
| 歩数 / batch / 学習率 | 2,000 / 64 / 5e-6 | **同じ** |
| 動かす範囲 | 全部（4.14B） | **同じ** |
| カメラ拡張 | あり | **同じ** |
| **データ** | `libero_combined_20hz`（19,533 本） | **＋こちらの 389 本 × 11 = 23,812 本** |
| 所要 | **3 時間 49 分**（6.85 秒/step） | **同じ見込み ＋ 段取り 1 時間** |

**loss の物差しも `all111` がある——終わりで 0.309**（40 指示文だけの走りは 0.23〜0.25。
[reference/19](../../reference/19_weights_catalog.md)）。**prototype が 0.3 前後なら正常、
1.0 を超えていたら正規化が壊れている。**

## 5. 正規化統計——**混ぜると配布データの見え方まで変わる。だから据え置く**

### 5.1 どの値が学習に使われるのかを確かめた

| 分かったこと | 出どころ |
|---|---|
| **pi05 は state と action を mean/std で正規化する** | `rp3000` と `all111` の `config.json` の `normalization_mapping` が `{ACTION: MEAN_STD, STATE: MEAN_STD, VISUAL: IDENTITY}`。**画像は正規化されない** |
| **学習は「データセットの `meta/stats.json`」を使う。重みに入っている値は使わない** | `lerobot_train.py` 315〜332 行が `preprocessor_overrides["normalizer_processor"]["stats"] = dataset.meta.stats` で**上書きする**（`--init-from` で始めて `--resume` でないときはこの枝に入る） |
| **だから混ぜた瞬間に、配布データ 111 課題の正規化も変わる** | 上の 2 つの帰結 |
| pi05 の既定は本来 `QUANTILES` だが、**こちらの重みは `MEAN_STD` を持っているのでそちらが勝つ** | `configuration_pi05.py:74-76` が既定・`config.json` が上書き。**分位点は使われない**（**使われていたら困った**——5.2 参照） |

### 5.2 統計の再計算が正しいことを 3 通りで検算した

道具を足した——**`solution/tools/track3_reversal/check_merge_stats.py`**。
lerobot の `aggregate_feature_stats` と同じ式（**平均はコマ数の重み付き平均、
分散は群間の散らばりを足した並列アルゴリズム**）を独立に実装して突き合わせる。

```bash
# ① 同じものを混ぜたら平均と分散は動かないこと
python3 solution/tools/track3_reversal/check_merge_stats.py verify \
  --parts work/probe_024/proto_x1 work/probe_024/proto_x1 --merged work/probe_024/proto_x2
# ② 違う 3 つを混ぜた実物と、式で合成した値が一致すること
python3 solution/tools/track3_reversal/check_merge_stats.py verify \
  --parts work/probe_017/reverse_lerobot_book work/probe_017/reverse_lerobot_soup \
          work/probe_017_mug/reverse_lerobot_mug \
  --merged work/probe_017/reverse_lerobot_all --tol 1e-6
```

| 検算 | 結果 |
|---|---|
| ① **コピーで平均・分散が動かない** | **最大差 0.000e+00**（コマ数だけ 67,775 → 135,550 → 271,100 と増える） |
| ② **book ＋ soup ＋ mug の合成 = `reverse_lerobot_all` の実物** | **最大差 3×10⁻¹⁸**（float64 の丸め） |
| ③ **`meta/stats.json` と、生 parquet からの直接計算** | **mean 最大差 5×10⁻⁸・std 最大差 3×10⁻⁷・min/max は完全一致** |

**③ で分位点だけは合わない**（`q50` が最大 1.44 ずれる）。
**lerobot はエピソードごとの分位点をコマ数で重み付き平均しているだけ**なので、
**集合全体の分位点にはなっていない。**平均と分散は同じ扱いでも数学的に厳密だが、
**分位点は近似である。****pi05 が `MEAN_STD` を使っているので今は無害だが、
`QUANTILES` に切り替えるなら統計を作り直すこと。**

### 5.3 混ぜると配布データの正規化がどれだけ動くか（**実測値で計算した**）

配布側の統計は推測ではない——**`siruku6/pi05_all111` の `002000/` に同梱された
normalizer（`policy_preprocessor_step_2_normalizer_processor.safetensors`）を
落として読んだ。**`count` が 3,028,708 で `libero_combined_20hz` と一致する。

```bash
python3 solution/tools/track3_reversal/check_merge_stats.py predict \
  --provided <all111 の normalizer .safetensors> \
  --ours work/probe_017/reverse_lerobot_all/meta/stats.json --copies 1 5 11 22 45
```

**「配布のずれ」は、混ぜる前に平均 0・分散 1 だった配布データが、混ぜた後にどこへ動くか。**

| N | 割合 | **配布の action の倍率** | 配布の state の倍率 | こちらの action の倍率 |
|---|---|---|---|---|
| 1 | 2.19% | **0.817〜0.999** | 0.990〜1.005 | 1.03〜3.99 |
| **11** | **19.75%** | **0.426〜0.992** | 0.919〜1.050 | 1.03〜2.08 |
| 22 | 32.99% | **0.342〜0.987** | 0.874〜1.089 | 1.02〜1.67 |
| 45 | 50.17% | **0.283〜0.981** | 0.825〜1.149 | 1.01〜1.38 |

**割合 20% で、配布データの回転チャンネルが 0.43 倍に縮む。**
`rp3000` はその 2.3 倍の尺度で 111 課題を覚えているので、**2,000 歩・学習率 5e-6 で
出力の尺度を作り直させることになる。****Track1・Track2 を壊す筋である。**

### 5.4 縮む原因は**こちらのデータの回転指令が配布の 4〜5 倍広いこと**

| action の次元 | 配布の std | **こちらの std** | 倍率 |
|---|---|---|---|
| 0〜2（並進） | 0.329 / 0.379 / 0.444 | 0.340 / 0.570 / 0.490 | 1.03〜1.50 |
| **3〜5（回転）** | **0.040 / 0.062 / 0.078** | **0.152 / 0.150 / 0.380** | **3.8 / 2.4 / 4.9** |
| 6（グリッパ） | 0.929 | 1.000 | 1.08 |

**配布データの回転指令は ±0.375 に収まっているのに、こちらは ±1（クリップの上限）に
届いている。**配布の mean/std で正規化すると、こちらのデータは次のようになる。

| | 値 |
|---|---|
| 回転チャンネルの \|z\| の最大 | **24.8**（次元 3） |
| **どこか 1 次元でも \|z\| > 5 のコマ** | **21.1%** |
| **同 \|z\| > 10** | **15.1%** |
| その \|z\|>10 のコマがどこにあるか | **エピソードの前半に集中**（先頭 10% のコマの **52.6%** が該当。0.5 以降は 3% 以下） |

**前半に集中しているので、原因は復路ではなく、こちらが合成した接近・掴み・持ち上げの
区間である。**掴み姿勢へ手首を回す指令を毎コマ上限で出しているためである
（[reference/30](../../reference/30_reverse_grasp_pose.md)）。

**これは 020・023 のデータ生成へ返す課題である**——**接近の回転を配布の封筒
（±0.375）に収まる速さへ落とすか、対称性を使って回転量そのものを減らす。**
**ただしコマ数の上限 300 と正面衝突するので、この依頼では触らない。**

### 5.5 判断：**この 1 本は正規化を配布データのものに据え置く**

**据え置くと、111 課題の見え方は混ぜる前と 1 ビットも変わらない。**
道具を足した（`freeze`。**書き換えるのは `observation.state` と `action` だけ**で、
画像は `IDENTITY` なので触らない。元の値は `stats.json.bak` に残す）。

| 道 | 配布 111 課題 | こちらの 389 本 | 判断 |
|---|---|---|---|
| **据え置く（採用）** | **変化なし** | 回転の 15% が \|z\|>10 の外れ値になる | **採用。**`grad_clip_norm 10` が効くので発散は抑えられる。**1 変数の比較が成り立つ** |
| 混ぜた値を使う | **回転が 0.43 倍に縮む** | 収まる | **採らない。**Track1・Track2 が落ちたとき原因を切り分けられない |

**推論との食い違いは起きない。**評価時に読まれるのは**学習が書き出した
チェックポイント同梱の preprocessor** なので、据え置いた値がそのまま使われる。

**尺度の変更に耐えられた実績は ±8% しか無い。**9/5 の `pi05_all111` は
`libero_plus`（2,238,036 コマ）から `libero_combined_20hz`（3,028,708 コマ）へ
乗り換えた 1 本で、**そのとき std が動いた幅は action 0.95〜1.08 倍・
state 0.95〜1.04 倍**である（`rp3000` と `all111` の normalizer を突き合わせた）。
**0.43 倍が耐えられるかを示す一次情報は無い。**だから据え置く。

**出発点の重みを `lerobot/pi05_libero_base` に戻す道もある**——base は
正規化統計を持っていないので（`normalizer_processor` の `stats` が空・
`.safetensors` も無い）、**最初の学習が尺度を決め、食い違いが起きる余地が無い。**
**ただし `rp3000` 相当まで戻すのに batch 64 で 9,000 歩・約 17 時間かかる。**
**この 1 本では採らない**（1 変数の比較が崩れ、Track3 が 0.000 のままでも
「データが効かない」のか「学習が足りない」のか分けられなくなる）。
**本番でどうするかは [020 の 9 節](020_reverse_dataset_regeneration.md) にある。**

**smoke で loss が 1.0 を超えたら、据え置きをやめて混ぜた値で回し直す**
（`stats.json.bak` を戻すだけでよい）。**そのときは Track1 の退化も合わせて報告すること。**

## 6. やること

### 6.0 こちらの 389 本を上げる（**セッションに入る前に済ませる**）

```bash
cd /home/ubuntu/src/PARC2026_pre
tar -czf /tmp/proto389.tar.gz -C work/probe_017 reverse_lerobot_all   # 391 MB
hf upload siruku6/reverse_lerobot_389 /tmp/proto389.tar.gz proto389.tar.gz \
    --repo-type dataset --private
```

**9/7 に上げたのは `siruku6/reverse_lerobot_389`（データセット・非公開）の
`proto389.tar.gz`（395 MB・35 MB/s で 11 秒）である。**

**道具を 2 つ一緒に渡す。**Blackwell では docker を使わず素の python で回すので、
ファイルそのものが要る（どちらも依存は lerobot と numpy だけ）。

| ファイル | 何に使うか |
|---|---|
| `solution/tools/scripted_demo/npz_to_lerobot.py` | `align` / `merge` / `inspect` |
| **`solution/tools/track3_reversal/check_merge_stats.py`** | **正規化統計の検算と据え置き**（5 節） |

### 6.1 配布 tar を展開する（[howto/14](../../howto/14_blackwell_session.md) の節をそのまま）

```sh
. ~/env_pi05.sh
DATA=/opt/dlami/nvme/data/libero_combined_20hz
mkdir -p "$DATA"
time tar -xf ~/dataset/lerobot/libero_combined_20hz.tar -C "$DATA" --strip-components=1
python3 -c "
import json; i = json.load(open('$DATA/meta/info.json'))
print(i['total_episodes'], i['total_frames'], i['total_tasks'], i['fps'])"
# 19533 3028708 111 20 と出ること。違ったら止める
```

### 6.2 こちらの 389 本を 11 倍にする（**小さいので速い。3 分**）

**倍々で組む。**11 = 8 + 2 + 1 なので **5 回**の merge で済む
（1 本ずつ足すと 10 回・書き込み 26 GB になる）。**中間物も含めて約 14 GB 使う。**

```sh
P=/opt/dlami/nvme/data/proto
# **`hf download` は $HOME ではなく HF キャッシュに置く。**返ってきた path を使う
TAR=$(find "${HF_HOME:-$HOME/.cache/huggingface}/hub/datasets--siruku6--reverse_lerobot_389" \
        -name proto389.tar.gz | head -1)
echo "$TAR"                                            # 空でないこと
mkdir -p "$P"; tar -xzf "$TAR" -C "$P"                 # $P/reverse_lerobot_all
python3 -c "
import json; i = json.load(open('$P/reverse_lerobot_all/meta/info.json'))
print(i['total_episodes'], i['total_frames'], i['total_tasks'], i['fps'])"
# 389 67775 3 20 と出ること
M() { python ~/npz_to_lerobot.py merge \
        --base "$1" --base-repo-id local/a \
        --add  "$2" --add-repo-id  local/b \
        --output "$3" --output-repo-id local/proto || exit 1; }
M $P/reverse_lerobot_all $P/reverse_lerobot_all $P/x2      #   778 本
M $P/x2  $P/x2                     $P/x4                   # 1,556 本
M $P/x4  $P/x4                     $P/x8                   # 3,112 本
M $P/x8  $P/x2                     $P/x10                  # 3,890 本
M $P/x10 $P/reverse_lerobot_all    $P/x11                   # 4,279 本 / 745,525 コマ
```

**`x11` が 4,279 本 / 745,525 コマ / 3 タスクであることを確かめてから次へ進む。**

### 6.3 配布データに 1 度だけ混ぜる（**10〜20 分・25 GB 書く**）

```sh
python ~/npz_to_lerobot.py align --base "$DATA" --add "$P/x11"    # features の飾りを揃える
MERGED=/opt/dlami/nvme/data/proto20
python ~/npz_to_lerobot.py merge \
  --base "$DATA"   --base-repo-id lerobot/libero_combined_20hz \
  --add  "$P/x11"  --add-repo-id  local/t3_reverse_x11 \
  --output "$MERGED" --output-repo-id local/t3_proto20
python3 -c "
import json; i = json.load(open('$MERGED/meta/info.json'))
print(i['total_episodes'], i['total_frames'], i['total_tasks'])"
# 23812 3774233 114 と出ること。違ったら止める
```

**混ぜたら、正規化統計を配布データのものに据え置く**（**理由は 5 節**）。

```sh
# ① 据え置く前に、混ぜた統計が式どおりかを検算する（ここは一致すること）
python3 ~/check_merge_stats.py verify --parts "$DATA" "$P/x11" --merged "$MERGED"
# ② 配布データの値へ据え置く
python3 ~/check_merge_stats.py freeze --merged "$MERGED" --source "$DATA"
#    action: std の倍率 0.426〜0.992 を 1.0 に戻す  と出ること（割合 20% のとき）
# ③ もう一度 verify を打つと、こんどは action と state が「ちがう」と出る。
#    **それが据え置きが効いた証拠である**（元の値は stats.json.bak にある）
```

**`align` を先に打つ。**打たずに `merge` すると
`Same features is expected` で落ちる（**落ちるだけで壊れはしない**）。

### 6.4 smoke（**20 歩・batch 2。5 分**）

```sh
cd "$CODE_DIR"
python train_pi05.py --run-name proto20_smoke --smoke-test \
    --init-from /opt/dlami/nvme/weights/rp3000 \
    --dataset-repo local/t3_proto20 --dataset-root "$MERGED"
```

| 見るもの | 期待値 | 違ったら |
|---|---|---|
| **`dataset.num_episodes`** | **23812** | **19533 なら混ざっていない。14347 なら乗り換えていない。どちらもその場で止める** |
| `num_learnable_params` | 4143404816 | 693422112 なら行動側しか動かない |
| loss（20 歩） | **0.2〜0.5** | **1.0 超なら据え置きをやめる**——`stats.json.bak` を戻して混ぜた値で回し直し、**Track1 の退化も合わせて報告する**（5.5 節） |

### 6.5 本番の 1 本（**3 時間 49 分**）

```sh
NW=$(python -c "import os;print(min(os.cpu_count()//2, 16))")
nohup python train_pi05.py --run-name pi05_proto20 \
    --init-from /opt/dlami/nvme/weights/rp3000 \
    --dataset-repo local/t3_proto20 --dataset-root "$MERGED" \
    --steps 2000 --batch-size 64 --num-workers "$NW" \
    --learning-rate 5e-6 --save-freq 500 \
    --freeze-vision-encoder false --train-expert-only false \
    --camera-viewpoint-augment \
    > $OUTPUT_ROOT/pi05_proto20.nohup 2>&1 &
```

**`--max-demo-steps` は付けない。**付けると配布側の長いデモも落ちて、
`all111` との 1 変数比較が崩れる（**こちらの 389 本で 300 コマを超えるのは 1 本だけ**・最長 371）。

**チェックポイントを HF へ上げるループは [howto/14](../../howto/14_blackwell_session.md) の B 節を
名前だけ差し替えて打つ**（`pi05_proto20`）。

### 6.6 測る（**RTX6000。EGL が効く方**）

```bash
cd /home/ubuntu/src/PARC2026_pre
bash solution/tools/eval/run_final_eval.sh track3     # 4 タスク × 2 本
bash solution/tools/eval/run_final_eval.sh track1     # 退化していないかの確認
```

**比べる相手は `rp3000` と `all111`（どちらも Track3 は 0.000）である。**

## 7. できあがりの条件

1. `dataset.num_episodes` が **23812** と出たログ
2. **`verify` が一致し、`freeze` が「0.426〜0.992 を 1.0 に戻す」と出たログ**（5 節）
3. 2,000 歩が終了コード 0 で終わり、loss が **0.3 前後**
4. `run_final_eval.sh track3` の点（4 タスク × 2 本）と、`track1` の点

**4 つが揃えば終わりである。点が 0.000 のままでも終わりである**——
**「逆向きデータを 2 割混ぜても動かない」は、それ自体が次の判断を変える。**

## 8. やらなくてよいこと

- **データの作り直し。**この 389 本は**擦りと落下を直す前**に作ったものである
  （皿が 21 mm 動く・缶が 37 mm 落ちる。直した後の作り方は
  [reference/31](../../reference/31_reverse_place_without_contact.md)）。
  **直した版で作り直すのは [023](023_local_eval_dataset.md) の仕事である。**
  **だからこの学習の点は「下限」であって、方式の評価ではない。**
- **two mugs を足すこと。**この 389 本には入っていない。
- **歩数を伸ばすこと・割合を変えて何本も回すこと。**まず 1 本だけ。
- **接近の回転を配布データの封筒に収めること。**外れていることは測ってあるが
  （[reference/30 の 8 節](../../reference/30_reverse_grasp_pose.md)）、
  **直すのは生成側の仕事で、コマ数の上限 300 と衝突する。**この 1 本は据え置きで逃げる。
- **提出物を組むこと。**この重みは出さない。

## 9. 踏んだ穴（打ちながら足す）

| 症状 | 原因と対処 |
|---|---|
| `tar: ~/proto389.tar.gz: Cannot open` | **`hf download` は `$HOME` に置かない。**HF キャッシュの `hub/datasets--<repo>/snapshots/<sha>/` に入る。6.2 の `find` で解決する（9/7 に踏んだ） |
| `~/pi05_code/train_pi05.py` が無い | **`siruku6/pi05_code` は入れ子である。**実体は `~/pi05_code/solution/train/train_pi05.py`。`CODE_DIR` はそこを指す（9/7 に踏んだ） |
| `hf download` が `Repository not found` | **データセットのリポジトリには `--repo-type dataset` が要る**（9/7 に踏んだ） |

## 10. 報告の書き方

| 欄 | 例 |
|---|---|
| `dataset.num_episodes` | 23812 |
| 所要・終了コード | 3 時間 51 分・0 |
| loss（最後の 10 点の平均） | 0.31 |
| **正規化を据え置いたか**（据え置かなかったならその理由） | 据え置いた／smoke の loss が 1.2 だったので混ぜた値に戻した |
| **Track3 の点**（4 タスク × 2 本・タスクごとの成否） | 本 1/2・缶 0/2・赤マグ 0/2・two mugs 0/2 |
| Track1 の点 | 0.4xx（`rp3000` は 0.492） |
| 踏んだ穴 | 1 行ずつ |

## 11. 関連

| 何 | どこ |
|---|---|
| Blackwell の毎セッション手順 | [howto/14](../../howto/14_blackwell_session.md) |
| **接近の回転が配布の分布を外れること**（原因と手立ての候補） | **[reference/30 の 8 節](../../reference/30_reverse_grasp_pose.md)** |
| 学習の条件と過去の点・loss | [reference/19](../../reference/19_weights_catalog.md) |
| 配布データの中身と重なり | [reference/25](../../reference/25_dataset_inventory.md) 8 節 |
| 数の関係（デモ × init の 2 因子） | [reference/06](../../reference/06_training_data_composition.md) 末尾 |
| **明日の手元評価用データ**（4 タスク・作れる組を全部） | **[023](023_local_eval_dataset.md)** |
| **本番学習用データ**（2 因子を振って 1 タスク 50 本） | **[020](020_reverse_dataset_regeneration.md)** |

## 12. 後片付け（この依頼で作った手元の一時物）

**2.8 GB が残っている**——`work/probe_024/proto_x1`（391 MB）・`proto_x2`（780 MB）・
`proto_x4`（1.6 GB）。**このマシンでは `rm -rf` が拒否される**ので消せていない。
**`proto_x1` 以外は要らない。**ユーザが消すか、許可が下りたら消す。

| 消し残し | 大きさ | いつのもの |
|---|---|---|
| `work/probe_024/proto_x2` / `proto_x4` | **2.4 GB** | 9/7 のコピー検査 |
| `work/probe_017/reverse_lerobot_all_tmp1` / `reverse_lerobot_bs` / `reverse_lerobot` | 511 MB | 017 の中間物 |
| `work/probe_018/cost_npz` | 244 MB | 018 の所要計測 |
