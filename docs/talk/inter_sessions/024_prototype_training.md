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
（`EpisodeAwareSampler`・`shuffle=True`。**配布ではない、学習イメージ
`parc2026-train-pi05` の `/opt/lerobot` で 9/7 に確認**。pi05 は `drop_n_last_frames` を
持たないので、落ちるコマも無い）。

**1 周は 3,774,233 コマ ＝ batch 64 で 58,973 歩である。**2,000 歩は**その 3.4%** にすぎない。

| 起きること | 数 |
|---|---|
| 同じコマが 2 度引かれる（1 周を超える） | **起きない。**1 周の 3.4% しか回さない |
| **こちらの実体 1 コマが 2 回以上当たる**（11 本のコピーの別々が引かれる） | **5.4%**（期待値 0.373 のポアソン近似） |
| **こちらの実体 1 コマが 1 度も当たらない** | **68.9%** |

**つまり心配すべきは「学習しすぎ」ではなく「ほとんど見られない」側である。**
2 割にしても、こちらのコマの 7 割は 1 度も見られない。**それでも 2.19% のときの
0.041 回から 9 倍に増えるので、初報を取るには足りる。**

**全部を 1 度は見せたいなら、2 割のまま 5,400 歩（10.2 時間）か、N = 45（割合 5 割）である。**
**Blackwell の残りは 45 時間ほどなので、まずは 2,000 歩で様子を見る。**

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

## 5. やること

### 5.0 こちらの 389 本を上げる（**セッションに入る前に済ませる**）

```bash
cd /home/ubuntu/src/PARC2026_pre
tar -czf /tmp/proto389.tar.gz -C work/probe_017 reverse_lerobot_all   # 391 MB
# Blackwell へ渡す手立ては howto/14 の 1 節（HF の非公開リポジトリ経由が確実）
```

**`npz_to_lerobot.py` も一緒に渡す。**Blackwell では docker を使わず素の python で回すので、
`solution/tools/scripted_demo/npz_to_lerobot.py` の 1 ファイルが要る（依存は lerobot だけ）。

### 5.1 配布 tar を展開する（[howto/14](../../howto/14_blackwell_session.md) の節をそのまま）

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

### 5.2 こちらの 389 本を 11 倍にする（**小さいので速い。3 分**）

**倍々で組む。**11 = 8 + 2 + 1 なので 4 回の merge で済む
（1 本ずつ足すと 11 回・書き込み 26 GB になる）。

```sh
P=/opt/dlami/nvme/data/proto
mkdir -p "$P"; tar -xzf ~/proto389.tar.gz -C "$P"      # $P/reverse_lerobot_all
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

### 5.3 配布データに 1 度だけ混ぜる（**10〜20 分・25 GB 書く**）

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

**`align` を先に打つ。**打たずに `merge` すると
`Same features is expected` で落ちる（**落ちるだけで壊れはしない**）。

### 5.4 smoke（**20 歩・batch 2。5 分**）

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
| loss（20 歩） | **0.2〜0.5** | **1.0 超なら正規化が壊れている。**そこで止めて報告 |

### 5.5 本番の 1 本（**3 時間 49 分**）

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

### 5.6 測る（**RTX6000。EGL が効く方**）

```bash
cd /home/ubuntu/src/PARC2026_pre
bash solution/tools/eval/run_final_eval.sh track3     # 4 タスク × 2 本
bash solution/tools/eval/run_final_eval.sh track1     # 退化していないかの確認
```

**比べる相手は `rp3000` と `all111`（どちらも Track3 は 0.000）である。**

## 6. できあがりの条件

1. `dataset.num_episodes` が **23812** と出たログ
2. 2,000 歩が終了コード 0 で終わり、loss が **0.3 前後**
3. `run_final_eval.sh track3` の点（4 タスク × 2 本）と、`track1` の点

**3 つが揃えば終わりである。点が 0.000 のままでも終わりである**——
**「逆向きデータを 2 割混ぜても動かない」は、それ自体が次の判断を変える。**

## 7. やらなくてよいこと

- **データの作り直し。**この 389 本は**擦りと落下を直す前**に作ったものである
  （皿が 21 mm 動く・缶が 37 mm 落ちる。直した後の作り方は
  [reference/31](../../reference/31_reverse_place_without_contact.md)）。
  **直した版で作り直すのは [023](023_local_eval_dataset.md) の仕事である。**
  **だからこの学習の点は「下限」であって、方式の評価ではない。**
- **two mugs を足すこと。**この 389 本には入っていない。
- **歩数を伸ばすこと・割合を変えて何本も回すこと。**まず 1 本だけ。
- **提出物を組むこと。**この重みは出さない。

## 8. 報告の書き方

| 欄 | 例 |
|---|---|
| `dataset.num_episodes` | 23812 |
| 所要・終了コード | 3 時間 51 分・0 |
| loss（最後の 10 点の平均） | 0.31 |
| **Track3 の点**（4 タスク × 2 本・タスクごとの成否） | 本 1/2・缶 0/2・赤マグ 0/2・two mugs 0/2 |
| Track1 の点 | 0.4xx（`rp3000` は 0.492） |
| 踏んだ穴 | 1 行ずつ |

## 9. 関連

| 何 | どこ |
|---|---|
| Blackwell の毎セッション手順 | [howto/14](../../howto/14_blackwell_session.md) |
| 学習の条件と過去の点・loss | [reference/19](../../reference/19_weights_catalog.md) |
| 配布データの中身と重なり | [reference/25](../../reference/25_dataset_inventory.md) 8 節 |
| 数の関係（デモ × init の 2 因子） | [reference/06](../../reference/06_training_data_composition.md) 末尾 |
| **明日の手元評価用データ**（4 タスク・作れる組を全部） | **[023](023_local_eval_dataset.md)** |
| **本番学習用データ**（2 因子を振って 1 タスク 50 本） | **[020](020_reverse_dataset_regeneration.md)** |

## 10. 後片付け（この依頼で作った手元の一時物）

**2.8 GB が残っている**——`work/probe_024/proto_x1`（391 MB）・`proto_x2`（780 MB）・
`proto_x4`（1.6 GB）。**このマシンでは `rm -rf` が拒否される**ので消せていない。
**`proto_x1` 以外は要らない。**ユーザが消すか、許可が下りたら消す。

| 消し残し | 大きさ | いつのもの |
|---|---|---|
| `work/probe_024/proto_x2` / `proto_x4` | **2.4 GB** | 9/7 のコピー検査 |
| `work/probe_017/reverse_lerobot_all_tmp1` / `reverse_lerobot_bs` / `reverse_lerobot` | 511 MB | 017 の中間物 |
| `work/probe_018/cost_npz` | 244 MB | 018 の所要計測 |
