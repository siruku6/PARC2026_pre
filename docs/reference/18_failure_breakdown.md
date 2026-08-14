# 18. π0.5 が落とした 11 件は 4 つの型に割れる

**「落とした分はすべて『届いたのに 1mm 動かして失格』」は間違いだった。**
実際に失格の内訳を数字と映像で当たると、**打つ手が正反対の 4 つの型**に割れる。
そのうち**2 件は誰がやっても取れない。**

対象は π0.5 24,000 歩・`n_action_steps`=16 の 36 件
（`work/gpu_eval/0814_113657_pi05_24k_n16_ep1_max300`。成功 25 / 衝突 6 / 未完了 5）。

```bash
# この文書の元データ（1 行 1 エピソード）
grep "結果" work/gpu_eval/0814_113657_pi05_24k_n16_ep1_max300/eval.log \
  | grep -v " 成功 " | sed 's/^.*\[結果\] //'

# 映像を静止画に畳んで見る（cv2 が要るのでキットの容器で回す）
docker run --rm -u "$(id -u):$(id -g)" \
    -v "$PWD/solution/tools:/host/tools:ro" \
    -v "$PWD/work/gpu_eval/0814_113657_pi05_24k_n16_ep1_max300:/out" \
    parc2026 python /host/tools/summarize_rollout_videos.py /out/videos
```

---

## 1. 4 つの型

| 型 | 件数 | 中身 | 打つ手 |
|---|---|---|---|
| **A. 物が落ちてくる** | **2** | 1 歩目で失格。**ゴールには届いている** | **無い**（採点ハーネス側） |
| **B. 別の物を大きく動かした** | 2 | 191mm・201mm。的を取り違えた／押した物が隣を押した | 学習（どれが的かを見分ける） |
| **C. あと数 mm** | 2 | **2.8mm・6.1mm。ゴール到達済み** | **軌道**（n・EMA・余裕の取り方） |
| **D. 課題ができていない** | 5 | ゴール未達で 300 歩打ち切り | 学習（能力そのもの） |

**C だけが今日つまみで動かせる分である。** 2 件 = 36 件のうち 0.056。

## 2. 型 A —— 3 つの別モデルで 0.1mm も違わず同じ値が出る

| タスク | 初衝突 | ぶつけた物体 | ゴール |
|---|---|---|---|
| `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_light_10` | **1 歩** | ketchup_1=544.6mm, orange_juice_1=385.7mm, milk_1=385.7mm | **到達** |
| `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_light_15` | **1 歩** | 同じ 3 つ・同じ値 | **到達** |

**π0.5 n=10・π0.5 n=16・weak3x n=8 の 3 つで同じ値が出る。**

```bash
for d in 0814_091240_pi05_24k_n10 0814_113657_pi05_24k_n16 0814_101939_weak3x_n8; do
    grep "LIVING_ROOM_SCENE2" work/gpu_eval/${d}_ep1_max300/eval.log | grep 結果
done
```

原因は採点ハーネスにある。`pipeline/rollout.py:165-186` は状態を入れてから
**空の行動を 10 歩だけ送って**物体の初期位置を記録する。

```python
for _ in range(10):
    obs, _, _, _ = env.step(dummy_action)
object_init_pos = {...}       # ← まだ落下中の物体は「空中の位置」が初期位置になる
```

**残りの落下がそのまま変位に化ける。** 判定は
`pipeline/rollout.py:241` の `collided = any(d > collision_threshold ...)`、
しきい値は 1mm、距離は**各軸の差の絶対値の和**（`np.sum(np.abs(...))`）である。

**方策は 1 歩で物を 10cm 動かせない。** 制御周期は 0.05 秒で、成功した回の
手先は 1 歩あたり 5mm 前後しか進まない。**1〜2 歩目で 100mm 超は落下と断定できる。**

> **`summarize_gpu_eval.py` はこれを見逃していた。** 判定が
> 「同じ変位が 3 回以上」だけだったため、**1 タスク 1 本の 36 件では
> 2 件どまりで発火しなかった**（84 件の一覧は 1 タスク 3 本なので発火する）。
> 2026-08-14 に**回数に頼らない物理の道**を足した
> （`_IMPOSSIBLE_COLLISION_STEPS` / `_IMPOSSIBLE_DISPLACEMENT_MM`）。
> **これで方策の衝突率が 0.167 から 0.111 に下がった。**

**36 件の一覧で取れる上限は 34 件である。** 25/34 = 0.735。

## 3. 型 C —— つまみで動くのはここだけ

| タスク | ぶつけた物体 | 初衝突 | ゴール |
|---|---|---|---|
| `pick_up_the_black_bowl_on_the_cookie_box_..._tb_13` | **cookies_1=2.8mm** | 113 歩 | 到達 |
| `pick_up_the_cream_cheese_and_place_it_in_the_basket_light_24` | **tomato_sauce_1=6.1mm** | 51 歩 | 到達 |

どちらも**指示文が名前を出している「隣の物」か、その近くの物**である。
掴む対象のすぐ隣に別の物があり、指を降ろすときに触っている。

**n=10 との比較で、この 2 件は n=16 のほうが悪い。**

| タスク | n=10 | n=16 |
|---|---|---|
| cookie_box tb_13 | 衝突 2.6mm | 衝突 **2.8mm** |
| cream_cheese light_24 | **成功** | 衝突 **6.1mm** |

**つまり n を上げると滑らかさ（平均ジャーク −14%）は良くなるが、
この型の余裕は良くならない。** n は的を外している可能性がある。

## 4. 型 B・D —— 学習でしか動かない

| 型 | タスク | 決め手 |
|---|---|---|
| B | `pick_up_the_black_bowl_on_the_ramekin_..._light_29` | **akita_black_bowl_2=191mm**。黒椀が 2 つあり、**別の椀を掴んだ**（握り x9・ゴール未達） |
| B | `push_the_plate_to_the_front_of_the_stove_table_23` | cream_cheese_1=201mm。握り x0 なので押しただけだが、**押した先に別の物があった** |
| D | `pick_up_the_black_bowl_next_to_the_plate_..._table_16` | 握り **x13**・終盤 50 歩 0.36m。**掴み直しで空回り** |
| D | `pick_up_the_black_bowl_on_the_wooden_cabinet_..._table_3` | 終盤 0.12m |
| D | `LIVING_ROOM_SCENE6_put_the_white_mug_..._light_14` | 終盤 **0.08m**。止まりかけ |
| D | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_light_31` | 経路 **0.86m** しかない・握り x2。**ほとんど動いていない** |
| D | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_29` | 終盤 0.24m |

**B の 2 件も D の 5 件も、共通するのは「黒椀が複数ある」「2 段階の課題」である。**

## 5. いちばん大きい弱点は光である

| 見た目の変え方 | 成功率 | 本数 |
|---|---|---|
| **Light Conditions** | **0.56** | 16 |
| Background Textures | 0.80 | 20 |
| （接尾辞で割ると）light | 0.56 | 16 |
| table | 0.77 | 13 |
| tb | 0.86 | 7 |

**light の 16 本で 7 件落としている。** これは n を変えても動かなかった
（n=10 も n=16 も 0.56）。**LIBERO-plus 全体でも light が最多（1,142 件）である。**

## 6. これが判断に効くところ

| 効くこと | 中身 |
|---|---|
| **「衝突対策で 5 件拾える」は 3 件だった** | 到達して失格の 5 件のうち 2 件は落下物。**残り 3 件のうち軌道で届きそうなのは 2 件** |
| **n はこの型に効いていない** | 平均ジャークは下がるが、2.8mm・6.1mm の余裕は広がらない。**採点の総合スコアには滑らかさが入る（[17_scoring_formula.md](17_scoring_formula.md)）ので無駄ではないが、成功件数は動かない** |
| **残りの大半（7 件）は能力の話** | 光への強さと、黒椀の見分け・2 段階の課題。**つまみでは動かない** |
| 36 件の上限は 34 件 | 比べるときは 25/34 = 0.735 で読む |
