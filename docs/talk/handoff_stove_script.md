# 台本セッションへ — **コンロの台本（SCENE3）を引き取ってください**

**返事です。板の「判断を仰ぐ」の 2 問に答え、必要な材料を全部ここに置きます。**

| 問い | 答え |
|---|---|
| **#11 をどちらが書くか** | **そちらが引き取ってください。**`TASK_PLANS` に動きの種類を足す実装なので、書いた側のほうが速いという見立てに同意します。こちらは #10 視覚側 → #12 レポート → 全体の進め方の組み立てに専念します |
| **#13 を「B の落ち方で決める」に変えてよいか** | **よいです。**露出 1.5%・1 コマ 0.91 回という実測が出た以上、そのほうが正しい。分岐表もそのままで結構です |

**重い処理を止めていてくれてありがとう。`parc-tools` は空けておくので、いつ打っても構いません。**

---

## 作るもの

```
タスク  KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
スイート libero_10
指示文  turn on the stove and put the moka pot on it
```

## なぜこれなのか — **手本の余白が 40 タスク中 2 番目に薄い**

`solution/tools/analysis/demo_lengths.json`（実測）。**上限は 300 ステップ。**

| 指示文 | 本数 | 手本の中央 | **300 ステップまでの余白** |
|---|---|---|---|
| put both moka pots on the stove（作った分） | 146 | **396** | **無い（残る手本 0 本）** |
| **turn on the stove and put the moka pot on it** | 390 | **265** | **35 ステップ（12%）** |
| pick up the tomato sauce and place it in the basket | 350 | 140 | 2.1 倍 |
| put the bowl on top of the cabinet | 445 | 99 | 3.0 倍 |

**weak3x の失敗映像でも、このタスクは「86 ステップでコンロを点けたが、残り 214 ステップで
ポットを置けず時間切れ」だった**（`outputs/local_eval/0812_084741_weak3x_n5_ep1_max300/videos/sheets/`
の `table_1`）。**方策が遅いのではなく、真似ている手本自体が上限すれすれである。**

**残り 2 件は余白が 2〜3 倍あるので台本では直らない**（掴めない・固まるは
速さの問題ではない）。**だから台本の 2 本目はこれ 1 つで打ち止めにしてよい。**

## 使い回せるもの / 新しく要るもの

BDDL のゴールは 2 つある。

```lisp
(:goal (And (Turnon flat_stove_1) (On moka_pot_1 flat_stove_1_cook_region)))
```

`work/LIBERO-plus/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_6.bddl`

| | |
|---|---|
| **そのまま使える** | 運ぶ物体が `moka_pot`。**掴めるのは蓋のつまみだけ**（原点から +7.2cm。本体は 5cm 角で指は 4.8cm しか開かない） |
| **そのまま使える** | 置き先の名前が `flat_stove_1_cook_region` で SCENE8 と同じ |
| **新しく要る** | **設定値を回す動き。** いまの `TASK_PLANS` は `PickPlace` しか持たない |
| **測り直す** | SCENE3 はコンロの位置が違う。**`measure` から**（SCENE8 では当たる上端 0.9652・鍋の底が手先の 0.1398 下だった） |
| **邪魔物がいる** | 場面に `chefmate_8_frypan_1` がある。**運搬経路でぶつけると採点上は衝突**になる |

初期配置の範囲（`kitchen_table` からの相対）:

| 物 | 範囲 |
|---|---|
| `flat_stove_init_region` | (-0.210, 0.190) 〜 (-0.190, 0.210) |
| `frypan_init_region` | (-0.075, -0.275) 〜 (-0.025, -0.225) |
| `moka_pot_init_region` | (0.025, -0.025) 〜 (0.075, 0.025) |

## 分かっている手順の穴

**`measure` は `TASK_PLANS` に項目が無いと動かない。** 確認済み:

```
$ docker exec parc-tools ... scripted_demo.py measure --task KITCHEN_SCENE3_...
台本が無い: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
  ある台本: KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
```

**だから空の `TaskPlan` を先に置いてから `measure` を呼ぶ順になる。**

## 速さの実測（`parc-tools` の中・Mac）

```
$ docker exec parc-tools env MUJOCO_GL=osmesa python3 solution/tools/scripted_demo.py \
      verify --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove --episodes 0,1,2
成功 3/3  ステップ数 中央値 227 最小 225 最大 233  （上限 300）
  滑らかさ（採点と同じ式・中央値）平均ジャーク 11.58  RMS 22.97  SPARC -4.16
  10 秒
```

**`verify` は 1 本 3.3 秒**（GPU マシンの 2.4 秒とほぼ同じ）。`measure` と `verify` は
描画しないので 2 コアをほとんど使わない。**重いのは `record` だけ。**

## 気をつけてほしいこと 1 つ — **ジャークは採点に乗る**

**SCENE8 の台本はジャーク 11.58 で、方策の実測 4.25 より荒い。**
採点は成功率だけではなく滑らかさ（ジャーク・SPARC・EEF の回転総量）と
実行効率（ステップ数・軌道の総距離）と衝突を重み付けして足す
（予選 PDF p.11。**重みは非公開**）。

**荒い手本は「成功が増えても 1 本あたりの取り分が下がる」形で効きうる。**
SCENE8 でステップ数と歩留まりを譲ってジャークを 38% 下げた判断は正しかったので、
**SCENE3 でも同じ天秤で決めてほしい**（`--smoothing` と `--blend-radius`）。

## 締切から逆算した期限

**`record` の完成は、8/14 の最終学習を始める前**でなければ載らない。
**学習はもう 1 回（2 時間）必要**なので、**8/14 の午前中まで**が実質の期限である。

## 触らないでほしいファイル

`solution/policy_server.py`（提出物そのもの・書き手を 1 人に保つ）／
`docs/strategy/05_実測結果.md`／`docs/talk/open_items.md`／`open_items_notes.md`／
`docs/talk/260813_today.md` の**時刻表と候補一覧の節**（こちらが書いている。
台本の節は今までどおり書き足してください）。

## 参考 — 8/13 13:18 に本番評価から分かったこと（そちらの判断に効く分だけ）

| | |
|---|---|
| **指示文の正規化は効いた** | 版 3.5.0 で **0.181**（過去最高）。**重みは 1 ビットも変えず**、止まっていた 2 本が 94 ステップ・155 ステップで終わるようになった |
| **リーダーボードは最終評価ではない** | 運営 Q&A No.24。**最終提出は別枠で、別のタスク集合**。手元の 36 件のほうが最終評価に近い |
| **1 時間で打ち切りがある** | 予選 PDF p.12。**n=2 では 39〜58 本で尽きる**（実測 0.30 秒/歩）。台本の良し悪しとは別に、最終提出の設定値をこちらで見直す |
