# 36. 指示文の書き換えの対応表 —— **採点器が送る 16 文が、モデルに何として届くか**

> **状態: 測って並べた（2026-09-12）** ／ 依頼は
> [talk/inter_sessions/031](../talk/inter_sessions/031_instruction_text_rewriting.md)
>
> **しきい値は動かしていない。`solution/policy_server.py` は 0.65 のままである。**
> ここは判定の材料だけを置く。**決めるのはユーザである。**
> 穴の話は [knowledge/23](../knowledge/23_self_made_demos_need_the_instruction_the_model_sees.md)、
> 8/27 の事故は [knowledge/15](../knowledge/15_instruction_normalizer_inverted_a_task.md) にある。

**読み終えたときに分かること。**
採点器が送る 16 文が、`InstructionNormalizer` を通って**モデルに何として届くか**。
そして**その書き換えが正しいかどうかを、何を根拠に言えるか。**

| 節 | 中身 |
|---|---|
| [1 なぜこの表が要るか](#1-なぜこの表が要るか--065-と-099-のどちらが正しいかを決める根拠がない) | 提出済みの zip でしきい値が割れている |
| [2 文はどこにあるか](#2-文はどこにあるか--4-か所) | 一次資料の場所と行番号 |
| [3 正しさを何で測るか](#3-正しさを何で測るか--task_id-という答え合わせ) | **`task_id` が答えを持っていた** |
| [4 対応表（16 行）](#4-対応表16-行) | **この節が成果物である** |
| [5 意味が変わる 2 行の根拠](#5-意味が変わる-2-行の根拠--bddl-を読む) | bddl の `:init` と `:goal` |
| [6 しきい値 3 点の比較](#6-しきい値-3-点の比較) | 0.50 / 0.65 / 0.99 |
| [7 本番のタスクは未知である](#7-本番のタスクは未知である--16-文で誤りゼロが保証すること) | この表が保証すること・しないこと |
| [8 同じ表を出し直す](#8-同じ表を出し直す) | 打つコマンド |

---

## 1. なぜこの表が要るか —— **0.65 と 0.99 のどちらが正しいかを決める根拠がない**

`InstructionNormalizer` は、採点器が送ってくる文を**学習データの言い方へ寄せてから
モデルへ渡す。**予選ではこれが効いた（版 3.1.0 → 3.5.0 で 0.166 → 0.181・
歩数 1,922 → 1,617）。運営 Q&A の No.29／No.31 で認められた処理である。

**ところが、提出した zip ごとにしきい値が違う。**

| どこ | しきい値 | 意味 |
|---|---|---|
| `solution/policy_server.py`（いまの木） | **0.65** | 8/27 に 0.50 から上げた |
| 版 5.2.0（9/1）・版 14.0.0（9/11）の Track3 | **0.99** | どの文にも当たらない＝寄せを実質切ってある |

**どちらが正しいかを決める根拠が「1 件の誤りを見つけた」しか無かった。**
寄せる回数を減らせば安全側だが、**寄せることで得ていた利益も減る。**
どの文が得をして、どの文が損をしているかは、1 行ずつ見ないと分からない。
**この表はそのために作った。**

---

## 2. 文はどこにあるか —— **4 か所**

| # | 何の文か | 場所 |
|---|---|---|
| 1 | **採点器が送る文**（Track1・2） | `~/src/b_PARC2026_final/compe/t1/T1_TASKS.csv`・`t2/T2_TASKS.csv` の **`instruction` 列** |
| 2 | **採点器が送る文**（Track3） | `~/src/b_PARC2026_final/compe/t3/assets/bddl_files/libero_t3/*.bddl` の **`(:language …)`** |
| 3 | **寄せ先の候補**（40 種） | `work/artifacts/libero_plus/meta/tasks.parquet` の索引 |
| 4 | **寄せる実装** | `solution/policy_server.py` の `InstructionNormalizer`（324〜325 行） |

**1 と 2 は合わせて 16 文である**（Track1 が 4・Track2 が 8・Track3 が 4）。

**3 は実装の 40 種と完全に一致していることを確かめた。**
`_TRAINING_INSTRUCTIONS`（`policy_server.py`）と `tasks.parquet` の索引 40 件は
集合として同じである。**寄せ先の候補が実装と学習データでずれていることはない。**

**寄せるかどうかは 2 つの門で決まる。**点数だけではない。

| 門 | 値 | 意味 |
|---|---|---|
| `_MIN_LEXICAL` | **0.65** | 1 番の点数がこれ未満なら寄せない |
| `_MIN_LEXICAL_MARGIN` | **0.05** | **1 番と 2 番の差**がこれ未満なら寄せない（迷ったら寄せない） |

**`exact`（学習データと同じ文）は、この 2 つの門を通らない。**辞書引きで自分に
写るだけなので、**しきい値を 0.99 にしても素通りにはならない。**

---

## 3. 正しさを何で測るか —— **`task_id` という答え合わせ**

**依頼は bddl の `:goal` を読んで確かめよと書いていた。**それはやった（5 節）。
**だがその前に、もっと強い答え合わせが CSV の中にあった。**

`T1_TASKS.csv`・`T2_TASKS.csv` は、**送ってくる文と、採点器が実際に読み込む課題を
同じ行に持っている。**

```
task_num,task_id,instruction,libero_plus_id,suite,category,difficulty_level
1,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_table_2,Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate,80,...
     ^^^^^^^^^^^^^^^^^^ 採点器が読み込む課題 = 素の LIBERO タスク名 + 摂動の印
```

`compe/t1/register.py` は、この `task_id` から bddl と初期状態を引き、
`instruction` 列をそのまま `Task.language` にしてモデルへ送る。
**つまり `task_id` は「この文が本当は何のタスクか」の答えそのものである。**
末尾の摂動の印（`_table_2`・`_view_…`・`_noise_…` など）を落とせば素の
タスク名が残り、**寄せた先の文と 1 文字ずつ比べられる。**

**結果は 12 / 12 で一致した。**

```
$ python3 solution/tools/suite/audit_grader_instructions.py --task-id
...
一致 12 / 12
```

**Track1・Track2 の 12 文すべてで、寄せた先は採点器が読み込む課題そのものだった。**
`top layer` → `top drawer` も `on the bowl` → `in the bowl` も、
**`task_id` が「それが正しい」と言っている。**

> **Track3 の 4 文にはこの答え合わせが使えない。** bddl には `task_id` に当たる
> 欄が無い。4 文とも素通りするので（6 節）、いまは寄せ先の正しさを問う場面が無い。

---

## 4. 対応表（16 行）

**`語の差` は語の単位の差分である。**`∅` は「その位置に語が無い」という意味。
**`分類` は 3 段階**で、根拠が要るものは 5 節に書いた。

### 4.1 書き換えが起きる 10 行（Track1・Track2）

| # | 出典 | 採点器が送る文 | モデルが見る文 | 語の差 | 点 / 差 | 分類 |
|---|---|---|---|---|---|---|
| 1 | `T1_TASKS.csv:2` | Pick the akita black bowl in the top **layer** of the wooden cabinet and place it on the plate | pick **up** the black bowl in the top **drawer** of the wooden cabinet and place it on the plate | `∅`→`up` ／ `akita`→`∅` ／ **`layer`→`drawer`** | 0.708 / **0.077** | **呼び名**（5.1） |
| 2 | `T1_TASKS.csv:3` | **Pick** the tomato sauce and place it in the basket | **pick up** the tomato sauce and place it in the basket | `∅`→`up` | 0.970 / 0.326 | 無害 |
| 3 | `T1_TASKS.csv:4` | **Pick** the milk and place it in the basket | **pick up** the milk and place it in the basket | `∅`→`up` | 0.963 / 0.439 | 無害 |
| 4 | `T2_TASKS.csv:2` | **Pick** the orange juice and place it in the basket | **pick up** the orange juice and place it in the basket | `∅`→`up` | 0.975 / 0.543 | 無害 |
| 5 | `T2_TASKS.csv:3` | Put the cream cheese **on** the bowl | put the cream cheese **in** the bowl | **`on`→`in`** | 0.902 / 0.315 | **関係**（5.2） |
| 6 | `T2_TASKS.csv:4` | **Pick** the ketchup and place it in the basket | **pick up** the ketchup and place it in the basket | `∅`→`up` | 0.963 / 0.439 | 無害 |
| 7 | `T2_TASKS.csv:5` | **Pick** the salad dressing and place it in the basket | **pick up** the salad dressing and place it in the basket | `∅`→`up` | 0.975 / 0.543 | 無害 |
| 8 | `T2_TASKS.csv:6` | Pick the **akita** black bowl on the **cookies** box and place it on the plate | pick **up** the black bowl on the **cookie** box and place it on the plate | `∅`→`up` ／ `akita`→`∅` ／ `cookies`→`cookie` | 0.805 / 0.121 | 呼び名 |
| 9 | `T2_TASKS.csv:8` | Pick the **akita** black bowl on the ramekin and place it on the plate | pick **up** the black bowl on the ramekin and place it on the plate | `∅`→`up` ／ `akita`→`∅` | 0.762 / 0.121 | 呼び名 |
| 10 | `T2_TASKS.csv:9` | Pick the **akita** black bowl between the plate and the ramekin and place it on the plate | pick **up** the black bowl between the plate and the ramekin and place it on the plate | `∅`→`up` ／ `akita`→`∅` | 0.818 / 0.144 | 呼び名 |

### 4.2 書き換えが起きない 6 行

**`exact` の 2 行は、学習データと同じ文である。**小文字化だけで自分に写る。

| # | 出典 | 文 | どうなるか | 点 / 差 | 分類 |
|---|---|---|---|---|---|
| 11 | `T1_TASKS.csv:5` | Put the bowl on the stove | `exact`・小文字化のみ | 1.000 / 0.346 | 無害 |
| 12 | `T2_TASKS.csv:7` | Turn on the stove | `exact`・小文字化のみ | 1.000 / 0.367 | 無害 |

**Track3 の 4 行は 0.65 に届かないので素通りする。**`2 番手` は、寄ってしまった
場合に飛ばされる先である。

| # | 出典 | 文 | 1 番手（寄りかけた先） | 点 / 差 | 分類 |
|---|---|---|---|---|---|
| 13 | `take_the_alphabet_soup_out_of_the_tray_….bddl:3` | take the alphabet soup out of the tray and place it on the table in front of the ketchup | pick up the alphabet soup and place it in the basket | 0.402 / **0.047** | 無害（素通り） |
| 14 | `take_the_red_mug_off_the_left_plate_….bddl:3` | take the red mug off the left plate and put it back on the table behind the other two mugs | put the white mug on the left plate and put the yellow and white mug on the right plate | 0.317 / 0.096 | 無害（素通り） |
| 15 | `take_the_two_mugs_off_the_plates_….bddl:3` | take the white mug off the left plate and the yellow and white mug off the right plate, then place … | put the white mug on the left plate and put the yellow and white mug on the right plate | **0.607** / 0.156 | **0.50 だと逆に寄る**（5.3） |
| 16 | `take_the_yellow_book_off_the_shelf_….bddl:3` | take the yellow book off the top of the shelf and put it back on the table on the left | pick up the book and place it in the back compartment of the caddy | 0.357 / **0.025** | 無害（素通り） |

### 4.3 内訳

| 分類 | 件数 | どの行か |
|---|---|---|
| **無害**（大文字小文字・`pick`→`pick up`・素通り） | **11** | 2・3・4・6・7・11・12・13・14・15・16 |
| **呼び名が変わる**（`akita` が落ちる・単複・`layer`→`drawer`） | **4** | **1**・8・9・10 |
| **関係が変わる**（前置詞・方向・否定） | **1** | **5** |

**寄せ先が誤っている行は 0 件である**（3 節の `task_id` 突き合わせで 12 / 12 一致）。

---

## 5. 意味が変わる 2 行の根拠 —— **bddl を読む**

**依頼は「関係が変わる」に入った行だけ根拠を書けと言っていた。**
**5.1 は「呼び名」に分類したが、同じ根拠を付けた。**理由は本文に書く。

根拠に使った bddl は、評価イメージ `parc2026-final` の中の
`/workspace/LIBERO-plus/libero/libero/bddl_files/` から取り出したものである
（8 節に取り出し方）。

### 5.1 `top layer` → `top drawer` —— **同じ机に黒い椀が 2 つある**

**この行は「呼び名」だが、危険度は「関係」と同じである。**
`bddl_files/libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_table_2.bddl`
を読むと、**黒い椀が 2 つ置いてある。**

```lisp
(:objects
  akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl   ; 見分けのつかない椀が 2 つ
  ...)
(:obj_of_interest
  akita_black_bowl_1     ; 採点されるのはこちら
  plate_1)
(:init
  (In akita_black_bowl_1 wooden_cabinet_1_top_region)   ; 引き出しの中
  (On akita_black_bowl_2 wooden_cabinet_1_top_side)     ; 棚の上（囮）
  (Open wooden_cabinet_1_top_region)                    ; 引き出しは開いている
  ...)
(:goal
  (And (On akita_black_bowl_1 plate_1)))
```

**`top layer`（上の段）は、引き出しの中とも棚の上とも読める。**
そして**どちらにも椀がある。**取り違えれば `akita_black_bowl_2` を掴むことになり、
`(On akita_black_bowl_1 plate_1)` は永久に満たされない。

**書き換えは正しい。**寄せた先は引き出し（`In … top_region`）を指しており、
`task_id` もそう言っている（3 節）。

**そして 2 番手が囮そのものである。**

| | 文 | 点 |
|---|---|---|
| 1 番手 | pick up the black bowl **in the top drawer of** the wooden cabinet and place it on the plate | 0.708 |
| 2 番手 | pick up the black bowl **on** the wooden cabinet and place it on the plate | 0.631 |

**差は 0.077 しかない**——16 文のうちで最も狭い。2 番手は `akita_black_bowl_2`
（棚の上の椀）のタスクである。**差の門（0.05）がぎりぎりで正解側を守っている。**

> **判定してほしいのはここである。**書き換えは正しいが、**余裕が 0.077 と 0.05 の
> 差しかない。**未知の言い回しで語が 1 つ動けば、囮側へ倒れうる。

### 5.2 `on the bowl` → `in the bowl` —— **前置詞は何も指していない**

`bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl` はこうなっている。

```lisp
(:objects
  akita_black_bowl_1 - akita_black_bowl   ; 椀は 1 つだけ
  cream_cheese_1 - cream_cheese
  ...)
(:init
  (On cream_cheese_1 main_table_cream_cheese_region)
  (On akita_black_bowl_1 main_table_akita_black_bowl_region))
(:goal
  (And (On cream_cheese_1 akita_black_bowl_1)))   ; 述語は On である
```

**2 つのことが分かる。**

1. **椀は 1 つしかない。**`on` と `in` のどちらで呼んでも、動かす物と行き先は
   同じ対（`cream_cheese_1` と `akita_black_bowl_1`）である。**前置詞は行き先を
   選り分けていない。**
2. **`:goal` の述語は `On` である。**タスク名は「`in` the bowl」なのに、
   採点条件は `On` で書かれている。**LIBERO 自身が椀に対して `on` と `in` を
   区別していない。**

**書き換えは無害である。**位置関係は変わらない。**変わるのは呼び方だけで、
それを学習データ側の呼び方に揃えている。**`task_id` も
`put_the_cream_cheese_in_the_bowl` である。

### 5.3 8/27 に見つけた 1 件 —— **寄せ先が `:init` そのものだった**

**これは素通りする行だが（0.607 < 0.65）、しきい値を 0.50 に戻すと寄る。**
何が起きるかを `compe/t3/assets/bddl_files/libero_t3/take_the_two_mugs_off_the_plates_and_place_them_on_the_table.bddl`
で確かめた。

```lisp
(:init
  (On porcelain_mug_1 plate_1)          ; 白い椀は左の皿の上にある
  (On white_yellow_mug_1 plate_2))      ; 黄白の椀は右の皿の上にある
(:goal
  (And (On porcelain_mug_1 living_room_table_porcelain_mug_init_region)
       (On white_yellow_mug_1 living_room_table_white_yellow_mug_init_region)))
                                        ; どちらも机の上へ下ろせ＝皿から外せ
```

寄せ先の候補（0.607）は
`put the white mug on the left plate and put the yellow and white mug on the right plate` である。

**これは `:init` の状態を言葉にしたものである。**つまり寄せてしまうと、
**モデルは「始まりの状態を作れ」と言われる**——`:goal` のちょうど逆である。
**8/27 に 0.65 へ上げた判断は、この bddl で裏が取れている。**

> **この課題には学習データが 1 本も無い。** `reverse_lerobot_v2`（3 種）と
> `jaist_mix_v2_x8`（43 種 = 40 + 3）を調べたところ、**Track3 の 4 文のうち
> 3 文は 1 文字も違わずに入っているが、この 2 つのマグの文だけが入っていない。**
> **いちばん寄せ間違えやすい文が、いちばん手本を持っていない。**

---

## 6. しきい値 3 点の比較

**`exact` の 2 行はしきい値を通らない**（2 節）ので、どの値でも動かない。

| # | 出典 | 点 | 差 | **0.50** | **0.65**（いまの木） | **0.99** |
|---|---|---|---|---|---|---|
| 1 | `T1_TASKS.csv:2` | 0.708 | 0.077 | 寄る | **寄る** | 素通り |
| 2 | `T1_TASKS.csv:3` | 0.970 | 0.326 | 寄る | **寄る** | 素通り |
| 3 | `T1_TASKS.csv:4` | 0.963 | 0.439 | 寄る | **寄る** | 素通り |
| 11 | `T1_TASKS.csv:5` | 1.000 | 0.346 | exact | **exact** | exact |
| 4 | `T2_TASKS.csv:2` | 0.975 | 0.543 | 寄る | **寄る** | 素通り |
| 5 | `T2_TASKS.csv:3` | 0.902 | 0.315 | 寄る | **寄る** | 素通り |
| 6 | `T2_TASKS.csv:4` | 0.963 | 0.439 | 寄る | **寄る** | 素通り |
| 7 | `T2_TASKS.csv:5` | 0.975 | 0.543 | 寄る | **寄る** | 素通り |
| 8 | `T2_TASKS.csv:6` | 0.805 | 0.121 | 寄る | **寄る** | 素通り |
| 12 | `T2_TASKS.csv:7` | 1.000 | 0.367 | exact | **exact** | exact |
| 9 | `T2_TASKS.csv:8` | 0.762 | 0.121 | 寄る | **寄る** | 素通り |
| 10 | `T2_TASKS.csv:9` | 0.818 | 0.144 | 寄る | **寄る** | 素通り |
| 13 | `take_the_alphabet_soup_….bddl:3` | 0.402 | 0.047 | 素通り | 素通り | 素通り |
| 14 | `take_the_red_mug_….bddl:3` | 0.317 | 0.096 | 素通り | 素通り | 素通り |
| 15 | `take_the_two_mugs_….bddl:3` | 0.607 | 0.156 | **寄る（誤り）** | 素通り | 素通り |
| 16 | `take_the_yellow_book_….bddl:3` | 0.357 | 0.025 | 素通り | 素通り | 素通り |

**判定が変わる行は 11 行**（16 行中）。**ただし変わり方は 2 つに割れている。**

| 動かす区間 | 変わる行 | 中身 |
|---|---|---|
| **0.50 → 0.65** | **1 行** | 15 だけ。**誤って寄る 1 件が消える。正しい書き換えは 1 つも失わない。** |
| **0.65 → 0.99** | **10 行** | 正しい書き換えが**全部消える。**残るのは `exact` の 2 行だけ |

**これが取引の形である。**

- **0.65 は、誤りを消して利益を全部残している。**16 文の上では最も良い。
- **0.99 は、誤りも利益も両方捨てる。**素通りの 4 行（Track3）には何の影響も無い
  ——**あの 4 行は 0.65 でも既に素通りしている。**

> **提出済みの zip について。**版 5.2.0（9/1）と版 14.0.0（9/11）は Track3 を
> 0.99 にしてあるが、**この 16 文で見るかぎり 0.99 と 0.65 は Track3 で同じ
> 振る舞いをする**（4 行とも素通り）。**配布の 4 文に対しては、あの変更は
> 何も変えていない。**効くとすれば本番の未知の文に対してだけである（7 節）。

---

## 7. 本番のタスクは未知である —— **「16 文で誤りゼロ」が保証すること**

**16 文は配布されている example であって、本番のタスク集合ではない。**
`compe/t3/README.md` に「本番評価は非公開タスクを含む別のタスクセットで実施される」
とある。だからこの表の読み方には限りがある。

**保証すること。**採点器が**この言い方の癖**（`Pick` と書く・`akita` を付ける・
`cookies` と複数形にする・`top layer` と呼ぶ）で文を作っているかぎり、
**0.65 は 12 文すべてを正しい課題へ戻す。**癖は課題ごとにばらばらではなく、
**同じ書き換え規則が全体にかかっている**ように見える（10 行のうち 9 行が
`pick`→`pick up` を含む）。本番も同じ規則で作られているなら、同じように効く。

**保証しないこと。3 つある。**

1. **未知の文が、どの既存タスクにも当たらないとは言えない。**15 の 2 つのマグの
   文がそうだったように、**逆向きの文が語の重なりだけで 0.6 台に届くことがある。**
   0.65 はその 1 件の上に置いた線であって、**0.65 と 0.708 の隙間は 0.058 しかない。**
   本番に 0.65〜0.71 の誤りが 1 件でもあれば、この線は守らない。
2. **差の門（0.05）の余裕が足りない行がある。**行 1 は差が 0.077 で、
   **囮の椀のタスクが 2 番手に付けている**（5.1）。語が 1 つ動けば逆へ倒れうる。
3. **16 文は Track3 の書き換えを 1 度も試していない。**4 行とも素通りなので、
   **Track3 で寄せ先が正しいかどうかは、この表では 1 件も検証できていない。**
   0.99 の是非をこの 16 文から決めることはできない。

**まとめると、この表が言えるのは「0.65 は 0.50 より厳密に良く、0.99 より
Track1・Track2 で良い」までである。**Track3 で 0.65 と 0.99 のどちらが良いかは、
**配布の 4 文がどちらでも素通りする以上、この一次資料からは決まらない。**

---

## 8. 同じ表を出し直す

**道具は `solution/tools/suite/audit_grader_instructions.py` にある。**
提出物の `InstructionNormalizer` をそのまま呼ぶので、
`policy_server.py` のしきい値を変えればこの表も動く。

```bash
cd ~/src/PARC2026_pre

# 4 節の対応表（16 文・寄せ先・2 番手・語の差）
python3 solution/tools/suite/audit_grader_instructions.py

# 6 節のしきい値 3 点
python3 solution/tools/suite/audit_grader_instructions.py --sweep

# 3 節の答え合わせ（寄せ先 vs task_id）
python3 solution/tools/suite/audit_grader_instructions.py --task-id
```

**torch も重みも要らない。**`policy_server.py` の `class MyPolicy` より前だけを
実行するので、素の python で通る。

**5 節が読んでいる bddl は評価イメージの中にある。**取り出すにはこうする。

```bash
IMAGE=parc2026-final bash solution/tools/regen/run_in_container.sh \
    solution/tools/suite/dump_libero_bddl.py
# -> work/instr_audit/bddl_files/ に 16,334 件
```

**言い換え 1,527 種を含む広い母集団で傾向を見たいときは、こちらではなく
[`check_instruction_normalization.py`](../../solution/tools/suite/check_instruction_normalization.py) を使う**
（`--sweep` でしきい値を振れる）。**役割が違う**——あちらは傾向、ここは 16 文の判定である。
