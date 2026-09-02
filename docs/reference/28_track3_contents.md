# 28. Track3 の中身（配布された 4 件と、母集団の作り）

**Track3 だけは課題の定義ファイルが実物で配られている。**Track1・Track2 は一覧で
タスク名を指定する形だが、Track3 は `.bddl`（課題の定義）と初期状態のファイルが入っている
（`b_PARC2026_final/compe/t3/assets/`）。**手元で採点と同じものを走らせられるのは
Track3 だけである**——ただし本番の採点は非公開タスクを含む別の集合だと明記されている。

**この文書は「何が配られていて、母集団がどう作られているか」だけを持つ。**
**だから何をするかは [../plan/17_after_track3_disclosure.md](../plan/17_after_track3_disclosure.md) にある。**

| 節 | 中身 |
|---|---|
| 1 | 配布された 4 件と、その目標条件 |
| 2 | **運営の開示（8/28）—— Track3 は「逆向きのタスク」である** |
| 3 | **混入した検査スクリプトから読める母集団の作り** |

## 1. 配布された 4 件と、その目標条件

| 課題 |
|---|
| take the alphabet soup out of the tray and place it on the table（トレイからスープの缶を出して机に置く） |
| take the red mug off the left plate and put it on the table（左の皿から赤いマグを取って机に置く） |
| take the two mugs off the plates and place them on the table（皿から 2 つのマグを取って机に置く） |
| take the yellow book off the shelf and put it on the table（棚から黄色い本を取って机に置く） |

**どれも「置いてあるものを取って、机の決まった場所に置く」型である。**

| 定義ファイルから読めること | 中身 |
|---|---|
| **目標は「机の狭い領域に置くこと」** | 黄色い本の課題の目標は `(On yellow_book_2 study_table_yellow_book_left_init_region)` で、**領域の広さは 15 cm 角ほど**（`:ranges` が `-0.125 -0.325 0.025 -0.175`）。**`..._init_region` は、その物の元の置き場である** |
| **指示文は定義ファイルの中のほうが詳しい** | ファイル名より長い。例: `take the yellow book off the top of the shelf and put it back on the table on the left` |
| **棚そのものが `:obj_of_interest` に入っている**（`wooden_two_layer_shelf_1`） | つまり**棚を動かしても 1 mm の衝突判定では失格にならない** |

## 2. 運営の開示（8/28）—— Track3 は「逆向きのタスク」である

**2026-08-28 15:12 の告知**（`../materials/notifications/260828_1512_track3_contents.md`）。
配布物に運営内部用の検査スクリプトが混入していたため、**公平性のため全参加者へ開示し、
本選 1 では変更せず実施する**とされた。

> **Track 3 は、LIBERO の既存タスクを逆順にしたタスクで構成されます**
> （例: 「A を B に置く」に対して「B から A に戻す」）。

**1 節の観察が、これで全部つながる。**

| 1 節で書いたこと | 逆向きという説明 |
|---|---|
| どれも「置いてあるものを取って、机の決まった場所に置く」型 | **順方向（机の物をどこかへ置く）の裏返しだから** |
| 目標が「机の狭い領域」で、名前が `..._init_region` | **その物の元の置き場である。**「元に戻す」課題なので当然である |
| 学習データにこの型が 1 件も無い | **LIBERO は順方向しか持っていない。**`libero_10` と `libero_90` の SCENE 名タスク 100 件を数え直しても、`take ... off` / `out of` / `put it back` に当たるものは 0 件である |

**手元で回したときの動きも説明が付く**——アルファベットスープの課題で、方策は
クリームチーズの箱を掴んでトレイへ入れた（**順方向をやっている**。
[../knowledge/17_track3_picks_the_wrong_object.md](../knowledge/17_track3_picks_the_wrong_object.md)）。

## 3. 混入した検査スクリプトから読める母集団の作り

**`b_PARC2026_final/scripts/check_no_leaks.py` は「配布物に残ってはいけない語」の
一覧である。**運営は**このファイルを配布物から削除せず残す**と明言した。
**一覧そのものから、母集団の作りが読める。**

**fixture** = コンロ・棚・引き出しのように、シーンに据え置かれた物（LIBERO のコードの呼び名）。
**掴んで運ぶ対象の物体とは別で、自由関節を持たない。**

| 読めること | 元の記述 |
|---|---|
| **各タスクは順方向の BDDL から作られている** | `source_forward_bddl`（「タスク生成の元 BDDL を示すキー」） |
| **作り方は 4 通りに分類されている** | `A_direct_flip`（直接反転）・`C_pure_primitive`（純粋プリミティブ）・`D_role_swap`（役割交換）・`E_cross_fixture`（fixture またぎ） |
| **Track3 は 2 つの小トラックに分かれる** | `T3a` / `T3b` |
| **元タスクは SCENE 名のものである** | 禁止語に `(KITCHEN\|LIVING_ROOM\|STUDY)_SCENE\d+`「元タスク（順方向 BDDL）の識別子」 |
| **6 つのファミリーに分かれ、各 2 桁の連番がつく** | `[fF][1-6]_\d{2}`「ファミリー記号つき連番」 |
| 内部の命名は `rev_*` / `REV_*` | 「逆転タスクの内部命名」 |
| **評価は 4 つの段に分かれる** | `1_配布` / `2_予選` / **`3_Omni`** / **`4_最終`** |

**元タスクが SCENE 名に限られることが、いちばん効く。**
手元の LIBERO には **SCENE 名のタスクが 100 個**ある（`libero_90` に 90・`libero_10` に 10）。
**そのうち学習データ（`libero_plus` の 40 タスク）に入っているのは `libero_10` の 10 個だけで、
`libero_90` の 90 個は 1 つも入っていない**
（[25_dataset_inventory.md](25_dataset_inventory.md)）。
**`libero_spatial` / `object` / `goal` の平たい名前のタスクは、元になっていない。**

**採点ログの `タスク用途: 3_Omni` は、毎日の枠が第 3 段であることを示している。
最終評価は `4_最終` で別の集合である。**——**毎日の点をそのまま最終成績と読まないこと。**
