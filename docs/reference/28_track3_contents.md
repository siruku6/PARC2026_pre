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
| 4 | **トークナイザは Track3 のボトルネックではない**（9/3 に測った） |
| 5 | **配布 init とタスクの幾何を実測した**（9/6）——本は立っている・腕は home・目標領域は 15 cm 角・**init は 4 タスク × 各 50 行**・**倒れた本は配布にも LIBERO-90 にも無い** |

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

### 1.1 対応する順方向がどこにあるか（**two mugs だけ別扱い**）

**逆向きデータは順方向デモの時刻反転で作る**ので、4 件それぞれに「対応する
順方向」が要る。3 件は `libero_90` に bddl・init・デモが揃っているが、
**two mugs だけ揃っていない。**

| T3 の課題 | 対応する順方向 | bddl / init | デモ |
|---|---|---|---|
| 缶 | `LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray` | libero_90 | あり |
| 赤マグ | `LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate` | libero_90 | あり |
| 本 | `STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf` | libero_90 | あり |
| **two mugs** | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | **libero_90 に無い**。016 が組み立てたもの（`work/probe_016/t3assets`）を使う。**公式配布ではない** | **配布データセットに `task_index 30` で 313 本**（214〜330 コマ） |

**「順方向 2 本の合成が要る」は誤りである。**組み合わせタスクのデモは 1 本で
丸ごと入っている。`libero_90` にあるのは単体 2 本
（`..._put_the_white_mug_on_the_left_plate` と
`..._put_the_yellow_and_white_mug_on_the_right_plate`）だが、**そちらを合成する
必要は無い。**

**two mugs の goal だけ述語が 2 つある。**

```
(And (On porcelain_mug_1   living_room_table_porcelain_mug_init_region)
     (On white_yellow_mug_1 living_room_table_white_yellow_mug_init_region))
```

つまり**1 本の中で 2 つの物を別々の場所へ置く**。逆向きの作り方は
[reference/30 の 7 節](30_reverse_grasp_pose.md) にある。

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

## 4. トークナイザは Track3 のボトルネックではない（**9/3 に測った**）

**結論。トークナイザは学習しない。学習してはいけない。そして Track3 の 4 文は
いま在るトークナイザで 1 文字も取りこぼしていない。**

**測ったもの**（`quant_train` の中で `model_weights/vlm` を `AutoTokenizer` に読ませた）。

| 文 | トークン数 | `<unk>` |
|---|---|---|
| `take the alphabet soup out of the tray …` | 14 | **0** |
| `take the red mug off the left plate …` | 14 | **0** |
| `take the two mugs off the plates …` | 13 | **0** |
| `take the yellow book off the top of the shelf …` | 20 | **0** |
| （比較）学習側の `pick up the alphabet soup and place it in the basket` | 11 | 0 |

**鍵になる語句も、普通の単語のまま分かれている。**
`off the shelf` → `off` / `▁the` / `▁shelf`、`out of the tray` → `out` / `▁of` / `▁the` / `▁tray`、
`alphabet soup` → `alphabet` / `▁soup`。**語彙は 257,152 語で、Track3 に出てくるのは
すべて普通の英単語である。**

**学習してはいけない理由が 2 つある。**

| 理由 | 中身 |
|---|---|
| **ID の割り当てが変わると、事前学習した埋め込みが全部無効になる** | **`embed_tokens.weight` は行番号がトークン ID である。**実物の形が `[257152, 2048]` で、**行数が語彙数とぴったり一致する**（`pi05_rp3000/model.safetensors` のヘッダを読んだ。キーは `model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`）。語彙を作り直すと語と行の対応が崩れるので、**埋め込みを一から学び直すことになる** |
| **推論時に読むのは同梱した `vlm/` である** | `policy_server.py` の 763 行が `tokenizer_processor` の `tokenizer_name` に `<重み>/vlm` を渡す。**学習時と 1 バイトでも違うと、同じ文が違う ID になる** |

**言語側の重みは、すでに学習している。**`--freeze-vision-encoder false --train-expert-only false`
なので、`_set_requires_grad`（**429〜437 行**）は**何も凍結しない**
（行は学習に使っている lerobot 0.6.0 の実体で数えた——
`/opt/lerobot/src/lerobot/policies/pi05/modeling_pi05.py`。常駐コンテナ `quant_train` の中にある）
——PaliGemma（画像側 ＋ Gemma の言語側 ＋ `embed_tokens`）と action expert の全部が対象である。
**学習ログの `num_learnable_params=4143404816` がそれで、行動側だけなら 693,422,112 になる。**

> **だから Track3 で足りていないのは「言語の学習量」ではない。**
> **足りていないのは 2 つである。**①**その言い方をほぼ見ていない**
> ——学習データの指示文 2,066 種のうち `take … off / out of` は **1 件**だけである
> （`open_items.md` の `F72`）。②**似た物の中から選べていない**
> ——9/3 の測定で腕が対象外の物を 10〜50 cm 運んでいた
> （[../knowledge/17_track3_picks_the_wrong_object.md](../knowledge/17_track3_picks_the_wrong_object.md)）。
> **どちらもデータの問題で、トークナイザの問題ではない。**

---

## 5. 配布 init とタスクの幾何——**シムで実測した**（2026-09-06）

**黄色い本の課題（`take_the_yellow_book_off_the_shelf_and_put_it_on_the_table`）について、
配布された `.pruned_init` を実際に読み込んで測った。**逆向きデータを自分で作るときの
前提がここで決まるので、推測で進めないこと。

### 5.1 配布 init の中身——**本は立っており、腕は home にある**

50 本の初期状態のうち先頭 4 本を復元して測った。**4 本とも同じ形である。**

| 測ったもの | 値 |
|---|---|
| 本の姿勢 | **立っている**（厚み軸のワールド z 成分 = 0.000＝厚み軸が水平） |
| 本の下端 | 棚の天板から **+3.2 mm** 浮く |
| 棚の手前端からのはみ出し | **−34 〜 −99 mm**（＝**はみ出していない**。天板の内側に収まっている） |
| 腕の初期 eef 位置 | `[-0.298, 0.000, 1.173]`（**home 相当**。棚から離れている） |

**ここが効く。本が立っていれば厚み軸（23.6 mm）は水平を向くので、横から普通に
挟むだけで掴める。**下に指を入れる隙間も、はみ出しも要らない。

### 5.2 物の寸法とグリッパの開き幅——**幅は挟めない**

| 対象 | 値 |
|---|---|
| 本（`yellow_book_2`）の外形 | **23.6 × 97.8 × 122.2 mm**（box geom の半サイズ ×2） |
| グリッパ（Panda）の最大開き幅 | **約 80 mm**（指の関節可動域 ±0.04 m） |

**2 番目に短い辺（97.8 mm）は 80 mm を超えるので、挟めない。**
掴めるのは最短辺の 23.6 mm だけである。つまり**本のどの向きであっても、
指の閉じ方向を「厚み軸」に合わせるしかない**。

**この帰結が、順方向デモの最終状態を逆向きの初期状態に使うときに効く。**
順方向デモは本を**寝かせて**置くので、厚み軸が鉛直になる。すると挟むには
**下側の指を本の下に入れる**必要があり、天板からの浮きは 3 mm しかないので、
**棚の手前端からはみ出した部分でしか掴めない。**

実測した 3 本（task 92 = `STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf`）:

| エピソード | 順再生後のはみ出し | 逆向きで掴めたか |
|---|---|---|
| 4061 | **+26.5 mm** | **掴めた** |
| 4063 | **+32.6 mm** | **掴めた** |
| 4062 | **+3.4 mm** | **掴めない** |

**はみ出し量が成否をそのまま決めている。**閾値は 3.4 mm（不可）と 26.5 mm（可）の
間にある。

### 5.3 目標領域は 15 cm 角——**順方向 bddl の同名領域と間違えないこと**

`study_table_yellow_book_left_init_region` は**同じ名前で、定義ファイルによって
広さが 7.5 倍違う**。

| 定義ファイル | `:ranges` | 箱の半分 |
|---|---|---|
| 順方向 `libero_90/STUDY_SCENE4_...` | `(-0.060 -0.26 -0.04 -0.24)` | **±10 mm**（物を置く初期配置用の極小領域） |
| **配布 T3** | `(-0.125 -0.325 0.025 -0.175)` | **±75 mm** |

**逆向き goal の判定は必ず配布 T3 の env で行うこと。**順方向 env の中で同名の
述語を評価すると、7.5 倍狭い箱で採点してしまう（実際に一度これで
「境界が厳しすぎる」と誤った結論を出した）。

### 5.4 順方向デモと配布 T3 は、物の姿勢が一致しない

| | 本の姿勢 | はみ出し | 腕 |
|---|---|---|---|
| **配布 T3 の init** | 立っている | 無し | home |
| 順方向デモ（task 92）の最終状態 | **寝ている** | +3〜+33 mm | 棚の上（放した直後） |

**運営の言う「逆順」は意味の上での逆であって、順方向デモの最終状態がそのまま
T3 の初期状態になっているわけではない。**順方向デモの最終状態を逆向きの初期状態に
使うと、**本番には無い難しさ（寝た本を掴む）が入る。**

### 5.5 配布 init は「4 件」ではなく **4 タスク × 各 50 行**である

**`.pruned_init` は 1 ファイルに初期状態が 1 本ではなく 50 本入っている。**
`compe/t3/assets/` に入っているファイルが 4 つなのは**タスクが 4 つだから**であって、
初期状態が 4 通りという意味ではない。デモ軌道（hdf5/parquet）は**1 本も入っていない**。

| タスク | `.pruned_init` の shape |
|---|---|
| take_the_alphabet_soup_out_of_the_tray… | (50, 97) |
| take_the_red_mug_off_the_left_plate… | (50, 84) |
| take_the_two_mugs_off_the_plates… | (50, 84) |
| take_the_yellow_book_off_the_shelf… | **(50, 58)** |

各行は MuJoCo のシム状態（時刻＋qpos＋qvel）で、`set_init_state()` にそのまま渡せる。
列数がタスクごとに違うのは、シーンに置かれている物の数が違うため。

**50 行は本当に散っている**（黄色い本の課題で、列ごとの分散から測った）。

| 振られている量 | 幅 |
|---|---|
| 本の x | **114 mm**（−0.256 〜 −0.142） |
| 本の y | **49 mm**（+0.272 〜 +0.321） |
| 本のヨー角 | **約 43°** |
| 本の z | 一定（1.0953＝棚の天板の上） |

**逆向きデータの初期状態の多様性は、この 50 行がそのまま使える。**
本を自分で置き直す必要はない。

**ただし倒れた本は 1 行も無い。**50 行のクォータニオンは x, y 成分の最大が
**0.0009**（傾きにして 0.05°）で、振られているのはヨーだけである。
**LIBERO-90 にも無い**——`libero_90` の init のうち book を含む 28 ファイル
（50 行版・100 行版とも）は**すべて `|qx|, |qy| = 0.0000`（厳密にゼロ）**だった。
寝た本が出てくるのは「棚や仕切りに本を平らに置いた**後**」＝デモの結果としてだけで、
**初期状態としては学習データのどこにも存在しない。**

### 5.6 env をまたいで物の姿勢を写すときの落とし穴

**(a) 同じ名前の fixture でも、env が違えば場所が違う。**

| | `wooden_two_layer_shelf_1_top_side` の位置 |
|---|---|
| 順方向 `libero_90/STUDY_SCENE4_...` | `(-0.2046, 0.3051, 1.0939)` |
| 配布 T3 | `(-0.1932, 0.2866, 1.0939)` |
| | **ずれ 21.7 mm**（z は一致） |

どちらも `seed=42` で fixture を固定した上での値である。**順再生の最終状態の
物の姿勢を、別の env の init へ座標のままコピーしてはいけない。**はみ出し量は
数 mm で掴めるかどうかが変わるので、21.7 mm は結論を変える。**基準になる site
（順方向 goal の領域 site）の座標系で測り直してから差し込むこと。**

**(b) `yellow_book_2` の body 原点は本の**底面**にある。**

`qpos` の xyz は本の中心ではない。実測で body の z = 1.09525、geom の AABB は
z = 1.09496 〜 1.22036（高さ 125.4 mm）。**中心だと思って天板の高さと比べると
「本が天板に埋まっている」ように見える。**

| 本のローカル軸 | geom 半サイズ | 立っているときの向き |
|---|---|---|
| **x** | **13.7 mm（厚み）** | 水平 |
| y | 55.4 mm | 水平 |
| z | 62.6 mm | **鉛直** |

**厚み軸はローカル x** なので `object_is_upright(quat, thin_axis_local=0)` の
既定は正しい。ただしヨーだけの回転では x 軸は必ず水平になるため、**この判定は
「倒れているか」を見分けるのには使えても、「厚み軸がどれか」の検証にはならない。**

**シムを回すときの落とし穴は
[28_mppi_data_regeneration_survey.md](28_mppi_data_regeneration_survey.md) の 5.1〜5.2 節に
まとめてある。**

