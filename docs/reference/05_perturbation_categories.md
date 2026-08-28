# 05. LIBERO-plus の摂動 7 カテゴリと、Track 1 に出る範囲

**「評価側は見た目の摂動だけ」という言い方は強すぎた。** 8/11 に運営 Q&A の原文へ
戻って確かめ直した記録である。**確定していることと、こちらの推測を分けて書く。**

件数は引き写していない。**すべて手元の `task_classification.json` を数えた結果**で、
再現の手順を各節に付けた。

---

## 結論だけ

| | 中身 |
|---|---|
| **確定** | **7 カテゴリ全部ではない**（運営が明言） |
| **確定** | 判定の基準は「同一タスク・同一ドメインにおける、**テクスチャなどの摂動**」 |
| **確定** | 公開されている 4 タスクは `_light_N` と `_table_N` だけ |
| **推測（強い）** | Light Conditions ＋ Background Textures は入る（**2,218 件・22.1%**） |
| **推測（弱い）** | **Sensor Noise も入りうる。** 画素値の歪みは「見た目の摂動」と読めるため |
| **不明** | 難易度の範囲、元タスクが 40 タスク内か（**どちらも回答を断られた**） |

> **前回「2 カテゴリだけ」と書いたのは踏み込みすぎである。** 運営が言ったのは
> 「全部ではない」と基準の文言までで、**カテゴリ名は挙げていない。**

---

## 7 カテゴリの一覧

**摂動の中身は LIBERO-plus 公式リポジトリの README（論文の実装）の記述である。**
件数と接尾辞は手元の分類表を数えた。

| # | カテゴリ | 件数 | 名前の接尾辞 | 何を変えるか | 名前から分かるか |
|---|---|---|---|---|---|
| 1 | Sensor Noise | **1,601** | `_noise_N` | **画素値の歪みと画質の劣化** | ○ |
| 2 | Camera Viewpoints | **1,599** | `_initstate_N` | カメラの位置・向き・画角 | **×** |
| 3 | Robot Initial States | **1,550** | `_initstate_N` | アームの初期姿勢 | **×** |
| 4 | Language Instructions | **1,537** | `_initstate_N` | **指示文を LLM で言い換える** | **×** |
| 5 | Objects Layout | **1,525** | `_add_N`（829）/ `_levelN_sampleM`（696） | 紛らわしい物体を足す・対象物をずらす | ○ |
| 6 | **Light Conditions** | **1,142** | `_light_N` | **明るさ・向き・色・影** | ○ |
| 7 | **Background Textures** | **1,076** | `_table_N`（613）/ `_tb_N`（463） | **背景と面の見た目** | ○ |
| | 合計 | **10,030** | | | |

**下線の 6・7 が Track 1 の本命である**（合計 2,218 件 = 22.1%）。

### 名前だけでは 3 カテゴリを見分けられない

**2・3・4 は接尾辞が同じ `_initstate_N` である。** 分類表の `category` 列だけが
正解を持つ。とくに**指示文の摂動は、違いが名前ではなく instruction 側にある。**

```
Robot Initial States   ..._place_it_on_the_plate_view_0_0_100_0_0_initstate_1
Camera Viewpoints      ..._place_it_on_the_plate_view_0_0_100_2_352_initstate_0
Language Instructions  ..._place_it_on_the_plate_language_1_view_0_0_100_0_0_initstate_0
```

**集計側はこれで一度間違えた**（3 つ全部を Camera Viewpoints として数えていた）。
いまは分類表の CSV を明示して渡し、名前から分からないものは「不明」と言わせている
（[04_track1_population.md](04_track1_population.md)）。

### 接尾辞は末尾で見る

**variant** = 元のタスク 1 つに摂動を 1 つ当てたもの（`.bddl` 1 個）。初期配置は元タスクのものを共有するので、**1 つの variant から最大 50 エピソード作れる。**

`table` と `tb` を「`_table_` を含むか」で分けると **636 / 440** になり、
正しい 613 / 463 と **23 件ずれる。** 名前の途中に `table` を含む `_tb_N` の variant があるためである。

```
pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_tb_1
                          ^^^^^ ここに引っかかる          ^^^^ 本当の接尾辞
```

### 難易度は 1〜5 に散っている

| 難易度 | 1 | 2 | 3 | 4 | 5 | 無記載 |
|---|---|---|---|---|---|---|
| 全 10,030 件 | 1,644 | 2,202 | 2,094 | 1,886 | 2,083 | 121 |
| **6・7 の 2,218 件** | 344 | 468 | **507** | 405 | 373 | 121 |

**2 カテゴリに絞っても難易度は偏らない。** 検証用スイートを 6・7 だけで組んでも、
難易度の幅は母集団と同じものが取れる。

---

## 運営が答えたこと — 原文

**質問（Q&A No.21、種別「評価方法/ルール」、回答者 西浦）**

> (1) リーダーボードおよび予選最終評価で使用されるタスクは、上記 7 カテゴリの
> すべてを含みますか。それとも一部のカテゴリに限定されていますか？
> (2) difficulty_level（1〜5）は全水準が含まれますか？
> (3) 評価タスクの元になっているタスクは、lerobot/libero_plus に含まれる
> 40 タスクの範囲内でしょうか。

**回答**

> （1）**7カテゴリすべてが含まれているわけではありません。** 回答可能な範囲では、
> 予選スライドのTrack 1のタスク詳細で説明されている
> **「同一タスク・同一ドメインにおける、テクスチャなどの摂動」**に該当すると
> 考えられるものが含まれています。
> （2）申し訳ございませんが、回答いたしかねます。
> （3）こちらも申し訳ございませんが、回答いたしかねます。

### ここから読めることと、読めないこと

| | |
|---|---|
| **読める** | 7 つ全部ではない。**基準は「同一タスク・同一ドメイン」＋「テクスチャなどの摂動」** |
| **読めない** | どのカテゴリか。**運営はカテゴリ名を 1 つも挙げていない** |

**基準に当てて 7 つを仕分けると、こうなる。これはこちらの解釈である。**

| カテゴリ | 同一タスク・同一ドメインか | 「テクスチャなど」か | 見立て |
|---|---|---|---|
| **Background Textures** | ○ | **○（そのもの）** | **ほぼ確実に入る** |
| **Light Conditions** | ○ | **○（見た目）** | **ほぼ確実に入る** |
| **Sensor Noise** | ○ | **△（画素値の歪み）** | **入りうる。見落とすと痛い** |
| Camera Viewpoints | ○ | × | 入りにくい |
| Robot Initial States | ○ | × | 入りにくい |
| Language Instructions | ○ | ×（**指示文であって見た目でない**） | 入りにくい |
| Objects Layout | **△（物体が増えるとドメインが変わる）** | × | 入りにくい |

**公開されている 4 タスクが `_light_N` と `_table_N` だけである**という事実が、
上 2 つの見立てを補強している（[02_evaluation_data.md](02_evaluation_data.md)）。

> **Sensor Noise を捨てないこと。** 1,601 件と最多であり、
> **「画質の劣化」は素直に見た目の摂動である。** 学習データを絞るときは
> **`_noise_N` を混ぜておくほうが取りこぼしが小さい**（絞る目的は密度を上げること
> なので、3 カテゴリでも 10,030 件中 3,819 件＝38% と、まだ 2.6 倍の密度になる）。

---

## 論文が報告している弱点 — こちらの状況と噛み合う

README の「Key Findings」より。**評価対象の見立てと合わせて読むと効く。**

| 論文の指摘 | こちらへの意味 |
|---|---|
| **カメラ視点とアームの初期姿勢に極端に弱い**（95% → 30% 未満） | **その 2 つは Track 1 に入りにくい。** つまり**いちばん苦しい摂動は当たらない** |
| **言語指示をほぼ無視している**（Vision-Action モデルのように振る舞う） | **指示文の正規化（項目 8）の期待値が下がる。** モデルが読んでいない |
| **摂動を組み合わせると、単独の効果からは予測できない相互作用が出る** | **1 カテゴリずつ測った成功率を足して本番を予測できない** |

---

## 再現の手順

```bash
# 7 カテゴリの件数・接尾辞・難易度
python3 - <<'PY'
import json, collections, re
from pathlib import Path
raw = json.loads(Path(
    "work/LIBERO-plus/libero/libero/benchmark/task_classification.json").read_text())
entries = [{**r, "suite": r.get("suite", s)} for s, recs in raw.items() for r in recs]
suffix = re.compile(r"_(light|table|tb|noise|add|initstate)_\d+$")
shapes = collections.defaultdict(collections.Counter)
for e in entries:
    found = suffix.search(e["name"])
    shapes[e["category"]][found.group(1) if found else "(なし)"] += 1
for category, count in collections.Counter(e["category"] for e in entries).most_common():
    print(f"{category:24s}{count:6d}  {dict(shapes[category])}")
print("合計", len(entries))
PY
```

```bash
# 運営の回答（No.21）を原文で読む
python3 - <<'PY'
import csv, glob
for row in csv.reader(open(sorted(glob.glob("docs/*260810*.csv"))[0])):
    if len(row) > 6 and row[1].strip() == "21":
        print(row[6].strip())
PY
```

## 出典

| 事実 | 出典 |
|---|---|
| 7 カテゴリの件数・接尾辞・難易度 | `work/LIBERO-plus/libero/libero/benchmark/task_classification.json`（revision `4976dc3`）を数えた |
| **各カテゴリが何を変えるか** | `work/LIBERO-plus/README.md` の「7 Perturbation Dimensions」（論文 *In-depth Robustness Analysis for Vision-Language-Action Models* の公式実装） |
| **7 つ全部ではないこと・判定の基準** | **Q&A CSV No.21（回答者 西浦）の原文** |
| 公開 4 タスクの接尾辞 | [02_evaluation_data.md](02_evaluation_data.md) |
| 論文が報告する弱点 | `work/LIBERO-plus/README.md` の「Key Findings」 |
| 2,218 件・22.1% の内訳 | [04_track1_population.md](04_track1_population.md) |
