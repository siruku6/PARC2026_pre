# 04. Track 1 に出そうな変種の母集団

外部でまとめられた LIBERO-plus の集計（2026-08-11 に共有されたもの）を、
**手元の LIBERO-plus とベースモデルのソースで数え直した記録**である。

数字は引き写していない。**すべて `work/LIBERO-plus` の実物と
`work/lerobot-src` のソースを数えた結果**で、各節に再現の手順を付けた。

> **なぜ数え直したか。** 母集団の比が分かると、検証用スイートの 84 件が
> 母集団を代表しているかを判定できる。**代表していない軸があれば、
> そこの弱さは測っても見えない。**

---

## 結論だけ

| 分かったこと | 効くところ |
|---|---|
| Track 1 候補は **2,218 件**（全 10,030 件の 22.1%）。`light` 1,142 / `table` 613 / `tb` 463 | 検証用スイートの割り方 |
| **接尾辞は末尾で判定しないと 23 件ずれる**（`_table_` を名前の途中に含む `_tb_` の変種） | 集計コード。**直した** |
| 衝突しうる相手は平均 **3.10 個**・最多 5 個。**55 件は 0 個** | 衝突率の読み方。**CSV に列を足した** |
| LeRobot の画像拡張は**明るさ ±20% まで**。評価側の振れ幅（×0.71〜×1.70）を**覆えていない** | 学習の設定（項目 22） |
| ただし**学習データ自体は 0.086〜0.723 を含む。** 足りないのは拡張ではなく密度の可能性 | 同上。**先に測る** |

---

## Track 1 候補は 2,218 件

運営 Q&A No.21 は「同一タスク・同一ドメインにおける、テクスチャなどの摂動」が
含まれると答えている（7 カテゴリ全部ではない）。公開されている example 4 件も
**Background Textures と Light Conditions の 2 カテゴリだけ**である
（[02_evaluation_data.md](02_evaluation_data.md)）。

> **8/11 追記。運営はカテゴリ名を 1 つも挙げていない。**
> 「2 カテゴリだけ」はこちらの推測である。**Sensor Noise（1,601 件・最多）も
> 「画質の劣化」として基準に当てはまりうる。** 原文と 7 カテゴリの仕分けは
> [05_perturbation_categories.md](05_perturbation_categories.md) に分けた。

この 2 カテゴリを候補と見て数えると次のようになる。

| 区分 | 件数 |
|---|---|
| 分類表の全件 | 10,030 |
| **Background Textures ＋ Light Conditions** | **2,218（22.1%）** |
| うち `_light_N` | 1,142 |
| うち `_table_N` | 613 |
| うち `_tb_N` | 463 |
| 元になる primitive | 40 |

スイート別も偏っていない（libero_10 563 / libero_goal 560 / libero_spatial 550 /
libero_object 545）。難易度は 1〜5 に散り、L3 が最多で 507 件、121 件は無記載である。

```bash
python3 - <<'PY'
import json, collections, re
from pathlib import Path
raw = json.loads(Path(
    "work/LIBERO-plus/libero/libero/benchmark/task_classification.json"
).read_text())
entries = [{**r, "suite": r.get("suite", s)} for s, recs in raw.items() for r in recs]
t1 = [e for e in entries
      if e["category"] in ("Background Textures", "Light Conditions")]
tail = re.compile(r"_(light|table|tb)_\d+$")
print(len(entries), len(t1), f"{len(t1) / len(entries):.1%}")
print(collections.Counter(tail.search(e["name"]).group(1) for e in t1))
print(collections.Counter(e["suite"] for e in t1))
PY
```

**外部の集計と 1 件も違わなかった**（2,218 / 22.1% / 1,142 / 613 / 463 / 40 /
545〜563）。**この母集団の定義は信頼して使える。**

> **ただし「候補」である。** 本番の出題は非公開で、運営は「元タスクが
> `lerobot/libero_plus` の 40 タスクの範囲内か」には回答していない（Q&A No.21）。

## 接尾辞は末尾で見る — 含むかどうかで見ると 23 件ずれる

`table` と `tb` を「名前に `_table_` を含むか」で分けると **636 / 440** になる。
正しい 613 / 463 との差 23 件は、**名前の途中に `table` を含む `_tb_N` の変種**である。

```
pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_tb_1
                          ^^^^^ ここに引っかかる          ^^^^ 本当の接尾辞
```

[summarize_gpu_eval.py](../../solution/tools/summarize_gpu_eval.py) の
カテゴリ判定はこの形（含むかどうか・`_table_` を `_tb_` より先に見る）だったので、
**末尾一致に直した**。試験で固定してある
（[tests/test_rollout_report_line.py](../../tests/test_rollout_report_line.py)）。

**同時に、もっと大きな穴が見つかった。** 変種名から摂動カテゴリを当てるのは
**7 つのうち 4 つまでしかできない。**

| カテゴリ | 名前の形 | 名前から分かるか |
|---|---|---|
| Light Conditions | `..._light_N` | ○ |
| Background Textures | `..._table_N` / `..._tb_N` | ○ |
| Sensor Noise | `..._view_…_noise_N` | ○ |
| Objects Layout | `..._add_N` / `..._levelN` | ○ |
| **Camera Viewpoints** | `..._view_…_initstate_N` | **×** |
| **Language Instructions** | `..._view_…_initstate_N` | **×** |
| **Robot Initial States** | `..._view_…_initstate_N` | **×** |

下 3 つは**名前がまったく同じ形**で、分類表（CSV の `category` 列）だけが
正解を持つ。とくに指示文の摂動は、**違いが名前ではなく instruction 側にある。**

```bash
# 3 カテゴリがすべて同じ形であることを数える
python3 - <<'PY'
import json, collections, re
from pathlib import Path
raw = json.loads(Path(
    "work/LIBERO-plus/libero/libero/benchmark/task_classification.json"
).read_text())
by = collections.defaultdict(collections.Counter)
for suite, records in raw.items():
    for record in records:
        stem = re.search(r"_(light|table|tb|noise|add|initstate)_\d+$", record["name"])
        by[record["category"]][stem.group(1) if stem else "-"] += 1
for category, shapes in sorted(by.items()):
    print(f"{category:24s}", dict(shapes))
PY
```

**そのため集計側を 2 つ直した。**

| 直したこと | それまで何が起きていたか |
|---|---|
| CSV を `run_gpu_eval.sh` から明示して渡す | 探す場所が 1 段ずれていて**見つかっていなかった**（`work/` が抜けていた） |
| 名前から当てられないものは「不明」と言う | 全部 Camera Viewpoints として数えていた。**学習の的を決める表が静かに間違う** |

## 衝突しうる相手は、タスクによって 0 個から 5 個まで

採点は「操作対象以外の物体が初期位置から 1mm を超えて動いたら失格」である
（[pipeline/rollout.py](../../pipeline/rollout.py) の
`success = bool(done) and not collided`）。**相手が何個いるかはタスクによって違う。**

Track 1 候補 2,218 件の bddl を全部読んで数えた。

| 相手の数 | 件数 |
|---|---|
| 0 個（**衝突では落ちない**） | 55 |
| 1 個 | 417 |
| 2 個 | 125 |
| 3 個 | 893 |
| 4 個 | 104 |
| 5 個 | 624 |

平均 3.10・中央 3・最多 5 である（外部の集計は「平均約 2.7・最大 5」で、
最大は一致した。平均の差は数え方の違いと見る）。

**衝突率をそのまま方策の比較に使えない。** 0 個のタスクを分母に入れると
薄まるので、[summarize_gpu_eval.py](../../solution/tools/summarize_gpu_eval.py) は
**相手がいる回だけの衝突率**も出すようにした。数は
[make_validation_suite.py](../../solution/tools/make_validation_suite.py) が
CSV の `n_nontarget_objects` 列に書く。

```bash
# 84 件のスイートについて数える（CSV を作り直すだけで出る）
python3 solution/tools/make_validation_suite.py \
    --libero-plus work/LIBERO-plus \
    --bddl-root work/LIBERO-plus/libero/libero/bddl_files \
    --init-states-root work/LIBERO-plus/libero/libero/init_files \
    --output /tmp/VAL_TASKS.csv --per-cell 3 | tail -20
```

手元の 84 件では平均 3.44・最多 7・0 個は 1 件だった（`_add_` の摂動が
物体を増やすため、候補 2,218 件の最多 5 個より多くなる）。

> **採点が見ているものと厳密には一致しない。** ハーネスは観測の `*_pos` という
> 鍵を数えるので、bddl の物体名とはずれる（什器も含まれる）。**目安である。**

## 画像拡張は、評価側の明るさの幅を覆えていない

外部の集計は、初期フレームの平均輝度を 15 件測って次を報告している。

| 接尾辞 | 平均輝度 |
|---|---|
| `light` | 0.287 |
| `table` | 0.405 |
| `tb` | 0.405（0.212〜0.689） |

いま学習で有効にしている拡張（`--dataset.image_transforms.enable=true`）の
中身を、`work/lerobot-src` のソースで確かめた。

| 変換 | 範囲 |
|---|---|
| brightness | **×0.8〜1.2** |
| contrast | ×0.8〜1.2 |
| saturation | ×0.5〜1.5 |
| hue | ±0.05 |
| sharpness | ×0.5〜1.5 |
| affine | ±5°・平行移動 5% |

6 種類から**最大 3 種類**が各フレームに掛かる（`max_num_transforms=3`）。
**gamma は無い。**

```bash
sed -n '/^class ImageTransformsConfig/,/^def make_transform/p' \
    work/lerobot-src/src/lerobot/transforms/transforms.py
```

`table`/`tb` の平均 0.405 を基準にすると、`light` の 0.287 は **×0.71**、
`tb` の上端 0.689 は **×1.70** に当たる。**どちらも ×0.8〜1.2 の外側である。**

> **ただし「だから拡張を広げるべき」とは即断できない。**
> 学習データ（`lerobot/libero_plus`、14,347 エピソード）の平均輝度は
> **0.086〜0.723** で、選んだ 200 エピソードも同じ幅を張っている
> （[01_retrain.md](../howto/01_retrain.md) の実行例）。
> **評価側の幅はデータ側の幅に収まっている。**
>
> つまり足りないのは**届く範囲ではなく、タスクごとの密度**かもしれない
> （1 タスク 5 本しか使っていない）。平均輝度は 1 次元の代理値でしかなく、
> **同じ明るさでまったく違う模様**がありうる。
> **どちらなのかを測ってから決める**（[plan/02_appearance_and_data.md](../plan/02_appearance_and_data.md)）。

## 出典

| 事実 | 出典 |
|---|---|
| 摂動カテゴリと件数、難易度 | `work/LIBERO-plus/libero/libero/benchmark/task_classification.json`（revision `4976dc3`） |
| 変種の物体と操作対象 | 同 `bddl_files/<suite>/<変種名>.bddl` を 2,218 件読んだ |
| 出題は 7 カテゴリ全部ではない | Q&A CSV No.21（西浦） |
| 成功判定に衝突が入ること | [pipeline/rollout.py](../../pipeline/rollout.py) |
| 画像拡張の既定値 | `work/lerobot-src/src/lerobot/transforms/transforms.py` の `ImageTransformsConfig` |
| 学習データの明るさの幅 | [01_retrain.md](../howto/01_retrain.md)（`train.py` が起動時に出す 2 行） |
| 外部の集計（比較対象） | 2026-08-11 に共有された LIBERO-plus の EDA 報告 |
