# 5. 評価するタスクを決める（検証用スイートと小分け）

> **いまは [`ce.build_suite()`](../../../solution/tools/colab_eval.py) が
> 5-1 と 5-2 を通してやる。** 塊の数は
> [colab_eval_config.py](../../../solution/tools/colab_eval_config.py) の `CHUNKS`。
> このページは**なぜ example 4 件では駄目なのか**と、
> **変種名の扱いを間違えると黙って壊れる**ことを残す。

**未知のタスクで評価し、Colab が切れても失わない形にする。**
**`ce.setup_kit()` のあと、`ce.run()` の前に 1 度だけ実行する。**

## 5-1. 検証用スイートへ差し替える

**example 4 件での成功率は本番を予測できない。** 0.875 を出しても本番は 0.096
だった。原因は `compe/t1/README.md` にそのまま書いてある。

> 本番評価は非公開タスクを含む別のタスクセットで実施される。

そこで **LIBERO-plus の 10,030 変種から example 以外を選び、`libero_t1` の
中身を差し替える。** `register.py` は CSV を読んでスイートを作るだけなので、
CSV を替えれば評価パイプラインは何も変えずに動く。

**摂動カテゴリは 7 つとも入れる。** example 4 件は Background Textures と
Light Conditions しか含まないが、それは 4 サンプルにすぎない。運営 Q&A No.21 の
「7 カテゴリすべてではない」も、どれが出るかは言っていない。

絞ると、当たっていれば無駄がなく、**外れていれば何も見えない**。全部入れて
カテゴリ別に集計すれば、当たっていてもその行を読めばよく、外れていれば気づける。
**どこが弱いかが分かれば学習の的も決まる。** スイートも 4 つとも混ぜる。

**B のあと、手順 6 の前に 1 度だけ実行する。** 元へ戻すには B から流し直す。

まず [make_validation_suite.py](../../../solution/tools/make_validation_suite.py) を
`/content/` へアップロードしておく。**手順 1-1 で上げていれば、もう置かれている。**

#### 変種の大半は自分の bddl を持たない（8/10 に判明）

**LIBERO-plus の 10,030 変種のうち 6,287 件は `<変種名>.bddl` が存在しない。**
手元に clone して数えた実測である。カメラ視点・ロボット初期姿勢・観測ノイズの摂動は
**変種名そのものに数値として埋め込まれている。**

```
..._place_it_on_the_plate_view_0_0_100_2_352_initstate_0_noise_3
                          └ 水平・垂直・倍率%・端点回転・端点垂直
                                              └ ロボット初期姿勢   └ ノイズ
```

`libero/libero/envs/env_wrapper.py` が環境を作るときに `_view_` で切って解釈するので、
**bddl は元タスクの分だけあればよい。**

ここを踏み外すと、どちらも**黙って**間違う。

| やってしまうこと | どうなるか |
|---|---|
| 「bddl が無い」として飛ばす | **7 カテゴリのうち 4 つが丸ごと消える。** 84 件のはずが 36 件になった |
| `task_id` を実体のある名前へ書き換える | ファイルは開けるが**摂動が全部無効**になり、素のタスクを評価してしまう |

`make_validation_suite.py` は `task_id` を**一切書き換えず**、解決したパスは
存在確認と指示文の読み取りにだけ使う。

##### ただし配布キット側も `_view_` を知らない

**`env_wrapper.py` が切ってくれるのは、環境を作るときだけである。**
配布キットの [pipeline/environment.py](../../../pipeline/environment.py) は
衝突判定のために bddl を**自分で開く**。

```python
bddl_path = suite.get_task_bddl_file_path(task_id)   # 変種名そのまま
...
def parse_obj_of_interest(bddl_file):
    txt = Path(bddl_file).read_text()                # ← ここで FileNotFoundError
```

`get_task_bddl_file_path` は `_view_` を切らないため、**変種名の bddl が
実体でないと、タスクを読み込む時点で落ちる。** 例外はトラック全体を止めるので、
`_view_` を含む行に当たった瞬間に評価が終わる。

**2026-08-10 に実際にこうなった。** 84 タスクのうち 1〜3 行目
（`_table_` / `_tb_`）だけが走り、4 行目（`_view_`）で落ちて 173 秒で終了した。
終了コードは 0 で、**画面上は正常に見えた。**

下のセルが `parse_obj_of_interest` にも同じ規則を入れる。

#### 初期状態ファイルの決め方（8/10 に作り直した）

**正規表現で接尾辞を剥がす方式は捨てた。** 摂動の付け方はカテゴリごとに違う。

| カテゴリ | 変種名の末尾 |
|---|---|
| Background Textures | `_table_7` / **`_tb_3`** |
| Light Conditions | `_light_15` |
| Sensor Noise | `_noise_2` |
| Language Instructions | `_language_25` |
| Objects Layout | `_add_13` |
| Camera Viewpoints | `_view_0_0_100_0_0` |
| Robot Initial States | `view_0_0_100_0_0_initstate_1` |

正規表現に分岐を足していく形では**必ず取りこぼす。**

**一方、初期状態ファイルは元タスク分しか存在しない**（`init_files/libero_spatial/` は
10 件のみ）。**実在するファイル名との最長前方一致で決めれば取りこぼしようがない。**

`make_validation_suite.py` がこの方式で決めた結果を CSV の `init_states_file` 列へ
書き出し、下のセルが `register.py` にその列を読ませる。
**example の 4 件（列が無い CSV）でも従来どおり動く。**

```python
# === 5-1. 検証用スイートへ差し替える ===
import shutil
from pathlib import Path

# --- 1. キットにパッチを当てる（手順 2 で当てていれば「適用済み」と出る） ---
# **中身は patch_kit_for_variants.py にある。** Docker でもこれを使うので、
# 手元で確かめたものとまったく同じコードが当たる。
run(f"{KIT_PYTHON} /content/patch_kit_for_variants.py --kit-dir {KIT_DIR}", tail=10)

# --- 2. CSV を用意する ---
# **前の回と比べるなら、作り直さずに前の回の CSV を使う。**
# 1 件でも違うタスクが混ざると、1 件ずつ対にした比較（符号検定）が成り立たない。
# 8/11 の 84 件はこれである:
#   outputs/0811_101824_base_n5_ep1_max300/VAL_TASKS.csv
# これを /content/VAL_TASKS.csv へ上げておけば、下の生成は飛ばせる。
if Path("/content/VAL_TASKS.csv").is_file():
    print("既にある /content/VAL_TASKS.csv を使う（前の回と同じ 84 件で比べられる）")
else:
    run(f'PYTHONPATH="{KIT_DIR}/LIBERO-plus:{KIT_DIR}:{KIT_DIR}/compe" '
        f'{KIT_DIR}/venv/bin/python /content/make_validation_suite.py '
        f'--libero-plus {KIT_DIR}/LIBERO-plus '
        f'--exclude {KIT_DIR}/compe/t1/T1_TASKS.csv '
        f'--output /content/VAL_TASKS.csv --per-cell 3',
        tail=60, log="val_suite.log")

# --- 3. example と差し替える ---
# **元の CSV は退避する。** 比較のために example へ戻すことがある。
original = KIT_DIR / "compe/t1/T1_TASKS.csv"
backup = KIT_DIR / "compe/t1/T1_TASKS.csv.example"
if not backup.exists():
    shutil.copy(original, backup)
shutil.copy("/content/VAL_TASKS.csv", original)

import csv as _csv
rows = list(_csv.DictReader(original.open()))
print(f"\n差し替えた: {len(rows)} タスク（example は {backup.name} に退避）")
for row in rows[:7]:
    print(f"  {row['suite']:16s} {row['category']:22s} {row['task_id'][:44]}")
print(f"  ... 計 {len(rows)} 件")

# --- 4. 評価を回す前に、全部開けるか確かめる ---
# **82 分回してから「4 行目で落ちていた」と分かるのは高すぎる。**
# 配布キットと LIBERO-plus の実際のコードと同じ手順でパスを組み立てて、
# 開けない行をここで名指しする。落ちたらこのセルで止まるので、
# 手順 8 へ進む前に気づける。
#
# `--kit-dir` を渡すのは、**パッチが当たっているかを実物で見る**ため。
# 「当てたつもり」で当たっていない状態を見逃さない。
run(f'{KIT_DIR}/venv/bin/python /content/check_validation_suite.py '
    f'--csv /content/VAL_TASKS.csv --libero-plus {KIT_DIR}/LIBERO-plus '
    f'--kit-dir {KIT_DIR}',
    tail=30, log="val_check.log")
```

**先頭 7 件が 7 カテゴリになるように並べてある。** `--max-tasks 5` で
5 件だけ回すと 5 カテゴリを踏むので、**疎通確認が 1 カテゴリに偏らない。**

#### まず 5 件だけ回して、最後まで行くか見る（5 分）

**84 件を 82 分かけて回す前に、これをやる。** 8/10 は 3 タスクで止まったのに
終了コード 0 で終わり、**画面上は正常に見えた。**

**手順 6 の `MAX_TASKS` を `5` にして、手順 6 → 8 を流す。**
手順 8 のセルは触らない（手で書き換えると消し忘れる）。

```python
MAX_TASKS = 5               # 疎通確認。**終わったら None に戻す**
```

**見るのはスコアではなく次の 3 点である。**

| 見るところ | 通っている | 失敗している |
|---|---|---|
| `[結果]` 行の数 | **5 本**そろう | 途中で止まる |
| 所要時間 | 4〜6 分 | **極端に短い**（3 分未満）|
| 手順 9 の出力 | 成功率と 95%CI が出る | 「評価が例外で落ちている」 |

**5 本そろったら `MAX_TASKS = None` に戻して、手順 6 → 8 を流し直す。**

#### エピソード数の決め方

`N_EPISODES` は**タスクあたり**の本数である。タスク数が増えるので、
example のときと同じ感覚で設定すると時間が跳ねる。

タスク数は `4 スイート × 7 カテゴリ × --per-cell` になる。

**`N_EPISODES` は 1 にする。** タスクが 4 件から 84 件へ増えるので、
example のときの 20 をそのまま使うと 1,680 エピソードになって終わらない。

所要は 1 ステップあたり `38.6 ms + 850 ms / n` で見積もれる。
`MAX_STEPS=300` かつ多くが打ち切りまで走る前提（1 本あたり約 280 ステップ）。

| `--per-cell` | タスク数 | `N_EPISODES` | n=10 | n=5 |
|---|---|---|---|---|
| **2** | **56** | **1** | 32 分 | **55 分** |
| 3 | 84 | 1 | 48 分 | 82 分 |
| 3 | 84 | **20** | — | **27 時間（事故）** |

**タスク数を増やすほうがエピソード数を増やすより価値が高い。**
測りたいのは「見たことのない摂動に耐えるか」であって、同じタスクを
何度も回しても分からない。カテゴリごとに 12 本（4 スイート × 3）あれば、
「このカテゴリが弱い」は読める。

#### 例に戻すとき

```python
shutil.copy(KIT_DIR / "compe/t1/T1_TASKS.csv.example",
            KIT_DIR / "compe/t1/T1_TASKS.csv")
```

---

## 5-2. 84 件を 3 つに割る（1 秒・初回のみ）

**1 本で回すと 2.6 時間かかり、Colab のセッション切断に耐えない。**
8/11 に **56 件目（1 時間 43 分）で落ちて、そのぶんを失った。** 画面を触っていても切れる。

**28 件・51 分ずつに割る。** 落ちてもその 51 分だけで済む。
**それでも落ちるなら `CHUNKS` を 6 にする**（1 塊 14 件・26 分）。

```python
# === 84 件を 3 つに割る ===
# **カテゴリは巡回順に並んでいる**（先頭 7 件が 7 カテゴリ）。だから
# 続きの 28 行を取れば、どの塊にも 7 カテゴリが 4 件ずつ入る。
import csv
from pathlib import Path

CHUNKS = 3
source = Path("/content/VAL_TASKS.csv")
rows = list(csv.reader(source.open(newline="")))
header, body = rows[0], rows[1:]

Path("/content/chunks").mkdir(exist_ok=True)
size = -(-len(body) // CHUNKS)              # 切り上げ
category = header.index("category")
for index in range(CHUNKS):
    part = body[index * size:(index + 1) * size]
    path = Path(f"/content/chunks/VAL_{index + 1}.csv")
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(header)
        writer.writerows(part)
    kinds = len({row[category] for row in part})
    print(f"{path.name}: {len(part)} 件 / {kinds} カテゴリ")
```

**どの塊も 7 カテゴリを含んでいること**を確かめる。1 つでも欠けていたら
並び順が巡回でなくなっているので、`--per-cell` を変えた CSV を使っていないか見る。

**回し方。** `手順 6` の `CHUNK` を **1 → 2 → 3** と変えて、
毎回 `手順 6` → `手順 8` → `F` を通す。**`F` は塊ごとに Drive へ退避する。**

| | |
|---|---|
| 1 つあたり | **28 件・約 51 分**（実測 110 秒／本） |
| 落ちたとき | **その塊だけやり直す** |
| ジャークと SPARC | **塊ごとに出る。** 3 つの平均を件数で重み付けして使う |

> **`CHUNK = None` なら 84 件を一度に回す。** GPU マシンのように
> 切断が無い環境ではそちらでよい（ジャークも 1 つの値で出る）。

---

**次は [6. 評価の設定](06_settings.md) へ。**
