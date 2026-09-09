# データセット構成の調査

**何のための文書か:** 本選で配布された6つのデータセットのタスク構成を調べ、Track1・Track2・Track3 それぞれにどのデータセットを使うべきかを決める根拠を示す。

## 項目一覧

| 節 | 中身 |
|---|---|
| [調査結果の要約](#調査結果の要約) | 8/25・8/31・**9/2** の 3 回ぶんの結論 |
| 1 | `Sylvest/libero_plus` の 40 タスクの内訳 |
| 2 | **LIBERO の SCENE 名タスクは 100 個**（Track3 の母集団） |
| 3 | 配布された 6 つのデータセットの構成 |
| 4 | **データセット間の重複**（`libero_plus` は標準スイート 4 つを完全に含む） |
| 5 | **トラックごとに使うべきデータセット**（**9/2 に書き換えた。配布 tar 1 つで足りる**） |
| 6 | 追加の観点（`no_noops`・エピソード長・`libero_90` の 73・Track2 の非公開セット） |
| 7 | **公開版の `meta/` を落として数え直した**（8/31。形式・大きさ・指示文と場面の重なり） |
| **8** | **配布 tar `libero_combined_20hz` の中身**（**9/2。`total_tasks` は 111・乗り換えで踏む穴・`splits` は無視してよい**）＋ **9/5: 逆向きタスク 3/4 の順方向デモが tar にある・`action` の規約と gripper の符号化混在（8.7）・`state` の読み方と相対回転の組み方（8.8）** |
| 数え方 | 打ったコマンド |

## 調査結果の要約

- **`Sylvest/libero_plus`（現在の学習データ）の 40 タスク中、10 個が SCENE 名タスク（libero_10 由来）**
- **LIBERO の SCENE 名タスクは libero_10 に 10 個、libero_90 に 90 個、合計 100 個**
- **Track3（逆向きタスク）の母集団は SCENE 名タスクに限られる（運営の check_no_leaks.py より）**
- **spatial・object・goal の 30 タスクは SCENE 名ではない（Track3 に使えない）**

**2026-08-31 に公開版の `meta/` を落として数え直して分かったこと**（7 節）:

- **`libero_90` の「73 タスク」は指示文を重複を除いて数えた数で、90 個の bddl のうち 89 個が入っている**（欠けは 1 個だけ）
- **IPEC-COMMUNITY の 5 スイートは LeRobot v2.1、`lerobot/libero_plus` だけ v3.0**
- **`lerobot/libero_combined_20hz` は Hugging Face に無い。**配布された tar からしか中身を読めない（**9/2 に読んだ。8 節**）
- **配布の `dataset_IPEC-COMMUNITY.zip`（5.11 GB）は、公開版 5 スイートの合計 5.32 GB とほぼ同じ大きさである**

**2026-09-02 に配布 tar の `meta/info.json` を読んで分かったこと**（8 節）:

- **`libero_combined_20hz` の `total_tasks` は 111 である。****中身は `libero_plus` ＋ `libero_90` ＋ `libero_10` ＋ `libero_spatial` ＋ `libero_object`** で、エピソード 19,533・コマ 3,028,708 が 5 つの足し算と 1 件も違わない（**`libero_goal` だけ入っていない**）
- **これに乗り換えると、Track3 の母集団 100 場面のうち 99 場面が順方向で学習に入る**（いまは 10 場面だけ）。**課題名は 40 → 111、エピソードは 1.36 倍になる**
- **`v3.0` である。**いま学習に使っている `lerobot/libero_plus` と同じ形式なので、**変換が要らない**（標準スイートを単体で足すなら `v2.1` の変換が要る）
- **配布 GPU 環境の `~/dataset/lerobot/` に読み出し専用で置かれている。**`~/data` へ展開して使う——**21 GB をどこかへ転送する必要が無い**
- **ただし「これを足せば Track3 が動く」の根拠は弱い。****配布ベースラインはこのデータで学習されていて、それでも Track3 は 0.000 である**（8.3 節）
- **`info.json` の `"splits": {"train": "0:432"}` は無視してよい。**同梱の LeRobot でも `solution/train/` でも、`splits` は書かれるだけでエピソードの選択には 1 度も読まれない（8.4 節）
- **乗り換えには 3 か所の書き換えが要る。**`_DATASET_REPO`・`_DATASET_REVISION`・`--dataset-root` で、**`--dataset-root` を忘れると足した 5,186 本が 1 本も選ばれないまま、エラーも出ずに学習が走る**（8.2 節）

**2026-09-05 に会社の RTX6000 で tar の `tasks.parquet` と bddl を突き合わせて分かったこと**（8.5 節）:

- **Track3 の逆向きタスク 3/4（黄色い本・赤いマグ・2 つのマグ）の順方向デモは配布 tar に実在する**。**「黄色い本の順方向は無い」という初判断は誤りで、修正済み**——タスクの指示文は「the book」（「yellow book」ではない）で、一致させなきゃいけないのは指示文の文字列ではなくシーン（fixtures・objects）の構成である
- **デモの使い道は会社マシンのローカルだけで足りる。Blackwell からの転送は不要**（tar は 20 GB でこのマシンに実在する）
- **例外はスープ 1 タスクだけ**（スープの順タスクは全て `basket` 向けで、逆向きタスクが `tray` を使うため別場面）

---

## 1. `Sylvest/libero_plus` の 40 タスクの内訳

現在の学習に使っている `Sylvest/libero_plus` の 40 タスクは、以下の内訳である。

| 由来スイート | タスク数 | SCENE 名タスクか |
|---|---|---|
| `libero_10` | 10 | ✓（全て SCENE 名） |
| `libero_spatial` | 10 | ✗ |
| `libero_object` | 10 | ✗ |
| `libero_goal` | 10 | ✗ |
| 合計 | 40 | 10 個が SCENE 名 |

**この表から何が決まるか:** 現在の学習データは Track3 の母集団（SCENE 名タスク）を 10 種類しかカバーしていない。

### `libero_plus` に含まれる SCENE 名タスク（10 個）

```bash
# 抽出コマンド
python3 << 'EOF'
import json
from pathlib import Path

episodes_path = Path('/home/ubuntu/.cache/huggingface/hub/datasets--Sylvest--libero_plus_lerobot/snapshots/22c57433fef692b5b9ecc0795344daac7fa867a5/meta/episodes.jsonl')
tasks = set()
with open(episodes_path, 'r') as f:
    for line in f:
        ep = json.loads(line)
        if 'tasks' in ep and len(ep['tasks']) > 0:
            tasks.add(ep['tasks'][0])
print(f'{len(tasks)} タスク')
EOF
```

実際のタスク（10 個、全て libero_10 由来）:

1. turn on the stove and put the moka pot on it
2. put the black bowl in the bottom drawer of the cabinet and close it
3. put the yellow and white mug in the microwave and close it
4. put both moka pots on the stove
5. put both the alphabet soup and the cream cheese box in the basket
6. put both the alphabet soup and the tomato sauce in the basket
7. put both the cream cheese box and the butter in the basket
8. put the white mug on the left plate and put the yellow and white mug on the right plate
9. put the white mug on the plate and put the chocolate pudding to the right of the plate
10. pick up the book and place it in the back compartment of the caddy

---

## 2. LIBERO の SCENE 名タスクは 100 個（libero_10 + libero_90）

手元の LIBERO-plus ソースから bddl ファイルを数えた結果。

```bash
# 抽出コマンド
ls -1 /home/ubuntu/src/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_10/*.bddl \
  | sed 's|.*/||' | sed 's|\.bddl$||' \
  | grep -E "^(KITCHEN|LIVING_ROOM|STUDY)_SCENE[0-9]+" \
  | grep -v "_add_" | grep -v "_view_" | grep -v "_language_" | grep -v "_level" | grep -v "copy" | grep -v "_sample" \
  | grep -v "_[0-9]\+$" \
  | wc -l
# 結果: 10

ls -1 /home/ubuntu/src/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_90/*.bddl \
  | sed 's|.*/||' | sed 's|\.bddl$||' \
  | wc -l
# 結果: 90
```

| スイート | SCENE 名タスク数 | 備考 |
|---|---|---|
| `libero_10` | 10 | 全て SCENE 名（variant を除く元タスク） |
| `libero_90` | 90 | 全て SCENE 名（variant を除く元タスク） |
| 合計 | **100** | Track3 の母集団 |

**この表から何が決まるか:** Track3 の元タスクは最大 100 種類ある。libero_90 を学習に加えない限り、Track3 の 90 種類には対応できない。

---

## 3. 本選で配布された 6 つのデータセットの構成

公開版の `meta/info.json` から数えた（2026-08-25・[06_training_data_composition.md](06_training_data_composition.md) 135 行目以降）。

| データセット | fps | エピソード | コマ | タスク | 1 本の平均コマ | SCENE 名タスク |
|---|---|---|---|---|---|---|
| `libero_spatial_no_noops` | 20 | 432 | 52,970 | 10 | 122.6 | 0 |
| `libero_object_no_noops` | 20 | 454 | 66,984 | 10 | 147.5 | 0 |
| `libero_goal_no_noops` | 20 | 428 | 52,042 | 10 | 121.6 | 0 |
| `libero_10_no_noops` | 20 | 379 | 101,469 | 10 | 267.7 | **10** |
| `libero_90_no_noops` | 20 | 3,921 | 569,249 | 73 | 145.2 | **73** |
| **`Sylvest/libero_plus`** | 20 | **14,347** | **2,238,036** | **40** | 156.0 | **10** |

**この表から何が決まるか:** 

- **`libero_90_no_noops` だけが新しい SCENE 名タスクを追加する**（他のデータセットは libero_plus に既に含まれている）
- **`libero_10_no_noops` は 1 本が飛び抜けて長い（267.7 コマ）**。300 歩の打ち切りに対して余裕が無い。Track3 の課題（2 段構えのタスク）も同様に長い可能性がある。
- **`no_noops` は動いていないコマを抜いた版**。方策が「止まっていること」まで覚えないようにするため。

---

## 4. データセット間の重複（タスク名の集合として）

| データセット A | データセット B | 重複タスク数 | 備考 |
|---|---|---|---|
| `libero_plus` (40) | `libero_10` (10) | 10 | libero_plus は libero_10 を完全に含む |
| `libero_plus` (40) | `libero_spatial` (10) | 10 | libero_plus は libero_spatial を完全に含む |
| `libero_plus` (40) | `libero_object` (10) | 10 | libero_plus は libero_object を完全に含む |
| `libero_plus` (40) | `libero_goal` (10) | 10 | libero_plus は libero_goal を完全に含む |
| `libero_plus` (40) | `libero_90` (73) | 0 | **場面としては重複なし。ただし指示文は 2 件重なる**（7.3） |
| `libero_10` (10) | `libero_90` (73) | 0 | SCENE 番号が異なる（10 vs 90）。**指示文は 1 件重なる**（7.3） |

**この表から何が決まるか:**

- **標準スイート 4 つ（spatial・object・goal・10）を統合してもタスクの種類は増えない**（全て libero_plus に含まれる）
- **タスクを増やせるのは `libero_90` を入れたときだけ**

---

## 5. トラックごとに使うべきデータセット

> **【9/2 に書き換えた】3 行とも「配布の `libero_combined_20hz` 1 つ」で足りる。**
> **あれは `libero_plus` ＋ `libero_90` ＋ `libero_10` ＋ `libero_spatial` ＋ `libero_object` である**（8 節）。
> **下の表の「使うべきデータセット」は、そのどれもがあの 1 つに収まっている**
> ——**唯一入っていないのは `libero_goal`（428 本・10 課題）で、
> あれは SCENE 名タスクではないので Track3 には使えず、Track1・Track2 では
> 「エピソードを増やす」ぶんにしか効かない。**
>
> **だから「自分で混ぜる」という選択肢は、9/2 以降は考えなくてよい。**

| トラック | 中身 | 使うべきデータセット | **配布 tar 1 つで足りるか** | 根拠 |
|---|---|---|---|---|
| **Track1** | 学習済みタスク・背景と照明が変わる | `libero_plus` + 標準スイート 4 つ | **足りる**（`libero_goal` の 428 本だけ欠ける） | エピソード数を増やす（摂動への頑健性）。タスクの種類は増えない |
| **Track2** | 学習済みタスク・視点・姿勢・配置・雑音が変わる | `libero_plus` + 標準スイート 4 つ + **`libero_90`** | **足りる**（同上） | エピソード数 + タスクの種類を両方増やす。difficulty_level（task_classification.json）から非公開セットが引かれる |
| **Track3** | 既存タスクの逆向き（SCENE 名に限られる） | **`libero_10` + `libero_90`** | **足りる。**母集団 100 場面のうち **99 場面**が入る（7.5 節。欠けは `LIVING_ROOM_SCENE2` の 1 個） | Track3 の母集団は SCENE 名タスク（100 個）。spatial・object・goal は使えない |

**ただし Track3 については、これで届くという意味ではない。**
**配布ベースラインはこの 99 場面を見て学習していて、それでも Track3 は 0.000 である**（8.3 節）。
**「順方向の場面を見る」ことと「逆向きに物を選べる」ことは別である。**

**Track3 の根拠（詳細）:**

運営の検査スクリプト `/home/ubuntu/src/b_PARC2026_final/scripts/check_no_leaks.py` にある禁止語:

```python
r"(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+"
```

この正規表現に「元タスク（順方向 BDDL）の識別子」という説明が付いている。つまり、**Track3 の元タスクは SCENE 名のものに限られる**。

---

## 6. 追加の観点

### (1) `no_noops` の有無が学習に効く理由

`no_noops` は動いていないコマ（行動がほぼゼロのステップ）を落としてある。落とさないと、方策が「止まっていること」まで覚えてしまい、実行時に不要な停止が発生する。

### (2) 1 エピソードの長さと 300 歩の打ち切り

| データセット | 平均コマ | 300 歩との余裕 |
|---|---|---|
| `libero_10_no_noops` | 267.7 | 32.3 コマ（11%）|
| その他 | 120〜160 | 140〜180 コマ（47〜60%）|

**`libero_10` と Track3 の長いタスクは、300 歩で完遂できるギリギリの長さである。** 学習時のデータ長と評価時の制限時間がどちらも 300 歩なので、方策が最適な経路を学べないと失敗する。

### (3) `libero_90` のタスク数が 73 である理由 —— **2026-08-31 に答えが出た**

公開版の `meta/info.json` には 73 タスクと記録されているが、手元の bddl には 90 個の基本タスクがある。

**8/31 に数え直した結果、73 は指示文を重複を除いて数えた数であった**（7.5）。
**90 個の bddl のうち 89 個が入っており、落ちているのは
`LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket` の 1 個だけである。**

**下の「残り 17 種類」という読みは外れである。**同じ指示文が 2〜3 の場面で使い回されており、
その分だけタスク数が縮んで見えていた。~~本選の非公開セットでは残り 17 種類も採点に出る可能性がある。~~

### (4) Track2 の非公開セットは difficulty_level から引かれる

Track2 の配布物（`task_classification.json`）には difficulty_level が 1〜5 まであり、公開された 8 行は見本である。**母集団は task_classification.json に列挙された variant の全部であり、倍率ではなく difficulty_level が運営の評価軸である。** 実測すると、手先の精度が 2 cm を超えると成功率が崩れ始める（向きの誤差が効き始める）。

---

## 7. 公開版の `meta/` を落として数え直した（2026-08-31）

**動画は 1 本も落としていない。**Hugging Face から `meta/` だけを取り出して数えた
（`allow_patterns=['meta/*']`。合計で数十 MB）。数えたのは
[../../solution/tools/analysis/count_dataset_overlap.py](../../solution/tools/analysis/count_dataset_overlap.py)
である。**配布物（`~/dataset` の実物）とはまだ突き合わせていない。**

**`lerobot/libero_combined_20hz` は Hugging Face に無い**（`RepositoryNotFoundError`）。
**あれの中身は、配布された tar からしか読めない。**
**→ 9/2 に配布物から読んだ。8 節にある。**

**ここから 1 つ決まる。****配布環境（消える機械）から丸ごと運び出す価値があるのは
`libero_combined_20hz.tar` だけである。**標準スイート 5 つと LIBERO-plus は
Hugging Face から引き直せるので、**運ぶのは `meta/` だけでよい。**

### 7.1 `info.json` の 5 項目 —— **5 スイートは LeRobot v2.1、`libero_plus` だけ v3.0**

| データセット | Hugging Face の名前 | codebase | エピソード | コマ | タスク | fps |
|---|---|---|---|---|---|---|
| `libero_spatial` | `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` | **v2.1** | 432 | 52,970 | 10 | 20 |
| `libero_object` | `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot` | **v2.1** | 454 | 66,984 | 10 | 20 |
| `libero_goal` | `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` | **v2.1** | 428 | 52,042 | 10 | 20 |
| `libero_10` | `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot` | **v2.1** | 379 | 101,469 | 10 | 20 |
| `libero_90` | `IPEC-COMMUNITY/libero_90_no_noops_lerobot` | **v2.1** | 3,921 | 569,249 | **73** | 20 |
| `libero_plus` | `lerobot/libero_plus` | **v3.0** | 14,347 | 2,238,036 | 40 | 20 |

**エピソード数・コマ数・タスク数は 3 節の表（8/25 に数えた値）と 1 件も食い違わない。**

**この表から何が決まるか。****libero_90 を足す日にやることが 2 つある。**

1. **v2.1 が読めるかを確かめる。**`train_pi05.py` は `--dataset-root` を
   `--dataset.root=` として `lerobot-train` に渡すだけである（351〜352 行目）。
   **だから読めるかどうかは、入っている `lerobot` の版で決まる。まだ確かめていない。**
   確かめ方は、学習を回す機械で `python3 -c "import lerobot; print(lerobot.__version__)"` を打ち、
   `libero_90` を渡して 1 歩だけ回すことである。
2. **`train_pi05.py` の 2 行を書き換える。**`_DATASET_REPO = "lerobot/libero_plus"`（153 行目）と
   `_DATASET_REVISION`（154 行目）は定数で、`--dataset.repo_id` と `--dataset.revision` として
   毎回渡される（349〜350 行目）。**`--dataset-root` を別のデータセットへ向けても、この 2 つは
   `libero_plus` を指したままである。**

### 7.2 Hugging Face から落とすときの大きさ —— **5 スイートで 5.32 GB**

| データセット | 大きさ | ファイル数 |
|---|---|---|
| `libero_spatial_no_noops_1.0.0_lerobot` | 0.37 GB | 1,301 |
| `libero_object_no_noops_1.0.0_lerobot` | 0.54 GB | 1,367 |
| `libero_goal_no_noops_1.0.0_lerobot` | 0.34 GB | 1,289 |
| `libero_10_no_noops_1.0.0_lerobot` | 0.64 GB | 1,142 |
| **`libero_90_no_noops_lerobot`** | **3.43 GB** | 11,769 |
| 合計 | **5.32 GB** | 16,868 |

**この表から何が決まるか。**配布物の `dataset_IPEC-COMMUNITY.zip` は **5,113,568,944 バイト
（5.11 GB）**で、この 5.32 GB とほぼ同じである。**あの zip には libero_90 が入っているとみて良い**
（確認は `unzip -l dataset_IPEC-COMMUNITY.zip | grep -c libero_90`）。
**入っていなければ Hugging Face から 3.43 GB を落とせば足りる。**

### 7.3 指示文の集合の重なり（6 つの総当たり）

**数えたのは指示文（`meta/tasks.jsonl` の `task`）であって、タスク名（bddl のファイル名）ではない。**
**この 2 つは別物である**（7.4 で分けて数える）。

| A | B | A のみ | 共通 | B のみ |
|---|---|---|---|---|
| `libero_spatial` (10) | `libero_plus` (40) | 0 | **10** | 30 |
| `libero_object` (10) | `libero_plus` (40) | 0 | **10** | 30 |
| `libero_goal` (10) | `libero_plus` (40) | 0 | **10** | 30 |
| `libero_10` (10) | `libero_plus` (40) | 0 | **10** | 30 |
| `libero_90` (73) | `libero_plus` (40) | 71 | **2** | 38 |
| `libero_10` (10) | `libero_90` (73) | 9 | **1** | 72 |
| `libero_goal` (10) | `libero_90` (73) | 9 | **1** | 72 |
| 標準スイート 4 つの間（6 通り） | | 10 | **0** | 10 |

**この表から何が決まるか。**

- **標準スイート 4 つは `libero_plus` に完全に含まれる。**4 節の表がそのまま裏付けられた。
- **`libero_90` と `libero_plus` は、指示文では 2 件だけ重なる**——`turn on the stove` と
  `pick up the book and place it in the back compartment of the caddy` である。
  **ただし場面（bddl）としては重なっていない。**同じ文が別の場面で使い回されているだけである
  （`turn on the stove` は `libero_goal` の場面と `KITCHEN_SCENE3`・`KITCHEN_SCENE9`）。
- **つまり `libero_90` を足すと、方策は「見たことのある文が、見たことのない場面で出てくる」状況に置かれる。**

### 7.4 SCENE 名タスク（bddl のファイル名）の重なり

**指示文から bddl を引いて数えた。**引き方は 2 つ用意した——bddl の `(:language ...)` 行と、
**bddl のファイル名から場面の名前を外した残り**である。**LIBERO-plus の variant の bddl は言い回しが
書き換わっている**（`pick up the black bowl …` が `pick the akita black bowl …` になっている）ので、
ファイル名の側が要る。

| A | B | A のみ | 共通 | B のみ |
|---|---|---|---|---|
| `libero_10` (11) | `libero_plus` (13) | 0 | **11** | 2 |
| `libero_10` (11) | `libero_90` (90) | 9 | **2** | 88 |
| `libero_90` (90) | `libero_plus` (13) | 86 | **4** | 9 |
| `libero_goal` (2) | `libero_90` (90) | 0 | **2** | 88 |
| `libero_spatial` (0) / `libero_object` (0) | 何と比べても | 0 | **0** | — |

**この表は上限である。**12 の指示文が 2〜3 の場面に当たるので（7.5）、**指示文からは
「どの場面が録られたのか」が決まらない。**`libero_10` が 10 個ではなく 11 個、
`libero_plus` が 10 個ではなく 13 個と出ているのは、この当たり過ぎの分である。

**この表から何が決まるか。****Track3 に効くのは `libero_90` だけである**という 4 節の結論は動かない。
**`libero_spatial` と `libero_object` は SCENE 名の bddl を 1 つも持たない**——
ファイル名が指示文そのものになっており、場面の名前が入っていない。

### 7.5 **`libero_90` の「73 と 90」は、17 タスクの欠けではなかった**

**90 個の bddl のうち 89 個が配布データに入っている。落ちているのは 1 個だけである。**

| 数えたもの | 値 |
|---|---|
| `bddl_files/libero_90` の bddl | **90 個** |
| 73 の指示文から引ける bddl | **89 個** |
| 1 つも入っていない bddl | **1 個**——`LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket` |
| 2 場面以上に当たる指示文 | **12 件**（うち 3 場面が 4 件） |

**根拠は 1 本あたりのエピソード数である。**1 場面だけの指示文 61 件は **34〜50 本（中央値 45）**、
**2 場面に当たる 8 件は 76〜100 本、3 場面に当たる 4 件は 91〜146 本**であった。
**倍になっているので、どちらの場面も録られている。**

**内訳は 61 + 8×2 + 4×3 = 89 で、bddl の 90 個から 1 個だけ足りない。**

**この表から何が決まるか。**6 節 (3) の「残り 17 種類が非公開セットに出るかもしれない」という
読みは外れである。**73 は指示文を重複を除いて数えた数であって、タスクが 17 個欠けているのではない。**
**`libero_90` を足せば、Track3 の母集団 100 個のうち 99 個が順方向で学習に入る。**

---

## 数え方（打ったコマンド）

### 2026-08-31 の 7 節（公開版の `meta/` だけを落として数える）

```bash
# meta/ だけを落とす（動画は落ちない）
python3 - <<'EOF'
from huggingface_hub import snapshot_download
for r in ['IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot',
          'IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot',
          'IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot',
          'IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot',
          'IPEC-COMMUNITY/libero_90_no_noops_lerobot']:
    print(snapshot_download(repo_id=r, repo_type='dataset', allow_patterns=['meta/*']))
EOF

# 重なりを数える（H は ~/.cache/huggingface/hub）
cd ~/src/PARC2026_pre && H=~/.cache/huggingface/hub
python3 solution/tools/analysis/count_dataset_overlap.py \
  --meta libero_spatial=$(ls -d $H/datasets--IPEC-COMMUNITY--libero_spatial_no_noops_1.0.0_lerobot/snapshots/*/)meta \
  --meta libero_object=$(ls -d $H/datasets--IPEC-COMMUNITY--libero_object_no_noops_1.0.0_lerobot/snapshots/*/)meta \
  --meta libero_goal=$(ls -d $H/datasets--IPEC-COMMUNITY--libero_goal_no_noops_1.0.0_lerobot/snapshots/*/)meta \
  --meta libero_10=$(ls -d $H/datasets--IPEC-COMMUNITY--libero_10_no_noops_1.0.0_lerobot/snapshots/*/)meta \
  --meta libero_90=$(ls -d $H/datasets--IPEC-COMMUNITY--libero_90_no_noops_lerobot/snapshots/*/)meta \
  --meta libero_plus=$(ls -d $H/datasets--lerobot--libero_plus/snapshots/*/)meta \
  --bddl-root work/LIBERO-plus/libero/libero/bddl_files
```

**配布物（tar・zip）を同じ道具で数えるには、`meta/` だけを取り出してから同じ `--meta` に渡す。**

```bash
mkdir -p ~/meta_only
# tar から meta/ だけを取り出す（動画は取り出さない）
tar -xf libero_combined_20hz.tar --wildcards '*/meta/*' -C ~/meta_only
# zip も同じ。-j は付けない（付けると 5 スイートの meta が同じ名前でぶつかる）
unzip dataset_IPEC-COMMUNITY.zip '*/meta/*' -d ~/meta_only
```



### `Sylvest/libero_plus` のタスク一覧

```bash
python3 << 'EOF'
import json
from pathlib import Path

episodes_path = Path('/home/ubuntu/.cache/huggingface/hub/datasets--Sylvest--libero_plus_lerobot/snapshots/22c57433fef692b5b9ecc0795344daac7fa867a5/meta/episodes.jsonl')
tasks = set()
with open(episodes_path, 'r') as f:
    for line in f:
        ep = json.loads(line)
        if 'tasks' in ep and len(ep['tasks']) > 0:
            tasks.add(ep['tasks'][0])

print(f'Total unique tasks: {len(tasks)}')
for task in sorted(tasks):
    print(task)
EOF
```

### libero_10 と libero_90 の SCENE 名タスクを数える

```bash
# libero_10 の元タスク（variant を除く）
ls -1 /home/ubuntu/src/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_10/*.bddl \
  | sed 's|.*/||' | sed 's|\.bddl$||' \
  | grep -E "^(KITCHEN|LIVING_ROOM|STUDY)_SCENE[0-9]+" \
  | grep -v "_add_" | grep -v "_view_" | grep -v "_language_" | grep -v "_level" | grep -v "copy" | grep -v "_sample" \
  | grep -Ev "_[0-9]+$" \
  | wc -l

# libero_90 の元タスク（variant なし）
ls -1 /home/ubuntu/src/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_90/*.bddl | wc -l
```

### SCENE 名タスクの一覧を出力

```python
from pathlib import Path
import re

def get_true_base_tasks(path):
    all_files = [f.stem for f in path.glob('*.bddl')]
    base_tasks = []
    for f in all_files:
        if any(pattern in f for pattern in ['_add_', '_level', '_light_', '_table_', '_tb_', 
                                               '_view_', '_language_', 'copy', '_sample']):
            continue
        if re.search(r'_\d+$', f):
            after_scene = re.sub(r'^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_', '', f)
            if re.search(r'_\d+$', after_scene):
                continue
        base_tasks.append(f)
    return sorted(base_tasks)

libero_10_path = Path('/home/ubuntu/src/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_10')
libero_90_path = Path('/home/ubuntu/src/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_90')

libero_10_base = get_true_base_tasks(libero_10_path)
libero_90_base = get_true_base_tasks(libero_90_path)

print(f"libero_10: {len(libero_10_base)} tasks")
print(f"libero_90: {len(libero_90_base)} tasks")
```

---

## 8. `libero_combined_20hz` の中身が分かった（2026-09-02）

**7 節では「Hugging Face に無いので配布された tar からしか読めない」と書いた。
9/2 にユーザが配布物の `meta/info.json` を読んだ。**

| 項目 | 値 |
|---|---|
| `codebase_version` | **`v3.0`**（いま学習に使っている `lerobot/libero_plus` と同じ） |
| `total_episodes` | **19,533** |
| `total_frames` | **3,028,708** |
| **`total_tasks`** | **111** |
| `fps` | 20 |
| `robot_type` | `panda` |
| カメラ | **2 つだけ**——`observation.images.wrist` と `observation.images.front`（どちらも 256×256×3・av1） |
| `observation.state` | float32・8 次元 `[eef_pos(3), 軸角(3), 指 qpos(2)]`・**生スケール** |
| `action` | float32・7 次元 `[並進(3), 回転(3), gripper(1)]`・**規約は 8.7 節** |

### 8.1 5 つのデータセットの足し算とぴったり合う —— `libero_goal` だけが入っていない

**7.1 節で数えた 6 つの値の組み合わせを総当たりしたところ、
エピソード数とコマ数の両方が一致する組み合わせは 1 つだけだった。**

| 中身 | エピソード | コマ | 課題名 |
|---|---|---|---|
| `Sylvest/libero_plus` | 14,347 | 2,238,036 | 40 |
| `libero_90` | 3,921 | 569,249 | 73 |
| `libero_10` | 379 | 101,469 | 10 |
| `libero_spatial` | 432 | 52,970 | 10 |
| `libero_object` | 454 | 66,984 | 10 |
| **合計** | **19,533** | **3,028,708** | 143（単純な足し算） |
| **配布 tar の値** | **19,533** | **3,028,708** | **111** |

**エピソードもコマも 1 件も違わない。****`libero_goal`（428 本・52,042 コマ・10 課題）だけが入っていない。**

**課題名は 143 の単純な足し算に対して 111 なので、32 個が重複している。**
**この 32 は、4 節と 7.3 節で 8/25〜8/31 に数えた重なりとぴったり合う。**

| 重なり | 数 | 出典 |
|---|---|---|
| `libero_10` ⊂ `libero_plus` | 10 | 4 節（`libero_plus` は 4 つの標準スイートを完全に含む） |
| `libero_spatial` ⊂ `libero_plus` | 10 | 同 |
| `libero_object` ⊂ `libero_plus` | 10 | 同 |
| `libero_plus` と `libero_90` の指示文の重なり | **2** | 7.3 節 |
| **合計** | **32** | **143 − 32 = 111。配布 tar の `total_tasks` と一致する** |

**`libero_10` と `libero_90` の重なり 1 件（7.3 節）は、この 32 の中に入っている**
——**`libero_10` の課題はすでに `libero_plus` の重複として数えているからである。**

**つまり課題名の側でも、8.1 節の組み合わせが正しいことが裏づいた。**
**エピソード数・コマ数・課題名の 3 つが独立に一致している。**

**いま学習に使っているのは `libero_plus` の 40 課題だけである。**
**だから配布 tar に乗り換えると、課題名が 40 → 111 に増える**（**新しく増えるのは 71 個**）。
**エピソードは 1.36 倍、コマは 1.35 倍になる。**

### 8.2 これで消えた作業が 2 つある

| 消えた項目 | 何が要らなくなったか |
|---|---|
| **`libero_90` を自分で足す**（`open_items.md` の `F48`・段 2 の前半） | **配布 tar にもう入っている。**LeRobot 0.6.0 は複数のデータセットを受け付けないので結合コードを書く必要があったが、**それが丸ごと要らない** |
| **形式の変換**（`F57`） | **`v3.0` なので、いま学習に使っているものと同じである。**7.1 節では標準スイート 5 つが `v2.1` で `libero_plus` だけ `v3.0` だと分かっていて、**変換に 1 日使う可能性**を見ていた。**配布 tar を使うならその心配が消える** |

**そして運ぶ手間も要らない。**配布 GPU 環境（Blackwell サーバ）の `~/dataset/lerobot/` に
**読み出し専用で置かれている**（`~/data` へ展開してから使う。`resources/infra_and_docker.md`）。
**21 GB をどこかへ転送する必要が無い。**

#### 乗り換えるときに踏む穴 —— 2 か所とも直さないと、黙って古いデータで学習が走る

**`train_pi05.py` の定数 2 行を直すだけでは足りない。`--dataset-root` も要る。**
**逆に `--dataset-root` だけでも足りない。両方である。**

| 直すもの | 場所 | 忘れるとどうなるか |
|---|---|---|
| **`_DATASET_REPO = "lerobot/libero_plus"`** | `train_pi05.py:153`（`--dataset.repo_id` として渡る。349 行目） | **`lerobot-train` が受け取る名前が `libero_plus` のままになる** |
| **`_DATASET_REVISION`** | `train_pi05.py:154`（`--dataset.revision` として渡る。350 行目） | 同上 |
| **`--dataset-root` を展開先へ向ける** | `train_pi05.py:214`（`--dataset.root` として渡る。351〜352 行目） | **エピソード番号の選び方が Hub から取った `libero_plus` の 14,347 本になり、足した 5,186 本が 1 本も選ばれないまま学習が始まる。****エラーにならないので気付けない** |

**3 つ目は `train.py` の `_resolve_dataset_root`（733 行目）の説明にそのまま書かれている**
——**「混ぜたデータセットを渡されたら Hub へ取りに行かない。取りに行くと
`lerobot/libero_plus` の 14,347 本ぶんの番号が返り、足した分が 1 本も選ばれないまま
学習が始まる（エラーにならないので気付けない）」。**

**番号選びが読むのは `meta/episodes/*.parquet` であって、`info.json` の `splits` ではない**
（`_select_without_holdout` → `_load_episode_metadata`。840 行目）。

**確認したい仮定。****`--dataset.root` を配布 tar の展開先へ向けたまま
`--dataset.repo_id` を `libero_plus` のままにしたときに、`lerobot-train` が
どう振る舞うかは確かめていない**（`root` が優先されて動くのか、`revision` の
食い違いで落ちるのか）。**だから乗り換える 1 本目は、必ず数歩だけ回して
`[train]` が出す `--dataset.episodes` の長さを見る**——
**14,347 本ぶんなら失敗、19,533 本ぶんに近ければ成功である。**

### 8.3 ただし「`libero_90` を足せば Track3 が動く」の根拠は、むしろ弱くなった

**配布資料の記述では、配布ベースラインモデルはこの `libero_combined_20hz` で学習されている。**
**そのベースラインの点は Track1 0.286・Track2 0.165・**
**Track3 0.000 である**（`reference/19` の「本選の提出と点」の最終行）。

**つまり `libero_90` の 73 課題を見た重みが、すでに Track3 で 0 点を出している。**

| これが意味すること | 中身 |
|---|---|
| **「場面を見たことがないから動けない」だけでは説明が付かない** | ベースラインは `libero_90` の場面を見ていて、それでも 0 / 8 だった |
| **`knowledge/17` の診断と噛み合う** | 診断は**「似た物が複数ある中から、言葉で 1 つを選べていない」**である。順方向の場面を見ることは、逆向きの物の選び方を教えない |
| **残る反論** | **ベースラインは LoRA 5,000 歩で、我々のフルパラメータ学習より弱い**（Track1 0.286 対 我々の 0.492）。**「弱すぎて 0 だった」のか「データでは届かない」のかは、この 1 点からは分けられない** |

**確認したい仮定。「ベースラインが `libero_combined_20hz` で学習された」は
`resources/infra_and_docker.md` の記述であって、資料のページ番号を突き合わせていない。**
**もしベースラインが `libero_plus` だけで学習されていたなら、上の議論は成り立たず、
`libero_90` を足す価値は元に戻る。**

### 8.4 `splits` の `"train": "0:432"` は無視してよい（読まれていない）

**配布 tar の `info.json` には `"splits": {"train": "0:432"}` と入っている。**
**432 は `libero_spatial` のエピソード数で、全体の 19,533 と合っていない。**
**これを読んで学習が 432 本だけになる心配は無い。**

**同梱している LeRobot（`work/lerobot-src`）と `solution/train/` を全部見たところ、
`splits` は書き込まれるだけで、エピソードの選択には 1 度も読まれていない。**

| 場所 | 何をしているか |
|---|---|
| `datasets/dataset_metadata.py:624` | `{"train": f"0:{total_episodes}"}` を**書く** |
| `datasets/aggregate.py:718` | 束ねた合計で**書く** |
| `datasets/dataset_tools.py:938`・`1889` | **書く** |
| `datasets/dataset_tools.py:1107-1109` | 元の値を**写す** |
| `datasets/dataset_tools.py:1108-1109` | **読む唯一の箇所。**ただし「元の値を写す」ためで（`# Preserve original splits if available`）、**エピソードは選んでいない** |
| `scripts/lerobot_edit_dataset.py` | **データセットを分割する別の道具**の引数。学習では通らない |
| **`solution/train/`** | **1 件も出てこない**（`grep -rn splits solution/train/` が空） |

**エピソードを選ぶために読んでいる箇所は 1 つも無い。**
**選び方は `meta/episodes/*.parquet` を読む**（`train.py:840` の `_select_without_holdout`）。

### 8.5 配布 tar のタスクが Track3（黄色い本タスクの順方向）と対応する（2026-09-05）

**Track3 の逆向きタスク「棚から黄色い本を下ろす」の順方向は、配布 tar の `task 92`
（`STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf`）である。**
**2 つの bddl を直接突き合わせて確認した**（2026-09-05・会社の RTX6000 上で実行）。

| | 順方向 `STUDY_SCENE4_…top_of_the_shelf`（tar `task 92`） | 逆向き（配布 `compe/t3` 黄色い本タスク） |
|---|---|---|
| `:fixtures` | `study_table`＋`wooden_two_layer_shelf_1` | 同一 |
| `:objects` | `black_book_1`・`yellow_book_1 yellow_book_2` | 同一 |
| `:obj_of_interest` | `yellow_book_2`・棚 | 同一 |
| `yellow_book_2` の初期 | 机上 `yellow_book_left_init_region` | 棚 `top_side` |
| `yellow_book_2` の目標 | 棚 `top_side` | 机上 `yellow_book_left_init_region` |

**初期配置と目標条件がきれいに反転している**（他の物体の配置は同じ）。
**この場面の順方向 bddl は `libero_90` の `STUDY_SCENE4` に 4 件あり、4 件とも tar に入っている**
（`task 62/79/92/106`）。

**注意 —— 2 つの落とし穴:**

1. **タスクの指示文は「the yellow book」ではなく「the book」である。**
   指示文の文字列で `yellow` を探すと全て取りこぼす。
   **シーン（fixtures・objects の構成）で一致させよ。**
2. **tar のエピソードにはフルなシミュレータ状態が無い。**
   `observation.state`（8 次元）と `action`（7 次元）だけで、物体の位置は含まれない。
   **「順方向デモの最終状態を `.pruned_init` に書き出す」には、
   `action` 列をシミュレータで順再生して最終フレームの状態を捕まえる手順になる。**
   この順再生自体が「デモがこちらのシミュレータで成立するか」の確認にもなる。

**Track3 の 4 タスクに対する tar の順方向デモの対応表**（2026-09-05・実測）:

| 配布の逆向きタスク | 順方向（反転元） | tar にデモがあるか |
|---|---|---|
| 黄色い本を棚から机へ | `task 92`（上） | **ある** |
| 赤いマグを左の皿から机へ | `put the red mug on the left plate`（`task 59`・`LIVING_ROOM_SCENE5`） | **ある** |
| 2 つのマグを皿から机へ | `put the white mug on the left plate and put the yellow and white mug on the right plate`（`task 30`・`LIVING_ROOM_SCENE5`） | **ある** |
| スープをトレイから机へ | スープをトレイへ置く順タスク（soup の順タスクは全て `basket` 向け・tray 向けは `cream cheese`） | **無い**（`basket` と `tray` は別 fixture） |

**3/4 の逆向きタスクの順方向デモは、配布 tar（会社の RTX6000 に 20 GB で実在する
`/home/ubuntu/src/train_data/libero_combined_20hz.tar`）から取れる。
Blackwell からの転送は不要。**

### 8.6 引き直すコマンド

```bash
# 配布 tar の info.json（打つのは配布物がある機械。Hugging Face には無い）
tar -xOf libero_combined_20hz.tar --wildcards '*/meta/info.json'

# 課題名の重なりの 2 個を特定する（まだ打っていない）
tar -xOf libero_combined_20hz.tar --wildcards '*/meta/tasks*'

# splits が読まれていないことの確認
grep -rn "\.splits" work/lerobot-src/src/lerobot/

# 8.5 節の確認（9/5・会社の RTX6000。pyarrow は pyenv の python3.12.14）
tar -xOf libero_combined_20hz.tar libero_combined_20hz/meta/info.json
# tasks.parquet の全タスクを出力（タスク文字列は `__index_level_0__` 列）
python3 -c "
import pyarrow.parquet as pq
for r in pq.read_table('libero_combined_20hz/meta/tasks.parquet').to_pylist():
    print(r['task_index'], r['__index_level_0__'])"
# シーンの一致は指示文ではなく fixtures・objects で見る（2 つの bddl を突き合わせる）
```

### 8.7 `action` 列の規約 —— 正規化された相対差分の指令値（2026-09-05）

**`action[:6]` は「今の姿勢からの相対差分」を ±1 に正規化した OSC への指令値である。**
絶対姿勢でもなければ、達成された差分（state の引き算）でもない。
**そのままシミュレータへ流して再生できる。**

根拠は評価用コンテナ（`parc2026-final`）でコントローラ設定を直接読んだこと:

| 実測した設定 | 値 |
|---|---|
| コントローラ | `OSC_POSE`（`OperationalSpaceController`） |
| `control_delta` | **`true`** ＝ 指令は相対差分 |
| `input_max` / `input_min` | `+1` / `-1` ＝ 入力は正規化済み |
| `output_max` | `[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`（並進 m・回転 rad） |
| `control_freq` | 20 Hz（1 指令 = 25 シムステップ） |

**014 が測った scale はこれと整合する。** 並進 0.0111 m/単位 ÷ 0.05 = 0.22、
回転 0.101 rad/単位 ÷ 0.5 = 0.20。**並進と回転が独立に同じ ≒0.2 を出す**のは、
「20 Hz の 1 ステップでは指令の 2 割しか到達しない」という応答比 1 つで説明が付く。
**`action` が「1 フレームの state 差分の定数倍」でないのはこのためで、
規約が違うからではない。**

**逆向きデモを作るときは符号を反転すること。** 相対差分なので、順方向が
d0..d(T-1) で x0 → xT へ動くなら、戻る経路は -d(T-1)..-d0 である。
順序だけ逆にすると元の場所には戻らない。

#### `action[6]`（gripper）は**エピソードによって符号化が違う**

**これは 2 種類のデータセットを足したことによる混在で、`libero_combined_20hz`
全体を走査して確かめた**（2026-09-05）。

| エピソード番号 | 値 | 意味 | 本数 |
|---|---|---|---|
| 0〜5047 | `{0, 1}` | **1 = 開く・0 = 閉じる** | 4,827 |
| 5186〜19532 | `{-1, +1}` | **-1 = 開く・+1 = 閉じる**（robosuite そのまま） | 13,736 |
| 2222〜5185 | `+1` のみ（**最後まで一定**） | 掴まないデモ。`{0,1}` 系なので **+1 ＝ 開いたまま** | 359 |
| 5193〜19454 | `-1` のみ（**最後まで一定**） | 掴まないデモ。±1 系なので **-1 ＝ 開いたまま** | 611 |

**4 行を足すと 19,533 になる**（13,736 + 4,827 + 611 + 359）。
下の 2 行は**開閉が 1 度も起きないデモ**で、指令が 1 値しか出てこない。

> **⚠ 復号はエピソード番号の範囲で分岐してはいけない。「負の値が出るか」で判定する。**
> 上の表の範囲は**重なっている**——`+1` 一定の 359 本は 2222〜5185 に散らばり、
> そのうち **138 本は `{0,1}` 系の上限 5047 を越えて 5048〜5185 に居る**。
> 「`ep <= 5047` なら `{0,1}` 系」と書くと、この 138 本が ±1 系と誤判定され、
> **`+1`（開いたまま）が「ずっと閉じる」に反転する。**
> 指の開き幅で確かめた結果、`+1` 一定の 359 本は全て幅 0.078 m（開）である
> （0〜5047 で平均 0.0788・5048〜5185 で平均 0.0782）。**±1 系で `+1` 一定＝
> ずっと閉じている、というエピソードはこのデータセットには 1 本も無い**ので、
> 符号による判定は 19,533 本すべてで正しく効く。

指の開き幅（`state[6] - state[7]`）と突き合わせて確認した。
例（ep 4061・0/1 系）: f0-52 は `+1` で幅 0.078（開いている）、f53-132 は `0` で
0.024 まで閉じ、f133 以降 `+1` で 0.079 へ開く。
例（ep 5195・±1 系）: f0 は `-1` で幅 0.078、f34 で `+1` にして 0.006 まで閉じる。

**`task 92`（黄本の順方向）の 44 本は全部 0/1 系である。**
生のまま robosuite へ流すと `+1` が「閉じる」・`0` が「そのまま」と解釈され、
**掴む／放すが完全に入れ替わって本を掴めない**。0/1 系は `a → 1 - 2a` で直す。
実装は `solution/tools/track3_reversal/reversal_probe.py` の `decode_gripper`。

**シム側の落とし穴**（状態復元で復らないもの・据え置きの配置・predicate の呼び方）は
[28_mppi_data_regeneration_survey.md](28_mppi_data_regeneration_survey.md) の 5.1 節。

### 8.8 `observation.state` の読み方と、姿勢を組み直すときの落とし穴（2026-09-05）

**`observation.state` は「フレーム毎の絶対姿勢」として健全である。**
2 本（ep 5195・92 コマ／ep 5225・184 コマ）を読んで確かめた——8 次元が全コマ揃い、
NaN は 0、eef_pos は静的コピーではない（先頭 50 コマで 237〜247 mm 動く）。
**逆向きデモの初期姿勢を元デモの最終コマから取る、という使い方はこの列で成立する。**

| 中身 | 実測 |
|---|---|
| `[0:3]` eef_pos | 生の m。範囲 ±0.48 |
| `[3:6]` 姿勢 | **軸角**（回転ベクトル）。ノルムが回転角 |
| `[6:8]` 指 qpos | **左右で符号が対称**（`state[7] ≒ -state[6]`）。**開き幅は `state[6] - state[7]`** で 0.004〜0.080 m |

指の値は **0/1 の離散値ではなく連続なサーボ位置**である（閉 0.002 m ↔ 開 0.040 m／指 1 本）。
±1 のバンバンは `action` 側の表現であって、`state` 側には無い。

#### ⚠ 相対回転は必ず SO(3) の合成で組む。軸角の引き算は誤り

姿勢を差分に直すとき、**軸角ベクトルを成分ごとに引いてはいけない**
（`np.diff(state[:, 3:6])`）。これは相対回転ではないので、姿勢が単位行列から離れるほど壊れる。
**`R_rel = R[t+1] @ R[t].T`（world 座標系）**を組んでから軸角に戻すこと。

| 組み方 | `action[3:6]` との成分相関（ep 5195／5225） |
|---|---|
| **world `R[t+1] @ R[t].T`** | **0.814／0.802** |
| body `R[t].T @ R[t+1]` | −0.225／−0.182（**符号が反転する**） |
| 軸角の引き算（誤り） | 0.221／0.335 |

**body で組むと相関が負に振れる**ので、取り違えると静かに逆向きの回転が入る。

#### `action[t]` は `t → t+1` の差分（1 コマ遅れではない）

並進の成分相関で判定した: `action[t]` ↔ `t→t+1` が 0.9525／0.9501、
`action[t]` ↔ `t-1→t` が 0.8891／0.9271。**前者を採る。**

#### 応答比はエピソードごとに当て直す

8.7 節の応答比（≒0.2）は**定数ではない**。全 19,533 本から 20 本を無作為抽出して
当て直すと、**並進は平均 0.01110 m/単位・標準偏差 0.00211（±19%）・範囲 0.00580〜0.01293**、
回転は平均 0.10149 rad/単位・±10%。接触や荷重で変わるためで、データの異常ではない。
**平均値を焼き込まないこと**——最小の 0.0058 は平均の半分である。

**なお逆向きデモの生成にこの定数は要らない。** 保存された `action` を
`-actions[::-1]` で符号反転して使うのが実装（8.7 節）で、そこに応答比は現れない。
定数が要るのは「state から action を組み直す」経路を採る場合だけである。

#### 出どころ

`~/src/train_data/verify014_calib.py`（`offset`／`scale`／`lag`／`sweep`／`gripper`／
`roundtrip` の 6 チェック・1.4 秒）。**リポジトリ外**にあるので、必要になったら
`solution/tools/track3_reversal/` へ移すこと。展開済みデータは
`~/src/train_data/libero_combined_20hz_extracted/`。
