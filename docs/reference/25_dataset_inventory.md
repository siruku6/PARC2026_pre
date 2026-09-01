# データセット構成の調査

**何のための文書か:** 本選で配布された6つのデータセットのタスク構成を調べ、Track1・Track2・Track3 それぞれにどのデータセットを使うべきかを決める根拠を示す。

## 調査結果の要約

- **`Sylvest/libero_plus`（現在の学習データ）の 40 タスク中、10 個が SCENE 名タスク（libero_10 由来）**
- **LIBERO の SCENE 名タスクは libero_10 に 10 個、libero_90 に 90 個、合計 100 個**
- **Track3（逆向きタスク）の母集団は SCENE 名タスクに限られる（運営の check_no_leaks.py より）**
- **spatial・object・goal の 30 タスクは SCENE 名ではない（Track3 に使えない）**

**2026-08-31 に公開版の `meta/` を落として数え直して分かったこと**（7 節）:

- **`libero_90` の「73 タスク」は指示文を重複を除いて数えた数で、90 個の bddl のうち 89 個が入っている**（欠けは 1 個だけ）
- **IPEC-COMMUNITY の 5 スイートは LeRobot v2.1、`lerobot/libero_plus` だけ v3.0**
- **`lerobot/libero_combined_20hz` は Hugging Face に無い。**配布された tar からしか中身を読めない
- **配布の `dataset_IPEC-COMMUNITY.zip`（5.11 GB）は、公開版 5 スイートの合計 5.32 GB とほぼ同じ大きさである**

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
ls -1 /home/ubuntu/lab/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_10/*.bddl \
  | sed 's|.*/||' | sed 's|\.bddl$||' \
  | grep -E "^(KITCHEN|LIVING_ROOM|STUDY)_SCENE[0-9]+" \
  | grep -v "_add_" | grep -v "_view_" | grep -v "_language_" | grep -v "_level" | grep -v "copy" | grep -v "_sample" \
  | grep -v "_[0-9]\+$" \
  | wc -l
# 結果: 10

ls -1 /home/ubuntu/lab/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_90/*.bddl \
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

| トラック | 中身 | 使うべきデータセット | 根拠 |
|---|---|---|---|
| **Track1** | 学習済みタスク・背景と照明が変わる | `libero_plus` + 標準スイート 4 つ | エピソード数を増やす（摂動への頑健性）。タスクの種類は増えない |
| **Track2** | 学習済みタスク・視点・姿勢・配置・雑音が変わる | `libero_plus` + 標準スイート 4 つ + **`libero_90`** | エピソード数 + タスクの種類を両方増やす。difficulty_level（task_classification.json）から非公開セットが引かれる |
| **Track3** | 既存タスクの逆向き（SCENE 名に限られる） | **`libero_10` + `libero_90`** | Track3 の母集団は SCENE 名タスク（100 個）。spatial・object・goal は使えない |

**Track3 の根拠（詳細）:**

運営の検査スクリプト `/home/ubuntu/lab/b_PARC2026_final/scripts/check_no_leaks.py` にある禁止語:

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
cd ~/lab/PARC2026_pre && H=~/.cache/huggingface/hub
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
ls -1 /home/ubuntu/lab/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_10/*.bddl \
  | sed 's|.*/||' | sed 's|\.bddl$||' \
  | grep -E "^(KITCHEN|LIVING_ROOM|STUDY)_SCENE[0-9]+" \
  | grep -v "_add_" | grep -v "_view_" | grep -v "_language_" | grep -v "_level" | grep -v "copy" | grep -v "_sample" \
  | grep -Ev "_[0-9]+$" \
  | wc -l

# libero_90 の元タスク（variant なし）
ls -1 /home/ubuntu/lab/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_90/*.bddl | wc -l
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

libero_10_path = Path('/home/ubuntu/lab/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_10')
libero_90_path = Path('/home/ubuntu/lab/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_90')

libero_10_base = get_true_base_tasks(libero_10_path)
libero_90_base = get_true_base_tasks(libero_90_path)

print(f"libero_10: {len(libero_10_base)} tasks")
print(f"libero_90: {len(libero_90_base)} tasks")
```
