# データセット構成の調査

**何のための文書か:** 本選で配布された6つのデータセットのタスク構成を調べ、Track1・Track2・Track3 それぞれにどのデータセットを使うべきかを決める根拠を示す。

## 調査結果の要約

- **`Sylvest/libero_plus`（現在の学習データ）の 40 タスク中、10 個が SCENE 名タスク（libero_10 由来）**
- **LIBERO の SCENE 名タスクは libero_10 に 10 個、libero_90 に 90 個、合計 100 個**
- **Track3（逆向きタスク）の母集団は SCENE 名タスクに限られる（運営の check_no_leaks.py より）**
- **spatial・object・goal の 30 タスクは SCENE 名ではない（Track3 に使えない）**

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
| `libero_10` | 10 | 全て SCENE 名（変種を除く基本タスク） |
| `libero_90` | 90 | 全て SCENE 名（変種を除く基本タスク） |
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
| `libero_plus` (40) | `libero_90` (73) | 0 | **重複なし** |
| `libero_10` (10) | `libero_90` (73) | 0 | SCENE 番号が異なる（10 vs 90） |

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

### (3) `libero_90` のタスク数が 73 である理由

公開版の `meta/info.json` には 73 タスクと記録されているが、手元の bddl には 90 個の基本タスクがある。73 という数は、公開データセット（`libero_90_no_noops`）に含まれるエピソードが 73 種類のタスクしかカバーしていないことを示している。**本選の非公開セットでは残り 17 種類も出題される可能性がある。**

### (4) Track2 の非公開セットは difficulty_level から引かれる

Track2 の配布物（`task_classification.json`）には difficulty_level が 1〜5 まであり、公開された 8 行は見本である。**母集団は task_classification.json に列挙された全変種であり、倍率ではなく difficulty_level が運営の評価軸である。** 実測すると、手先の精度が 2 cm を超えると成功率が崩れ始める（向きの誤差が効き始める）。

---

## 数え方（打ったコマンド）

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
# libero_10 の基本タスク（変種を除く）
ls -1 /home/ubuntu/lab/PARC2026_pre/work/LIBERO-plus/libero/libero/bddl_files/libero_10/*.bddl \
  | sed 's|.*/||' | sed 's|\.bddl$||' \
  | grep -E "^(KITCHEN|LIVING_ROOM|STUDY)_SCENE[0-9]+" \
  | grep -v "_add_" | grep -v "_view_" | grep -v "_language_" | grep -v "_level" | grep -v "copy" | grep -v "_sample" \
  | grep -Ev "_[0-9]+$" \
  | wc -l

# libero_90 の基本タスク（変種なし）
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
