# libero_plus_bowl5

*English: [README.md](README.md)*

`lerobot/libero_plus`（14,347 本）から、**採点中に近くの物へぶつかる率がいちばん高かった
「黒い丼」5 課題**を外し、**より丁寧に掴んで置く台本の手本に入れ替えた**ものである。
それ以外は `libero_plus` のままで、**動画はハードリンクで再利用している**（作り直していない）。

| | |
|---|---|
| 本数 | **14,085** |
| コマ数 | **2,264,704** |
| 課題数 | **40** |
| 形式 | LeRobot v3.0・20 Hz |

## なぜこの 5 課題か

**5 つとも、別の物の隣か上にある黒い丼を掴む課題である。**丼は指より広いので真上からは掴めず、
**縁の壁をまたぐ**必要がある。

- `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`

**配布の手本では、人が掴む半径（手首が回転軸から約 4.0 cm）が、縁の壁の実際の中心
（約 4.71 cm）より内側にある。**そのため片方の指が先に壁に当たり、**掴むのではなく押してしまう。**
測定した LIBERO-plus 全課題で、不要な接触によって失っていた約 17 ポイントのうち、
**11.3 ポイントがこの一族だった。**

## 中身の内訳

| 部分 | 本数 | 注 |
|---|---|---|
| `libero_plus` から丼 5 課題を除いたもの | 12,467 | `lerobot/libero_plus` からハードリンク。手を入れていない |
| 撮り直した台本の手本（丼 5 課題） | 1,618 | 外した 1,880 本の代わり |
| **合計** | **14,085** | 40 課題・2,264,704 コマ・20 fps |

**台本の方策（`solution/tools/scripted_demo/scripted_demo.py` の `RimPickPlace`）で作った。**
**掴む半径を縁の壁の実際の中心に合わせ、接触の直前で降りる勢いを落としてある。**
初期状態は 2 通りから引いた——評価器が使う課題ごと 50 通りの摂動姿勢と、
人の手本の初期姿勢の周りから引いた 50 通り（5 課題のうち 4 課題）。

**目標に届かなかった本と、対象外の物を 1 mm 以上動かした本は捨てている。**
だから約 1,880 本のうち 1,618 本しか残っていない。
（この一族は、掴みのずれ 9.67 cm までなら接触なしの成功率 130/150・86.7 % と別途測ってある
——元の人の手本に含まれるずれ 1.0 cm 以下よりはるかに広い範囲である。）

**指示文は「配布データの言い回し」に合わせてある**——採点時の生の BDDL の文ではない。
**評価の仕組みが、採点器の指示文を学習データの 40 個の文字列のどれかに写してから方策へ渡す**ため、
語彙の外の言い回しで撮った手本は、その対応表に載らず**推論時に一度も引かれない。**

## 形式

`lerobot/libero_plus` と同じ列である。

| 列 | 型 | 形 |
|---|---|---|
| `observation.images.front` | video | [256, 256, 3] |
| `observation.images.wrist` | video | [256, 256, 3] |
| `observation.state` | float32 | [8] |
| `action` | float32 | [7] |

## 使い方

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("siruku6/libero_plus_bowl5")
```

## 狙いと限界

- **`lerobot/libero_plus` の差し替えとして作った。**配布データとの意図的な違いは丼 5 課題だけなので、
  **他を固定して比べるためのものである。**
- 撮り直した手本は**模擬器の中だけのもの**（LIBERO ＋ robosuite/MuJoCo の Franka Panda）。
  `libero_plus` の他の部分と同じである。
- **触ったのは名前を挙げた 5 課題だけである。**初期姿勢と見た目も振ったものは
  [`libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose)
  にある（16,820 本）。
- **学習に効くかは、はっきりしない。**行動側だけを学習させて入れ替えたところ、
  手元の Track2 の合成点は **0.3656 → 0.3354 と下がった**（1 つのカメラ摂動条件に偏って落ちた）。
  全パラメータ学習では 0.4004（比べた 4 つの重みで最良）だったが、
  **「配布データで全パラメータ学習」という対照が無い**ので、
  データの効果か学習のやり方の効果かは切り分けられていない。

## ライセンスと出典

| 部分 | ライセンス |
|---|---|
| LIBERO（Copyright (c) 2023 UT-Austin Robot Perception and Learning Lab / Lifelong-Robot-Learning） | MIT |
| `lerobot/libero_plus`（手を入れずハードリンクした部分） | MIT（LIBERO から継承） |
| 撮り直した台本の手本（このリポジトリの寄与） | MIT |
| robosuite | MIT |
| MuJoCo | Apache-2.0 |
