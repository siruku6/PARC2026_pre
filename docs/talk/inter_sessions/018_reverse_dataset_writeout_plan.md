# 018 逆向きデータを「学習に載る形」で書き出すまでの段取り

**前提はここまでで揃っている。**逆向きの軌道は t3 ルートで 150/150 通った
（[016 の末尾](016_reverse_transfer_probe_on_current_weights.md)）。
残っているのは**成否を測るだけの道具に、書き出し口を付けること**である。

## 1. 必要なものと、その置き場

| 要るもの | 事実の置き場 |
|---|---|
| **1 本に何が要るか（5 つ）・出来上がりの形・npz の鍵・踏む穴** | **[reference/29](../../reference/29_training_data_requirements.md)** |
| データセットの規約（配布物を数えた結果・`action` の意味・gripper の符号化） | [reference/25 の 8 節](../../reference/25_dataset_inventory.md) |
| 配布 T3 の中身（init は 4 タスク × 各 50 行・幾何・env をまたぐ写し替え） | [reference/28_track3_contents](../../reference/28_track3_contents.md) |
| シムを回すときの落とし穴（状態復元・fixture・時刻反転） | [reference/28_mppi_...](../../reference/28_mppi_data_regeneration_survey.md) の 5.1〜5.2 節 |

| 使う道具 | 状態 |
|---|---|
| [`track3_reversal/gripper_shift_sweep.py`](../../../solution/tools/track3_reversal/gripper_shift_sweep.py) | 逆向きを回して成否を測る。**書き出し口が無い** |
| [`track3_reversal/replay_video_probe.py`](../../../solution/tools/track3_reversal/replay_video_probe.py) | `rollout_phased` 等の本体 |
| [`scripted_demo/scripted_demo.py`](../../../solution/tools/scripted_demo/scripted_demo.py) の `SavedEpisode` | **npz の取り決めの唯一の置き場。写しを作らない** |
| [`scripted_demo/npz_to_lerobot.py`](../../../solution/tools/scripted_demo/npz_to_lerobot.py) | `build` / `inspect` / `align` / `merge`。**そのまま使える** |

## 2. まだ無いもの

**4 つだけである。**どれも `track3_reversal` 側にある。

| # | 無いもの | いまの状態 |
|---|---|---|
| 1 | `action` 列 | `rollout_phased` が `acts` を集めているのに**返していない** |
| 2 | `observation.state` の**指 2 次元** | eef 位置と軸角しか記録していない。`robot0_gripper_qpos` が要る |
| 3 | **手首カメラ** | `agentview` しか描画していない。`robot0_eye_in_hand` を足す |
| 4 | 指示文の読み出し | bddl の `:language` を読んでいない |

**指示文を書く作業は無い。**配布 T3 の bddl に入っている
（[reference/29 の 4 節](../../reference/29_training_data_requirements.md)）。

## 3. 他タスクへ広げられるか——**順方向が要る**

逆向き軌道は順方向デモの時刻反転で作るので、**T3 の各タスクに対応する順方向が
`libero_90` にあるか**が前提になる。調べた結果:

| T3 タスク | 対応する順方向 | 可否 |
|---|---|---|
| yellow book | `STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf` | **済**（150/150） |
| alphabet soup | `LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray` | **1:1 の逆。行ける** |
| red mug | `LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate` | **1:1 の逆。行ける** |
| two mugs | `..._put_the_white_mug_on_the_left_plate` ＋ `..._put_the_yellow_and_white_mug_on_the_right_plate` | **2 本の合成が要る。後回し** |

## 4. やる順番

**1 → 2 → 3 → 4。**1 はどちらの方向にも必要なので、これが最初に来る。

| # | やること | なぜその順か |
|---|---|---|
| **1** | **書き出し口を作る**（2 節の 4 つ）。npz を `SavedEpisode` の鍵で書く | これが無いと何も検証できない。ここまでで変換から先は既存コードに乗る |
| **2** | 本タスク 1 本を通しで検証。`build` → `inspect` | **ファイルが揃うことと、学習が読めることは別である。**動画の符号化はここで初めて落ちる |
| **3** | **soup と red mug へ広げる**。掴む姿勢の作り方（`clamped` + standoff）がそのまま効くか | 1 タスクだけのデータで学習すると崩れる実績がある（[reference/29 の 5 節](../../reference/29_training_data_requirements.md)） |
| **4** | two mugs（2 本の合成）と、はみ出し対策 | どちらも 3 までの結果が出てから要否を判断する |

## 5. 決めてあること

- **失敗した回は入れない。**ep4062 の `t3-lying`／`forward-final` は落ちるので外す。
  `npz_to_lerobot.py build` が `success` を見て既定で捨てるので、`success` を
  正しく書けばよい。
- **`t3` ルート（150 本）が主力。**本番と同じ形（腕は home・本は立っている）で、
  本の xy 114×49 mm・ヨー 42° の振れをそのまま持っている。
- **`t3-lying`（2 本）は汎化のための添え物。**倒れた本の初期状態は配布 T3 にも
  LIBERO-90 にも 1 つも無いので、隠しタスクに備えるなら自分で作るしかない。
  ただしこのルートは init 行を変えても結果が変わらない（本の姿勢を上書きするため）
  ので、**デモ 1 本につき 1 通りしか作れない。**
