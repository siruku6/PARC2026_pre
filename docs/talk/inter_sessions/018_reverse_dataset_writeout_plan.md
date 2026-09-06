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

## 2. 書き出し口に足りなかったもの——**4 つとも入れた**

どれも `track3_reversal` 側にあった。`--write-npz` を付けると `SavedEpisode`
の鍵で書き出す。

| # | 足りなかったもの | 入れた形 |
|---|---|---|
| 1 | `action` 列 | `rollout_phased` に `capture=True` を足し、`acts` を返す |
| 2 | `observation.state` の**指 2 次元** | `obs_state8` が `robot0_gripper_qpos` を連結する |
| 3 | **手首カメラ** | `make_render_env` を複数カメラ対応にし、`robot0_eye_in_hand` を足す |
| 4 | 指示文の読み出し | `env.language_instruction` をそのまま書く |

**指示文を書く作業は無い。**配布 T3 の bddl に入っている
（[reference/29 の 4 節](../../reference/29_training_data_requirements.md)）。

## 3. 他タスクへ広げられるか——**順方向が要る**

逆向き軌道は順方向デモの時刻反転で作るので、**T3 の各タスクに対応する順方向が
`libero_90` にあるか**が前提になる。調べた結果:

| T3 タスク | 対応する順方向 | 可否 |
|---|---|---|
| yellow book | `STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf` | **済**（150/150） |
| alphabet soup | `LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray` | **通った**（真上からの握りが要る。[reference/30 の 4 節](../../reference/30_reverse_grasp_pose.md)） |
| red mug | `LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate` | **通った**（順方向デモ 20 本中 14 本が 2/2） |
| two mugs | `..._put_the_white_mug_on_the_left_plate` ＋ `..._put_the_yellow_and_white_mug_on_the_right_plate` | **2 本の合成が要る。後回し** |

## 4. やる順番

**1 → 2 → 3 → 4。**1 はどちらの方向にも必要なので、これが最初に来る。

| # | やること | なぜその順か | 状態 |
|---|---|---|---|
| **1** | **書き出し口を作る**（2 節の 4 つ）。npz を `SavedEpisode` の鍵で書く | これが無いと何も検証できない。ここまでで変換から先は既存コードに乗る | **済** |
| **2** | 本タスク 1 本を通しで検証。`build` → `inspect` | **ファイルが揃うことと、学習が読めることは別である。**動画の符号化はここで初めて落ちる | **済** |
| **3** | **soup と red mug へ広げる**。掴む姿勢の作り方（`clamped` + standoff）がそのまま効くか | 1 タスクだけのデータで学習すると崩れる実績がある（[reference/29 の 5 節](../../reference/29_training_data_requirements.md)） | 生成中 |
| **4** | two mugs（2 本の合成）と、はみ出し対策 | どちらも 3 までの結果が出てから要否を判断する | 未着手 |

**3 で分かったことは 2 つあり、どちらも [reference/30](../../reference/30_reverse_grasp_pose.md) に置いた。**

1. **掴む姿勢は、物の向きが順方向デモと一致していないと移せない。**缶は順方向で
   立っているのに配布 T3 ではトレイの中で横倒しで、移すと壁に阻まれる。真上から
   直径を挟む姿勢を別に作って通した（0/4 → **4/4**）。
2. **順方向デモには再生が再現しないものが混ざる。**マグは 20 本中 6 本で物が
   1 mm も動かず、その 6 本は逆向きもほぼ全滅した。**物が動いたか**でふるう。

本タスクの 150 本は LeRobot 形式まで出来ている（26,443 コマ、npz 9.7 GB →
140 MB）。

**3 のあと、経路そのものに 2 つ穴が見つかった。どちらも直して
[reference/31](../../reference/31_reverse_place_without_contact.md) に置いた。**

3. **掴んだ物を横へ引きずって、下の皿やトレイを動かしていた。**掴んだら先に
   真上へ 80 mm 逃げ、復路には**高さの床を張って**合流する（コマを飛ばすと姿勢の
   変化まで飛んで荷物が傾く）。皿 21.0 mm → **0.9 mm**、トレイ 6.4 mm → **0.1 mm**。
4. **空中で放して落としていた。**放す時刻を「順方向が実際に挟み終えたコマ」の
   反転にし、そこから**触れるまで下ろしてから**開く。缶の落下 36.9 mm →
   **2.6 mm**、置いた位置も goal 領域の中心まで 60.8 mm → **1.2 mm**。

5. **赤マグだけ、傾けたまま置いていた。**皿の上に傾いて載っているマグを、その
   傾きごと写し取って運んでいた。放す前に**荷物の立つ軸を鉛直へ直す**と
   10.5° → 0.3°、落下 7.5 mm → **0.4 mm**。**缶に同じことをしてはいけない**
   （最後の一手で 90° 回してケチャップを 116 mm 薙ぎ倒す）。

**既存の 389 本はこの 3 つを直す前のものである。**作り直す手順は
[020](020_reverse_dataset_regeneration.md)（本・缶は `LIFT=0.08 RELEASE=contact`、
赤マグはそれに `LEVEL=1`）。

**順方向デモのふるい（上の 2）は仕組みになっていない。**手で選んでいる。
配布の順方向 14,347 本を含めてどう外すかは [019](019_forward_replay_failure_exclusion.md)。

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
