# 019 再生しても再現しない順方向デモを、学習データから外す

> **状態: 依頼中** ／ 受け側: 未定 ／ 更新: 2026-09-07
>
> **まだ実施しなくてよい。**現状・再現手順・方針の概要をここに置く。
> **どう外すか（判定の設計と、外す範囲）を決めるところから、この依頼に含まれる。**

## 1. 何を疑っているか

**配布された学習データの中に「記録された行動を再生しても、その通りに動かない
デモ」が混じっている。**そういうデモは、物を掴めていない・落としている・目標に
届いていない動きを含む。**それをそのまま学習すると、失敗する動かし方を教えて
いることになる。**

これは 2 つの母集団に別々に効く。

| 母集団 | いまの扱い | 疑い |
|---|---|---|
| **配布の順方向データ**（`libero_plus` 系、14,347 本。[reference/06](../../reference/06_training_data_composition.md)） | **そのまま全部学習に入れている** | **未測定。**何本が再現しないのか誰も数えていない |
| **こちらで作った逆向きデータ**（389 本。[reference/30 の 9 節](../../reference/30_reverse_grasp_pose.md)） | 逆向き goal が立った回だけ書いている | 元にした順方向デモは**手で選んだ**。ふるいが仕組みになっていない |

## 2. 分かっていること

### 2.1 再現しないデモは実在し、割合も小さくない

**赤マグの順方向デモ 20 本を再生して数えた**（[reference/30 の 5 節](../../reference/30_reverse_grasp_pose.md)）。

| 順再生で物が動いたか | 本数 | そこから作った逆向きの成否 |
|---|---|---|
| 動いた | 14 | **全部 2/2** |
| 動かなかった（1 mm 未満） | 6 | 0/2 が 5 本・1/2 が 1 本 |

**20 本中 6 本（3 割）が再現しない。**缶でも 1 本見つかっている
（`ep3698`。動画は [work/shares](../../../work/shares/README.md) の
`FORWARD_soup_ep3698.mp4`）。

### 2.2 「goal が立ったか」では判定できない

順再生のあとに `env.check_success()` を見る手もあるが、**当てにならない**。

- 逆向き goal が 2/2 通ったマグの `ep2199` / `ep2202` は、**順方向 goal が
  `False`** だった（`work/probe_017_mug/gripper_shift_sweep.json`）。
- 原因は述語の相手である。マグの順方向 goal は `plate_1` という**物**が相手で、
  region site を前提にした評価が噛み合わない（[reference/30 の 5 節](../../reference/30_reverse_grasp_pose.md)）。

**いま使えている判定は「順再生で物が動いたか」だけである。**これは効くが、
「動いたが目標には届いていない」を見逃す。

### 2.3 **再現しない理由の 1 つは「シーンが違う」である**（9/7 に判明）

**指示文が同じでもシーンは 1 つとは限らない。**標準スイート由来のタスクは
`_add_1`…`_add_N` の摂動版が **1 指示文あたり 203 シーン**あり、デモはその
203 シーンに散っている（[reference/06 の末尾](../../reference/06_training_data_composition.md)）。

**素のシーン 1 つで全部を再生すれば、当然その多くは再現しない。**two mugs で
20 本中 10 本が落ちたのはこれで説明が付く（デモ 313 本 / 203 シーン）。
一方 `libero_90` 由来のタスク（本・缶・赤マグ）は**シーンが 1 つしかない**ので、
この原因は当たらない——そちらの再現しないデモは別の理由である。

**どのエピソードがどの variant かを示す列は無い**（[reference/06](../../reference/06_training_data_composition.md) の
「variant 名はどの列にも無い」）。**だから「再生して確かめる」以外の方法が無い。**
この依頼の判定を設計するとき、**2 つの母集団を混ぜないこと。**

### 2.4 配布データには成功フラグが無い

LeRobot v3.0 の `meta` に `is_success` 列は無い
（`solution/tools/track3_reversal/host_extract_actions.py` の冒頭に実測が
書いてある）。**メタデータを読むだけでは選別できず、1 本ずつ再生するしかない。**

## 3. 再現手順

### 3.1 順方向デモを npz に抜く（ホスト側）

```bash
cd /home/ubuntu/src/PARC2026_pre
python3 solution/tools/track3_reversal/host_extract_episode.py \
  --dataset-root /home/ubuntu/src/train_data/libero_combined_20hz_extracted \
  --task-index 49 --out work/probe_018/ep_check.npz
```

`--task-index` はタスク番号。指示文で引くなら `host_extract_actions.py` の
`--task-text` を使う。**pyarrow はホスト側にしか無い**ので、コンテナでは動かない。

### 3.2 再生して、物が動いたかを見る（コンテナ）

```bash
OUT=$PWD/work/probe_018 TASK=mug EP_NPZ=ep_check.npz \
  MODE=phased INIT_SOURCE=t3 GRASP_FRAME=clamped STANDOFF=0.06 \
  T3_INIT=0,1 MAX_APPROACH=120 \
  bash solution/tools/track3_reversal/run_gripper_shift_sweep.sh "0" ""
```

1 本ごとにこの行が出る。**`本 A -> B` の差が「順再生で物が動いた量」**である。

```
=== ep 2199  T=129  初期状態 #12 (3.2 mm) ===
  順再生: ずれ 中央 4.1 mm  :goal=False  本 [x y z] -> [x y z]
  !! 順再生が goal に届いていない。この ep の逆向きは参考値
```

同じ値は `work/probe_018/gripper_shift_sweep.json` の
`book_start` / `book_forward_end` / `forward_goal` にも入る。

**注意: `run_gripper_shift_sweep.sh` は逆向きまで回すので重い。**判定だけが
目的なら、順再生の区間（`gripper_shift_sweep.py` の `main` の「1)」の箱）だけを
呼ぶ小さなプローブを作るほうが速い。

### 3.3 いまの学習データがどれを使っているか

| もの | 置き場 |
|---|---|
| 逆向き 389 本（本 150・缶 100・マグ 139） | `work/probe_017/reverse_lerobot_all` |
| 元にした順方向デモ | 本 `ep4061,4062,4063` ／ 缶 `ep3696,3697` ／ マグ `ep2206,2199,2202` |
| 順方向デモの npz | `work/probe_017/ep_data.npz`・`ep_soup.npz`・`work/probe_017_mug/ep_mug20.npz` |

## 4. 対処の方針（概要）

**決めることは 3 つある。順に片付ける。**

### 4.1 判定をどう作るか

「順再生で物が動いたか」を出発点にして、**見逃している「動いたが届いていない」を
どう拾うか**を決める。候補は次のとおりで、どれを採るかは測ってから決めてよい。

- 物の終点が、bddl の goal 領域の箱に入っているかを**幾何で**見る
  （`gripper_shift_sweep.eval_in_t3` と同じやり方。述語を呼ばずに済む）
- 再生の腕のずれ（`err_m` の中央値）に閾を置く
- 記録された `observation.state` の終点と、再生の終点を突き合わせる

### 4.2 どこまで外すか

**逆向き 389 本のほうは、既に成否で絞ってあるので優先度は低い。**効きが大きい
のは配布の順方向 14,347 本のほうだが、**そちらは 1 本ずつ再生するとシムの時間が
効いてくる。**まず 1 タスクぶんを測って「何割が落ちるか」「1 本あたり何秒か」を
出し、全体に掛ける値打ちがあるかを判断する。

### 4.3 外したあと何が変わるかを測る

**データを減らすと必ず良くなるとは限らない。**除外版と現行版で学習を回して
比べる段取りが要る（[reference/11](../../reference/11_training_schedule.md)）。
比較の前に、**除外で何本減るのか**だけは先に出しておくこと。

## 5. 関連

| 何 | どこ |
|---|---|
| **対になる依頼——目当て以外の物を動かしたデモを外す** | **[025](025_collision_episode_exclusion.md)**（**再生は 1 回で、こちらのラベルと 025 のラベルの両方が取れる。**実施するときは同じ走りに相乗りさせる） |
| 順方向デモのふるい（実測と、goal 述語が使えない理由） | [reference/30 の 5 節](../../reference/30_reverse_grasp_pose.md) |
| 逆向きデータの作り方全体 | [018](018_reverse_dataset_writeout_plan.md) |
| 学習データ 14,347 本の内訳 | [reference/06](../../reference/06_training_data_composition.md) |
| シムを回すときの落とし穴 | [reference/28_mppi_...](../../reference/28_mppi_data_regeneration_survey.md) 5.1〜5.2 節 |
