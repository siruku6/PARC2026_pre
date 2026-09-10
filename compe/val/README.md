# compe/val — 手元で評価を回すためのタスク一覧

**本番の採点に出るタスクは非公開で、配布キットに入っている例は
[compe/t1/T1_TASKS.csv](../t1/T1_TASKS.csv) の 4 行しか無い。**
それでも評価の仕組みは「CSV の 1 行から LIBERO のタスクを組み立てる」だけなので、
**同じ列の CSV を自分で書けば、手元でも本番と同じ形で好きなタスクを測れる**
（[docs/reference/02](../../docs/reference/02_evaluation_data.md)）。

**ここはその「自分で書いた CSV」の置き場である。**
1 枚が 1 つの問いに対応していて、問いが違えば行の選び方も違う。
**どれを渡すかを間違えると、出てきた数字が答えている問いごと変わる。**
だからこの README は、**残してある CSV それぞれについて「何を知るための行なのか」**を書く。

| 節 | 中身 |
|---|---|
| [1. 回し方](#1-回し方) | どの CSV も同じ口から渡す。列の意味 |
| [2. 残してある一覧](#2-残してある一覧) | **何のために残しているか**の表 |
| [3. 評価実行例](#3-評価実行例) | 実際に回して出た数字のごく概要 |
| [4. 消したもの](#4-消したもの2026-09-10) | 消した 3 枚と、戻し方 |

---

## 1. 回し方

**どの CSV も、同じ 1 つの口から渡す。**
**評価する側のコードは、この CSV を配布キットの CSV の上にかぶせて読む。**
だから回すときに変えるのはファイル名だけで、他の手順は同じである。

| どのキットか | 環境変数 | 打つもの |
|---|---|---|
| **本選キット**（Track1 / Track2 / Track3） | **`TASKS_CSV`** | [`solution/tools/eval/run_final_eval.sh`](../../solution/tools/eval/run_final_eval.sh) |
| 予選キット（36 件の評価） | `EVAL_CSV` | [`solution/tools/eval/run_docker_eval.sh`](../../solution/tools/eval/run_docker_eval.sh) |

```bash
# 本選キットで Track2 を 72 行 × 3 エピソード = 216 本回す
MUJOCO_GL=egl TASKS_CSV=compe/val/TRACK2_VAL72.csv TRACK_EPISODES=3 \
  WEIGHTS=$PWD/work/artifacts/pi05_rp3000 LABEL=val72base \
  bash solution/tools/eval/run_final_eval.sh track2
```

**回す前に、その CSV の行が成り立つかを確かめられる。**
場面の定義（`.bddl`）と初期状態の一覧（`.pruned_init`）が実在しない行が 1 つでもあると、
例外がトラック全体を止めてしまう。

```bash
python3 solution/tools/eval/check_validation_suite.py \
    --csv compe/val/TRACK1_VAL29.csv \
    --libero-plus work/LIBERO-plus --kit-dir ../b_PARC2026_final
```

### 1.1 列の意味

**列は配布キットの `T1_TASKS.csv` と同じである**（`HOLDOUT_TASKS.csv` だけは別物・2 節）。

> **ここで「摂動」とは、元のタスクに見た目や配置の変更を 1 つ当てたもののことである。**
> LIBERO-plus は 7 種類（物の配置・カメラ視点・ロボット初期姿勢・言語指示・照明・
> 背景テクスチャ・センサ雑音）を定義していて、全部で 10,030 個の variant がある
> （[docs/reference/05](../../docs/reference/05_perturbation_categories.md)）。

| 列 | 中身 |
|---|---|
| `task_num` | ファイルの中での通し番号。評価側は使わない |
| `task_id` | **variant の名前。**元タスク名に摂動の接尾辞が付く（`_light_29`・`_tb_15`・`_view_45_0_100_0_0_initstate_0` など） |
| `instruction` | 方策に渡す指示文 |
| `libero_plus_id` | `task_classification.json` での通し番号。摂動なしの行は空 |
| `suite` | `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` |
| `category` | 摂動の種類（`Base` は摂動なし） |
| `difficulty_level` | **書き方が一定でない。**運営の 1〜5、`L2`〜`L5`、こちらで書いた `az+45` が混じる |
| `init_states_file` | **開くべき初期状態のファイル名。**接尾辞を正規表現で剥がすと取りこぼすので、実在するファイル名との最長前方一致で決めて書き出してある |
| `n_nontarget_objects` | 対象外の物体の数。**1 mm 動かしたら失敗**という判定に効く |

---

## 2. 残してある一覧

**Track1 側（背景・照明）と Track2 側（視点・初期姿勢）で、母集団がまったく違う。**
`EVAL_TASKS.csv` の 36 行には Track2 の摂動が 1 件も入っていない
（背景 20・照明 16・カメラ 0・初期姿勢 0）。**この 2 つを混ぜて読まないこと。**

| ファイル | 行数 | **何を知るための行か** | 状態 | 作り直し方 |
|---|---|---|---|---|
| **`EVAL_TASKS.csv`**（＋ `.meta.json`） | 36 | **Track1 の親元。**「学習が引かなかった実演」だけから作った検証用の集合で、予選期からの提出候補の比べ相手はすべてこれ。多くの道具の既定値でもある（`run_docker_eval.sh`・量子化の 4 本・Colab 評価） | **現役**（親元として） | `python3 solution/tools/suite/make_eval_split.py`。根拠は [reference/13](../../docs/reference/13_holdout_episodes.md) |
| **`TRACK1_VAL29.csv`**（＋ `.meta.json`） | 29 | **いまの Track1 の測り。**36 行のうち、本選キットが初期状態を開けずに落ちる `_tb_NN` の 7 行を外したもの。キットの `T1_TASKS.csv` は 4 行しか無く、1 本の入れ替わりが 8 ポイント動くので使えない | **現役**（9/10 に作った。まだ回していない） | `.meta.json` に選び方と、`check_validation_suite.py` での確かめ方がある |
| **`TRACK2_VAL72.csv`** | 72 | **いまの Track2 の測り。**12 タスク × 6 条件（摂動なし・カメラ方位角 +45°/−60°・ロボット初期姿勢 0.1/0.2/0.4）。12 タスクは 8/24 の 240 本で摂動なしなら 3 回とも 300 歩未満だったものである | **現役** | [plan/16 の 3 節](../../docs/plan/16_finals_initiative_priority.md) に 12 タスクと 6 条件の作り方がある |
| **`TRACK2_BASE12.csv`** | 12 | **`TRACK2_VAL72.csv` の摂動なし 12 行だけ。**摂動を当てる前に「素のタスクで壊れていないか」を 12 本で早く見るために使う（[knowledge/20](../../docs/knowledge/20_probe_object_choice_without_rollout.md)） | **現役** | `TRACK2_VAL72.csv` から `category` が `Base` の行を抜く |
| **`COOKIE12.csv`** | 12 | **衝突（対象外の物を動かすこと）で落ちている行だけを集めた 12 行。**`TRACK2_VAL72.csv` のうちクッキー箱の 2 タスク × 6 条件で、推論の設定を 1 つ変えて衝突が減るかを見るのに使う。**比べ相手の数字が既にある**（3.4 節） | **残す**（設定を比べるとき） | `TRACK2_VAL72.csv` から該当タスクの行を抜く |
| **`TRACK2_ROBOT_LADDER60.csv`** | 60 | **ロボットの初期姿勢だけを段にして測る 60 行**（12 タスク × [摂動なし ＋ 運営の難易度 L2〜L5]）。「どのくらい姿勢がずれると崩れ始めるか」を答えた行で、**答えは出ている**（3.5 節） | **測り終えた**（作り直しの手はあるので残す） | `python3 solution/tools/probe/make_robot_initstate_ladder.py --output compe/val/TRACK2_ROBOT_LADDER60.csv` |
| **`TRACK2_PROBE.csv`** | 40 | **Track2 の 4 カテゴリを横並びにする 40 行**（8 タスク × [摂動なし ＋ カメラ・初期姿勢・物の配置・センサ雑音]）。1 カテゴリ 8 件しか無いので**順位を付ける道具であって本評価ではない。**加えて、**カメラの学習データを作るときに「評価で使う variant」として除く一覧**にもなっている | **測り終えた**（除外の一覧として現役） | `python3 solution/tools/probe/make_track2_probe.py --output compe/val/TRACK2_PROBE.csv` |
| **`MOKA_TASKS.csv`** / **`MOKA_BASE.csv`** | 8 / 1 | **`put both moka pots on the stove` だけを集めた行。**このタスクは手本が全部 341 歩以上あり `--max-demo-steps 300` で落ちるので、**36 行に 1 つも入っていない。**scripted policy で作った手本の効きを見るための入り口である（[howto/05](../../docs/howto/05_scripted_demos.md) の 5〜8 節） | **残す**（howto/05 が入力として指している。ただし SmolVLA 期の線で、π0.5 では回していない） | `python3 solution/tools/suite/make_task_suite.py --instruction "put both moka pots on the stove" --count 8 --output compe/val/MOKA_TASKS.csv` |
| **`HOLDOUT_TASKS.csv`** | 8 | **これだけ評価用ではない。****学習から外すタスクの一覧**で、`train.py --holdout-tasks` と検証集合づくりの両方がこの 1 枚を読む。列も別（`instruction,suite,category,source`） | **残す**（道具は動く。ただし**いまの 36 行はタスク単位ではなくエピソード単位で分けたので、これを使っていない**） | `python3 solution/tools/suite/make_holdout_split.py` |

> **ここに無いもの。**`VAL_TASKS.csv` という名前がスクリプトの中に出てくるが、これは
> `make_validation_suite.py` がその場で作る出力で、**git には入れていない。**
> 評価を回すと実行ごとのディレクトリへ `TASKS_CSV.csv` として控えが残るので、
> **後から「あの数字はどの行で出したのか」を引けるのはそちらである。**

---

## 3. 評価実行例

**ここは要約だけを置く。**設定・重み・読み方の詳しい話はリンク先が持つ。

### 3.1 `EVAL_TASKS.csv`（36 行）—— **並び順を当てられないことが分かっている**

**同じ 36 行・同じ設定・同じ重みで 5 回回すと、成功率は 0.694 / 0.694 / 0.694 / 0.778 / 0.667 と
11.1 ポイント動いた**（`work/gpu_eval/0824_125446_floor36_ep5_...`）。歩数の合計なら ±8.9% に収まる。
**そして手元で +3 件だった変更が、本番では 0.264 → 0.169 にも 0.264 → 0.485 にもなった。**
**この 36 行は良し悪しの順位を付けられない**（[knowledge/07](../../docs/knowledge/07_local_eval_cannot_rank.md)）。
本番と繋がっている量は成功率ではなく歩数である（[knowledge/06](../../docs/knowledge/06_step_count_drives_score.md)）。

### 3.2 `TRACK1_VAL29.csv`（29 行）—— **まだ 1 度も回していない**

9/10 に作ったところである。最初の用途は、**接触の多い学習データを掃除した重み
（`pi05_clean24k`）と掃除前（`pi05_full_local`）を Track1 で並べること**で、
29 行 × 3 エピソード = 87 本を RTX6000 で回す
（[reference/32](../../docs/reference/32_current_tasks.md) の `F` の行）。
**外した 7 行の選び方と、掃除で学習データが 0 本になった 1 行の注意は
[`TRACK1_VAL29.meta.json`](TRACK1_VAL29.meta.json) が持つ。**

### 3.3 `TRACK2_VAL72.csv`（72 行）—— **崩れるのはカメラを回したとき**

`work/final_eval/0828_002049_track2_val72base`（`pi05_rp3000`・3 エピソードずつ 216 本・
`score_composite.md`）。**成功率 0.7222・衝突率 0.2269・合計点 0.3674。**

| 条件 | 成功率 | 衝突率 |
|---|---|---|
| 摂動なし | 0.806 | 0.167 |
| ロボット初期姿勢 0.1 / 0.4 | 0.806 | 0.167 |
| ロボット初期姿勢 0.2 | 0.694 | 0.278 |
| **カメラ方位角 +45° / −60°** | **0.611** | 0.278 / 0.306 |

**失敗の 82% は衝突である**（[plan/16 の 3.1 節](../../docs/plan/16_finals_initiative_priority.md)）。
衝突の中身を 1 本ずつ見るために同じ 72 行を回し直したものが
`work/final_eval/0828_134500_track2_collision_probe` で、こちらは
`results/object_disp.jsonl` に 216 行（1 本 1 行）を持つ。

### 3.4 `COOKIE12.csv`（12 行）—— **推論の設定 3 つは全部落ちた**

8/28 に同じ 12 行 × 3 = 36 本で、基準（`EMA 0.3`・`blend 0`）と 3 通りを比べた。

| | 基準 | `blend_steps` 4 | `ema_alpha` 0.5 | `ema_alpha` 0（切る） |
|---|---|---|---|---|
| 合計点 | **0.2055** | 0.1288 | 0.2011 | 0.1436 |
| 成功率 | **0.389** | 0.250 | 0.389 | 0.278 |
| 衝突率 | **0.611** | 0.750 | 0.611 | 0.694 |

**3 つとも不採用。**とくに `blend_steps` 4 は 8/14・8/23 の 2 組では衝突率が下がって見えていて、
**衝突を減らしたい集合で測り直したら逆に出た**（[reference/19](../../docs/reference/19_weights_catalog.md) の
「版を上げずに手元で測った設定」）。

### 3.5 `TRACK2_ROBOT_LADDER60.csv`（60 行）—— **L3 から合計点が半分になる**

`work/final_eval/0828_151854_track2_robot_ladder`（180 本・3,804 秒）。

| 段 | 成功率 | 打ち切り率 | 全回込みの平均歩数 | 合計点 |
|---|---|---|---|---|
| 摂動なし | 0.833 | 0.028 | 126.7 | **0.431** |
| L2 | 0.778 | 0.056 | 137.1 | 0.393 |
| L3 | 0.500 | 0.333 | 195.6（**+54.4%**） | **0.238** |
| L4 | 0.394 | 0.333 | 204.7 | 0.191 |
| L5 | 0.462 | 0.436 | 213.7 | 0.224 |

**段より鋭く分かれたのは「手先が 2 cm 動くかどうか」だった**——
2 cm 未満は歩数 +6.2%（同じ設定で回し直したときのばらつき ±6.5% の内側）で、
2〜4 cm で +42.0%・4〜6 cm で +57.7%・6 cm 以上で +87.5% と増え続ける
（[reference/21](../../docs/reference/21_weakness_map.md) の 1・4 節）。

### 3.6 `TRACK2_PROBE.csv`（40 行）—— **成功率と歩数で順位が逆になる**

`work/gpu_eval/0822_092556_track2_probe_...`（`pi05_rp3000`・1 セル 1 本）。
**成功率ではカメラ視点は崩れて見えなかったが、歩数で見ると最も崩れていた。**

| カテゴリ | 成功率 | **摂動なしで完走した件の歩数比** |
|---|---|---|
| カメラ視点 | 88%（7/8） | **+34.6%** |
| センサ雑音 | 75%（6/8） | +24.4% |
| ロボット初期姿勢 | 88%（7/8） | +17.5% |
| 物の配置 | 88%（7/8） | +15.6% |

**1 カテゴリ 8 件なので強い結論ではない**（[reference/20 の 4 節](../../docs/reference/20_track2_perturbation_categories.md)）。

### 3.7 `TRACK2_BASE12.csv`（12 行）—— **逆向きデータを 2 割混ぜたら 0.9167 → 0.5833**

9/8 に `pi05_rp3000`・`pi05_all111/002000`・`pi05_proto20/002000` を 12 行で並べた
（[reference/19 の「9/8」の節](../../docs/reference/19_weights_catalog.md)）。
**0.9167 → 0.7500（逆向きデータ抜きでも下がる）→ 0.5833。**
**同じ日の 72 行でも 0.5833 が出た**ので、12 行という少なさによる見かけではない。

### 3.8 `MOKA_TASKS.csv`（8 行）—— **8 件とも 0**

`work/gpu_eval/0813_165734_weak3x_moka_...`（`weak3x`）と
`0813_174816_scripted_mix_moka_...`（scripted policy の手本を混ぜた重み）。
**どちらも 8 件すべて 0.0%。**混ぜた側の重みは残っていない
（[reference/19](../../docs/reference/19_weights_catalog.md) の `smolvla/scripted_mix_20000`）。

---

## 4. 消したもの（2026-09-10）

**どれも 8/14 の 1 回きりの比較のために `EVAL_TASKS.csv` から行を抜いたもので、
そこで出た結論は [reference/19](../../docs/reference/19_weights_catalog.md) に移してある。**
**古い文書がこれらの名前で指しているときは、下の「戻し方」で作り直せる。**

| 消したファイル | 何だったか | 戻し方 |
|---|---|---|
| `KNOB_TASKS.csv`（12 行） | **`EVAL_TASKS.csv` の先頭 12 行そのもの。**8/14 に推論の設定を 1 つずつ変えて比べた | `head -13 compe/val/EVAL_TASKS.csv > KNOB_TASKS.csv` |
| `EVAL_TASKS_FAST12.csv`（12 行） | 36 行から 12 行を抜いて速く回すためのもの | `git show 32641f6:compe/val/EVAL_TASKS_FAST12.csv` |
| `EVAL_TASKS_RESCUED5.csv`（5 行） | 36 行のうち、設定を変えて解けるようになった 5 行 | `git show 653398a:compe/val/EVAL_TASKS_RESCUED5.csv` |

**3 枚を指していた文書は、どちらも過去の記録である**——
[plan/archive/06](../../docs/plan/archive/06_eval_gap_measurement.md)（2026-08-22 に取り下げ済み）と、
その日ごとの作業記録（`docs/talk/` の下）である。
