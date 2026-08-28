# 001 衝突と判定されたとき、どの物体がどれだけ動いたかを数える

> **状態: 依頼中** ／ 受け側: 未定 ／ 更新: 2026-08-28

## 何をしてほしいか

**評価のロールアウト 1 回ごとに「操作対象でない物体が、それぞれ何 mm 動いたか」を
記録する仕掛けを足し、1 本回して分布を出してほしい。**
成果物は **(1) 記録する仕掛けのコード** と **(2) 変位の分布を並べた表**である。

## なぜ要るか

**いま手元の失敗の 82% が衝突である。**次に何をするかが、この分布で分かれる。

| 分布がこうなら | 次にやること |
|---|---|
| **1〜3 mm に固まっている**（かすった） | 動きを滑らかにする・接触の手前で減速する、といった**推論側の後処理**に意味がある |
| **数 cm に散らばっている**（倒した・押しのけた） | **後処理では届かない。**学習でしか直らないと決まる |

**この判定なしに衝突対策へ着手すると、外れたときに何も残らない。**

## 前提となる事実

**判定の場所は `pipeline/rollout.py` である**（会社の RTX6000 と、このリポジトリの
`work/final_eval` から回すときに使う本選キット `/home/ubuntu/lab/b_PARC2026_final`）。

- **`rollout.py:233-234`**

  ```python
  collided = any(d > collision_threshold for d in object_max_disp.values())
  success = bool(done) and not collided
  ```

  **つまり「ゴールに着いた」だけでは成功にならない。**

- **しきい値は 1 mm。** `pipeline/total_score.py` の内蔵既定が
  `"collision": {"enabled": True, "threshold_m": 0.001}`。

- **監視される物体**は `rollout.py:170-178` で決まる。観測 `obs` のキーのうち
  `_pos` で終わり、`robot0` で始まらず、`_to_robot0_eef_pos` で終わらず、
  **`obj_of_interest`（そのタスクの操作対象）に入っていないもの。**
  初期位置は、ダミー行動を 10 ステップ流したあとの値を基準にしている。

- **`object_max_disp` は既に計算されている。** 各物体の「初期位置からの最大変位」である。
  **いまはしきい値と比べた真偽だけが使われ、値そのものは捨てられている。**
  ここを書き出すのが依頼の中心である。

**8/28 に測った衝突率**（`work/final_eval/0828_002049_track2_val72base`・72 タスク × 3 本 = 216 本）。

| 条件 | 成功率 | 衝突率 |
|---|---|---|
| Base（摂動なし） | 0.806 | **0.167** |
| Camera 方位角 +45° | 0.611 | 0.278 |
| Camera 方位角 −60° | 0.611 | **0.306** |
| Robot 摂動量 0.1 / 0.2 / 0.4 | 0.806 / 0.694 / 0.806 | 0.167 / 0.278 / 0.167 |
| **全体** | **0.722** | **0.227** |

## やること

1. **`solution/tools/eval/kit_patches/` に `rollout.py` を置く。**
   `/home/ubuntu/lab/b_PARC2026_final/pipeline/rollout.py` をそのまま写し、
   **`object_max_disp` を書き出す行だけを足す。**
   同じ場所の `scorer.py` が同じやり方（キットの中を直さず、こちらに置いてマウントする）で
   既に動いているので、**それに倣うこと。**

2. **書き出す形は 1 行 1 ロールアウトの JSONL** にする。置き場所は実行ディレクトリの
   `object_disp.jsonl`。1 行の中身：

   ```json
   {"task": "<タスク名>", "episode": 0, "success": false, "collided": true,
    "obj_of_interest": ["alphabet_soup_1"],
    "max_disp_m": {"wooden_tray_1": 0.0043, "ketchup_1": 0.0001}}
   ```

3. **`solution/tools/eval/run_final_eval.sh` にマウントを足す。**
   173 行目あたりの `SCORER_PATCH` の直後に、同じ形で `ROLLOUT_PATCH` を足し、
   `docker run` の引数へ `"${rollout_mount[@]}"` を差し込む。
   **書き出し先はコンテナの中なので、`$OUTPUT_DIR` を見せているマウントの下に置くこと。**

4. **ユーザに 1 本回してもらう。**打つコマンドをそのまま貼れる形で用意すること。
   **回すのは会社の RTX6000 で、`docker run` はこちらのマシンからは通らない。**
   集合は 72 行の `compe/val/TRACK2_VAL72.csv` を使う（下のコマンドの形）。

   ```
   TASKS_CSV=compe/val/TRACK2_VAL72.csv TRACK_EPISODES=3 MUJOCO_GL=egl \
     PARC_N_ACTION_STEPS=16 PARC_EMA_ALPHA=0.3 LABEL=collision_probe \
     solution/tools/eval/run_final_eval.sh track2
   ```

5. **分布を出す。**

## できあがりの条件

- `object_disp.jsonl` が 216 行あり、うち衝突した回の数が
  `results/server_8000.json` の衝突率 × 216 と一致する
- 下の 3 つの表が書けている

## 報告の書き方

**表と数値だけでよい。長い考察は要らない。**

1. **衝突した回の最大変位の分布。** 区切りは
   `1〜2 mm / 2〜5 mm / 5〜10 mm / 10〜50 mm / 50 mm 以上` の 5 段。件数と割合。
2. **どの物体が犯人か。** 物体名ごとの件数、上位 10 個。
3. **条件ごと**（Base / Camera / Robot）に 1 の分布がどう変わるか。

**最後に 1 行だけ、「かすった」寄りか「倒した」寄りかの判定を書く。**

## やらなくてよいこと

- **衝突を減らす対策そのもの。**この依頼は数えるところまでである。
- 採点式や合計点の話（別で `solution/tools/eval/score_composite.py` として作ってある）。
- `NOW.md` の書き換え。

## 報告

（受け側がここに書く）
