# 17. 採点の「総合スコア」は成功率だけではない

**結論から。手元の評価で出る総合スコアは成功率そのものだが、
採点環境（リーダーボード・最終評価）の総合スコアは、成功率に
滑らかさや衝突を重み付きで足したものである。**
**「ジャークは採点されない」は間違いだった。**

以下、項目ごとに出典と、再確認のしかたを付ける。

---

## 0. 式そのものは公開されている（2026-08-28 に追記）

**運営の資料に式が載っている。**重みと正規化の中身だけが非公開である。

```
smooth metrics = w1*time + w2*jerk + w3*SPARC + w4*trajectory + w5*rotation

Total Score = (1/N) * Σ success_i * (1 - collision penalty_i) * smooth metrics_i
```

| 公開されていること | 中身 |
|---|---|
| **成功は掛け算の門である** | **失敗した回の点は 0 になる。**なめらかさがどれだけ良くても拾えない |
| **collision penalty は [0,1] の 2 値** | 衝突した回も同じく 0 になる |
| **重み付きの和に入るのは 5 つだけ** | time・jerk・SPARC・trajectory・rotation。**成功率は入っていない** |
| 重みの合計 | `w1 + ... + w5 = 1`。「特定の指標に偏らせないための重み付け」と説明されている |
| 正規化 | 各指標を 0〜1 の連続値へ写す。**式は非公開** |

**目盛りの手がかりが 1 つある。予選では全 rollout が成功しても点は 0.48 だった。**
つまり `smooth metrics` の実力値がその辺りに来る正規化になっている。

**手元でこの形の点を出す道具は `solution/tools/eval/score_composite.py` である。**
重みは 5 等分、正規化は `基準値 /（基準値 + 実測）`。
**この目盛りで、8/28 の 72 行は「全部成功したら 0.4988」になり、0.48 とよく合う。**

**点は評価の実行ディレクトリに残る。**`run_final_eval.sh` が評価のあと自動でこの道具を
呼び、`<実行ディレクトリ>/score_composite.md` に**点と計算式を一緒に**書く。
採点側の式と違えている 3 点（タスクごとに平均してから掛ける・`(1 - collision)` を
掛けない・全部落ちたタスクの歩数に 300 を当てる）もそのファイルに書き出される。
**あとから「この点はどの式で出したのか」を引くのに、コードを読む必要はない。**

以下の 1〜4 節は、**式が公開される前に採点ログから読み取った経緯**である。
結論は変わっていない。



---

## 1. 配布ハーネスの既定は「成功率の平均」だけ

| 見たもの | 内容 |
|---|---|
| `pipeline/rollout.py:52-56` | `success_rate` = そのタスクで成功したエピソード数 ÷ 本数。**1 本ごとは成功か失敗の 2 値**（部分点は無い） |
| `pipeline/scorer.py:102` | `overall_success = np.mean([ts.success_rate for ts in task_scores])` |
| `pipeline/scorer.py:130` | `overall_score = float(overall_success)` |
| `pipeline/scorer.py:105-127` | ジャーク・衝突率・SPARC・経路長は `overall_metrics` に**報告されるだけ**で、`overall_score` には入らない |

```bash
sed -n '95,145p' pipeline/scorer.py      # 上の 3 行を確認する
sed -n '47,64p'  pipeline/rollout.py
```

**だから手元の 36 件の 0.667 や 0.694 は、ちょうど 24/36・25/36 である**
（`work/gpu_eval/*/summary.txt` の `mean_success_rate` と一致する）。

## 2. ただし「重みを足す道」が既定で切られているだけである

`pipeline/total_score.py:14-20` の内蔵既定がこれである。

```python
_BUILTIN_TOTAL_SCORE_DEFAULT = {
    "control_dt": 0.05,
    "use_total_score_as_overall": False,
    "weights": {},
    "collision": {"enabled": True, "threshold_m": 0.001},
}
```

**`use_total_score_as_overall` と `weights` という名前の設定が用意されている。**
配布された `scorer.py` はこの 2 つを一度も読まないが、
**設定を差し替えれば「重み付きの合計点を総合スコアにする」道が開く**ことを
名前が示している。

`load_scoring_config()` は `total_score_config.json` と
`normalization_config.json` を探し、**無ければ内蔵既定に落ちる**
（`pipeline/total_score.py:61-62` が警告を出す）。**この 2 つは配布物に入っていない。**

```bash
ls total_score_config.json normalization_config.json   # どちらも無い
```

## 3. 採点環境はその設定ファイルを「秘匿ファイル」として持ち込む

`evaluate.py:33-41`（採点側の入口）にこう並んでいる。

```python
_EVAL_SECRET_PATHS = [
    "/workspace/LIBERO-plus",
    "/workspace/compe",
    "/workspace/pipeline",
    "/workspace/scoring_config.json",
    "/workspace/total_score_config.json",
    "/workspace/normalization_config.json",
]
```

```bash
sed -n '33,41p' evaluate.py
grep -n "public_score\|overall_score" evaluate.py
```

**採点環境では中身の違う設定ファイルが置かれる**、というのがこの一覧の意味である。
そして `evaluate.py:202-204` で `public_score` は各トラックの
`overall_score` の平均になる ——**つまり秘匿設定が `overall_score` の中身を決める。**

## 4. 本番のスコアの値そのものが、成功率だけでないことを示している

**採点ログで数えたエピソードは 8 本である**
（`outputs/260814_114832_pi0.5_bench.md` の `/reset` が 8 回）。

```bash
grep -c "POST /reset" outputs/260814_114832_pi0.5_bench.md
```

**成功率の平均だけなら、8 本のときに出る値は必ず 0.125 刻みになる**
（1 タスク 1 本なら k/8、4 タスク × 2 本でも同じ集合）。
ところが返ってきた値は 10 回すべて 0.125 の倍数でない。

| 版 | 本番スコア |
|---|---|
| 0.1.0 | **0.05832** |
| 0.2.0 | 0.096 |
| 0.3.0 | 0.1409 |
| 2.0.0 | **0.09507** |
| 3.0.0 | 0.144 |
| 3.1.0 | 0.166 |
| 3.3.0 | 0.080 |
| 3.4.0 | 0.168 |
| 3.5.0 | 0.181 |
| 3.6.0 | **0.184** |
| 3.8.0 | **0.177** |

**とくに 0.05832 と 0.09507 は、小さい分数では作れない。**
**成功率のほかに連続値の項が入っている、と読むしかない。**

（出典はスコアの履歴表 [../howto/02_build_submission.md](../howto/02_build_submission.md)
だが、**元の値は毎回の採点ログである**。ログは `outputs/` に残っている。）

## 5. これが判断に効くところ

| 効くこと | 中身 |
|---|---|
| **ジャークは点になる** | 「滑らかさは動いたが点は動かない」という読みは撤回する。`n_action_steps` を上げてジャークが −14% 下がったことは、**本番では点として現れうる** |
| **0.184 と 0.177 の 0.007 差は、成功本数の差とは限らない** | 滑らかさの差でも同じだけ動く。**手元の 36 件（成功率だけ）では、この差を再現して測れない** |
| **手元では composite を測れない** | 重みが秘匿なので、**測る方法はリーダーボードに出すことだけ**である |
| 衝突は 1mm で失格のまま | `collision.threshold_m` は内蔵既定でも 0.001（`pipeline/total_score.py:19`） |

**残っている不明点：重みの内訳（成功率が何割か）は分からない。**
0.1.0 が 0 本成功だったとすれば、滑らかさ側だけで 0.058 出ることになるが、
**1 本成功していた場合と区別できない**（1/8 = 0.125 に重み 0.47 以下が掛かっても
0.058 になる）。**分解するには、成功本数が同じで滑らかさだけ違う 2 回の提出が要る。**
