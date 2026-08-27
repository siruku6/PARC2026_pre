# 15. 今回のπ0.5学習の学習率は、研究側の値とどう違うか

**【退避・2026-08-27】学習率の正本は [../11_training_schedule.md](../11_training_schedule.md) である。**この文書は「今回の値が研究側の既定と一致しているか」を確かめた当時の記録である。

**結論を先に。学習率の高さ(2.5e-5)と下げ方(cosineで10分の1まで)は、
lerobotのπ0.5既定値・Physical Intelligence自身のπ0既定値と完全に一致している。**
SmolVLA側で踏んだ「出典のない組み合わせを作ってしまう」失敗
([11_training_schedule.md](../11_training_schedule.md))は、今回は起きていない。

**ただし、Physical Intelligenceが公開しているLIBERO向けの設定
(`pi05_libero`)とは形が違う。** あちらは「バッチ256で、学習率を下げずに
ほぼ一定に保つ」recipe である。違いは下の3節にまとめた。

> **この文書が出典に使うもの。**
>
> | 出典 | 何が分かるか |
> |---|---|
> | `lerobot` v0.6.0 の [`policies/pi05/configuration_pi05.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/policies/pi05/configuration_pi05.py) | 今回の学習が使っている既定値 |
> | `lerobot` v0.6.0 の [`optim/schedulers.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/optim/schedulers.py) | 下げ方の計算式そのもの |
> | `lerobot` v0.6.0 の [`policies/smolvla/configuration_smolvla.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/policies/smolvla/configuration_smolvla.py) | SmolVLA側の既定値(SmolVLAの著者自身のコード) |
> | `openpi` の [`training/config.py`](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/config.py) と [`training/optimizer.py`](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/optimizer.py) | **π0.5を作った研究チーム自身の設定**(論文よりこちらが実物) |
> | `lerobot/pi05_libero_base` の [`config.json`](https://huggingface.co/lerobot/pi05_libero_base/raw/main/config.json) | 出発点の重みに同梱されている値 |
> | 実行中コンテナの学習ログ(`docker logs`)に出る`lr:` | **実際にこのマシンで動いている値** |
>
> どれも2026-08-13に確認した。SmolVLAの論文(arXiv 2506.01844)の本文は読めていない
> ため、SmolVLA側は著者のコードの既定値と重みのconfigだけを根拠にしている。

## 1. 今回の学習が実際にやっていること(ログで裏を取った)

渡しているのは学習率の高さ(`--policy.optimizer_lr=2.5e-5`)だけで、
**下げ方は`train_pi05.py`から一切指定していない**ので、lerobotの既定値が使われる。
その既定値は、出発点の重み`lerobot/pi05_libero_base`の`config.json`と同じ値である
(`optimizer_lr=2.5e-5`、`scheduler_warmup_steps=1000`、
`scheduler_decay_steps=30000`、`scheduler_decay_lr=2.5e-6`)。

**ステップ数を24,000にしたので、下げ方は自動で24,000 ステップに縮められている。**
lerobot v0.6.0の`CosineDecayWithWarmupSchedulerConfig.build()`が、
`num_training_steps < num_decay_steps`のときに立ち上げと下げ幅を
`num_training_steps / num_decay_steps`倍する作りになっているためである
(24000/30000 = 0.8倍 → 立ち上げは1,000 ステップではなく**800 ステップ**、
下げきるのは30,000 ステップではなく**24,000 ステップ**)。

| 歩 | 学習率 |
|---|---|
| 0 | 3.1e-8(ほぼ0から立ち上げる) |
| 800(立ち上げ終わり) | **2.49e-5(ここが頂点)** |
| 6,000 | 2.17e-5 |
| 12,000(折り返し) | 1.38e-5 |
| 18,000 | 5.8e-6 |
| 24,000(終わり) | **2.5e-6** |

**この表は計算だが、計算式が合っていることは実行中のログで確認した。**
ログの`lr:`は20 ステップぶんの平均なので、同じように平均して比べる。

| ログの歩 | ログの`lr` | 自動調整あり(立ち上げ800 ステップ)の計算 | 自動調整なし(1,000 ステップ)の計算 |
|---|---|---|---|
| 60 | 1.6e-6 | **1.61e-6** | 1.29e-6 |
| 80 | 2.2e-6 | **2.23e-6** | 1.79e-6 |
| 100 | 2.9e-6 | **2.86e-6** | 2.29e-6 |
| 200 | 6.0e-6 | **5.98e-6** | 4.78e-6 |

4点すべて自動調整ありの方に一致する。**つまり「24,000 ステップで学習率が
下がりきる」形になっており、途中で切れなければ最後まで下がる。**

## 2. 3つの設定を並べる

| | **今回のπ0.5** | lerobotのπ0.5既定<br>(= openpiのπ0既定) | openpi公式の`pi05_libero` | SmolVLA(lerobot既定) | 過去のSmolVLA追加学習<br>(本番0.016) |
|---|---|---|---|---|---|
| 頂点の学習率 | **2.5e-5** | 2.5e-5 | 5e-5 | 1e-4 | 3e-4 |
| 立ち上げ(warmup) | **800 ステップ** | 1,000 ステップ | **10,000 ステップ** | 1,000 ステップ | 100 ステップ |
| 最後の学習率 | **2.5e-6** | 2.5e-6 | **5e-5(下げない)** | 2.5e-6 | 3e-5 |
| 下げきるステップ数 | **24,000** | 30,000 | 1,000,000(30,000 ステップでは効かない) | 30,000 | 20,000 |
| 下げ方の形 | cosineで10分の1へ | 同じ | **立ち上げ後ほぼ一定** | cosineで40分の1へ | cosine |
| バッチ | **16** | 32 | **256** | 32 | 記録なし |
| 重みの平均(EMA) | **なし** | 0.99(π0のLIBERO設定) | **0.999** | なし | なし |
| 学ぶ範囲 | **action expertだけ**(VLM固定) | 全部 | 全部 | 全部 | action expertのLoRA |
| AdamW | betas (0.9,0.95)・weight_decay 0.01・勾配打ち切り 1.0 | 同じ | b2 0.95・weight_decay 1e-10・打ち切り 1.0 | betas (0.9,0.95)・weight_decay 1e-10・打ち切り 10 | 同じ |

## 3. 研究側(openpiの`pi05_libero`)と形が違う点

**1. あちらは学習率を下げない。** `warmup_steps=10_000, peak_lr=5e-5,
decay_steps=1_000_000, decay_lr=5e-5`——頂点と最後が同じ値なので、
30,000 ステップの学習の間は「10,000 ステップかけて5e-5まで上げ、あとはずっと5e-5」に
なる。今回のcosineで10分の1まで下げる形とは別物である。

**2. バッチが16倍違う(256 対 16)。** 学習率はバッチと一緒に動かすのが
経験則なので、単純に比べられない。バッチに比例させる考え方なら
`5e-5 × 16/256 ≈ 3e-6`、平方根で合わせる考え方なら
`5e-5 × √(16/256) ≈ 1.3e-5`。**今の2.5e-5は、その両方より高い側にある。**
ただしlerobotの既定2.5e-5はバッチ32を前提にした値なので、
「バッチ16で2.5e-5」が大外れというほどでもない。

**3. あちらは重みの平均(EMA 0.999)を使っている。** 学習の揺れを均す仕組みで、
lerobotのπ0.5にはこれに当たる設定が無い。**下げ方をcosineにしていることが、
EMAが無いぶんの揺れを終盤で抑える役目を部分的に担っている**とも読める。

**4. あちらはVLMまで含めて全部学習する。** 今回はメモリの都合で
`train_expert_only=true`(action expertだけ)なので、**同じ学習率でも
動くパラメータの数が違う**(4.14Bのうち約693M)。固定して学ぶ範囲が
狭いほうが壊れにくい方向なので、学習率を下げる強い理由にはならない。

## 4. SmolVLAで踏んだ失敗との違い

[11_training_schedule.md](../11_training_schedule.md)に記録した失敗は、
**「配布ノートブックの学習率(3e-4、3,000 ステップ×バッチ1向け)を、20,000 ステップの
学習にそのまま持ち込んだ」**というものだった。本番スコアが0.1409から
0.016に落ちている。

今回はそれと違い、**学習率と下げ方を一切いじっていない**
(出発点の重みのconfigと同じ値をそのまま使い、ステップ数だけ24,000にした)。
ステップ数を変えたぶんは、1節のとおりlerobotが自動で下げ方に反映している。

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| 学習率を`pi05_libero`寄り(頂点5e-5・下げない・立ち上げ長め)にした方がスコアで勝るか | 同じステップ数で両方回して比べる以外に方法が無い。今回の残り時間では1本しか回せない |
| バッチ16に対して2.5e-5が高すぎないか | 学習曲線(`loss`)が終盤で暴れるかどうかを見る。暴れるなら次の試行で下げる |
| SmolVLAの論文本文が報告している学習率 | arXiv 2506.01844 のPDFを`pdf-reading`スキルで読む(この文書ではコードの既定値で代用した) |
