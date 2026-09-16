# 38. 学習率の系譜 —— π0.5 と MolmoAct2

TOC:
- [0. このファイルの役割](#0-このファイルの役割)
- [1. 結論](#1-結論)
- [2. π0.5 の系譜と学習率](#2-π05-の系譜と学習率)
- [3. MolmoAct2 の系譜と学習率](#3-molmoact2-の系譜と学習率)
- [4. 分割して学習するときの学習率](#4-分割して学習するときの学習率)
- [5. 出典](#5-出典)

## 0. このファイルの役割

**学習を何回かに分けて回すとき、学習率を毎回どうするかを決めるための一次情報を置く。**
「上流は各段階でいくつを使っていたか」「減衰させていたか」を、土台ごとに引けるようにする。

**このファイルが持たないもの。**こちらが実際に回した走りの学習率と歩数は
[19_weights_catalog.md](19_weights_catalog.md)、SmolVLA で踏んだ失敗の経緯は
[11_training_schedule.md](11_training_schedule.md)、当時の確認記録は
[archive/15_pi05_lr_schedule.md](archive/15_pi05_lr_schedule.md) にある。
**ここは上流（Physical Intelligence と Allen AI）が何をしていたかだけを持つ。**

## 1. 結論

1. **π0.5 を作った本人たちの微調整レシピは、warmup のあと学習率を動かさない**
   （`decay_lr` を `peak_lr` と同じ値にしてある）。LIBERO も DROID も同じ形である。
2. **MolmoAct2 は頂点の 10% まで cosine で落とすが、頂点の値は段階が変わっても据え置く**
   （事前学習・後段学習・各ロボットへの微調整・LIBERO・LeRobot 版まで同じ）。
3. **どちらも「段階をまたいでも頂点の学習率は変えない」。**分割学習で毎回同じ学習率を使うことは、
   上流の慣行から外れない。

## 2. π0.5 の系譜と学習率

### 2.1 段階の一覧

| # | 段階 | 学習率 | 規模 | 分かり方 |
|---|---|---|---|---|
| 1 | PaliGemma 3B（SigLIP ＋ Gemma 2B） | — | — | 上流の VLM。π0 系の土台 |
| 2 | π0 / π0.5 の事前学習 → `pi05_base` | **公開されていない** | 公開されていない | π0.5 論文（arXiv:2504.16054）・π0 論文（arXiv:2410.24164）とも本文に "learning rate" の記述が **0 件**。openpi にも事前学習の設定は無い |
| 3 | **`pi05_libero`**（LIBERO 微調整） | **5e-5（warmup 後は一定）** | batch 256・30,000 歩・全パラメータ・EMA 0.999 | openpi の `TrainConfig` |
| 4 | `lerobot/pi05_libero_base` への変換 | 学習していない | — | モデルカードが openpi を参照実装と明記 |

### 2.2 `pi05_libero` の設定（減衰しない理由）

```python
batch_size=256,
lr_schedule=CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5,
                                decay_steps=1_000_000, decay_lr=5e-5),
optimizer=AdamW(clip_gradient_norm=1.0),   # b1 0.9 / b2 0.95 / weight_decay 1e-10
ema_decay=0.999,
weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
num_train_steps=30_000,
```

- **`decay_lr` が `peak_lr` と同じ**なので、cosine の底が頂点と一致する＝減衰しない。
- さらに `decay_steps` が学習の 33 倍あるので、二重に減衰が効かない。
- **DROID 側も同じ形**である——`pi05_droid` と `pi05_full_droid_finetune` は
  warmup 1,000・peak 5e-5・`decay_lr` 5e-5・100,000 歩・batch 256。
- 凍結は無い（LoRA 用の設定だけが `freeze_filter` を持つ）。**全パラメータ学習である。**

### 2.3 `lerobot/pi05_libero_base` の `config.json` を学習の記録と読まない

同梱の `optimizer_lr=2.5e-5`・`scheduler_warmup_steps=1000`・`scheduler_decay_steps=30000`・
`scheduler_decay_lr=2.5e-6` は、**LeRobot 側の学習の既定値**であって、
**この重みを作ったときの値ではない**（作ったのは 2.2 節の openpi 設定）。
こちらの追加学習は何も渡さなければこの既定値で回るので、**上流の LIBERO 微調整とは別物の形になる。**

## 3. MolmoAct2 の系譜と学習率

| # | 段階 | 本体 | ViT | つなぎ | action expert | 規模 |
|---|---|---|---|---|---|---|
| 1 | Molmo2（Qwen3-4B）の mid-training | — | — | — | — | 外部の土台 |
| 2 | Molmo2-ER 段階 1（身体性への特化） | 2e-5 | 6e-6 | 2e-4 | — | 20K 歩・実効 batch 64。**論文に明記が無く、公式起動スクリプトの既定値**（`pretrain.py`） |
| 3 | Molmo2-ER 段階 2（混ぜ戻し） | 「その他は Molmo2 に従う」 | — | — | — | +1.5K 歩 |
| 4 | MolmoAct2 事前学習（離散 VLA） | **1e-5** | **5e-6** | **5e-6** | — | 実効 batch 128・64×H100・約 5,760 GPU 時間 |
| 5 | MolmoAct2 後段学習 → `allenai/MolmoAct2` | **1e-5** | **5e-6** | **5e-6** | **5e-5** | 100K 更新・実効 batch 128・約 2,300 GPU 時間 |
| 6 | `MolmoAct2-LIBERO` | 1e-5 | 5e-6 | 5e-6 | 5e-5 | 50K 更新・実効 batch 64・32×H100（採用は 40K 時点） |
| 7 | **`MolmoAct2-LIBERO-LeRobot`** | 1e-5 | 5e-6 | 5e-6 | 5e-5 | 10,000 歩・実効 batch 256（32×8 枚） |

- **4 段目以降は値が動いていない。**段階が変わっても頂点は据え置く設計である。
- 減衰は cosine で**頂点の 10%**（公式 olmo 側 `SchedulerConfig(alpha_f=0.1)`・warmup は各群 200 歩）。
  LeRobot 版の設定は warmup 200・`scheduler_decay_lr` 1e-6。
- **公式の起動スクリプトの既定値では action expert だけ 1e-4** である（README の全パラメータ微調整の例は 5e-5）。
  つまり action expert は **5e-5〜1e-4 が公式の範囲**。LoRA のときだけ本体・ViT・つなぎを 5e-5 まで上げている。

## 4. 分割して学習するときの学習率

### 4.1 なぜ問題になるか

**LeRobot の `--init-from`（重みだけ引き継ぐ口）は学習の状態を読まない。**
だから新しい実行は毎回 0 歩目から始まり、**warmup が引き直される。**
減衰する設定のままだと、2 回目以降は「上げて・下げて」を繰り返すので、
**歩数を足しても学習量が足し算にならない**（実測と換算は [19 の 2.1.1 節](19_weights_catalog.md)）。

### 4.2 一定に保つには（LeRobot の実装）

`lerobot/optim/schedulers.py` の cosine は次の形である。

```python
alpha = decay_lr / peak_lr
decayed = (1 - alpha) * cosine_decay + alpha
```

| やりたいこと | 渡すもの | 結果 |
|---|---|---|
| **warmup 後に一定**（openpi の LIBERO レシピの再現） | `--policy.scheduler_decay_lr` を `--policy.optimizer_lr` と同じ値にする | alpha ＝ 1 になり、頂点のまま平らになる |
| 減衰させる（LeRobot 既定） | 何も渡さない | 2.5e-5 → 2.5e-6 |

**`decay_steps` を大きくする手は効かない。**LeRobot は `num_training_steps < num_decay_steps` のとき、
warmup と減衰を実際の歩数に**自動で縮める**（`build()` の auto-scale）。1,000,000 を渡しても、
学習長にぴったり合った cosine になるだけである。

### 4.3 どちらを選ぶか

- **分割して回すなら一定が扱いやすい。**上流の慣行に合い、累積歩数をそのまま学習量として読める。
- **1 本で完結させるなら減衰でよい。**こちらの既存の走り（`pi05_bowl5_full`・`pi05_combined_initpose_full`）は
  減衰ありなので、**それらと 1 変数で比べたいときは減衰のまま揃える。**
- **混ぜて比べない。**同じ比較の中で片方だけ一定にすると、データの差と学習率の形の差が重なる。

## 5. 出典

| 出典 | 何が分かるか | 確認日 |
|---|---|---|
| openpi [`src/openpi/training/config.py`](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/config.py) | `pi05_libero`・`pi05_droid` の学習率・batch・歩数・EMA | 2026-09-16 |
| openpi [`src/openpi/training/optimizer.py`](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/optimizer.py) | `CosineDecaySchedule` と `AdamW` の既定値 | 2026-09-16 |
| π0.5 論文 arXiv:2504.16054 / π0 論文 arXiv:2410.24164 | **学習率の記述が無い**ことの確認（全文検索） | 2026-09-16 |
| `lerobot/pi05_libero_base` の `config.json` とモデルカード | 同梱の値は LeRobot の既定であり学習の記録ではないこと | 2026-09-16 |
| LeRobot v0.6.0 [`src/lerobot/optim/schedulers.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/optim/schedulers.py) | 一定に保つ条件（alpha ＝ 1）と auto-scale の挙動 | 2026-09-16 |
| `allenai/MolmoAct2-LIBERO-LeRobot` の `train_config.json` | 7 段目の学習率・warmup・batch・歩数 | 2026-09-16 |
| molmoact2 [`experiments/launch_scripts/train_lerobot.py`・`sft.py`・`pretrain.py`](https://github.com/allenai/molmoact2) | 2・4〜6 段目の学習率と scheduler（`alpha_f=0.1`） | 2026-09-16 |
| MolmoAct2 論文 arXiv:2605.02881（4.1〜4.3 節） | 4〜6 段目の学習率・実効 batch・GPU 時間 | 2026-09-16 |
