# MolmoAct2-LIBERO LoRA（24,000 ステップ）

*English: [README.md](README.md)*

[`allenai/MolmoAct2-LIBERO-LeRobot`](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot)
の上に載せた LoRA。学習データは
[`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)。

**測定があるのはこちらである。**40,000 ステップまで続けたものは
[`siruku6/molmoact2_lora_v1_40k`](https://huggingface.co/siruku6/molmoact2_lora_v1_40k) にあるが、
**一度も評価していない。数字が要るならこちらを使うこと。**

## これは何か

| | |
|---|---|
| 土台 | `allenai/MolmoAct2-LIBERO-LeRobot`（MolmoAct2 5B・Apache 2.0） |
| 学習の仕方 | VLM 側にだけ LoRA（`enable_lora_vlm: true`。action expert は触っていない） |
| rank / alpha / dropout | 64 / 16 / 0.05 |
| ステップ | **24,000** |
| batch | 4 |
| 最適化 | adamw・lr 1e-5・weight_decay 0.0・betas (0.9, 0.95)・eps 1e-6・grad_clip_norm 1.0 |
| スケジューラ | cosine_decay_with_warmup・warmup 200・peak 1e-5 → 1e-6 |
| 種 | 1000 |
| 画像を崩す細工 | **あり**（明るさ・コントラスト・彩度・色相・鮮鋭度・回転 ±5° から最大 3 つ） |
| 枠組み | LeRobot v0.6.0 |

**重みはマージせず、PEFT が付ける名前のまま保存してある。**読み込むには `peft` と
`enable_lora_vlm: true` が要る。

## 評価

摂動を加えた LIBERO の 84 行 × 5 本 ＝ 420 本・EGL 描画・`samples=1`・再計画の切り替えなし。
方策の衝突率からは、どの方策にも避けられない場面側の落下物を除いてある。

| 模型 | 成功率 | ゴール到達率 | 方策の衝突率 | 滑らかさ |
|---|---|---|---|---|
| π0.5 `bowl5_full`（別系統・全パラメータ学習） | **0.798** | **0.881** | 0.138 | **0.503** |
| **この模型（24,000）** | 0.774 | 0.833 | 0.138 | 0.494 |
| 16,000 ステップ・細工なし・batch 2 | 0.695 | 0.752 | 0.150 | 0.482 |

**摂動の種類ごとに割ると**（24,000 と π0.5 の対照）、

| 行数 | 中身 | π0.5 `bowl5_full` | この模型 |
|---|---|---|---|
| 12 | 素（摂動なし） | 0.917 | **0.917** |
| 60 | センサ雑音 | **0.790** | 0.763 |
| 12 | ロボット初期姿勢 | 0.600 | 0.600 |

正直な読み。

- **π0.5 の対照と同等で、上ではない。**種類ごとの差はどれも、実測した実行ごとの揺れ（約 0.05）の内側である。
- **16,000 → 24,000 の伸びは 1 変数の比較ではない。**細工の off→on、batch の 2→4、
  ステップの 16k→24k が同時に変わっている。
- **この評価セットには、カメラ視点・背景・照明の行が 1 行も無い。**
  **見た目への頑健さについて何も言えない**——そしてそれは、コンペの課題分布の半分を占めていたものである。
- **学習していない素の土台のほうが、コンペの採点では上だった**（2 つのトラックで 0.530 と 0.424）。
  **私たちの LoRA は、どのチェックポイントも有効な採点を 1 度も得ていない。**

## 使い方と、通信の無い環境での落とし穴

[`siruku6/molmoact2_lora_v1_40k`](https://huggingface.co/siruku6/molmoact2_lora_v1_40k) を見てほしい
——読み込みのコードは同じで、そちらに失敗の形を詳しく書いてある。要点だけ言うと、
**設定が `allenai/MolmoAct2-FAST-Tokenizer` を Hub のリポジトリ名として持っているので、
通信の無い環境では方策が起動しない。****そのトークナイザの写しを `fast_action_tokenizer/`
として同梱してある**ので、`discrete_action_tokenizer` にそこを指させること。

もう 1 つ。**この模型には adaptive resampling が効かない。**
`samples=3` ＋ `PARC_ADAPTIVE_CADENCE` で 13 課題の部分集合で 0.846、
`samples=1` で 0.877 だった。**π0.5 には効く。**
**設定を模型の系統をまたいで持ち込まないこと。**

## ライセンスと出自

| 何 | 権利者 | ライセンス |
|---|---|---|
| 土台の重み `allenai/MolmoAct2-LIBERO-LeRobot` | Allen Institute for AI | Apache 2.0 |
| LoRA（このリポジトリ） | 作者 | Apache 2.0 |
| LeRobot | The HuggingFace team | Apache 2.0 |
| LIBERO / LIBERO-plus の実演データ | Lifelong Robot Learning ほか | MIT |

## 関連

- 40,000 ステップまで続けたもの: [`siruku6/molmoact2_lora_v1_40k`](https://huggingface.co/siruku6/molmoact2_lora_v1_40k)
- データ: [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)
