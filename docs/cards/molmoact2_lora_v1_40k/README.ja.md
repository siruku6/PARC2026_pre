# MolmoAct2-LIBERO LoRA（40,000 ステップ）

*English: [README.md](README.md)*

[`allenai/MolmoAct2-LIBERO-LeRobot`](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot)
の上に LoRA を載せて、**LIBERO** の操作課題で学習したものである。
学習に使った実演データは、**対象外の物体への接触**と**ロボット初期姿勢の分布の狭さ**という
2 つの失敗の原因を減らす目的で組み直してある。

ロボット操作のコンペ **PARC2026**（本選 1）のために学習した。

**このチェックポイントには成績が無い。**[評価](#評価)を読んでほしい——下の数字は
24,000 ステップの兄弟のものであり、**40,000 ステップのほうは締切までに 1 度も採点されていない。**
出発点として使えるものであって、確かめられた結果ではない。

## これは何か

| | |
|---|---|
| 土台 | `allenai/MolmoAct2-LIBERO-LeRobot`（MolmoAct2 5B・Apache 2.0） |
| 学習の仕方 | VLM 側にだけ LoRA（`enable_lora_vlm: true`。action expert は触っていない） |
| rank / alpha / dropout | 64 / 16 / 0.05 |
| ステップ | **40,000** |
| データ | [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)（21,642 本・111 課題） |
| 枠組み | LeRobot v0.6.0 |

**重みはマージせず、PEFT が付ける名前のまま保存してある**（`base_model.model.*`・`.base_layer.*`）。
読み込むには `peft` と、設定の `enable_lora_vlm: true` が要る。

## 学習

```
steps          40000
batch_size     4
optimizer      adamw, lr 1e-5, weight_decay 0.0, betas (0.9, 0.95), eps 1e-6
               grad_clip_norm 1.0
scheduler      cosine_decay_with_warmup
               warmup 200, decay 40000, peak_lr 1e-5, decay_lr 1e-6
seed           1000
chunk_size     10      n_action_steps 10
```

画像を崩す細工は**入れている**（`image_transforms.enable: true`。明るさ 0.8〜1.2・
コントラスト 0.8〜1.2・彩度 0.5〜1.5・色相 ±0.05・鮮鋭度 0.5〜1.5・
回転 ±5° と平行移動 ±5% から最大 3 つ）。**これは土台の重み自身が学習に使っていた細工と同じ向きである。**

8,000 ステップごとに保存した。このリポジトリには `040000/` が入っている。

## 評価

**このチェックポイントの成績は無い。**以下は 24,000 ステップの兄弟を、
摂動を加えた LIBERO の 84 行 × 5 本 ＝ 420 本で測ったものである
（EGL 描画・`samples=1`・再計画の切り替えなし）。「方策の衝突率」からは、
どの方策にも避けられない場面側の落下物を除いてある。

| 模型 | 成功率 | ゴール到達率 | 方策の衝突率 | 滑らかさ |
|---|---|---|---|---|
| π0.5 の対照（全パラメータ学習・別系統） | **0.798** | **0.881** | 0.138 | **0.503** |
| **この系統の 24,000 ステップ** | 0.774 | 0.833 | 0.138 | 0.494 |
| この系統の 16,000 ステップ（細工なし・batch 2） | 0.695 | 0.752 | 0.150 | 0.482 |

読むときの注意を 4 つ。

- **16,000 → 24,000 の伸びは本物だが、1 変数の比較ではない。**同時に 3 つ変えている——
  画像の細工が off → on、batch が 2 → 4、ステップが 16k → 24k。
  **だから「細工が効いた」とは、この表からは言えない。**
- **24,000 ステップは π0.5 の対照を上回っていない。**差 −0.024 は、私たちが測った
  実行ごとの揺れ（約 0.05）の内側なので、**「同等」と読むのが正しい。**
- **この評価セットには、カメラ視点・背景・照明の摂動が 1 行も入っていない。**
  内訳は素 12 行・センサ雑音 60 行・ロボット初期姿勢 12 行である。
  **見た目に対する頑健さについて、この表から言えることは無い。**
- **学習していない素の土台のほうが、コンペの採点では高かった**（2 つのトラックで 0.530 と 0.424）。
  **40,000 ステップの LoRA が素の土台を上回るかどうかは、分かっていないし試してもいない。**

## 使い方

```python
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.configs.policies import PreTrainedConfig

config = PreTrainedConfig.from_pretrained("path/to/checkpoint")
config.checkpoint_path = "path/to/checkpoint/molmoact2_base_small"
policy = MolmoAct2Policy.from_pretrained("path/to/checkpoint", config=config, strict=True)
```

`peft` を入れておくこと。`molmoact2_base_small/`（設定・トークナイザ・正規化の統計）は
このリポジトリに同梱してあり、必須である。

### 通信の無い環境で動かす

**設定には、行動を離散化するトークナイザとして Hub のリポジトリ名が書かれている**
（`discrete_action_tokenizer: "allenai/MolmoAct2-FAST-Tokenizer"`）。
LeRobot は読み込み時に、**その文字列をまずローカルのパスとして解釈し、
無ければ `snapshot_download` へ落ちる。**
**通信の無い環境では、その落ちた先で必ず失敗し、方策は起動しない。**

先に落として、設定からローカルのディレクトリを指させればよい。

```bash
hf download allenai/MolmoAct2-FAST-Tokenizer --local-dir ./fast_action_tokenizer
```

```python
config.discrete_action_tokenizer = "./fast_action_tokenizer"
```

**私たちはこれでコンペの提出を 1 つ失った**（サーバーが起動せず 0 点）。
頼る前に確かめる価値がある。

## ライセンスと出自

| 何 | 権利者 | ライセンス |
|---|---|---|
| 土台の重み `allenai/MolmoAct2-LIBERO-LeRobot` | Allen Institute for AI | Apache 2.0 |
| LoRA（このリポジトリ） | 作者 | Apache 2.0 |
| LeRobot | The HuggingFace team | Apache 2.0 |
| LIBERO / LIBERO-plus の実演データ | Lifelong Robot Learning ほか | MIT |

**学習データそのものはここに含まれていない**（含まれるのは学習の結果の重みだけである）。

## 関連

- データ: [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)
- 土台: [`allenai/MolmoAct2-LIBERO-LeRobot`](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot)
