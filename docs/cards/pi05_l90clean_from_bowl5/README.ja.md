# π0.5 LIBERO —— `bowl5_full` の続きを、libero_90 を掃除したデータで学習

*English: [README.md](README.md)*

[`siruku6/pi05_bowl5_full`](https://huggingface.co/siruku6/pi05_bowl5_full) の `005500` から、
[`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)
で**全パラメータを 8,000 歩続けた**もの。このデータは、親が学習した 40 課題の LIBERO-plus の枝に
`libero_90` と標準スイートを足した 111 課題である。

チェックポイント: `002000`・`004000`・`006000`・`008000`。

**使う前に読んでほしい。この重みは、親の課題分布での成績を手放して `libero_90` での成績を買っている。**
**それが良い取引かどうかは、何を評価するかで決まる。私たちにとっては良くなかった。**

## これは何か

| | |
|---|---|
| 出発点 | `siruku6/pi05_bowl5_full` の `005500`（これ自体が `lerobot/pi05_libero_base` の全パラメータ学習） |
| 学習した範囲 | **全部 4.14B**（`freeze_vision_encoder: false`・`train_expert_only: false`） |
| ステップ | 8,000（2,000 ごとに保存） |
| batch | 64 |
| 最適化 | adamw・lr 5e-5・weight_decay 0.01・betas (0.9, 0.95)・eps 1e-8・grad_clip_norm 1.0 |
| スケジューラ | cosine_decay_with_warmup・warmup 200・peak 5e-5 → 2.5e-6 |
| 種 | 42 |
| 画像を崩す細工 | **なし** |
| 枠組み | LeRobot v0.6.0 |

## 評価

`008000` を、親自身と並べて測った。摂動を加えた LIBERO の行 × 5 本・EGL 描画。

| 評価セット | 何が入っているか | `bowl5_full`（親） | **`l90clean` の `008000`** |
|---|---|---|---|
| 84 行 × 5（素・センサ雑音・初期姿勢） | 111 課題のうち `libero_90` 側は入っていない | **0.798** | 0.793 |
| 29 行 × 5（背景の質感・照明） | **狙っている課題分布そのもの** | **0.697** | 0.690 |
| 72 行 × 5（カメラ視点・初期姿勢） | **狙っている課題分布そのもの** | **0.792** | 0.767 |
| `libero_90` の 45 課題 | **この模型の学習データに入っており、親には入っていない** | 0.156 | **0.422** |

**伸びは全部 `libero_90` 側から来ている。そこはこの模型が学習し、親は学習していない課題である。**
**私たちが気にしていた課題分布から引いたどの集合でも、2 つは見分けが付かない**
（差は 0.005〜0.025 で、実測した実行ごとの揺れは約 0.05）。

### コンペの点について

この重みは PARC2026 の本選に提出して **0.075** だった。同じトラックで、
**推論の設定が 1 バイトも同じ**親は **0.498** である。

**あの点は、この重みを測ったものではない。**提出した書庫の `model_weights/vlm/` に
**間違ったトークナイザ**が入っていた——π0.5 の Gemma ではなく SmolVLM2（`GPT2Tokenizer`）の
ディレクトリで、**指示文が全部別のトークン列になっていた。**
後で再現した。同じ重み・同じ設定・同じ 2 課題で、トークナイザのディレクトリだけを入れ替えて
**0.000（誤り）対 0.500（正しい）**である。

**したがって正直な要約はこうなる——この重みには有効なコンペの点が無い。**
**手元の評価が言っていることはそのまま立つ**——狙っている分布では親と同等、`libero_90` では上。

## 使い方

```python
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.policies import PreTrainedConfig

config = PreTrainedConfig.from_pretrained("path/to/008000")
config.load_vlm_weights = False        # VLM の重みはこのチェックポイントに入っている
config.vlm_model_name = "path/to/008000/vlm"
policy = PI05Policy.from_pretrained("path/to/008000", config=config)
```

**`vlm/` に入っているのは Gemma のトークナイザである**——4 ファイル・
`tokenizer_class: GemmaTokenizer`。**配置用のディレクトリを手で組むなら、この値を確かめること。**
**`merges.txt` と `vocab.json` があるディレクトリは π0.5 のトークナイザではない。**
**そして方策は動いてしまう**——指示に従えないまま、黙って。

## ライセンスと出自

`lerobot/pi05_libero_base` から派生したもので、その配布は **Gemma Terms of Use** による。

> Gemma is provided under and subject to the Gemma Terms of Use found at
> [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms)

同 Terms の Section 3.2（Use Restrictions）と
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) が適用される。
LeRobot は Apache 2.0、LIBERO / LIBERO-plus の実演データは MIT である。

## 関連

- 親: [`siruku6/pi05_bowl5_full`](https://huggingface.co/siruku6/pi05_bowl5_full)
- データ: [`siruku6/libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)
- 公開したもの全体の見取り図: [PARC2026 の模型索引](https://github.com/siruku6/PARC2026_pre/blob/submission/v1-base-weights/docs/cards/README.ja.md)
