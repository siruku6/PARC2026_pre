# 03. 画像を読む側と行動を出す側で学習率を分ける

**やりたいこと。** SmolVLA の追加学習で、**画像を読む側（VLM）を行動を出す側
（action expert）より小さい学習率で更新する。**

**結論。できる。** LeRobot v0.6.0 に必要な仕組みは揃っており、**書き換えるのは
メソッド 1 つだけ**である。ただし**その前に、画像側の重みを学習の対象にする作業が必要**で、
いまの設定では画像側に学習される重みが 1 つも無い。

> **この文書は単体で読めるように書いた。** 事実には出典を付け、
> 確認しなおすコマンドも置いた。**根拠は同梱した LeRobot v0.6.0 のソース**
> （`work/artifacts/vendor/lerobot/`。原本は `work/lerobot-src/src/lerobot/`）と、
> **重みに同梱された設定ファイル**である。
> リポジトリ内の他の `.md` は出典にしていない。
>
> 以下、行番号は同梱 LeRobot 内の相対パスで示す。

## 前提：いま何が学習されているのか

**LoRA の層が挿入されるのは 2 か所だけである。** `policies/smolvla/modeling_smolvla.py:496-505`
の `_get_default_peft_targets` が返す対象がこれである。

```python
common_projections = (
    "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
)
target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
return {"target_modules": target_modules, "modules_to_save": []}
```

| LoRA の層が挿入される | 挿入されない |
|---|---|
| action expert の q / v 射影（`lm_expert`） | **画像を読む側（SigLIP）** |
| `state_proj`・`action_in_proj`・`action_out_proj`・`action_time_mlp_in`・`action_time_mlp_out` | **VLM 本体（言語側を含む）** |

**したがって `freeze_vision_encoder` と `train_expert_only` を切り替えても、
LoRA を使う限り学習される重みは変わらない。** 画像側には学習対象が無いので、
**学習率を分けても、その学習率で更新される重みが存在しない。**

一方、出発点の重み `lerobot/smolvla_libero_plus@7bb70aa5` は
`use_peft=false` / `freeze_vision_encoder=false` / `train_expert_only=false` で、
**画像を読む側まで含めた全パラメータ**を 20,000 ステップ・バッチ 32・学習率 1e-4 で
学習している（重みに同梱の `train_config.json`）。

**つまり「画像側の学習率を下げる」の実体は、
「いま 1 度も更新されていない画像側の重みを、action expert より小さい学習率で更新し始める」ことである。**

## 手順 1：画像側の重みを学習の対象にする

**コード修正は不要。** `PeftConfig` に `target_modules` があり
（`configs/default.py:117-147`）、既定は `None`（＝上の既定対象を使う）である。

```python
class PeftConfig:
    target_modules: list[str] | str | None = None    # :124
    method_type: str = "LORA"                        # :133
    r: int = 16                                      # :141
    lora_alpha: int | None = None                    # :147
```

`lerobot-train` に `--peft.target_modules=<正規表現>` を渡せば上書きできる。
既定の対象に VLM 側を足す形にする。**既定の対象を正規表現から消さないこと。**
消すと action expert 側に LoRA が挿入されず、行動を出す側が学習されなくなる。

```
--peft.target_modules='(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)|model\.vlm_with_expert\.vlm\..*\.(q|v)_proj)'
```

**最後の 1 項が足した分である。** 正規表現は `peft` が
モジュール名に対して照合する。**正規表現が何個のモジュール名に一致したかを必ず確認すること**（[手順 3](#手順-3設定が反映されたかを確かめる)）。
`modules_to_save` は既定で空なので、LoRA を挿入した層以外はチェックポイントに保存されない。

> **`--peft.target_modules` に何を渡すかは実測で決める。** 上は
> 「VLM の注意の q / v だけ」に絞った例である。
> **一致するモジュールを増やすほど更新される重みが増え、出発点の重みが持っている性能を損なう危険も大きくなる。**

## 手順 2：重みを 2 群に分け、群ごとに学習率を指定する

### 仕組みは揃っている

`OptimizerConfig.build()` の docstring（`optim/optimizers.py:60-78`）が
**パラメータグループを公式に受け付けると書いている。**

> `params`: Parameters to optimize. Accepts multiple formats …
> **`Iterable[dict]`: List of param groups with 'params' key and optional
> 'lr', 'weight_decay' overrides (e.g., ACT, VQBeT policies)**

`AdamWConfig.build()`（同 `:101-111`）は受け取ったものをそのまま
`torch.optim.AdamW(params, **kwargs)` に渡す。**グループ側に `lr` があれば
その値が使われ、`lr` を書かなかった群は `AdamWConfig` の `lr` が使われる。**

### 同じ狙いの前例が入っている

`XVLAAdamWConfig`（`optim/optimizers.py:132-229`）が、**名前に `vlm` を含む重みを
別グループにして `lr * 0.1`・`weight_decay * 0.1` にしている。**
docstring はこう書いている。

> The Vision-Language Model (VLM) is trained with 1/10 of the base learning rate
> for stable optimization, while all other components use the full LR.
> **This LR ratio is crucial for achieving strong and stable finetuning performance.**

**上流が「この比率が重要」と明言している。** 1/10 は出発点として妥当な値である。

その実装（同 `:182-220`）はこの形である。

```python
assert isinstance(params, dict), "Custom LR optimizer requires `named_parameters()` as inputs."
vlm_group, soft_prompt_group, other_group = [], [], []
for name, p in params.items():
    if not p.requires_grad:
        continue
    if "vlm" in name.lower():
        vlm_group.append(p)
    ...
param_groups = [
    {"params": vlm_group, "lr": self.lr * 0.1, "weight_decay": self.weight_decay * 0.1, "name": "vlm"},
    ...
]
param_groups = [g for g in param_groups if len(g["params"]) > 0]
```

### 書き換えが必要な 1 か所

**SmolVLA は平たい列を返している。** `modeling_smolvla.py:274-275`。

```python
def get_optim_params(self) -> dict:
    return self.parameters()
```

`XVLAAdamWConfig` は `assert isinstance(params, dict)` で
`named_parameters()` を要求するため、**`--optimizer.type` を渡すだけでは通らない。**
ここを差し替える必要がある。

## 実装方針：`get_optim_params()` がグループを返すようにする

**方針 A を採る。** 比較した 3 つを先に示す。

| 方針 | 変えるところ | 評価 |
|---|---|---|
| **A（採用）** | `SmolVLAPolicy.get_optim_params()` がグループを返す | **最小。** optimizer と scheduler の設定はそのまま使える |
| B | `get_optim_params()` が `dict(self.named_parameters())` を返し、CLI で `--optimizer.type` を XVLA 系に差し替える | **同じ書き換えが必要なうえ、scheduler も CLI で指定し直す必要がある**（下記） |
| C | 2 つの optimizer を使う（`MultiAdamConfig`、同 `:231-263`） | 過剰。SmolVLA は 1 つのモデルである |

方針 B で scheduler の指定まで必要になる理由は `configs/train.py:249-253` にある。

```python
if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
    raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
elif self.use_policy_training_preset and not self.resume:
    self.optimizer = active_cfg.get_optimizer_preset()
    self.scheduler = active_cfg.get_scheduler_preset()
```

**optimizer だけを差し替えることはできない。** `--optimizer.*` を使うには
`use_policy_training_preset=false` にする必要があり、そのとき
`--scheduler.*` も自分で全部渡さなければならない。
**方針 A なら preset のまま（`AdamWConfig` ＋ `CosineDecayWithWarmupSchedulerConfig`）で済む。**

### 変更の中身

`policies/smolvla/modeling_smolvla.py` の `get_optim_params()` を、
**学習される重みだけを 2 つに分けて返す**形にする。

```
処理の流れ（擬似コード）

  名前付きの重みを全部見る
  `requires_grad` が偽の重みは対象から除外する
      LoRA を使う場合、LoRA を挿入した層以外はここで除外される
  名前が画像・言語側（VLM）に属するなら「画像側」へ
  そうでなければ「行動側」へ

  画像側のグループには学習率を 設定値 × 比率 で明示する
      比率は環境変数か設定から読む。既定 0.1（上流 XVLA と同じ）
  行動側のグループには学習率を書かない
      書かなければ AdamWConfig の lr がそのまま使われる

  重みが 0 本の群はリストから除外する
      画像側に LoRA を挿入していない実行では、群が 1 つだけになる
  2 つのグループの重みの数を必ず表示する
      **画像側が 0 本なら、手順 1 の正規表現が 1 つも一致していない**
```

**どちらの群に分類するかの判定条件に注意する。** `lm_expert`（行動側）は
`model.vlm_with_expert.lm_expert...`、画像・言語側は
`model.vlm_with_expert.vlm...` である。**どちらも `vlm_with_expert` を含む**ため、
XVLA のコードをそのままコピーして `"vlm" in name.lower()` で判定すると、**action expert の重みまで画像側の群に分類されてしまう。**
`lm_expert` を先に判定するか、`.vlm.` で照合する。

### 置き場所

**学習を始める前に、LeRobot のソースファイルを書き換える。**
同じことをしている前例が 2 つある。

| 前例 | 何をしているか |
|---|---|
| 配布ノートブック `examples/smolvla_libero_spatial_lora.ipynb` | LeRobot を clone したあと `smolvlm_with_expert.py` の dtype と `lerobot_train.py` のログ出力を書き換えている |
| `solution/train/train.py` の `_patch_dtype_for_gpu` | bf16 非対応の GPU で同じ dtype 置換をしている |

**提出物の推論側とは無関係である。** 提出物が同梱する
`work/artifacts/vendor/lerobot/` は推論だけに使い、`get_optim_params()` は呼ばれない。

置換は**書き換え対象の文字列が 1 個だけ見つかった場合に限り実行し、0 個または 2 個以上なら例外を投げて止める**こと。
LeRobot の版が変わったときに黙って壊れないようにする
（`solution/tools/build/vendor_lerobot.py` が同じ方針を採っている）。

## スケジューラは比率を壊さない（確認済み）

**scheduler が群ごとの学習率を上書きしてしまうと 2 つの学習率が同じ値になるため、先に確かめた。**

SmolVLA の preset は `CosineDecayWithWarmupSchedulerConfig`
（`policies/smolvla/configuration_smolvla.py:141-147`）で、
その `build()`（`optim/schedulers.py:144-182`）は `LambdaLR` を返す。
`lr_lambda` が返すのは**係数**である。

| 区間 | 返す値 |
|---|---|
| warmup 中 | `1/(warmup+1)` から 1 へ線形に上がる**係数** |
| 減衰中 | `alpha = decay_lr / peak_lr` として `(1-alpha)*cos + alpha`。**比**である |

**`LambdaLR` は、群ごとに保持している `base_lr` と この係数を掛け算した値を、その群の学習率として設定する。**
したがって

- warmup でも減衰でも**グループ間の比率は保たれる**
- 画像側の下限は `画像側の base_lr × alpha` になり、**曲線ごと比例して下がる**

**scheduler 側の変更は要らない。**

## 手順 3：設定が反映されたかを確かめる

**「設定したつもり」を防ぐ。** 学習を長く回す前に 3 つ確認する。

| 確認 | 見るもの | 効いていないときの見え方 |
|---|---|---|
| **画像側の重みが学習の対象になったか** | グループごとの重みの数（上の擬似コードで表示させる） | **画像側が 0 のまま。** `--peft.target_modules` の正規表現がどのモジュール名にも一致していない |
| **学習率が 2 本あるか** | 学習ログの `lr` 表示、または `optimizer.param_groups` の `lr` | 1 本しか出ない |
| **比率が保たれているか** | warmup 直後と減衰後の 2 時点で 2 つのグループの `lr` を出す | 比が 0.1 から動いていたら scheduler の扱いを見直す |

学習を始める前に、**短く回して確かめる**のが安い。

```bash
# 5 ステップだけ回してグループの内訳と lr を見る
python train.py --run-name lr_split_smoke --steps 5 --save-freq 5 \
    --peft.target-modules '<手順 1 の正規表現>'     # 渡し方は train.py の `--` 以降でもよい
```

> `solution/train/train.py` は `--` の後ろに書いたものを `lerobot-train` へ
> そのまま渡す。**同じ設定を 2 回渡した場合、後ろに書いた値が採用される。**

## 代わりに失うもの

**この変更は、信用できるどちらの出典からも外れた条件になる。**

| 出典 | 学習範囲 | 学習率 |
|---|---|---|
| 出発点の重み（`train_config.json`） | 全パラメータ | 1e-4 の 1 本 |
| 配布ノートブック | action expert のみ（LoRA） | 3e-4 の 1 本 |
| **この方針** | **expert ＋ 画像側の一部（LoRA）** | **2 本（画像側 1/10）** |

**そのうえで、効いたかどうかを確かめる手立てが今は無い。**

- **採点環境は 8 エピソードしか走らない。** 採点ログの `POST /reset` の回数を
  数えると 8 である（`outputs/prod_eval_results/*.md` の 3 本すべて）。
  **1 本の結末が変わるだけでスコアが大きく動く。**
- **手元の 84 件の検証は本番を予測しなかった。** 追加学習した重みは手元で
  0.500（素は 0.476）だったが、本番では 0.016（素は 0.1409）だった。
- 運営 Q&A（No.29 の回答）は「評価タスク集合・各エピソードの初期状態・乱数シードは
  採点環境に固定値として組み込まれており、提出のたびに変わることはない」と
  答えている。**同じ 8 本なので向きは読めるが、小さな差は読めない。**

**したがって次の順序を勧める。**

1. **本番に近い形で測れる小さな検証集合を先に用意する。** これが無いと、
   学習率を分けた効果を判定できない
2. 学習率を分けた回と分けない回を、**同じ検証集合で**比べる
3. 手元で差が出たものだけを提出に回す

**学習の条件は必ず残すこと。** バッチ数と使ったエピソードは重みの
`config.json` に残らない。`solution/train/merge.py` は
`train_config.json` と `episodes.json` を `merged_model/` へ同梱するので、
**学習マシンからコピーするときは、この 2 つのファイルも一緒にコピーすること。**

## 出典と、確認しなおす手順

| 事実 | 出典 |
|---|---|
| LoRA の層が挿入される対象 | `policies/smolvla/modeling_smolvla.py:496-505` |
| `target_modules` を CLI から渡せる | `configs/default.py:117-147`（特に `:124`） |
| optimizer がグループを受け付ける | `optim/optimizers.py:60-78`（docstring）、`:101-111`（AdamW） |
| VLM を 1/10 にする前例と「比率が重要」の記述 | `optim/optimizers.py:132-229` |
| SmolVLA が平たい列を返している | `policies/smolvla/modeling_smolvla.py:274-275` |
| preset を使わない設定にすると scheduler も自分で指定する必要がある | `configs/train.py:249-253` |
| scheduler が係数を返す（比率が保たれる） | `optim/schedulers.py:144-182`、`policies/smolvla/configuration_smolvla.py:141-147` |
| 出発点は全パラメータ学習・1e-4 | `work/artifacts/model_weights_base/train_config.json` |
| 本番は 8 エピソード | `outputs/prod_eval_results/*.md` の `POST /reset` の回数 |
| 評価条件が提出間で固定 | 運営 Q&A CSV No.29 の回答 |

```bash
# LoRA の層が挿入される対象を原本で確認する
sed -n '496,505p' work/artifacts/vendor/lerobot/policies/smolvla/modeling_smolvla.py

# optimizer がグループを受け付けることを確認する
sed -n '60,78p' work/artifacts/vendor/lerobot/optim/optimizers.py

# scheduler が係数を返すことを確認する
sed -n '163,182p' work/artifacts/vendor/lerobot/optim/schedulers.py

# 同梱コードが v0.6.0 の原本と一致しているか
diff work/lerobot-src/src/lerobot/optim/optimizers.py \
     work/artifacts/vendor/lerobot/optim/optimizers.py
```

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| `--peft.target_modules` に渡した正規表現が VLM のどのモジュール名に一致するか | 短く回して、群ごとの重みの本数と、一致したモジュール名を表示させる |
| 画像側の重みも更新すると VRAM と 1 ステップあたりの時間がどれだけ増えるか | 同じ短い回で計測する。**画像側を更新しない場合の 20,000 ステップで約 55 分が比較の基準** |
| 1/10 が適切な比率か | 上流の XVLA が使っている値をそのまま流用しただけである。**この構成で測った値ではない** |
| 画像側に LoRA を挿入した重みを `merge.py` が正しくマージできるか | `merge.py` は `merge_and_unload` を使い、残った `lora_` を検査して例外にする。**対象が増えても仕組みは同じだが、実行して確かめること** |
