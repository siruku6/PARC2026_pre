# 11. 追加学習の学習率とステップ数（本番 0.1409 → 0.016 の一次情報）

**追加学習した重みを提出したら、本番スコアが 0.1409 から 0.016 に落ちた。**
その学習が何をしていたのかを、**一次情報だけ**で組み直す。

> **この文書が出典に使うもの。**
>
> | 出典 | 何が分かるか |
> |---|---|
> | 配布ノートブック [../../examples/smolvla_libero_spatial_lora.ipynb](../../examples/smolvla_libero_spatial_lora.ipynb) | 運営が示した学習の条件 |
> | 配布 [../../README.md](../../README.md) | 学習は「参考例」で提出には必須でない、とだけ書かれている |
> | `work/artifacts/model_weights_base/train_config.json` と `config.json` | **出発点の重みが作られた条件**（HF が重みに同梱） |
> | `work/train/smolvla_lora_all_suites/merged_model/config.json` | **こちらの学習が書き出した実物** |
> | `outputs/prod_eval_results/*.md` | 採点環境から持ち帰ったログ（スコアの行は機械の出力） |
>
> **`solution/` 以下の説明文と、`docs/` の他の `.md` は出典にしていない。**

## 3 つの条件を並べる

**この表は「学習率の組み立てを 3 者で比べる」ためのものである。**流した学習そのものの
一覧（実行名・機械・所要・ログの置き場・出た重み・出た点）は
[19_weights_catalog.md](19_weights_catalog.md) にある。


| | 出発点の重み | **こちらの学習** | 配布ノートブック |
|---|---|---|---|
| 本番スコア | **0.1409** | **0.016** | （提出していない） |
| ステップ数 | 20,000 | **20,000** | 3,000 |
| 学習率 | 1e-4 | **3e-4** | 3e-4 |
| 立ち上げ（warmup） | 1,000 | **100** | 100 |
| 最後の学習率 | 2.5e-6 | **3e-5** | 3e-5 |
| 学ぶ範囲 | **全パラメータ** | action expert だけ（LoRA） | action expert だけ（LoRA） |
| バッチ | 32 | **記録が残っていない** | 1 |
| データ | `pepijn223/libero_plus_lerobot` 全部 | **記録が残っていない** | `lerobot/libero_plus` の 10 タスク × 5 本 |

**ステップ数は推測ではない。** ノートブックもこちらの学習も
`--policy.scheduler_decay_steps={ステップ数}` を渡しており、**この値は重みの `config.json` に残る。**
こちらの学習の `config.json` は `scheduler_decay_steps: 20000` である。

## 分かったこと

**学習率の組み立てが、どちらの出典にも無い形になっている。**

- ノートブックの 3e-4 / warmup 100 / 最後 3e-5 は、**3,000 ステップ × バッチ 1 のための値**である。
- 出発点の 20,000 ステップは、**1e-4 / warmup 1,000 / 最後 2.5e-6** で回されている。

**こちらの学習は、前者の学習率で後者のステップ数を回した。**
ステップ数はノートブックの 6.7 倍、学習率は出発点の 3 倍、立ち上げは 10 分の 1、
最後の学習率は 12 倍である。**この組み合わせを認めている出典は無い。**

さらに、学ぶ範囲が出発点と違う。出発点は
`freeze_vision_encoder=false` / `train_expert_only=false` / `use_peft=false`
で**画像を読む側まで含めて全部**学習している（`train_config.json`）。
こちらは `freeze_vision_encoder=true` / `train_expert_only=true` で LoRA なので、
**動くのは action expert だけ**である。ノートブックもそうしている。

## 分からないこと（記録が残っていない）

| 事柄 | なぜ分からないか |
|---|---|
| **バッチ数** | `config.json` に残らない。学習の出力先 `work/train/smolvla_lora_all_suites/` には `merged_model/` しか無く、`train_config.json` もチェックポイントも残っていない |
| **使ったエピソードの本数** | 同じ理由。`--dataset.episodes` は重みの設定に残らない |
| 1 コマを何回見たか | 上の 2 つが要る。**データのコマ数も未確認**（下のコマンドで取れる） |

**バッチ数とデータ量が分からないと、「学習しすぎたのか」は判定できない。**
ここは埋めるべき穴である。

## 次に学習するとき、どこをどう直すか

### 結論：学習率まわりは直す必要がない。すでに揃っている

**学習中のチェックポイントの `config.json` と出発点の `config.json` を
全項目突き合わせた結果、次の値は完全に一致していた。**

| 項目 | 値 |
|---|---|
| `optimizer_lr` | 1e-4 |
| `scheduler_warmup_steps` | 1,000 |
| `scheduler_decay_lr` | 2.5e-6 |
| `scheduler_decay_steps` | 20,000 |
| `optimizer_betas` / `eps` / `weight_decay` / `grad_clip_norm` | `[0.9, 0.95]` / 1e-8 / 1e-10 / 10.0 |
| `max_state_dim` / `max_action_dim` | 32 / 32 |
| `chunk_size` / `num_steps` | 50 / 10 |
| `resize_imgs_with_padding` | `[512, 512]` |
| `attention_mode` / `num_vlm_layers` | `cross_attn` / 16 |

**差が出たのは `empty_cameras`・`input_features`（推論では打ち消される。
[10_observation_interface.md](10_observation_interface.md)）と、
学習時だけ効く `freeze_vision_encoder`・`train_expert_only`・`use_peft`、
それに覚え書き（`pretrained_path`・`repo_id`・`push_to_hub`）だけである。**

**8/11 に崩壊した組み合わせ（3e-4 / warmup 100 / 20,000 ステップ）は再現していない。**
`train.py` は開始時にその組み合わせを検出して警告を出す。

`--resume` で続けるときも壊れない。`_build_resume_command` は
`--config_path` でチェックポイントの `train_config.json` を読ませ、
`--policy.*` や `--steps` を重ねて渡さないため、**スケジュールが途中で
別物にならない。**

### 直すべきは 3 つ。いずれも `train.py` の既定値ではない

| # | 直すところ | どう直すか | なぜ |
|---|---|---|---|
| **1** | **1 回の学習を 2 台に分けない** | 最後まで同じ GPU で通す。20,000 ステップで約 55 分なので分ける理由が無い | `_patch_dtype_for_gpu` は bf16 非対応の GPU で VLM を fp16 へ書き換える（配布ノートブックと同じ置換）。**途中で機種が変わると、前半と後半で数値の型が違う重みになる。** 8/11 の回は 0〜9,000 ステップが Colab T4（fp16）、9,500〜20,000 ステップが H100（bf16）だった |
| **2** | **合否を判定する道具** | 84 件を「壊れていないかの確認」に限る。本番に寄せた小さい集合を用意する | **本番は 8 エピソードで、84 件とは別の母集団である**（[22_production_scale.md](22_production_scale.md)）。84 件で 0.500 と 0.476 が並んでも 8 本の結果は言い当てられない。**ここを直さないと、次の学習が良くなったかどうかを提出前に判断できない** |
| 3 | 学習の条件を持ち出す | 対応済み。`merge.py` が `train_config.json` を `merged_model/` へ同梱する（**`dataset.episodes` もそこに入る**ので使ったエピソードも分かる） | 8/11 の回は `merged_model/` だけを持ち帰ったため、バッチ数とエピソード本数が失われた |

### 決めるべき判断：LoRA が当たる範囲を広げるか

**これは事実ではなく判断である。** 前提となる事実は原本で確認した。

`_get_default_peft_targets`（同梱 LeRobot `policies/smolvla/modeling_smolvla.py:496-505`）が
返す対象は次の 2 つだけである。

```
model.vlm_with_expert.lm_expert.*.(q|v)_proj
model.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)
```

**視覚を読む側と VLM 本体は 1 つも含まれない。**
したがって **`freeze_vision_encoder` や `train_expert_only` を切り替えても、
LoRA を使う限り学習される重みは変わらない。**

一方、出発点の重みは `use_peft=false` / `freeze_vision_encoder=false` /
`train_expert_only=false` で、**画像を読む側まで含めて全パラメータ**が
20,000 ステップ学習されている（`train_config.json`）。

**つまりこちらの学習は、すでに全部学習済みのモデルに対して、行動を出す側の
一部だけを動かしている。** 効き幅を変えるなら選択肢は 3 つある。

| 選び方 | 変え方 | 引き受けるもの |
|---|---|---|
| **いまのまま（LoRA・expert だけ）** | 何もしない | **崩れにくい。** ただし動かせる幅が狭い |
| LoRA の対象を広げる | `--peft.target_modules` に VLM 側を渡す | 視覚と行動の対応を学び直せるが、**出発点が積み上げたものを崩しうる** |
| LoRA を外して全パラメータ | `--peft.method_type` を渡さない | 出発点と同じ形。**VRAM と時間が増え、崩れる幅も最大** |

**8 エピソードしか観測できない以上、崩れる幅が大きい選び方は割に合わない**
と見ている。判断材料は [22_production_scale.md](22_production_scale.md) にある。

### 数えておくこと

**1 コマを何回見るかは、出発点と同じ濃さに揃える。**
20,000 ステップ × バッチ 32 は 640,000 コマ分の引き出しにあたる。
出発点はこれを 2,238,036 コマのデータに当てて 0.29 回だった。
**こちらのデータのコマ数は未確認である。**

```bash
curl -s https://huggingface.co/datasets/lerobot/libero_plus/raw/main/meta/info.json \
  | python3 -c "
import json, sys
c = json.load(sys.stdin)
frames = c['total_frames']
print(f\"コマ数 {frames:,}  1 コマあたり {20000 * 32 / frames:.2f} 回\")
"
```

**0.3 回前後なら出発点と同じ濃さである。** 大きく超えるならバッチかステップ数を下げる。

## いま回っている学習について確かめること

**下の 4 つを見れば、同じ組み合わせを繰り返していないかが分かる。**

```bash
# 学習の出力先（GPU マシン側）で
run=<出力先>/checkpoints/last/pretrained_model
python3 -c "
import json; c = json.load(open('$run/config.json'))
for k in ('optimizer_lr', 'scheduler_warmup_steps',
          'scheduler_decay_lr', 'scheduler_decay_steps'):
    print(k, c.get(k))
"

# バッチ数とエピソードは LeRobot が書くこちらに残る
python3 -c "
import json; t = json.load(open('<出力先>/train_config.json'))
print('batch', t.get('batch_size'), 'episodes', t['dataset'].get('episodes'))
"
```

**学習率が 3e-4 でステップ数が 20,000 なら、崩壊した回と同じ組み合わせである。**

## 出典を取り直す手順

```bash
# 出発点が作られた条件（HF から revision 固定で取り直せる）
curl -s https://huggingface.co/lerobot/smolvla_libero_plus/raw/7bb70aa5bc92b82c9239142775d3a173103567ff/train_config.json \
  | python3 -c "
import json, sys
c = json.load(sys.stdin); p = c['policy']
print(c['steps'], c['batch_size'], p['optimizer_lr'],
      p['scheduler_warmup_steps'], p['scheduler_decay_lr'])
"

# 学習データのコマ数（1 コマを何回見たかを出すのに要る）
curl -s https://huggingface.co/datasets/lerobot/libero_plus/raw/main/meta/info.json \
  | python3 -c "
import json, sys
c = json.load(sys.stdin)
print({k: c[k] for k in ('total_episodes', 'total_frames', 'total_tasks')})
"

# ノートブックの条件
python3 -c "
import json
nb = json.load(open('examples/smolvla_libero_spatial_lora.ipynb'))
for cell in nb['cells']:
    src = ''.join(cell['source'])
    if 'LEARNING_RATE' in src and 'STEPS' in src:
        print(src)
        break
"
```

**観測の受け口（画像の枚数と状態の並び）の食い違いは、
悪化の原因ではない。** 理由は [10_observation_interface.md](10_observation_interface.md) にある。

## 重み側の 2 つの安い手（2026-08-25 追記）

### 追加学習後の重みと、学習前の重みを混ぜる（WiSE-FT）

**学習を 1 秒も回さずに候補を作れる。** 追加学習で未知の条件への強さが削れる現象は、
画像の分野で WiSE-FT として知られている ── **学習前と学習後の重みを比率で足し合わせると、
未知の分布での成績が上がる。** 比率を 5 通り作っても GPU の学習は 0 秒で、消費するのは
評価の枠だけである。

**予選で 1 度試している。道具も残っている**（`solution/tools/weights/blend_weights.py`）。
ただし**当てた相手が違う。**

| 予選でやったこと | 結果 |
|---|---|
| SmolVLA の**壊れた学習**（LoRA 20,000 ステップ・本番 0.016）に、素の重みを 75% 混ぜた（λ=0.25） | **0.016 → 0.09507**。壊れた学習を救う効果はあったが、素の 0.1409 には届かなかった |

**うまくいった学習に混ぜたことは 1 度も無い。** π0.5 でも試していない。
**4.2.0（Track1 で 0.370・ベースラインを 0.084 上回る）に混ぜるのが、今回の未踏の使い方である。**

**ロボットの方策で同じ効果を報告した論文は、2026-08-25 の調査では見つからなかった。**
近いものは ConSFT（arXiv 2605.08879）で、π0・π0.5 を対象に、控えめな追加学習で
元の能力の保持が平均 20 ポイント以上良くなると報告している。

**本選 Track2 でこれを試す理由がある** ── 配布ベースライン（LoRA・5,000 ステップ）の
Track2 が 0.165 で、こちらの全パラメータ学習（4.2.0）が 0.154 だった。
**全パラメータ学習が未知ドメインへの強さを削っている可能性がある。**

### 学習中の重みの移動平均（EMA）

**π0 の公式実装（openpi）の説明には、メモリが足りないときの最後の手段として
「EMA を切る」と書かれている ── つまり既定では入っている。**
手元の lerobot 経由の π0.5 学習には EMA が無い（`solution/train/` に該当の記述なし）。
未知の条件での安定性に効く見込みがあり、実装は学習ループへ移動平均を 1 本足すだけである。
