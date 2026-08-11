# 11. 追加学習の学習率と歩数（本番 0.1409 → 0.016 の一次情報）

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

| | 出発点の重み | **こちらの学習** | 配布ノートブック |
|---|---|---|---|
| 本番スコア | **0.1409** | **0.016** | （提出していない） |
| 歩数 | 20,000 | **20,000** | 3,000 |
| 学習率 | 1e-4 | **3e-4** | 3e-4 |
| 立ち上げ（warmup） | 1,000 | **100** | 100 |
| 最後の学習率 | 2.5e-6 | **3e-5** | 3e-5 |
| 学ぶ範囲 | **全パラメータ** | action expert だけ（LoRA） | action expert だけ（LoRA） |
| バッチ | 32 | **記録が残っていない** | 1 |
| データ | `pepijn223/libero_plus_lerobot` 全部 | **記録が残っていない** | `lerobot/libero_plus` の 10 タスク × 5 本 |

**歩数は推測ではない。** ノートブックもこちらの学習も
`--policy.scheduler_decay_steps={歩数}` を渡しており、**この値は重みの `config.json` に残る。**
こちらの学習の `config.json` は `scheduler_decay_steps: 20000` である。

## 分かったこと

**学習率の組み立てが、どちらの出典にも無い形になっている。**

- ノートブックの 3e-4 / warmup 100 / 最後 3e-5 は、**3,000 歩 × バッチ 1 のための値**である。
- 出発点の 20,000 歩は、**1e-4 / warmup 1,000 / 最後 2.5e-6** で回されている。

**こちらの学習は、前者の学習率で後者の歩数を回した。**
歩数はノートブックの 6.7 倍、学習率は出発点の 3 倍、立ち上げは 10 分の 1、
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

## 次に学習するときに決めておくこと

**学習率と歩数は、どちらかの出典に揃える。混ぜない。**

| 選び方 | 学習率 / warmup / 最後 | 歩数 | 根拠 |
|---|---|---|---|
| **出発点の続きを学ぶ** | 1e-4 / 1,000 / 2.5e-6 | 20,000 | `train_config.json`。**長く回すならこちら** |
| ノートブックをなぞる | 3e-4 / 100 / 3e-5 | 3,000 | ノートブック cell 8。**短く回すならこちら** |

**バッチ数とエピソード本数は、次の学習では必ず記録に残す。**
`--dataset.episodes` を渡したかどうかも含めて、学習の出力先に
LeRobot が書く `train_config.json` を捨てないこと。**今回それが無いために、
悪化の原因を数字で確かめられない。**

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

**学習率が 3e-4 で歩数が 20,000 なら、崩壊した回と同じ組み合わせである。**

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
