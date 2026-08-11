# 10. モデルに渡す観測の形（画像の枚数と状態の並び）

**出発点の重みと、追加学習した重みでは、設定ファイルの宣言が食い違っている。**
本番スコアが 0.1409 から 0.016 に落ちた原因を調べる途中で見つかった。

**結論を先に書く。この食い違いは推論では打ち消されるので、悪化の原因ではない。**
悪化の原因は小さなデータで繰り返しすぎたことで、そちらは
[../strategy/05_実測結果.md](../strategy/05_実測結果.md) にある。

この文書は**追いかけなくてよい理由**を残すためのものである。設定ファイルを並べると
明らかに違って見えるため、次に見た人が同じ道を辿り直す。

> 事実には出典を付けた。**リポジトリ内の他の `.md` は出典にしていない。**
> 根拠は重みに同梱された設定ファイル、同梱した LeRobot v0.6.0 のソース、
> [../../solution/train/train.py](../../solution/train/train.py) の実物である。

## 宣言は確かに食い違っている

| | 出発点（本番 0.1409） | 追加学習した重み（本番 0.016） |
|---|---|---|
| 画像の名前 | `camera1` `camera2` `camera3` | **`front` `wrist`** |
| 空の枠（`empty_cameras`） | **2** | **0** |
| 状態の次元（宣言） | **6** | **8** |
| 学習に使ったデータ | `pepijn223/libero_plus_lerobot` | `lerobot/libero_plus` |

出発点の側は `config.json` と `train_config.json` の両方が同じことを言っているので、
**宣言どおりの形で学習された**と読める。

## 食い違う理由は学習スクリプトの 3 行

[../../solution/train/train.py:519-521](../../solution/train/train.py#L519-L521) が毎回これを渡す。

```
--policy.input_features=null
--policy.output_features=null
--policy.empty_cameras=0
```

`input_features=null` にすると、LeRobot は**学習データ側の列名から受け口を組み直す**。
`lerobot/libero_plus` のカメラは `front` と `wrist` なので、そのまま名前になる。
**この 3 行は固定文字列なので、今後の学習もすべて同じ受け口になる。**

## それでも推論では打ち消される

理由は 3 つあり、いずれも同梱した LeRobot のソースで確かめられる。

### 1. 状態はどちらも 8 次元である（宣言の 6 は使われない）

出発点の `config.json` は状態を 6 次元と言うが、**実際に使う正規化の統計量は 8 次元**である。

```
observation.state.mean  [8]
observation.state.std   [8]
```

計算は要素ごとの `(x - mean) / std` なので、宣言の数字ではなく**統計量の形**で決まる。
渡す側（`LiberoProcessorStep`）は位置 3 ＋ 回転 3 ＋ グリッパ 2 の **8 次元を常に組む**
（`processor/env_processor.py`）。さらに `prepare_state` が `max_state_dim=32` まで
零で伸ばすため、**出発点と追加学習した重みで状態はまったく同じものになる。**

同じ統計量ファイルには画像の統計量も `front` `wrist` の名前で入っているが、
`norm_map` が `VISUAL: IDENTITY` なので**画像は正規化されない**。名前が合わなくても影響しない。

### 2. 零で埋める枠は「欠けた数」ではなく `empty_cameras` で止まる

`modeling_smolvla.py` の `prepare_images` はこう書かれている。

```python
for num_empty_cameras in range(len(missing_img_keys)):
    if num_empty_cameras >= self.config.empty_cameras:
        break
```

**埋める枠数は「欠けている数」と `empty_cameras` の小さい方**になる。
評価環境が渡すのは `front` と `wrist` の 2 枚だけなので、実際の枠数はこうなる。

| | 実際の画像 | 零で埋めた枠 | 合計 |
|---|---|---|---|
| 出発点 | 2（`camera1` `camera2` に名前を付け替え） | **2** | **4 枠** |
| 追加学習した重み | 2（`front` `wrist`） | 0 | **2 枠** |

名前の付け替えは**出発点の重みに同梱された前処理**が行っている。参加者が足したものではない。

```json
"rename_map": {
  "observation.images.front": "observation.images.camera1",
  "observation.images.wrist": "observation.images.camera2"
}
```

### 3. 零で埋めた枠は注意から外れ、位置番号も進めない

埋めた枠は `mask` が 0 で入る。`embed_prefix` はその `mask` をそのまま `pad_masks` に積み、
呼び出し側が 2 つに使う。

| 使い道 | 効果 |
|---|---|
| `make_att_2d_masks(prefix_pad_masks, ...)` | **埋めた枠は注意の対象から外れる** |
| `torch.cumsum(prefix_pad_masks, dim=1) - 1` | **位置番号が進まない**ので、後続のトークンの番号がずれない |

画像 1 枚ごとに目印のトークンを足す仕組み（`add_image_special_tokens`）もあるが、
**両方の重みで `false`** なので、枠数の差がトークン数に出ることもない。

**つまり枠が 4 でも 2 でも、モデルが実際に見るものは同じ 2 枚である。**
差は「零の画像を 2 枚 SigLIP に通す時間」だけで、出力には出ない。

## 別の話として、出発点は 3 台目のカメラを失っている

出発点は `camera1` `camera2` `camera3` の**実カメラ 3 台**で学習されている
（`train_config.json` の `input_features` はどれも `[3, 256, 256]`。
空の 2 枠だけが `[3, 480, 640]` で区別できる）。

評価環境が渡すのは 2 枚なので、**`camera3` は零で埋められる。**
出発点の重みは、学習時に使えた視点 1 つを本番では失った状態で 0.1409 を出している。

**これは追加学習で直せる話ではない。** 評価環境が 3 枚目を渡さないため、
3 台前提に戻しても埋める枠が増えるだけである。

## 学習データは 2 つあり、同じものではない

[01_base_model.md](01_base_model.md) が未確認として挙げていた
「`pepijn223/libero_plus_lerobot` と `lerobot/libero_plus` が同一か」に答えが出た。

**列の名前が違うので、同一ではない。**

| | カメラの名前 | 状態 |
|---|---|---|
| `pepijn223/libero_plus_lerobot`（出発点が使った） | `camera1` `camera2` `camera3` | 6 次元 |
| `lerobot/libero_plus`（こちらの学習が使う） | `front` `wrist` | 8 次元 |

**中身の軌道が同じかどうかは、これだけでは分からない。** 分かったのは列の作り方が違うことである。

## 出典と、確認しなおす手順

| 事実 | 出典 |
|---|---|
| 出発点の宣言と学習データ | `work/artifacts/model_weights_base/train_config.json`、`config.json` |
| 状態の統計量が 8 次元 | `policy_preprocessor_step_5_normalizer_processor.safetensors` の見出し |
| 名前の付け替え | `work/artifacts/model_weights_base/policy_preprocessor.json` |
| 状態を 8 次元に組む場所 | 同梱 LeRobot `processor/env_processor.py` の `LiberoProcessorStep` |
| 埋める枠数の決まり方 | 同梱 LeRobot `policies/smolvla/modeling_smolvla.py` の `prepare_images` |
| 埋めた枠が注意から外れること | 同上 `embed_prefix` と `sample_actions` の `prefix_position_ids` |
| 受け口を組み直している 3 行 | [../../solution/train/train.py:519-521](../../solution/train/train.py#L519-L521) |

```bash
# 宣言の食い違いを並べて見る
work/venv310/bin/python -c "
import json
for n in ['model_weights_base', 'model_weights']:
    c = json.load(open(f'work/artifacts/{n}/config.json'))
    imgs = [k.split('.')[-1] for k in c['input_features'] if 'images' in k]
    print(n, 'empty=', c['empty_cameras'],
          'state=', c['input_features']['observation.state']['shape'], imgs)
"

# 実際に使われる統計量の形（宣言ではなくこちらが正）
work/venv310/bin/python - <<'PY'
import json, struct
p = 'work/artifacts/model_weights_base/policy_preprocessor_step_5_normalizer_processor.safetensors'
with open(p, 'rb') as f:
    head = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
print({k: v['shape'] for k, v in head.items() if 'state' in k})
PY
```

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| 零の画像 2 枚を SigLIP に通す時間 | 出発点と追加学習した重みで 1 歩の時間を比べる。**本番は 1 時間に対し実測 305 秒なので、困っていない** |
| 2 つの学習データの軌道が同じか | 両方の `meta/info.json` と、同じタスクのエピソード本数を突き合わせる |
| 状態の 8 次元の並びが学習データと一致しているか | `lerobot/libero_plus` の `observation.state` の統計量と、`LiberoProcessorStep` が組む値の分布を比べる。**ここは未確認のまま残っている** |
