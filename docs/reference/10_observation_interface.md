# 10. 観測の受け口の違いは推論で打ち消される

**出発点の重みと、追加学習した重みでは、設定ファイルの宣言が食い違っている。**
本番スコアが 0.1409 から 0.016 に落ちた原因を調べる途中で見つかった。

**結論：この食い違いは推論では打ち消される。悪化の原因ではない。**
ただし**打ち消しには前提条件がある**（[下](#打ち消しが崩れる条件)）。
その条件は今の設定では満たされているが、**設定を 1 つ変えると崩れる。**

> **出典。** 重みに同梱された設定ファイル、同梱した LeRobot v0.6.0 のソース、
> 配布ノートブック、採点環境から持ち帰ったログ。
> **`solution/` 以下の説明文と `docs/` の他の `.md` は出典にしていない。**
>
> **本番で走るコードは原本と同一である。** 同梱した
> `policies/smolvla/modeling_smolvla.py` と v0.6.0 の原本
> （`work/lerobot-src`）の差は `Unpack` の import 先が
> `typing_extensions` になった 1 か所だけで（Python 3.10 対応）、
> `processor/env_processor.py` はバイト単位で一致する。
>
> 以下、行番号のみの引用は `work/artifacts/vendor/lerobot/policies/smolvla/` 内の
> `modeling_smolvla.py`（**M**）と `smolvlm_with_expert.py`（**S**）を指す。

## 宣言は確かに食い違っている

| | 出発点（本番 0.1409） | 追加学習した重み（本番 0.016） |
|---|---|---|
| 画像の名前 | `camera1` `camera2` `camera3` ＋空 2 枠 | **`front` `wrist`** |
| `empty_cameras` | **2** | **0** |
| 状態の次元（宣言） | **6** | **8** |
| 学習に使ったデータ | `pepijn223/libero_plus_lerobot` | `lerobot/libero_plus` |

**学習中のチェックポイントと出発点を全項目突き合わせた結果、推論に関わる差は
`empty_cameras` と `input_features` の 2 つだけである。**
`resize_imgs_with_padding`（両方 `[512,512]`）、`max_state_dim`（32）、
`chunk_size`（50）、`num_steps`（10）、`attention_mode`（`cross_attn`）、
`num_vlm_layers`（16）、`add_image_special_tokens`（`false`）は一致する。
残りの差（`freeze_vision_encoder`・`train_expert_only`・`use_peft`）は学習時のみ効き、
`pretrained_path`・`repo_id`・`push_to_hub` は覚え書きである。

**この 3 行は運営が配ったノートブックのものである。**
[../../examples/smolvla_libero_spatial_lora.ipynb](../../examples/smolvla_libero_spatial_lora.ipynb)
の `lerobot-train` を組むセルに `--policy.input_features=null`
`--policy.output_features=null` `--policy.empty_cameras=0` がそのまま入っている。
`input_features=null` にすると LeRobot が学習データの列名から受け口を組み直し、
`lerobot/libero_plus` のカメラ名 `front` `wrist` がそのまま採用される。
**参加者側の細工ではなく配布手順どおりの結果である。**

## 観測が渡る経路

### 1. 届くのは常に画像 2 枚だけ

[../../solution/policy_server.py:326-338](../../solution/policy_server.py#L326-L338) が
`agentview_image` と `robot0_eye_in_hand_image` から
`observation.images.front` / `.wrist` を作る。**重みが何であれこの 2 つである。**

### 2. state は LeRobot 側が 8 次元に組む

`LiberoProcessorStep`（`processor/env_processor.py:28`）が
位置 3 ＋ クォータニオンを軸角に直した 3 ＋ グリッパ 2 を連結し（同 `:38-41, 68-73`）、
画像を 180 度反転する（同 `:58-59`）。**両方の重みで同じコードが走る。**

### 3. 名前の対応

| | `rename_map`（同梱の `policy_preprocessor.json`） |
|---|---|
| 出発点 | `front→camera1`, `wrist→camera2` |
| 追加学習した重み | **`{}`（空。素通し）** |

どちらも「自分の `config.json` が宣言した 2 台の名前」に一致する。
**出発点側の付け替えは HuggingFace が同梱したものである。**

### 4. 正規化 —— 宣言の 6 次元は使われない

`norm_map` は両方 `VISUAL: IDENTITY` なので**画像は正規化されない。**
`STATE: MEAN_STD` の統計量は、`policy_preprocessor_step_5_normalizer_processor.safetensors`
の見出しを読むと**両方 `[8]`** である。

```
observation.state.mean  [8]      observation.state.std  [8]
```

計算は要素ごとの `(x - mean) / std` なので、**`config.json` の宣言（6）ではなく
統計量の形（8）で決まる。** 宣言は使われない。

### 5. state の詰め方

`prepare_state` が `pad_vector(state, self.config.max_state_dim)`（M`:488`）で
32 次元まで零で伸ばす。`max_state_dim` は両方 32。

**ここまでで state の経路は完全に同一である。** 残る差は画像の枠数だけになる。

### 6. 枠数は `empty_cameras` で止まる

`prepare_images`（M`:416`）は宣言した画像のうち batch にあるものを present、
無いものを missing に分ける（M`:422-423`）。零詰めの数は

```python
for num_empty_cameras in range(len(missing_img_keys)):
    if num_empty_cameras >= self.config.empty_cameras:   # M:450
        break
    img = torch.ones_like(img) * -1                      # M:452
    mask = torch.zeros_like(mask)                         # M:453
```

**`min(欠けている数, empty_cameras)` で止まる。**

| | 実カメラ | 零詰め | 合計 |
|---|---|---|---|
| 出発点 | 2 | 欠け 3・上限 2 → **2** | **4 枠** |
| 追加学習した重み | 2 | 欠け 0 → 0 | **2 枠** |

零詰め枠は**実枠の後ろ**に積まれる（M`:430-445` の後に `:447-454`）。
中身は直前の実画像と同じ形の全 −1、`mask` は 0 である。

画像の並び順は `image_features`（`configs/policies.py:151-154`）が
`input_features` の並びを保つため、どちらも (agentview, wrist) の順になる。

観測キューは `populate_queues`（`policies/utils.py:30-47`）が
**batch にあるキーだけを詰める**ため、出発点側で `camera3` などの空キューが
できても触られない。`_get_action_chunk`（M`:285-287`）の積み直しも
batch のキーだけを回すので、例外にならない。

## 増えた 2 枠が出力を変えない理由

**5 つある。どれも欠けると成立しない。**

### 理由 1. 自己注意で鍵として外れる

```python
cumsum = torch.cumsum(att_masks, dim=1)                          # M:130
pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]     # M:132
att_2d_masks = att_2d_masks & pad_2d_masks                       # M:133
```

`pad_2d_masks[i, j]` は **i と j の両方**が有効でなければ立たない。
零詰め枠は j として全クエリから落ちる。

### 理由 2. action expert の交差参照でも外れる ← ここが要

`attention_mode` が `cross_attn` なので、expert は prefix の
キー・値キャッシュを参照する。**そこにも同じ遮蔽が渡っている。**

```python
prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(...)   # M:897
full_att_2d_masks = torch.cat([prefix_pad_2d_masks, ...], dim=2) # M:901
```

prefix 側の自己注意も `attention_mask[:, :seq_len, :seq_len]` を切り出して使う
（S`:310`）。**この行が無ければ打ち消しは成立しなかった。**

### 理由 3. 注意ブロックの構造がずれない

画像は `att_masks += [0] * (num_img_embs)`（M`:679`）、
state は `att_masks += [1] * (states_seq_len)`（M`:716`）を積む。
`cumsum(att_masks)`（M`:130`）は**零を足しても変わらない**ので、
言語は 0 番ブロック・state は 1 番ブロックのままである。

### 理由 4. 位置番号がずれない

`prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1`（M`:835`）。
零詰め枠は `pad_mask` が 0 なので番号を進めない。
suffix 側も `prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)`（M`:902`）で
**有効な数だけ**数える。

### 理由 5. 遮蔽は加算ではなく上書きである

注意の実装は 1 つしかない。`get_attention_interface` は分岐せず
`eager_attention_forward` を返す（S`:512-514`）。その中身が決定的である。

```python
big_neg = torch.finfo(att_weights.dtype).min                          # S:550
masked_att_weights = torch.where(attention_mask[...], att_weights, big_neg)  # S:551
```

**`torch.where` は値を上書きする。** 加算（`+ (-inf)`）ではないため、
遮蔽した位置のスコアが NaN であっても `big_neg` に置き換わり、**NaN が
実トークンの行へ混ざらない。** さらに `-inf` ではなく `finfo.min`（有限）なので、
**行が丸ごと遮蔽されても softmax が NaN にならない**（下で数値を示す）。

## 数値で確かめた

torch を手元に用意できなかったため、**S`:543-560` の演算順序を float32 で
そのまま写して確かめた**（`struct` で各演算後に float32 へ丸める）。
`torch.where` → 行の最大を引く softmax → `probs @ V` の順である。

| 確かめたこと | 結果 |
|---|---|
| 遮蔽値は softmax で厳密に 0 になるか | `exp(finfo.min − 最大)` は行の最大が 0.5 / 12 / −30 / 100 のいずれでも **`0.0`**。差が約 −3.4e38 で float64 の exp すら落ちる |
| 零詰め枠を足して実トークンの出力が変わるか | **ビット単位で一致**。零枠の確率は厳密に `0.0`、実枠 4 つの確率もすべてビット一致 |
| 零枠のスコアが NaN / inf でも汚れないか | **汚れない**（`torch.where` が上書きするため） |
| **行が丸ごと遮蔽された零枠自身**はどうなるか | すべて `big_neg` → 最大を引くと全て 0 → **一様分布 1/6** になり出力は有限。**NaN を作らない** |
| もし `-inf` を使う実装だったら | `exp(-inf − (−inf)) = exp(nan)` で**全要素 NaN**。次の層で V が NaN になり実トークンへ伝播する |

**つまり `finfo.min` を使っているという実装上の選択が、打ち消しの成立条件そのものである。**

再現するには次を実行する（依存なし）。

```bash
python3 - <<'PY'
import math, struct
f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
BIG_NEG = f32(-3.4028234663852886e+38)          # torch.finfo(float32).min

def row(scores, values, mask, head_dim=64):
    w = [f32(f32(s) * f32(head_dim ** -0.5)) for s in scores]
    w = [w[j] if mask[j] else BIG_NEG for j in range(len(w))]   # torch.where
    m = max(w)
    e = [f32(math.exp(f32(x - m))) if x - m > -740 else 0.0 for x in w]
    s = f32(sum(e))
    p = [f32(v / s) for v in e]
    out = 0.0
    for j in range(len(p)):
        out = f32(out + f32(p[j] * values[j]))
    return out, p

sc = [3.7182818, -1.4142135, 0.5772157, 2.7182818]
va = [0.3141593, -0.7071068, 1.6180340, -0.2718282]
a, _ = row(sc, va, [True] * 4)
b, p = row(sc + [9999.0, -8888.0], va + [123456.0, -654321.0],
           [True] * 4 + [False] * 2)
print(struct.pack('f', a) == struct.pack('f', b), p[4], p[5])
PY
```

## 打ち消しが崩れる条件

**打ち消しは無条件ではない。** 次のどれかが変わると成立しなくなる。

| 条件 | いまの値 | 崩れると何が起きるか |
|---|---|---|
| **`add_image_special_tokens`** | **両方 `false`** | 真にすると 1 枠ごとに **`mask` が 1 の**目印トークンが増える（M`:651-664, 680-693`）。**遮蔽されないので枠数の差が出力に出る** |
| **零詰め枠の値ベクトルが有限であること** | 満たす | `probs` が 0 でも `0.0 * NaN = NaN` になる。零枠は全 −1 の画像（M`:452`）で SigLIP を通っても有限、各層の出力も一様平均で有限 |
| 遮蔽の当て方 | `torch.where`（上書き） | 加算実装なら NaN が伝播する |
| `attention_mode` | `cross_attn` | 別の経路では M`:897` の遮蔽が使われるか要確認 |

## 実測での裏付けと、していないこと

**裏付け。** 出発点の設定（4 枠・うち 2 枠が遮蔽）は採点環境で **3 回**走り、
0.058 / 0.096 / 0.1409 を返している（`outputs/prod_eval_results/*.md`）。
**遮蔽枠が注意を汚すなら、この 3 回はサーバー落ちか無意味な行動になっていた。**
少なくとも「NaN で壊れてはいない」ことは本番で確認済みである。

**していないこと。**

| 事柄 | なぜ |
|---|---|
| **本物のモデルを 2 つの設定で走らせて行動を突き合わせる** | 手元に torch も numpy も無く、`docker run` と依存の追加が許可されていない |
| PyTorch の `matmul` が系列長で足し合わせ順を変えないことの確認 | 遮蔽枠の寄与は厳密に 0 だが、**非零項の足し合わせ順が変わる可能性は残る。** その場合でも差は下位ビットに限られる |
| 出発点を 2 枠で走らせた場合の本番スコア | 提出枠を使う。優先度は低い |

**決着させる実測は 1 項目だけ変えれば足りる。**
出発点の重みの `empty_cameras` を 2 → 0 にすると、
宣言も前処理も重みもそのままで枠数だけが 4 → 2 になる。
同じ `noise` を渡して `predict_action_chunk` を 2 回呼び、行動を突き合わせればよい。

## 打ち消されない違いが 1 つある

**出発点は `camera3` を実カメラとして学習している。**
`train_config.json` の `input_features` で `camera1` `camera2` `camera3` は
どれも `[3, 256, 256]`、空枠だけが `[3, 480, 640]` なので区別できる。

評価環境が渡すのは 2 枚なので、**`camera3` は零詰め枠に置き換わる。**
出発点の重みは、学習時に使えた視点 1 つを失った状態で 0.1409 を出している。
**評価側が 3 枚目を渡さないため、こちら側では直せない。**

なお、これは追加学習した重みには当てはまらない。
`lerobot/libero_plus` のカメラは `front` `wrist` の 2 台なので、
**学習時と推論時で枚数が一致している。**

## 学習データが 2 つあり、同じものではない

[01_base_model.md](01_base_model.md) が未確認として挙げていた
「`pepijn223/libero_plus_lerobot` と `lerobot/libero_plus` が同一か」に答えが出た。
**列の作り方が違うので、同一ではない。**

| | カメラの名前 | state |
|---|---|---|
| `pepijn223/libero_plus_lerobot`（出発点が使った） | `camera1` `camera2` `camera3` | 6 次元 |
| `lerobot/libero_plus`（こちらの学習が使う） | `front` `wrist` | 8 次元 |

**軌道の中身が同じかどうかは、これだけでは分からない。**

## 出典と、確認しなおす手順

| 事実 | 出典 |
|---|---|
| 宣言の食い違い | 各重みの `config.json`、出発点の `train_config.json`（HF 同梱） |
| 推論に関わる差が 2 項目だけ | 学習中のチェックポイントの `config.json` と出発点の全項目突き合わせ |
| 3 行の出どころ | 配布ノートブックの `lerobot-train` を組むセル |
| state の統計量が 8 次元 | `policy_preprocessor_step_5_normalizer_processor.safetensors` の見出し |
| 名前の付け替え | 各重みの `policy_preprocessor.json` |
| state を 8 次元に組む場所 | `processor/env_processor.py:28,38-41,68-73` |
| 零詰めの枠数と中身 | M`:416,422-423,450,452-453` |
| 遮蔽の作り方 | M`:130,132,133` |
| 交差参照への遮蔽 | M`:897,901` と S`:310` |
| 位置番号 | M`:835,902` |
| 注意実装が 1 つだけ | S`:512-514` |
| 上書きと `finfo.min` | S`:550-551` |
| 本番で 3 回走った | `outputs/prod_eval_results/*.md` |

```bash
# 推論に関わる差だけを並べる
python3 -c "
import json
a = json.load(open('work/artifacts/model_weights_base/config.json'))
b = json.load(open('work/artifacts/model_weights/config.json'))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print(k, '|', a.get(k), '->', b.get(k))
"

# 実際に使われる統計量の形（宣言ではなくこちらが正）
python3 - <<'PY'
import json, struct
p = ('work/artifacts/model_weights_base/'
     'policy_preprocessor_step_5_normalizer_processor.safetensors')
with open(p, 'rb') as f:
    head = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
print({k: v['shape'] for k, v in head.items() if 'state' in k})
PY

# 同梱コードが原本と一致しているか
diff work/lerobot-src/src/lerobot/policies/smolvla/modeling_smolvla.py \
     work/artifacts/vendor/lerobot/policies/smolvla/modeling_smolvla.py
```
