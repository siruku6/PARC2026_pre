# 09. 使ったモデルは何をしているのか

レポートには「どんなモデルを使ったか」を書く欄がある。読む人は VLA
（画像と言葉を受け取ってロボットの動きを出すモデル）を初めて見るかもしれない。
**そこで、専門知識を前提にせずにこのモデルを説明できる材料を集める。**

これを読むと、次の 3 つがそのまま書ける。

- **何を使ったか** — 名前の由来、大きさ、出発点の重みが何で学習済みか
- **どう動くか** — 画像と指示文を読む側と、動きを作る側の 2 段構え。行動の作り方
- **自分たちが何を足したか** — LoRA がモデルのどこに当たっていて、何個の重みを学んだか

> **この文書の書き方について。** 事実には 1 件ずつ出典を付けた。出典は論文（arXiv）、
> HuggingFace のモデルカードと `meta/info.json`、重みに同梱の設定ファイル、LeRobot v0.6.0 のソース、
> この repository のコードのいずれかで、**他の `.md` は出典にしていない。**
>
> 手元で数えた値には「実測」と書き、数え直す手順を末尾に置いた。
> **図は描かず、論文と HuggingFace の該当箇所を指す**（下の「図はどこを見るか」）。

## 1 行でいうと

| | |
|---|---|
| モデル名 | **SmolVLA**（HuggingFace 製）。出発点の重みは `lerobot/smolvla_libero_plus` revision `7bb70aa5` |
| 受け取るもの | カメラ 2 台の画像、ロボットの姿勢 8 個の数値、英語の指示文 1 本（例: `Pick the milk and place it in the basket`） |
| 出すもの | 未来 50 ステップ分の動き（1 ステップは 7 個の数値） |
| 大きさ | **4.50 億個の重み**（実測）。ファイルは 865 MiB |
| ライセンス | Apache-2.0（`lerobot/smolvla_libero_plus` のモデルカード） |
| こちらが足したもの | **LoRA。37 か所の線形層だけを学習し、学習後に元の重みへ畳み込んだ** |

## 名前の由来 — 「小さい」＋「見る・言葉・動く」

`SmolVLA` は 2 つの語をつないだ名前である。**`VLA` は Vision-Language-Action
（視覚・言語・行動）で、画像と言葉を受け取って行動を出すモデルを指す。**
`Smol` は small を崩した綴りで、論文の要旨も自らを "a small, efficient,
community-driven VLA" と説明している（[arXiv:2506.01844](https://arxiv.org/abs/2506.01844)、
題名は "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"、2025-06-02）。

小さいことは売り文句である。論文は、既存の VLA が「しばしば数十億個の重みを持つ」ため
学習も配備も高くつくと述べ、**1 枚の GPU で学習でき、民生用の GPU や CPU で動くこと**を
狙いに挙げている（同要旨）。土台にしている視覚言語モデルも同じ「Smol」一族の
SmolVLM2 で、名前はそこから続いている。

## 何を受け取り、何を出すか

出発点の重みに同梱された設定ファイルと、提出物側の前処理コードから読める。

| | 中身 | 出典 |
|---|---|---|
| 画像 | 正面カメラと手首カメラの 2 枚。256×256 で受け取り、**内部で 512×512 へ余白を足して拡大**する | 重み同梱の `config.json`（`input_features`、`resize_imgs_with_padding: [512, 512]`） |
| 姿勢 | **8 個の数値** = 手先の位置 3 ＋ 手先の向き 3 ＋ 指の開き 2 | LeRobot v0.6.0 [`LiberoProcessorStep`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/processor/env_processor.py)（コード内の注記も `shape=(8,), # [eef_pos(3), axis_angle(3), gripper(2)]`） |
| 指示文 | 英語 1 本。**48 個の単位（トークン）に切って、足りない分は詰め物で埋める** | 同 `config.json`（`tokenizer_max_length: 48`、`pad_language_to: "max_length"`） |
| 出す行動 | **7 個の数値 × 50 ステップ分。** 先頭 6 個は手先をどれだけ動かすかの差分、7 個目は指の開閉 | 同 `config.json`（`chunk_size: 50`、`action` の shape 7）。7 個目は逆正規化統計の min/max がちょうど ±1（実測） |

**7 個目だけ性質が違う。** 前半 6 個は連続値だが、指の開閉は ±1 しか取らない
入り切りの信号である（実測: 逆正規化統計の std は 0.999 で、ほぼ両端に張り付いている）。
後から動きをならす処理を入れるときは、前半 6 個だけを対象にする。

## 大きさ — 4.5 億。うち動きを作る側が 1 億

論文は "contains 450 million parameters, with approximately 100 million dedicated to the
action expert"（4.5 億個のうち約 1 億が action expert）と書いている。**手元の重みを数えたら、その通りだった。**

| 部分 | 重みの数（実測） | 全体比 | 何をする部分か |
|---|---|---|---|
| 言語側（SmolLM2 の 16 層） | 204.6 M | 45.5% | 指示文と画像を並べて読む |
| **action expert** | **98.2 M** | **21.8%** | **動きを作る** |
| 視覚側（SigLIP 12 層） | 86.4 M | 19.2% | 画像を細かい単位に変える |
| 語彙の出力層・画像と言語のつなぎ | 59.1 M | 13.1% | 元は文章を書くための部分（行動生成では出力層は使わない） |
| 状態・行動・時刻の射影 5 層 | 1.6 M | 0.4% | 数値をモデルの内部表現に変換する |
| **合計** | **450.0 M** | 100% | ファイルは 906,712,520 バイト = 865 MiB、ほぼ bfloat16 |

## 2 段構え — 見て読む側と、動きを作る側

**SmolVLA は 2 つのモデルが縦につながった形をしている。**
前半の視覚言語モデル（VLM）が画像・指示文・姿勢を読んで中間結果を作り、
後半の action expert がそれを参照しながら行動を作る。

| | 前半：VLM | 後半：action expert |
|---|---|---|
| 正体 | **SmolVLM2-500M-Video-Instruct**。画像側は SigLIP、言語側は SmolLM2-360M | LeRobot が同じ形の層を新しく組んだ小型版 |
| 層の数 | **32 層のうち前半 16 層だけを使う** | 16 層 |
| 1 層の幅 | 960 | **720（VLM の 0.75 倍）** |
| 学習済みか | 画像 34%・動画 33%・文章 20% の 330 万件で事前学習済み | 出発点の重みを作るときに学習された |
| 出典 | 重み同梱 `train_config.json`（`vlm_model_name`、`num_vlm_layers: 16`）、[SmolVLM2 のモデルカード](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)、同梱 `vlm/config.json`（`num_hidden_layers: 32`） | 同 `train_config.json`（`expert_width_multiplier: 0.75`、`self_attn_every_n_layers: 2`）、実測の層数と行列の形 |

**なぜ半分で切るのか。** 論文は層を N 層で打ち切る比較を載せ、「N を全体の半分にすると
速さと性能の釣り合いが良い」と述べている（同論文の該当節では、500M の VLM を半分で切るほうが
256M の小さい VLM を丸ごと使うより成績が良い）。**画像 1 枚は 64 個の単位まで削られる**
（論文の記述であり、同梱 `vlm/config.json` の `scale_factor: 4` と 512×512 入力から
32×32 = 1,024 個の切れ端を 16 分の 1 にまとめる計算とも合う）。

**action expert の 16 層は、2 種類が交互に並んでいる。**
偶数番の層は作りかけの行動どうしを見比べ、**奇数番の層は VLM が計算した中間結果の控え
（キャッシュ）を読みに行く**（LeRobot v0.6.0
[`smolvlm_with_expert.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
の `forward` が `layer_idx % self_attn_every_n_layers == 0` で振り分けており、設定値は 2）。
論文はこの交互配置が、どちらか片方だけより良いと報告している。

## 行動の出し方 — 雑音から 10 回で削り出す（flow matching）

**このモデルは行動を一度に書き下さない。** 出発点は乱数（正規分布の雑音）で、
そこから少しずつ形を整えて行動列に変えていく。この作り方を **flow matching** と呼ぶ。

| | 学習のとき | 推論のとき |
|---|---|---|
| すること | 正解の行動と雑音を割合 `t` で混ぜ、**「雑音 − 正解」という向き**を当てさせる | **純粋な雑音（`t`=1）から始め、当てた向きの分だけ引く**のを繰り返す |
| 回数 | 1 回（混ぜる割合はランダムに引く） | **10 回**（`t` を 1.0 から 0.1 刻みで 0 へ） |
| 出力 | 向きの予測と正解の差（二乗誤差） | 50 ステップ分の行動 |
| 出典 | LeRobot v0.6.0 [`modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/smolvla/modeling_smolvla.py) の `VLAFlowMatching.forward`（`x_t = t·noise + (1−t)·actions`、目標 `u_t = noise − actions`） | 同 `sample_actions`（`dt = −1/num_steps`）。`num_steps: 10` は重み同梱 `config.json` |

**推論で 10 回まわすのは行動を作る側だけである。** 画像と指示文を読む VLM の計算は
最初の 1 回だけで、その結果を控えておいて 10 回使い回す（同 `sample_actions` が
`fill_kv_cache=True` で 1 度だけ VLM を通している）。だから重い側は 1 回しか動かない。

**50 ステップ分まとめて作る意味。** 1 回の推論で未来 50 ステップを作れるので、1 ステップごとに推論しなくても動ける。
提出物では**作った 50 ステップのうち先頭 5 ステップだけ使って作り直している**
（[policy_server.py:161](../../solution/policy_server.py#L161)、環境変数 `PARC_N_ACTION_STEPS` で変更できる）。

さらに、動きのガタつきを抑えるため直前の行動を 3 割混ぜている
（[policy_server.py:298-302](../../solution/policy_server.py#L298-L302)。前半 6 次元のみが対象）。

## 出発点の重み — `lerobot/smolvla_libero_plus` は何を学んだか

**この重みは 3 段重ねでできている。**
SmolVLM2（画像と文章で事前学習）→ `lerobot/smolvla_base`（ロボットのデータで学習）→
`lerobot/smolvla_libero_plus`（LIBERO-plus のデータで学習）の順である
（モデルカードの `base_model: lerobot/smolvla_base`、同梱 `train_config.json` の
`pretrained_path: "lerobot/smolvla_base"`）。

3 段目の学習条件は、重みに同梱されている `train_config.json` に全部書かれている。

| 設定 | 値 |
|---|---|
| 学習データ | `pepijn223/libero_plus_lerobot`（`episodes: null` = **全部使う**） |
| その量 | **14,347 エピソード / 2,238,036 コマ / 40 種類の指示文**、ロボットは Panda、20 コマ毎秒 |
| ステップ数 × バッチ | **20,000 × 32**（1 コマあたり平均 0.29 回しか見ていない計算になる） |
| 学習率 | 1e-4 から 2.5e-6 へ余弦で下げる。立ち上げに 1,000 ステップ |
| 画像の加工 | **無効**（`image_transforms.enable: false`） |
| 学習した重み | **全部**（`peft: null`、`use_peft: false`。アダプタを使わない全パラメータ学習） |

出典は同梱 `train_config.json`、データ量は
[`meta/info.json`](https://huggingface.co/datasets/lerobot/libero_plus/raw/f3f49f426d75030177b18778374005bc12ccd588/meta/info.json)
（`lerobot/libero_plus` revision `f3f49f42` のもので、`pepijn223/libero_plus_lerobot` の
`meta/info.json` も同じ 14,347 / 2,238,036 / 40 である）。

**LIBERO-plus とは何か。** ロボット操作の標準的な試験である LIBERO に対し、
**見え方や言い方を変えた 10,030 通りの課題を足して、モデルの弱さを暴くために作られた**
ものである（[github.com/sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)、
論文 "LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models"
[arXiv:2510.13626](https://arxiv.org/abs/2510.13626)）。変える要素は 7 種類
（物の配置、カメラ位置、ロボットの初期姿勢、指示文の言い換え、照明、背景の柄、センサの雑音）で、
同 repository は「わずかな変更で成績が 95% から 30% を下回るまで落ちる」と報告している。

## こちらが足した LoRA はどこに当たっているか

**LoRA は、元の重みを固定したまま、小さな差分だけを学ぶやり方である。**
1 つの線形層 `W` の隣に細長い行列 2 枚（`A`：入力 → 16 次元、`B`：16 次元 → 出力）を置き、
`W + B·A` として使う。学習するのは `A` と `B` だけなので、学ぶ重みが桁で減る。

**当たる場所は LeRobot が正規表現で決め打ちしている。** 自分で選んだのではない。

```python
# LeRobot v0.6.0 src/lerobot/policies/smolvla/modeling_smolvla.py
# SmolVLAPolicy._get_default_peft_targets
common_projections = (
    "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
)
target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
return {"target_modules": target_modules, "modules_to_save": []}
```

この正規表現に当たる線形層を数えたら **37 か所**だった（実測）。

| 場所 | 数 | 中身 |
|---|---|---|
| action expert の `q_proj` と `v_proj` | **32** | 16 層 × 2 種類。奇数番の層の `v_proj` は VLM の控えを読む側でもある |
| 射影 5 層 | **5** | 姿勢の入口、行動の入口と出口、時刻を混ぜる 2 層 |
| **合計** | **37** | **学習する重みは 742,656 個 = 全体 4.50 億の 0.17%**（実測） |

**視覚側と言語側には 1 つも当たっていない。** 正規表現が `lm_expert` と 5 つの射影しか
指していないためである。設定に `freeze_vision_encoder` と `train_expert_only` という
固定の旗があるが、**LoRA を使う間はこの 2 つを変えても学習される重みは変わらない**
（LeRobot v0.6.0 [`pretrained.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/pretrained.py)
の `wrap_with_peft` が全パラメータを `requires_grad_(False)` にしてからアダプタを付けるため）。

**視覚側まで学ばせたいなら、`--peft.target_modules` を自分で書くか、LoRA を外して
全パラメータ学習にするしかない。** 固定の旗をいじっても何も起きない。

**細長い行列の幅（rank）は 16、強さ（alpha）も 16 である**
（[train.py:96-97](../../solution/train/train.py#L96-L97) と
[train.py:399-401](../../solution/train/train.py#L399-L401)。
`alpha / rank = 1.0` なので、学んだ差分をそのままの大きさで足すことになる）。
`modules_to_save` が空なので、**アダプタ以外に保存される重みは 1 つも無い。**

**提出物にアダプタは残らない。** 学習後に `merge_and_unload` で元の重みへ畳み込むため、
提出する重みは出発点と同じ 500 個のテンソル・同じ 4.50 億個で、
中身だけが 37 か所ぶん変わっている（[merge.py:124](../../solution/train/merge.py#L124)）。

## 図はどこを見るか

**自分で描かず、一次資料の図を指す。** レポートに図を載せる場合も出典付きで引用する。

| 見たいもの | 場所 |
|---|---|
| **全体像（VLM → action expert → 行動）** | 論文 Figure 1。[arXiv:2506.01844（HTML 版）](https://arxiv.org/html/2506.01844v1)。VLM の後ろの層を鋏で切る絵と、交互に並ぶ 2 種類の層が描かれている |
| 同じ構造の簡易図 | LeRobot のソース内にある図。[`modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/smolvla/modeling_smolvla.py) の `class VLAFlowMatching` の説明部分 |
| 出発点の重みの中身 | [HuggingFace `lerobot/smolvla_libero_plus` revision 7bb70aa5](https://huggingface.co/lerobot/smolvla_libero_plus/tree/7bb70aa5bc92b82c9239142775d3a173103567ff) |
| 土台の視覚言語モデル | [HuggingFace `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) |
| 学習データ | [HuggingFace `lerobot/libero_plus`](https://huggingface.co/datasets/lerobot/libero_plus)、[LIBERO-plus の repository](https://github.com/sylvestf/LIBERO-plus) |

## レポートに載せる短い版

そのまま貼れる形にした。**ステップ数などの回ごとに変わる数値は `<>` で空けてある。**

> **使用モデル.** ベースは HuggingFace が公開する SmolVLA（arXiv:2506.01844）であり、
> ロボット操作の試験セット LIBERO-plus で学習済みの重み `lerobot/smolvla_libero_plus`
> （revision `7bb70aa5`、Apache-2.0）を出発点とした。総パラメータは 4.50 億で、
> 画像と指示文を読む視覚言語モデル（SmolVLM2-500M の前半 16 層）と、
> 動きを作る action expert（0.98 億）が縦に並ぶ構成である。
>
> **行動の生成.** モデルは 1 回の推論で未来 50 ステップ分の行動（各 7 次元）をまとめて出力する。
> 生成は flow matching により、正規分布の雑音から 10 回の更新で行動列へ変換する形で行う。
> 重い視覚言語モデル側の計算は 1 回だけで、10 回の更新はその中間結果を再利用する。
>
> **追加学習.** 出発点の重みは固定し、LoRA（rank 16、alpha 16）で 37 か所の線形層のみを学習した。
> 対象は action expert 16 層の q/v 射影 32 か所と、状態・行動・時刻の射影 5 か所であり、
> 学習した重みは 742,656 個（全体の 0.17%）である。
> 学習後はアダプタを元の重みへ畳み込んだため、提出物にアダプタは残らない。
>
> **学習データ.** `lerobot/libero_plus`（revision `f3f49f42`、14,347 エピソード /
> 2,238,036 コマ / 40 種類の指示文）を用い、`<ステップ数>` 歩 × バッチ `<バッチ>` で学習した。

## 再確認する手順

**重みを数え直す**（この文書の「実測」はすべてこの出力である）。

```bash
# 総数・部分ごとの内訳・LoRA が当たる線形層の数
python3 - <<'PY'
import json, struct, re, collections
p = "work/artifacts/model_weights.bak/model.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
hdr.pop("__metadata__", None)
numel = lambda s: __import__("math").prod(s)
common = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
pat = re.compile(rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common}))")
tot = sum(numel(v["shape"]) for v in hdr.values())
hits = [k for k in hdr if pat.search(k) and k.endswith(".weight")]
lora = sum(16 * sum(hdr[k]["shape"]) for k in hits)   # rank 16: A(16×in) + B(out×16)
print(f"総パラメータ {tot:,} / テンソル {len(hdr)}")
print(f"LoRA が包む線形層 {len(hits)} / 学習する重み {lora:,}（{100*lora/tot:.2f}%）")
PY
```

**出発点の学習条件を読む。**

```bash
python3 -c "import json;d=json.load(open('work/artifacts/model_weights.bak/train_config.json'));print(d['dataset']['repo_id'], d['steps'], d['batch_size'], d['policy']['optimizer_lr'], d['peft'])"
```

**LoRA の当たる場所を上流のソースで確かめる。**

```bash
curl -sL https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/src/lerobot/policies/smolvla/modeling_smolvla.py \
  | grep -n -A 10 "_get_default_peft_targets"
```

**データ量を確かめる。**

```bash
curl -sL https://huggingface.co/datasets/lerobot/libero_plus/raw/f3f49f426d75030177b18778374005bc12ccd588/meta/info.json \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['total_episodes'], d['total_frames'], d['total_tasks'], d['fps'], d['robot_type'])"
```
