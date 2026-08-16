# 03. GPU マシンで評価を回す

**放っておける。** 端末を閉じても続き、結果はマシン上に残る。

Colab には 3 つの困りごとがあった。**GPU マシンなら 3 つとも消える。**

| 困りごと | GPU マシンでは |
|---|---|
| 見ていないと切れる（学習は 9,100 ステップで切れた） | **切れない。** `--detach` で流して寝てよい |
| `/content` は消えるので Drive へ逃がす仕掛けが要る | **消えない。** 退避漏れで結果を失う経路が無い |
| GPU が当たるかどうかが運次第 | 常にある |

## 何を測るのか（既定が 36 件に変わった）

**既定で測るのは 36 件である。** [compe/val/EVAL_TASKS.csv](../../compe/val/EVAL_TASKS.csv)
に入っており、**学習が 1 度も引かなかった実演がある指示文だけ**で作ってある。

**以前の 84 件は使わない。** 84 件は学習に使ったタスクを検証にも使っていたため、
**手元の順序が採点環境と逆になった。**

| 重み | 84 件の成功率 | 採点環境の総合スコア |
|---|---|---|
| 素 | 0.476 | **0.1409** |
| LoRA 20,000 ステップ | **0.500**（素より上に見えた） | **0.016** |

理由と作り方は [04_proxy_eval.md](04_proxy_eval.md) にある。
一覧を作り直すのは `solution/tools/make_eval_split.py` である。

| | 件数 | 所要 | いつ使うか |
|---|---|---|---|
| **`EVAL_TASKS.csv`（既定）** | **36** | **約 72 分**（1 件 2 分・8/13 実測） | **重みの採否を判断するとき** |
| 84 件を生成 | 84 | 2 時間 21 分 | **壊れていないかの確認だけ。** `EVAL_CSV=` と空にすると戻る |
| `--check` | 5 | 10 分 | 配線が通っているかを見るとき |

**この 36 件がどういう根拠で選ばれたか**は
[../reference/13_holdout_episodes.md](../reference/13_holdout_episodes.md) にある。
**「選んだ変種そのものが学習に入っていない」とは言えない**点もそこに書いた。

> **一覧は「どの学習に対して未使用か」で決まる。** いまのものは
> `outputs/train2/weak3x_max300/episodes.json`（weak3x の回）を根拠にしている。
> **別の学習の重みを測るなら作り直す**（根拠は `EVAL_TASKS.meta.json` の `record`）。

## 使い方

**イメージを 2 つ並べて使う。どちらも作り直さない。**

```bash
# 1. まず 5 件だけ。5 分ほどで配線が通っているかが分かる
bash solution/tools/run_docker_eval.sh --check

# 2. 通ったら 36 件。放っておける形で流す
bash solution/tools/run_docker_eval.sh --detach
tail -f work/gpu_eval/launcher.log
```

**一覧が無いと、その場で止まる。** 黙って 84 件に落ちることはない。

```
  **測るタスクの一覧がない: .../compe/val/EVAL_TASKS.csv**
  作る: python3 solution/tools/make_eval_split.py --record <episodes.json>
  84 件を生成する道に戻す: EVAL_CSV= bash solution/tools/run_docker_eval.sh
```

事前に 2 つのイメージが要る。**学習に使ったものがそのまま使える。**

```bash
docker build -t parc2026-train solution/train   # 学習で作ってあるはず
docker build -t parc2026 .                      # 配布キット
```

初回だけ `parc2026-train-server` を派生させる（**HTTP サーバーの依存 3 つを足すだけ**。
下の「なぜ派生させるか」）。2 回目以降はキャッシュが効く。

## 2 つの重みを比べる

**本番スコア 0.1409 を出した重みは、こちらで学習したものではない。**
`lerobot/smolvla_libero_plus@7bb70aa5` をそのまま使っていた
（[01_base_model.md](../reference/01_base_model.md)）。**だから失われていない。**
revision を固定して取り直せば同じものが手に入る。

```bash
# 素の重み（0.1409 の構成）を取り寄せる。869 MB
bash solution/tools/fetch_base_weights.sh

# 素の重みで評価する（36 件で約 1 時間）
MUJOCO_GL=osmesa WEIGHTS=work/artifacts/model_weights_base LABEL=base \
    bash solution/tools/run_docker_eval.sh --detach

# 学習後の重みで測る（終わってから。さらに約 1 時間）
MUJOCO_GL=osmesa LABEL=lora20k \
    bash solution/tools/run_docker_eval.sh --detach
```

> **同時に流さないこと**（下の警告と同じ理由）。

### 36 件が判断に使えるかを先に検定する

**この 36 件を信じてよいかは、答えの分かっている重みで確かめる。**
採点環境の結果が 3 つある。

| 重み | 置き場所 | 採点環境の総合スコア |
|---|---|---|
| 素 | `work/artifacts/model_weights_base` | **0.1409** |
| λ=0.25 のブレンド | `work/artifacts/model_weights_lam025` | 0.095 |
| LoRA 20,000 ステップ | `outputs/train2/weak3x_max300/merged_model` ほか | **0.016** |

> **⚠️ この「置き場所」は実行環境によって変わる。`WEIGHTS` に渡す前に、その環境で
> `ls` して実在を確かめること。** 上の `outputs/train2/...` は書いた時点の配置であり、
> 別の GPU マシンでは同じ中身が違う場所にあることがある
> （8/13 に実機で `outputs/` ではなく `work/` 配下だったことがあった）。
>
> **GPU マシンでは大体、次のどちらかに置かれている。**
> - `work/artifacts/model_weights_<label>` — `install_weights.py` で取り込み済み。
>   `vlm/`（SmolVLM の設定とトークナイザ）を含み、そのまま `WEIGHTS` に渡せる
> - `work/train2/<設定>/merged_model` — マージセルが出した生の出力。`vlm/` を
>   **含まない**ため、そのまま `WEIGHTS` に渡すと
>   `エラー: .../vlm が無い` で止まる。先に `install_weights.py` を通すこと

**素と LoRA 20,000 ステップの 2 つで足りる。** この 2 つは採点環境のスコアが大きく違っており、λ=0.25 は素との差が 1 本ぶん（0.046）しかないので判定に使わない。

```bash
python3 solution/tools/compare_with_production.py \
    --pair "work/gpu_eval/<base の回>=outputs/prod_eval_results/260810_評価結果_ema=0.3あり.md" \
    --pair "work/gpu_eval/<lora20k の回>=outputs/prod_eval_results/260811_評価結果_ema=0.3あり_初回学習後.md"
```

**順序が食い違ったら、この 36 件も判断に使えない。** 読み方は
[04_proxy_eval.md](04_proxy_eval.md) にある。

### 1 本目が終わった時点で見る数字

**成功件数だけで、この集合に判別力があるか分かる。**

| 成功件数 | 意味 | 次 |
|---|---|---|
| **10〜25 件** | 天井も床もない | **2 本目（LoRA 20k）へ進む** |
| 0〜2 件 | 難しすぎて全方策が 0 に張り付く | `make_eval_split.py --offset 1` で別の変種にして作り直す |
| 34〜36 件 | 易しすぎて悪化を検出できない | 同じく作り直す |

`summary.txt` の「結末」で**打ち切り 300 ステップに張り付いた本数**も見る。
そこが大半なら、成功件数が中間でも判別力は低い。

> **【8/13・1 本目は通った】素の重みで 21/36（0.583）。判別力はある。**
> 回は `work/gpu_eval/0813_085045_base_n5_n5_ep1_max300_ema03_s1/`
> （`a67a056a`・`n=5` `ema 0.3` `samples=1`・36 件そろい）。
> 打ち切りに張り付いたのは 8 件（22%）だけで、上にも下にも余地がある。
>
> | | 値 |
> |---|---|
> | 成功率 | **0.583**（21/36）95%CI [0.422, 0.729] |
> | ゴール到達 | 0.667（24/36）。**うち 3 件を衝突で失格** |
> | 衝突率 / 空回り率 | 0.194 / **0.083** |
> | 平均ジャーク | **4.304** |
>
> **この 4 つが今日以降のものさしである** ——
> [../plan/04_training_targets.md](../plan/04_training_targets.md) に集めた。
> **84 件で決めた 3 条件の合格ライン（空回り 0.131・ジャーク 4.223）は
> そのまま使えない。** 測った集合が違うので 36 件では 0.083・4.304 になる。
>
> **次は同じ設定値で LoRA 20k を回し、`compare_with_production.py` にかける。**
> **カテゴリ別の表は読まないこと** ——36 件は 1 指示文 1 変種なので、
> `tb` 0.29 / `table` 0.85 はタスクの偏りで説明がつく（04 の 5 節）。

### 設定値は採点時に合わせる

**既定は版 3.4.0 で `n=2` `samples=3` に変わっている。**
比較する採点結果 2 本はどちらも `n=5` `ema 0.3` `samples=1`（`policy_server 0.3.0`）
なので、**明示的に渡さないと重み以外が 2 つ違ってしまう。**

```bash
PARC_N_ACTION_STEPS=5 PARC_SAMPLES=1 PARC_EMA_ALPHA=0.3 …
```

**流し始めたら `server.log` の先頭 1 行を必ず読む。** 値が違えばその回は捨てる。

**重みの取り違えも機械的に潰す。** 素の重みは
`shasum -a 256 work/artifacts/model_weights_base/model.safetensors` が
`a67a056ab0761afa5f5efef8edd45065cef4494735231b7eeeb8328e71fcfb72` である。

**出力先の名前に `LABEL` が入る**ので、どちらの回か後から取り違えない。

```
work/gpu_eval/0811_143022_base_n5_ep1_max300/
work/gpu_eval/0811_161500_lora15k_n5_ep1_max300/
```

比べるのは 3 つである。

| 見るもの | どこに出るか |
|---|---|
| **摂動カテゴリ別の成功率** | `summary.txt`。**どのカテゴリで負けたかが学習の的になる** |
| 結末の内訳 | 同じ。**打ち切りに張り付いた本数**が増えていれば届いていない |
| **失敗した回の映像** | `videos/`。`stalled` か `wandering` かで打つ手が変わる |

```bash
python3 solution/tools/summarize_rollout_videos.py work/gpu_eval/<回>/videos
```

> **Mac では直接動かない。`parc2026` の Docker で回す。**
> mp4 を読むには `cv2` と `numpy` が要るが、**Mac 側のどの python にも入っていない**
> （`work/venv*` も homebrew の python も。`pip` すら無い venv がある）。
> **`parc2026` イメージには両方入っている**
> （`setup.sh` が `opencv-python-headless==4.11.0.86` と `numpy==1.26.4` を入れる）。
> **`ffmpeg` コマンドは要らない** —— opencv の wheel が FFmpeg を同梱している。
>
> ```bash
> docker run --rm -v "$PWD:/w" -w /w parc2026 \
>     python3 solution/tools/summarize_rollout_videos.py outputs/local_eval/<回>/videos
> ```
>
> **`videos/sheets/<動画名>.png` が 1 本 8 コマで出る。** 動画を再生しなくても、
> **ステップ数・グリッパの開閉回数・結末・衝突した物体**がコマに焼かれているので読める。
> 8/13 にこれで「言い換えの回は掴む物を間違えている」ことを確かめた
> （[../talk/260812_briefing.md](../talk/260812_briefing.md) の 5-3 節）。

> **同時に流さないこと。** どちらも GPU を使い、方策サーバーのコンテナ名が
> 同じ（`parc-policy`）である。**片方が終わってから次を流す。**

## 指示文の正規化の精度を測る（意味で寄せる道）

**語の重なりで寄せる分は手元で測れる**（torch が要らない。
[02_build_submission.md](02_build_submission.md) の「指示文の正規化を測る」）。
**測れないのは「意味で寄せる道」だけである** —— 同梱の SmolVLM の言語側を
実際に動かすので、torch と重みが要る。

```bash
docker run --rm --gpus all -v "$PWD:/w" -w /w parc2026-train-server \
    python3 solution/tools/check_instruction_normalization.py --embed --sweep
```

**`parc2026` では動かない。** あのイメージに torch はあるが transformers が無い。
**方策サーバーが動くイメージ（`parc2026-train-server`）で回す**
（下の「なぜイメージが 2 つ要るのか」）。**重みを読むので 1〜2 分かかる。**

**見るのは 1 つだけ。** 「本番の 2 カテゴリで誤って寄せた数」が **0 のまま**で、
**言い換えを寄せた数が 468 種から増える**なら、
`policy_server.py` の `_USE_EMBEDDING` を `True` にする価値がある。
**誤りが増えるなら入れない**（いま解けている回を壊しうる）。

## 設定

| 変数 | 既定 | 意味 |
|---|---|---|
| `WEIGHTS` | `work/artifacts/model_weights` | **測る重み。** 比べるときはここを変える |
| `LABEL` | （空） | 出力先の名前に足す語 |
| **`EVAL_CSV`** | **`compe/val/EVAL_TASKS.csv`** | **測るタスクの一覧（36 件）。** 空にすると 84 件を生成する道に戻る |
| `MAX_STEPS` | `300` | 本番と同じ打ち切り |
| `N_EPISODES` | `1` | **タスクあたり**の本数。36 件あるので 1 のまま |
| `PER_CELL` | `3` | 84 件を生成するときのスイート × カテゴリあたりの本数。**`EVAL_CSV` を空にしたときだけ効く** |
| `CATEGORIES` | （空） | 同上。生成するときだけ効く。下を見る |
| `RECORD` | `1` | ロールアウトを mp4 に撮る。**失敗した回だけ** |
| `PARC_*` | （空） | 推論の設定値。空なら `policy_server.py` の既定が効く |

**一覧を別のものに差し替えるときは `EVAL_CSV` を渡す。**

```bash
# 指示文ごとに 2 件へ増やした一覧で測る
python3 solution/tools/make_eval_split.py --record outputs/train2/episodes.json \
    --per-task 2 --output compe/val/EVAL_TASKS_x2.csv
EVAL_CSV=compe/val/EVAL_TASKS_x2.csv \
    bash solution/tools/run_docker_eval.sh --detach
```

### カテゴリを絞って短く回す（84 件を生成する道でだけ効く）

> **`EVAL_CSV` を空にしたときだけの話である。** 既定の 36 件は
> すでに Background Textures と Light Conditions の 2 カテゴリだけで作ってあり
> （採点環境の摂動がそれだと運営 Q&A No.21 が答えている）、
> **`CATEGORIES` を渡しても効かない。** 36 件から絞るなら
> `make_eval_split.py --categories` で一覧を作り直す。

**1 カテゴリが 12 件である**（`--per-cell 3` × スイート 4 つ。生成器を回して
数えた値で、7 つ全部だと 84 件になる）。**件数がそのまま時間になる。**

```bash
EVAL_CSV= CATEGORIES='Light Conditions,Background Textures' \
    bash solution/tools/run_docker_eval.sh   # 84 件 2 時間 21 分 → 24 件 39 分
```

時間はどちらも実測である。**カテゴリごとの内訳**は下の
「描画は osmesa を使う」の節に表で置いた。
**絞る組み合わせを変えるときは、そこの分数を足す。**

**区切りは読点である。空白では切れない。** `Light Conditions` のように
名前そのものに空白が入っているためで、空白で並べると
「そんなカテゴリは無い」と言ってその場で止まる（黙って件数が減ることはない）。

使える名前は 7 つ。`make_validation_suite.py` の `_DEFAULT_CATEGORIES` にある。

`Background Textures`（机や壁の見た目）・`Camera Viewpoints`（カメラの位置と向き）・
`Language Instructions`（指示文の言い換え）・`Light Conditions`（照明）・
`Objects Layout`（物体の配置と種類）・`Robot Initial States`（ロボットの初期姿勢）・
`Sensor Noise`（観測ノイズ）

## なぜイメージが 2 つ要るのか

**評価側と方策側は依存が正面衝突する。** 1 つに混ぜられない。

| | 必要なもの | イメージ |
|---|---|---|
| 評価パイプライン | `numpy==1.26.4` / `gym==0.25.2` / `robosuite==1.4.0` / CPU torch | `parc2026` |
| 方策 | LeRobot v0.6.0 / **GPU torch** / transformers | `parc2026-train` |

**本番採点も 2 プロセス構成である。** 同じ形をそのまま使う。
Docker のネットワークを 1 つ作って繋ぐので、HTTP は
コンテナ間で完結する。

### なぜ学習イメージから派生させるか

**`parc2026-train` には HTTP サーバーの依存が入っていない。**
LeRobot が `fastapi` を宣言しているのは `phone` extra（遠隔操作用）だけで、
`[training,smolvla,peft]` では入らない。`msgpack` と `uvicorn` も同様である。

提出物の `requirements.txt` が 3 つを「必須」に挙げているのはこのためで、
**採点環境では venv に入るが、学習イメージには無い。**

いまは 1 行の派生で足している。

```dockerfile
FROM parc2026-train
RUN pip install --no-cache-dir "fastapi>=0.68" "uvicorn>=0.15" "msgpack>=1.0"
```

> **学習用と評価用のイメージを分けるのは、それ自体は筋がよい。**
> 学習イメージは `wandb` や `datasets` を抱えており、推論には要らない。
> ただし**差は 3 パッケージしかない**ので、いまは派生で足りている。
> 分けるなら [open_items.md](../talk/open_items.md) の 19 として置いた。

## 描画は osmesa を使う（このマシンで EGL は原理的に使えない）

**このマシン（H100-20C）では EGL で止まる。** 8/11 に確かめた。

| 描画 | 空の場面 1 枚 | LIBERO の場面 | 1 ステップ | 84 件を流したとき |
|---|---|---|---|---|
| **`osmesa`（CPU）** | 0.8 ms | **動く** | **384 ms** | **2〜3 時間** |
| `egl`（GPU） | 0.2 ms | **最初の数歩で止まる** | — | — |

**右端は 84 件で測ったときの値である**（既定の 36 件なら約 1 時間）。

**原因が判明した。この vGPU は計算専用の製品である。**

```
$ nvidia-smi -q | grep -iA3 -e virtualization -e vgpu
    Virtualization Mode  : VGPU
    Product Name         : NVIDIA Virtual Compute Server
```

**`NVIDIA Virtual Compute Server`（vCS）は OpenGL / Vulkan に対応しない。**
描画が使えるのは vWS（Virtual Workstation）系だけである。
**こちら側では直せない。** `osmesa` を使う。

```bash
MUJOCO_GL=osmesa bash solution/tools/run_docker_eval.sh --check
```

> **速さの測定を「動く証拠」にしてはいけない。**
> 最初はこう書いていた ——「描画を 50 枚測り、8 ms を超えたら止める」。
> **EGL はその測定を 0.2 ms で通ったうえで、実際の場面で止まった。**
> **いまなら 0.2 ms 自体が不自然だと分かる**（GPU 描画でも速すぎる）。
> あれは描けていたのではなく、実質何も描いていなかった。
>
> しかも `os.environ.setdefault("MUJOCO_GL", "egl")` と書いていたため、
> **osmesa を指定した回にも「GPU 描画で動いている」と表示していた。**
> いまは実際に使っている方式を出す。

**推論は GPU のままである。** 遅いのは描画だけで、1 ステップ 384 ms のうち
170 ms ほどが推論（`850 ms / n=5`）、残りが描画である。

**速くする手はスイートを絞ることだけになった。**
**件数がそのまま時間になる。** 下は見積もりではなく、84 件を 1 回流した
ログから測った値である（下の「かかった時間を自分で測る」でやり方を出す）。

| スイート | 件数 | 実測 | 指定 |
|---|---|---|---|
| **`EVAL_TASKS.csv`（既定）** | **36** | **約 1 時間**（1 件 1.67 分から算出） | 何も渡さない |
| 7 カテゴリを生成 | 84 | **2 時間 21 分** | `EVAL_CSV=` |
| 本命 2 カテゴリを生成 | **24** | **39 分** | `EVAL_CSV= CATEGORIES='Light Conditions,Background Textures'` |
| 先頭 5 件（`--check`） | 5 | 10 分 | `--check` |

**36 件の「約 1 時間」だけは実測ではない。** 1 件 1.67 分（84 件の実測から）を
かけた値である。**1 回流したら実測に書き換えること。**

**1 件あたり 1.67 分。カテゴリによって 1.2〜2.1 分の幅がある。**
**打ち切りに当たった回が長い**（84 件のうち 38 件が 300 ステップまで走った。
成功した 40 件は中央 152 ステップで終わっている）。絞ったときの時間は下の表で足す。

| カテゴリ | 件数 | 合計 | 1 件 | ステップ数の中央 |
|---|---|---|---|---|
| `Language Instructions` | 12 | 25.0 分 | 2.09 分 | 300 |
| `Light Conditions` | 12 | 22.6 分 | 1.88 分 | 140 |
| `Robot Initial States` | 12 | 20.9 分 | 1.74 分 | 300 |
| `Objects Layout` | 12 | 20.6 分 | 1.71 分 | 300 |
| `Sensor Noise` | 12 | 19.1 分 | 1.59 分 | 208 |
| `Background Textures` | 12 | 16.5 分 | 1.38 分 | 200 |
| `Camera Viewpoints` | 12 | 14.6 分 | 1.22 分 | 157 |

> **出典は `outputs/0811_101824_base_n5_ep1_max300/` の `eval.log`**
> （素の重み・84 件・`MAX_STEPS=300`）。`=== タスク` 行と `[結果]` 行の
> 時刻の差を取り、隣の `VAL_TASKS.csv` の `category` 列で束ねた。
> **重みが変われば時間も変わる**（打ち切りに当たる本数が変わるため）。
>
> **件数は生成器を回して数えた**（`--per-cell 3` で 1 カテゴリ 12 件）。
> **前に「2 カテゴリ＝36 件」と書いてあったのは誤りである。**

**`MAX_STEPS` は 300 のまま動かさないこと**（本番と同じ値である）。
詳しくは [open_items.md](../talk/open_items.md) の 29。

### かかった時間を自分で測る

```bash
run=outputs/0811_101824_base_n5_ep1_max300      # 自分の回に読み替える
grep '\[結果\]' "$run/eval.log" | sed -n '1p;$p' | cut -c1-19
```

**最初の `[結果]` ではなく最初の `=== タスク` から数えること。**
1 件目の所要（約 2 分）が抜け落ちる。**方策サーバーが重みを読む 1〜2 分は
この外側である**（`server.log` を見る）。

## 集計が出すもの

**6 つ。どれも「次に何をするか」が変わる情報である。**

| 見出し | なぜ要るか |
|---|---|
| **件数** | **途中で落ちても終了コードは 0 になる。** 全件そろったかを最初に見る |
| 結末 | 成功 / 衝突 / 未完了。**ゴール到達後に衝突で落ちた本数**を分けて出す |
| **摂動カテゴリ別** | **学習の的を決めるのはこの表。** 弱い順に並ぶ |
| **衝突の切り分け** | **「方策の衝突率」と「環境を含む衝突率」を並べて出す**（下） |
| ぶつけた物体 | 偏りが出る。同じ変位が 3 回以上出たら回数を示す |
| 録画 | 撮れた本数と、見るためのコマンド |

### 衝突は 2 つの率で出る

8/10 に、摂動が違う 6 回で `ketchup_1=544.6mm` が 0.1 mm も違わずに出た。
**方策の振る舞いではなく、環境側で最初から動いていた**（瓶が空から落ちてくる。
原因は `pipeline/rollout.py` が初期位置を捉える前に 10 ステップしか落ち着かせないこと）。

**そのままだと衝突率が 2.25 倍に水増しされる。** 8/10 の 25 件では 0.360 が
実際には 0.160 だった。

| 出る数字 | 何に使うか |
|---|---|
| **方策の衝突率** | **衝突対策の効き目を測る。** 環境由来を外した分 |
| 環境を含む衝突率 | **本番のスコアに乗るのはこちら。** 採点側のコードなので直せない |

### 2 回を並べて比べる

**84 件の報告を 2 つ見比べると見落とす。** 差だけを出す。

```bash
python3 solution/tools/summarize_gpu_eval.py \
    --output-dir work/gpu_eval/0811_..._lora \
    --compare    work/gpu_eval/0811_..._base
```

**出せなかった行は名指しされる。** 行が無いのを「差が無い」と読み違えるのが
いちばん危ない（滑らかさの指標が消えると、**本番の悪化 8.8 倍のうち 4.4 倍**を
見落とす）。

> **本番のスコアは手元で計算できない。** 配点は `total_score_config.json` と
> `normalization_config.json` にあるが、**どちらも配布されていない。**
> キットの `overall_score` は成功率そのものである。
> **比べられるのは成功率と滑らかさの指標までで、順位は予測できない。**

## 落ちたときに見る順番

| 症状 | 見るところ |
|---|---|
| **`エラー: 重みが無い` / `vlm が無い` が出て、出力先も作られない** | **`merged_model/` を `WEIGHTS` に渡した。** あれには `vlm/` が入らない（`install_weights.py` が素から引き継ぐ設計）。0/5 は出力先を作る前に止まるので、**ディレクトリが無いのは正常な止まり方である。** 直し方は [../plan/04_training_targets.md](../plan/04_training_targets.md) の 7 節。8/13 に踏んだ |
| **`84 タスクを読み込み` と出る** | **このマシンのリポジトリが古い。** `git pull` する。8/13 に踏んだ ——「渡された一覧を使う」が出ない版だと黙って 84 件を生成する。タスク名に `_view_`・`_language_`・`_noise_`・`_add_` が現れるのも同じ合図（36 件は `_table_`・`_tb_`・`_light_` だけ） |
| **`確かめた行: 36 / 落ちる行: 36 —— 初期状態が無い`** | **CSV に `init_states_file` 列が無い。** `make_eval_split.py` を新しくして作り直す。8/13 に踏んだ ——`register.py` の `_SUFFIX_RE` は `_light_*` と `_table_\d+` しか剥がさないので、**`_tb_N` は必ず落ちる** |
| **方策サーバーが消えた** | **台本が途中で終わった症状である**（`trap cleanup EXIT`）。原因は `launcher.log` にある。**止めるときも評価側のコンテナを止める** ——`parc-policy` を先に落とすとパイプラインが例外を握って終了コード 0 で終わり、中途の結果が残る |
| `torch か lerobot が入っていない` | **ホストの venv を見ている。** Docker 版（このページ）を使う |
| **画面が無言のまま進まない** | **50 ステップごとに `[進捗]` が出るようになっている**（8/11 に直した）。それも出ないなら本当に止まっている |
| `[parc] 行が無い` | 古い `policy_server.py` を見ている。`solution/` が正 |
| 描画が 8 ms を超える | `--gpus all` と `NVIDIA_DRIVER_CAPABILITIES` |
| 件数が足りない | `summary.txt` の「例外で測れなかった回」 |
| 結果が 1 件も無い | `eval.log` の末尾。CSV の差し替えかパッチを疑う |
| 出力先のファイルが消せない | **持ち主が root になっている。** 台本の最後の `chown` まで進んでいない。`sudo rm -rf` か、`docker run --rm -v "$PWD/work/gpu_eval:/g" --entrypoint chown parc2026 -R "$(id -u):$(id -g)" /g` |

## 進み具合の見方

**評価中は 50 ステップごとに 1 行出る。** 全文は `eval.log` に残る。

```
=== タスク 3/84: pick_up_the_ketchup_..._light_49 ===
[進捗] pick_up_the_ketchup_..._light_49: 50/300 steps (7.4s)
[結果] pick_up_the_ketchup_..._light_49 ep0: 衝突 137 steps  ...
```

**画面へ出すのは進み具合・結末・エラーだけ**にしてある。全部流すと
1 タスクで 300 行を超え、読めなくなる。

> **EGL の行が描画の測定のあとに出ることがある。**
> python が終わるときの後片付けで出るもので、`--- 描画の測定はここまで ---`
> より下に現れる。**握りつぶしていない** — この環境に何を足したかは
> 手元から見えないので、本当の不具合を隠す側を選ばない。

> **ホストで直接回す道**（`run_gpu_eval.sh`）も残してある。
> LeRobot と GPU torch の入った venv がホストにあるなら、そちらが速い。
> **学習を Docker でやった場合はホストに venv が無いので、このページを使う。**
