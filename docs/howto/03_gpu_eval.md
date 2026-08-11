# 03. GPU マシンで評価を回す

**放っておける。** 端末を閉じても続き、結果はマシン上に残る。

Colab には 3 つの困りごとがあった。**GPU マシンなら 3 つとも消える。**

| 困りごと | GPU マシンでは |
|---|---|
| 見ていないと切れる（学習は 9,100 歩で切れた） | **切れない。** `--detach` で流して寝てよい |
| `/content` は消えるので Drive へ逃がす仕掛けが要る | **消えない。** 退避漏れで結果を失う経路が無い |
| GPU が当たるかどうかが運次第 | 常にある |

## 使い方

**イメージを 2 つ並べて使う。どちらも作り直さない。**

```bash
# 1. まず 5 件だけ。5 分ほどで配線が通っているかが分かる
bash solution/tools/run_docker_eval.sh --check

# 2. 通ったら全件。放っておける形で流す
bash solution/tools/run_docker_eval.sh --detach
tail -f work/gpu_eval/launcher.log
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

# 素の重みで測る（2〜3 時間）
MUJOCO_GL=osmesa WEIGHTS=work/artifacts/model_weights_base LABEL=base \
    bash solution/tools/run_docker_eval.sh --detach

# 学習後の重みで測る（終わってから。さらに 2〜3 時間）
MUJOCO_GL=osmesa LABEL=lora15k \
    bash solution/tools/run_docker_eval.sh --detach
```

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

> **同時に流さないこと。** どちらも GPU を使い、方策サーバーのコンテナ名が
> 同じ（`parc-policy`）である。**片方が終わってから次を流す。**

## 設定

| 変数 | 既定 | 意味 |
|---|---|---|
| `WEIGHTS` | `work/artifacts/model_weights` | **測る重み。** 比べるときはここを変える |
| `LABEL` | （空） | 出力先の名前に足す語 |
| `MAX_STEPS` | `300` | 本番と同じ打ち切り |
| `N_EPISODES` | `1` | **タスクあたり**の本数。84 件あるので 1 のまま |
| `PER_CELL` | `3` | スイート × カテゴリあたり。全体で 84 件になる |
| `RECORD` | `1` | ロールアウトを mp4 に撮る。**失敗した回だけ** |
| `PARC_*` | （空） | 推論のつまみ。空なら `policy_server.py` の既定が効く |

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

| 描画 | 空の場面 1 枚 | LIBERO の場面 | 1 歩 | 84 件 |
|---|---|---|---|---|
| **`osmesa`（CPU）** | 0.8 ms | **動く** | **384 ms** | **2〜3 時間** |
| `egl`（GPU） | 0.2 ms | **最初の数歩で止まる** | — | — |

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

**推論は GPU のままである。** 遅いのは描画だけで、1 歩 384 ms のうち
170 ms ほどが推論（`850 ms / n=5`）、残りが描画である。

**速くする手はスイートを絞ることだけになった。**
1 タスクが 300 歩 × 384 ms = 約 2 分なので、件数がそのまま時間になる。

| スイート | 件数 | 時間 |
|---|---|---|
| 7 カテゴリ | 84 | 2.7 時間 |
| 本命 2 カテゴリ | **36** | **1.2 時間** |

**`MAX_STEPS` は 300 のまま動かさないこと**（本番と同じ値である）。
詳しくは [open_items.md](../talk/open_items.md) の 29。

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
原因は `pipeline/rollout.py` が初期位置を捉える前に 10 歩しか落ち着かせないこと）。

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
| `torch か lerobot が入っていない` | **ホストの venv を見ている。** Docker 版（このページ）を使う |
| **画面が無言のまま進まない** | **50 歩ごとに `[進捗]` が出るようになっている**（8/11 に直した）。それも出ないなら本当に止まっている |
| `[parc] 行が無い` | 古い `policy_server.py` を見ている。`solution/` が正 |
| 方策サーバーが落ちた | `docker logs parc-policy`。重みのパスか GPU メモリ |
| 描画が 8 ms を超える | `--gpus all` と `NVIDIA_DRIVER_CAPABILITIES` |
| 件数が足りない | `summary.txt` の「例外で測れなかった回」 |
| 結果が 1 件も無い | `eval.log` の末尾。CSV の差し替えかパッチを疑う |

## 進み具合の見方

**評価中は 50 歩ごとに 1 行出る。** 全文は `eval.log` に残る。

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
