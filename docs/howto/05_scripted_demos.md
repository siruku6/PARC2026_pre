# 05. 手本を台本で作る

**Colab は要らない。** 必要な環境は 2 つとも既にある。

> **撮影（`record`）に GPU は要らない。** 物理と描画だけなので、**手元の Mac の
> Docker で回せる**（8/13 の 50 本はそうやって撮った）。GPU が要るのは学習
> （6 節）だけで、**撮影と学習は別のマシンで同時に進められる。**
>
> **どちらのマシンでも描画は CPU である**（2 節）。GPU マシンの vGPU は
> OpenGL に対応しない。速くする手は CPU の並列だけ（3 節）。

| イメージ | 中身 | 何を回すか |
|---|---|---|
| `parc2026` | Python 3.10 ＋ mujoco / robosuite / LIBERO-plus | `measure` / `verify` / **`record`** |
| `parc2026-train` | **Python 3.12 ＋ lerobot v0.6.0** | **`build` / `merge`** ＋ 学習 |

**2 つに分かれているのは Python の版が違うからである。** lerobot は
PEP 695 の型構文（`def f[T: X](...)`）を使うので 3.12 が要り、キットは
採点環境に合わせて 3.10 に固定されている。だから間に npz を挟む
（設計の理由は [../strategy/06_scripted_demos.md](../strategy/06_scripted_demos.md)）。

## 先に用意するもの

| | 確かめ方 |
|---|---|
| Docker | `docker run --rm parc2026 python -c "import robosuite"` |
| ディスク | 学習データ 16 GB ＋ 重み ＋ 生成物（1 本 85 MB） |

**`record` に GPU は要らない。** 物理と描画だけで、方策を動かさないためである。
そして**どちらのマシンでも描画に GPU が使えない**（下の 2 節）。GPU が要るのは
学習（6 節）だけである。

| 段 | 回す場所 | なぜ |
|---|---|---|
| `record` / `check` | **手元の Mac の Docker でよい** | CPU しか使わない。GPU マシンを空けておける |
| `build` / `merge` / 学習 | GPU マシン | 学習が GPU を使う。`build` 自体は CPU でもよい |

**データセットは 1 本 1.4 MB しかない**ので、Mac で撮って GPU マシンへ送るのが速い。

```bash
rsync -a work/scripted/lerobot/ <GPU マシン>:~/src/PARC2026_pre/work/scripted/lerobot/
```

## 1. イメージを 2 つ作る

```bash
docker build -t parc2026 .                      # 配布キット（10〜20 分）
docker build -t parc2026-train solution/train   # 学習側。学習で作ってあるはず
```

## 2. 描画は CPU（osmesa）を使う

**どちらのマシンでも EGL（GPU 描画）は使えない。** GPU マシンは vGPU が
`NVIDIA Virtual Compute Server` で OpenGL / Vulkan に対応せず（経緯と
`nvidia-smi` の出力は [03_gpu_eval.md](03_gpu_eval.md) の「描画は osmesa を
使う」）、手元の Mac の Docker は VM の中なので GPU がそもそも見えない。

| 描画 | 空の場面 1 枚 | LIBERO の場面 |
|---|---|---|
| **`osmesa`（CPU）** | 0.8 ms | **動く** |
| `egl`（GPU） | 0.2〜0.4 ms | **最初の数歩で止まる** |

> **速さを測って「動く証拠」にしてはいけない。** EGL は 1 枚 0.4 ms という
> 良い値を出したうえで、実際の場面で止まる。**その速さ自体が「何も描いて
> いない」印である。** 2026-08-13 にこの罠を 2 度目に踏んだ —— 同じ測定を
> 書き直して「GPU を掴んでいる」と判断した。**確認は本物の場面を 1 本
> 通すことでしか取れない。**

## 3. 手本を記録する

```bash
mkdir -p work/scripted
docker run --rm \
    -e MUJOCO_GL=osmesa \
    -v "$PWD/solution/tools:/host/tools:ro" \
    -v "$PWD/work/scripted:/out" \
    parc2026 python /host/tools/scripted_demo.py record \
    --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \
    --episodes 0 --output /out/base

# Linux のホストでは root で書かれるので、持ち主を戻す
# **手元の Mac（colima）では要らない**——共有の仕組みが自分のものとして見せる
docker run --rm -v "$PWD/work/scripted:/out" --entrypoint chown parc2026 \
    -R "$(id -u):$(id -g)" /out
```

**`--user` と `HOME` は指定しない。** LIBERO は import されるとき
`$HOME/.libero/config.yaml` を探し、**無ければ `input()` で聞いてくる**
（`libero/libero/__init__.py`）。バッチでは標準入力が無いので
`EOFError: EOF when reading a line` で落ちる。この設定はキットの
`setup.sh` がビルド時に `/root/.libero/` へ書いており、`/root` は他の
利用者から読めない。**root で走らせて、あとで `chown` するのが素直である**
（評価側の [run_docker_eval.sh](../../solution/tools/run_docker_eval.sh) も同じ形）。

### 並列にできるが、コア数が上限である

**評価は並列にできない**（方策サーバーと GPU が 1 台ずつ）。**しかし `record` は
どちらも要らない**ので、CPU のコア数まで並べられる。

**上限は Docker が見ているコア数である。** 手元の Mac では VM に割り当てた数で
決まり（colima の既定は **2**）、Mac 本体のコア数は関係ない。GPU マシンでも
「GPU があるからコアも多い」は成り立たない。**必ず数えてから `N` を決めること。**

```bash
docker run --rm parc2026 nproc                     # 撮影に使う側で数える
docker info --format 'CPUs={{.NCPU}} Mem={{.MemTotal}}'
# 手元の Mac（colima）は CPUs=2 / Mem=10405670912（9.69 GiB）だった
```

**Mac で撮るなら colima の割り当てを増やすのが一番効く。** `colima stop &&
colima start --cpu 6 --memory 12` のように取り直せば、そのぶん並列にできる。

コア数より多く並べても総量は増えず、奪い合いとメモリ圧迫で遅くなる。
**1 個あたり 1.25 GB 使う**（実測。画像の置き場 118 MB ＋ mujoco の描画用
バッファ ＋ 実行時）ので、メモリからも上限が決まる。

> **2 コア・9.69 GB の VM（手元の Mac の colima）で 8 並列にして実際に壊した。** MEM は 8.7 GB
> まで埋まり、**CPU 合計は 200% のうち 78% しか出ず**、1 個あたり
> **BLOCK I/O が 65 GB**（swap を叩いている印）、1 個は OOM で消えた。
> **CPU% が上限に届かず BLOCK I/O が異常に多いときは、並べすぎである。**

```bash
N=2                                    # nproc の値。**大きくしても速くならない**
for i in $(seq 0 $((N-1))); do
    eps=$(seq "$i" "$N" 49 | paste -sd,)
    docker run -d --name "rec$i" \
        -e MUJOCO_GL=osmesa -e LP_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
        -v "$PWD/solution/tools:/host/tools:ro" \
        -v "$PWD/work/scripted:/out" \
        parc2026 python /host/tools/scripted_demo.py record \
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \
        --episodes "$eps" --output /out/base
done
docker logs -f rec0
```

**`LP_NUM_THREADS=1` と `OMP_NUM_THREADS=1` が要る。** osmesa（llvmpipe）は
内部で勝手に複数スレッドを使うので、指定しないとコンテナ同士がコアを
奪い合って並列の意味が薄れる。

| | 直列 | 2 並列（2 コアでの上限） |
|---|---|---|
| 1 本 | 111 秒 | **172 秒**（実測） |
| 50 本 | 約 92 分 | **約 72 分**（実測から） |

**2 コアで 2 並列にしても 1.3 倍しか稼げない**（92 分 → 72 分）。1 本の撮影が
1 コアを使い切れておらず、並べると 1 本ずつが 1.5 倍遅くなるためである。
**2 倍を見込んで予定を立てないこと。** 大きく稼ぐならコアを増やす（上の
`colima start --cpu`）か、撮影と学習を別のマシンに分ける。

メモリは 1 本あたり **1.25 GB**（実測）。**2 並列で 2.5 GB。**

**`--episodes` を省くと初期配置 50 通り全部を回す。** 変種を変えるときは
`--variant` と `--output` を差し替える（**物理は同じなので同じ台本が通る**）。

## 4. 形が合っているかを確かめる（lerobot 不要）

**変換の前に必ず通す。** ここで止まれば、学習側へ持って行く前に気づける。

```bash
docker run --rm -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    parc2026 python /host/tools/npz_to_lerobot.py check --npz-dir /data/base
```

混ぜる相手の宣言と突き合わせるなら `--against <libero_plus の meta/info.json>` を足す。
**食い違ったら 0 以外で終わる。**

## 5. 学習データの形式に変換する

**ここから Python 3.12 側。** 道具はイメージに入っていないのでマウントする
（`solution/train/Dockerfile` は `train.py` と `merge.py` だけを COPY する）。

**`--gpus all` を付けない。** `build` も `merge` も GPU を使わないので、
**手元の Mac でもそのまま回る**（付けると colima では
`could not select device driver` で落ちる）。

```bash
# 作った分だけで 1 つのデータセットにする
docker run --rm \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    parc2026-train python /host/tools/npz_to_lerobot.py build \
    --npz-dir /data/base --output /data/lerobot --repo-id local/scripted_moka

# 既存の学習データに混ぜる（提出候補を作るときはこちら）
docker run --rm \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /host/tools/npz_to_lerobot.py merge \
    --base /root/.cache/huggingface/lerobot/lerobot/libero_plus \
    --add /data/lerobot --output /data/merged
```

**`merge` は 16 GB を複製する。** ディスクを先に確認すること。

**書き出したら、学習を流す前に読ませて確かめる。** ファイルが揃っていることと
学習が読めることは別で、**動画の符号化が壊れていれば読み出しで初めて落ちる**
（LeRobot の自動フォールバックは torchcodec が「入っているか」しか見ない）。

```bash
docker run --rm \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    parc2026-train python /host/tools/npz_to_lerobot.py inspect \
    --root /data/lerobot --repo-id local/scripted_moka
```

**1 コマ目を実際に読んで形を出す。** ここが通れば学習も読める。

## 6. 学習する（GPU マシン）

**撮影を待たなくてよい。** 撮影は CPU だけなので、**別のマシンで並行して学習を
回せる。** データセットを送ってから始める（1 本 1.4 MB）。

```bash
rsync -a work/scripted/lerobot/ <GPU マシン>:~/src/PARC2026_pre/work/scripted/lerobot/
```

**まず 10 歩だけ回して、起動するかを見る。** 既定の引数では番号選びを通らない
ので学習データのメタデータを取りに行かず、要るのは出発点の重み 869 MB だけである
（一度学習していれば cache にある）。**8/13 に 1 本のデータセットで通した**
——`dataset.num_frames=233` / `num_episodes=1` を読み、10 歩を 10 秒、
学習する重みは 743K（LoRA）。

```bash
docker run --rm --gpus all \
    -v "$PWD/work/train:/out" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /workspace/train.py \
    --run-name scripted_smoke --steps 10 --save-freq 10 -- \
    --dataset.repo_id=local/scripted_moka --dataset.root=/data/lerobot
```

通ったら本番の歩数で回す。`train.py` は `--` の後ろを**最後に**付けるので、
データセットの差し替えは**コード変更なしで通る**（同じ設定を 2 度渡すと後ろが勝つ）。

```bash
docker run --gpus all --rm \
    -v "$PWD/work/train:/out" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /workspace/train.py --run-name scripted_only -- \
    --dataset.repo_id=local/scripted_moka --dataset.root=/data/lerobot
```

**2 つ気をつける。**

**混ぜたデータで絞り込みも掛けるときは `--dataset-root` を使う**（8/13 に追加。
`train.py`）。`--` の後ろで `--dataset.root=` を渡すだけでは**番号選びには効かない**
——`--emphasize-tasks` / `--max-demo-steps` / `--episodes-per-task` は
`lerobot/libero_plus` のメタデータを Hub から落として番号を決めるので、
**足した分が 1 本も選ばれないまま、エラーも出さずに学習が始まる。**

```bash
docker run --gpus all --rm -v "$PWD/work/train:/out" -v "$PWD/work/merged:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /workspace/train.py --run-name scripted_mix \
    --dataset-root /data --max-demo-steps 300 --emphasis-multiplier 3 \
    --emphasize-tasks "put both moka pots on the stove" ...
```

**渡すと本数とタスク数が印字される。** 混ぜ損ねはそこでしか気付けない
（14,347 ＋ 45 = **14,392 本**と出るのが正しい）。

| 気をつけること | なぜ |
|---|---|
| **`--emphasize-tasks` に台本のタスクを入れる** | 入れないと**得意タスクの間引き（27%）を台本も受けて 45 本のうち 12 本しか使われない**（[../talk/260813_today.md](../talk/260813_today.md)） |
| **`--dataset.revision` は上書き不要**（8/13 実測） | 固定値が渡ったままでも `root` のローカルを読む。`revision` を見に行くのは Hub から取るときだけである |
| **正規化の統計値は渡したデータセットのものに差し替わる** | 下記 |

**出発点の重みが持っている統計値は使われない。** lerobot の学習は
`normalizer_processor` の `stats` を `dataset.meta.stats` で上書きし
（`lerobot_train.py` の `preprocessor_overrides`）、その中身は
**`<root>/meta/stats.json`** である（`datasets/dataset_metadata.py:184` の
`load_stats(self.root)`）。**`repo_id` は関係ない。**

| | 影響 |
|---|---|
| ① 作った分だけ | 基準が 1 タスク 45 本から計算し直される。**同じタスクの中では筋が通るが、他タスクの評価には使えない**（忘却とは別の理由でも壊れる） |
| ② 混ぜる | `merge_datasets` が**エピソードごとの統計を足し合わせる**（`aggregate_stats`）。14,347 本に 45 本なので**ずれは 0.3% 相当** |

**作った分だけで学習したものは提出できない。** 1 タスクぶんの手本しか無いので
残り 39 タスクの性能が落ちる（8/11 の追加学習は本番 0.1409 → 0.016 に崩れた）。
`build` は「台本を真似られるか」を見るため、`merge` が提出候補を作るためのものである。

## 7. 評価する — **既定の 36 件では測れない**

**このタスクは 36 件に入っていない**（146 本すべてが 341 歩以上で、
`--max-demo-steps 300` に全部落ちるため）。**このタスクの変種だけを集めた
一覧が `compe/val/MOKA_TASKS.csv`（8 件）にある**ので、それを渡す
（作り直すときは `solution/tools/make_task_suite.py`）。

```bash
EVAL_CSV=compe/val/MOKA_TASKS.csv MUJOCO_GL=osmesa \
    WEIGHTS="$PWD/work/train/<run 名>/merged_model" LABEL=<run 名> \
    bash solution/tools/run_docker_eval.sh --detach
```

手順の詳細は [03_gpu_eval.md](03_gpu_eval.md) にある。

> **8 件は照明 4 種と背景 4 種で、素の見た目は入っていない。** 一方
> **台本の 45 本は素の見た目だけで撮った。** つまりこの 8 件は
> 「動きを学べたか」と「見た目の変化を越えられるか」を**同時に**問う。
> **0/8 だったときに原因が分かれない**ので、そのときは素の 1 件
> （`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`・接尾辞なし）を
> 診断用に回す。**ただし初期配置が学習と重なるので、成功しても
> 「覚えた」までしか言えない。**
>
> 変種でも撮れば根本から消える（**物理は同じなので台本は変えない**。
> `--variant` と `--output` を差し替えるだけ。手元の Mac で 45 本 72 分）。

## 時間の見積もり

**下の実測はすべて手元の Mac の Docker（colima・2 コア / 9.69 GiB）である。**
GPU マシンで撮る場合は `nproc` から測り直すこと。

| 段 | 直列 | 2 並列 |
|---|---|---|
| `verify` 50 通り（描画なし） | 153 秒 | — |
| `record` 1 本 | **111 秒**（233 歩） | **172 秒** |
| `record` 50 本 | 約 92 分 | **約 72 分**（実測。45 本が成功） |
| `record` 変種 4 種で 200 本 | 6.2 時間 | **約 4.8 時間** |
| `build` 45 本 | **10 分 31 秒**（実測・**14 秒/本**） | — |

**`build` は `record` の 1/12 の時間で済む**（1 本 14 秒 対 172 秒）。
出来上がりは **45 本で 64 MB**（front 41.2 MB ＋ wrist 23.6 MB）で、
元の npz 3.8 GB の **1.7%** である。

**`record` の中身は描画が支配的である。** 実測 111 秒 / 233 歩 = 1 歩 476 ms で、
うち物理は 39 ms しかない。**GPU では描けないので、速くする手は並列だけである。**
そして**2 コアでの並列は 1.3 倍しか稼げない**（上の 92 分 → 72 分）ので、
大きく縮めたければコアを増やすしかない。

> **npz は圧縮していない。** 動画に再エンコードするだけの中継物で、画像
> 118 MB を deflate しても 1.7 倍しか縮まないのに 1 本 4.3 秒かかる。
> 1 本 85 MB になるので、50 本で 4.3 GB を見ておく。
>
> **変換後は 1 本 1.4 MB になる**（AV1・front 0.9 MB ＋ wrist 0.5 MB）。
> 50 本で 70 MB なので、**npz は変換したら消してよい。**

## 詰まったときの見どころ

| 症状 | 原因 |
|---|---|
| `EOFError: EOF when reading a line` | `$HOME/.libero/config.yaml` が見えていない。**`--user` と `HOME` を外す** |
| 場面の描画で止まる | `MUJOCO_GL=egl` になっている。**このマシンでは使えない。** `osmesa` にする |
| 並列にしても速くならない | `LP_NUM_THREADS=1` を渡していない、または **`N` が Docker の見ているコア数を超えている**（`nproc` で数える。手元の Mac は 2） |
| `docker stats` の CPU% が上限に届かない／BLOCK I/O が数十 GB | **並べすぎで swap を叩いている。** `N` を下げる |
| コンテナが消える（`Status` が 137） | OOM で殺された。**`N` × 1.25 GB が実装メモリに収まるか**を確かめる |
| 出力を消せない | root で書かれている。上の `chown` を回す |
| `ModuleNotFoundError: scripted_demo` | 標準入力から流し込むと `__file__` が使えない。`PYTHONPATH` に道具の置き場を入れる |
