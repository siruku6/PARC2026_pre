# 05. 手本を台本で作る

**Colab は要らない。** 必要な環境は 2 つとも既にある。

## どの節から始めるか

**上から順に読む文書ではない。** やることで入り口が変わる。

| やること | 入り口 | 8/13 時点 |
|---|---|---|
| `put both moka pots on the stove` の手本を**混ぜて学習する** | **8 節** | **ここから。**1〜5 は済み |
| 別のタスクの台本を書く | 1 節（環境）→ **[../strategy/06](../strategy/06_scripted_demos.md) の「作り方」** | 中継点を差し替える |
| 撮り直す・変種で本数を増やす | **3 節**（`record`） | 45 本ある |
| npz をデータセットにする | **5 節**（`build`） | 済み（45 本 / 10,339 コマ） |
| 作った分だけで学習する | 6 節 | **やっても測れない**（6 節の「基準が潰れる」） |

**`merge`（混ぜる）は 8 節にしかない。** 5 節は `build` までである。

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

| 手 | 回す場所 | なぜ |
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
    parc2026 python /host/tools/scripted_demo/scripted_demo.py record \
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
（評価側の [run_docker_eval.sh](../../solution/tools/eval/run_docker_eval.sh) も同じ形）。

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
        parc2026 python /host/tools/scripted_demo/scripted_demo.py record \
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
    parc2026 python /host/tools/scripted_demo/npz_to_lerobot.py check --npz-dir /data/base
```

混ぜる相手の宣言と突き合わせるなら `--against <libero_plus の meta/info.json>` を足す。
**食い違ったら 0 以外で終わる。**

## 5. 学習データの形式に変換する（`build`）

**ここから Python 3.12 側。** 道具はイメージに入っていないのでマウントする
（`solution/train/Dockerfile` は `train.py` と `merge.py` だけを COPY する）。

**`--gpus all` を付けない。** `build` は GPU を使わないので**手元の Mac でも
そのまま回る**（付けると colima では `could not select device driver` で落ちる）。

```bash
docker run --rm \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    parc2026-train python /host/tools/scripted_demo/npz_to_lerobot.py build \
    --npz-dir /data/base --output /data/lerobot --repo-id local/scripted_moka
```

**混ぜる（`merge`）はこの節にはない。8 節にまとめてある。** 16 GB の複製・
持ち主の戻し・関門 A の確認まで通しで要るので、途中だけを真似ると必ず抜ける。

**書き出したら、学習を流す前に読ませて確かめる。** ファイルが揃っていることと
学習が読めることは別で、**動画の符号化が壊れていれば読み出しで初めて落ちる**
（LeRobot の自動フォールバックは torchcodec が「入っているか」しか見ない）。

```bash
docker run --rm \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    parc2026-train python /host/tools/scripted_demo/npz_to_lerobot.py inspect \
    --root /data/lerobot --repo-id local/scripted_moka
```

**1 コマ目を実際に読んで形を出す。** ここが通れば学習も読める。

## 6. 学習する（GPU マシン）

**撮影を待たなくてよい。** 撮影は CPU だけなので、**別のマシンで並行して学習を
回せる。** データセットを送ってから始める（1 本 1.4 MB）。

```bash
rsync -a work/scripted/lerobot/ <GPU マシン>:~/src/PARC2026_pre/work/scripted/lerobot/
```

**まず 10 ステップだけ回して、起動するかを見る。** 既定の引数では番号選びを通らない
ので学習データのメタデータを取りに行かず、要るのは出発点の重み 869 MB だけである
（一度学習していれば cache にある）。**8/13 に 1 本のデータセットで通した**
——`dataset.num_frames=233` / `num_episodes=1` を読み、10 ステップを 10 秒、
学習する重みは 743K（LoRA）。

```bash
docker run --rm --gpus all \
    -v "$PWD/work/train:/out" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /workspace/train.py \
    --run-name scripted_smoke --steps 10 --save-freq 10 -- \
    --dataset.repo_id=local/scripted_moka --dataset.root=/data/lerobot
```

通ったら本番のステップ数で回す。`train.py` は `--` の後ろを**最後に**付けるので、
データセットの差し替えは**コード変更なしで通る**（同じ設定を 2 度渡すと後ろが勝つ）。

```bash
docker run --gpus all --rm \
    -v "$PWD/work/train:/out" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /workspace/train.py --run-name scripted_only -- \
    --dataset.repo_id=local/scripted_moka --dataset.root=/data/lerobot
```

**2 つ気をつける。**

**混ぜたデータで絞り込みもするときは `--dataset-root` を使う**（8/13 に追加。
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
| ① 作った分だけ | **基準が潰れる。** 1 タスクを 1 つの台本で解くので、`observation.state` の std が 4 次元でほぼ 0（素の 0.00〜0.10 倍）、姿勢の指令は std が **0**。**わずかなずれが何十倍に拡大されて方策へ渡り、戻れなくなる**（8/13 に 0/10。[../strategy/06](../strategy/06_scripted_demos.md) に数字） |
| ② 混ぜる | `merge_datasets` が**エピソードごとの統計を足し合わせる**（`aggregate_stats`）。14,347 本に 45 本なので**ずれは 0.3% 相当。問題は起きない** |

**だから「作った分だけで学習して測る」は台本の良し悪しを測れない。**
測るなら `meta/stats.json` の基準を素の重みのものに差し替えてから回すこと。

**作った分だけで学習したものは提出できない。** 1 タスクぶんの手本しか無いので
残り 39 タスクの性能が落ちる（8/11 の追加学習は本番 0.1409 → 0.016 に崩れた）。
`build` は「台本を真似られるか」を見るため、`merge` が提出候補を作るためのものである。

### 学習した重みを評価に渡すまでに 2 つ手当てが要る

**`merge.py` の出力はそのままでは評価に渡せない。** 2 つとも `PermissionError`
や「`vlm` が無い」で止まるので、先に済ませる。

```bash
# ① 持ち主を戻す（Linux のホストでは root で書かれる）
docker run --rm -v "$PWD/work/train:/out" --entrypoint chown parc2026-train \
    -R "$(id -u):$(id -g)" /out/<run 名>

# ② 名前を付けて置き、SmolVLM の設定とトークナイザを足す（merge.py は出さない）
cp -r work/train/<run 名>/merged_model work/artifacts/model_weights_<label>
cp -r work/artifacts/model_weights_base/vlm work/artifacts/model_weights_<label>/vlm
```

**引き継ぎ元は `model_weights_base` である。** GPU マシンには
`work/artifacts/model_weights/`（提出用の 1 つの置き場）を作っておらず、
**回ごとの名前で並べてある**（`model_weights_base` `model_weights_weak3x` …）。
手順の元は [01_retrain.md](01_retrain.md) の A-7 にある。

> **`vlm/` はどの回のものでも同じである。** 中身は SmolVLM の設定と
> トークナイザ 11 ファイル（約 4.9 MB）で、**重みは 1 つも入っていない。**
> 学習も `merge.py` も触らないので、どの回のものも `model_weights_base` の
> 複製である。疑うときは全部のハッシュを並べて確かめられる。
>
> ```bash
> for d in work/artifacts/*/vlm; do
>     echo "$(cd "$d" && find . -type f | sort | xargs sha256sum | sha256sum | cut -c1-16)  $d"
> done
> ```

**`install_weights.py` を `--dry-run` 無しで実行しないこと。** 既定の取り込み先が
`work/artifacts/model_weights`（**提出する重みそのもの**）である。実験の重みで
置き換える理由は無い。**GPU マシンでは比較相手を明示する**と、テンソル構成まで
見てくれる（書き換えは起きない）。

```bash
python solution/tools/weights/install_weights.py work/train/<run 名>/merged_model \
    --destination work/artifacts/model_weights_base --dry-run
```

## 7. 評価する — **既定の 36 件では測れない**

**このタスクは 36 件に入っていない**（146 本すべてが 341 ステップ以上で、
`--max-demo-steps 300` に全部落ちるため）。**このタスクの変種だけを集めた
一覧が `compe/val/MOKA_TASKS.csv`（8 件）にある**ので、それを渡す
（作り直すときは `solution/tools/suite/make_task_suite.py`）。

```bash
EVAL_CSV=compe/val/MOKA_TASKS.csv MUJOCO_GL=osmesa \
    WEIGHTS="$PWD/work/artifacts/model_weights_<label>" LABEL=<label> \
    bash solution/tools/eval/run_docker_eval.sh --detach
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

## 8. 混ぜて学習する（提出候補）— **上から順に 6 手**

**ここが本命である。** 作った分だけで学習したものは提出できず、
**測ることさえできない**（6 節の「基準が潰れる」）。**混ぜる側にその問題は無い。**

**全部 GPU マシンで行う。** `merge` には `lerobot/libero_plus`（16 GB）が要り、
それがあるのは学習を回したマシンだけである。**判断の枠組み（3 つの関門）と
どの指示文を重くするかは [../talk/260813_today.md](../talk/260813_today.md) にある。**

```bash
# 0. 前提。**足りなければここで止まる**
du -sh ~/.cache/huggingface/lerobot/lerobot/libero_plus   # 16 GB あるか
df -h .                                                   # 17 GB 以上空いているか
```

```bash
# 1a. 列の書き方を相手に合わせる（**1 度だけ。数秒**）
docker run --rm \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /host/tools/scripted_demo/npz_to_lerobot.py align \
    --base /root/.cache/huggingface/lerobot/lerobot/libero_plus \
    --add /data/lerobot
```

**これを飛ばすと `merge` が `Same features is expected` で落ちる。**
形は同じでも**飾りが違う**ためである。`lerobot/libero_plus` は古い版で
作られており、v0.6.0 が書く `features` とは 2 か所で書き方が違う
（非動画の列に `fps` を書くか・動画の `info` が `video.is_depth_map` か
`is_depth_map` か）。`align` は**`dtype`・`shape`・`names` の一致を確かめてから
飾りだけコピーする**ので、本当に形が違うときは止まる。元の `info.json` は
`info.json.bak` に残る。

```bash
# 1b. 混ぜる（16 GB を複製する）
#     **書き出し先を -v で直接指さない。** 親を渡し、その下の未作成の名前にする
rm -rf work/merged
docker run --rm \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work:/work" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /host/tools/scripted_demo/npz_to_lerobot.py merge \
    --base /root/.cache/huggingface/lerobot/lerobot/libero_plus \
    --add /work/scripted/lerobot --add-repo-id local/scripted_moka \
    --output /work/merged

docker run --rm -v "$PWD/work:/w" --entrypoint chown parc2026-train \
    -R "$(id -u):$(id -g)" /w/merged        # Linux では root で書かれる
```

**`-v "$PWD/work/merged:/merged"` と書くと必ず失敗する。** lerobot は書き出し先が
既にあると作らない（`mkdir(exist_ok=False)`）のに、**docker が先に作ってしまう。**
しかもそれは取り付け先そのものなので、コンテナの中からは消せない。

**`--add-repo-id` を忘れないこと**（既定は `local/scripted_demos`）。
最後に**足した分のエピソード番号**（`14347〜14391`）が印字される。

```bash
# 2. ゲート A：配線を 10 ステップで見る（5 分）
docker run --gpus all --rm -v "$PWD/work/train:/out" -v "$PWD/work/merged:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /workspace/train.py --run-name scripted_mix_smoke \
    --dataset-root /data --steps 10 --save-freq 10 \
    --max-demo-steps 300 --emphasis-multiplier 3 \
    --emphasize-tasks "put both moka pots on the stove" <weak3x の 3 つ>
```

**`<weak3x の 3 つ>` は [01_retrain.md](01_retrain.md) の A-2-3 にある指示文を
そのまま並べる**（**完全一致。1 文字違うと 0 本になる**）。**台本のタスクは
4 つ目として足す** —— 入れないと、45 本のうち 12 本しか使われない
（得意タスクの間引きを受けるため。理由は
[../talk/260813_today.md](../talk/260813_today.md)）。

**`14,392 本 / 40 タスク` と印字されるのを確かめる。** ここでしか混ぜ損ねに
気付けない（14,347 のままなら足せていない）。

```bash
# 3. ゲート B：本番のステップ数（20,000 ステップ・約 2 時間）
#    2 と同じ引数から --steps / --save-freq / --run-name を差し替えるだけ
```

```bash
# 4. 重みを整える（6 節と同じ 2 手）
#    **`chown` は `merge.py` の「後」に打つ。** merge.py も root で走るので、
#    先に打つと出来たての merged_model がまた root 所有になり、cp が
#    `Permission denied` で落ちる（8/13 に実際に踏んだ）
docker run --rm --gpus all -v "$PWD/work/train:/out" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /workspace/merge.py --run-name scripted_mix
docker run --rm -v "$PWD/work/train:/out" --entrypoint chown parc2026-train \
    -R "$(id -u):$(id -g)" /out/scripted_mix
cp -r work/train/scripted_mix/merged_model work/artifacts/model_weights_scripted_mix
cp -r work/artifacts/model_weights_base/vlm work/artifacts/model_weights_scripted_mix/vlm

# **並んでいることを目で確かめる。** model.safetensors と vlm が同じ階層に要る
ls work/artifacts/model_weights_scripted_mix
```

> **やり直すときは、先に置き場を消すこと。** `cp -r A B` は **B が既にあると
> 「B の中に A を作る」**ので、2 度打つと `model_weights_…/merged_model/…` と
> 1 段深くなり、**評価が「重みが無い」で落ちる。** 8/13 に踏んだ形はこうだった
> —— 1 回目の `cp` が 907 MB のところで `Permission denied`（`chown` を
> `merge.py` の前に打っていた）で落ち、小さい json だけが入り、2 回目が入れ子。
> **`rm -rf` してから `cp` を打ち直す**のが確実である。

> **【8/13】`compe/val/MOKA_TASKS.csv` は作り直した。前の版は 8 件とも
> 「初期状態が無い」で落ちる。** `init_states_file` の列が無かったため、評価側が
> 接尾辞を正規表現で剥がして初期状態を探しに行き、実在しない
> `KITCHEN_SCENE8_..._light_11.pruned_init` を開こうとしていた
> （実在するのは接尾辞を剥がした `KITCHEN_SCENE8_....pruned_init` のほう）。
> **`make_task_suite.py` が `make_validation_suite._build_rows` を呼ぶようにして直した。**
> 手元にある CSV が古いときは作り直す（数秒・GPU 不要）。
>
> ```bash
> python3 solution/tools/suite/make_task_suite.py \
>     --instruction "put both moka pots on the stove" \
>     --count 8 --output compe/val/MOKA_TASKS.csv
> ```
>
> **列があることを目で確かめる。** 先頭行に `init_states_file` が要る。
> **`compe/val/MOKA_BASE.csv` にも列は無いが、あちらは接尾辞が無いので
> 剥がしても同じ名前になり、偶然通る。**

```bash
# 5. モカポット 8 件で測る（20 分）。**比べる相手を先に測る**
EVAL_CSV=compe/val/MOKA_TASKS.csv MUJOCO_GL=osmesa \
    WEIGHTS="$PWD/work/artifacts/model_weights_weak3x" LABEL=weak3x_moka \
    bash solution/tools/eval/run_docker_eval.sh --detach     # 0/8 のはず

EVAL_CSV=compe/val/MOKA_TASKS.csv MUJOCO_GL=osmesa \
    WEIGHTS="$PWD/work/artifacts/model_weights_scripted_mix" LABEL=scripted_mix_moka \
    bash solution/tools/eval/run_docker_eval.sh --detach
```

```bash
# 6. ゲート C：他のタスクを壊していないか（36 件・72 分）
EVAL_CSV=compe/val/EVAL_TASKS.csv MUJOCO_GL=osmesa \
    WEIGHTS="$PWD/work/artifacts/model_weights_scripted_mix" LABEL=scripted_mix_36 \
    bash solution/tools/eval/run_docker_eval.sh --detach
```

**評価は 2 つ同時に走らせられない。** 起動時に前回の残骸（`parc-policy` と
`parc-eval-net`）を消しに行くので、**走っている評価が途中で終わる。**
1 つずつ流すこと。

| 詰まったら | |
|---|---|
| `Same features is expected` | 1a の `align` を飛ばしている（形ではなく書き方の食い違い） |
| `FileExistsError: /merged` | 書き出し先を `-v` で直接指している。**親を渡す**（1b） |
| `14,347 本` と出る | 足せていない。`--dataset-root` が混ぜた側を指しているか |
| `PermissionError` | root 所有。上の `chown` を回す |
| `vlm が無い` | 4 の `cp` を飛ばしている |
| 方策サーバーが消えた | 評価を 2 つ同時に流した |

## 時間の見積もり

**下の実測はすべて手元の Mac の Docker（colima・2 コア / 9.69 GiB）である。**
GPU マシンで撮る場合は `nproc` から測り直すこと。

| 手 | 直列 | 2 並列 |
|---|---|---|
| `verify` 50 通り（描画なし） | 153 秒 | — |
| `record` 1 本 | **111 秒**（233 ステップ） | **172 秒** |
| `record` 50 本 | 約 92 分 | **約 72 分**（実測。45 本が成功） |
| `record` 変種 4 種で 200 本 | 6.2 時間 | **約 4.8 時間** |
| `build` 45 本 | **10 分 31 秒**（実測・**14 秒/本**） | — |

**`build` は `record` の 1/12 の時間で済む**（1 本 14 秒 対 172 秒）。
出来上がりは **45 本で 64 MB**（front 41.2 MB ＋ wrist 23.6 MB）で、
元の npz 3.8 GB の **1.7%** である。

**`record` の中身は描画が支配的である。** 実測 111 秒 / 233 ステップ = 1 ステップ 476 ms で、
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
| `ModuleNotFoundError: scripted_demo` | 標準入力から流し込むと `__file__` が使えない。`PYTHONPATH=<リポジトリ>/solution/tools/scripted_demo` を付けて実行する |
