# 05. 手本を台本で作る（GPU マシン）

**Colab は要らない。** 必要な環境は 2 つとも既にある。

> **描画は CPU である。** このマシンでは GPU で描けない（2 節）。
> GPU を使うのは学習（6 節）だけで、**手本づくりは CPU の並列で速くする**（3 節）。

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
そして**このマシンでは描画に GPU が使えない**（下の 2 節）。GPU が要るのは
学習（6 節）だけである。

## 1. イメージを 2 つ作る

```bash
docker build -t parc2026 .                      # 配布キット（10〜20 分）
docker build -t parc2026-train solution/train   # 学習側。学習で作ってあるはず
```

## 2. 描画は CPU（osmesa）を使う

**このマシンで EGL（GPU 描画）は原理的に使えない。** vGPU が
`NVIDIA Virtual Compute Server` で、OpenGL / Vulkan に対応しないためである。
経緯と `nvidia-smi` の出力は [03_gpu_eval.md](03_gpu_eval.md) の
「描画は osmesa を使う」にある。

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

# root で書かれるので、持ち主を戻す
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

### 遅さは並列で取り戻す

**評価は並列にできない**（方策サーバーと GPU が 1 台ずつ）。**しかし `record` は
どちらも要らない。** CPU のコア数だけ並べられる。

```bash
N=8                                    # nproc を見て決める
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

| | 直列 | 8 並列 |
|---|---|---|
| 50 本 | **83 分** | **約 11 分** |
| 変種 4 種で 200 本 | 5.5 時間 | **約 45 分** |

メモリは 1 本あたり画像の置き場 118 MB ＋ 実行時 0.5 GB 程度。8 並列で 5 GB。

**`--episodes` を省くと初期配置 50 通り全部を回す。** 変種を変えるときは
`--variant` と `--output` を差し替える（**物理は同じなので同じ台本が通る**）。

## 4. 形が合っているかを確かめる（lerobot 不要）

**変換の前に必ず通す。** ここで止まれば、学習側へ持って行く前に気づける。

```bash
docker run --rm -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    -e PYTHONPATH=/host/tools:/workspace:/workspace/LIBERO-plus:/workspace/compe \
    parc2026 python /host/tools/npz_to_lerobot.py check --npz-dir /data/base
```

混ぜる相手の宣言と突き合わせるなら `--against <libero_plus の meta/info.json>` を足す。
**食い違ったら 0 以外で終わる。**

## 5. 学習データの形式に変換する

**ここから Python 3.12 側。** 道具はイメージに入っていないのでマウントする
（`solution/train/Dockerfile` は `train.py` と `merge.py` だけを COPY する）。

```bash
# 作った分だけで 1 つのデータセットにする
docker run --rm --gpus all \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /host/tools/npz_to_lerobot.py build \
    --npz-dir /data/base --output /data/lerobot --repo-id local/scripted_moka

# 既存の学習データに混ぜる（提出候補を作るときはこちら）
docker run --rm --gpus all \
    -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /host/tools/npz_to_lerobot.py merge \
    --base /root/.cache/huggingface/lerobot/lerobot/libero_plus \
    --add /data/lerobot --output /data/merged
```

**`merge` は 16 GB を複製する。** ディスクを先に確認すること。

## 6. 学習する

`train.py` は `--` の後ろを**最後に**付けるので、データセットの差し替えは
**コード変更なしで通る**（同じ設定を 2 度渡すと後ろが勝つ）。

```bash
docker run --gpus all --rm \
    -v "$PWD/work/train:/out" -v "$PWD/work/scripted:/data" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    parc2026-train python /workspace/train.py --run-name scripted_only -- \
    --dataset.repo_id=local/scripted_moka --dataset.root=/data/lerobot
```

**2 つ気をつける。**

| 気をつけること | なぜ |
|---|---|
| **`--episodes-per-task` / `--emphasize-tasks` / `--max-demo-steps` と併用しない** | これらは `lerobot/libero_plus` のメタデータを別に落として番号を決めるので、**差し替えたデータセットとは番号が対応しない。** 混ぜたデータで絞るなら `--dataset.episodes` を `--` の後ろで自分で渡す |
| **`--dataset.revision` の扱いは未確認** | `train.py` が固定値を渡す。ローカルのデータセットに revision は無いので、上書きが要るかは**実機で 1 回試さないと分からない** |

**作った分だけで学習したものは提出できない。** 1 タスクぶんの手本しか無いので
残り 39 タスクの性能が落ちる（8/11 の追加学習は本番 0.1409 → 0.016 に崩れた）。
`build` は「台本を真似られるか」を見るため、`merge` が提出候補を作るためのものである。

## 7. 評価する

いつもどおり。手順は [03_gpu_eval.md](03_gpu_eval.md) にある。

## 時間の見積もり

| 段 | 手元の Mac | GPU マシン（直列） | GPU マシン（8 並列） |
|---|---|---|---|
| `verify` 50 通り | 153 秒 | 同程度 | — |
| `record` 1 本 | **100 秒** | 実測して埋める | 同じ |
| `record` 50 本 | 83 分 | 同程度 | **約 11 分** |

**`record` の中身は描画が支配的である**（1 歩あたり物理 39 ms ＋ 描画
256×256 を 2 枚）。**GPU では描けないので、速くする手は並列しかない。**

> **npz は圧縮していない。** 動画に再エンコードするだけの中継物で、画像
> 118 MB を deflate しても 1.7 倍しか縮まないのに 1 本 4.3 秒かかる。
> 1 本 85 MB になるので、50 本で 4.3 GB を見ておく。

## 詰まったときの見どころ

| 症状 | 原因 |
|---|---|
| `EOFError: EOF when reading a line` | `$HOME/.libero/config.yaml` が見えていない。**`--user` と `HOME` を外す** |
| 場面の描画で止まる | `MUJOCO_GL=egl` になっている。**このマシンでは使えない。** `osmesa` にする |
| 並列にしても速くならない | `LP_NUM_THREADS=1` を渡していない。llvmpipe がコアを奪い合う |
| 出力を消せない | root で書かれている。上の `chown` を回す |
| `ModuleNotFoundError: scripted_demo` | 標準入力から流し込むと `__file__` が使えない。`PYTHONPATH` に道具の置き場を入れる |
