# 05. GPU マシンで手本を作る

**Colab は要らない。** 必要な環境は 2 つとも既にある。

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
| NVIDIA ドライバ ＋ nvidia-container-toolkit | `docker run --rm --gpus all parc2026 nvidia-smi` |
| **`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`** | 下の 2 節。**無いと黙って 11 倍遅くなる** |
| ディスク | 学習データ 16 GB ＋ 重み ＋ 生成物（1 本 85 MB） |

```bash
sudo mkdir -p /usr/share/glvnd/egl_vendor.d
sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null <<'EOF'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
EOF
```

## 1. イメージを 2 つ作る

```bash
docker build -t parc2026 .                      # 配布キット（10〜20 分）
docker build -t parc2026-train solution/train   # 学習側。学習で作ってあるはず
```

## 2. 描画が GPU を掴んでいるかを測る

**必ず先に測る。** libglvnd は「どのベンダ実装を使うか」を `egl_vendor.d` の
JSON で決めており、NVIDIA の登録が無いと **例外も警告も出さずに** Mesa の
ソフトウェア実装に落ちる。

```bash
docker run --rm --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e MUJOCO_GL=egl \
    parc2026 python -c "
import os, time, mujoco
model = mujoco.MjModel.from_xml_string(\"<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>\")
r = mujoco.Renderer(model, 256, 256); d = mujoco.MjData(model)
r.update_scene(d); r.render()
s = time.time()
for _ in range(50): r.update_scene(d); r.render()
print(f'1 枚 {(time.time()-s)/50*1000:.1f} ms')
os._exit(0)"
```

| 出た値 | 判定 |
|---|---|
| **0.4 ms**（実測） | GPU を掴んでいる |
| 8 ms 超 | **ソフトウェア描画。** JSON を置いてやり直す |

`NVIDIA_DRIVER_CAPABILITIES` に **`graphics` が要る。** nvidia-container-runtime は
既定で `compute,utility` しか通さず、描画を落とす。

> **`os._exit(0)` を付けているのは、終了時の後片付けで EGL のエラーが
> 出るのを避けるため**である（`Exception ignored in: Renderer.__del__`）。
> 測り終わったあとに起きるので値には影響しないが、紛らわしい。

> **球 1 個が描けることと、台所の場面が描けることは別である。** vGPU の一部の
> 構成では描画が許可されておらず、単純な図形は通っても実際の場面で止まる。
> 次の 3 節が本当の確認になる。止まったら `MUJOCO_GL=osmesa` に落として
> 切り分ける（遅いが動く）。

## 3. 手本を記録する

```bash
mkdir -p work/scripted
docker run --rm --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e MUJOCO_GL=egl \
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

見るところは 2 つ。

- **`ep00 ... 成功 True`** が出れば、場面も描けている
- **1 本あたりの秒数。** 描画は 0.4 ms でも**物理計算は CPU** なので、そこが下限になる

通ったら本数を増やす。**`--episodes` を省くと初期配置 50 通り全部を回す。**

```bash
# 見た目の摂動を掛けた変種で撮り直す。**物理は同じなので同じ台本が通る**
for variant in _light_2 _table_28 _tb_9; do
    docker run --rm --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e MUJOCO_GL=egl \
        -v "$PWD/solution/tools:/host/tools:ro" -v "$PWD/work/scripted:/out" \
        parc2026 python /host/tools/scripted_demo.py record \
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \
        --variant "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove${variant}" \
        --output "/out${variant}"
done
```

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

| 段 | CPU（手元の Mac） | GPU マシン |
|---|---|---|
| `verify` 50 通り | 153 秒 | ほぼ同じ（描画しない） |
| `record` 1 本 | **100 秒** | 実測して埋める |
| `record` 50 本 | 83 分 | 同上 |

**`record` の中身は描画が支配的だった**（CPU で 1 歩 300 ms）。GPU では
1 枚 0.4 ms まで下がるので、**物理計算（1 歩 39 ms・CPU）が下限になる。**

> **npz は圧縮していない。** 動画に再エンコードするだけの中継物で、画像
> 118 MB を deflate しても 1.7 倍しか縮まないのに 1 本 4.3 秒かかる。
> 1 本 85 MB になるので、50 本で 4.3 GB を見ておく。

## 詰まったときの見どころ

| 症状 | 原因 |
|---|---|
| `EOFError: EOF when reading a line` | `$HOME/.libero/config.yaml` が見えていない。**`--user` と `HOME` を外す** |
| 描画が 8 ms を超える | `10_nvidia.json` か `NVIDIA_DRIVER_CAPABILITIES=...,graphics` |
| 球は描けるのに場面で止まる | vGPU で描画が許可されていない。`MUJOCO_GL=osmesa` へ落とす |
| 出力を消せない | root で書かれている。上の `chown` を回す |
| `ModuleNotFoundError: scripted_demo` | 標準入力から流し込むと `__file__` が使えない。`PYTHONPATH` に道具の置き場を入れる |
| `Exception ignored in: Renderer.__del__` | 終了時の後片付け。**結果には影響しない** |
