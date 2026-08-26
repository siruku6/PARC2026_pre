# 会社の RTX6000 で本選の自己評価を回せるようにする（持ち込み手順）

**このファイルだけで作業が終わるように書いた。**別のファイルを開く必要はない。

対象マシンは `eng003sasaki01`（会社の RTX6000・Quadro RTX 6000 24 GB・Ubuntu 24.04・
ディスク残り 455 GB）。**ここへ持ち込む理由は EGL（GPU 描画）が効くことである。**
JAIST サーバの vGPU は OpenGL に対応せず CPU 描画しか選べないため、評価の待ち時間の
大半が描画で消えている。同じ描画スクリプト（240×320 を 50 枚）で 9.7 ms/枚 → **0.6 ms/枚**
という差を 2026-08-26 に実測した。

作成: 2026-08-26。

| # | 手順 | 所要 | 並行 |
|---|---|---|---|
| 0 | [前提を数える](#手順-0-前提を数える) | 2 分 | — |
| 1 | [docker を sudo なしで使えるようにする](#手順-1-docker-を-sudo-なしで使えるようにする) | 2 分 | — |
| 2 | [コードを置く](#手順-2-コードを置く) | 5 分 | — |
| 3 | [方策イメージを作る](#手順-3-方策イメージを作る) | 20〜60 分 | 4 と並行可 |
| 4 | [重みを落とす](#手順-4-重みを落とす) | 10〜30 分 | 3 と並行可 |
| 5 | [疎通を 1 本流す](#手順-5-疎通を-1-本流す) | 10 分 | — |
| 6 | [本番の 4 本を流す](#手順-6-本番の-4-本を流す) | 40 分〜 | — |

読み終えたときに分かること: **この 6 手順を順に踏めば、`run_final_eval.sh` が
会社の RTX6000 で EGL で回る状態になる。**

## 要るもの 4 つ

**評価は 2 プロセス構成なので、置くものが 4 つある。**

| 要るもの | 何か | 会社の RTX6000 の現状 |
|---|---|---|
| **評価イメージ** `parc2026-final` | 運営の配布キットを固めたもの。物理シミュレーションと描画をする | **ビルド済み** |
| **方策イメージ** `parc2026-train-pi05` | torch と LeRobot が入った学習イメージ。重みを読んで `/act` に答える | **無い**（手順 3 で作る） |
| **重み** 8.8 GB | π0.5 の `model.safetensors` ＋ `vlm/` | **無い**（手順 4 で落とす） |
| **コード** | `policy_server.py` と `run_final_eval.sh` | **無い**（手順 2 で置く） |

## 手順 0 前提を数える

実行場所: **会社の RTX6000 のホスト**（コンテナの外）。

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
docker images parc2026-final
df -h ~
ls -d ~/b_PARC2026_final 2>/dev/null || echo "配布キットの場所が違う"
ssh -T git@github.com 2>&1 | head -1
nproc; free -g | head -2
```

**分かること。** 評価イメージがあるか、配布キットがどこにあるか、GitHub へ ssh で
届くか、ディスクが 20 GB 以上空いているか。

**配布キットの場所が `~/b_PARC2026_final` でない場合**、手順 5・6 のコマンドに
`KIT_ROOT=<実際の場所>` を足す。`run_final_eval.sh` は既定で「自分のリポジトリの隣の
`b_PARC2026_final`」を見る。

## 手順 1 docker を sudo なしで使えるようにする

**このマシンでは `docker run` に `sudo` が要る。**`run_final_eval.sh` は中で
`docker run` を 4 回呼ぶので、スクリプト全体を `sudo` で回すか、自分を `docker`
グループに入れるかの二択になる。**後者を推す。**`sudo` だと `$HOME` の解決と
出力ファイルの持ち主で毎回つまずく。

```bash
sudo usermod -aG docker "$USER"
```

**このあと一度ログアウトして入り直す**（`newgrp docker` でもよいが、そのシェルだけに効く）。
入り直したら `docker info >/dev/null && echo OK` で確かめる。

**代償を 1 行で。** `docker` グループは実質 root と同じ権限になる。共用機なら
`sudo -E bash solution/tools/eval/run_final_eval.sh ...` で通す道を選ぶ
（`-E` を落とすと `$HOME` が `/root` になり、HF キャッシュのマウント元が変わる）。

## 手順 2 コードを置く

**配布キットの隣に `PARC2026_pre` を置く。**並びが崩れるとスクリプトがキットを見つけられない。

```
~/
├── b_PARC2026_final/     ← 配布キット（既にある）
└── PARC2026_pre/         ← これから置く
```

```bash
cd ~
git clone -b submission/v1-base-weights git@github.com:siruku6/PARC2026_pre.git
cd PARC2026_pre
git log --oneline -1
```

**GitHub へ ssh で届かない場合**、`https://github.com/siruku6/PARC2026_pre.git` に
Personal Access Token で clone する。

**注意。** clone した中身に `policy_server.py` の版 4.5.0 と `run_final_eval.sh` が
含まれているかを確かめる。含まれていなければ JAIST サーバ側で push が済んでいない。

```bash
grep -m1 '_VERSION' solution/policy_server.py
ls solution/tools/eval/run_final_eval.sh
```

## 手順 3 方策イメージを作る

実行場所: **ホスト**。GPU は要らない。**手順 4 と同時に流してよい。**

```bash
cd ~/PARC2026_pre/solution/train
docker build -f Dockerfile.pi05 -t parc2026-train-pi05 .
```

**所要 20〜60 分**（torch cu128 と LeRobot の pip install が大半）。放っておける。
JAIST サーバでの出来上がりは 9.86 GB だった。

**CUDA 12.8 で組んだイメージをドライバ 595 で動かすことになるが、これは前方互換で問題ない**
（新しいドライバは古い CUDA を動かせる）。

終わったら次で確かめる。

```bash
docker run --rm --gpus all parc2026-train-pi05 \
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

**`run_final_eval.sh` は、これに `fastapi`/`uvicorn`/`msgpack` を足した派生イメージを
初回に自動で作る**（`parc2026-train-pi05-server-<image_id>`）。手作業は要らない。

## 手順 4 重みを落とす

実行場所: **ホスト**。**手順 3 と同時に流してよい。**

Ubuntu 24.04 は `pip install` をシステムに直接させないので、仮想環境を 1 つ作る。

```bash
python3 -m venv ~/venv-hf
~/venv-hf/bin/pip install -q -U "huggingface_hub[cli,hf_xet]"
~/venv-hf/bin/hf auth login          # 対話。write 権限は要らない
```

```bash
cd ~/PARC2026_pre
mkdir -p work/artifacts
~/venv-hf/bin/hf download siruku6/pi05_full_runpod \
  --local-dir work/artifacts/pi05_rp3000
du -sh work/artifacts/pi05_rp3000
ls work/artifacts/pi05_rp3000/vlm
```

**期待する結果。** 8.8 GB。`model.safetensors` が 9,354,xxx,xxx バイト、`vlm/` に
`tokenizer.json` ほか 4 ファイル。**`vlm/` が無いと評価は 5 秒で落ちる。**

**この重みは予選最終提出（版 4.2.0）のものであり、`policy_server.py` 4.5.0 が
既定で使う重みと同じである。**HF 上の `siruku6/pi05_full_runpod` は非公開なので
`hf auth login` が要る。

**別の道。** `siruku6/parc2026-submission` の `submission_4.5.0.zip`（9.378 GB）を
落として展開すると、`model_weights/` として同じ重みが出てくるうえ、提出した
`policy_server.py` そのものも入っている。**提出物の中身を確かめたいときはこちら。**
展開に 9.4 GB 余分に要る。

```bash
~/venv-hf/bin/hf download siruku6/parc2026-submission submission_4.5.0.zip --local-dir /tmp/sub
unzip -q /tmp/sub/submission_4.5.0.zip -d /tmp/sub/x   # → /tmp/sub/x/model_weights
```

## 手順 5 疎通を 1 本流す

**本番の 4 本を流す前に、1 タスクだけで通し、EGL が効くかとタスクあたりの秒数を測る。**

```bash
cd ~/PARC2026_pre
MUJOCO_GL=egl TRACK_EPISODES=1 LABEL=smoke \
  bash solution/tools/eval/run_final_eval.sh track2
```

所要 10 分ほど（重みの読み込みに 1〜2 分、8 タスク × 1 本）。**画面に出る
`[parc] policy_server 4.5.0 …` の行で、効いている設定値を確認する。**

**分かること 3 つ。**

| 見るもの | どこ | 期待 |
|---|---|---|
| bf16 が cc 7.5 で動くか | 落ちずに `/health OK` が出るか | **ここが唯一の未確認点**（下の「確認したい仮定」） |
| EGL が効いたか | `work/final_eval/*_track2_smoke/eval.log` に `read_pixels` の失敗が無いこと | 落ちない |
| 1 本の所要 | 同じ出力の末尾と `results/*.json` の `mean_avg_episode_time_sec` | JAIST サーバの osmesa では 63.1 秒だった |

**落ちたときの報告のしかた。** `work/final_eval/<日時>_track2_smoke/` の
`server.log` の末尾 40 行と `eval.log` の末尾 40 行を送ってもらえれば読める。

## 手順 6 本番の 4 本を流す

**測りたいのは「並進の倍率をどこに置くか」と「その差が雑音より大きいか」である。**
4 本は互いに 1 変数だけ違う。

| # | 設定 | 何を知るための本か |
|---|---|---|
| 1 | 倍率 1.0・種 0 | 基準 |
| 2 | **倍率 1.15**・種 0 | 提出した 4.5.0 そのもの |
| 3 | 倍率 1.30・種 0 | 上げ過ぎると崩れるかどうか |
| 4 | 倍率 1.15・**種 1** | **雑音の大きさ。**これが 1・2 の差より大きければ 2 の結果は読めない |

```bash
cd ~/PARC2026_pre
for spec in "1.0 0 g100_s0" "1.15 0 g115_s0" "1.3 0 g130_s0" "1.15 1 g115_s1"; do
  set -- $spec
  MUJOCO_GL=egl TRACK_EPISODES=1 LABEL="$3" \
    PARC_GAIN_TRANSLATION="$1" PARC_SEED="$2" \
    bash solution/tools/eval/run_final_eval.sh track2
done
```

所要は 1 本あたり手順 5 と同じ（osmesa なら 10 分、EGL でどこまで縮むかは手順 5 で分かる）。
**4 本まとめて放っておける。**

終わったら次を送ってもらえれば私が読む。

```bash
cd ~/PARC2026_pre
for d in work/final_eval/*_track2_g*/; do
  echo "== $d"
  python3 -c "
import json,glob,sys
for f in glob.glob('$d/results/*.json'):
    t=json.load(open(f))['tracks'][0]; m=t['overall_metrics']
    print(' score', t['overall_score'],
          'collision', m['mean_collision_rate'],
          'steps', m['mean_avg_steps_to_success'],
          'path', round(m['mean_cartesian_path_length'],3),
          'orient', round(m['mean_orientation_path_length'],3))
"
done
```

## 比べる相手について 1 つ注意

**2026-08-25 に JAIST サーバで測った Track2 の 8 行（成功 0.75・衝突 0.25・
ステップ 129.5・軌道 1.027・向き 0.886）は、版 4.4.0 の既定値で回したものである。**

```
[parc] policy_server 4.4.0  num_inference_steps=20  n_action_steps=16  samples=3  seed=off
```

4.5.0 の既定値は `num_inference_steps=10`・`samples=1`・`seed=0` なので、**上の 4 本と
8/25 の数字は 3 変数違う。直接の比較には使えない。**上の 4 本どうしの比較だけが 1 変数である。

## 確認したい仮定

| 仮定 | 根拠 | 外れると何が変わる |
|---|---|---|
| **bf16 が compute capability 7.5 で動く** | 重みは `dtype=bfloat16` で保存されており、cc 7.5 に bf16 の専用回路は無い。torch は多くの演算を代替経路で通すが、全部が通る保証は無い | 落ちるなら fp32 で読む必要があり、VRAM が 11.9 GB → 約 19 GB に増える（24 GB には載るが余裕が消える） |
| 配布キットが `~/b_PARC2026_final` にある | 評価イメージがビルド済みなので clone はされている、と読んだだけ | `KIT_ROOT` を渡す必要がある |
| GitHub へ ssh で届く | 未確認 | Personal Access Token で https clone する |

**手順 5 の 1 本で 3 つとも片付く。**

## あとで見る資料

| 知りたいこと | ファイル |
|---|---|
| どのマシンに何があるか | [../resources/infra_and_docker.md](../resources/infra_and_docker.md) |
| HF に何が置いてあるか | [../resources/huggingface.md](../resources/huggingface.md) |
| HF の転送で詰まったとき | [08_hf_transfer_troubleshooting.md](08_hf_transfer_troubleshooting.md) |
| 評価が CPU 描画で遅い話の実測 | [../knowledge/14_quantization_and_eval_bottleneck.md](../knowledge/14_quantization_and_eval_bottleneck.md) |
| ディスクを空ける | [09_free_disk_space.md](09_free_disk_space.md) |
