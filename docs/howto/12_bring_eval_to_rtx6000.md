# 会社の RTX6000 で本選の自己評価を回せるようにする（持ち込み手順）

**このファイルだけで作業が終わるように書いた。**別のファイルを開く必要はない。

対象マシンは `eng003sasaki01`（会社の RTX6000・Quadro RTX 6000 24 GB・Ubuntu 24.04・
ディスク残り 357 GB）。**ここへ持ち込む理由は EGL（GPU 描画）が効くことである。**
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

**どこまで済んだかはここに書かない。**[../talk/open_items.md](../talk/open_items.md)と
リポジトリ直下の `NOW.md` が持つ。**このファイルはマシンを借り直すたびに上から
やり直すものなので、状態を混ぜると次に借りたとき消し忘れて嘘になる。**

読み終えたときに分かること: **この 6 手順を順に踏めば、`run_final_eval.sh` が
会社の RTX6000 で EGL で回る状態になる。**

## 要るもの 4 つ

**評価は 2 プロセス構成なので、置くものが 4 つある。**

| 要るもの | 何か | どこで用意するか |
|---|---|---|
| **評価イメージ** `parc2026-final` | 運営の配布キットを固めたもの。物理シミュレーションと描画をする | 手順 0 で確かめる |
| **方策イメージ** `parc2026-train-pi05` | torch と LeRobot が入った学習イメージ。重みを読んで `/act` に答える | 手順 3 で作る |
| **重み** 8.8 GB | π0.5 の `model.safetensors` ＋ `vlm/` | 手順 4 で落とす |
| **コード** | `policy_server.py` と `run_final_eval.sh` | 手順 2 で置く |

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

**2026-08-26〜27 に実際に流した結果。**

| 項目 | 値 |
|---|---|
| GPU | `Quadro RTX 6000` / 24,576 MiB / cc **7.5** / ドライバ **595.84** |
| vCPU / RAM | 16 / 47 GiB |
| ディスク | 489 GB のうち **357 GB 空き**（評価イメージを作ったあと） |
| 評価イメージ | `parc2026-final:latest` `a174b42ff901`・**ディスク上 69.2 GB**（中身 26.8 GB） |
| GitHub | ssh が通る（`Hi siruku6! You've successfully authenticated`） |
| `docker` を sudo 無しで | 8/26 は拒否された。**8/27 に解決**（手順 1 と [knowledge/16](../knowledge/16_docker_group_needs_session_restart.md)） |
| **配布キットの場所** | **`~/src/b_PARC2026_final`。`~/src/PARC2026_pre` の隣なので `KIT_ROOT` は要らない** |

**`KIT_ROOT` は存在確認にしか使われていない。**`pipeline` は評価イメージの中に入っている
ので、`run_final_eval.sh` は 39 行目でキットの場所を求めるだけで、そのあと参照していない。
**それでも隣に無いと `set -e` で落ちるので、無い場合だけ `KIT_ROOT=<実際の場所>` を渡す。**
探し方:

```bash
find ~ -maxdepth 4 -type d -name "*PARC2026_final*" 2>/dev/null
```

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

**ログインし直しても直らないことがある**（VS Code Remote-SSH の Reload Window だけでは
足りなかった実例）。[../knowledge/16_docker_group_needs_session_restart.md](../knowledge/16_docker_group_needs_session_restart.md)
に切り分け手順を残した。

## 手順 2 コードを置く

**配布キットの隣に `PARC2026_pre` を置く。**並びが崩れるとスクリプトがキットを見つけられない。

**2026-08-27 の実機は `~/src/` の下に 2 つ並べた。**

```
~/src/
├── b_PARC2026_final/     ← 配布キット（既にある）
└── PARC2026_pre/         ← これから置く
```

```bash
mkdir -p ~/src && cd ~/src
git clone git@github.com:siruku6/PARC2026_pre.git
cd PARC2026_pre
git log --oneline -1
```

**`main` が最新である**（8/26 に PR #11 が merge された）。ブランチを指定する必要は無い。

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
cd ~/src/PARC2026_pre/solution/train
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

[pyenv install 手順](https://github.com/pyenv/pyenv?utm_source=chatgpt.com#installation)

```bash
git clone git@github.com:pyenv/pyenv.git ~/.pyenv

# 一般のUbuntuの場合
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

source .bashrc
pyenv install --list
pyenv install 3.12.14
pyenv local 3.12.14
python --version

python -m venv ~/.venv
source ~/.venv/bin/activate

pip install -q -U huggingface_hub
hf auth login          # 対話。write 権限は要らない
```

```bash
cd ~/src/PARC2026_pre
mkdir -p work/artifacts
hf download siruku6/pi05_full_runpod \
  --local-dir work/artifacts/pi05_rp3000
du -sh work/artifacts/pi05_rp3000
ls work/artifacts/pi05_rp3000/vlm
```

**期待する結果。** 8.8 GB。`model.safetensors` が 9,354,xxx,xxx バイト、`vlm/` に `tokenizer.json` ほか 4 ファイル。**`vlm/` が無いと評価は 5 秒で落ちる。**

**この重みは予選最終提出（版 4.2.0）のものであり、`policy_server.py` 4.6.0 が既定で使う重みと同じである。**

<!-- **別の道。** `siruku6/parc2026-submission` の `submission_4.5.0.zip`（9.378 GB）を落として展開すると、`model_weights/` として同じ重みが出てくるうえ、提出した `policy_server.py` そのものも入っている。**提出物の中身を確かめたいときはこちら。**
展開に 9.4 GB 余分に要る。

```bash
hf download siruku6/parc2026-submission submission_4.5.0.zip --local-dir work/artifacts/submission_4.5.0.zip
unzip -q work/artifacts/submission_4.5.0.zip -d work/artifacts/submission_4.5.0   # → work/artifacts/submission_4.5.0/model_weights
``` -->

## 手順 5 疎通を 1 本流す

**本番の 4 本を流す前に、1 タスクだけで通し、EGL が効くかとタスクあたりの秒数を測る。**

```bash
cd ~/src/PARC2026_pre
MUJOCO_GL=egl TRACK_EPISODES=1 LABEL=smoke \
  bash solution/tools/eval/run_final_eval.sh track2
```

所要 10 分ほど（重みの読み込みに 1〜2 分、8 タスク × 1 本）。**画面に出る
`[parc] policy_server 4.5.0 …` の行で、効いている設定値を確認する。**

**分かること 3 つ。**

| 見るもの | どこ | 期待 |
|---|---|---|
| bf16 が cc 7.5 で動くか | 落ちずに `/health OK` が出るか | **8/27 に確認済み。動く** |
| EGL が効いたか | `work/final_eval/*_track2_smoke/eval.log` に `read_pixels` の失敗が無いこと | 落ちない |
| 1 本の所要 | 同じ出力の末尾と `results/*.json` の `mean_avg_episode_time_sec` | JAIST サーバの osmesa では 63.1 秒だった |

**【2026-08-27 の実測値】**このときの出力は
`work/final_eval/0827_151131_track2_smoke/` にある。

| 見たもの | 結果 |
|---|---|
| 立ち上がり | **通った。**`bf16` は cc 7.5 でも動く（fp32 へ落とす必要は無い） |
| 描画 | **EGL で通った。**`read_pixels` の失敗は 1 件も無い |
| **1 本の所要** | **30.6 秒。JAIST サーバの osmesa は 60.6 秒だったので、ちょうど半分になった** |
| 合計ステップ数 | 1,626 |
| 成功率 | 0.375（3/8） |

**評価まるごとで 2 倍速い。**箱 1 個の描画では 16 倍（9.7 ms → 0.6 ms）だったが、
評価には物理計算と方策の推論も含まれるので、実際の縮みは 2 倍である。**それでも
1 本 10 分が 5 分になるので、1 日に流せる本数が倍になる。**

**この回は版 4.5.0 で動いていた**（`gain=(1.15,1.0)`）。clone が並進の倍率を
1.0 へ戻す前の版だったためである。**結果として、思わぬ収穫があった。**

| | JAIST・倍率 1.0 | **RTX6000・倍率 1.15** |
|---|---|---|
| 成功率 | **0.750**（6/8） | **0.375**（3/8） |
| 落ちたタスク | cookie_box・ramekin | 左の 2 つ ＋ **orange_juice・cream_cheese・salad_dressing** |

**採点側では Track2 が 0.154 → 0.035 に崩れていた。手元の 8 行も 0.750 → 0.375 に
崩れた。向きが一致している。**ただしマシンと描画方式も同時に変わっているので、
**手順 6 の 1 本目（倍率 1.0・EGL）を流すまでは 1 変数ではない。**

### 指標が 2 つ足りないときは、採点器が差し替わっていない

**json に `mean_collision_rate` と `mean_sparc` が無ければ、採点器の差し替えが
効いていない。**JAIST 側の出力には 16 個あるのに 14 個しか出なかったことがある
（2026-08-27）。**衝突は採点のゲートなので、無いままでは手元の点が衝突を見ていない。**

**原因はキットの版ではない。**両方 `2059989`（配布そのもの）の上にあり、
`pipeline` のコードに差は無い。**JAIST 側の `pipeline/scorer.py` に、コミットして
いない作業ツリーの変更が入っていた**（30 行の追加）。本選の配布版はこの 2 つの計算を
落としており、予選の配布版から戻したものが JAIST にだけ残っていた。
**clone した RTX6000 にはそれが無い。**

**直し方。**キットの中を直すとまたマシンごとにズレるので、
`solution/tools/eval/kit_patches/scorer.py` としてこちらのリポジトリに置き、
**`run_final_eval.sh` が評価コンテナの `/workspace/pipeline/scorer.py` へ
上書きマウントする**ようにした。`git pull` すれば RTX6000 でも同じ指標が出る。
**評価イメージの作り直しは要らない。**

```bash
cd ~/src/PARC2026_pre && git pull
ls solution/tools/eval/kit_patches/scorer.py     # あれば効く
```

**効いていることは、次の評価の出力に `-- 採点器を差し替える` の行が出て、
json に `mean_collision_rate` が入ることで分かる。**

**落ちたときの報告のしかた。** `work/final_eval/<日時>_track2_smoke/` の
`server.log` の末尾 40 行と `eval.log` の末尾 40 行を送ってもらえれば読める。

## 手順 6 本番の 4 本を流す

**測りたいのは「手元の 8 行が、採点の結果を再現できるか」である。**

**答えは既に採点側で出ている。**版 4.5.0（並進の倍率 1.15）は 8/26 の採点で
**Track1 が 0.370 → 0.136、Track2 が 0.154 → 0.035** と両方崩れた（4.2.0 との差は
倍率と乱数の種だけ）。**だから 1 と 2 の間に大きな差が出るのが正解である。**

| # | 設定 | 何を知るための本か |
|---|---|---|
| 1 | 倍率 1.0・種 0 | 基準（採点では Track1 0.370・Track2 0.154 に相当） |
| 2 | **倍率 1.15**・種 0 | **採点では Track2 0.035 だった。ここが 1 より明確に悪くなるか** |
| 3 | 倍率 1.30・種 0 | もっと上げたときの向き（採点では未測定） |
| 4 | 倍率 1.15・**種 1** | **雑音の大きさ。**これが 1・2 の差より大きければ、手元の 8 行では何も判定できない |

**この 4 本が「手元の評価は使えるか」の試験そのものである。**

| 結果 | 意味 |
|---|---|
| 1 が 2 より明確に良く、4 が 2 に近い | **手元の 8 行は採点の向きを再現する。**以後は設定の絞り込みを手元でやれる |
| 1 と 2 の差が 4 と 2 の差に埋もれる | **手元の 8 行では判定できない。**エピソード数を増やすか、別の測り方を作る |
| 1 が 2 より悪い | **手元の評価は採点と逆を向く。**手元の数字で設定を決めてはいけない |

```bash
cd ~/src/PARC2026_pre
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
cd ~/src/PARC2026_pre
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

## つまずきやすい点

| 症状 | 原因と直し方 |
|---|---|
| `docker` が `permission denied` | [../knowledge/16](../knowledge/16_docker_group_needs_session_restart.md)。`id -nG` に `docker` が無ければセッションが古い |
| 起動して 5 秒で落ちる | 重みの隣に `vlm/` が無い。`pi05_rp3000` から**コピー**する（リンクではない） |
| `set -e` でいきなり終わる | リポジトリの隣に `b_PARC2026_final` が無い。`KIT_ROOT` を渡す |
| json の指標が 14 個しかない | 採点器の差し替えが効いていない。`git pull` してから流し直す |
| GitHub へ ssh が通らない | Personal Access Token で https clone する |

## あとで見る資料

| 知りたいこと | ファイル |
|---|---|
| どのマシンに何があるか | [../resources/infra_and_docker.md](../resources/infra_and_docker.md) |
| HF に何が置いてあるか | [../resources/huggingface.md](../resources/huggingface.md) |
| HF の転送で詰まったとき | [08_hf_transfer_troubleshooting.md](08_hf_transfer_troubleshooting.md) |
| docker がログインし直しても sudo なしで使えないとき | [../knowledge/16_docker_group_needs_session_restart.md](../knowledge/16_docker_group_needs_session_restart.md) |
| 評価が CPU 描画で遅い話の実測 | [../knowledge/14_quantization_and_eval_bottleneck.md](../knowledge/14_quantization_and_eval_bottleneck.md) |
| ディスクを空ける | [09_free_disk_space.md](09_free_disk_space.md) |
