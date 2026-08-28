# 10. このマシンを作り替えたあと、元通りにする

**作り替えるとディスクは全部消える。** git と HF に置いたものは残るが、それ以外は消える。
この 1 枚だけで、作り替える前の持ち出しと、作り替えたあとの復旧が終わる形にした。

**復旧にかかる時間は、待ち時間がほとんどである。** 手を動かす部分は 30 分ほどで、
残りは docker のビルドと HF からの取り直しを放っておく時間になる。

| 段 | 実測に基づく所要時間 | 放っておけるか |
|---|---|---|
| 0. GPU ドライバとライセンス | 不明（自分では決められない） | いいえ |
| 1〜2. 持ち出しと秘密の作り直し | 15 分 | いいえ |
| 3. OS の下ごしらえ | 15 分 | ほぼ |
| 4. docker image を 3 つビルド | 60〜90 分 | はい |
| 5. HF から 29 GB 取り直す | 30〜60 分 | はい |
| 6. 常駐コンテナを起こす | 1 分 | いいえ |
| 7. 復旧できたかを確かめる | 5 分 | はい |

## 項目一覧

| 節 | 中身 |
|---|---|
| 0 | **GPU ドライバの .deb がどこにも無い。** ここだけは自分では復旧できない |
| 1 | 作り替える前に走らせる 1 本のスクリプト |
| 2 | 秘密の 4 件は tar に入れず、手で作り直す |
| 3 | OS の下ごしらえ（apt 19 個・リポジトリ 2 つ） |
| 4 | docker image は 3 つ。派生 image は作らなくてよい |
| 5 | HF から取り直すもの（モデル 5・データセット 3） |
| 6 | **常駐コンテナ 2 つの起動コマンド。実測から復元した** |
| 7 | 復旧できたことを確かめる 3 つのコマンド |
| 8 | 取り直さないと決めたもの |

---

## 0. GPU ドライバの .deb がどこにも無い —— ここだけは自分では復旧できない

**復旧の中で唯一、自分の手では終わらせられない段である。** GPU が使えなければ学習も
評価も 1 つも動かないので、**作り替えを始める前に決着させること。**

いま入っているドライバと、取り直せるかどうかは次のとおり。

| | 実測値 |
|---|---|
| GPU | `NVIDIA H100-20C`（1 枚の H100 を分けた持ち分で VRAM 20480 MiB） |
| ドライバ | `nvidia-linux-grid-570` バージョン `570.133.20`。カーネル側は NVIDIA UNIX Open Kernel Module |
| apt のどのリポジトリから来たか | **どこからでもない。** `apt-cache policy nvidia-linux-grid-570` が `/var/lib/dpkg/status` の 1 行しか返さない |
| .deb の現物 | **ディスク上に無い。** `~/` にあるのは `nvidia-linux-grid-535_535.129.03_amd64.deb`（392 MB）で、これは古い版 |
| `~/vGPU/15.4/` の中身 | vGPU 15.4（`525.147.01`）の配布物と PDF 5 本。**入っている 570 とは別世代** |
| ライセンス | `nvidia-smi -q` が `NVIDIA Virtual Compute Server` を `Licensed` と返す |
| ライセンスの出どころ | `/etc/nvidia/ClientConfigToken/`。**root しか読めず、`~/` には控えが無い** |
| `/etc/nvidia/gridd.conf` | `ServerAddress=` が空で `FeatureType=0`。つまりトークンのファイルで効かせている |
| `~/vGPU/license/` のトークン | `..._expire_2024_03_04.tok`。**2024 年に期限切れで、いま効いているものではない** |

`nvidia-smi` が返す期限（`2026-8-24 13:32:46 GMT`）は 1 日単位で先に延びていくもので、
`nvidia-gridd` が動いている限り自動で更新される。**切れかけているわけではない。**
問題は期限ではなく、**トークンと .deb の現物がどちらも root 領域にしか無く、
新しいマシンで同じものを用意する手が自分の側に無いこと**である。

**やること: マシンを用意する人に、次の 2 つを新しいマシンに入れておいてもらう。**

1. `nvidia-linux-grid-570`（`570.133.20`）またはそれ以降の vGPU ゲストドライバ
2. `/etc/nvidia/ClientConfigToken/` のクライアント設定トークン

`nvidia-smi` が動いて `License Status: Licensed` と出れば、この段は終わりである。

---

## 1. 作り替える前に走らせる 1 本のスクリプト

```bash
cd ~/lab/PARC2026_pre
bash solution/tools/machine/backup_machine_state.sh --with-transcripts
```

`~/machine_backup/` に 2 つの tar.gz ができる。**実測で 28 MB と 190 MB。**

| 中身 | 何のためか |
|---|---|
| `manifest/` | 元の姿の実測値。`pip freeze`・`apt-mark showmanual`・`docker inspect`・`nvidia-smi`・HF キャッシュに居たリポジトリの一覧 |
| `manifest/docker-run-recovered.sh` | **常駐コンテナの起動コマンドを `docker inspect` から組み立てたもの。** 6 節と同じ内容が機械的に出る |
| `manifest/git-unsaved.md` | git に取り残しがあるリポジトリと、リモートに無いブランチ |
| `files/` | git にも HF にも無い一次資料。訓練ログ 20 本・評価 57 回分（mp4 を外して 44 MB）・`NOW.md`・`lab/.claude/`・`~/.claude` の設定と memory |
| `SECRETS_TODO.md` | 2 節と同じ内容 |

**`manifest/git-unsaved.md` が空になるまで作り替えを始めないこと。** 空でなければ、
そこに出たリポジトリで commit と push を済ませる。

運ぶ先は HF の private リポジトリでよい（`~/.ssh` の秘密鍵は入っていない）。
scp は 1.7 MB/s しか出ないので、190 MB 側は HF 経由のほうが速い
（[08_hf_transfer_troubleshooting.md](08_hf_transfer_troubleshooting.md)）。

---

## 2. 秘密の 4 件は tar に入れず、手で作り直す

**わざと入れていない。** tar は HF を経由して運ぶので、秘密鍵を入れると運んだ先すべてに
鍵が残る。どれも数分で作り直せる。

| 何 | 元の置き場 | 新しいマシンでどうするか |
|---|---|---|
| SSH 秘密鍵 | `~/.ssh/id_ed25519`・`~/.ssh/id_rsa` | `ssh-keygen -t ed25519` で作り直し、公開鍵を GitHub の Settings → SSH keys に登録する。**古い鍵は GitHub 側から消す** |
| HF のトークン | `~/.cache/huggingface/token` | `hf auth login`。トークンは huggingface.co/settings/tokens で作る（write 権限が要る） |
| Claude Code の資格情報 | `~/.claude/.credentials.json` | `claude` を起動してログインし直す |
| vGPU のライセンストークン | `/etc/nvidia/ClientConfigToken/` | **自分では復旧できない。** 0 節のとおり用意してもらう |

`~/.ssh/known_hosts` は秘密ではないので tar の `files/home/ssh_known_hosts` に入っている。

---

## 3. OS の下ごしらえ

作り替え前の実測値は Ubuntu 22.04.3 LTS・カーネル 5.15.0-92-generic・**2 コア**・
メモリ 28 GB・ディスク 298 GB（うち 196 GB 使用）だった。

**2 コアという数字は評価の所要時間を決めている。** 手元の評価の壁時計は方策ではなく
CPU での画面生成（osmesa というソフトウェア描画）で律速されていて、方策が占めるのは
18〜29% しかない（[../knowledge/14_quantization_and_eval_bottleneck.md](../knowledge/14_quantization_and_eval_bottleneck.md)）。
**コアを増やすと、方策を速くするどの手よりも評価が速くなる。**

### 作り替えで GPU が vGPU でなくなるなら、評価が数倍速くなる余地がある

**CPU 描画を強制しているのは vGPU だからである。** いま `MUJOCO_GL=osmesa` を固定して
いるのは、H100-20C のような vGPU が OpenGL/EGL 描画に対応せず、EGL を指定すると評価が
50〜99 ステップ目で `read_pixels` で落ちた実績があるため
（[../resources/infra_and_docker.md](../resources/infra_and_docker.md) の 1 節）。

新しいマシンの GPU が vGPU の切り出しではなく 1 枚まるごとになるなら、
**`MUJOCO_GL=egl` が通る可能性がある。** 通れば画面生成が GPU 側に移り、
評価の壁時計の 7〜8 割を占めていた部分が縮む。復旧が済んだら、5 件だけの
`--check` を `MUJOCO_GL=egl` で 1 回試して、落ちなければ壁時計を比べる価値がある。
**落ちたら osmesa に戻すだけなので、試すこと自体の危険は無い。**

apt のリポジトリを 2 つ足す。Ubuntu 本体以外はこの 2 つだけだった。

```bash
# nvidia-container-toolkit（コンテナから GPU を見せるため）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# docker（作り替え前は server 29.4.1・ストレージドライバ overlay2 だった）
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
```

自分で入れていた apt パッケージは 40 個あり、Ubuntu の素の構成に含まれるものを除くと
次の 19 個だった（tar の `manifest/apt-manual.txt` が全 40 個の一覧）。

```bash
sudo apt-get update && sudo apt-get install -y \
  awscli cockpit cockpit-podman containerd.io curl \
  docker-buildx-plugin docker-ce docker-ce-cli docker-compose-plugin \
  gnupg hashalot libflashrom1 libftdi1-2 \
  nvidia-container-toolkit openssh-server podman python3.10-venv zip
sudo usermod -aG docker "$USER"    # 入り直すまで効かない
```

`nvidia-linux-grid-570` はこの一覧に入っているが、**0 節のとおり apt では入らない。**

Claude Code は `~/.local/bin/claude` に 2.1.241 が入っていた。`apt` のリポジトリに
`downloads.claude.ai/claude-desktop/apt/stable` も足してあったが、CLI 自体は
公式のインストーラで入る。

ホスト側の `.venv` は HF への上げ下げ専用の小さいもので、Python 3.10.12 に 30 個しか
入っていない。**torch も LeRobot も入っていない**（それらは docker image の中にある）。

tar を `~/machine_backup/machine_state/` に展開してあるものとして、次で戻る。

```bash
cd ~/lab/PARC2026_pre
python3.10 -m venv .venv
grep -v '^Python ' ~/machine_backup/machine_state/manifest/host-venv.txt > /tmp/host-venv-requirements.txt
.venv/bin/pip install -r /tmp/host-venv-requirements.txt
```

`host-venv.txt` は 1 行目が `Python 3.10.12` で、2 行目以降が `pip freeze` の出力に
なっている。`grep -v` はその 1 行目を落としている。

---

## 4. docker image は 3 つ。派生 image は作らなくてよい

作り替え前は image が 7 つあったが、**自分でビルドするのは 3 つだけ**である。

```bash
cd ~/lab/PARC2026_pre
docker build -t parc2026 .                                                    # 配布キット
docker build -t parc2026-train solution/train                                 # 学習（SmolVLA 系）
docker build -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train   # 学習（π0.5）
```

| image | 実測サイズ | 元 | 中身 |
|---|---|---|---|
| `parc2026` | 27.6 GB | ルートの `Dockerfile` | `FROM ubuntu:22.04`。中で `setup.sh` を走らせる。`MUJOCO_GL=osmesa` |
| `parc2026-train` | 9.49 GB | `solution/train/Dockerfile` | SmolVLA 系の学習用 |
| `parc2026-train-pi05` | 9.86 GB | `solution/train/Dockerfile.pi05` | `FROM python:3.12-slim`。torch 2.11.0（cu128）・torchcodec 0.11.1・torchvision 0.26.0・LeRobot v0.6.0・bitsandbytes 0.49.2 |

**残る 4 つ（`parc2026-train-server`・`parc2026-train-pi05-server-*`）はビルドしない。**
[../../solution/tools/eval/run_docker_eval.sh](../../solution/tools/eval/run_docker_eval.sh) が
評価のたびに `parc2026-train-pi05` から派生させて作る。

**`parc2026` のビルドは `setup.sh` 経由で LIBERO-plus を `--depth 1` で clone する。
リビジョンが固定されていない。** 作り替え前の手元は `4976dc3`（`task_classification.json`
に難易度のラベルを足した版）だった。上流が進んでいると評価環境が変わるので、ビルド後に
`git -C LIBERO-plus log --oneline -1` を見て `4976dc3` と違っていたら、
その差が評価の数字に効くかを先に確かめること。

---

## 5. HF から取り直すもの

`~/.cache/huggingface` は 29 GB あった。**中身は tar に入れていない。** 内訳は
`hub/` が 14 GB、`lerobot/` が 15 GB（LeRobot 形式に変換したデータセット）、
`datasets/` が 227 MB。

| 種類 | リポジトリ | 何に使うか |
|---|---|---|
| モデル | `lerobot/pi05_libero_base` | π0.5 の学習の出発点 |
| モデル | `google/paligemma-3b-pt-224` | π0.5 の中の VLM |
| モデル | `siruku6/pi05_stage1_24000`（private） | 24000 ステップの重み。ローカルの `work/train/pi05_full_local` と同じもの |
| モデル | `siruku6/pi05_full_runpod`（private） | RunPod で 3000 ステップ回した重み |
| モデル | `siruku6/smolvla-libero-plus-finetunes` | `weak3x_20000` |
| データセット | `lerobot/libero_plus` | 学習データ。**リビジョン `f3f49f426d75030177b18778374005bc12ccd588` に固定してある**（`solution/train/train_pi05.py` の `_DATASET_REVISION`） |
| データセット | `Sylvest/LIBERO-plus` | `assets.zip`。`setup.sh` が取る |
| データセット | `Sylvest/libero_plus_lerobot` | 変換前のもの |

提出 zip 4 本（`submission_361.zip`・`submission_pi05_n16.zip`・`submission_rp3000_blend0.zip`・
`submission_rp3000_i20.zip`）は `siruku6/parc2026-submission`（private）にあり、
**ローカルには現物が無い。**

キャッシュに `models--siruku6--parc2026-archive` の残骸があるが、**このリポジトリは
2026-08-18 に消した**（commit `c3b8f1e`）。取り直せないし、取り直す必要も無い。

### 重みを評価に使うときの落とし穴 2 つ

作り替えとは関係なく毎回効くので、復旧直後にも同じところで詰まる。

1. **重みの隣に `vlm/` ディレクトリが無いと評価が 5 秒で死ぬ。** 学習が吐く
   チェックポイントには `vlm/` が付かない。`pi05_stage1_24000` と `pi05_full_runpod` の
   HF 側には入っているので、そこから**コピーする**（symlink は不可）。
2. **`config.json` の `compile_model` が `true` だと評価が必ず落ちる。** 冷えた
   inductor のキャッシュを埋めるのに 1177.9 秒かかる（`work/quant_sweep/RANK_CHECK_REPORT.md`
   の実測）。待ちの上限は 2 か所にあり、どちらにも間に合わない。
   **`false` に直すのが唯一の手である。**

   | どこ | 上限 | 変えられるか |
   |---|---|---|
   | 手元の評価 [`run_docker_eval.sh:238`](../../solution/tools/eval/run_docker_eval.sh) の `for _ in $(seq 1 300)` | 300 秒 | スクリプトを直せば変えられる |
   | 配布キットの [`evaluate.py:123`](../../evaluate.py) の `os.environ.get("SERVER_TIMEOUT", "120")` | 既定 120 秒 | 環境変数で上げられるが、**本番採点は配布キットの [`evaluate.sh:25`](../../evaluate.sh) が `SERVER_TIMEOUT=120` を固定して呼ぶので、本番では上げられない** |

---

## 6. 常駐コンテナ 2 つの起動コマンド

**リポジトリのどの手順書にも、この 2 つの完全な形は書かれていなかった。**
[../../solution/tools/quant/run_quant_sweep.sh](../../solution/tools/quant/run_quant_sweep.sh)
の案内は `docker run -d --gpus all --name quant_train ... parc2026-train-pi05 sleep infinity`
と `...` で省略されている。下は動いている現物の `docker inspect` から起こしたものである。

```bash
# ① π0.5 の学習・量子化の実験用。GPU を使う
docker run -d --name quant_train \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --network bridge \
  -v "$HOME/.cache/huggingface:/hf_home" \
  -v "$HOME/lab/PARC2026_pre/work/train:/out" \
  parc2026-train-pi05 sleep infinity

# ② 配布キット側の確認用。GPU もネットワークも使わない
docker run -d --name parc2026_verify \
  --network none \
  parc2026 sleep infinity
```

image が持っている設定（自分で `-e` を足す必要は無い）は次のとおり。

| | `quant_train` | `parc2026_verify` |
|---|---|---|
| image | `parc2026-train-pi05` | `parc2026` |
| 作業ディレクトリ | `/opt/pi05` | `/workspace` |
| 実行ユーザー | `1000:1000` | root |
| ネットワーク | `bridge` | `none` |
| GPU | 全部見せる | 見せない |
| 主な環境変数 | `HF_HOME=/hf_home`・`OUTPUT_ROOT=/out`・Python 3.12.14 | `MUJOCO_GL=osmesa`・`LIBERO_ROOT=/workspace/LIBERO-plus`・`PYTHONPATH=/workspace/LIBERO-plus:/workspace:/workspace/compe` |

**`--user "$(id -u):$(id -g)"` を外さないこと。** これを付けずに回すと、bind mount した
`work/train/` の中身が root 所有で残り、後から消せなくなる
（[07_train_pi05_local.md](07_train_pi05_local.md) の同じ節）。

**この 2 つは Claude では起こせない。** このマシンでは組織の制約で Bash ツールからの
`docker run`（新規コンテナの作成）が拒否される。`docker exec`（既にあるコンテナへの実行）は
通る。だから**復旧のこの段だけは人が叩く**必要がある。

---

## 7. 復旧できたことを確かめる 3 つのコマンド

上から順に、軽いものから重いものへ並べてある。

```bash
# ① GPU がコンテナから見えるか（数秒）
docker exec quant_train nvidia-smi --query-gpu=name,driver_version --format=csv

# ② 配布キット側の単体テスト（4 ファイル・76 件。実測 18 秒）
docker exec parc2026_verify python -m pytest /workspace/tests -q

# ③ 評価の配線が通るか。5 件だけ流す（5 分）
bash solution/tools/eval/run_docker_eval.sh --check
```

① と ② は `docker exec` なので Claude でも流せる。**③ は中で `docker run` を呼ぶので
人が叩く必要がある**（組織の制約で Claude の Bash ツールからの `docker run` は拒否される）。

③ が通れば復旧は終わりである。**84 件の本番相当を流して作り替え前の数字と比べる意味は薄い。**
2026-08-23 に**同じ行・同じ設定で 3 回**流して乱数の床を測ってあり、次のとおり大きい
（[../knowledge/06_step_count_drives_score.md](../knowledge/06_step_count_drives_score.md)
の「この指標の床を初めて測った」節）。

| 見るもの | 3 回の値 | 振れ幅 |
|---|---|---|
| 32 行すべての合計歩数 | 6,833 / 6,489 / 5,141 | **+32.9%** |
| 打ち切りに一度も達しなかった 6 行の合計（Base 側） | 828 / 877 / 793 | +10.6% |
| 同（摂動側） | 692 / 779 / 738 | +12.6% |

**打ち切りが混ざらない行に絞っても床は約 11% ある。** さらに「300 ステップに達したか」
という成否そのものが、24 行のうち 14 行で 3 回の結果が割れた。**作り替え前後の数字の差が
この幅より小さければ、何も語らない。**

**終了コードは成否の判定に使わないこと。** 2 つの理由がある。

- 評価は途中で落ちても終了コード 0 を返す。全件そろったかを最初に見る
  （[03_gpu_eval.md](03_gpu_eval.md) の同じ節）
- 長時間の `docker exec` を Claude の Bash ツール経由で流すと、成功していても
  終了コード 120 が返る（6 時間の学習が何度も「失敗」と記録された）。
  **コードで裏づけた話ではなく、実際に踏んだ症状である**

どちらも、出来たファイルとログの完了行で判断すれば避けられる。

---

## 8. 取り直さないと決めたもの

**判断は 2026-08-23 にユーザーが下した。** 消えるに任せる。

| 何 | 大きさ | なぜ取り直さないか |
|---|---|---|
| 量子化・compile の実験チェックポイント 5 本 | 55 GB | 量子化の候補 5 つはすべて撤退済み。`-53%` は `torch.compile` の効果だと対照実験で確定していて、float8 は速度もメモリも動かさなかった |
| `work/train/pi05_full_local` の `training_state` | 2.1 GB | 24000 ステップから学習を再開する予定が無い。重み側は HF にある |
| 評価の mp4 470 本 | 1.26 GB | 数字は log・json・csv 側にあり、そちらは tar に入っている |
| `work/scripted` で作ったエピソード | 200 MB | `record_tier2_robot_init.sh` などで作り直せる |
| docker image 7 本 | 87 GB | Dockerfile が git 管理下。**そもそも CPU が変わるので作り直しになる** |
| `~/.cache/huggingface` | 29 GB | 5 節のとおり HF から取り直す |
| `work/LIBERO-plus`・`work/lerobot-src`・`.venv`・`.mypy_cache`・`~/.vscode-server` | 2.4 GB | すべて生成物 |

---

## 確認したい仮定

| 仮定 | 外れた場合に変わること |
|---|---|
| 新しいマシンにも vGPU ドライバとライセンストークンが用意される | **0 節が塞がると復旧そのものが止まる。** GPU 無しで進める範囲は配布キット側の単体テストだけになる |
| `~/vGPU/15.4`（3.1 GB）と `~/nvidia-linux-grid-535...deb`（392 MB）は取り直せる | 取り直せないなら、この 3.5 GB も持ち出す側に移す。ただしどちらも**いま入っている 570 とは別世代**なので、そのままでは使えない |
| `~/libero_plus_episodes.parquet`（14 MB）は手で作ったものではない | 手で作ったものなら作り方が失われる。判断が付かないので tar には入れてある |
| 上流の LIBERO-plus が `4976dc3` から進んでいても評価の数字は変わらない | 4 節のとおり、ビルド後に差分を見る必要が出る |

## あとで見る資料

| 知りたいこと | 場所 |
|---|---|
| どの重みが HF のどこにあるか | [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md)・[../resources/huggingface.md](../resources/huggingface.md) |
| 計算資源と image・コンテナの現状 | [../resources/infra_and_docker.md](../resources/infra_and_docker.md) |
| 学習の回し方 | [01_retrain.md](01_retrain.md)・[07_train_pi05_local.md](07_train_pi05_local.md) |
| 評価の回し方 | [03_gpu_eval.md](03_gpu_eval.md) |
| 重みを別のマシンへ運ぶとき | [08_hf_transfer_troubleshooting.md](08_hf_transfer_troubleshooting.md) |
| 過去に踏んだ事故 | [../knowledge/12_workbench_accidents.md](../knowledge/12_workbench_accidents.md) |
