# 15. RunPod でフルパラメータ ＋ 画像拡張を 1 回回す（設計）

**この 1 回分の設計だけを持つ文書である。** 何度でも使う手順（イメージのビルドと push・
Pod の作り方・batch の探り方・チェックポイントの間引き・持ち帰り）は
[../howto/06_runpod_pi05.md](../howto/06_runpod_pi05.md) の 4〜12 節にある。
**こちらには「今回どう設定するか」と「なぜそうするか」しか書かない。**

作成: 2026-08-27。


**1 回目（8/17）と違うのは 2 つだけである。出発点が `rp3000` になり、画像拡張が入る。**
**手順そのものは [howto/06](../howto/06_runpod_pi05.md) の 4〜12 節がそのまま使える。**この節はその 2 つの差分と、
今回決めた数字だけを書く。

| | 1 回目（8/17・実行済み） | **2 回目（8/28）** |
|---|---|---|
| 出発点 | `pi05_stage1_24000`（行動側だけ 24,000 歩） | **`pi05_full_runpod`（＝ `rp3000`。1 回目の結果そのもの）** |
| 動かす範囲 | 全部（4.14B） | **全部（同じ）** |
| 画像拡張 | 無し | **`--camera-viewpoint-augment`（8 種のうち最大 4 種を毎回）** |
| batch | 64 | **64（同じ。128 は 75.77GB で落ちた）** |
| 学習率 | 5e-6 → 2.5e-6 | **5e-6 → 2.5e-6（同じ）** |
| 出た点 | 予選 **0.485**・本選 Track1 0.370 | ？ |

**なぜ画像拡張を RunPod でやるのか。** 手元の H100-20C（20 GB）では画像側を学習できない。
`train_expert_only=true` は PaliGemma 全体（画像側を含む）を凍結する実装であり
（[modeling_pi05.py](../../work/lerobot-src/src/lerobot/policies/pi05/modeling_pi05.py) の
`_set_requires_grad`）、画像側を動かすには `false` にして 4.14B 全部を対象にするしかない。
AdamW の状態だけで 33 GB 必要になるので 20 GB には載らない。**画像を歪めるなら、
それを見る側も学習できる場所でやる。**

### 時間と費用（1 回目の実測 6.2 smp/s から計算した）

**A100 80GB PCIe・batch 64・全部学習で 6.2 smp/s**（[howto/06](../howto/06_runpod_pi05.md) 13 節の実測表）。
1 歩あたり 10.32 秒である。

| 歩数 | 見るコマ数 | 所要 | 費用（$1.8〜2.2/時） |
|---|---|---|---|
| 3,000（1 回目と同じ） | 192,000 | **8 時間 36 分** | $15〜19 |
| **4,500** | **288,000** | **12 時間 54 分** | **$23〜28** |
| 6,000 | 384,000 | 17 時間 12 分 | $31〜38 |

**画像拡張を入れると CPU 側の変換が増えるので、これより遅くなる可能性がある。**
`--num-workers 16` にして、`data_s`（データ待ちの秒数）が `updt_s` に対して
大きくなっていないかを最初の 100 歩で見る。**大きければ workers を増やす。**

### Pod の作り方は 2 通りある。**どちらでも同じ環境になる**

**イメージは `sirukufarios/parc2026-train-pi05` である**（2026-08-14 に push。
名前を記録していなかったので 8/27 に
[../resources/infra_and_docker.md](../resources/infra_and_docker.md) へ書いた）。

| 道 | やること | 所要 | どんなときに |
|---|---|---|---|
| **A（推奨）: 8/14 のイメージを使う** | **`sirukufarios/parc2026-train-pi05`** で Pod を作る | **15 分** | **こちらでよい。**作り直す必要は無い（下） |
| **B（確実）: 公式テンプレート ＋ `runpod_setup.sh`** | 「RunPod PyTorch」テンプレートで Pod を作り、`bash runpod_setup.sh` を 1 回流す | **35 分**（pip の時間） | イメージが無い／名前が分からないとき |

**古いイメージのままでよい。**8/14 以降に `Dockerfile.pi05` へ入った変更は量子化の
配線だけ（`bitsandbytes`・`torchao` の追加と `experiments/` の COPY）で、量子化の
5 候補はすべて廃止した。現在の `train_pi05.py` は `from experiments import quantization`
を実行するが、**その `__init__.py` は `argparse`・`os`・`pathlib` しか import せず、
重い候補は量子化のオプションを渡したときだけ遅延して読まれる。**
**古いのは焼き込まれた `train_pi05.py` だけで、それは `/workspace` の版で上書きする。**

**B は確実に通る。**`runpod_setup.sh` は torch 2.11.0+cu128・torchcodec 0.11.1・
torchvision 0.26.0・lerobot v0.6.0 を版まで固定して `/opt/lerobot` へ入れる。
**`Dockerfile.pi05` と同じ組み合わせである。**イメージを作り直して 9.7 GB を
上げ直す必要は無い（上り回線が読めないので、そちらの方が危ない）。

### 先にやること 3 つ（Pod を作る前ではなく、Pod へ入った直後）

**(1) コードを落とす。**押してあるイメージの `train_pi05.py` は 8/14 のもので、
`--camera-viewpoint-augment` が入っていない。**scp は要らない。HF に置いた。**

```bash
pip install -q -U "huggingface_hub[cli,hf_xet]"
hf auth login
cd /workspace
hf download siruku6/parc2026-submission runpod_code_260827.tar.gz --local-dir /tmp/code
tar -xzf /tmp/code/runpod_code_260827.tar.gz -C /workspace
ls /workspace          # train.py train_pi05.py experiments/ runpod_setup.sh prune_checkpoints.sh
```

`sha256` は `128b8fd34e1ba064ea3c196f0c35c64272cc1c2bdcb28205b4790c9e3b3df5e4`。
**`train_pi05.py` は隣の `train.py` と `experiments/` を import するので、3 つが
同じ場所に無いと動かない。**だから `/workspace` に展開して `/workspace` から起動する。

**(2) 出発点の重みを落とす。**

```bash
hf download siruku6/pi05_full_runpod --local-dir /workspace/rp3000
du -sh /workspace/rp3000        # 8.8 GB
```

**(3) チェックポイントの置き場を数える。**フルパラメータのチェックポイントは
**1 個 25 GB**。`--save-freq 1500` で 1,500 / 3,000 / 4,500 の 3 個が残るので
**75 GB** 使う。Volume Disk 150 GB のうち初期重みとデータセットで 30 GB 使うので、
**105 GB で収まる。間引きは要らない。**

```bash
df -h /workspace
```

**105 GB を超える見込みになったら**、[howto/06](../howto/06_runpod_pi05.md) 11 節の
`nohup` の 1 行で間引きを仕掛ける。

### 本番のコマンド

**`--num-workers` は Pod の vCPU 数を見てから決める**（[howto/06](../howto/06_runpod_pi05.md)
0 節）。`nproc` の半分から 16 の小さい方でよい。

```bash
nproc                      # ここを見てから下の 16 を決める
tmux new -s train
cd /workspace
python train_pi05.py --run-name pi05_full2_aug \
    --init-from /workspace/rp3000 \
    --camera-viewpoint-augment \
    --steps 4500 --batch-size 64 --num-workers 16 \
    --learning-rate 5e-6 --save-freq 1500 \
    --freeze-vision-encoder false --train-expert-only false
```

`Ctrl+b` → `d` で抜ける。戻るのは `tmux attach -t train`。

**渡した 2 つのフラグは、そのまま LeRobot へ届く**（`train_pi05.py:329-330` で
`--policy.freeze_vision_encoder=false --policy.train_expert_only=false` を組み立てる）。
**`false` は小文字である。**

### 立ち上がったことを 3 点で確かめる（最初の 20 歩）

| 見るもの | 期待する値 | 外れたときの意味 |
|---|---|---|
| `All keys loaded successfully!` | 出る | `--init-from` が効いていない |
| `num_learnable_params` | **4143404816（4B）** | **693M なら画像側が凍ったままである。**`--train-expert-only false` が渡っていない |
| 20 歩目の `loss` | **0.20 前後** | 1.5 を超えていたら重みを読めていない（最初から学習し直している） |

**`num_learnable_params` が一番大事である。** ここが 693M だと、今回の狙い
（画像側も学習する）がまるごと空振りになる。

### 持ち帰り

**`scp` は 1.7 MB/s しか出ない。HF の非公開リポジトリを使う**（[howto/06](../howto/06_runpod_pi05.md) 13 節の実測で
8.8 GB が 1 時間超 → 9.4 GB が 16 分）。

```bash
hf upload siruku6/pi05_full2_aug /workspace/out/pi05_full2_aug/checkpoints/004500/pretrained_model . --private
```

**降ろしたあと、評価の前に `vlm/` を足すこと。** 学習の出力には入らないので、
`rp3000` から**コピー**する（リンクではなくコピー）。無いと評価は 5 秒で落ちる。

```bash
cp -r work/artifacts/pi05_rp3000/vlm <降ろした先>/
```

### この回で比べられること・比べられないこと

**判断に使うのは `rp3000` との比較である。**手元の Track2 8 行（版 4.6.0 の設定）で
`rp3000` は成功 0.750・衝突 0.250・歩数 122.7 なので、そこと並べる。

**ただし 1 変数ではない。**`rp3000` との差は「フルパラメータの追加学習」と
「画像拡張」の 2 つある。**切り分けは、同じ日に JAIST で回している
行動側だけ 24,000 歩の結果と合わせて読む。**

| 見えた形 | 読み |
|---|---|
| RunPod が良く、JAIST も良い | 追加学習そのものが効いている。拡張の寄与は不明 |
| RunPod が良く、JAIST は変わらない | **画像側を学習したこと（＋拡張）が効いている** |
| RunPod が変わらない | 追加学習では動かない。設定側と学習データ側へ戻る |

**拡張なしのフルパラメータ学習を対照として回すと 1 変数になるが、費用が倍になる。
今回は回さない。**

