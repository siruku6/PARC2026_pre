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

### 先にやること 3 つ（Pod を作る前）

**(1) 押してあるイメージの `train_pi05.py` は 8/14 のものであり、`--camera-viewpoint-augment`
が入っていない。** `Dockerfile.pi05` はビルド時に `train_pi05.py` を `COPY` するので、
イメージを作り直して push し直すか、Pod の中でファイルを差し替えるかになる。
**押し直しは 1 時間と 10 GB の上りを使うので、差し替えを推す。**

Pod へ入ったあと、このマシンから 3 つを送る（合計 200 KB 未満）。

```bash
# JAIST サーバ側で実行する。<pod> は 5.3 節で確認した接続先
cd /home/ubuntu/lab/PARC2026_pre/solution/train
scp -P <ポート> train.py train_pi05.py root@<ホスト>:/workspace/
scp -P <ポート> -r experiments root@<ホスト>:/workspace/
```

**Pod 側では `/workspace` から起動する**（`train_pi05.py` は隣の `train.py` と
`experiments/` を import するので、3 つが同じ場所に無いと動かない）。

**(2) 出発点の重みを Pod へ落とす。** `rp3000` は HF の非公開リポジトリにある。

```bash
pip install -q -U "huggingface_hub[cli,hf_xet]"
hf auth login
hf download siruku6/pi05_full_runpod --local-dir /workspace/rp3000
du -sh /workspace/rp3000        # 8.8 GB になる
```

**(3) チェックポイントの間引きを先に仕掛ける。** フルパラメータ学習の
チェックポイントは **1 個 25 GB** で、LeRobot は古いものを消さない。
Volume Disk 150 GB のうち、初期重みとデータセットで 30 GB 使うので、
**間引き無しだと 4 個目で尽きる。**[howto/06](../howto/06_runpod_pi05.md) 11 節の `nohup` の 1 行をそのまま使う。

### 本番のコマンド

```bash
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

