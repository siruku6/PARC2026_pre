# 19. `libero_combined_bowl5_initpose_l90clean` で学習する（RunPod・π0.5 と MolmoAct2）

TOC:
- [0. このファイルの役割](#0-このファイルの役割)
- [1. どちらの土台で回すか](#1-どちらの土台で回すか)
- [2. 共通の準備](#2-共通の準備)
- [3. π0.5 で回す](#3-π05-で回す)
- [4. MolmoAct2 で回す](#4-molmoact2-で回す)
- [5. 見積もりの根拠](#5-見積もりの根拠)
- [6. 詰まったときの分かれ道](#6-詰まったときの分かれ道)

## 0. このファイルの役割

**9/16 に作った `libero_combined_bowl5_initpose_l90clean` で 1 本学習を回すまでを、
迷わず打ち切れる形にする。**土台は 2 つ選べる（π0.5 と MolmoAct2）ので、
**選び方（1 節）→ 共通の準備（2 節）→ 選んだほうの手順（3 節か 4 節）**の順に読めばよい。

**このファイルが持たないもの。**RunPod のアカウントの作り方・料金の感覚・イメージの作り方は
[06_runpod_pi05.md](06_runpod_pi05.md) にある。π0.5 の学習設計の経緯（なぜ 5,500 歩・なぜ 2.5e-5 か）は
[16_bowl_regen_train_eval.md の 10 節](16_bowl_regen_train_eval.md#10-第-3-段初期姿勢を振った-combined-のデータで全パラメータ学習するblackwell914)、
MolmoAct2 を土台に替える話の全体像は
[talk/inter_sessions/032](../talk/inter_sessions/032_molmoact2_libero_adoption.md) にある。
**このファイルは「新しいデータで 1 本回す」ことだけを持つ。**

**使うデータ。**HF `siruku6/libero_combined_bowl5_initpose_l90clean`（非公開・21,642 本・
3,431,137 コマ・111 課題・27.8 GB）。`libero_combined_bowl5_initpose` との違いは
**libero_90 部分の汚れた 250 本を、同じ課題・同じ本数の綺麗な手本に置き換えたことだけ**である
（詳細は [resources/datasets.md](../resources/datasets.md)）。本数・課題ごとの本数は変わらず、
コマ数だけ 5,803 減っている。**置き換えた 250 本はデータセットの末尾に付いているので、
本の番号で前の版と対応は取れない。**

## 1. どちらの土台で回すか

**今日 1 本回して評価まで進めたいなら π0.5 を選ぶ。**MolmoAct2 は 1 サンプルあたり 4〜12 倍重く、
同じサンプル数を見るのに枚数を増やさないと日をまたぐ。

| | **π0.5**（`pi05_libero_base` から） | **MolmoAct2**（`MolmoAct2-LIBERO-LeRobot` から） |
|---|---|---|
| 1 歩（batch 64・全パラメータ） | **6.85〜10.3 秒**（実測） | **40〜80 秒**（推定。1 枚では batch 64 が載らない） |
| 5,500 歩（＝35.2 万サンプル） | **10.5〜15.7 時間** | 61〜122 時間（1 枚）／31〜61 時間（2 枚） |
| 要る VRAM | 約 49 GB（batch 64） | 60.1 GiB（batch 32）／約 80 GiB（batch 64・外挿） |
| 最低限の板 | 80 GB 1 枚 | **80 GB 2 枚**（48 GB は全パラメータ学習では不可） |
| 手順の確かさ | **何度も回して実測がある** | **未実行。**スクリプトは書いたが 1 度も通していない |
| 期待 | **提出済みの最良重み `pi05_bowl5_full` から新しいデータへ寄せる**（3.2 節）。比べる相手はその出発点自身 | 素の重みが Track2 で π0.5 より上（0.3858 対 0.3674）。ただし提出には追加学習が必須 |

**歩数を減らせば MolmoAct2 も現実的になる**（LIBERO を解ける方策から始めるので、
公式の 320 万サンプルは要らない）。その見積もりは 4 節にある。

## 2. 共通の準備

### 2.1 データに `bowl5_expected.json` を足す（**どちらの手順でも必須**）

`runpod_initpose_ft.sh` も `runpod_molmoact2_ft.sh` も、降ろしたデータが手元で組んだものと同じかを
`$DS/bowl5_expected.json` と突き合わせて確かめる。**新しいデータセットにはこのファイルがまだ無い**
ので、先に 1 つだけ上げておく（この機械には 9/16 に作成済み）。

```bash
cd /home/ubuntu/src/PARC2026_pre/work/artifacts/libero_combined_bowl5_initpose_l90clean
hf upload siruku6/libero_combined_bowl5_initpose_l90clean \
    bowl5_expected.json bowl5_expected.json --repo-type dataset
```

中身は `{"total_episodes": 21642, "total_frames": 3431137, "total_tasks": 111}` である。

### 2.2 HF の空きを確かめる

**チェックポイントの置き場は HF だけで、Pod を消すとローカルは消える。**上がらなければ
学習が丸ごと失われる（[16 の 10.1 節](16_bowl_regen_train_eval.md#101-打つ前に--hf-の空きここで詰まると重みが消える)）。

**1 個あたりの大きさと、空きが尽きたときの振る舞いは 10.1 節にある。**このファイルが足すのは
**MolmoAct2 のチェックポイントが 1 個約 11 GB**（π0.5 は約 9.4 GB）ということだけである。

**保存の個数は `SAVE_FREQ` で決まる。**空きが心細いときは最後の 1 個だけ保存する。
使用量は `hf` の API で数えられるので、**打つ直前にその場で数える**（書き写した値はすぐ古くなる）。

## 3. π0.5 で回す

### 3.1 Pod を作る

欄の埋め方は [16 の 10.6 節 (2)](16_bowl_regen_train_eval.md#106-runpod-で回すときblackwell-が空かない日の代わり914-夜に準備) と同じで、
**Volume だけ 150 GB → 160 GB にする**（データが 27.8 GB あるため）。

**GPU ごとの 1 歩の秒数と 5,500 歩の所要時間も、同じ 10.6 節 (2) の表を見る**（Blackwell 96 GB 6.85 秒・
A100 80 GB 10.29 秒の実測つき）。この走りで違うのは Volume の大きさだけである。

### 3.2 打つコマンド

**`pi05_bowl5_full`（9/13 に提出した最良の重み）から続けて、新しいデータへ寄せる**（ユーザ 9/16）。
学習率は **warmup 200 歩で 5e-5 まで上げ、そこから cosine で 5e-6（頂点の 1 割）まで下げる**。
**1 本で完結させる形**である。

**なぜこの形か。**

- **5e-5 は上流 openpi が微調整で使う値**である（`pi05_libero`・`pi05_droid` とも peak 5e-5。
  [reference/38](../reference/38_lr_lineage.md)）。
- **ただし上流は「warmup のあと一定」で、それは EMA 0.999 と組で使われている。**LeRobot v0.6.0 の
  π0.5 学習に EMA は無いので、一定のまま止めると**止めた瞬間の揺れをそのまま持った重み**になる。
  1 本で終えるので、頂点の 1 割まで下げて締める。
- **何回かに分けて積み上げるなら `DECAY_LR=5e-5`（＝下げない）にする。**その場合は歩数がそのまま
  足し算になる代わりに、最後に短い「冷やし」の走り（数百歩を 5e-6）を足して締める。

```bash
export HF_HOME=/workspace/hf_home
hf auth login
hf download siruku6/pi05_code --local-dir /workspace/pi05_code

# まず速さだけ測る（上げ先を作らず、100 歩で片付ける）
TRIAL=1 BATCH=64 \
  DATA_REPO=siruku6/libero_combined_bowl5_initpose_l90clean \
  DATASET_ID=local/libero_combined_bowl5_initpose_l90clean \
  DS=/workspace/data/libero_combined_bowl5_initpose_l90clean \
  bash /workspace/pi05_code/solution/train/runpod_initpose_ft.sh

# 本番（pi05_bowl5_full の 005500 から・warmup 200 → 5e-5 → 5e-6）
RUN=pi05_l90clean_from_bowl5 \
  INIT_REPO=siruku6/pi05_bowl5_full INIT_STEP=005500 \
  STEPS=5500 BATCH=64 LR=5e-5 WARMUP=200 DECAY_LR=5e-6 SAVE_FREQ=1500 \
  DATA_REPO=siruku6/libero_combined_bowl5_initpose_l90clean \
  DATASET_ID=local/libero_combined_bowl5_initpose_l90clean \
  DS=/workspace/data/libero_combined_bowl5_initpose_l90clean \
  bash /workspace/pi05_code/solution/train/runpod_initpose_ft.sh
```

- **`DECAY_LR` は 9/16 に `runpod_initpose_ft.sh` へ足した口である。**渡さなければ今までどおり
  LeRobot 既定の 2.5e-6 まで下げる。`LR` と同じ値を渡すと下がらない（`alpha = decay_lr / peak_lr` が 1）。
- **5e-5 は `pi05_bowl5_full` を作ったときの頂点（2.5e-5）の 2 倍である。**上流が 5e-5 を使うのは
  汎用の `pi05_base` から特化させるときで、今回は**すでに特化し終えた重み**から始める。
  丼 5 課題で作り込んだ掴み方が薄まる方向に振れうるので、**最初のチェックポイント（1,500 歩）を
  先に VAL72 で測り、`pi05_bowl5_full` から落ちていないことを見てから残りを回す**こと。
  落ちていたら `LR=2.5e-5` へ下げて打ち直す。
- **正規化統計が変わる。**出発点は `libero_plus_bowl5`（40 課題）の統計で学習されており、
  今回は 111 課題の統計になる。差は行動 0.99〜1.06 倍・状態 0.97〜1.03 倍（9/14 実測）と小さいが、
  **入力の目盛りが少し変わることは起きる。**
- **`SAVE_FREQ` は HF の空きと相談する。**1,500 なら 4 個（約 37.6 GB）。
- 種は `train_pi05.py` の呼び出しで 42 に固定されている。
- **`RUN` は過去の走りと別の名前にする。**同じ名前だと HF の同じ `005500/` を上書きし合う。

**この走りは C（`pi05_combined_initpose_full`）との 1 変数比較ではない**——出発点も学習率も違う。
C と 1 変数で比べたいときは、`INIT_REPO=base STEPS=5500 BATCH=64 LR=2.5e-5 WARMUP=500`（`DECAY_LR` は渡さない）
で別に 1 本打つ。

### 3.3 離席の前に確かめる 5 つ

スクリプトが最後に出す。読み方は [16 の 10.3 節](16_bowl_regen_train_eval.md#103-離席の前にスクリプトの出力で-5-つ確かめる) と同じだが、
**このデータでの期待値だけ違う**。

| 見るもの | 期待 |
|---|---|
| 実際／期待の本数 | `21642 3431137 111` が両側に出る |
| `num_learnable_params` | `4143404816`（全パラメータ） |
| `dataset.num_episodes` | `21642` |
| 出発点 | `初期重みにローカルのチェックポイントを使う` がログに出る（`INIT_REPO=base` のときは出ない） |
| GPU メモリ | batch 64 で約 49 GB |
| 上げるループの開始行 | `開始 RUN=...` が `upload_ckpt_$RUN.log` にある |

**学習率も 1 度だけ確かめておく。**ログの `lr:` は 20 歩の平均なので、warmup 200 歩なら
200 歩の直後で 5e-5 に届いているはずである。届いていなければ `DECAY_LR` か `WARMUP` の渡し漏れを疑う。

### 3.4 戻ったら

重みの取り込みと評価は既存の手順をそのまま使う——
[16 の 6.4 節](16_bowl_regen_train_eval.md#64-終わったら重みを取り込む)（取り込み）と
[reference/37](../reference/37_local_eval_for_weight_selection.md)（VAL72 での選定）。

**比べる相手は出発点そのもの、`pi05_bowl5_full` である。**同じ評価セットで並べて、
上回っていれば「新しいデータへ寄せたことが効いた」と読める。**下回っていたら学習率が高すぎた可能性が
先に立つ**ので、`LR=2.5e-5` で打ち直して同じ比較をやり直す。
参考として `pi05_combined_initpose_full`（C）も同じ表に並べてよいが、
**出発点も学習率も違うので、差の原因をデータだけに帰することはできない。**

## 4. MolmoAct2 で回す

### 4.1 π0.5 と違うところ

**スクリプトは [`solution/train/runpod_molmoact2_ft.sh`](../../solution/train/runpod_molmoact2_ft.sh)（9/16 に作成・未実行）。**
`train_pi05.py` は `--policy.type=pi05` を固定で組み立てるので使えない。中で `lerobot-train` を直に叩く。

| 何 | なぜ要るか |
|---|---|
| `--rename_map` | 出発点は `observation.images.image` / `wrist_image` を期待し、こちらのデータは `front` / `wrist` |
| `--policy.scheduler_decay_steps` | チェックポイントの設定が `null` で、draccus が int に読めず起動時に落ちる |
| `CKPT_MODE=small`（既定） | 起動時に捨てられる素の重み 21.8 GB を読ませないための直し。`full` なら素直に降ろす |
| `peft` | **LoRA を使うときだけ。**全パラメータ学習なら不要（学習イメージには入っていない） |

**勾配累積は lerobot v0.6.0 に無い。**実効 batch は `BATCH × 枚数`だけで決まる。

### 4.2 Pod を作る

| batch（1 枚あたり） | 全パラメータ学習のメモリ | 80 GB（74.5 GiB） | 96 GB（89.4 GiB） |
|---|---|---|---|
| 8 / 16 | 48.3 / 49.8 GiB（実測） | 入る | 入る |
| 32 | 60.1 GiB（実測） | **入る** | 入る |
| 64 | 約 80 GiB（外挿） | 入らない | 入るがギリギリ |

- **48 GB の板は全パラメータ学習では使えない**（batch 8 でも 48.3 GiB 要り、44.7 GiB に収まらない）。
  枚数を増やしても各 GPU がモデル全体を持つので変わらない。使うなら LoRA か action expert だけになる。
- **おすすめは 80 GB × 2 枚で、1 枚あたり batch 32（実効 64）。**1 枚 batch 64 と同じ仕事量を半分の時間で終える。
  SXM（NVLink）を選べるほうがよい（毎歩 5B 分の勾配を足し合わせる）。
- Volume 200 GB・Shared Memory 8 GB 以上・CPU 16 コア以上。

### 4.3 試し走り → 本番

```bash
# 実測を取る（π0.5 と同じ summarize_speed.py が 1 歩の秒数・データ待ち・メモリの山を出す）
for b in 16 32; do
  TRIAL=1 GPUS=2 BATCH=$b bash /workspace/pi05_code/solution/train/runpod_molmoact2_ft.sh
done

# 本番（実効 batch 64・2,000 歩・500 歩ごとに保存）
RUN=molmoact2_l90clean GPUS=2 BATCH=32 STEPS=2000 SAVE_FREQ=500 \
  bash /workspace/pi05_code/solution/train/runpod_molmoact2_ft.sh
```

**歩数の目安は 2,000〜4,000 歩（実効 batch 64 で 12.8 万〜25.6 万サンプル）。**
公式の LIBERO 微調整は 320 万サンプルだが、それは素の土台を LIBERO 特化にする量である。
今回は既に LIBERO を解ける方策を摂動入りのデータへ寄せるだけなので、こちらの π0.5 の経験
（2,000 歩 × 64 ＝ 12.8 万サンプルでデータ差が判定できた）のほうが近い。

| 枚数（80 GB 級） | 1 歩（実効 batch 64） | 2,000 歩 | 4,000 歩 |
|---|---|---|---|
| 1 枚 | 40〜80 秒 | 22〜44 時間 | 44〜89 時間 |
| **2 枚** | **20〜40 秒** | **11〜22 時間** | 22〜44 時間 |
| 4 枚 | 10〜20 秒 | 6〜11 時間 | 11〜22 時間 |

### 4.4 学習率は出発点の系譜と同じにしてある

スクリプトの既定（本体 1e-5 ／ ViT 5e-6 ／ つなぎ 5e-6 ／ action expert 5e-5）は、出発点の重みに
同梱の `train_config.json` と同じである。**この値は系譜の 4 段目以降ずっと変わっていない**ので、
「効かない大きさ」ではない。**段階ごとの一覧と出典は
[reference/38 の 3 節](../reference/38_lr_lineage.md#3-molmoact2-の系譜と学習率)にある。**

**上げるなら action expert だけ**（`EXPERT_LR=1e-4`。公式の起動スクリプトの既定値がこれ）。
VLM 側を上げる例は LoRA のときだけで、全パラメータ学習で 1e-5 を超えた例は公式にも無い。

## 5. 見積もりの根拠

**実測と推定を混ぜないために、どこから来た数字かを残す。**

| 数字 | 出どころ |
|---|---|
| π0.5 6.85 秒/歩（batch 64・全パラメータ） | Blackwell での実測（[reference/19](../reference/19_weights_catalog.md) の `pi05_proto20`・`pi05_ladder2500`） |
| π0.5 10.24〜10.29 秒/歩 | RunPod での実測（8/14・8/28） |
| MolmoAct2 の VRAM（batch 8/16/32） | 上流の測定表（`lerobot/policies/molmoact2/README.md`。H100 80GB・bf16・勾配チェックポイント有効） |
| MolmoAct2 batch 64 の約 80 GiB | **外挿**（16→32 の増え方 0.64 GiB/サンプルを伸ばした値） |
| MolmoAct2 0.8〜1.6 サンプル/秒/枚 | **推定**（論文 arXiv:2605.02881 の GPU 時間から逆算。後段学習 100K 更新で 2,300 GPU 時間、LIBERO 微調整 50K 更新で 1,150 GPU 時間） |
| 学習率の選び方（5e-5・減衰・EMA が無いこと） | [reference/38](../reference/38_lr_lineage.md)（上流の設定と、LeRobot で一定に保つ条件） |

**MolmoAct2 の時間は 1 度も実測していない。**4.3 節の試し走りで置き換えること。

## 6. 詰まったときの分かれ道

| 症状 | 見るところ | 直し方 |
|---|---|---|
| `bowl5_expected.json が無い` で止まる | 2.1 節 | 小さい JSON を 1 つ上げる |
| 降りたデータが期待と違う | `$DS/meta/info.json` と `$DS/bowl5_expected.json` | `$DS` を消して降ろし直す（目録だけ先に降ろして突き合わせるので、27.8 GB を運ぶ前に分かる） |
| 1 歩も進まず落ちる（多 GPU） | `/dev/shm` の大きさ | Pod の Shared Memory を 8 GB 以上にする（9/10 に π0.5 で踏んだ） |
| out of memory | 4.2 節の表 | batch を下げる。全パラメータ学習で 48 GB の板は不可 |
| `hf upload が失敗` が続く | 2.2 節 | HF を空ける。上げるループは 60 秒ごとにやり直すので、空けば自動で上がる |
| MolmoAct2 が起動時に落ちる | `--policy.scheduler_decay_steps` と `--rename_map` | 4.1 節の 2 つを渡しているか確かめる |
