# Colab GPU で評価を回す

**リポジトリから8ファイルを Colab へコピーし、セルを5つ貼って呼び出すだけの手順である。**
中身はすべて [solution/tools/colab/colab_eval.py](../../solution/tools/colab/colab_eval.py) にある。

| 節 | 内容 |
|---|---|
| [π0.5 にも対応した](#π05track2にも対応したただし-gpu-の選び方だけ未確認である) | `POLICY_KIND` で切り替える。T4 と CPU の可否 |
| [上げる8ファイル](#colab-へ上げるのは-8-ファイルで集めるのはスクリプトが行う) | `collect_colab_files.py` が集める |
| [貼るセル5つ](#colab-に貼るのはセル-5-つだけ) | そのままコピーできる形 |
| [値を変える](#値を変えるのは-colab_eval_configpy-だけ) | 設定10項目の表 |
| [出力の置き場](#結果は-1-本ごとに-drive-へ出る) | 落ちても直前まで残る |
| [遅いとき](#1-本目が-200-秒を超えたら-egl-を疑う) | EGL がソフトウェア実装に落ちる罠 |
| [2 プロセスと av](#依存を2プロセスに分けるのとav-が要る理由) | 依存が正面衝突する。`av` が要る理由 |
| [使い分け](#colab-は速くないgpu-マシンが空いていればそちらで回す) | 実測 110 秒／本 |
| [本番との違い](#本番との違い) | Python・CUDA の版、通信の可否 |

## π0.5（Track2）にも対応した。ただし GPU の選び方だけ未確認である

**`POLICY_KIND` で切り替える**（`colab_eval_config.py`）。`"pi05"` と `"smolvla"` の
違いは `colab_eval.py` の `POLICY_SPECS` 1 か所に集めてある。

| 違うところ | SmolVLA（Track1） | π0.5（Track2） |
|---|---|---|
| 重みの出どころ | Drive へ手で上げる（912 MB） | **HF の非公開リポジトリから落とす。**`siruku6/pi05_stage1_24000`（`vlm/` 同梱） |
| トークナイザ | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` を別に落とす | **落とさない**（重みに同梱） |
| 正規化の safetensors | `..._step_5_...` | **`..._step_2_...`** |
| lerobot の extra | `[smolvla]` | **`[pi]`**（手元の学習イメージと同じ v0.6.0 ＋ `[pi]`） |
| 測るタスク | 84 件を生成（`PER_CELL`） | **`compe/val/EVAL_TASKS.csv`（36 件）を上げる**。手元の評価と同じ集合 |
| `n_action_steps` の既定（`policy_server.py`） | 8 | 16 |

**Drive は要らなくなった。** 重みは 8.8 GB あるが、`snapshot_download` で Colab へ
直接落ちてくる（Drive のブラウザ画面から 8.8 GB を上げる必要はない）。
**非公開リポジトリなので、Colab で先に HF へログインすること。**

### 未確認: T4 で π0.5 が動くかどうか

**π0.5 の dtype は bfloat16 である**（`config.json`）。bf16 は compute capability 8.0
以上の GPU でしか本来の形で動かない。**T4 は 7.5 である。**

SmolVLA には fp16 へ書き換える処置が入っているが（`colab_eval.py` の
`POLICY_SPECS` の `bf16_patch`）、**π0.5 には同じ手が無い。** dtype を決めているのは
`policy_server.py` の `_build_pi05` で、`bfloat16` でなければ float32 になる
分岐しか無く、**float32 の π0.5 は T4 の 15 GB に載らない**（`_build_pi05` の
docstring が記録している fp32 での組み立ては 16.6 GB）。

そのため `setup_policy()` は、bf16 非対応の GPU では**止めずに警告だけ出して進む。**
動くかどうかは疎通確認（`ce.run(chunk=1, max_tasks=5)`）で分かる。落ちるなら
cc 8.0 以上（L4・A100 など）が要る。

### CPU では回せない

**1 推論に CPU 1 コアで 195 秒かかる**（2026-08-13 実測。
[07_train_pi05_local.md](07_train_pi05_local.md) の表。
重みの読み込みだけでも 302 秒）。1 エピソードは 300 ステップ ÷
`n_action_steps` 16 ＝ 約 19 回の推論なので、

    19 回 × 195 秒 ＝ 約 1.03 時間／エピソード（1 コア）
    36 件 → 約 37 時間（1 コア）／約 19 時間（2 コアで半分になったとして）

Colab の接続が保たない。**しかもこの 195 秒は下限である** —— 測ったのは
`samples=1`・ノイズ除去 10 回の構成で、手元の 36 件評価と同じ構成
（`samples=3`・20 回）は 1 推論の計算量がその約 6 倍になる。
**CPU は「行動が出るか」を見る用途にしか使えない。**

## Colab へ上げるのは 8 ファイルで、集めるのはスクリプトが行う

**手で選ぶと取り違えるので、`collect_colab_files.py` にコピーを作らせる。**

```bash
# π0.5（Track2）—— 手元の 36 件評価と同じ集合を渡す
cd /home/ubuntu/lab/PARC2026_pre
python3 solution/tools/colab/collect_colab_files.py \
    --val-tasks compe/val/EVAL_TASKS.csv
open work/colab_upload

# SmolVLA（Track1）—— 前の回と1件ずつ対にして比べるとき
python3 solution/tools/colab/collect_colab_files.py \
    --val-tasks outputs/0811_101824_base_n5_ep1_max300/VAL_TASKS.csv
```

集まるのは `colab_eval.py`・`colab_eval_config.py`・`policy_server.py`・
`make_validation_suite.py`・`check_validation_suite.py`・`patch_kit_for_variants.py`・
`rollout_recorder.py`・`summarize_rollout_videos.py` の8つ（どれも小さい）と、
用途を書いた `README.txt` である。**`work/colab_upload/` の中身をまとめて Colab の
左のファイルペインへ、`/content/` の直下へドラッグする。**

> `ce.prepare()` が数えるのは 7 件である。`colab_eval.py` 自身は import した時点で
> 読めているので、そろっているかの確認対象に入っていない（`colab_eval.py` の
> `UPLOADED_FILES`）。

`--val-tasks` に前の回の CSV を渡すと一緒に集まる。**前の回と1件ずつ対にして比べたい
ときは必ず渡す** ——1件でも違うタスクが混ざると符号検定が成り立たない。

**π0.5 では重みを運ぶ作業が無い。** `WEIGHTS_SOURCE = "hf"` にしておけば
`prepare_weights()` が `snapshot_download` で 8.8 GB を直接落とす。**非公開
リポジトリなので、先に Colab で HF へログインしておくこと**（下のセル 2）。

SmolVLA を Drive から測るときだけ（`WEIGHTS_SOURCE = "drive"`）、重み（912 MB）を
Drive のブラウザ画面へフォルダごとドラッグする。置き場は
`MyDrive/parc2026/FOR_COLAB_EVAL/<重みのフォルダ名>` にして、その名前を
`colab_eval_config.py` の `DRIVE_WEIGHTS_DIR` に書く。
**Drive への上げ下ろしは Colab の実行時間を消費しない。**

## Colab に貼るのはセル 5 つだけ

**ランタイムのタイプを GPU に変更**してから、セルを 5 つ作る。

```python
# セル 1 —— 読み込む（ファイルを上げ直したときも、これを流し直すだけでよい）
import importlib, sys
sys.path.insert(0, "/content")
import colab_eval as ce
importlib.reload(ce)
```

```python
# セル 2 —— HF へログイン（π0.5 の重みが非公開リポジトリにあるため。
#           SmolVLA を Drive から測るなら要らない）
from huggingface_hub import login
login()               # トークンを貼る。read 権限で足りる
```

```python
# セル 3 —— 準備（上から順に。合わせて 15 分ほど）
ce.prepare()          # Drive をつなぐ・7 ファイルが上がっているか・GPU と EGL
ce.setup_kit()        # clone・setup.sh・パッチ（5〜10 分）
ce.prepare_weights()  # 重みを用意し、ハッシュで名指しする
ce.setup_policy()     # 方策サーバーの依存と配置
ce.build_suite()      # 上げた CSV へ差し替え、塊に割る
```

```python
# セル 4 —— まず疎通確認（5 タスク。**36 件を回す前に必ずこれを通す**）
ce.run(chunk=1, max_tasks=5)
```

```python
# セル 5 —— 評価（π0.5 なら 1 つ 12 件。数字を変えて 3 回実行する）
ce.run(chunk=1)
```

**触るのは `chunk` の数字だけである。** 1回の呼び出しで、設定 → 評価 → 集計 →
Drive への退避まで通る。前の関数が作ったものが無ければ、どの関数を先に通すべきかを
名指しで止める。

| したいこと | 書くこと |
|---|---|
| 疎通確認（5 タスク・5 分） | `ce.run(chunk=1, max_tasks=5)`。**設定に残らないので戻し忘れが起きない** |
| どの塊が終わったか | `ce.status()` |
| 84 件を一度に（切断が無い環境） | `ce.run()` |
| 前のサーバーが居残ったとき | `ce.stop_server()` |

## 値を変えるのは colab_eval_config.py だけ

**書き換えたらそのファイルだけを `/content/` へ上げ直し、セル1から流し直す。**
`colab_eval.py` は毎回読み直すので、ランタイムの再起動は要らない。

| 設定 | 既定 | 変えるとき |
|---|---|---|
| `POLICY_KIND` | `pi05` | **どちらの模型を積むか。**`pi05` か `smolvla` |
| `WEIGHTS_SOURCE` | `hf` | `hf`（π0.5・非公開リポジトリ）か `drive`（手で上げたもの） |
| `DRIVE_WEIGHTS_DIR` | `…/model_weights_lam025` | `WEIGHTS_SOURCE = "drive"` のときだけ効く |
| `EXPECT_DIGEST` | `d46bffcf` | **意図した重みかを読み込む前に照合する。違えば止まる** |
| `N_EPISODES` | `1` | 1 タスクあたりの本数。手元の評価と揃えるため 1 のまま |
| `MAX_STEPS` | `300` | 本番の打ち切りと同じ値 |
| `CHUNKS` | `3` | 36 件なら 1 塊 12 件。落ちるなら増やす |
| `PER_CELL` | `3` | 84 件を生成するときだけ効く。**CSV を上げているときは使われない** |
| `POLICY_ENV` | 推論の設定値 5 つ | **π0.5 は手元の 36 件評価と同じ 5 つを明示する。**値は `policy_server.py` の既定と同じなので挙動は変わらないが、保存先の名前と記録に残る |
| `RECORD_DIR` / `RECORD_WHICH` | `/content/videos` / `failures` | 録画。`None` で撮らない |

**`ce.run()` は使う値を全部画面に出す。** 意図と違えばそこで止められる。

## 結果は 1 本ごとに Drive へ出る

**ランタイムが落ちても、Drive に退避したぶんは残る。**

| 出るもの | 置き場 | いつ |
|---|---|---|
| `[結果]` 行 | 画面 ＋ Drive `parc2026/eval/live/<日時>_<設定>.results.txt` | **1 本ごと** |
| 成功率・95%CI・衝突率・jerk / SPARC | 画面 | 完走時（JSON から） |
| ぶつけた物体・摂動カテゴリ別・タスク別 | 画面 | 毎回（ログから作るので落ちた回でも出る） |
| 全文ログ・結果行の控え・動画・一式 | Drive `parc2026/eval/<日時>_<設定>/` | 塊ごと |

名前は条件から作る（`0812_042900_n5_ep1_chunk1` なら「n=5・1 本ずつ・塊 1」）。
**時刻は Colab の既定の UTC なので、日本時間より 9 時間前に見える。**

やり直すのは落ちた塊だけでよい。`build_suite()` は `/content/VAL_TASKS.csv` が
上がっていればそれをそのまま使い、作り直さないので、前の回と1件ずつ対にして
比べられる（π0.5 では手元と同じ 36 件がそのまま入る）。

## 1 本目が 200 秒を超えたら EGL を疑う

**Colab では EGL がソフトウェア実装に落ちて、例外も警告も出さずに 11 倍遅くなる。**
`/usr/share/glvnd/egl_vendor.d/` に NVIDIA の登録が無いと、libglvnd は Mesa の
ソフトウェア実装を掴み、`env.step` が 38.6 ms から 431.8 ms になる。
`ce.prepare()` が `10_nvidia.json` を自分で書いて直す（`colab_eval.py` の `prepare()`）。

**`run()` の最初の `[結果]` 行の時刻を見る。** 100〜130 秒なら想定どおり。
200 秒を超えていたら止めて、`ce.prepare()` の EGL の出力を読み直す。

**この 100〜130 秒は Track1（SmolVLA・`n_action_steps=5`）の数字である。**
π0.5 では 1 本の秒数を測っていない（推論 1 回が重い代わりに回数が 3 分の 1 になる）。
π0.5 の 1 本目で判断するときは、EGL が効いているかを
`ce.prepare()` の出力（`10_nvidia.json`）だけで確かめること。

## 依存を2プロセスに分けるのと、av が要る理由

**評価側と方策側は依存が正面衝突するので、HTTP で分ける**（本番採点も同じ2プロセス構成）。
評価パイプラインは `numpy==1.26.4` / `gym==0.25.2` / `robosuite==1.4.0` / CPU torch という
古い固定版を要求し、方策は LeRobot v0.6.0 / GPU torch / transformers という新しい版を要求する。

**`av>=15,<16` を入れるのは、Colab に `datasets` が既定で入っているためである。**
`lerobot/policies/factory.py` が groot の設定を import し、その連鎖で
`groot/processor_groot.py` の `if TYPE_CHECKING or _datasets_available:` に到達する。
`datasets` が入っているとこの分岐が通り、`lerobot/datasets/__init__.py` が
`require_package("av")` を呼ぶ。**提出物の環境には `datasets` が無いので分岐が飛ばされ、
`av` は要らない。** 版を固定するのは、素で入れると 17 系になり
`lerobot/datasets/pyav_utils.py` が使う `av.option` が消えていて `AttributeError` に
なるため（`colab_eval.py` の `setup_policy()`）。

それでも `av` 関連で落ちるときは、`pip uninstall -y datasets` で分岐そのものを
通らなくする方が確実である。この評価では `datasets` を使わない。

## Colab は速くない。GPU マシンが空いていればそちらで回す

**実測（8/11）で 110 秒／本、84 件で 2.6 時間。** GPU マシン（osmesa・n=5）の
98.6 秒／本・2.3 時間とほぼ同じである。描画では Colab が 11 倍速いが、`n_action_steps=5`
では推論が支配的（60 回 × 0.85 秒）なので差が消える。

**だから「評価は Colab、学習は GPU マシン」という分け方は、速さを理由には選べない。**
Colab を使う理由は「GPU マシンが学習で埋まっているあいだも評価できる」ことだけである。

| 使い分け | |
|---|---|
| GPU マシンが空いている | **そちらで回す。** 切断が無く、`--detach` で放っておける |
| GPU マシンが学習中 | Colab で回す。塊に割る（π0.5 の 36 件なら 12 件ずつ 3 回） |

**π0.5 では、この比較がどうなるかを測っていない。** 手元の GPU マシンでは
36 件 51 分（85 秒／本）で、その 7〜8 割が CPU 描画である
（[../knowledge/14_quantization_and_eval_bottleneck.md](../knowledge/14_quantization_and_eval_bottleneck.md)）。
Colab では描画が約 11 倍速い代わりに、割り当てられる GPU は H100 より遅い。
**打ち消し合う量が読めないので、1 塊目の秒数で判断する。**

**学習セッションとは別のアカウントで開くこと。** 同一アカウントで2つ GPU を使うと
片方が止められる。

## 本番との違い

| 項目 | 内容 |
|---|---|
| 通信 | Colab は通信できる。採点環境は遮断される。**オフラインの確認は Docker 側で行う** |
| Python の版 | Colab は 3.12、採点環境は 3.10。ここでの成功は 3.10 の保証にならない |
| CUDA の版 | Colab は CUDA 12 系、採点環境は 13。`torchvision` の互換はここでは確認できない |

## あとで見る資料

| 知りたいこと | ファイル |
|---|---|
| 各関数が実際に何をしているか | [solution/tools/colab/colab_eval.py](../../solution/tools/colab/colab_eval.py) |
| 変えられる設定の全部 | [solution/tools/colab/colab_eval_config.py](../../solution/tools/colab/colab_eval_config.py) |
| 配布キットへのパッチの中身（`register.py` と `parse_obj_of_interest`） | [solution/tools/eval/patch_kit_for_variants.py](../../solution/tools/eval/patch_kit_for_variants.py) |
| GPU マシンで評価を回す（放っておける） | [03_gpu_eval.md](03_gpu_eval.md) |
| 実測値の一覧 | [../strategy/05_実測結果.md](../strategy/05_実測結果.md) |
