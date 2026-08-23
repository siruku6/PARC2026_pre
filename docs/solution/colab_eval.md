# Colab GPU で評価を回す

**リポジトリから8ファイルを Colab へコピーし、セル3つを貼って呼び出すだけの手順である。**
中身はすべて [solution/tools/colab/colab_eval.py](../../solution/tools/colab/colab_eval.py) にある。

| 節 | 内容 |
|---|---|
| [Track1 用である](#この手順は-track1-用でtrack2-では未検証である) | π0.5 では動かない。何を直せば動くか |
| [上げる8ファイル](#colab-へ上げるのは-8-ファイルで集めるのはスクリプトが行う) | `collect_colab_files.py` が集める |
| [貼るセル3つ](#colab-に貼るのはセル-3-つだけ) | そのままコピーできる形 |
| [値を変える](#値を変えるのは-colab_eval_configpy-だけ) | 設定7項目の表 |
| [出力の置き場](#結果は-1-本ごとに-drive-へ出る) | 落ちても直前まで残る |
| [遅いとき](#1-本目が-200-秒を超えたら-egl-を疑う) | EGL がソフトウェア実装に落ちる罠 |
| [2 プロセスと av](#依存を2プロセスに分けるのとav-が要る理由) | 依存が正面衝突する。`av` が要る理由 |
| [使い分け](#colab-は速くないgpu-マシンが空いていればそちらで回す) | 実測 110 秒／本 |
| [本番との違い](#本番との違い) | Python・CUDA の版、通信の可否 |

## この手順は Track1 用で、Track2 では未検証である

**`colab_eval.py` と `colab_eval_config.py` は SmolVLA・912 MB の重みを前提にしている。**
Track2 で使っている π0.5 は読み込むだけで約 18 GB 必要なので、この流れには乗らない。

| 前提にしている値 | どこ | Track2 で何が変わるか |
|---|---|---|
| ベース重み `lerobot/smolvla_libero_plus` | `colab_eval.py:97` | π0.5 の重みに差し替える必要がある |
| `EXPECT_DIGEST = "952171f9"`（λ=0.25 の重み。予選4回目の提出物） | `colab_eval_config.py` | 測る重みのハッシュ先頭8桁に書き換える |
| 検証用スイート 84 件（4 スイート × 7 カテゴリ × 3） | `colab_eval_config.py` の `PER_CELL` | Track2 の摂動スイートとは中身が違う |
| 重みを Drive 経由で運ぶ（912 MB） | `prepare_weights()` | 18 GB の転送時間を測っていない |

**Track1 の重みを測り直すならそのまま使える。** π0.5 で使うなら上の4点を直すこと。

## Colab へ上げるのは 8 ファイルで、集めるのはスクリプトが行う

**手で選ぶと取り違えるので、`collect_colab_files.py` にコピーを作らせる。**

```bash
cd /home/ubuntu/lab/PARC2026_pre
python3 solution/tools/colab/collect_colab_files.py \
    --val-tasks outputs/0811_101824_base_n5_ep1_max300/VAL_TASKS.csv
open work/colab_upload
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

重み（912 MB）は別で、Drive のブラウザ画面へフォルダごとドラッグする。置き場は
`MyDrive/parc2026/FOR_COLAB_EVAL/<重みのフォルダ名>` にして、その名前を
`colab_eval_config.py` の `DRIVE_WEIGHTS_DIR` に書く。
**Drive への上げ下ろしは Colab の実行時間を消費しない。**

## Colab に貼るのはセル 3 つだけ

**ランタイムのタイプを GPU に変更**してから、セルを3つ作る。

```python
# セル 1 —— 読み込む（ファイルを上げ直したときも、これを流し直すだけでよい）
import importlib, sys
sys.path.insert(0, "/content")
import colab_eval as ce
importlib.reload(ce)
```

```python
# セル 2 —— 準備（上から順に。合わせて 15 分ほど）
ce.prepare()          # Drive をつなぐ・7 ファイルが上がっているか・GPU と EGL
ce.setup_kit()        # clone・setup.sh・パッチ（5〜10 分）
ce.prepare_weights()  # 重みを用意し、ハッシュで名指しする
ce.setup_policy()     # 方策サーバーの依存と配置
ce.build_suite()      # 検証用スイートへ差し替え、84 件を 3 つに割る
```

```python
# セル 3 —— 評価（1 つ 28 件・約 51 分。数字を変えて 3 回実行する）
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
| `USE_FINETUNED` / `DRIVE_WEIGHTS_DIR` | `True` / `…/model_weights_lam025` | 評価する重みを替える |
| `EXPECT_DIGEST` | `952171f9` | **意図した重みかを読み込む前に照合する。違えば止まる** |
| `N_EPISODES` | `1` | 1 タスクあたりの本数。84 件では 1 のまま |
| `MAX_STEPS` | `300` | 本番の打ち切りと同じ値 |
| `CHUNKS` | `3` | 51 分でも落ちるなら 6（1 塊 14 件・26 分） |
| `PER_CELL` | `3` | 検証用スイートの大きさ。4 スイート × 7 カテゴリ × この数 |
| `POLICY_ENV` | 空 | 推論の設定値。**重みを比べるときは空** |
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
上がっていれば同じ 84 件を作り直さないので、前の回と1件ずつ対にして比べられる。

## 1 本目が 200 秒を超えたら EGL を疑う

**Colab では EGL がソフトウェア実装に落ちて、例外も警告も出さずに 11 倍遅くなる。**
`/usr/share/glvnd/egl_vendor.d/` に NVIDIA の登録が無いと、libglvnd は Mesa の
ソフトウェア実装を掴み、`env.step` が 38.6 ms から 431.8 ms になる。
`ce.prepare()` が `10_nvidia.json` を自分で書いて直す（`colab_eval.py` の 325〜340 行）。

**`run()` の最初の `[結果]` 行の時刻を見る。** 100〜130 秒なら想定どおり。
200 秒を超えていたら止めて、`ce.prepare()` の EGL の出力を読み直す。

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
なるため（`colab_eval.py` の 535〜542 行）。

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
| GPU マシンが学習中 | Colab で回す。ただし 28 件ずつ 3 回に割る |

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
| GPU マシンで評価を回す（放っておける） | [../howto/03_gpu_eval.md](../howto/03_gpu_eval.md) |
| 実測値の一覧 | [../strategy/05_実測結果.md](../strategy/05_実測結果.md) |
