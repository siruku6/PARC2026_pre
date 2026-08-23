# 19. 重みの一覧 —— 名前・提出した版・中身・置き場所

**「`pi05/full_local_024000` ってどれだっけ」を 1 か所で引けるようにした表である。**
重みの名前・その重みで出した提出物の版番号・実際にどういうモデルなのかを 1 行に
並べてある。

出典は 3 つ。**どれも一次資料である。**

| 何の出典か | どこ |
|---|---|
| 版番号と本番スコアと zip の sha256 | `../howto/02_build_submission.md` の版の表（もとは `outputs/` の採点ログ） |
| 学習の条件（固定した所・ステップ数・batch・学習率） | 各重みの隣の `train_config.json` |
| 推論の設定（n・infer・平滑化・samples） | 各重みの隣の `INFERENCE.md` |
| 重みの sha256 | 各重みの隣の `PROVENANCE.md`。HuggingFace 側は LFS の `sha256` |

**この文書が索引そのものである。**8/18 まで `siruku6/parc2026-archive` の
`index.json` が索引だったが、中身が空になったのでリポジトリごと消した。

## 名前についての注意（2 つとも読み違えやすい）

**`pi05_full_local` は全パラメータ学習ではない。**「full」は π0.5 の実行の名前として
付いただけで、この実行は**行動側（action expert）だけを動かし、カメラ画像を扱う部分
と VLM は固定していた**（`train_expert_only=true`・`freeze_vision_encoder=true`）。
全部を動かしたのは RunPod の 3,000 ステップの方だけである。

**`pi05_stage1_24000`（HuggingFace のリポジトリ名）と `pi05/full_local_024000`
（この文書での呼び名）は同じ重みである。**sha256 `d46bffcf…` で一致を確認済み。

## 一覧

| この文書での呼び名 | 出した版 | 実際にどういうモデルか | **本番スコア** | 重みの sha256 |
|---|---|---|---|---|
| `smolvla/base` | **0.1.0・0.2.0・0.3.0** | **追加学習していない素の重み。**`lerobot/smolvla_libero_plus` rev `7bb70aa5`。表示される 20,000 ステップは上流がこれを作るのに回した分。**上流から取り直せるので、どこにも上げていない** | 0.05832・0.096・**0.1409** | `a67a056a…` |
| `smolvla/lora20k` | **1.0.0** | Colab で LoRA を 20,000 ステップ。**設定の一次資料が残っていないので作り直せない。**どこにも上げていない | **0.016**（素の 8.8 分の 1 に落ちた） | `2cd9faab…` |
| （重みは残っていない） | **2.0.0** | 素 75% ＋ `lora20k` が学習で動いた分 25%（λ=0.25）。`solution/tools/weights/blend_weights.py` で作り直せる | 0.09507 | — |
| **`weak3x_20000`**（＝ HF `smolvla-libero-plus-finetunes`。**ここだけ公開**） | **3.0.0〜3.7.0** | SmolVLA に LoRA を 20,000 ステップ。**苦手なタスクの手本を 3 倍に増やし、300 ステップを超える手本を落とした**（`weak3x` はこの 2 つを指す） | 0.144 → **0.184**（SmolVLA 系の最高。版 3.6.0／3.6.1） | `a8cb8aee…` |
| `smolvla/text40k_40000` | **なし（見送り）** | LoRA を当てる先を言語を扱う側まで広げて 40,000 ステップ。**weak3x の 2 条件を外しており、`train_config.json` も残っていない。**どこにも上げていない | — | `ed9b6372…` |
| `smolvla/scripted_mix_20000` | **なし** | 台本で自分で作った手本（`solution/tools/scripted_demo/scripted_demo.py`）を混ぜて 20,000 ステップ。**測ったのは moka の 8 件だけで 0.000。**`train_config.json` も残っていない。どこにも上げていない | — | `0793c54e…` |
| `pi05/full_local_old_000500` | **なし** | π0.5 に切り替えた最初の 500 ステップ。**成功率を測った記録が無い。**この実行の `optimizer_state` は 8/14 の容量整理で消したので、ここから学習を再開する道は無い | — | `deaaed6b…` |
| **`pi05/full_local_024000`**（＝ HF `pi05_stage1_24000`・**1 回目の学習**） | **3.8.0〜3.13.0** | π0.5 を `lerobot/pi05_libero_base` rev `a217bfd3` から **行動側だけ** 24,000 ステップ。更新したのは **693M** パラメータ。batch 16・学習率 2.5e-5 → 2.5e-6・ローカル H100-20C で 9 時間 31 分 | 0.177（3.8.0）・**0.264（3.9.0）**・0.169（3.10.0） | `d46bffcf…` |
| **`pi05/full_runpod_003000`**（＝ HF `pi05_full_runpod`・**2 回目の学習**） | **4.0.0〜4.4.0** | **1 回目の学習の 24,000 ステップから出発**し、固定を全部外して 3,000 ステップ。更新したのは **4.14B** パラメータ（全部）。batch 64・学習率 5e-6 → 2.5e-6・RunPod A100 80GB で 8 時間 33 分 | **0.485（4.2.0）＝予選の最終提出・42 位** | `0c5d542e…` |

**上の 2 本は親子である。**`full_runpod_003000` の `train_config.json` の
`pretrained_path` が `/workspace/stage1` を指している。狙いと loss の推移は
`../knowledge/09_two_stage_training.md` にある。

## 版番号だけから引きたいとき

**設定だけを変えた版は、重みが同じでも別の版番号を持っている。**1 版 1 行で並べる。
「変えたもの」は**すぐ上の版から何を動かしたか**である。

| 版 | 重み | すぐ上の版から変えたもの | **本番スコア** |
|---|---|---|---|
| **0.1.0** | `smolvla/base` | （最初の版）後処理なし・n=50 | 0.05832 |
| **0.2.0** | `smolvla/base` | n=50 → 10 | 0.096 |
| **0.3.0** | `smolvla/base` | n=10 → 5 ＋ EMA 0.3 | **0.1409**（素の最高） |
| **1.0.0** | `smolvla/lora20k` | **重みだけ**（設定は 0.3.0 と同じ） | 0.016 |
| **2.0.0** | λ=0.25 で混ぜたもの | **重みだけ** | 0.09507 |
| **3.0.0** | `smolvla/weak3x_20000` | **重みだけ**（n=5） | 0.144 |
| **3.1.0** | `smolvla/weak3x_20000` | n=5 → 2 | 0.166 |
| **3.2.0** | `smolvla/weak3x_20000` | n=5 のまま ＋ samples 1 → 3 | 未提出 |
| **3.3.0** | `smolvla/weak3x_20000` | n=2 → 1 | 0.080（**本物の悪化**） |
| **3.4.0** | `smolvla/weak3x_20000` | n=2 ＋ samples 1 → 3 | 0.168（3.1.0 と同じ・時間 2.7 倍） |
| **3.5.0** | `smolvla/weak3x_20000` | **指示文の正規化を入れる**（n=2・samples 1） | 0.181 |
| **3.6.0** | `smolvla/weak3x_20000` | n=2 → 5 | **0.184（SmolVLA 系の最高）** |
| **3.6.1** | `smolvla/weak3x_20000` | `reset` の印字を落とす（**挙動は 3.6.0 と同じ**） | 0.184 相当。**SmolVLA 期の最後の提出** |
| **3.7.0** | `smolvla/weak3x_20000` | n=5 → 8 | 未提出（**落ちたとき用に用意**） |
| **3.8.0** | `pi05/full_local_024000` | **重みを π0.5 に替える**（n=10・infer 10） | 0.177 |
| **3.9.0** | `pi05/full_local_024000` | n=10 → 16 | **0.264** |
| **3.10.0** | `pi05/full_local_024000` | chunk の継ぎ目の平滑化 0 → 4 | 0.169（**手元 36 件では最高だったのに本番は負けた**） |
| **3.11.0** | `pi05/full_local_024000` | 平滑化を外し RTC（horizon 4）を入れる | 未提出（**平滑化より強く悪化**） |
| **3.12.0** | `pi05/full_local_024000` | ノイズ除去の回数 infer 10 → 20 | 未提出（単独では未測定） |
| **3.13.0** | `pi05/full_local_024000` | infer 20 ＋ samples 1 → 3 | 未提出（**手元 36 件でゴール到達 34/36**） |
| **4.0.0** | `pi05/full_runpod_003000` | **重みを RunPod 3,000 歩に替える**（平滑化 4 のまま） | 未提出（**平滑化が本番で負けたので破棄**） |
| **4.1.0** | `pi05/full_runpod_003000` | samples を π0.5 でも使えるようにした（既定 1 ＝ 4.0.0 と同じ挙動） | **zip 自体を作っていない** |
| **4.2.0** | `pi05/full_runpod_003000` | 平滑化 4 → 0（**3.9.0 と同一の設定＝重みだけの 1 変数**） | **0.485。予選の最終提出・42 位** |
| **4.3.0** | `pi05/full_runpod_003000` | infer 10 → 20 | 未提出 |
| **4.4.0** | `pi05/full_runpod_003000` | infer 20 ＋ samples 1 → 3 | 未提出 |

**未提出の版もこの表に入っている。**zip の sha256 と採点ログの日時は
`../howto/02_build_submission.md` の版の表にある。

## 版ごとの設定（差分ではなく、そのままの値）

**上の表は「すぐ上の版から何を変えたか」なので、ある版の設定を知るには遡る必要が
ある。この表は遡らなくて済むように、全部の値をそのまま並べたものである。**zip を
消しても、ここと重みがあれば組み直せる。

**出典の「採点ログ」は一次資料である。**`policy_server.py` は起動時に設定を 1 行
印字するので（[policy_server.py:777](../../solution/policy_server.py#L777)）、
**提出した版は本番の採点ログにそのままの値が残っている。**`outputs/` は git で
追跡している。

| 版 | 重み | n | infer | blend | EMA | samples | RTC | 正規化 | 出典 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1.0 | base | 50 | — | 0 | なし | 1 | — | なし | 文書 |
| 0.2.0 | base | 10 | — | 0 | なし | 1 | — | なし | 文書 |
| **0.3.0** | base | **5** | — | 0 | **0.3** | 1 | — | なし | **採点ログ** |
| 1.0.0 | lora20k | 5 | — | 0 | 0.3 | 1 | — | なし | 文書 |
| **2.0.0** | λ=0.25 | **5** | — | 0 | **0.3** | 1 | — | なし | **採点ログ** |
| **3.0.0** | weak3x | **5** | — | 0 | **0.3** | 1 | — | なし | **採点ログ** |
| **3.1.0** | weak3x | **2** | — | 0 | **0.3** | 1 | — | なし | **採点ログ** |
| 3.2.0 | weak3x | 5 | — | 0 | 0.3 | 3 | — | なし | 文書（未提出） |
| **3.3.0** | weak3x | **1** | — | 0 | **0.3** | 1 | — | なし | **採点ログ** |
| **3.4.0** | weak3x | **2** | — | 0 | **0.3** | **3** | — | なし | **採点ログ** |
| **3.5.0** | weak3x | **2** | — | 0 | **0.3** | 1 | — | **入り** | **採点ログ** |
| **3.6.0** | weak3x | **5** | — | 0 | **0.3** | 1 | — | **入り** | **採点ログ** |
| 3.6.1 | weak3x | 5 | — | 0 | 0.3 | 1 | — | 入り | 3.6.0 と同一の挙動 |
| **3.7.0** | weak3x | **8** | — | 0 | **0.3** | 1 | — | **入り** | **採点ログ**（未提出だが手元で起動した） |
| **3.8.0** | **24000** | **10** | **10** | **0** | **0.3** | **1** | off | **入り** | **採点ログ** |
| **3.9.0** | **24000** | **16** | **10** | **0** | **0.3** | **1** | off | **入り** | **採点ログ** |
| **3.10.0** | 24000 | **16** | **10** | **4** | **0.3** | **1** | off | **入り** | **採点ログ** |
| 3.11.0 | 24000 | 16 | 10 | 0 | 0.3 | 1 | **on(horizon 4)** | 入り | 文書（未提出） |
| 3.12.0 | 24000 | 16 | **20** | 0 | 0.3 | 1 | off | 入り | 文書（未提出） |
| 3.13.0 | 24000 | 16 | **20** | 0 | 0.3 | **3** | off | 入り | 文書（未提出） |
| 4.0.0 | rp3000 | 16 | 10 | **4** | 0.3 | 1 | off | 入り | 文書（未提出） |
| 4.1.0 | rp3000 | 16 | 10 | 4 | 0.3 | 1 | off | 入り | 4.0.0 と同一の挙動・zip なし |
| **4.2.0** | **rp3000** | **16** | **10** | **0** | **0.3** | **1** | **off** | **入り** | **採点ログ。予選の最終提出** |
| 4.3.0 | rp3000 | 16 | **20** | 0 | 0.3 | 1 | off | 入り | 文書（未提出） |
| 4.4.0 | rp3000 | 16 | **20** | 0 | 0.3 | **3** | off | 入り | **`policy_server.py` の現在の既定値そのまま**（`__version__ = "4.4.0"`） |

- **n** ＝ `PARC_N_ACTION_STEPS`。1 回の推論で作った行動列のうち実際に実行する本数
- **infer** ＝ `PARC_NUM_INFERENCE_STEPS`。flow matching のノイズ除去の回数（π0.5 だけ）
- **blend** ＝ `PARC_BLEND_STEPS`。行動列の継ぎ目を直前の行動へ寄せる本数。**0 ＝ 寄せない**
- **EMA** ＝ `PARC_EMA_ALPHA`。出した行動を前の行動と混ぜる係数
- **samples** ＝ `PARC_SAMPLES`。1 手ごとに何本サンプルして medoid を選ぶか
- **RTC** ＝ `PARC_RTC`。実時間で継ぎ目を埋める仕組み。**入れると悪化した**
- **正規化** ＝ `PARC_NORMALIZE_INSTRUCTION`。タスク文の表記を寄せる処理

### zip を組み直す手順

**環境変数では足りない。**採点環境はこちらの環境変数を渡してくれないので、
**`policy_server.py` の既定値そのものを書き換えてから組む。**

```bash
# 1. policy_server.py の __version__ と、上の表の値に合う既定値を書き換える
#    （今のコードは 4.4.0 そのままなので、4.4.0 なら 1 文字も触らない）
# 2. 組む。--weights で重みの置き場を指す（zip の中では model_weights/ になる）
work/venv310/bin/python solution/tools/build/build_submission.py \
    --weights work/artifacts/pi05_rp3000 \
    --output work/submission_4.3.0.zip
work/venv310/bin/python validate_submission.py --static
```

**書き換える既定値の行**（[policy_server.py](../../solution/policy_server.py)）:

| 設定 | 行 |
|---|---|
| `__version__` | 176 |
| `PARC_N_ACTION_STEPS` | 670（π0.5 は 16・SmolVLA は 8） |
| `PARC_NUM_INFERENCE_STEPS` | 684 |
| `PARC_BLEND_STEPS` | 723 |
| `PARC_EMA_ALPHA` | 724 |
| `PARC_SAMPLES` | 735 |
| `PARC_RTC` | 745（`PARC_RTC_HORIZON` は 747） |
| `PARC_NORMALIZE_INSTRUCTION` | 763 |

**手元の評価だけなら書き換えは要らない。**7 つとも環境変数で上書きできるので、
`PARC_NUM_INFERENCE_STEPS=20 bash solution/tools/run_docker_eval.sh …` で足りる。
**採点環境はこちらの環境変数を渡してくれないので、提出するときだけ既定値を直す。**

**組み直した zip の sha256 は元と一致しない**（時刻が入る）。提出済みの版の
バイト列を後から示す必要があるときは、`../howto/02_build_submission.md` の
版の表に記録した sha256 が唯一の手がかりになる。

## 何がどこにあるか

**HuggingFace 側は 2026-08-18 に済んでいる。**「このマシン」の欄だけがまだ手つかずで、
`work/admin/cleanup_postqualifier.sh` を流すと下の姿になる。

**公開しているのは `weak3x_20000` だけである。**最終提出（4.2.0）とその 1 段目
（24,000 歩）は非公開のままにした。公開の方はコンペとは無関係な
「`lerobot/libero_plus` で SmolVLA を追加学習した重み」として置いてある。

| 重み | HuggingFace | このマシン |
|---|---|---|
| `pi05/full_runpod_003000` | **非公開** `pi05_full_runpod`（`vlm/` と `INFERENCE.md` 込み） | `work/artifacts/pi05_rp3000`（`work/train/pi05_full_runpod/003000` と同じ実体） |
| `pi05/full_local_024000` | **非公開** `pi05_stage1_24000`（同上） | `work/artifacts/model_weights_pi05_024000`（同上）。`checkpoints/024000/training_state/` も残してある |
| `weak3x_20000` | **公開** [`smolvla-libero-plus-finetunes`](https://huggingface.co/siruku6/smolvla-libero-plus-finetunes) | **消す** |
| `smolvla/base` | 無い。上流 `lerobot/smolvla_libero_plus` rev `7bb70aa5` | **消す**（上流から取り直せる） |
| `smolvla/text40k_40000`・`scripted_mix_20000`・`lora20k` | **無い** | **消す。消したら作り直せない** |
| `pi05/full_local_old_000500` | **無い** | **消す**（成功率を測った記録が無く、使い道がない） |

**zip は 3 本だけ残した。**残す基準は「バイト列そのものに意味があるか」である。
組み直せる版は、重みと上の設定表があれば作れるので置いていない。

| 提出物の zip | HuggingFace **非公開** `parc2026-submission` | このマシン |
|---|---|---|
| **4.2.0**（最終提出・0.485・`5cf6b983…`） | `submission_rp3000_blend0.zip` | 手元には無い（8/14 に消した） |
| **3.9.0**（0.264・`5a9eead5…`） | `submission_pi05_n16.zip` | 手元には無い |
| **3.6.1**（SmolVLA 期の最後・`e5fdfd3b…`） | `submission_361.zip` | 消す |
| 3.8.0・3.12.0・3.13.0・4.3.0・4.4.0 | **8/18 に削除** | 消す |

**消した 5 本は、重みと `solution/` から組み直せる。**重みは 3.x が
`pi05/full_local_024000`、4.x が `pi05/full_runpod_003000` で、どちらもこのマシンと
HuggingFace の両方にある。違うのは `policy_server.py` の既定値だけで、その値は
上の「版ごとの設定」に全部書いてある。

**ただし 3.8.0 は提出済みである。**組み直した zip の sha256 は元と一致しないので、
**「本番に出したのはこのバイト列だ」と後から示す道は無くなった。**残るのは
`../howto/02_build_submission.md` の版の表に記録した `ea951d5d…` という値だけである。
提出済みの 3 本のうち 3.9.0 と 4.2.0 は現物を残したので、そちらは示せる。

片づけの段取りと、消した理由は `../plan/05_postqualifier_cleanup.md` にある。

## 引き直すコマンド

```bash
# HuggingFace にある重みと、その sha256 と公開・非公開
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
for m in api.list_models(author='siruku6'):
    i = api.model_info(m.id, files_metadata=True)
    for s in i.siblings:
        if s.rfilename.endswith('model.safetensors'):
            print(f\"{'非公開' if i.private else '公開  '} {s.lfs.sha256[:12]} {m.id}/{s.rfilename}\")"

# 最終提出の推論設定（n・infer・平滑化・samples）
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
print(open(hf_hub_download('siruku6/pi05_full_runpod', 'INFERENCE.md')).read())"

# ある重みが行動側だけの学習か、全パラメータか
.venv/bin/python -c "
import json
d = json.load(open('work/artifacts/pi05_rp3000/train_config.json'))['policy']
print('行動側だけ' if d['train_expert_only'] else '全パラメータ')"
```
