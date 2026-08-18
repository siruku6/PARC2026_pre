# 19. 重みの一覧 —— 名前・提出した版・中身・置き場所

**「`pi05/full_local_024000` ってどれだっけ」を 1 か所で引けるようにした表である。**
重みの名前・その重みで出した提出物の版番号・実際にどういうモデルなのかを 1 行に
並べてある。

出典は 3 つ。**どれも一次資料である。**

| 何の出典か | どこ |
|---|---|
| 版番号と本番スコアと zip の sha256 | `../howto/02_build_submission.md` の版の表（もとは `outputs/` の採点ログ） |
| 学習の条件（固定した所・ステップ数・batch・学習率） | 各重みの隣の `train_config.json` |
| 重みの sha256 と置き場所 | HuggingFace `siruku6/parc2026-archive` の `index.json` |

## 名前についての注意（2 つとも読み違えやすい）

**`pi05_full_local` は全パラメータ学習ではない。**「full」は π0.5 の実行の名前として
付いただけで、この実行は**行動側（action expert）だけを動かし、カメラ画像を扱う部分
と VLM は固定していた**（`train_expert_only=true`・`freeze_vision_encoder=true`）。
全部を動かしたのは RunPod の 3,000 ステップの方だけである。

**`pi05_stage1`（HuggingFace のリポジトリ名）と `pi05/full_local_024000`（退避先での
名前）は同じ重みである。**sha256 `d46bffcf…` で一致を確認済み。

## 一覧

| 退避先での名前 | 出した版 | 実際にどういうモデルか | **本番スコア** | 重みの sha256 |
|---|---|---|---|---|
| `smolvla/base` | **0.1.0・0.2.0・0.3.0** | **追加学習していない素の重み。**`lerobot/smolvla_libero_plus` rev `7bb70aa5`。表示される 20,000 ステップは上流がこれを作るのに回した分 | 0.05832・0.096・**0.1409** | `a67a056a…` |
| `smolvla/lora20k` | **1.0.0** | Colab で LoRA を 20,000 ステップ。**設定の一次資料が残っていないので作り直せない。**退避先には上げていない | **0.016**（素の 8.8 分の 1 に落ちた） | `2cd9faab…` |
| （重みは残っていない） | **2.0.0** | 素 75% ＋ `lora20k` が学習で動いた分 25%（λ=0.25）。`solution/tools/blend_weights.py` で作り直せる | 0.09507 | — |
| `smolvla/weak3x_20000` | **3.0.0〜3.7.0** | SmolVLA に LoRA を 20,000 ステップ。**苦手なタスクの手本を 3 倍に増やし、300 ステップを超える手本を落とした**（`weak3x` はこの 2 つを指す） | 0.144 → **0.184**（SmolVLA 系の最高。版 3.6.0／3.6.1） | `a8cb8aee…` |
| `smolvla/text40k_40000` | **なし（見送り）** | LoRA を当てる先を言語を扱う側まで広げて 40,000 ステップ。**前処理 4 ファイルが欠けており、weak3x の 2 条件も外している。**退避先には上げていない | — | `ed9b6372…` |
| `smolvla/scripted_mix_20000` | **なし** | 台本で自分で作った手本（`solution/tools/scripted_demo.py`）を混ぜて 20,000 ステップ。**測ったのは moka の 8 件だけで 0.000。**退避先には上げていない | — | `0793c54e…` |
| `pi05/full_local_old_000500` | **なし** | π0.5 に切り替えた最初の 500 ステップ。**成功率を測った記録が無い。**この実行の `optimizer_state` は 8/14 の容量整理で消したので、ここから学習を再開する道は無い | — | `deaaed6b…` |
| **`pi05/full_local_024000`**（＝ `pi05_stage1`・**1 回目の学習**） | **3.8.0〜3.13.0** | π0.5 を `lerobot/pi05_libero_base` rev `a217bfd3` から **行動側だけ** 24,000 ステップ。更新したのは **693M** パラメータ。batch 16・学習率 2.5e-5 → 2.5e-6・ローカル H100-20C で 9 時間 31 分 | 0.177（3.8.0）・**0.264（3.9.0）**・0.169（3.10.0） | `d46bffcf…` |
| **`pi05/full_runpod_003000`**（＝ `pi05_full_runpod`・**2 回目の学習**） | **4.0.0〜4.4.0** | **1 回目の学習の 24,000 ステップから出発**し、固定を全部外して 3,000 ステップ。更新したのは **4.14B** パラメータ（全部）。batch 64・学習率 5e-6 → 2.5e-6・RunPod A100 80GB で 8 時間 33 分 | **0.485（4.2.0）＝予選の最終提出・42 位** | `0c5d542e…` |

**上の 2 本は親子である。**`full_runpod_003000` の `train_config.json` の
`pretrained_path` が `/workspace/stage1` を指している。狙いと loss の推移は
`../knowledge/09_two_stage_training.md` にある。

## 版番号だけから引きたいとき

**設定だけを変えた版は、重みが同じでも別の版番号を持っている。**どの版がどの重みか。

| 版 | 重み | 版の間で変えたもの |
|---|---|---|
| 0.1.0 → 0.2.0 → 0.3.0 | `smolvla/base` | 実行するチャンクの長さ n=50 → 10 → 5 ＋ EMA 0.3 |
| 1.0.0 | `smolvla/lora20k` | — |
| 2.0.0 | （λ=0.25 で混ぜたもの） | 混ぜる割合 |
| 3.0.0 → 3.1.0 → 3.3.0 → 3.4.0 → 3.5.0 → 3.6.0 → 3.6.1 → 3.7.0 | `smolvla/weak3x_20000` | n・samples・指示文の正規化・印字（3.6.1 は挙動が 3.6.0 と同じ） |
| 3.8.0 → 3.9.0 → 3.10.0 → 3.11.0 → 3.12.0 → 3.13.0 | `pi05/full_local_024000` | n=10 → 16・chunk 継ぎ目の平滑化・RTC・ノイズ除去の回数 |
| 4.0.0 → 4.2.0 → 4.3.0 → 4.4.0 | `pi05/full_runpod_003000` | 平滑化を外す・ノイズ除去の回数・samples |

**未提出の版もこの表に入っている。**提出したかどうかは
`../howto/02_build_submission.md` の版の表の「本番スコア」の欄で見る。

## 何がどこにあるか

**この表は片づけが終わったあとの姿である。**2026-08-18 の時点で終わっているのは
`parc2026-archive` への SmolVLA 2 本だけで、**π0.5 の 2 本は退避先にまだ上げられて
いない**（非公開リポジトリの容量上限。`../plan/05_postqualifier_cleanup.md` の 3.4）。
**ローカルの削除もまだ実施していない**（`work/cleanup_postqualifier.sh` を流すと下の
姿になる）。

| 重み | HuggingFace | このマシン |
|---|---|---|
| `pi05/full_runpod_003000` | `pi05_full_runpod`（＋ `parc2026-archive` は 3.4 待ち） | `work/artifacts/pi05_rp3000`（`work/train/pi05_full_runpod/003000` と同じ実体） |
| `pi05/full_local_024000` | `pi05_stage1`（同上） | `work/artifacts/model_weights_pi05_024000`（同上）。`checkpoints/024000/training_state/` も残してある |
| `smolvla/weak3x_20000` | `parc2026-archive` | **消す** |
| `smolvla/base` | `parc2026-archive` | **消す**（上流から取り直せる） |
| `smolvla/text40k_40000`・`scripted_mix_20000`・`lora20k` | **無い** | **消す。消したら作り直せない** |
| `pi05/full_local_old_000500` | **無い** | **消す** |

| 提出物の zip | HuggingFace `parc2026-submission` | このマシン |
|---|---|---|
| 4.2.0（最終提出・`5cf6b983…`） | `submission_rp3000_blend0.zip` | 手元には無い（8/14 に消した） |
| 3.6.1（保険・`e5fdfd3b…`） | `submission_361.zip` | 消す |
| 4.3.0（`7a2bc2b3…`）・4.4.0（`02819ca3…`） | `submission_4.3.0.zip`・`submission_4.4.0.zip` | 消す |
| それ以外（3.12.0・3.13.0・pi05・pi05_n16） | **消す** | 手元には無い |

片づけの段取りと、消した理由は `../plan/05_postqualifier_cleanup.md` にある。

## 引き直すコマンド

```bash
# 退避先の索引（重みの sha256・学習条件・成績が 1 本の JSON にある）
.venv/bin/python -c "
import json
from huggingface_hub import hf_hub_download
for e in json.load(open(hf_hub_download('siruku6/parc2026-archive', 'index.json'))):
    print(f\"{e['folder']:32s} {e['sha256_model'][:12]} steps={e['steps']} score={e['score']}\")"

# ある重みが行動側だけの学習か、全パラメータか
.venv/bin/python -c "
import json
d = json.load(open('work/artifacts/pi05_rp3000/train_config.json'))['policy']
print('行動側だけ' if d['train_expert_only'] else '全パラメータ')"
```
