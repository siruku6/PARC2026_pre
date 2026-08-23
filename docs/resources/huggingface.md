# HuggingFace 上のモデル一覧（現在地）

**このファイルは「今 HF に何が置いてあるか」を映す場所であり、過去の版を残さず、状態が変わったら
上書きする。** 重みを上げた／消した、公開／非公開を切り替えた、といった変化があったら、その都度
このファイルを直す。

読む目的: 学習・評価・提出のどれを次に回せるかを検討するとき、「どの重みが手元に無くても
取り直せるか」を確認する。

**書くのは「どのモデルがどこにあるか」という今の状態だけである。** 消し方・空け方・運び方の
ような手順や Tips はここに書かない（[README.md](README.md) の線引き）。ディスクを空ける手順は
[../howto/09_free_disk_space.md](../howto/09_free_disk_space.md)、運ぶときに詰まったら
[../howto/08_hf_transfer_troubleshooting.md](../howto/08_hf_transfer_troubleshooting.md)。

**[reference/19_weights_catalog.md](../reference/19_weights_catalog.md) との違い**: あちらは
「どの重みでどの版を提出して何点だったか」という履歴の索引で、消した重みも載っている。
こちらは**今この瞬間 HF に存在するものだけ**を、置き場所として並べる。

最終更新: 2026-08-23 03:13 UTC 時点の HF の状態（`HfApi.list_models` / `model_info` で確認）。
3 節のローカル側は同日に `du` と `stat` で確認。

| # | 節 | 結論 |
|---|---|---|
| 1 | [自分が上げたリポジトリ](#1-自分が上げたリポジトリは-4-つ公開は-1-つだけ) | 4 リポジトリ・計 48.7 GB。公開は weak3x だけ |
| 2 | [上流から取る素の重み](#2-素の重みは-2-つとも上流から取り直せるが-π05-は-gemma-ライセンス) | π0.5 は gemma ライセンスで扱いが違う |
| 3 | [ローカルにある重みの控えの有無](#3-ローカルにある重みのうち控えがあるのは-π05-の-2-本だけ) | π0.5 の 2 本は HF にある。量子化スイープの 4 本はどこにも無い |

## 1. 自分が上げたリポジトリは 4 つ、公開は 1 つだけ

**π0.5 の重み 2 本と提出 zip は非公開、SmolVLA の weak3x だけが公開されている。
これら 4 つで計 48.7 GB を使っている。**

| リポジトリ | 公開 | 中身 | 容量 | 最終更新 (UTC) | 何のために置いてあるか |
|---|---|---|---|---|---|
| `siruku6/pi05_full_runpod` | 非公開 | `model.safetensors` 9.35 GB ＋ `train_config.json`・`INFERENCE.md`・`vlm/` | 9.37 GB | 2026-08-18 03:06 | **予選の最終提出（版 4.2.0・0.485）の重み。**RunPod 3,000 ステップ。sha256 `0c5d542e…` |
| `siruku6/pi05_stage1_24000` | 非公開 | 同構成 | 9.37 GB | 2026-08-18 03:06 | 上の重みの**親**（1 回目の学習 24,000 ステップ）。sha256 `d46bffcf…` |
| `siruku6/parc2026-submission` | 非公開 | 提出 zip 4 本 | 29.05 GB | **2026-08-23 03:13** | **GPU マシンから Mac へ zip を運ぶ通り道。**scp（1.7 MB/s）だと 9.4 GB に 1 時間超かかるのを 16 分に短縮する |
| `siruku6/smolvla-libero-plus-finetunes` | **公開**（apache-2.0） | `weak3x_20000/` 一式 | 0.91 GB | 2026-08-18 03:04 | SmolVLA 系の最高スコア（0.184）の重み。sha256 `a8cb8aee…` |

`parc2026-submission` に入っている zip 4 本:

| ファイル | 容量 |
|---|---|
| `submission_rp3000_i20.zip` | 9.38 GB |
| `submission_rp3000_blend0.zip` | 9.38 GB |
| `submission_pi05_n16.zip` | 9.38 GB |
| `submission_361.zip` | 0.92 GB |

**上の 29.05 GB は今の版だけの合計で、git の履歴に残っている分は含まない。** 2026-08-18 に
コミット 11 本を 1 本へ畳んだが、その後も更新されている（最終更新が 8/23）ため、実際の消費は
これより大きい可能性がある。

このほか `siruku6/llm-jp-3-13b-it_lora2`（公開・0.51 GB・2024-12-03）が同じアカウントにあるが、
**これは PARC2026 とは無関係の過去の別プロジェクト**（llm-jp-3-13b の LoRA アダプタ）。

**`siruku6/parc2026-archive` は既に消えている**（HF に問い合わせると 404）。ローカルの HF
キャッシュに 36.5 KB の残骸が残っているだけなので、参照しているコードがあれば直す必要がある。

## 2. 素の重みは 2 つとも上流から取り直せるが、π0.5 は gemma ライセンス

**追加学習の出発点になる素の重みは 2 つとも上流の公開リポジトリにあり、自分では上げていない。
どちらもコミットを固定して使っている。ライセンスが違う点に注意する。**

| リポジトリ | 固定しているコミット | ライセンス | 上流の最終更新 | 使い道 |
|---|---|---|---|---|
| `lerobot/pi05_libero_base` | `a217bfd3` | **gemma** | 2026-01-22 | π0.5 系の学習の出発点。`pi05_stage1_24000` はここから 24,000 ステップ |
| `lerobot/smolvla_libero_plus` | `7bb70aa5` | apache-2.0 | 2026-04-11 | SmolVLA 系の学習の出発点。素のまま提出した版（0.1.0〜0.3.0）でもある |

固定しているコミットは、いずれも問い合わせた時点の上流の先頭と一致していた（上流が動いていない）。

π0.5 の gemma ライセンスは apache-2.0 と条件が違うため、提出物に同梱する際の表記は
[knowledge/04_licenses.md](../knowledge/04_licenses.md) に書いてあるものに合わせる。

## 3. ローカルにある重みのうち、控えがあるのは π0.5 の 2 本だけ

**π0.5 の重み 2 本は HF に同じ中身があり、手元から消えても取り直せる。量子化スイープの
4 本と `weak_undertrained` はどこにも控えが無い。**

| ローカルのパス | 容量 | 同じ中身がある場所 |
|---|---|---|
| `work/artifacts/pi05_rp3000` | 8.8 GB | HF `siruku6/pi05_full_runpod`（sha256 `0c5d542e…`）。ローカルでは `work/train/pi05_full_runpod/003000` と同じ実体 |
| `work/artifacts/model_weights_pi05_024000` | 8.8 GB | HF `siruku6/pi05_stage1_24000`（sha256 `d46bffcf…`）。ローカルでは `work/train/pi05_full_local/checkpoints/024000/pretrained_model` と同じ実体 |
| `work/train/pi05_quant_c0`・`c3`・`c3_compile` の 016000 | 各 11 GB | **どこにも無い。**作り直すには 16,000 ステップの再学習（C3＋compile で 3 時間 02 分。`work/train/pi05_quant_c3_compile.log` の最終行） |
| `work/train/pi05_quant_weak_undertrained` の 000500 | 11 GB | **どこにも無い**（500 ステップなので作り直しは短い） |
| `~/.cache/huggingface/hub/models--lerobot--pi05_libero_base` | 14 GB | 上流 `lerobot/pi05_libero_base` rev `a217bfd3`。学習・評価のたびに要る |
| `~/.cache/huggingface/lerobot` | 15 GB | 上流 `lerobot/libero_plus`（学習データ本体）。取り直しは遅い |

**上の 2 本は `work/artifacts/` と `work/train/` が同じ実体を指しているので、片方だけ消しても
1 バイトも空かない。** 消してよいかの判定と、消しても空かないときの見方は
[../howto/09_free_disk_space.md](../howto/09_free_disk_space.md) にある。

## 確認したい仮定

- HF アカウントの容量上限に対して今どれだけ余裕があるかは確認していません。上限に近いと、
  新しい zip を上げる前に古いものを消す必要が出ます。
- `parc2026-submission` の git 履歴が実際に何 GB を消費しているかは、API から測れていません
  （今の版の 29.05 GB しか分かりません）。
- 消えた `siruku6/parc2026-archive` を今も参照しているコードや手順書が残っていないかは
  未確認です。
