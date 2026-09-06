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
| **0** | [どの機械がどの資格情報で入っているか](#0-どの機械がどの資格情報で-hf-に入っているか831-実測) | **JAIST は OAuth で 9/12 に切れる。Blackwell はリポジトリを作れない** |
| 1 | [自分が上げたリポジトリ](#1-自分が上げたリポジトリは-4-つ公開は-1-つだけ) | 5 リポジトリ。公開は weak3x だけ |
| 2 | [上流から取る素の重み](#2-素の重みは-2-つとも上流から取り直せるが-π05-は-gemma-ライセンス) | π0.5 は gemma ライセンスで扱いが違う |
| 3 | [ローカルにある重みの控えの有無](#3-ローカルにある重みのうち控えがあるのは-π05-の-2-本だけ) | π0.5 の 2 本は HF にある。量子化スイープの 4 本はどこにも無い |

## 0. どの機械が、どの資格情報で HF に入っているか（8/31 実測）

**同じアカウントでも機械ごとに種類が違う。**種類の見分け方と、詰まったときの対処は
[../howto/08_hf_transfer_troubleshooting.md](../howto/08_hf_transfer_troubleshooting.md) にある。

| 機械 | 資格情報 | できること |
|---|---|---|
| **JAIST サーバ** | **OAuth のセッション。`expiresAt` は `2026-09-12T06:42:07Z`** | 読み・書き・**リポジトリの作成**。**9/12 を過ぎると黙って 403 になる**（本選 1 の締切 9/17 より前である。切れたら `hf auth login` をやり直す） |
| **Blackwell** | **fineGrained のアクセストークン**（`~/.hf_token`。`solution/train/omnicampus_setup.sh:253` が新しいシェルごとに `HF_TOKEN` を export するので、`hf auth login` をしても環境変数が勝つ） | 読み・書き。**リポジトリは作れない**ので、新しい置き場はブラウザか他の機械で作る |

## 1. 自分が上げたリポジトリは 10 個、うち非公開は 4 つ

**枠を食うのは非公開だけである**（公開のぶんは 1 バイトも数えない。9/3 に確定）。
**2026-09-05 に数え直した非公開の合計は 46.87 GB（無料枠は非公開 100 GB・空き 53.13 GB）。**
**数え直す 1 行は [../howto/08_hf_transfer_troubleshooting.md](../howto/08_hf_transfer_troubleshooting.md) の最後の節にある。**

> **容量は「いま見えているファイル」だけでは決まらない。HF は古い LFS の版も数える。**
> `siruku6/parc2026-submission` を zip の通り道として何度も上書きした結果、
> 見えている 4 本（29 GB）よりずっと大きくなり、**2026-08-28 に上限に達した。**
> **リポジトリごと消して 28 GB まで下がった。**
> 実数は `https://huggingface.co/settings/storage` で見る（「List LFS files」で内訳も出る）。
> 履歴だけを潰す `super_squash_history` もあるが、**枠に反映されるまで最大 36 時間かかる。**
> 出典: [Storage limits](https://huggingface.co/docs/hub/storage-limits)（2026-08-28 確認）。

| リポジトリ | 公開 | 中身 | 容量 | 最終更新 (UTC) | 何のために置いてあるか |
|---|---|---|---|---|---|
| `siruku6/pi05_trial1500` | **非公開** | 1,500 歩（カメラ拡張つき）の `pretrained_model` 3 つ・`ladder2500` の 2 つ・学習ログ | **37.49 GB** | 2026-09-02 19:37 | **本選 1 の提出の材料。****`001500` が採点側で Track1 0.492・Track2 0.233 を出した重みそのもの**（版 5.0.1）。**`vlm/` 同梱** |
| `siruku6/parc2026-submission` | **非公開** | **提出 zip だけ。使い捨て** | **9.38 GB** | 2026-08-31 06:06 | **運ぶだけの通り道。**提出し終えたら**リポジトリごと消す**（下の「分ける決まり」）。**消すだけでは容量が戻らないので `super_squash_history` で履歴も潰す。**`sha256` は [reference/19](../reference/19_weights_catalog.md) と [howto/02](../howto/02_build_submission.md) の版の表に残っている |
| `siruku6/pi05_code` | **非公開** | 学習コード一式（`train_pi05.py` など） | 0.00 GB | 2026-09-04 00:19 | **Blackwell のセッションごとに取り直す出どころ**（[howto/14](../howto/14_blackwell_session.md)） |
| `siruku6/pi05_all111` | **非公開** | **9/5 の走り（配布の 111 課題）の上げ先** | 0.00 GB | 2026-09-05 02:12 | `F48`。**`000500`〜`002000` の 4 個（37.4 GB）がこれから入る** |
| `siruku6/pi05_camdrop2500` | **公開**（gemma） | 場面カメラを 0.5 で潰した 2,500 歩の 5 個（`000500`〜`002500`）と学習ログ・`README.md`・`NOTICE` | 46.77 GB | 2026-09-05 02:14 | **12.0.0 の重み（版 12.0.0）。****9/5 に公開へ移して 46.77 GB を枠から外した**（[howto/08](../howto/08_hf_transfer_troubleshooting.md) の最後の節） |
| `siruku6/pi05_full_runpod` | **公開** | `model.safetensors` 9.35 GB ＋ `train_config.json`・`INFERENCE.md`・`vlm/` | 9.37 GB | 2026-08-31 21:52 | **予選の最終提出（版 4.2.0・0.485）の重み**＝`rp3000`。RunPod 3,000 ステップ。sha256 `0c5d542e…` |
| `siruku6/pi05_stage1_24000` | **公開** | 同構成 | 9.37 GB | 2026-08-31 21:52 | 上の重みの**親**（1 回目の学習 24,000 ステップ）。sha256 `d46bffcf…` |
| `siruku6/smolvla-libero-plus-finetunes` | **公開**（apache-2.0） | `weak3x_20000/` 一式 | 0.91 GB | 2026-08-18 03:04 | SmolVLA 系の最高スコア（0.184）の重み。sha256 `a8cb8aee…` |
| `siruku6/lerobot_distributed` | 公開 | 空 | 0.00 GB | 2026-08-31 06:17 | 使っていない |
| `siruku6/llm-jp-3-13b-it_lora2` | 公開 | LoRA アダプタ | 0.51 GB | 2024-12-03 14:09 | **PARC2026 とは無関係の別プロジェクト** |

### 分ける決まり —— **残すものと運ぶものを、同じリポジトリに入れない**

**リポジトリを丸ごと消すと容量は即座に戻るが、リポジトリの中のファイルだけ消しても
履歴の LFS が残る**（`super_squash_history` で潰せるが、**枠への反映に最大 36 時間**）。
だから「片方だけ捨てる」を成立させるには、**捨て方の違うものを別のリポジトリに置く。**

| 置き場 | 中身 | 寿命 |
|---|---|---|
| 重みのリポジトリ（`pi05_*`） | `pretrained_model` と学習ログ | **残す。**評価にも使う |
| `parc2026-submission` | **提出 zip だけ** | **使い捨て。**提出し終えたらリポジトリごと消す |

**枠そのものは減らない**（同じアカウントの 100 GB を分け合うだけである）。
効くのは捨て方で、**丸ごと消せる形にしておくと履歴ごと即座に戻る。**
`hf upload ... --private` は、消したリポジトリを履歴が空の状態で作り直す。

**もう 1 つ、捨てずに戻す手がある**——**非公開のまま置いておく必要が無くなったものは、公開へ移す。**
**9/5 に `pi05_camdrop2500`（46.77 GB）でこれをやり、非公開の合計が 93.64 → 46.87 GB に戻った**
（[../howto/08_hf_transfer_troubleshooting.md](../howto/08_hf_transfer_troubleshooting.md) の最後の節。
**公開する前に、ログにトークンが混じっていないかを見ること**）。

**`siruku6/parc2026-archive` は既に消えている**（HF に問い合わせると 404）。ローカルの HF
キャッシュに 36.5 KB の残骸が残っているだけなので、参照しているコードがあれば直す必要がある。

**本選の配布ベースラインモデル（π0.5 を LoRA で 5,000 ステップ）は HF ではなく、配布 GPU 環境の
`~/dataset/pi05_step005000_submission_py310.zip` に置かれている。**中身と参考スコアは
[infra_and_docker.md](infra_and_docker.md) の配布 GPU 環境の節にある。

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
4 本と `weak_undertrained`、幾何の接ぎ木の 2 本はどこにも控えが無い。**

| ローカルのパス | 容量 | 同じ中身がある場所 |
|---|---|---|
| `work/artifacts/pi05_rp3000` | 8.8 GB | HF `siruku6/pi05_full_runpod`（sha256 `0c5d542e…`）。ローカルでは `work/train/pi05_full_runpod/003000` と同じ実体 |
| `work/artifacts/model_weights_pi05_024000` | 8.8 GB | HF `siruku6/pi05_stage1_24000`（sha256 `d46bffcf…`）。ローカルでは `work/train/pi05_full_local/checkpoints/024000/pretrained_model` と同じ実体 |
| `work/artifacts/pi05_geo12k` | 12 GB | **どこにも無い。**幾何の接ぎ木 12,000 歩。ローカルでは `work/train/geo_vggt_v2/checkpoints/012000/pretrained_model` と同じ実体（inode `23704624`） |
| `work/artifacts/pi05_geo6k` | 12 GB | **どこにも無い。**同じ学習の 6,000 歩。`work/train/geo_vggt_v2/checkpoints/006000/pretrained_model` と同じ実体（inode `810838694`） |
| `work/train/pi05_quant_c0`・`c3`・`c3_compile` の 016000 | 各 11 GB | **どこにも無い。**作り直すには 16,000 ステップの再学習（C3＋compile で 3 時間 02 分。`work/train/pi05_quant_c3_compile.log` の最終行） |
| `work/train/pi05_quant_weak_undertrained` の 000500 | 11 GB | **どこにも無い**（500 ステップなので作り直しは短い） |
| `~/.cache/huggingface/hub/models--lerobot--pi05_libero_base` | 14 GB | 上流 `lerobot/pi05_libero_base` rev `a217bfd3`。学習・評価のたびに要る |
| `~/.cache/huggingface/lerobot` | 15 GB | 上流 `lerobot/libero_plus`（学習データ本体）。取り直しは遅い |

**`work/artifacts/` の 4 本は `work/train/` と同じ実体（ハードリンク）なので、片方だけ消しても
1 バイトも空かない。**

**`du` を 1 回の呼び出しで 2 つ並べると、後ろの数字が嘘になる。**`du` は同じ inode を
1 回しか数えないので、**先に来た側に容量が全部乗る。**

```
$ du -sh work/train work/artifacts     # ← 後ろの artifacts が 81 MB に見える
111G  work/train
81M   work/artifacts
$ du -sh work/artifacts; du -sh work/train   # ← 別々に呼べば正しい
40G   work/artifacts
111G  work/train
```

**2026-08-31 にこれを「重みが消えた」と読み違えた。**空き容量（`df` の 11 GB）は
どちらの読み方でも変わらないので、**迷ったら `df` を見る。** 消してよいかの判定と、消しても空かないときの見方は
[../howto/09_free_disk_space.md](../howto/09_free_disk_space.md) にある。

## 確認したい仮定

- 消えた `siruku6/parc2026-archive` を今も参照しているコードや手順書が残っていないかは
  未確認です。
