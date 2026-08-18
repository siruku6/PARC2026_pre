# 05. 予選のあと片づけ（本戦 3 回に向けた棚卸しと段取り）

**予選は 42 位・0.48464 で終わった**（`docs/knowledge/05_score_history.md`）。
このあと本戦が 3 回ある。**この文書は「何を残し、何を消し、どの順で手を動かすか」を
1 枚にまとめたもの**である。数字はすべて 2026-08-18 にこのマシンと HuggingFace で
実測した値。

## 0. いま何がどこにあるか（実測）

| 場所 | 使用量 | 空き |
|---|---|---|
| `/`（`/dev/mapper/ubuntu--vg-ubuntu--lv`） | 258 GB / 298 GB（87%） | **41 GB** |
| うち `/home/ubuntu/lab` | 136 GB | — |
| うち Docker（イメージ＋ビルド用の中間結果） | 約 114 GB | — |
| うち `~/.cache/huggingface` | 33 GB | — |
| うち `/home/ubuntu/src`（tampura。PARC とは別件） | 9.0 GB | — |
| HuggingFace の非公開リポジトリ 5 本 | 87.3 GB（**上限 100 GB に対して**） | 約 13 GB |

**41 GB しか空いていない。**提出物の zip は 1 本 9.4 GB なので、本戦で候補を数本
組むだけで足りなくなる。8/14 には空き容量が尽きて `policy_server.py` が 0 バイトに
なり、走っていた評価が黙って死んだ（`docs/knowledge/12_workbench_accidents.md`）。

## 1. まず気づいたこと 3 つ（消す前に手当てが必要）

### 1.1 HuggingFace の `siruku6/parc2026-archive` が消えている

**404 が返る。**ローカルのキャッシュには 8/14 15:24 に取ってきた `index.json` が
残っていて、そこには重み 7 本が索引されていた。

| 索引にあった重み | sha256 の頭 | ほかに控えがあるか |
|---|---|---|
| `pi05/full_local_024000` | `d46bffcf` | **ある** — `siruku6/pi05_stage1` と同一と確認 |
| `pi05/full_local_old_000500` | `deaaed6b` | **この 1 台だけ** |
| `smolvla/weak3x_20000` | `a8cb8aee` | **この 1 台だけ** |
| `smolvla/text40k_40000` | `ed9b6372` | **この 1 台だけ** |
| `smolvla/scripted_mix_20000` | `0793c54e` | **この 1 台だけ** |
| `smolvla/base` | `a67a056a` | **この 1 台だけ**（上流の `lerobot/smolvla_libero_plus` から取り直せる） |
| `smolvla/lora20k` | `2cd9faab` | **この 1 台だけ。**しかも設定の一次資料が無く、作り直せない |

**消えた理由は分かっていない。**意図的に消したのかどうかを確かめるまで、
上の「この 1 台だけ」の 5 本には触らない。

### 1.2 直近のコミットと 4 ファイルが GitHub に上がっていない

**【8/18 に解消】**当初「62 コミット先」と書いたが、これは `github/main` との差である。
リモートには `submission/v1-base-weights` ブランチが既にあり、**未 push は 7 コミット
だけだった。**8/18 に push 済み。

まだコミットしていないものが 4 ファイル残っている。

| ファイル | 中身 |
|---|---|
| `docs/knowledge/05_score_history.md`（+60 行） | 公開された順位表の節（42 位・0.48464） |
| `docs/knowledge/10_deciding_under_deadline.md`（+8 行） | 100 位を 0.4 と見積もったが実際は 0.42832 だった、という検証 |
| `docs/knowledge/README.md`（+26 −5 行） | 成果物 4 を完了に更新し、知見とスライドの対応表を追加 |
| `docs/report/internal_briefing.md`（600 行・未追跡） | 社内向けの参加報告。S1〜S24 ＋ 付録 |

**どの重みがどの成績に対応するかを書いてあるのは、この文書群だけである。**重みを
消す前にここを上げておかないと、残した重みの意味が分からなくなる。

### 1.3 採点ログ 16 本が git の管理外にある

`outputs/` は `.gitignore` に入っている（1.5 MB・16 ファイル）。ところが
`docs/knowledge/05_score_history.md` は「出典は `outputs/` の採点ログそのもの」と
書いていて、**本番スコアの一次資料はここにしか無い。**HuggingFace にも無い。

`/home/ubuntu/lab/tmp/リーダーボード.txt`（2.6 KB）も同じ立場にある。ただしこれは
他の参加者のアカウント名を含むので、リポジトリには持ち込まない方針だった。

## 2. 残すもの

**判断の軸は「本戦で参照するか」と「作り直せるか」の 2 つだけ。**

| 残すもの | 大きさ | 置き場所 | 理由 |
|---|---|---|---|
| **最終提出の重み**（RunPod 全パラメータ 3,000 歩） | 8.7 GB | HF `pi05_full_runpod` ＋ ローカル `work/artifacts/pi05_rp3000` | 0.485 を出した本体。sha256 `0c5d542e` が両方で一致済み |
| **local 24,000 歩の重み** | 8.7 GB | HF `pi05_stage1` ＋ ローカル | 0.264。上の重みと混ぜる候補 |
| 最終提出の zip 4.2.0 | 8.73 GB | HF `parc2026-submission` | 提出物そのもの。sha256 `5cf6b983` |
| 保険の zip 3.6.1 | 0.86 GB | 同上 | SmolVLA 系の最後の提出物 |
| 未提出の 4.3.0・4.4.0 | 17.5 GB | 同上 | infer 20・samples 3 を本戦で試すときの比較材料 |
| 採点ログ 16 本 | 1.5 MB | **git に入れる**（後述） | 本番スコアの一次資料 |
| 評価の記録（`work/gpu_eval/`、動画 265 本を含む） | 730 MB | ローカル | 前回どう失敗していたかを見返す |
| 配布キットの Docker イメージ `parc2026`  | 27.6 GB | ローカル | 採点環境と同じ構成。作り直すと時間がかかる |
| 学習コンテナ `parc2026-train-pi05` と `-server` | 9.66 GB（共有） | ローカル | 本戦の学習と評価にそのまま使う |
| `pi05_libero_base` の重み | 14 GB | `~/.cache/huggingface` | 本戦の学習の出発点 |
| `lerobot/libero_plus` のデータ | 15 GB | 同上 | 学習データ。取り直すと遅い |
| Xet のキャッシュ | 80 MB | 同上 | 消すと HF からの取り直しが全部やり直しになる |

## 3. 消すもの（回収できる量）

### 3.1 情報が 1 バイトも失われないもの（合計 約 85 GB）

| 消すもの | 回収 | なぜ安全か |
|---|---|---|
| Docker のビルド用の中間結果 | **56.8 GB** | 使っているイメージは 0 個。作り直せる |
| イメージ `sirukufarios/parc2026-train-pi05:v2` | **9.58 GB** | Docker Hub に上げてある |
| ローカルの zip 3 本（`rp3000_i20`・`rp3000_i20_s3`・`361`） | **18.3 GB** | 3 本とも HF 側の sha256 と一致を確認済み |
| イメージ `ubuntu:latest` | 0.10 GB | ビルド時に取り直す |

**イメージは共有レイヤの分を差し引くこと。**`docker system df` の
「RECLAIMABLE 38.2 GB」は当てにならない。`-v` を付けて `UNIQUE SIZE` を見ると、
`parc2026-train` を消しても 60 kB しか空かない（`parc2026-train-pi05` と
9.49 GB を共有している）。**4 本まとめて消したときだけ 9.49 GB が空く。**

### 3.2 判断が要るもの（合計 約 99 GB）

| 消すもの | 回収 | 判断 |
|---|---|---|
| **π0.5 の中間チェックポイント** | **86.6 GB** | 下の表のとおり |
| SmolVLA 系の重み（ローカル分） | 8.1 GB | **HF に上げ直してからローカルを消す**（3.3） |
| `work/venv310_check` | 1.4 GB | 作り直せる |
| SmolVLA 用のキャッシュ（`SmolVLM2-500M` 1.9 GB ＋ `smolvla_libero_plus` 0.87 GB） | 2.8 GB | SmolVLA に戻らないなら不要 |

π0.5 の中間チェックポイントの内訳:

| 消すもの | 本数 | 回収 |
|---|---|---|
| `pi05_full_local/checkpoints/{022000,022500,023000,023500}` を丸ごと（重み＋`training_state`） | 4 | 43.0 GB |
| `pi05_full_runpod/{001000,001500,002000}`（001000 は同じ重みを 2 か所に持っている） | 4 | 34.9 GB |
| `pi05_full_local_old/checkpoints/000500` ＋ `artifacts/model_weights_pi05_step500` | 1 | 8.7 GB |

**`024000/training_state/`（2.0 GB）だけは残す。**学習を 24,000 歩から続ける道を
残すためである。

**同じ中身を 2 つの名前で指しているだけの箇所（ハードリンク）が 3 組ある。**
片方だけ消しても 1 バイトも空かない。**両方消して初めて空く。**

| 組 | inode |
|---|---|
| `artifacts/model_weights_pi05_024000/model.safetensors` ＝ `train/pi05_full_local/checkpoints/024000/pretrained_model/model.safetensors` | 25474754 |
| `artifacts/model_weights_pi05_step500/model.safetensors` ＝ `train/pi05_full_local_old/checkpoints/000500/pretrained_model/model.safetensors` | 540905770 |
| `artifacts/pi05_rp3000/model.safetensors` ＝ `train/pi05_full_runpod/003000/model.safetensors` | 805401061 |

いっぽう `pi05_full_runpod/001000/` は `model.safetensors` と
`pretrained_model/model.safetensors` を**別の実体として 2 つ持っている**（inode
811427643 と 540902022）。ここは 8.7 GB がまるごと無駄になっている。

### 3.3 HuggingFace 側（回収 35 GB）

| 操作 | 対象 | 回収 |
|---|---|---|
| 削除 | `parc2026-submission` の `submission_3.12.0.zip`・`3.13.0`・`pi05`・`pi05_n16` | 34.9 GB |
| リポジトリごと削除 | `parc2026-tmp`（`speedtest.bin` 0.195 GB だけ。転送速度を測った残り） | 0.20 GB |
| **作り直して上げる** | `parc2026-archive`（`solution/tools/archive_push.py` で） | **−12 GB**（増える） |

`parc2026-archive` に上げ直すもの:

| 上げるもの | 大きさ | 索引に載せる成績 |
|---|---|---|
| `smolvla/weak3x_20000` | 0.85 GB | 36 件で 0.722・本番 0.184 |
| `smolvla/base`（比べるときの基準） | 0.85 GB | 84 件で 0.476 |
| `pi05/full_runpod_003000` | 8.7 GB | **本番 0.485。最終提出** |
| `pi05/full_local_024000` | 8.7 GB | 本番 0.264・36 件で 0.694 |

**`scripted_mix` と `lora20k` は上げない。**前者は 8 件だけで 0.000 という比べられ
ない数字しか無く、後者は設定の一次資料が無くて作り直せない。

**π0.5 の 2 本は `vlm/`（17 MB のトークナイザ）も一緒に上げる。**これが無いと、
重みを取ってきたあとに提出物を組む段でネットワークが要る。

**Xet が中身の重なりを見て差分だけ送るので、8.7 GB の 2 本はほぼ瞬時に終わる**
見込みである（`pi05_full_runpod` と `pi05_stage1` に同じ中身が既にある）。

### 3.4 【8/18 に判明】非公開リポジトリの容量上限に先に当たる

**SmolVLA の 2 本は上がったが、π0.5 の 2 本はどちらも次のエラーで落ちた。**

```
huggingface_hub.errors.BadRequestError:
  Private repository storage limit reached, please upgrade your plan to
  increase your private storage limit
```

アカウントは無料プラン（`whoami` の `isPro` が `False`）で、**非公開の上限は
100 GB。**実測した内訳（現在の版に載っているファイルだけの合計）:

| 非公開リポジトリ | 大きさ |
|---|---|
| `parc2026-submission`（zip 8 本） | 66.56 GB |
| `pi05_full_runpod` | 9.35 GB |
| `pi05_stage1` | 9.35 GB |
| `parc2026-archive`（SmolVLA 2 本を上げたところ） | 1.82 GB |
| `parc2026-tmp` | 0.21 GB |
| **合計** | **87.30 GB** |

**ここに履歴の分が乗る。**`parc2026-submission` は 8/14 に
`submission_pi05_blend4.zip`（9.38 GB）を消しているが、**消したのは現在の版から
だけで、実体は履歴に残っている。**足すと約 96.7 GB で、π0.5 の 9.35 GB を
入れる余地が無い。数字が合う。

**消すだけでは空かない。**HF の容量は履歴も数えるので、zip 4 本を消したうえで
**履歴を 1 コミットに畳む**（`HfApi.super_squash_history`）必要がある。畳めば、
現在の版から外れた 5 本ぶん（4 本 37.5 GB ＋ blend4 9.4 GB）が落ちる。

台本は `work/cleanup_hf.py` にある。**既定では何もせず、何をするかだけ表示する。**

```bash
.venv/bin/python work/cleanup_hf.py              # 表示だけ
CONFIRM=yes .venv/bin/python work/cleanup_hf.py  # 実行
```

**畳むと `parc2026-submission` のコミット 11 本が 1 本になる。**残るのは
「いつどの zip を上げたか」の日時だが、それは
`../howto/02_build_submission.md` の版の表と `outputs/` の採点ログに書いてある。

## 4. 手を動かす順番

**上から順に。**保全が終わる前に削除に入らない。

**8/18 に順番が 1 か所変わった。**HF の容量上限に当たったので、**もとの 10 番
（HF 側の削除）を 3 番の途中に入れないと 3 番が終わらない**（3.4）。

| # | やること | 誰が | 目安 | 状態 |
|---|---|---|---|---|
| 1 | 書きかけ 4 ファイルをコミットし、`submission/v1-base-weights` を GitHub へ push | 承認をもらってから | 数分 | 未承認 |
| 2 | `.gitignore` から `outputs/` を外し、採点ログ 16 本（1.5 MB）をコミットして push | 同上 | 数分 | **済**（`581f375`） |
| 3a | `archive_push.py` で `parc2026-archive` を作り直し、SmolVLA の 2 本を上げる | `.venv/bin/python` で実行できる | 数分 | **済** |
| 3b | **HF の zip 4 本を消し、履歴を畳み、`parc2026-tmp` を消す**（`work/cleanup_hf.py`） | 承認をもらってから | 数分・HF 側 47 GB | **待ち** |
| 3c | π0.5 の 2 本を `parc2026-archive` に上げる（`vlm/` 込み） | 3b のあと | 10〜30 分 | 待ち |
| 4 | 上げた 4 本の sha256 を HF 側の値と突き合わせる | 自動で確認できる | 数分 | 2 本 **済**（`a8cb8aee`・`a67a056a` 一致） |
| 5 | Docker のビルド用の中間結果を捨てる（`docker builder prune -f`） | ユーザー | 数分・**56.8 GB** | 待ち |
| 6 | イメージ 2 本を消す（`sirukufarios/…:v2`・`ubuntu:latest`） | ユーザー | **9.68 GB** | 待ち |
| 7 | ローカルの zip 3 本を消す | ユーザー | **18.3 GB** | 待ち |
| 8 | π0.5 の中間チェックポイントを消す（3.2 の表のとおり） | ユーザー | **86.6 GB** | 待ち |
| 9 | SmolVLA 系のローカル重みと `venv310_check` を消す | ユーザー | **9.5 GB** | 待ち |
| 11 | 「どれがどこにあるか」の表を 1 本足して push | — | 数分 | **済**（`../reference/19_weights_catalog.md`） |

**`rm` はこちらから実行できない**（`docs/knowledge/12_workbench_accidents.md`）。
5〜10 は台本を用意してユーザーが流す。

見込みの結果:

| | 前 | 後 |
|---|---|---|
| `/` の空き | 41 GB | **約 220 GB** |
| `/` の使用率 | 87% | **約 26%** |
| HF の非公開リポジトリ | 87.3 GB（履歴込み 約 96.7 GB） | **約 59 GB**（上限 100 GB に対して） |

## 5. まだ決まっていないこと

| # | 問い | 決めるのは |
|---|---|---|
| 1 | `parc2026-archive` が消えたのは意図的か。違うなら、同じことが起きない置き方にする | ユーザー |
| 2 | `/home/ubuntu/src/tampura`（8.8 GB）・`vGPU`（3.1 GB）・`nvidia-linux-grid-535…deb`（374 MB）は PARC とは別件。消してよいか | ユーザー |
| 3 | `tampura-tampura` の Docker イメージ（0.94 GB）も同じ | ユーザー |
| 4 | `リーダーボード.txt` を git に入れるか。他の参加者名を含むので、いまは `lab/tmp/` に置いたまま | ユーザー |
| 5 | `parc2026-submission` の履歴を畳んでよいか（3.4）。畳まないと容量が空かず、π0.5 の 2 本を退避先に上げられない | ユーザー |
