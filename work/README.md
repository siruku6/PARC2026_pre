# work/ の中身（2026-08-21 整理後）

`work/` は生成物置き場で git 管理外（このファイルだけ例外的に追跡）。消しても
`solution/tools/` のスクリプトで作り直せる想定。各ディレクトリの役割は以下。

| パス | 役割 |
|---|---|
| `train/` | 学習run（チェックポイント含む）。**触らない**：稼働中の学習コンテナが bind mount 先として使用中 |
| `quant_sweep/` | 量子化スイープの進行記録（`CANDIDATES.md`, `PROGRESS.md`, `results.csv`, `sweep.log`）。**触らない**：稼働中のtmuxセッションが追記中 |
| `artifacts/` | 提出物に組み込む、vendor化済みLeRobotと学習済み重み |
| `gpu_eval/` | GPU評価の生ログ（実行回ごとのディレクトリ） |
| `lerobot-src/` | LeRobot v0.6.0 の展開元（vendor化前のソース） |
| `patch/` | 配布キットへのパッチ関連の作業ファイル |
| `replay/` | ロールアウト再生の作業ファイル |
| `scripted/` | 台本から作った手本データ |
| `train2/` | 学習系docker bind mountの予約名（`docs/howto/01_retrain.md`・`03_gpu_eval.md`が名前を前提にしている）。現在は空 |
| `episode_lists/` | エピソード設定のjsonリスト（`text_lora32_40k_episodes.json`, `weak3x_max300_episodes.json`） |
| `logs/` | 過去の実行ログ。`eval/`（推論・提出物の実行ログ）・`build/`（zip組み立てログ）・`blend/`（重みブレンド関連ログ）に分類 |
| `sha256/` | 提出zip・重みのチェックサム |
| `admin/` | 後片付けスクリプト（`cleanup_extra.sh`, `cleanup_hf.py`, `cleanup_postqualifier.sh`）と、退避済みの印ファイル `.archive_pushed` |

`train/` と `quant_sweep/` 以外は2026-08-21時点で内容を移動・整理済み（元は`work/`直下に
ログ・sha256・後片付けスクリプトが平積みだった）。

説明の詳細は [`../docs/`](../docs/) 側にある（`docs/solution/README.md`の
「生成物の置き場」等）。このファイルは`work/`直下の構成だけを一言でまとめたもの。
