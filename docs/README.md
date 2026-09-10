# docs/ — 説明の置き場

**説明はすべてここにある。** `solution/` には動かすファイルだけを置く。

## 最重要な入り口

| 探しているもの | 行き先 |
|---|---|
| **いま手を動かしているタスク** | **[reference/32_current_tasks.md](reference/32_current_tasks.md)** ── 着手中のものだけ。候補の全体は [plan/22](plan/22_grasp_and_color_next_moves.md) |
| **今なにをやるべきか** | **[talk/open_items.md](talk/open_items.md)** ── アイデアを思いついたらここへ足す |
| **流した学習・重み・版・出した点** | **[reference/19_weights_catalog.md](reference/19_weights_catalog.md)（実験台帳）** |
| **次にどこを直して学習するか** | **[reference/11_training_schedule.md](reference/11_training_schedule.md)** |
| **手を動かす手順** | **[howto/](howto/)** ── 頻繁に使う手順書の一覧 |
| **今使える計算資源** | **[resources/infra_and_docker.md](resources/infra_and_docker.md)** |

## ディレクトリ

| パス | 使用頻度 | 中身 |
|---|:---:|---|
| **[talk/](talk/)** | ★★★ | **いま抱えているものと、その優先順位。まずここ。**一時ファイルの置き場なので、他のディレクトリの文書からここを指してはいけない（[README](talk/README.md)。指したくなったら情報を移す）。[open_items.md](talk/open_items.md)・[inter_sessions/](talk/inter_sessions/) |
| **[howto/](howto/)** | ★★★ | **手順書。作業前に確認。**上から順にやれば終わる形。何度も使うものだけを置く。1 回だけの回は [plan/](plan/) へ、踏んだ事故と回避策は [knowledge/](knowledge/) へ。[01_retrain.md](howto/01_retrain.md)・[03_gpu_eval.md](howto/03_gpu_eval.md)・[05_scripted_demos.md](howto/05_scripted_demos.md) |
| **[reference/](reference/)** | ★★★ | **調べて分かった事実。判断の前に確認。**モデル・評価データ・採点のタスクの母集団・設定の意味。実験台帳 [19_weights_catalog.md](reference/19_weights_catalog.md)・学習条件 [11_training_schedule.md](reference/11_training_schedule.md)・失敗の内訳 [18_failure_breakdown.md](reference/18_failure_breakdown.md) |
| [knowledge/](knowledge/) | ★★ | **踏んだ穴・外した読み・成果物の材料。**1 項目 1 ファイル。レポート必須欄 [01_report_requirements.md](knowledge/01_report_requirements.md)・ライセンス [04_licenses.md](knowledge/04_licenses.md)・採点ログ全記録 [05_score_history.md](knowledge/05_score_history.md) |
| [plan/](plan/) | ★★ | **1 つずつの手の設計。**何を測り、出た数字で何を決めるか。本選 1 の手の一覧 [22_grasp_and_color_next_moves.md](plan/22_grasp_and_color_next_moves.md)・GPU と提出枠 [14_finals1_gpu_and_submission_plan.md](plan/14_finals1_gpu_and_submission_plan.md) |
| **[resources/](resources/)** | ★ | **今使える** 計算資源・ディスク・メモリ・docker image・HF 上のモデルの一覧。状態が変わるたびに上書き（履歴は残さない）。手順や Tips は書かない（[README](resources/README.md) の線引き）。[infra_and_docker.md](resources/infra_and_docker.md)・[huggingface.md](resources/huggingface.md) |
| [strategy/](strategy/) | ★ | 何を狙うか、なぜそうするか。判断の材料。実測結果 [05_実測結果.md](strategy/05_実測結果.md) |
| [solution/](solution/) | — | 提出物と Colab の説明。コード本体は `../solution/` |
| [materials/](materials/) | — | **外部から取得した資料の現物。**ルールの PDF・運営 Q&A の CSV・運営の告知・講義資料。一次資料はここを見る（[knowledge/11_primary_sources.md](knowledge/11_primary_sources.md)）。**ここのファイルは書き換えない** |
| [outputs/submission_logs/](../outputs/submission_logs/) | — | 本番採点の生ログ。トラックごとに `<日付>_<版>.txt`（本選は `b_PARC2026_final/outputs/submission_logs/`） |

## 説明とコードの分け方

```
docs/solution/     説明（README・colab_eval・DOCKER_TRAIN）
solution/          動かすファイル（policy_server.py・requirements.txt・tools/・train/）
```

`solution/` に置いたものだけが git の追跡対象で、重み・同梱ソース・venv・zip は
`work/` へ出て管理外になる。
