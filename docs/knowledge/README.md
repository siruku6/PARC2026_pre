# knowledge —— 成果物を組み立てるための材料置き場

**1 項目 1 ファイル。**成果物ごとに書き下ろすのではなく、**材料を 1 度だけ書いて、
複数の成果物から引く。**

## 何を作るためのものか

| # | 成果物 | 締切 | 状態 |
|---|---|---|---|
| **1** | **予選レポート（PDF 2 ページ以内）** | **8/17 23:59** | **作成中。これが最優先** |
| 2 | 知見の Claude skill 化 | — | 未着手 |
| 3 | 人間側で意識する・準備することの整理 | — | 未着手 |
| 4 | 社内向けの参加報告 | — | 未着手 |

## 書き方の約束

**① 値には出典と、確かめ直すコマンドを付ける。**
このリポジトリの `.md` を出典にしない（[../../.claude/CLAUDE.md](../../.claude/CLAUDE.md) の §2.1）。
使ってよいのは PDF・運営 Q&A の CSV・採点ログ・`train_config.json`・重みそのもの・
`work/gpu_eval/` の測定結果である。

**② 2 と 3 は同じ教訓を別の角度から見たものなので、1 ファイルにまとめる。**
各ファイルの末尾に **「Claude 側」** と **「人間側」** を並べて書く。
書き分けるのは成果物を作るときにする。

**③ 日本語で書く。** ここは人が読んで成果物を組み立てる材料である。
**skill（成果物 2）は英語**で、そのとき書き起こす（機械向けは英語、という規約）。

## 項目の一覧

| ファイル | 中身 | 主に使う成果物 |
|---|---|---|
| [01_report_requirements.md](01_report_requirements.md) | **レポートの必須欄**（予選 PDF p14・p23 から） | 1 |
| [02_model_identity.md](02_model_identity.md) | モデル名・ベース・revision・ハッシュ・推論方式・chunk 長 | 1, 4 |
| [03_training_info.md](03_training_info.md) | 学習データ・歩数・学習対象・ハイパーパラメータ | 1, 4 |
| [04_licenses.md](04_licenses.md) | 権利関係（ベースは gemma・データは未記載・同梱コードは Apache-2.0） | 1 |
| [05_score_history.md](05_score_history.md) | **採点ログ 16 本から抜いた全記録**（版・設定・歩数・秒・スコア） | 1, 4 |
| [06_step_count_drives_score.md](06_step_count_drives_score.md) | **歩数が最も強い説明変数（r = −0.87）。悪くする要素と良くする要素** | 1, 2, 3, 4 |
| [07_local_eval_cannot_rank.md](07_local_eval_cannot_rank.md) | 手元評価は本番の順位を付けられない | 1, 2, 3, 4 |
| [08_chunk_length_follows_model_strength.md](08_chunk_length_follows_model_strength.md) | **chunk 長の最適値はモデルが強くなるほど大きくなる**（n=2 → 16） | 1, 2, 3, 4 |
| [09_two_stage_training.md](09_two_stage_training.md) | **2 段学習。行動側で頭打ちにしてから凍結を解く** | 1, 4 |
| [10_deciding_under_deadline.md](10_deciding_under_deadline.md) | **閾値がある勝負での危険の取り方。運ぶと出すを切り離す** | 2, 3, 4 |
| [11_primary_sources.md](11_primary_sources.md) | **1 次資料の当たり方。踏んだ 4 件の間違い** | 2, 3 |
| [12_workbench_accidents.md](12_workbench_accidents.md) | ディスク満杯・並行実行・大容量の運び方 | 2, 3 |

**成果物 1（レポート）は 8/15 に提出済み。**`../report/qualifier_report.md`。
**次は成果物 2（Claude skill 化）。**06〜12 の「Claude 側 / 人間側」の欄が材料になる。

