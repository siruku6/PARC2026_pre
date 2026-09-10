# knowledge —— 成果物を組み立てるための材料置き場

**1 項目 1 ファイル。**成果物ごとに書き下ろすのではなく、**材料を 1 度だけ書いて、
複数の成果物から引く。**

## 何を作るためのものか

| # | 成果物 | 締切 | 状態 |
|---|---|---|---|
| **1** | **予選レポート（PDF 2 ページ以内）** | 8/17 23:59 | **提出済み**（8/15）。`../report/qualifier_report.md` |
| **2** | **知見の Claude skill 化** | — | **完了。**skill 2 本（下記） |
| 3 | 人間側で意識する・準備することの整理 | — | 未着手（4 を先に作ることにした） |
| **4** | **社内向けの参加報告** | — | **完了。**`../report/internal_briefing.md`（発表 20〜30 分ぶんの骨子 ＋ 読み物） |

## 書き方の約束

**① 値には出典と、確かめ直すコマンドを付ける。**
このリポジトリの `.md` を出典にしない（`~/.claude/CLAUDE.md` の §2「Development Environment」）。
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
| [03_training_info.md](03_training_info.md) | 学習データ・ステップ数・学習対象・ハイパーパラメータ | 1, 4 |
| [04_licenses.md](04_licenses.md) | 権利関係（ベースは Gemma Terms of Use・データは MIT・同梱コードは Apache-2.0） | 1 |
| [05_score_history.md](05_score_history.md) | **採点ログ 16 本から抜いた全記録**（版・設定・ステップ数・秒・スコア） | 1, 4 |
| [06_step_count_drives_score.md](06_step_count_drives_score.md) | **ステップ数が本番スコアを最もよく説明した（r = −0.87）。悪くする要素と良くする要素** | 1, 2, 3, 4 |
| [07_local_eval_cannot_rank.md](07_local_eval_cannot_rank.md) | 手元評価は本番の順位を付けられない | 1, 2, 3, 4 |
| [08_chunk_length_follows_model_strength.md](08_chunk_length_follows_model_strength.md) | **chunk 長の最適値はモデルが強くなるほど大きくなる**（n=2 → 16） | 1, 2, 3, 4 |
| [09_two_stage_training.md](09_two_stage_training.md) | **学習を 2 回に分ける。行動側で頭打ちにしてから、視覚エンコーダの固定を外す** | 1, 4 |
| [10_deciding_under_deadline.md](10_deciding_under_deadline.md) | **本選に出られる最低点を越えないと意味がない勝負での危険の取り方。運ぶと出すを切り離す** | 2, 3, 4 |
| [11_primary_sources.md](11_primary_sources.md) | **1 次資料の当たり方。実際にやってしまった 4 件の間違い** | 2, 3 |
| [12_workbench_accidents.md](12_workbench_accidents.md) | ディスク満杯・並行実行・書けていなかった動画・大容量の運び方 | 2, 3 |
| [13_runpod_operations.md](13_runpod_operations.md) | **時間貸しの GPU マシンは期限ではなく残高で止まる。**ストレージの種類で $0 のとき消えるかが決まる・チェックポイント 25 GB の間引き・持ち帰りの経路 | 2, 3 |
| [14_quantization_and_eval_bottleneck.md](14_quantization_and_eval_bottleneck.md) | **低精度化が速さになる条件は3つあり、1つ欠けると効果がゼロか逆になる。**手元の評価は方策ではなくCPU描画で律速されている（方策はかかった時間の18〜29%） | 2, 3 |
| [15_instruction_normalizer_inverted_a_task.md](15_instruction_normalizer_inverted_a_task.md) | **指示文を学習側の言い方へ寄せる仕掛けが、1 課題を逆向きに壊していた** | 2, 3 |
| [16_docker_group_needs_session_restart.md](16_docker_group_needs_session_restart.md) | `usermod -aG docker` は、古いセッションが残っている間は効かない | 2, 3 |
| [17_track3_picks_the_wrong_object.md](17_track3_picks_the_wrong_object.md) | **Track3 で方策は順方向をやっている**（別の物を掴んでトレイへ入れた） | 3, 4 |
| [18_local_eval_missed_the_gain.md](18_local_eval_missed_the_gain.md) | **手元の 72 行は、採点側で 2 トラックとも上がった重みを「効果なし」と読んだ** | 2, 3, 4 |
| [19_infer_less_count_first.md](19_infer_less_count_first.md) | **推し量る前に数える。**配布物の中身・崖のある制約・期限が切れる資格情報の 3 件 | 2, 3 |
| [21_track3_stops_short_of_the_object.md](21_track3_stops_short_of_the_object.md) | **Track3 で掴めないのは迷いではない。**邪魔な物を場外へ移しても掴めず、**手先が 6〜20 cm 手前で止まる。**測る前に 2 回、嘘の数が出た | 3, 4 |
| [20_probe_object_choice_without_rollout.md](20_probe_object_choice_without_rollout.md) | **対象物を選べているかはロールアウト無しで測れる。**系譜 6 点で何を得て何を失ったか・**色はほとんど見ていない**（元から） | 3, 4 |


## 作成済みの成果物

| 成果物 | 置き場 | 概要 |
|---|---|---|
| **Claude skill（2 本）** | `~/.claude/skills/` | `ml-competition-workflow`・`experiment-bookkeeping`（日本語版: [skill_ja.md](skill_ja.md)・[skill_bookkeeping_ja.md](skill_bookkeeping_ja.md)） |
| **社内向けの参加報告** | `../report/internal_briefing.md` | スライド形式（31 分ぶん）。このディレクトリの材料を使って作成 |

