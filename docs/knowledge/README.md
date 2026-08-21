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
| [12_workbench_accidents.md](12_workbench_accidents.md) | ディスク満杯・並行実行・大容量の運び方 | 2, 3 |

## 成果物 2 —— 出来た skill の日本語版（確認用）

**skill の本体は英語で `~/.claude/skills/` に置いた**（機械向けは英語、という規約）。
**中身を確認・調整するための日本語版**をここに置く。

| 日本語版 | skill 本体 | 中身 |
|---|---|---|
| [skill_ja.md](skill_ja.md) | `ml-competition-workflow`（397 行） | **中身を読めない採点器にどう当たるか。**失格条件・手元の評価の限界と 2 つの対策・切り口・設定の探索・ベース重みが何をどこまで学習済みか・最終日の決め方 |
| [skill_bookkeeping_ja.md](skill_bookkeeping_ja.md) | `experiment-bookkeeping`（276 行＋例 5 本） | **長期プロジェクトの事務管理。**ディレクトリの分け方・実行と設定と重みとログとスコアの結び付け・台帳・消す順番・未決事項の一覧 |

**事務管理は元は競技側の skill に入っていたが、コンペ固有ではないので分けた。**

## 成果物 4 —— 社内向けの参加報告

**`../report/internal_briefing.md`。**読者は同僚のエンジニアとチームリーダーで、
システム開発には習熟しているが Physical AI の経験は無い、という前提で書いた。
**見出し 1 つがスライド 1 枚に対応する形**（S1〜S24 ＋ 付録 A1〜A3・31 分ぶん）。

| 使った材料 | どのスライドに入ったか |
|---|---|
| [06](06_step_count_drives_score.md) ステップ数 | S7・S8・S9・S11・S18 |
| [07](07_local_eval_cannot_rank.md) 手元評価の限界 | S6・S10 |
| [08](08_chunk_length_follows_model_strength.md) chunk 長 | S12 |
| [09](09_two_stage_training.md) 学習を 2 回に分ける | S20 |
| [10](10_deciding_under_deadline.md) 締切下の判断 | S21 |
| [11](11_primary_sources.md) 1 次資料 | S16 |
| [12](12_workbench_accidents.md) 作業環境の事故 | S14 |
| [02](02_model_identity.md) [03](03_training_info.md) [05](05_score_history.md) の値 | S2〜S5・付録 A2 |
| [05](05_score_history.md) の公開された順位 | **S4**（42 位・分布の形）・S23 |

**次は成果物 3（人間側で意識すること）。**06〜12 の「人間側」の欄が材料になる。

