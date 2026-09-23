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
| [22_reverse_data_interferes_with_same_objects.md](22_reverse_data_interferes_with_same_objects.md) | **逆向きデータは、同じ物体を使う順方向の課題を壊す。**共有する行は −24 ポイント、残りは −3 ポイント。**引き換えに Track3 が 11 提出ぶりに動いた**（0.000 → 0.0235） | 3, 4 |
| [23_self_made_demos_need_the_instruction_the_model_sees.md](23_self_made_demos_need_the_instruction_the_model_sees.md) | **自作した手本は、評価でモデルが実際に見る文で入れないと 1 度も引かれない。**摂動下再生 1,009 本のうち 626 本（62%）がこれを踏んでいた。**沈黙して失敗する**——学習は通り、損失も下がる | 2, 3 |
| [24_license_provenance.md](24_license_provenance.md) | **公開している 18 件のうち、条件に当たるのは MolmoAct2 の 1 件だけ。**LIBERO はデータが CC BY 4.0・LIBERO-plus は作者の HF が MIT・本選ルールは外部公開について無記載。**π0.5 の土台は Apache 2.0 ではなく Gemma 利用規約である** | 2, 3 |
| [25_lora_before_base_weights_drops_keys_silently.md](25_lora_before_base_weights_drops_keys_silently.md) | **LoRA を当ててから重みを読むと、PEFT がパラメータ名を書き換えて不一致になり、多くの実装は不一致キーを黙って捨てる。**MolmoAct2 で 1,295 個の safetensors が読めないまま学習が始まり、20 歩後の `loss:nan` としてしか症状が出なかった。**読み込み直後にキーの一致数を確かめる** | 2, 3 |
| [20_probe_object_choice_without_rollout.md](20_probe_object_choice_without_rollout.md) | **対象物を選べているかはロールアウト無しで測れる。**系譜 6 点で何を得て何を失ったか・**色はほとんど見ていない**（元から） | 3, 4 |
| [24_reached_but_zero_is_collision_at_grasp.md](24_reached_but_zero_is_collision_at_grasp.md) | **目標に届いたのに 0 点になる道は衝突だけ。**その失点の 65% は丼を掴む瞬間に出ており、**丼の手本を作り直すと 0.522 → 0.111 に落ちた**（6.1 節） | 3, 4 |
| [25_removing_bad_demos_teaches_nothing.md](25_removing_bad_demos_teaches_nothing.md) | **汚れた手本を外すだけでは衝突は減らない。**外すと汚れは消えるが、正解の手本は 1 本も入らない | 2, 3, 4 |
| [26_adding_data_lowered_the_score.md](26_adding_data_lowered_the_score.md) | **データを足して点が下がった 3 つの道**——課題を薄める・正規化の尺度を書き換える・同じ物に逆の動きを教える | 2, 3, 4 |
| [27_clipped_gradients_do_not_shrink_adamw_steps.md](27_clipped_gradients_do_not_shrink_adamw_steps.md) | `grdn` が 1.0 を超え続けても学習は止まっていない。疑うのは上限ちょうどに積み上がった指令 | 2, 3 |
| [28_continued_training_is_not_additive.md](28_continued_training_is_not_additive.md) | **続きの学習は足し算にならない。**2 トラックそろって上がったのは画像側まで開けた 1,500 歩の 1 本だけ | 3, 4 |
| [29_inference_knobs_ran_out.md](29_inference_knobs_ran_out.md) | **推論側の摘みは本選で尽きた。**8 本の採点で読めたのは壊した 1 件とトラックで向きが割れた 2 件だけ | 3, 4 |
| [30_leaderboard_is_an_instrument.md](30_leaderboard_is_an_instrument.md) | **リーダーボードは成績ではなく計器である。**本選の提出枠の使い方 | 2, 3, 4 |
| [31_time_reversal_gives_only_half_a_demo.md](31_time_reversal_gives_only_half_a_demo.md) | **逆向きの課題のデータは時刻反転では半分しか作れない。**掴む区間は自分で作り、壊れるのは継ぎ目 | 3, 4 |
| [32_splitting_work_across_sessions_and_machines.md](32_splitting_work_across_sessions_and_machines.md) | **複数のセッションと機械で分担するときに踏んだ穴。**同じ名前が別の中身を指し、古い 1 行が学習を二重に起こしかけた | 2, 3 |
| [33_offline_scoring_breaks_hub_names_in_configs.md](33_offline_scoring_breaks_hub_names_in_configs.md) | 設定に焼かれた Hub の名前は、通信の無い採点環境で 0 点になる | 2, 3 |
| [34_no_difference_is_not_permission_to_swap.md](34_no_difference_is_not_permission_to_swap.md) | 手元で「差が無い」ことは、入れ替えてよいという意味ではない | 2, 3 |
| [35_multibyte_after_variable_breaks_on_macos.md](35_multibyte_after_variable_breaks_on_macos.md) | 変数のすぐ後ろに全角文字を置くと macOS の bash で名前が繋がる | 2, 3 |
| [36_stale_tooling_teaches_the_wrong_source.md](36_stale_tooling_teaches_the_wrong_source.md) | 使わなくなった道具は、間違った出どころを教え続ける | 2, 3 |
| [37_unsettled_scenes_score_zero_forever.md](37_unsettled_scenes_score_zero_forever.md) | **場面が落ち着く前に基準を取るので、一部の行はどのモデルでも必ず 0 点になる。**選定に使っていた 45 行のうち 9 行（20%）がこれだった | 2, 3, 4 |


## 作成済みの成果物

| 成果物 | 置き場 | 概要 |
|---|---|---|
| **Claude skill（2 本）** | `~/.claude/skills/` | `ml-competition-workflow`・`experiment-bookkeeping`（日本語版: [skill_ja.md](skill_ja.md)・[skill_bookkeeping_ja.md](skill_bookkeeping_ja.md)） |
| **社内向けの参加報告** | `../report/internal_briefing.md` | スライド形式（31 分ぶん）。このディレクトリの材料を使って作成 |

