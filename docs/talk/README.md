# talk —— 一時ファイルの置き場。**他の文書から参照させない**

**ここにあるのは、その日の会話・その日の段取り・別セッションへの引き継ぎである。**
どれも**役目が終われば消す**。だから**他のディレクトリの文書がここを指してはいけない。**

## 1 つだけの決まり —— `docs/talk/` の外から `talk/` のファイルを指さない

**`howto/`・`knowledge/`・`reference/`・`plan/`・`report/`・`NOW.md` に
`talk/…` へのリンクを書かない。**ここは消える前提の置き場なので、**リンクを張った側が
必ず先に壊れる。**

**指したくなったら、それは「そのファイルにしか無い情報がある」ということである。**
リンクではなく**情報を移す。**行き先は決まっている。

| 移すもの | 行き先 |
|---|---|
| 2 回以上打つ手順・コマンドの並び | `howto/` に節を作る |
| 分かったこと・踏んだ穴・外した読み | `knowledge/` |
| 測った値・採否の理由・重みの素性 | `reference/19_weights_catalog.md` |
| 数え上げた事実（データセットの中身・摂動の種類） | `reference/` |
| 追う／追わないの結論 | `talk/open_items.md` の「決めた — 蒸し返さない」 |

**`talk/` の中どうしのリンクは張ってよい**（`open_items.md` → `open_items_notes.md` など）。
消すときに同じディレクトリの中だけを見れば済むからである。

## 消すときにやること

1. `grep -rn "<ファイル名>" docs/ NOW.md` で、指しているものを全部出す
2. **そのファイルにしか無い情報を上の表の行き先へ移す**
3. 残ったリンクを消すか、移し先へ差し替える
4. ファイルを消す

**引用の中に `talk/…:123` のような出典が残ることはある**（`260828_word_audit.md` の
語の洗い出しなど）。**引用した文そのものが書き写してあれば、それは残してよい**——
リンクではなく記録である。

## いま置いてあるもの

| ファイル | 何のためのものか | 消す目安 |
|---|---|---|
| [open_items.md](open_items.md) | **本選 1 の `open_items.md`。**候補と「決めた — 蒸し返さない」 | 本選が終わるまで残す |
| [open_items_notes.md](open_items_notes.md) | `open_items.md` の各行の根拠と経緯 | 同上 |
| [archive_qualifier_board.md](archive_qualifier_board.md) | 予選の `open_items.md` の退避先。**Track2 準備期（8/21〜8/26）のぶんは 9/2 に消した** | 同上 |
| [lessons_learned.md](lessons_learned.md) | 次のコンペへ持ち越すもの | **成果物である。**コンペ後も残す |
| `YYMMDD_today.md`（例: `260831_today.md`。**8/31 分は消した**） | **その日にやること 1 枚。**移動中に読める形 | **その日が終わったら消す。**残す値や結論は上の表の行き先へ移す |
| [260828_word_audit.md](260828_word_audit.md) | 日本語の言い換え候補 44 語の洗い出し | 置き換えが終わった語から消していく |
| [handoff_stove_script.md](handoff_stove_script.md) | 別セッションへの引き継ぎ | 受け取られたら消す |
| [inter_sessions/](inter_sessions/) | 別セッションへの依頼 1 件 1 ファイル | 閉じたら消す（[その README](inter_sessions/README.md)） |

**8/31 に 5 本消した**——`260813_today.md`・`260814_today.md`・`260821_today.md`・
`260826_27_submission.md`・`handoff_validation_set.md`。**中身の行き先はこうである。**

| 消した | 何だったか | 中身はどこにある |
|---|---|---|
| `260813_today.md` | 8/13 の段取り。scripted policy を混ぜる関門 A・B・C | [../howto/05_scripted_demos.md](../howto/05_scripted_demos.md) の 8 節。コンロを選んだ理由は [handoff_stove_script.md](handoff_stove_script.md)、なましが指に掛からないことは [../plan/19](../plan/19_smoothness_beyond_postprocessing.md) 4 節 |
| `260814_today.md` | 8/14 の段取り。0.485 で決着した日 | 「手元の 36 件では順位が付かない」は [../knowledge/07_local_eval_cannot_rank.md](../knowledge/07_local_eval_cannot_rank.md)、ステップ数が点を決めることは [../knowledge/06_step_count_drives_score.md](../knowledge/06_step_count_drives_score.md) |
| `260821_today.md` | 8/21 の夜に回したもの | 起動コマンドは `open_items_archive_260821.md` に書き写したが、**そちらも 9/2 に消した** |
| `260826_27_submission.md` | 8/26・8/27 の提出計画 | RTX6000 で評価を回すときの注意は [../howto/03_gpu_eval.md](../howto/03_gpu_eval.md) の「会社の RTX6000 では EGL が効く」 |
| `handoff_validation_set.md` | 検証集合を学習と分ける提案（8/12） | **採らなかった**——[../reference/13_holdout_episodes.md](../reference/13_holdout_episodes.md) の「タスク単位で外すのは採らなかった」。`libero_plus_id` を抜くだけでは足りないことは [../howto/04_proxy_eval.md](../howto/04_proxy_eval.md) |

**9/2 にさらに 5 本消した。**

| 消した | 何だったか | 中身はどこにある |
|---|---|---|
| `260825_briefing.md` | 8/25 の全体像（印刷して読む用・1,112 行） | **Track3 の中身は [../reference/28_track3_contents.md](../reference/28_track3_contents.md)。**本選のルールは [../plan/14_finals1_gpu_and_submission_plan.md](../plan/14_finals1_gpu_and_submission_plan.md) の「いまどこにいるか」と一次資料の `../materials/PARC2026開発コンペティション_本選_v1.1.pdf`。配布データセットの実測は [../reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md)。`samples` の選び方は [../reference/07_chunk_params.md](../reference/07_chunk_params.md)。取り違えの教訓 3 つは [../knowledge/19_infer_less_count_first.md](../knowledge/19_infer_less_count_first.md) |
| `260831_today.md` | 8/31 の段取り（1,121 行） | **Blackwell の手順は [../howto/14_blackwell_session.md](../howto/14_blackwell_session.md) と [../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md)。**HF のトークンの種類は [../howto/08_hf_transfer_troubleshooting.md](../howto/08_hf_transfer_troubleshooting.md) と [../resources/huggingface.md](../resources/huggingface.md) の 0 節。学習の結果は [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) の「8/31 の `pi05_ladder2500`」 |
| `handoff_pi05_stage2.md` | π0.5 の 2 段学習の引き継ぎ（8/13） | [../howto/07_train_pi05_local.md](../howto/07_train_pi05_local.md) の「段2を始めるときの注意 4 つ」。`--init-from` が歩数と学習率の刻みを引き継がないことは [../reference/27_blackwell_environment.md](../reference/27_blackwell_environment.md) |
| `handoff_training_targets.md` | 学習の的の並べ替え（8/13） | **中身は全部ほかにある**——的の候補は [../plan/04_training_targets.md](../plan/04_training_targets.md)、失敗の型は [../reference/18_failure_breakdown.md](../reference/18_failure_breakdown.md)、Q&A No.33 の線は [../reference/13_holdout_episodes.md](../reference/13_holdout_episodes.md) |
| `open_items_archive_260821.md` | Track2 準備期（8/21〜8/26）の板 | **残すものは無かった。**Track2 の 7 項目は [../plan/07_track2_preparation.md](../plan/07_track2_preparation.md)、予選から引き継いだ番号は [archive_qualifier_board.md](archive_qualifier_board.md) |

**同じ 8/31 に、`talk/` の外でも 3 本消した**——`reference/03_report_values.md`（予選レポート用の値。
提出済みで、値は `reference/19` と `knowledge/04` にある）・`howto/12_bring_eval_to_rtx6000.md`
（持ち込みは 8/26 に済んだ。回すときの注意は `howto/03` へ移した）・
`docs/PARC2026  テクニカル関連質問_260810.csv`（古い版。新しい版が同じ 204 行を全部持ち、35 行に回答が付いた。
新しい版は 9/1 に [../materials/](../materials/) へ移した）。
