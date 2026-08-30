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

**`talk/` の中どうしのリンクは張ってよい**（板 → `open_items_notes.md` など）。
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
| [open_items.md](open_items.md) | **本選 1 の板。**候補と「決めた — 蒸し返さない」 | 本選が終わるまで残す |
| [open_items_notes.md](open_items_notes.md) | 板の各行の根拠と経緯 | 同上 |
| [open_items_archive_260821.md](open_items_archive_260821.md) | Track2 準備期（8/21〜8/26）の板の退避先 | 番号で指されているので残す |
| [archive_qualifier_board.md](archive_qualifier_board.md) | 予選の板の退避先 | 同上 |
| [lessons_learned.md](lessons_learned.md) | 次のコンペへ持ち越すもの | **成果物である。**コンペ後も残す |
| [260810_briefing.md](260810_briefing.md)・[260812_briefing.md](260812_briefing.md)・[260825_briefing.md](260825_briefing.md) | その日の全体像（印刷して読む用） | 次の briefing が出たら消せる |
| [260826_training_run.md](260826_training_run.md) | 8/26 の学習 2 本の手順と条件 | 結果が `reference/19` に入れば消せる |
| `NNNNNN_today.md`（[260831_today.md](260831_today.md) など） | **その日にやること 1 枚。**移動中に読める形 | **その日が終わったら消す。**残す値や結論は上の表の行き先へ移す |
| [260828_word_audit.md](260828_word_audit.md) | 日本語の言い換え候補 44 語の洗い出し | 置き換えが終わった語から消していく |
| [handoff_pi05_stage2.md](handoff_pi05_stage2.md)・[handoff_stove_script.md](handoff_stove_script.md)・[handoff_training_targets.md](handoff_training_targets.md) | 別セッションへの引き継ぎ | 受け取られたら消す |
| [inter_sessions/](inter_sessions/) | 別セッションへの依頼 1 件 1 ファイル | 閉じたら消す（[その README](inter_sessions/README.md)） |

**8/31 に 5 本消した**——`260813_today.md`・`260814_today.md`・`260821_today.md`・
`260826_27_submission.md`・`handoff_validation_set.md`。**中身の行き先はこうである。**

| 消した | 何だったか | 中身はどこにある |
|---|---|---|
| `260813_today.md` | 8/13 の段取り。scripted policy を混ぜる関門 A・B・C | [../howto/05_scripted_demos.md](../howto/05_scripted_demos.md) の 8 節。コンロを選んだ理由は [handoff_stove_script.md](handoff_stove_script.md)、なましが指に掛からないことは [../plan/19](../plan/19_smoothness_beyond_postprocessing.md) 4 節 |
| `260814_today.md` | 8/14 の段取り。0.485 で決着した日 | 「手元の 36 件では順位が付かない」は [../knowledge/07_local_eval_cannot_rank.md](../knowledge/07_local_eval_cannot_rank.md)、ステップ数が点を決めることは [../knowledge/06_step_count_drives_score.md](../knowledge/06_step_count_drives_score.md) |
| `260821_today.md` | 8/21 の夜に回したもの | 起動コマンドは [open_items_archive_260821.md](open_items_archive_260821.md) に書き写した |
| `260826_27_submission.md` | 8/26・8/27 の提出計画 | RTX6000 へ置くものと回し方は [../howto/12_bring_eval_to_rtx6000.md](../howto/12_bring_eval_to_rtx6000.md) |
| `handoff_validation_set.md` | 検証集合を学習と分ける提案（8/12） | **採らなかった**——[../reference/13_holdout_episodes.md](../reference/13_holdout_episodes.md) の「タスク単位で外すのは採らなかった」。`libero_plus_id` を抜くだけでは足りないことは [../howto/04_proxy_eval.md](../howto/04_proxy_eval.md) |
