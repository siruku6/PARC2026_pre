# docs/ — 説明の置き場

**説明はすべてここにある。** `solution/` には動かすファイルだけを置く。

| 探しているもの | 行き先 |
|---|---|
| **いま手を動かしているタスク** | **[reference/32_current_tasks.md](reference/32_current_tasks.md)** ── 着手中のものだけ。候補の全体と枝分かれは [plan/22](plan/22_grasp_and_color_next_moves.md) |
| **今なにをやるべきか（候補の `open_items.md`）** | **[talk/open_items.md](talk/open_items.md)** ── 1 行ごとに「何が決まるか・誰が・資源・所要・前提・打つもの」を持つ。**アイデアを思いついたらここへ足す** |
| **手を動かす手順** | [howto/](howto/) |
| **測った数字** | [strategy/05_実測結果.md](strategy/05_実測結果.md) |
| **次にどこを直して学習するか** | **[reference/11_training_schedule.md](reference/11_training_schedule.md)** |
| **本選 Track2（位置・視点・ノイズ）の準備** | **[reference/20_track2_perturbation_categories.md](reference/20_track2_perturbation_categories.md)**（弱点の地図は [reference/21](reference/21_weakness_map.md)） |
| **これから作るものの設計** | [plan/](plan/) |
| **提出物のしくみ** | [solution/README.md](solution/README.md) |
| **次のコンペに持ち越すこと** | [talk/lessons_learned.md](talk/lessons_learned.md) |
| **レポートの必須欄と締切** | [knowledge/01_report_requirements.md](knowledge/01_report_requirements.md)（**書く値そのものは実験台帳とライセンスの 2 つから引く**——下の [reference/19](reference/19_weights_catalog.md)・[knowledge/04](knowledge/04_licenses.md)） |
| **流した学習・重み・版・出した点（実験台帳）** | **[reference/19_weights_catalog.md](reference/19_weights_catalog.md)** |
| **今使える計算資源・docker image は何か** | **[resources/infra_and_docker.md](resources/infra_and_docker.md)** |
| **Blackwell で測った速さ・容量・パスの実体・踏んだ罠** | **[reference/27_blackwell_environment.md](reference/27_blackwell_environment.md)**（**手順は [howto/14_blackwell_session.md](howto/14_blackwell_session.md)**。値はこの 1 つだけが持つ） |
| **HF にどのモデルが置いてあるか・ローカルの重みに控えがあるか** | **[resources/huggingface.md](resources/huggingface.md)** |
| **ディスクを空けたい** | **[howto/09_free_disk_space.md](howto/09_free_disk_space.md)** |
| **このマシンを作り替える／作り替えた** | **[howto/10_rebuild_machine.md](howto/10_rebuild_machine.md)** |

## ディレクトリ

| パス | 中身 |
|---|---|
| [materials/](materials/) | **外部から取得した資料の現物。**ルールの PDF・運営 Q&A の CSV・運営の告知・講義資料。**一次資料はここを見る**（[knowledge/11_primary_sources.md](knowledge/11_primary_sources.md)）。**ここのファイルは書き換えない** |
| [talk/](talk/) | いま抱えているものと、その優先順位。**まずここ。**ただし**一時ファイルの置き場なので、他のディレクトリの文書からここを指してはいけない**（[talk/README.md](talk/README.md)。指したくなったら情報を `howto/`・`knowledge/`・`reference/` へ移す） |
| [howto/](howto/) | 手順書。上から順にやれば終わる形で書いてある。**何度も使うものだけを置く。**1 回だけの回・1 回だけの設定は [plan/](plan/) へ、踏んだ事故と回避策は [knowledge/](knowledge/) へ |
| [solution/](solution/) | 提出物と Colab の説明。コード本体は `../solution/` |
| [strategy/](strategy/) | 何を狙うか、なぜそうするか。判断の材料 |
| [reference/](reference/) | 調べて分かった事実（モデル・評価データ・採点のタスクの母集団・設定の意味） |
| [resources/](resources/) | **今使える** 計算資源・ディスク・メモリ・docker image・HF 上のモデルの一覧。状態が変わるたびに上書きする（履歴は残さない）。**手順や Tips は書かない**（[resources/README.md](resources/README.md) の線引き） |
| [plan/](plan/) | **1 つずつの手の設計。** 何を測り、出た数字で何を決めるか |
| [outputs/submission_logs/](../outputs/submission_logs/) | 本番採点の生ログ。**トラックごとに `<日付>_<版>.txt`**（本選は `b_PARC2026_final/outputs/submission_logs/`） |

## よく開くファイル

| ファイル | いつ |
|---|---|
| **[talk/open_items.md](talk/open_items.md)** | **作業を始めるとき、次の手に迷ったとき。本選 1 の `open_items.md`**（8/27 に作り直した。それ以前（8/21〜8/26）のぶんは消した） |
| [talk/archive_qualifier_board.md](talk/archive_qualifier_board.md) | **予選の `open_items.md` の退避先。**他の文書が「`open_items.md` の 29」のように**番号で指しているものはここにある** |
| [talk/lessons_learned.md](talk/lessons_learned.md) | コンペが終わったあと。**次の 3 日の組み立て方**まで書いてある |
| [howto/01_retrain.md](howto/01_retrain.md) | 追加学習を回すとき |
| [howto/02_build_submission.md](howto/02_build_submission.md) | zip を作って出すとき |
| **[howto/03_gpu_eval.md](howto/03_gpu_eval.md)** | **評価を回すとき（GPU マシン。放っておける）** |
| **[howto/05_scripted_demos.md](howto/05_scripted_demos.md)** | **手本を scripted policy で作るとき（エピソードの作成は手元の Mac の Docker。GPU が要るのは学習だけ）** |
| **[howto/08_hf_transfer_troubleshooting.md](howto/08_hf_transfer_troubleshooting.md)** | **借りた GPU マシンから重みを運ぶとき。`hf upload` が 403 で止まったら** |
| **[howto/09_free_disk_space.md](howto/09_free_disk_space.md)** | **`df` の空きが減ってきたとき。消してよいものの判定と、「消したのに空かない」3 つの罠** |
| **[howto/10_rebuild_machine.md](howto/10_rebuild_machine.md)** | **マシンを作り替えてディスクが消えるとき。持ち出す 1 本のスクリプト・秘密の 4 件・常駐コンテナの起動コマンド** |
| **[knowledge/16_docker_group_needs_session_restart.md](knowledge/16_docker_group_needs_session_restart.md)** | **`usermod -aG docker` を入れたのに `permission denied` が続くとき。`id -nG` での切り分けと、VS Code の Remote-SSH で必要になる操作** |
| **[howto/06_runpod_pi05.md](howto/06_runpod_pi05.md)** | **RunPod でフルパラメータ学習を回すとき。14 節の長い学習の段取り・14.5 節の立ち上がり確認** |
| **[howto/14_blackwell_session.md](howto/14_blackwell_session.md)** | **配布 GPU 環境（Blackwell サーバ）に初めて入るとき。60 時間の減り方・下見・配布データの数え方・止め方。自分の Docker は持ち込めない** |
| **[howto/11_colab_eval.md](howto/11_colab_eval.md)** | **評価を回すとき（Colab。8 ファイルを上げてセルを 5 つ。π0.5 も回せる。T4 で動くかは未確認）** |
| [strategy/05_実測結果.md](strategy/05_実測結果.md) | 判断する前。数字はすべてここ |
| **[strategy/06_scripted_demos.md](strategy/06_scripted_demos.md)** | **手本が 300 ステップに収まらないタスクを扱うとき。scripted policy で手本を自分で作る** |
| **[reference/11_training_schedule.md](reference/11_training_schedule.md)** | **学習を流す前と、評価が出た直後。次にどこを直すか（ものさし・候補・外したもの）** |
| [reference/18_failure_breakdown.md](reference/18_failure_breakdown.md) | 評価の結果を読むとき。**失敗が 4 つの型に割れる内訳と、どの型に手を入れるか** |
| **[reference/20_track2_perturbation_categories.md](reference/20_track2_perturbation_categories.md)** | **本選 Track2 に手をつけるとき。摂動 4 カテゴリの中身・実測・対策（弱点の地図は [reference/21](reference/21_weakness_map.md)）** |
| **[plan/08_training_data_creation_cautions.md](plan/08_training_data_creation_cautions.md)** | **学習用デモを作る前。成功例だけでは復帰を学べない等、データ作成の落とし穴** |
| **[reference/19_weights_catalog.md](reference/19_weights_catalog.md)** | **学習を流したあと、点が返ってきたあと、`pi05/full_local_024000` のような名前を見たとき。流した学習・重み・版ごとの設定・本選のトラック別の点が 1 ファイルに並ぶ** |
| [reference/04_track1_population.md](reference/04_track1_population.md) | 採点のタスクの母集団を確かめるとき |
| **[reference/05_perturbation_categories.md](reference/05_perturbation_categories.md)** | **摂動 7 カテゴリの中身と、どれが出るか（運営回答の原文つき）** |
| **[reference/06_training_data_composition.md](reference/06_training_data_composition.md)** | **学習データ 14,347 本の中身。摂動カテゴリで絞れるかの判定** |
| [reference/07_chunk_params.md](reference/07_chunk_params.md) | 学習後に変えてよい設定はどれか（`n_action_steps` と `chunk_size`） |
| **[reference/12_lora_targets.md](reference/12_lora_targets.md)** | **LoRA をどこに当てるか。候補 4 つと、採用したものの理由** |

## 説明とコードの分け方

```
docs/solution/     説明（README・colab_eval・DOCKER_TRAIN）
solution/          動かすファイル（policy_server.py・requirements.txt・tools/・train/）
```

`solution/` に置いたものだけが git の追跡対象で、重み・同梱ソース・venv・zip は
`work/` へ出て管理外になる。
