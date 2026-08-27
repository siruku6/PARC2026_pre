# docs/ — 説明の置き場

**説明はすべてここにある。** `solution/` には動かすファイルだけを置く。

| 探しているもの | 行き先 |
|---|---|
| **今なにをやるべきか** | [talk/open_items.md](talk/open_items.md) |
| **手を動かす手順** | [howto/](howto/) |
| **測った数字** | [strategy/05_実測結果.md](strategy/05_実測結果.md) |
| **次にどこを直して学習するか** | **[plan/04_training_targets.md](plan/04_training_targets.md)** |
| **本選 Track2（位置・視点・ノイズ）の準備** | **[plan/07_track2_preparation.md](plan/07_track2_preparation.md)** |
| **これから作るものの設計** | [plan/](plan/) |
| **提出物のしくみ** | [solution/README.md](solution/README.md) |
| **次のコンペに持ち越すこと** | [talk/lessons_learned.md](talk/lessons_learned.md) |
| **レポートに書く値** | [reference/03_report_values.md](reference/03_report_values.md) |
| **流した学習・重み・版・出した点（実験台帳）** | **[reference/19_weights_catalog.md](reference/19_weights_catalog.md)** |
| **今使える計算資源・docker image は何か** | **[resources/infra_and_docker.md](resources/infra_and_docker.md)** |
| **HF にどのモデルが置いてあるか・ローカルの重みに控えがあるか** | **[resources/huggingface.md](resources/huggingface.md)** |
| **ディスクを空けたい** | **[howto/09_free_disk_space.md](howto/09_free_disk_space.md)** |
| **このマシンを作り替える／作り替えた** | **[howto/10_rebuild_machine.md](howto/10_rebuild_machine.md)** |

## ディレクトリ

| パス | 中身 |
|---|---|
| [talk/](talk/) | いま抱えているものと、その優先順位。**まずここ** |
| [howto/](howto/) | 手順書。上から順にやれば終わる形で書いてある |
| [solution/](solution/) | 提出物と Colab の説明。コード本体は `../solution/` |
| [strategy/](strategy/) | 何を狙うか、なぜそうするか。判断の材料 |
| [reference/](reference/) | 調べて分かった事実（モデル・評価データ・出題元のデータ・設定の意味） |
| [resources/](resources/) | **今使える** 計算資源・ディスク・メモリ・docker image・HF 上のモデルの一覧。状態が変わるたびに上書きする（履歴は残さない）。**手順や Tips は書かない**（[resources/README.md](resources/README.md) の線引き） |
| [plan/](plan/) | **1 つずつの手の設計。** 何を測り、出た数字で何を決めるか |
| [prod_eval_results/](../outputs/prod_eval_results/) | 本番採点の生ログと読み解き |

## よく開くファイル

| ファイル | いつ |
|---|---|
| [talk/open_items.md](talk/open_items.md) | 作業を始めるとき。**本選 Track2 の板**（3,300 文字。8/21 に作り直した） |
| [talk/archive_qualifier_board.md](talk/archive_qualifier_board.md) | **予選の板の退避先。**他の文書が「`open_items.md` の 29」のように**番号で指しているものはここにある** |
| [talk/lessons_learned.md](talk/lessons_learned.md) | コンペが終わったあと。**次の 3 日の組み立て方**まで書いてある |
| [howto/01_retrain.md](howto/01_retrain.md) | 追加学習を回すとき |
| [howto/02_build_submission.md](howto/02_build_submission.md) | zip を作って出すとき |
| **[howto/03_gpu_eval.md](howto/03_gpu_eval.md)** | **評価を回すとき（GPU マシン。放っておける）** |
| **[howto/05_scripted_demos.md](howto/05_scripted_demos.md)** | **手本を台本で作るとき（撮影は手元の Mac の Docker。GPU が要るのは学習だけ）** |
| **[howto/08_hf_transfer_troubleshooting.md](howto/08_hf_transfer_troubleshooting.md)** | **借りた GPU マシンから重みを運ぶとき。`hf upload` が 403 で止まったら** |
| **[howto/09_free_disk_space.md](howto/09_free_disk_space.md)** | **`df` の空きが減ってきたとき。消してよいものの判定と、「消したのに空かない」3 つの罠** |
| **[howto/10_rebuild_machine.md](howto/10_rebuild_machine.md)** | **マシンを作り替えてディスクが消えるとき。持ち出す 1 本のスクリプト・秘密の 4 件・常駐コンテナの起動コマンド** |
| **[howto/12_bring_eval_to_rtx6000.md](howto/12_bring_eval_to_rtx6000.md)** | **会社の RTX6000 で本選の自己評価を回すとき。EGL が効くのでここが評価の主力になる。持ち込む 4 つ（評価イメージ・方策イメージ・重み・コード）と、`docker` が sudo を要るときの回避** |
| **[howto/13_docker_sudo_group_troubleshooting.md](howto/13_docker_sudo_group_troubleshooting.md)** | **`usermod -aG docker` を入れたのに `permission denied` が続くとき。`id -nG` での切り分けと、VS Code の Remote-SSH で必要になる操作** |
| **[howto/14_blackwell_first_session.md](howto/14_blackwell_first_session.md)** | **配布 GPU 環境（Blackwell サーバ）に初めて入るとき。60 時間の減り方・下見・配布データの数え方・止め方。自分の Docker は持ち込めない** |
| **[howto/11_colab_eval.md](howto/11_colab_eval.md)** | **評価を回すとき（Colab。8 ファイルを上げてセルを 5 つ。π0.5 も回せる。T4 で動くかは未確認）** |
| [strategy/05_実測結果.md](strategy/05_実測結果.md) | 判断する前。数字はすべてここ |
| **[strategy/06_scripted_demos.md](strategy/06_scripted_demos.md)** | **手本が 300 ステップに収まらないタスクを扱うとき。台本で手本を自分で作る** |
| **[plan/04_training_targets.md](plan/04_training_targets.md)** | **学習を流す前と、評価が出た直後。**ものさし・天井・候補・外したもの・記録が 1 枚にある |
| [plan/01_failure_accounting.md](plan/01_failure_accounting.md) | 評価の結果を読むとき。**ゴール到達と衝突の読み分け方** |
| **[plan/07_track2_preparation.md](plan/07_track2_preparation.md)** | **本選 Track2 に手をつけるとき。摂動 4 カテゴリの中身・関連研究・対策・推奨アクションが 1 枚にある。まず 2.0 節（方針改訂）と 3 節（推奨アクション）** |
| **[plan/08_training_data_creation_cautions.md](plan/08_training_data_creation_cautions.md)** | **学習用デモを作る前。成功例だけでは復帰を学べない等、データ作成の落とし穴。07 の方針改訂の根拠** |
| **[reference/19_weights_catalog.md](reference/19_weights_catalog.md)** | **学習を流したあと、点が返ってきたあと、`pi05/full_local_024000` のような名前を見たとき。流した学習・重み・版ごとの設定・本選のトラック別の点が 1 ファイルに並ぶ** |
| [reference/04_track1_population.md](reference/04_track1_population.md) | 出題元のデータを確かめるとき |
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
