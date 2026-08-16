# docs/ — 説明の置き場

**説明はすべてここにある。** `solution/` には動かすファイルだけを置く。

| 探しているもの | 行き先 |
|---|---|
| **今なにをやるべきか** | [talk/open_items.md](talk/open_items.md) |
| **手を動かす手順** | [howto/](howto/) |
| **測った数字** | [strategy/05_実測結果.md](strategy/05_実測結果.md) |
| **次にどこを直して学習するか** | **[plan/04_training_targets.md](plan/04_training_targets.md)** |
| **これから作るものの設計** | [plan/](plan/) |
| **提出物のしくみ** | [solution/README.md](solution/README.md) |
| **次のコンペに持ち越すこと** | [talk/lessons_learned.md](talk/lessons_learned.md) |
| **レポートに書く値** | [reference/03_report_values.md](reference/03_report_values.md) |

## ディレクトリ

| パス | 中身 |
|---|---|
| [talk/](talk/) | いま抱えているものと、その優先順位。**まずここ** |
| [howto/](howto/) | 手順書。上から順にやれば終わる形で書いてある |
| [solution/](solution/) | 提出物と Colab の説明。コード本体は `../solution/` |
| [strategy/](strategy/) | 何を狙うか、なぜそうするか。判断の材料 |
| [reference/](reference/) | 調べて分かった事実（モデル・評価データ・出題元のデータ・設定の意味） |
| [plan/](plan/) | **1 つずつの手の設計。** 何を測り、出た数字で何を決めるか |
| [prod_eval_results/](../outputs/prod_eval_results/) | 本番採点の生ログと読み解き |

## よく開くファイル

| ファイル | いつ |
|---|---|
| [talk/open_items.md](talk/open_items.md) | 作業を始めるとき。手つかずのものが優先順位つきで並ぶ |
| [talk/lessons_learned.md](talk/lessons_learned.md) | コンペが終わったあと。**次の 3 日の組み立て方**まで書いてある |
| [howto/01_retrain.md](howto/01_retrain.md) | 追加学習を回すとき |
| [howto/02_build_submission.md](howto/02_build_submission.md) | zip を作って出すとき |
| **[howto/03_gpu_eval.md](howto/03_gpu_eval.md)** | **評価を回すとき（GPU マシン。放っておける）** |
| **[howto/05_scripted_demos.md](howto/05_scripted_demos.md)** | **手本を台本で作るとき（撮影は手元の Mac の Docker。GPU が要るのは学習だけ）** |
| **[howto/08_hf_transfer_troubleshooting.md](howto/08_hf_transfer_troubleshooting.md)** | **借りた GPU マシンから重みを運ぶとき。`hf upload` が 403 で止まったら** |
| **[solution/colab_eval/](solution/colab_eval/README.md)** | **評価を回すとき（Colab。貼るのは `colab_eval.py` の呼び出し 8 行）** |
| [solution/COLAB.md](solution/COLAB.md) | Colab で学習を回すとき |
| [strategy/05_実測結果.md](strategy/05_実測結果.md) | 判断する前。数字はすべてここ |
| **[strategy/06_scripted_demos.md](strategy/06_scripted_demos.md)** | **手本が 300 ステップに収まらないタスクを扱うとき。台本で手本を自分で作る** |
| **[plan/04_training_targets.md](plan/04_training_targets.md)** | **学習を流す前と、評価が出た直後。**ものさし・天井・候補・外したもの・記録が 1 枚にある |
| [plan/01_failure_accounting.md](plan/01_failure_accounting.md) | 評価の結果を読むとき。**ゴール到達と衝突の読み分け方** |
| [reference/04_track1_population.md](reference/04_track1_population.md) | 出題元のデータを確かめるとき |
| **[reference/05_perturbation_categories.md](reference/05_perturbation_categories.md)** | **摂動 7 カテゴリの中身と、どれが出るか（運営回答の原文つき）** |
| **[reference/06_training_data_composition.md](reference/06_training_data_composition.md)** | **学習データ 14,347 本の中身。摂動カテゴリで絞れるかの判定** |
| [reference/07_chunk_params.md](reference/07_chunk_params.md) | 学習後に変えてよい設定はどれか（`n_action_steps` と `chunk_size`） |
| **[reference/12_lora_targets.md](reference/12_lora_targets.md)** | **LoRA をどこに当てるか。候補 4 つと、採用したものの理由** |

## 説明とコードの分け方

```
docs/solution/     説明（README・COLAB・COLAB_EVAL・DOCKER_TRAIN）
solution/          動かすファイル（policy_server.py・requirements.txt・tools/・train/）
```

`solution/` に置いたものだけが git の追跡対象で、重み・同梱ソース・venv・zip は
`work/` へ出て管理外になる。
