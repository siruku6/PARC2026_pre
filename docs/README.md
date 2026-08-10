# docs/ — 説明の置き場

**説明はすべてここにある。** `solution/` には動かすファイルだけを置く。

| 探しているもの | 行き先 |
|---|---|
| **今なにをやるべきか** | [talk/open_items.md](talk/open_items.md) |
| **手を動かす手順** | [howto/](howto/) |
| **測った数字** | [strategy/05_実測結果.md](strategy/05_実測結果.md) |
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
| [reference/](reference/) | 調べて分かった事実（モデル・評価データ） |
| [prod_eval_results/](prod_eval_results/) | 本番採点の生ログと読み解き |

## よく開くファイル

| ファイル | いつ |
|---|---|
| [talk/open_items.md](talk/open_items.md) | 作業を始めるとき。手つかずのものが優先順位つきで並ぶ |
| [talk/lessons_learned.md](talk/lessons_learned.md) | コンペが終わったあと。**次の 3 日の組み立て方**まで書いてある |
| [howto/01_retrain.md](howto/01_retrain.md) | 追加学習を回すとき |
| [howto/02_build_submission.md](howto/02_build_submission.md) | zip を作って出すとき |
| [solution/COLAB_EVAL.md](solution/COLAB_EVAL.md) | Colab で評価を回すとき |
| [solution/COLAB.md](solution/COLAB.md) | Colab で学習を回すとき |
| [strategy/05_実測結果.md](strategy/05_実測結果.md) | 判断する前。数字はすべてここ |

## 説明とコードの分け方

```
docs/solution/     説明（README・COLAB・COLAB_EVAL・DOCKER_TRAIN）
solution/          動かすファイル（policy_server.py・requirements.txt・tools/・train/）
```

`solution/` に置いたものだけが git の追跡対象で、重み・同梱ソース・venv・zip は
`work/` へ出て管理外になる。
