# Colab GPU で評価を回す — **[colab_eval/](colab_eval/) へ移した**

**1 ファイル 1,681 行では読み進められなくなったので、手順ごとに分けた**（8/11）。
**実行する順に番号を振ってあるので、上から順に開けばよい。**

**→ [colab_eval/README.md](colab_eval/README.md)**

**さらに 8/12、セルの中身を
[colab_eval.py](../../solution/tools/colab_eval.py) の関数へ畳んだ。**
**Colab に貼るのは読み込み 4 行と呼び出し 8 行だけである。**
下の表は**何が起きているかを読むためのページ**の一覧になった。

| 手順 | | いま呼ぶもの | 前の呼び名 |
|---|---|---|---|
| [1](colab_eval/01_gpu_and_egl.md) | **準備**（Drive・ファイルの上げ・GPU と EGL） | `ce.prepare()` | セル A ＋ **旧 D-2 のアップロード** |
| [2](colab_eval/02_kit.md) | 配布キットをセットアップし、パッチを当てる | `ce.setup_kit()` | セル B |
| [3](colab_eval/03_weights.md) | 評価する重みを用意する | `ce.prepare_weights()` | セル C |
| [4](colab_eval/04_policy_server.md) | 方策サーバーの依存を入れ、配置する | `ce.setup_policy()` | セル D-1・D-2 |
| **[5](colab_eval/05_suite.md)** | **検証用スイートへ差し替え、84 件を 3 つに割る** | `ce.build_suite()` | **セル H ＋ E-0** |
| [6](colab_eval/06_settings.md) | 評価の設定 | **設定ファイルと `run()` の引数** | セル E-1 |
| [7](colab_eval/07_speed_check.md) | 遅さの原因を切り分ける（初回のみ） | （畳んでいない） | セル E-2 |
| [8](colab_eval/08_run.md) | 評価を実行する | **`ce.run(chunk=N)`** | セル E-3 |
| [9](colab_eval/09_read.md) | 結果を読む・集計する・退避する | **同じ `run()` の中** | セル F |
| [10](colab_eval/10_notes.md) | 参考（過去の測定と注意点） | — | セル G ＋ 注意点 |

> **並べ替えたところが 1 つある。** 旧 `セル H`（検証用スイートへの差し替え）は
> **文書の最後に置かれていたのに、セル B の直後に実行する必要があった。**
> **手順 5 として本来の位置へ移した。**
