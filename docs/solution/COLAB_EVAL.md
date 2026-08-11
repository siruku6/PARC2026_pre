# Colab GPU で評価を回す — **[colab_eval/](colab_eval/) へ移した**

**1 ファイル 1,681 行では読み進められなくなったので、手順ごとに分けた**（8/11）。
**実行する順に番号を振ってあるので、上から順に開けばよい。**

**→ [colab_eval/README.md](colab_eval/README.md)**

| 手順 | | 前の呼び名 |
|---|---|---|
| [1](colab_eval/01_gpu_and_egl.md) | **準備**（Drive・ファイルの上げ・GPU と EGL） | セル A ＋ **旧 D-2 のアップロード** |
| [2](colab_eval/02_kit.md) | 配布キットをセットアップし、パッチを当てる | セル B |
| [3](colab_eval/03_weights.md) | 評価する重みを用意する | セル C |
| [4](colab_eval/04_policy_server.md) | 方策サーバーの依存を入れ、配置する | セル D-1・D-2 |
| **[5](colab_eval/05_suite.md)** | **検証用スイートへ差し替え、84 件を 3 つに割る** | **セル H ＋ E-0** |
| [6](colab_eval/06_settings.md) | 評価の設定 | セル E-1 |
| [7](colab_eval/07_speed_check.md) | 遅さの原因を切り分ける（初回のみ） | セル E-2 |
| [8](colab_eval/08_run.md) | 評価を実行する | セル E-3 |
| [9](colab_eval/09_read.md) | 結果を読む・集計する・退避する | セル F |
| [10](colab_eval/10_notes.md) | 参考（過去の測定と注意点） | セル G ＋ 注意点 |

> **並べ替えたところが 1 つある。** 旧 `セル H`（検証用スイートへの差し替え）は
> **文書の最後に置かれていたのに、セル B の直後に実行する必要があった。**
> **手順 5 として本来の位置へ移した。**
