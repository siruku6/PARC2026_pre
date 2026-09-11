# solution/tools/ — 各ディレクトリの役割

**どこに何があるかだけを書いた索引である。** 使い方の説明は
[docs/solution/README.md](../../docs/solution/README.md) と `docs/howto/` にある。

| ディレクトリ | 役割 | いつ触るか |
|---|---|---|
| `build/` | 提出物 zip を組み立てる。LeRobot の同梱、ダイジェスト、素の重みの取り寄せ | 提出物を作り直すとき |
| `weights/` | 重みの取り込み・混合（λ）・検証・退避。Track1 と π0.5 の両方 | 学習が終わって重みを差し替えるとき |
| `suite/` | 評価用タスク一覧（CSV）を作る。ホールドアウトの切り出し、指示文の正規化 | 何を測るかを決めるとき |
| `probe/` | 本選 Track2 の摂動カテゴリを測る小さな探査用 CSV と、その結果の読み直し | どの摂動で崩れるかを調べるとき |
| `analysis/` | 評価結果どうしの比較。ステップ数での対応づけ、失敗の切り分け、見た目の散らばり | 2 回の評価を比べるとき |
| `viz/` | 行動列と動画で目で見る。学習データとの重ね描き、π0.5 の 1 手確認 | 数字では区別できないとき |
| `quant/` | 量子化候補（学習側・推論側）の実装とスイープ | 学習・推論を速くする検討のとき |
| `scripted_demo/` | シミュレータの中で scripted policy から手本を作り、学習形式へ変換する | 学習データを増やすとき |
| `motion/` | **自作デモの指令を配布データと同じ形にする共有部品**（封筒・Ruckig の通過点・なめらかさの物差し）。026・028・`scripted_demo` から呼ぶ | 手本の指令の形を直すとき |
| `colab/` | Colab GPU で評価を回す一式（**Track1 用。Track2 では未検証**） | GPU マシンが学習で埋まっているとき |
| `eval/` | **評価を回す一式**（13 ファイル）。Docker の起動、キットへのパッチ当て、ロールアウトの記録、集計 | 手元で測るとき。**入口は `eval/run_docker_eval.sh`** |

**`tools/` 直下にファイルは残っていない**（この README だけ）。評価パイプラインの
13 ファイル（`run_local_eval.sh`・`run_docker_eval.sh`・`run_gpu_eval.sh`・
`*_inside.sh`・`smoke_test_*`・`make_validation_suite.py`・
`patch_kit_for_variants.py`・`check_validation_suite.py`・`rollout_recorder.py`・
`summarize_*.py`）は 2026-08-23 に `eval/` へ移した。

**`eval/` を動かすときの要注意点が 1 つある。** `run_docker_eval.sh` と
`smoke_test_suite.sh` は **`solution/tools` 全体**をコンテナの `/host/tools` に
見せ、その中の `/host/tools/eval/...` を叩く形にしてある。**マウント元を
`solution/tools/eval` に変えてはいけない**——`eval/make_validation_suite.py` が
兄弟の `suite/make_holdout_split.py` を `parents[1] / "suite"` で読むので、
`eval/` だけを見せると `ModuleNotFoundError` になる（実測済み）。

## 移動するときに壊れる 2 箇所

- **`Path(__file__).resolve().parents[N]`**：リポジトリルートを自分の位置から
  数えているファイルが多い。1 階層深くすると `parents[2]` → `parents[3]` に
  直す必要がある。**直し忘れても import は通り、実行して初めて分かる。**
- **`sys.path.insert` でディレクトリをまたいで import しているもの**（4 箇所）：
  `eval/make_validation_suite.py` → `parents[1] / "suite"`、
  `probe/make_track2_probe.py` → `parents[1] / "eval"`、
  `analysis/compare_with_production.py` → `parents[1] / "eval"`、
  `viz/visualize_demo.py` → `parents[1] / "scripted_demo"`。
  **`--help` を叩けば分かる**——import はモジュールの先頭で走るので、
  引数を解釈する前に落ちる。
- **`patch_kit_for_variants.py` の `_refuse_own_repository` は 2 行を同時に直す。**
  `parents[3]` と `len(parents) < 4` の片方だけ直すと、「配布キットを自分自身の
  リポジトリと誤認して書き換えてしまう」事故防止が**警告なしに無効化される。**
  確かめ方: `python3 solution/tools/eval/patch_kit_for_variants.py --kit-dir .`
  が「このリポジトリ自身は書き換えません」と出して終了コード 1 になること。

文書側に書かれた `PYTHONPATH=...:solution/tools` のような実行例も、
モジュールを直接 import していれば同じ理由で壊れる。
