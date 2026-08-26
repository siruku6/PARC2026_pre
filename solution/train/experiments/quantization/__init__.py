"""π0.5 の学習に量子化候補を差し込むための、train_pi05.py 側の窓口。

**量子化の実体はこのパッケージの中の4ファイルにあり、train_pi05.py はここが公開する
6つの関数を呼ぶだけである。** 候補の中身（何をどう低精度にするか）を足したり外したり
しても、train_pi05.py を触らずに済むように分けてある。

    quant_optim.py               C1  bitsandbytes の 8-bit AdamW に差し替える
    quant_vision_4bit.py         C2  凍結中の vision encoder を 4bit 化する
    quant_float8_expert.py       C3  action expert の Linear を torchao の float8 にする
    lerobot_train_with_quant.py      上の候補を lerobot-train の前に import する薄いラッパー

**3候補はどれも、対応するオプションを指定しない限り一切効かない。** 既定はすべて
`default` で、指定しなければ量子化なしのときと一字一句同じコマンドが組まれる。
`experiments/` の下に置いてあるのはこのためで、C1・C2 は速度が出ないことを実測して
廃止した後も、その実測値の出どころとして残してある（work/quant_sweep/CANDIDATES.md）。

**C3 の「学習が -53% 速くなった」は float8 の効果ではなく torch.compile の効果である**
（float8 単体は +10.4% の遅化。docs/knowledge/14_quantization_and_eval_bottleneck.md）。
float8 を残すかどうかは、compile だけを入れた対照の測定で決まる。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# 量子化候補のどれかを使うときは "lerobot-train"（console script）の代わりに、
# 各候補モジュールを import してから呼ぶ薄いラッパーを使う（詳細は
# lerobot_train_with_quant.py の docstring）。どの候補を有効にするかは
# 環境変数で伝える（set_environment が train_pi05.py の main() から呼ばれる）。
_WRAPPER = Path(__file__).resolve().parent / "lerobot_train_with_quant.py"

# FP8 の Tensor Core 演算が要求する、活性化テンソルの trailing dimension の倍数。
_FLOAT8_BATCH_MULTIPLE = 16


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """量子化候補を選ぶ3つのオプションを、train_pi05.py の引数に足す。

    Parameters
    ----------
    parser : argparse.ArgumentParser
        train_pi05.py の `_parse_args()` が組み立てているパーサー。
    """
    parser.add_argument("--optimizer-mode", type=str, default="default",
                        choices=("default", "bnb8bit"),
                        help="量子化実験の候補 C1。default: 変更なし（pi05 の既定 AdamW "
                             "プリセットのまま）。bnb8bit: bitsandbytes の 8-bit AdamW に"
                             "差し替える（quant_optim.py。要 bitsandbytes）。"
                             "work/quant_sweep/CANDIDATES.md 参照")
    parser.add_argument("--vision-quant-mode", type=str, default="default",
                        choices=("default", "bnb4bit"),
                        help="量子化実験の候補 C2。bnb4bit: 凍結中の vision encoder "
                             "（SigLIP）だけを bnb.nn.Linear4bit に置換する"
                             "（quant_vision_4bit.py）。実験的 — 動くかどうかも含めて試す候補")
    parser.add_argument("--expert-quant-mode", type=str, default="default",
                        choices=("default", "float8"),
                        help="量子化実験の候補 C3。float8: 学習対象（action expert）の "
                             "Linear だけを torchao の float8 training に変換する"
                             "（quant_float8_expert.py）。要 compute capability >= 8.9"
                             "（この GPU は H100, 9.0 なので満たす）。実験的候補")


def selected(args: argparse.Namespace) -> bool:
    """量子化候補のどれかが選ばれているか。"""
    return (
        args.optimizer_mode != "default"
        or args.vision_quant_mode != "default"
        or args.expert_quant_mode != "default"
    )


def validate(args: argparse.Namespace) -> None:
    """選んだ候補が要求する前提を、学習を始める前に確かめる。

    **データセット読み込み・モデル構築（数分）を待たずに気づけるようにするためにある。**
    float8 の制約は実行時に Linear ごとの形として現れるので、そこまで進んでから
    落ちると、何が悪かったのかが分かりにくい。

    Raises
    ------
    SystemExit
        前提を満たしていないとき。何をどう直すかを本文に書いて止める。
    """
    if (args.expert_quant_mode == "float8"
            and args.batch_size % _FLOAT8_BATCH_MULTIPLE != 0):
        # FP8 の Tensor Core 演算は、実行時の活性化テンソルの trailing dimension が
        # 16 の倍数であることを要求する。これは Linear ごとに違う形で現れる——
        # action 系の投影は batch_size × chunk_size(50) が、time_mlp 系は
        # batch_size 単体がその trailing dimension になる（2026-08-20、実機で
        # batch_size=2 なら "...shape: (32x100)"、batch_size=8 なら
        # "...shape: (3072x8)" という異なる形の RuntimeError を確認済み——
        # 後者は 100=2×50 は 16 の倍数だが、8 自体は 16 の倍数でないために失敗した）。
        # **batch_size 自体が 16 の倍数であれば、両方の形を同時に満たす**
        # （batch_size×50 も自動的に 16 の倍数になる）。詳細は
        # quant_float8_expert.py の docstring。
        raise SystemExit(
            f"[train_pi05] --expert-quant-mode float8 は --batch-size が "
            f"{_FLOAT8_BATCH_MULTIPLE} の倍数であることを要求する"
            "（FP8 Tensor Core 演算の trailing dimension 制約）。"
            f"batch_size={args.batch_size} は条件を満たさない。"
            "詳しくは solution/train/experiments/quantization/quant_float8_expert.py の"
            "「バッチサイズの制約」参照"
        )


def set_environment(args: argparse.Namespace) -> None:
    """量子化候補モジュールのうちどれを import するかを、子プロセスへ環境変数で伝える。

    lerobot_train_with_quant.py がこれを読んで対応するモジュールを import する。
    ここでは os.environ を書き換えるだけで、train_pi05.py の `_run_and_log` 側の
    subprocess.Popen は env= を渡さず親の環境をそのまま継承するので、これだけで
    子プロセスに伝わる。
    """
    if args.optimizer_mode == "bnb8bit":
        os.environ["QUANT_OPTIMIZER_MODE"] = "bnb8bit"
    if args.vision_quant_mode == "bnb4bit":
        os.environ["QUANT_VISION_MODE"] = "bnb4bit"
    if args.expert_quant_mode == "float8":
        os.environ["QUANT_EXPERT_MODE"] = "float8"


def command_head(args: argparse.Namespace) -> list[str]:
    """学習コマンドの先頭（何を実行するか）を返す。

    Returns
    -------
    head : list[str]
        候補を1つも選んでいなければ `["lerobot-train"]`（今までと同じ console
        script）。選んでいれば、候補を import してから同じ main() を呼ぶ
        ラッパーの起動コマンド。
    """
    if selected(args):
        return ["python", str(_WRAPPER)]
    return ["lerobot-train"]


def extra_arguments(args: argparse.Namespace) -> list[str]:
    """選んだ候補が要求する、lerobot-train への追加引数を返す。

    Returns
    -------
    arguments : list[str]
        追加が要らなければ空のリスト。
    """
    if args.optimizer_mode != "bnb8bit":
        return []
    # pi05 は既定で use_policy_training_preset=true になっており、
    # PI05Config.get_optimizer_preset()/get_scheduler_preset() が AdamWConfig と
    # そのスケジュールを決め打ちで返す（lerobot/configs/train.py の validate() で
    # preset を使わないなら optimizer/scheduler を明示しろ、と要求される）。
    # そのため adamw_bnb8bit を登録するだけでは効かず、preset を明示的に切って
    # 上書きする必要がある。値は quant_optim.py の docstring 参照。
    return [
        "--use_policy_training_preset=false",
        "--optimizer.type=adamw_bnb8bit",
        f"--optimizer.lr={args.learning_rate}",
        "--scheduler.type=cosine_decay_with_warmup",
        f"--scheduler.peak_lr={args.learning_rate}",
        "--scheduler.decay_lr=2.5e-6",
        "--scheduler.num_warmup_steps=1000",
        "--scheduler.num_decay_steps=30000",
    ]


def describe(args: argparse.Namespace) -> str:
    """学習開始時のログに出す、選んだ候補の1行。"""
    return (f"optimizer_mode={args.optimizer_mode}、"
            f"vision_quant_mode={args.vision_quant_mode}、"
            f"expert_quant_mode={args.expert_quant_mode}")
