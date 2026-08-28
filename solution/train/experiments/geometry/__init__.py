"""幾何の接ぎ木（板 F25）を学習へ差し込むための、train_pi05.py 側の窓口。

**このパッケージの外から見える形は `experiments.quantization` と同じである**
——`add_arguments` / `selected` / `validate` / `set_environment` / `command_head` /
`extra_arguments` / `describe` の 7 つ。train_pi05.py はこの窓口を呼ぶだけで、
何をどう接ぐかは一切持たない（中身は vggt_fuser.py）。

**何を学習するか。** Evo-0（arXiv 2507.00416）は π0 を凍結し、
つなぎの層・LoRA・行動側の 3 つだけを学習する。こちらの既定もそれに合わせる
——`--train-expert-only true`（既定）のまま接ぐと、行動側とつなぎの層が動く。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_WRAPPER = Path(__file__).resolve().parent / "lerobot_train_with_geometry.py"

# VGGT の patch は 14 なので、1 辺は 14 の倍数でなければ落ちる。
_PATCH = 14


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """train_pi05.py の引数に、幾何の接ぎ木のつまみを足す。"""
    group = parser.add_argument_group("幾何の接ぎ木（実験・板 F25）")
    group.add_argument(
        "--geometry-fuse",
        choices=("none", "vggt"),
        default="none",
        help="vggt: 凍結した VGGT の幾何トークンを、π0.5 の画像トークンへ交差注意 1 層で"
             "混ぜる（Evo-0 arXiv 2507.00416 の形）。既定の none では今までと同じ",
    )
    group.add_argument(
        "--geometry-model",
        default="facebook/VGGT-1B",
        help="VGGT の重み。既定は門の無い非商用版。提出物に入れる段では商用可の "
             "facebook/VGGT-1B-Commercial に差し替える（板 F26）",
    )
    group.add_argument(
        "--geometry-image-size",
        type=int,
        default=252,
        help="VGGT へ渡す 1 辺。14 の倍数のみ。既定 252 は H100-20C の 20 GB に"
             "収めるための値で、518 だと batch 16 で 8.20 GB 要り上限に触る",
    )


def selected(args: argparse.Namespace) -> bool:
    """幾何の接ぎ木を使うかどうか。"""
    return getattr(args, "geometry_fuse", "none") != "none"


def validate(args: argparse.Namespace) -> None:
    """噛み合わない組み合わせを、学習を始める前に止める。"""
    if not selected(args):
        return
    if args.geometry_image_size % _PATCH != 0:
        raise SystemExit(
            f"--geometry-image-size は {_PATCH} の倍数でなければならない"
            f"（VGGT の patch_embed が assert する）: {args.geometry_image_size}"
        )
    from experiments import quantization

    if quantization.selected(args):
        raise SystemExit(
            "--geometry-fuse と量子化の候補は同時に使えない。"
            "どちらも『候補を import してから lerobot-train の main() を呼ぶ』"
            "ラッパーを学習コマンドの先頭に置くので、先頭は 1 つしか選べない"
        )


def set_environment(args: argparse.Namespace) -> None:
    """どう接ぐかを子プロセスへ環境変数で伝える。

    train_pi05.py の subprocess は env= を渡さず親の環境を継承するので、
    os.environ を書き換えるだけで伝わる（quantization と同じ）。
    """
    if not selected(args):
        return
    os.environ["GEO_FUSE_MODE"] = args.geometry_fuse
    os.environ["GEO_VGGT_MODEL"] = args.geometry_model
    os.environ["GEO_VGGT_IMAGE_SIZE"] = str(args.geometry_image_size)


def command_head(args: argparse.Namespace) -> list[str]:
    """学習コマンドの先頭。接がないなら今までどおり console script を使う。"""
    if selected(args):
        return ["python", str(_WRAPPER)]
    return ["lerobot-train"]


def extra_arguments(args: argparse.Namespace) -> list[str]:  # noqa: ARG001
    """lerobot-train への追加引数。接ぎ木は引数を要求しない。"""
    return []


def describe(args: argparse.Namespace) -> str:
    """ログに 1 行で書くための説明。"""
    if not selected(args):
        return "幾何の接ぎ木なし"
    return (
        f"幾何の接ぎ木 {args.geometry_fuse}"
        f"（{args.geometry_model} / {args.geometry_image_size}x{args.geometry_image_size}）"
    )
