#!/usr/bin/env python3
"""学習が吐いた π0.5 のチェックポイントを、**評価と提出に使える形**へ整える。

`policy_server.py` は重みを `model_weights/` から読み、**トークナイザを
`model_weights/vlm/` から読む**（版 3.8.0。SmolVLA と共通の置き場である）。
学習が吐くチェックポイントには `vlm/` が入っていないので、ここで足す。

π0.5 が外から要るのは**トークナイザだけ**である。PaliGemma 本体は設定から
組み立てられる（`lerobot v0.6.0` の `PaliGemmaWithExpertModel.__init__` が
`CONFIG_MAPPING["paligemma"]()` で作る）ので、**重みを Hub から取りに行かない。**
採点環境は外部通信が遮断されるため、この 1 点が起動の可否を分ける。

**`model.safetensors` はハードリンクで置く。** 8.7 GB を複製すると
チェックポイントの間引き（`prune_checkpoints.sh`）と競って disk を食う。
リンクなら追加の容量は 0 で、しかも**元のチェックポイントが間引かれても
中身は残る**（リンク数が 2 になるため）。

    # 学習の最終チェックポイントを取り込む
    python3 solution/tools/weights/install_pi05_weights.py \\
        work/train/pi05_full_local/checkpoints/024000/pretrained_model

    # 名前を明示する（既定は model_weights_pi05_<ステップ数>）
    python3 solution/tools/weights/install_pi05_weights.py <ckpt> --name pi05_24k

出来たディレクトリはそのまま評価に渡せる。

    WEIGHTS=$PWD/work/artifacts/model_weights_pi05_024000 LABEL=pi05_24k \\
    POLICY_IMAGE=parc2026-train-pi05 \\
        bash solution/tools/run_docker_eval.sh --check

Reads:
  - <checkpoint>/config.json, model.safetensors, policy_pre/postprocessor*
  - $HF_HOME/hub/models--google--paligemma-3b-pt-224/snapshots/*（トークナイザ）
Generates:
  - work/artifacts/<name>/（`model.safetensors` はハードリンク、`vlm/` は複製）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# solution/tools/weights/ の 3 階層上がリポジトリルート。
_REPO_ROOT = Path(__file__).resolve().parents[3]

# チェックポイントに必ずあるべきもの。1 つでも欠けたら取り込まない。
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)

# トークナイザの出どころ。**チェックポイントの `policy_preprocessor.json` に
# この repo 名が焼き込まれている**ので、同じものを手元から写す。
_TOKENIZER_CACHE_GLOB = "hub/models--google--paligemma-3b-pt-224/snapshots/*"

# `AutoTokenizer.from_pretrained` がローカルの置き場から読むのに要るもの。
# `tokenizer.json` が 17 MB で、残りは数十 KB である。
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path,
                        help="学習が吐いた pretrained_model ディレクトリ")
    parser.add_argument("--name", type=str, default=None,
                        help="work/artifacts/ の下に作る名前"
                             "（既定は model_weights_pi05_<ステップ数>）")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="トークナイザの置き場（既定は $HF_HOME から探す）")
    parser.add_argument("--force", action="store_true",
                        help="出力先が既にあっても作り直す")
    return parser.parse_args()


def _default_name(checkpoint: Path) -> str:
    """`.../checkpoints/024000/pretrained_model` から名前を組み立てる。"""
    step = checkpoint.parent.name if checkpoint.name == "pretrained_model" \
        else checkpoint.name
    return f"model_weights_pi05_{step}"


def _check_checkpoint(checkpoint: Path) -> None:
    """必要なファイルが揃っていて、中身が π0.5 であることを確かめる。

    **型を見るのは取り違えを防ぐためである。** SmolVLA のチェックポイントを
    渡しても `policy_server.py` は動いてしまい、π0.5 を測ったつもりで
    別の模型を測ることになる。

    Reads:
      - <checkpoint>/config.json
    """
    missing = [name for name in _REQUIRED_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise SystemExit(f"エラー: {checkpoint} に {' と '.join(missing)} が無い\n"
                         "  渡すのは checkpoints/<ステップ数>/pretrained_model である")

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "pi05":
        raise SystemExit(f"エラー: これは π0.5 ではない: type={config.get('type')!r}")


def _find_tokenizer(requested: Path | None) -> Path:
    """トークナイザの置き場を返す。

    Parameters
    ----------
    requested : Path | None
        `--tokenizer` の値。`None` なら `$HF_HOME` の中を探す。

    Returns
    -------
    tokenizer : Path
        `tokenizer.json` があるディレクトリ。
    """
    if requested is not None:
        if not (requested / "tokenizer.json").is_file():
            raise SystemExit(f"エラー: {requested} に tokenizer.json が無い")
        return requested

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    for snapshot in sorted(hf_home.glob(_TOKENIZER_CACHE_GLOB)):
        if (snapshot / "tokenizer.json").is_file():
            return snapshot

    raise SystemExit(
        "エラー: paligemma のトークナイザが見つからない\n"
        f"  探した場所: {hf_home / _TOKENIZER_CACHE_GLOB}\n"
        "  取り寄せる: hf download google/paligemma-3b-pt-224 "
        "--include 'tokenizer*' 'special_tokens_map.json' 'added_tokens.json'"
    )


def _link_or_copy(source: Path, destination: Path) -> str:
    """まずハードリンクを試し、駄目なら複製する。

    別のファイルシステムに跨ると `link` は `OSError` になる。そのときだけ
    複製に落とす（8.7 GB のぶんだけ disk を食う）。

    Returns
    -------
    how : str
        `"リンク"` または `"複製"`。
    """
    try:
        os.link(source, destination)
        return "リンク"
    except OSError:
        shutil.copy2(source, destination)
        return "複製"


def _install_tokenizer(tokenizer: Path, destination: Path) -> list[str]:
    """`vlm/` を作り、トークナイザのファイルを写す。

    Returns
    -------
    copied : list[str]
        写したファイル名。`added_tokens.json` は無い版もあるので、
        **あるものだけ**写す。
    """
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in _TOKENIZER_FILES:
        source = tokenizer / name
        if source.is_file():
            shutil.copy2(source, destination / name)
            copied.append(name)
    if "tokenizer.json" not in copied:
        raise SystemExit(f"エラー: {tokenizer} に tokenizer.json が無い")
    return copied


def main() -> None:
    args = _parse_args()
    checkpoint: Path = args.checkpoint.expanduser().resolve()
    _check_checkpoint(checkpoint)

    tokenizer = _find_tokenizer(args.tokenizer.expanduser().resolve()
                                if args.tokenizer else None)
    destination = _REPO_ROOT / "work" / "artifacts" / (args.name
                                                       or _default_name(checkpoint))
    if destination.exists():
        if not args.force:
            raise SystemExit(f"エラー: 出力先が既にある: {destination}\n"
                             "  作り直すなら --force")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    print(f"[install] 取り込み元 {checkpoint}")
    print(f"[install] 出力先     {destination}")

    for item in sorted(checkpoint.iterdir()):
        if item.is_dir():
            continue
        if item.name == "model.safetensors":
            how = _link_or_copy(item, destination / item.name)
            size_gb = item.stat().st_size / 2**30
            print(f"[install]   {item.name}  {size_gb:.2f} GB（{how}）")
        else:
            shutil.copy2(item, destination / item.name)

    copied = _install_tokenizer(tokenizer, destination / "vlm")
    print(f"[install]   vlm/  {len(copied)} ファイル（{tokenizer}）")

    print()
    print("  次はこれで評価に回す（**GPU が空いてから**）:")
    print(f"    WEIGHTS={destination} LABEL={destination.name.replace('model_weights_', '')} \\")
    print("    POLICY_IMAGE=parc2026-train-pi05 \\")
    print("        bash solution/tools/run_docker_eval.sh --check")


if __name__ == "__main__":
    main()
