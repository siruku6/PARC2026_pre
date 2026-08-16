#!/usr/bin/env python3
"""LoRA アダプタを初期重みへマージし、提出できる形のモデルを作る。

`train.py` が出すチェックポイントには LoRA アダプタ（`adapter_model.safetensors`）
しか入っていない。**そのままでは提出物として読めない。** 配布ノートブックの
マージセルと同じ処理をここで行う。

出力には **`vlm/`（SmolVLM の設定とトークナイザ）が入らない。** 評価にも提出にも
これが要るので、次は [docs/howto/01_retrain.md](../../docs/howto/01_retrain.md) の
A-6（検算）→ A-7（`vlm/` の足し方が 2 通り）へ進む。

    python merge.py                       # 最新のチェックポイントをマージ
    python merge.py --step 15000          # 特定のステップを指定

途中で打ち切った場合は、学習率がどこまで落ちているかを確認してから選ぶこと
（[docs/solution/COLAB.md](../../docs/solution/COLAB.md) の「途中のチェックポイントで打ち切るとき」）。

**学習の条件も一緒に入れる。** バッチ数と使ったエピソードは重みの `config.json` に
残らないため、これを持ち出さないと**後から悪化の原因を数字で確かめられない**。
実際に 8/11 の 20,000 ステップでは `merged_model/` だけを持ち帰ったせいで、
バッチ数とエピソード本数を一次情報で確かめられなくなった
（[docs/reference/11_training_schedule.md](../../docs/reference/11_training_schedule.md)）。
出発点の重みも `train_config.json` を同梱しており、推論では読まれない。

Reads:
  - $OUTPUT_ROOT/<run-name>/checkpoints/<step>/pretrained_model/**
  - $OUTPUT_ROOT/<run-name>/train_config.json（LeRobot が書く学習条件。`dataset.episodes` も入る）
  - lerobot/smolvla_libero_plus（HF、初期重み）
Generates:
  - $OUTPUT_ROOT/<run-name>/merged_model/**
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import struct
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

_BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
_BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"
_VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# チェックポイントから引き継ぐ前処理・後処理の定義。save_pretrained は出さない。
_PROCESSOR_PATTERNS = (
    "policy_preprocessor.json",
    "policy_preprocessor*.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor*.safetensors",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("OUTPUT_ROOT", "/out")))
    parser.add_argument("--run-name", type=str, default="smolvla_lora_all_suites")
    parser.add_argument("--step", type=int, default=None,
                        help="マージするステップ。省略すると最新を使う")
    return parser.parse_args()


def _find_checkpoint(output_dir: Path, step: int | None) -> Path:
    """マージ対象のチェックポイントを返す。

    Parameters
    ----------
    output_dir : Path
        実行ごとの出力ディレクトリ。
    step : int | None
        指定があればそのステップ、無ければ最新。

    Returns
    -------
    checkpoint : Path
        `<step>/pretrained_model` を指すパス。
    """
    root = output_dir / "checkpoints"
    if not root.is_dir():
        raise FileNotFoundError(f"チェックポイントがありません: {root}")

    candidates = sorted(
        (path for path in root.glob("*") if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not candidates:
        raise FileNotFoundError(f"チェックポイントがありません: {root}")

    if step is None:
        chosen = candidates[-1]
    else:
        matched = [path for path in candidates if int(path.name) == step]
        if not matched:
            available = ", ".join(path.name for path in candidates[-5:])
            raise FileNotFoundError(f"step {step} がありません。直近: {available}")
        chosen = matched[0]

    checkpoint = chosen / "pretrained_model"
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LoRA アダプタがありません: {checkpoint}")
    return checkpoint


def _read_safetensors_keys(path: Path) -> set[str]:
    """safetensors のヘッダだけを読み、テンソル名の集合を返す。"""
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return {name for name in header if name != "__metadata__"}


def _merge(checkpoint: Path, base_model: Path, destination: Path) -> None:
    """LoRA を初期重みへ畳み込み、提出できる形で保存する。"""
    from peft import PeftModel

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    config.pretrained_path = str(base_model)
    config.use_peft = False

    base_policy = SmolVLAPolicy.from_pretrained(base_model, config=config, strict=False)
    peft_policy = PeftModel.from_pretrained(
        base_policy, checkpoint, is_trainable=False, torch_device="cpu"
    )
    merged = peft_policy.merge_and_unload(safe_merge=True)

    # 提出物側は config を書き換えずに読むため、ここで整えておく。
    merged.config.use_peft = False
    merged.config.pretrained_path = None
    merged.config.push_to_hub = False
    merged.config.repo_id = None
    merged.config.device = None
    merged.config.load_vlm_weights = False
    merged.config.vlm_model_name = _VLM_REPO

    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    merged.save_pretrained(destination)

    for pattern in _PROCESSOR_PATTERNS:
        for source in checkpoint.glob(pattern):
            shutil.copy2(source, destination / source.name)

    del peft_policy, base_policy, merged
    gc.collect()


def _carry_conditions(output_dir: Path, destination: Path) -> None:
    """学習の条件を書いたファイルを、マージ結果の隣へ複製する。

    重みの `config.json` に残るのは学習率・warmup・`scheduler_decay_steps`
    までで、**バッチ数と使ったエピソードは残らない。** 学習マシンから
    `merged_model/` だけを持ち帰ると、その 2 つが永久に分からなくなる。

    Parameters
    ----------
    output_dir : Path
        実行ごとの出力ディレクトリ。
    destination : Path
        マージ結果の置き場所。

    """
    # **`train_config.json` だけでよい。** LeRobot はここに `dataset.episodes` も
    # 書くので、隣の `episodes.json` は同じ内容の写しである。
    # 提出物の zip は `model_weights/` を丸ごと取り込むため、
    # **同じ番号の列を 2 度入れない。**
    name = "train_config.json"
    source = output_dir / name
    if not source.is_file():
        print(
            f"[merge] 注意: {name} が見つかりません（{output_dir}）。"
            "**バッチ数と使ったエピソードを後から確かめられなくなります。**"
        )
        return
    shutil.copy2(source, destination / name)
    print(f"[merge] 学習の条件を同梱: {name}")


def main() -> None:
    args = _parse_args()
    output_dir: Path = (args.output_root / args.run_name).resolve()

    checkpoint = _find_checkpoint(output_dir, args.step)
    print(f"[merge] チェックポイント: {checkpoint.parent.name}")

    base_model = Path(snapshot_download(
        _BASE_MODEL_REPO, revision=_BASE_MODEL_REVISION,
        ignore_patterns=["eval/*", "eval/**"],
    ))

    destination = output_dir / "merged_model"
    _merge(checkpoint, base_model, destination)

    keys = _read_safetensors_keys(destination / "model.safetensors")
    remaining = sorted(name for name in keys if "lora_" in name.lower())
    if remaining:
        raise RuntimeError(f"LoRA が残っています: {remaining[:3]} …")

    _carry_conditions(output_dir, destination)

    produced = sorted(path.name for path in destination.iterdir())
    total = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    print(f"[merge] 完了: {destination}  {total / 1e6:.0f} MB")
    print(f"[merge] {produced}")
    # **次はマージの検算である。**`merge.py` 自身の検査は「LoRA が残っていないこと」
    # しか見ないので、ここを飛ばすと初期重みのままでも先へ進めてしまう。
    # ホスト側のパスはコンテナから分からないため、手順書の節番号で案内する。
    print("[merge] 次に: verify_merge.py で検算する（docs/howto/01_retrain.md の A-6）")
    print("[merge] **vlm/ はここに入っていない。** 評価・提出の前に足す（同 A-7）")


if __name__ == "__main__":
    main()
