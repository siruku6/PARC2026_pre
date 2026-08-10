#!/usr/bin/env python3
"""SmolVLA の LoRA 追加学習を、Colab を使わずに回す。

配布ノートブック（examples/smolvla_libero_spatial_lora.ipynb）と同じ学習を、
[docs/solution/COLAB.md](../../docs/solution/COLAB.md) の H の変更を適用した状態で実行する。

| | 変更 | 配布既定 |
|---|---|---|
| (a) | 全 40 タスク（4 スイート）を使う | libero_spatial の 10 タスクのみ |
| (b) | 1 タスク内は明るさで散らして選ぶ | エピソード番号の等間隔 |
| (c) | 画像拡張を有効にする | 無効 |

Colab と違い、出力はホストのディレクトリに残るため Drive 同期は要らない。
中断しても `--resume` で続きから始められる。

    python train.py                      # 最初から
    python train.py --resume             # 直近のチェックポイントから
    python train.py --steps 6000         # 短く回す

Reads:
  - lerobot/smolvla_libero_plus（HF、初期重み）
  - lerobot/libero_plus（HF、学習データ）
Generates:
  - $OUTPUT_ROOT/<run-name>/checkpoints/**（チェックポイント）
  - $OUTPUT_ROOT/<run-name>/episodes.json（選んだエピソードの記録）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

_BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
_BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"
_VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
_DATASET_REPO = "lerobot/libero_plus"
_DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"

_LEROBOT_DIR = Path(os.environ.get("LEROBOT_DIR", "/opt/lerobot"))
_BRIGHTNESS_COLUMN = "stats/observation.images.front/mean"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("OUTPUT_ROOT", "/out")),
                        help="出力の置き場（既定: $OUTPUT_ROOT または /out）")
    parser.add_argument("--run-name", type=str, default="smolvla_lora_all_suites",
                        help="実行の名前。再開時は同じ名前を指定する")
    parser.add_argument("--steps", type=int, default=20000,
                        help="学習ステップ数。学習率はこの値をかけて減衰する")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--episodes-per-task", type=int, default=5,
                        help="1 タスクあたりのエピソード数。5 を超えると転送量が跳ねる")
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="データ供給のプロセス数。動画デコードが律速なら増やす")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-image-transforms", action="store_true",
                        help="画像拡張を無効にする（既定は有効）")
    parser.add_argument("--video-backend", type=str, default="torchcodec",
                        choices=("torchcodec", "pyav"),
                        help="動画のデコーダ。torchcodec が torch と対で壊れたときの逃げ道")
    parser.add_argument("--resume", action="store_true",
                        help="直近のチェックポイントから再開する")
    return parser.parse_args()


def _patch_dtype_for_gpu() -> None:
    """bf16 非対応の GPU では VLM の読み込みを fp16 に書き換える。

    配布ノートブックが同じ置換をしている。揃えないと、学習を回した GPU に
    よってモデルの挙動が変わってしまう。Ampere 以降なら何もしない。

    `torch.cuda.is_bf16_supported()` が GPU の無い環境で何を返すかは torch の
    版によって違う（内部で `current_device()` を呼ぶ実装では例外になる）。
    版に依存しないよう、先に `is_available()` で切る。
    """
    if not torch.cuda.is_available():
        print("[train] GPU が無いため dtype の書き換えはしない")
        return

    if torch.cuda.is_bf16_supported():
        print("[train] bf16 対応の GPU。書き換えは不要")
        return

    target = _LEROBOT_DIR / "src/lerobot/policies/smolvla/smolvlm_with_expert.py"
    source = target.read_text(encoding="utf-8")
    if 'torch_dtype="bfloat16",' not in source:
        print("[train] 既に fp16 へ書き換え済み")
        return

    target.write_text(
        source.replace('torch_dtype="bfloat16",', 'torch_dtype="float16",', 1),
        encoding="utf-8",
    )
    print("[train] bf16 非対応の GPU のため fp16 へ書き換えた")


def _flatten_numbers(value: Any) -> list[float]:
    """入れ子になった配列を、平坦な float のリストにする。

    parquet の統計値は array([array([array([0.35])])]) のように object dtype で
    三重に入れ子になっており、np.mean がスカラーを返さない。再帰的にほどく。
    """
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    return [number for item in value for number in _flatten_numbers(item)]


def _load_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    """メタデータの parquet を読み、タスク名と明るさの列を足して返す。

    LeRobot の `load_episodes()` は `stats/` で始まる列を落とすため、
    `LeRobotDatasetMetadata` 経由では明るさが取れない。parquet を直接読む。
    """
    files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"メタデータが見つかりません: {dataset_root}")

    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["task"] = frame["tasks"].apply(
        lambda cell: cell if isinstance(cell, str) else str(cell[0])
    )
    frame["brightness"] = [
        float(np.mean(_flatten_numbers(cell))) for cell in frame[_BRIGHTNESS_COLUMN]
    ]
    return frame


def _choose_spread_by_value(
    episode_indices: list[int], count: int, value_by_episode: dict[int, float]
) -> list[int]:
    """値で並べ替えてから、両端を含めて等間隔に count 本選ぶ。"""
    ordered = sorted(episode_indices, key=lambda index: value_by_episode[index])

    if count >= len(ordered):
        return list(ordered)
    if count <= 1:
        return [ordered[len(ordered) // 2]]

    positions = [round(rank * (len(ordered) - 1) / (count - 1)) for rank in range(count)]
    return [ordered[position] for position in positions]


def _select_episodes(dataset_root: Path, episodes_per_task: int) -> list[int]:
    """全 40 タスクから、明るさで散らしてエピソードを選ぶ。

    Parameters
    ----------
    dataset_root : Path
        メタデータを取得済みのデータセットのルート。
    episodes_per_task : int
        1 タスクあたりの本数。

    Returns
    -------
    episode_indices : list[int]
        選んだエピソード番号の昇順リスト。
    """
    frame = _load_episode_metadata(dataset_root)
    brightness = dict(zip(frame["episode_index"].astype(int), frame["brightness"]))

    selected: list[int] = []
    for task_name, group in frame.groupby("task"):
        indices = group["episode_index"].astype(int).tolist()
        selected.extend(_choose_spread_by_value(indices, episodes_per_task, brightness))

    selected = sorted(selected)
    chosen = [brightness[index] for index in selected]
    whole = list(brightness.values())

    print(f"[train] {frame['task'].nunique()} タスク × {episodes_per_task} 本 "
          f"= {len(selected)} エピソード")
    print(f"[train]   選んだ分の明るさ {min(chosen):.3f} 〜 {max(chosen):.3f}")
    print(f"[train]   データセット全体 {min(whole):.3f} 〜 {max(whole):.3f}")
    return selected


def _find_resume_checkpoint(output_dir: Path) -> Path | None:
    """再開に使う最新のチェックポイントを返す。無ければ None。

    ディレクトリ名はゼロ埋めされるが桁数は steps に依存するため、
    文字列ではなく数値として並べる。
    """
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return None

    checkpoints = sorted(
        (path for path in root.glob("*") if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    return checkpoints[-1] if checkpoints else None


def _build_resume_command(
    args: argparse.Namespace, output_dir: Path, resume_from: Path
) -> list[str]:
    """再開用の引数を組み立てる。**最小限しか渡さない。**

    再開時は学習条件がチェックポイントの `train_config.json` から復元される。
    ここで `--policy.*` や `--dataset.*` を重ねて渡すと、元の条件を上書きして
    しまう。特に `--policy.path` は `_resolve_pretrained_from_cli()` が
    優先するため、渡すと再開そのものが黙って無視される。

    例外は `--output_dir` である。チェックポイントの config には**元の環境の
    パス**（Colab なら `/content/outputs/...`）が入っているため、明示的に
    上書きしないと存在しない場所へ書こうとする。

    バッチサイズとワーカー数は学習条件を壊さないので渡してよい。
    バッチサイズを変えるとデータ順の再現性は落ちるが、LeRobot は警告を出すだけで
    エポックとオフセットは正しく復元する。

    `--steps` は渡さない。学習率は `scheduler_decay_steps` で決まっており、
    ここを変えるとスケジュールが途中で別物になる。
    """
    return [
        "lerobot-train",
        "--resume=true",
        f"--config_path={resume_from / 'pretrained_model' / 'train_config.json'}",
        f"--output_dir={output_dir}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--persistent_workers={'true' if args.num_workers > 0 else 'false'}",
        f"--save_freq={args.save_freq}",
    ]


def _build_command(
    args: argparse.Namespace,
    base_model: Path,
    episode_indices: list[int],
    output_dir: Path,
) -> list[str]:
    """最初から学習するときの引数を組み立てる。"""
    command = [
        "lerobot-train",
        f"--policy.path={base_model}",
        f"--policy.vlm_model_name={_VLM_REPO}",
        "--policy.push_to_hub=false",
        "--policy.repo_id=null",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        f"--policy.optimizer_lr={args.learning_rate}",
        f"--policy.scheduler_decay_lr={args.final_learning_rate}",
        f"--policy.scheduler_warmup_steps={args.warmup_steps}",
        f"--policy.scheduler_decay_steps={args.steps}",
        f"--dataset.repo_id={_DATASET_REPO}",
        f"--dataset.revision={_DATASET_REVISION}",
        f"--dataset.episodes=[{','.join(map(str, episode_indices))}]",
        "--dataset.use_imagenet_stats=false",
        f"--dataset.video_backend={args.video_backend}",
        f"--output_dir={output_dir}",
        f"--job_name={args.run_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--persistent_workers={'true' if args.num_workers > 0 else 'false'}",
        "--env_eval_freq=0",
        "--eval_steps=0",
        f"--seed={args.seed}",
        "--save_checkpoint=true",
        f"--save_freq={args.save_freq}",
        "--save_checkpoint_to_hub=false",
        "--log_freq=100",
        "--wandb.enable=false",
        "--peft.method_type=LORA",
        f"--peft.r={args.lora_r}",
        f"--peft.lora_alpha={args.lora_alpha}",
    ]

    if not args.no_image_transforms:
        command.append("--dataset.image_transforms.enable=true")

    return command


def main() -> None:
    args = _parse_args()
    output_dir: Path = (args.output_root / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'なし'}")
    _patch_dtype_for_gpu()

    resume_from = _find_resume_checkpoint(output_dir) if args.resume else None
    if args.resume and resume_from is None:
        print("[train] 再開できるチェックポイントが無いため、最初から学習する")

    if resume_from is not None:
        # 再開時は学習条件をチェックポイントから復元する。エピソードの選び直しも
        # 初期重みの取得も要らない。ステップ数は training_state から読まれる。
        print(f"[train] 再開: {resume_from.name}（step は training_state から復元される）")
        command = _build_resume_command(args, output_dir, resume_from)
        print(f"[train] 学習を再開（batch {args.batch_size}）")
    else:
        print("[train] 初期重みを取得")
        base_model = Path(snapshot_download(
            _BASE_MODEL_REPO, revision=_BASE_MODEL_REVISION,
            ignore_patterns=["eval/*", "eval/**"],
        ))

        print("[train] データセットのメタデータを取得")
        dataset_root = Path(snapshot_download(
            _DATASET_REPO, revision=_DATASET_REVISION, repo_type="dataset",
            allow_patterns=["meta/**"],
        ))

        episode_indices = _select_episodes(dataset_root, args.episodes_per_task)
        (output_dir / "episodes.json").write_text(
            json.dumps({"episodes": episode_indices, "per_task": args.episodes_per_task}),
            encoding="utf-8",
        )

        command = _build_command(args, base_model, episode_indices, output_dir)
        print(f"[train] 学習を開始（{args.steps} ステップ、batch {args.batch_size}）")

    process = subprocess.run(command)
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
