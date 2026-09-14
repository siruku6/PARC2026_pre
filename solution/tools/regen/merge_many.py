#!/usr/bin/env python3
"""LeRobot データセットを何個でも 1 回で繋ぐ（`npz_to_lerobot.py merge` の多数版）。

**なぜ別にしたか。**`merge` は 2 つずつしか繋げない。足す部品が 19 個
（撃ち直しの塊 17・台本・道 A）あると 19 回繋ぐことになり、そのたびに
それまでの動画を全部写し直す。`merge_datasets` は最初から一覧を受け取るので、1 回で済む。

**先に `npz_to_lerobot.py align --base <base> --add <部品>` で列の書き方を揃えること。**
揃っていないと `merge_datasets` が「Same features is expected」で止まる。

用法（コンテナの中で打つ）:
    python solution/tools/regen/merge_many.py --base work/regen/assembly/bowl5_keep \\
        --add work/regen/lerobot_init_pose_v2/c01 --add work/regen/lerobot_scripted_v2 \\
        --output work/artifacts/libero_plus_bowl5_initpose \\
        --output-repo-id local/libero_plus_bowl5_initpose

Reads:
  - <base>・<add>…（LeRobot v3 のデータセット）
Generates:
  - <output>/ … base の後ろに add を順に繋いだデータセット（課題は指示文の文字列で束ねる）
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, metavar="DIR")
    parser.add_argument("--base-repo-id", default="local/base")
    parser.add_argument("--add", type=Path, action="append", required=True, metavar="DIR")
    parser.add_argument("--output", type=Path, required=True, metavar="DIR")
    parser.add_argument("--output-repo-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.output.exists():
        raise SystemExit(f"出力先が既にある: {args.output}（消してから打つ）")
    datasets = [LeRobotDataset(args.base_repo_id, root=args.base)]
    datasets += [LeRobotDataset(f"local/add_{number:02d}", root=path)
                 for number, path in enumerate(args.add)]
    for path, dataset in zip([args.base, *args.add], datasets):
        print(f"  {dataset.meta.total_episodes:>6} 本 {dataset.meta.total_frames:>9} コマ  {path}")
    try:
        merged = merge_datasets(datasets, output_repo_id=args.output_repo_id,
                                output_dir=args.output)
    except ValueError as error:
        if args.output.exists():
            shutil.rmtree(args.output)
        raise SystemExit(f"{error}\n**先に npz_to_lerobot.py align で列の書き方を揃えること**") from error
    print(f"合計 {merged.meta.total_episodes} 本 {merged.meta.total_frames} コマ "
          f"{merged.meta.total_tasks} 課題 → {args.output}")


if __name__ == "__main__":
    main()
