#!/usr/bin/env python3
"""繋いだデータセットの後ろの部分が、抜き出す前の元データと同じコマを指しているかを確かめる。

**なぜ要るか。**`rebuild_dataset.py` は動画を作り直さず、メタデータの「動画の何秒目から」だけを
写す。`merge_many.py` はそこへ繋ぎ目のずれを足す。**どちらかで番号やずれを取り違えると、
本数も課題数も合ったまま、別のエピソードの画が付く。**本数の門では見つからない。

**見るもの**（抜き出した本から 3 本 × 2 コマ）
- 正面カメラの画が元と 1 画素も違わない
- `observation.state` が元と一致する
- 指の値は `--expect-gripper-fix` のとき「元の {0,1} を `1 − 2a` で直した値」、なければ元と一致する

用法（コンテナの中で打つ。元データはリポジトリの下にハードリンクで置く——コンテナから見えるのはリポジトリだけ）:
    python solution/tools/regen/verify_appended_frames.py \\
        --merged work/artifacts/libero_combined_bowl5_initpose --offset 16828 \\
        --source work/regen/assembly/combined_link --keep-file work/regen/combined/head_keep.json \\
        --expect-gripper-fix

Reads:
  - <merged>・<source>（LeRobot v3）・<keep-file>（`{"episodes": [元の番号, …]}`。抜き出した順）
Generates:
  - 標準出力。1 つでも違えば終了コード 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merged", type=Path, required=True, metavar="DIR")
    parser.add_argument("--offset", type=int, required=True, metavar="N",
                        help="繋いだ部分の先頭のエピソード番号（= base の本数）")
    parser.add_argument("--source", type=Path, required=True, metavar="DIR")
    parser.add_argument("--keep-file", type=Path, required=True, metavar="JSON")
    parser.add_argument("--expect-gripper-fix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    keep = json.loads(args.keep_file.read_text())["episodes"]
    merged = LeRobotDataset("local/merged", root=args.merged)
    source = LeRobotDataset("local/source", root=args.source)
    bad = 0
    for position in (0, len(keep) // 2, len(keep) - 1):
        m = merged.meta.episodes[args.offset + position]
        o = source.meta.episodes[keep[position]]
        for offset in (0, int(o["length"]) // 2):
            a = merged[int(m["dataset_from_index"]) + offset]
            b = source[int(o["dataset_from_index"]) + offset]
            want = float(b["action"][6])
            if args.expect_gripper_fix and want >= 0:
                want = 1.0 - 2.0 * want
            checks = {"front": torch.equal(a["observation.images.front"], b["observation.images.front"]),
                      "wrist": torch.equal(a["observation.images.wrist"], b["observation.images.wrist"]),
                      "state": torch.equal(a["observation.state"], b["observation.state"]),
                      "arm": torch.equal(a["action"][:6], b["action"][:6]),
                      "gripper": float(a["action"][6]) == want,
                      "task": a["task"] == b["task"]}
            bad += not all(checks.values())
            print(f"  {args.offset + position} ← 元 {keep[position]} の {offset} コマ目: "
                  + " ".join(f"{k}={'OK' if v else 'NG'}" for k, v in checks.items()))
    print("一致" if bad == 0 else f"**{bad} か所で違う**")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
