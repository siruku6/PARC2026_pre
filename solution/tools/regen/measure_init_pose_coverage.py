#!/usr/bin/env python3
"""摂動下再生のデータが、**初期姿勢のどの範囲を覆えているか**を測る。

**なぜ要るか。**採点で崩れるかどうかを決めるのは、手先のずれの**大きさではなく向き**である
（`docs/reference/21_weakness_map.md` の 2 節——同じ課題で 3.24 cm が 3/3 成功し、
2.97 cm が 0/3）。**大きさだけを層化しても、向きが偏っていれば覆えていない。**
この道具は**大きさ・向き（姿勢のずれ）・ずれの方向**の 3 つを同じ物差しで出す。

**基準は配布データの課題ごとの中央値である。**腕の土台は動かないので、摂動なしの
手先の姿勢は課題の中でほぼ 1 点に集まる（実測：位置の中央 1.00 cm・向きの中央 0.10 度）。

| 出す数 | 何が分かるか |
|---|---|
| **位置のずれ** [cm] | 学習データに無い帯（2 cm 以上）に入っているか |
| **向きのずれ** [度] | **配布データは最大 0.76 度しか振れていない。**ここを覆えているか |
| **ずれの方向の偏り** | 単位ベクトルの平均の長さ（**0 なら偏り無し**）と、軸ごとの正負 |

**課題は指示文で突き合わせる。**寄せ済み（`--normalize-instructions`）のデータなら
全エピソードが配布データの 40 種のどれかに当たる。**当たらない本があれば、
それは評価で一度も引かれない本である**（`docs/knowledge/23_…`）ので、数えて報告する。

用法（手元で打つ。コンテナは要らない）:
    python3 solution/tools/regen/measure_init_pose_coverage.py \\
        --dataset work/regen/lerobot_init_pose/c01 \\
        --dataset work/regen/lerobot_init_pose/c02 \\
        --reference work/artifacts/libero_plus

Reads:
  - <dataset>/data/chunk-*/file-*.parquet・<dataset>/meta/tasks.parquet
  - <reference>/… 同じ形の配布データ
Generates:
  - 標準出力の表（`--json` を渡すとその内容を書き出す）
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "scripted_demo"))

from check_dataset_instructions import dataset_tasks  # noqa: E402
from motion.rotation import relative_axisangle  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, action="append", required=True,
                        metavar="DIR", help="測る側。**複数渡すと合算する**")
    parser.add_argument("--reference", type=Path, required=True, metavar="DIR",
                        help="基準にする配布データ")
    parser.add_argument("--json", type=Path, default=None, metavar="FILE")
    return parser.parse_args()


def _frame0(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """各エピソードの 0 コマ目の `observation.state` と、その課題名を返す。

    **0 コマ目だけを parquet の側で抜く。**全コマを pandas に移してから絞ると、
    配布データ（224 万コマ）で 13 倍遅い。
    """
    names = np.array(dataset_tasks(root))
    states, tasks = [], []
    for path in sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet"))):
        table = pq.read_table(path, columns=["observation.state", "task_index"],
                              filters=[("frame_index", "==", 0)])
        column = table.column("observation.state").combine_chunks()
        states.append(column.flatten().to_numpy().reshape(len(column), -1))
        tasks.append(names[table.column("task_index").to_numpy()])
    return np.concatenate(states).astype(np.float64), np.concatenate(tasks)


def _spread(states: np.ndarray, tasks: np.ndarray, reference: dict
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """基準からの位置 [cm]・向き [度]・方向を出す。**基準の無い課題は数えて飛ばす。**"""
    position, rotation, direction, missing = [], [], [], 0
    for state, task in zip(states, tasks):
        base = reference.get(task)
        if base is None:
            missing += 1
            continue
        offset = state[:3] - base[:3]
        norm = float(np.linalg.norm(offset))
        position.append(norm * 100.0)
        rotation.append(np.degrees(np.linalg.norm(
            relative_axisangle(base[3:6], state[3:6]))))
        direction.append(offset / max(norm, 1e-12))
    return (np.array(position), np.array(rotation), np.array(direction), missing)


def _row(label: str, values: np.ndarray, threshold: float) -> str:
    return (f"| {label} | {len(values)} | {np.median(values):.2f} "
            f"| {np.percentile(values, 90):.2f} | {values.max():.2f} "
            f"| {(values > threshold).mean() * 100:.1f}% |")


def main() -> None:
    args = _parse_args()

    ref_states, ref_tasks = _frame0(args.reference)
    reference = {task: np.median(ref_states[ref_tasks == task], axis=0)
                 for task in np.unique(ref_tasks)}
    ref_pos, ref_rot, _, _ = _spread(ref_states, ref_tasks, reference)

    pieces = [_frame0(root) for root in args.dataset]
    states = np.concatenate([piece[0] for piece in pieces])
    tasks = np.concatenate([piece[1] for piece in pieces])
    pos, rot, direction, missing = _spread(states, tasks, reference)

    print(f"測った側 {len(states)} 本（{len(args.dataset)} 個のデータセット）"
          f"／ 基準 {len(ref_states)} 本・{len(reference)} 課題")
    if missing:
        print(f"**指示文が配布データに無い本が {missing} 本ある**"
              f"（{missing / len(states) * 100:.1f}%）。"
              f"評価で一度も引かれないので、作り直しが要る")

    for title, ref_values, values, threshold in (
            ("位置のずれ [cm]（しきい 2 cm）", ref_pos, pos, 2.0),
            ("向きのずれ [度]（しきい 10 度）", ref_rot, rot, 10.0)):
        print(f"\n### {title}\n")
        print("| | 本数 | 中央 | 90% | 最大 | 超える割合 |")
        print("|---|---|---|---|---|---|")
        print(_row("配布データ", ref_values, threshold))
        print(_row("**測った側**", values, threshold))

    bias = float(np.linalg.norm(direction.mean(axis=0)))
    print(f"\n### ずれの方向\n\n**偏りの大きさ {bias:.3f}**（0 なら偏り無し）\n")
    print("| 軸 | 正 | 負 |")
    print("|---|---|---|")
    for axis, label in enumerate(("左右 x", "前後 y", "上下 z")):
        plus = (direction[:, axis] > 0).mean() * 100
        print(f"| {label} | {plus:.0f}% | {100 - plus:.0f}% |")
    print(f"\n位置と向きの相関 r = {np.corrcoef(pos, rot)[0, 1]:.3f}")

    if args.json:
        args.json.write_text(json.dumps({
            "episodes": int(len(states)), "missing_instruction": int(missing),
            "pos_cm": pos.tolist(), "rot_deg": rot.tolist(),
            "direction_bias": bias,
        }, ensure_ascii=False))
        print(f"\n書いた: {args.json}")


if __name__ == "__main__":
    main()
