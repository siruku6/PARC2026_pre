#!/usr/bin/env python3
"""初期姿勢の段（L2〜L5）を測った回の結果を、2 通りの軸で読み直す。

**成功時だけの歩数は使わない。** `eval.log` の「平均 N steps」は成功した回だけの
平均なので、落ちた回が除かれて効果が消える。ここが読むのは
`results/object_disp.jsonl` で、**1 エピソード 1 行・成功も失敗も入っている。**
`total_steps`・`success`・`collided` がそのまま入っているので、打ち切りの回も
300 歩として数えられる。

出す軸は 2 つである。

  - **運営の難易度**（`difficulty_level`。CSV の列をそのまま使う）
  - **手先のずれ（cm）**（`check_init_pose_validity.py` が出した JSON を読む）

どちらも**各行を「同じ元タスクの摂動なしの行」と対にして**比べる。
タスクの難しさがそろっていないので、対にしないと軸ではなくタスクを測ってしまう。

歩数比の定義は [reference/21](../../../docs/reference/21_weakness_map.md) の 1 節と
同じにしてある（同じタスクの Base に対する平均歩数の比・300 歩の打ち切りは 300 と
数える）。**8/24 の +47.8% と直接比べるためである。**

使い方:

    python3 solution/tools/probe/summarize_robot_ladder.py \\
        --run work/final_eval/0828_151854_track2_robot_ladder \\
        --validity work/artifacts/robot_initstate_ladder/init_pose_validity.json

Reads:
  - <run>/results/object_disp.jsonl（1 エピソード 1 行）
  - <run>/TASKS_CSV.csv（`category` と `difficulty_level`）
  - --validity に渡した JSON（`eef_shift_cm`。省くと cm の軸を出さない）
Generates:
  - 標準出力のみ
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

_MAX_STEPS = 300
_VARIANT_SEP = "_view_0_0_100_0_0_initstate_"
# 手先のずれで区切る境目（cm）。学習データが覆っているのは中央値 1.0 cm までである。
_SHIFT_BINS = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 99.0)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="評価の実行ディレクトリ")
    p.add_argument("--validity", type=Path, default=None,
                   help="check_init_pose_validity.py が出した JSON")
    return p.parse_args()


def _two_sided_binomial(n_up: int, n: int) -> float:
    """「増えるか減るかが五分五分」を帰無仮説にした両側検定の p 値。"""
    tail = sum(math.comb(n, i) for i in range(n_up, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


class _Episodes:
    """1 行（タスク変種）ぶんのエピソードをまとめて持つ。"""

    def __init__(self, episodes: list[dict]) -> None:
        self.steps = [e["total_steps"] for e in episodes]
        self.success = [bool(e["success"]) for e in episodes]
        self.collided = [bool(e["collided"]) for e in episodes]

    @property
    def mean_steps(self) -> float:
        return statistics.mean(self.steps)

    @property
    def success_rate(self) -> float:
        return sum(self.success) / len(self.success)

    @property
    def collision_rate(self) -> float:
        return sum(self.collided) / len(self.collided)

    @property
    def timeout_rate(self) -> float:
        return sum(s >= _MAX_STEPS for s in self.steps) / len(self.steps)

    @property
    def hit_timeout(self) -> bool:
        return any(s >= _MAX_STEPS for s in self.steps)


def _load(run: Path) -> tuple[dict[str, _Episodes], list[dict]]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    with open(run / "results/object_disp.jsonl") as f:
        for line in f:
            episode = json.loads(line)
            grouped[episode["task"]].append(episode)
    with open(run / "TASKS_CSV.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: _Episodes(v) for k, v in grouped.items()}, rows


def _print_pooled(measured: dict[str, _Episodes], rows: list[dict]) -> None:
    """全エピソードをまとめた表。段ごとに 1 行。"""
    groups: dict[str, list[_Episodes]] = collections.defaultdict(list)
    for row in rows:
        level = row["difficulty_level"] or "Base"
        if row["task_id"] in measured:
            groups[level].append(measured[row["task_id"]])

    print("## 段ごと（全エピソードをまとめた値）\n")
    print("| 段 | 本数 | 成功率 | 衝突率 | 打ち切り率 | 全回込みの平均歩数 |")
    print("|---|---|---|---|---|---|")
    base_steps = None
    for level in ["Base", "L2", "L3", "L4", "L5"]:
        if level not in groups:
            continue
        steps = [s for e in groups[level] for s in e.steps]
        success = [s for e in groups[level] for s in e.success]
        collided = [c for e in groups[level] for c in e.collided]
        mean_steps = statistics.mean(steps)
        if level == "Base":
            base_steps = mean_steps
        ratio = "" if base_steps is None or level == "Base" else \
            f"（Base 比 {100 * (mean_steps / base_steps - 1):+.1f}%）"
        print(f"| {level} | {len(steps)} | {sum(success) / len(success):.3f} | "
              f"{sum(collided) / len(collided):.3f} | "
              f"{sum(s >= _MAX_STEPS for s in steps) / len(steps):.3f} | "
              f"{mean_steps:.1f}{ratio} |")


def _paired(measured: dict[str, _Episodes], rows: list[dict],
            key_of: dict[str, str], order: list[str], title: str) -> None:
    """行を「同じ元タスクの Base」と対にして、区分ごとにまとめる。"""
    base_of = {row["task_id"]: row["task_id"].split(_VARIANT_SEP)[0] for row in rows}
    base_rows = {row["task_id"] for row in rows if row["category"] == "Base"}

    buckets: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        task_id = row["task_id"]
        if task_id in base_rows or task_id not in measured:
            continue
        key = key_of.get(task_id)
        if key is not None:
            buckets[key].append(task_id)

    print(f"\n## {title}（各行を同じ元タスクの Base と対にした値）\n")
    print("| 区分 | 元タスク数 | 歩数比 | 成功率の差 | 衝突率の差 | 壊した | 歩数が増えた | 二項検定 |")
    print("|---|---|---|---|---|---|---|---|")
    for key in order:
        variants = buckets.get(key)
        if not variants:
            continue
        per_task: dict[str, list[str]] = collections.defaultdict(list)
        for task_id in variants:
            per_task[base_of[task_id]].append(task_id)

        ratios, d_success, d_collision = [], [], []
        n_up = n_broke = n_eligible = 0
        for base_task, group in per_task.items():
            base = measured[base_task]
            mean_steps = statistics.mean(measured[t].mean_steps for t in group)
            ratios.append(mean_steps / base.mean_steps)
            n_up += mean_steps > base.mean_steps
            d_success.append(
                statistics.mean(measured[t].success_rate for t in group)
                - base.success_rate)
            d_collision.append(
                statistics.mean(measured[t].collision_rate for t in group)
                - base.collision_rate)
            # 「壊した」は Base が 3 回とも打ち切りに達していないタスクだけで数える。
            if not base.hit_timeout:
                n_eligible += 1
                n_broke += any(measured[t].hit_timeout for t in group)

        n = len(ratios)
        print(f"| {key} | {n} | {100 * (statistics.mean(ratios) - 1):+.1f}% | "
              f"{statistics.mean(d_success):+.3f} | {statistics.mean(d_collision):+.3f} | "
              f"{n_broke}/{n_eligible} | {n_up}/{n} | "
              f"p = {_two_sided_binomial(n_up, n):.4f} |")


def main() -> None:
    args = _parse_args()
    measured, rows = _load(args.run)
    print(f"# {args.run.name}\n")
    print(f"{len(measured)} 行 / "
          f"{sum(len(e.steps) for e in measured.values())} エピソード\n")

    _print_pooled(measured, rows)

    levels = {row["task_id"]: row["difficulty_level"] for row in rows
              if row["difficulty_level"]}
    _paired(measured, rows, levels, ["L2", "L3", "L4", "L5"], "運営の難易度")
    _paired(measured, rows, {t: "摂動 48 行すべて" for t in levels},
            ["摂動 48 行すべて"], "まとめて 1 行")

    if args.validity is None:
        return
    with open(args.validity) as f:
        validity = {r["task_id"]: r for r in json.load(f)}
    shift_of: dict[str, str] = {}
    for task_id in levels:
        shift = validity.get(task_id, {}).get("eef_shift_cm")
        if shift is None:
            continue
        for low, high in _SHIFT_BINS:
            if low <= shift < high:
                shift_of[task_id] = f"{low:.0f}〜{high:.0f} cm"
                break
    order = [f"{low:.0f}〜{high:.0f} cm" for low, high in _SHIFT_BINS]
    _paired(measured, rows, shift_of, order, "手先のずれ")


if __name__ == "__main__":
    main()
