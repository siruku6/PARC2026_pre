#!/usr/bin/env python3
"""疑似検証スイート1回の結果を、**カテゴリ別のステップ数**で読み直す。

`summarize_gpu_eval.py` はカテゴリ別の成功率しか出さない。しかし
[07_track2_preparation.md §3.1](../../docs/plan/07_track2_preparation.md)が
指摘するとおり、**手元の成功率は本番の順位を付けられない**
（knowledge/07_local_eval_cannot_rank.md）。本番と繋がっている量は
ステップ数、とくに「基準（Base）で完走していた件だけのステップ数」である
（knowledge/06_step_count_drives_score.md、r=-0.868）。

`compare_steps.py` はこの考え方を2つの回（学習前後など）の比較に実装している。
本スクリプトは同じ考え方を、**1回のスイート内でBase行と摂動行を突き合わせる**
形に適用する。`TRACK2_PROBE.csv`/`make_track2_probe.py`が作る「元タスク1つに
Base+4摂動の5行」という並びを前提に、5行ずつのまとまりごとにBase行を基準として
各摂動行と対にする。

**成功率（n=カテゴリごとの本数）は「壊れていないことの確認」にしか使わない。**
採否の判断にはステップ数比・打ち切り率を使うこと。

Reads:
  - <run-dir>/eval.log の `[結果]` 行
  - <run-dir>/VAL_TASKS.csv（5行区切りでBase+4摂動という前提）
Generates:
  - 標準出力のみ（ファイルは書かない）
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

# compare_steps.py と同じ値（採点環境の1エピソードの上限）。
_MAX_STEPS = 300

_RESULT_RE = re.compile(r"(\S+) ep0: (\S+) (\d+) steps")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run_docker_eval.sh の出力先")
    parser.add_argument(
        "--block-size", type=int, default=5,
        help="1タスクぶんの行数（Base+摂動数。既定5=Base+4カテゴリ）",
    )
    return parser.parse_args()


def _read_results(run_dir: Path) -> dict[str, tuple[str, int]]:
    """`eval.log` の `[結果]` 行から、task_id → (見出し, ステップ数) を作る。"""
    text = (run_dir / "eval.log").read_text(errors="replace")
    results: dict[str, tuple[str, int]] = {}
    for line in text.splitlines():
        if "[結果]" not in line:
            continue
        match = _RESULT_RE.match(line.split("[結果]", 1)[1].strip())
        if match:
            results[match.group(1)] = (match.group(2), int(match.group(3)))
    return results


def _load_blocks(run_dir: Path, block_size: int) -> list[list[dict]]:
    with open(run_dir / "VAL_TASKS.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) % block_size != 0:
        raise ValueError(
            f"行数 {len(rows)} が block-size {block_size} の倍数でない。"
            "Base+摂動の並びが崩れている可能性がある"
        )
    return [rows[i:i + block_size] for i in range(0, len(rows), block_size)]


def main() -> None:
    args = _parse_args()
    results = _read_results(args.run_dir)
    blocks = _load_blocks(args.run_dir, args.block_size)

    by_category: dict[str, list[tuple[int, str, int, str]]] = {}
    for block in blocks:
        base_rows = [r for r in block if r["category"] == "Base"]
        if len(base_rows) != 1:
            raise ValueError(f"1ブロックにBase行が{len(base_rows)}件ある: {block}")
        base_id = base_rows[0]["task_id"]
        base_outcome, base_steps = results[base_id]
        for row in block:
            if row["category"] == "Base":
                continue
            other_outcome, other_steps = results[row["task_id"]]
            by_category.setdefault(row["category"], []).append(
                (base_steps, base_outcome, other_steps, other_outcome)
            )

    print(f"{'カテゴリ':22} {'成功':>7} {'打ち切り':>9} {'中央値':>7} {'最大':>5}"
          f"   基準完走件のステップ比（採否の判断に使う量）")
    print("-" * 100)
    for category, items in sorted(by_category.items()):
        n = len(items)
        steps = [o_steps for _, _, o_steps, _ in items]
        succ_n = sum(1 for *_, o_outcome in items if o_outcome == "成功")
        timeout_n = sum(1 for s in steps if s >= _MAX_STEPS)

        # **基準（Base）が打ち切りに達していた組は外す。** 救われると大きく減って
        # 相殺されるため（compare_steps.py の _print_totals と同じ理由）。
        finished = [(b, o) for b, _, o, _ in items if b < _MAX_STEPS]
        base_total = sum(b for b, _ in finished)
        other_total = sum(o for _, o in finished)
        if base_total:
            ratio = f"{(other_total / base_total - 1) * 100:+.1f}% (n={len(finished)}/{n})"
        else:
            ratio = "—"

        print(f"{category:22} {succ_n:3}/{n:<3} {timeout_n:5}/{n:<3}"
              f" {statistics.median(steps):7.0f} {max(steps):5}   {ratio}")

    print()
    print("**成功率は「壊れていないことの確認」にだけ使う。採否の判断は上のステップ比で行う**")
    print("（07_track2_preparation.md §3.1、knowledge/06_step_count_drives_score.md）。")


if __name__ == "__main__":
    main()
