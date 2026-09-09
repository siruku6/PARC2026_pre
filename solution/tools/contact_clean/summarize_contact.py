#!/usr/bin/env python3
"""接触の判定表を読み、外れる割合と、境目の妥当さを見るための表を出す。

**外す前に見るべきものが 3 つある。**

  1. **物差し 2 つ（1 mm 超えたか／触れたか）で、それぞれ何組・何本が外れるか**
  2. **指示文ごとの内訳。**特定の課題だけが落ちるなら、掃除ではなく課題の問題である
  3. **いちばん動いた量の分布。**0.001 m という境目が妥当かを見る

用法:
    .venv/bin/python solution/tools/contact_clean/summarize_contact.py \\
        --scores work/artifacts/libero_states/contact_scores.json \\
        --baseline 10

Reads:
  - <scores> … score_contact.py が書いた判定表
Generates:
  - 標準出力の表だけ（--keep-out を渡したときは残す組の番号一覧も書く）
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLLISION_THRESHOLD_M = 0.001


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, required=True, metavar="JSON")
    parser.add_argument("--baseline", type=int, default=10, metavar="N",
                        help="ずれの基準にするコマ（既定 10）")
    parser.add_argument("--threshold", type=float, default=COLLISION_THRESHOLD_M,
                        metavar="M", help="外す境目（m・3 軸の絶対値の和）")
    parser.add_argument("--rule", choices=("displacement", "touch", "either"),
                        default="displacement", help="どの物差しで外すか")
    parser.add_argument("--keep-out", type=Path, default=None, metavar="JSON",
                        help="残すエピソード番号の一覧を書く先")
    parser.add_argument("--boost-instruction", action="append", default=None,
                        metavar="TEXT",
                        help="**掃除で痩せる課題**。この指示文の残ったぶんだけを"
                             "別の一覧に書き出す（あとでコピーして本数を戻すため）")
    parser.add_argument("--boost-out", type=Path, default=None, metavar="JSON",
                        help="--boost-instruction の残ったぶんを書く先")
    return parser.parse_args()


def _worst(row: dict, baseline: int) -> tuple[str, float]:
    """その組でいちばん動いた対象外の物体と、その量を返す。"""
    table = row["score"]["displacement_l1_m"].get(str(baseline), {})
    if not table:
        return ("", 0.0)
    name = max(table, key=table.get)
    return (name, table[name])


def _is_dropped(row: dict, baseline: int, threshold: float, rule: str) -> bool:
    """その組を外すか。"""
    over = _worst(row, baseline)[1] > threshold
    touched = bool(row["score"]["touched"])
    return {"displacement": over, "touch": touched, "either": over or touched}[rule]


def _print_rule_table(rows: list[dict], baseline: int, threshold: float) -> None:
    """物差し 3 通りの外れ方を並べる。"""
    episodes_total = sum(len(row["episodes"]) for row in rows)
    print(f"=== 物差しごとの外れ方（基準コマ {baseline}・境目 {threshold * 1000:.1f} mm）===")
    print(f"{'物差し':<26} {'組':>6} {'本':>7} {'割合':>8}")
    for rule, label in (("displacement", "(a) 1 mm を超えて動いた"),
                        ("touch", "(b) 触れた"),
                        ("either", "(a) か (b) のどちらか")):
        dropped = [row for row in rows if _is_dropped(row, baseline, threshold, rule)]
        episodes = sum(len(row["episodes"]) for row in dropped)
        print(f"{label:<26} {len(dropped):>6} {episodes:>7} "
              f"{episodes / episodes_total * 100:>7.1f}%")

    over = {id(row) for row in rows if _is_dropped(row, baseline, threshold, "displacement")}
    touch = {id(row) for row in rows if _is_dropped(row, baseline, threshold, "touch")}
    print(f"\n  両方に当たる組 {len(over & touch)}／(a) だけ {len(over - touch)}"
          f"／(b) だけ {len(touch - over)}")


def _print_per_task(rows: list[dict], baseline: int, threshold: float, rule: str) -> None:
    """指示文ごとの外れる割合を、高い順に並べる。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["instruction"]].append(row)

    lines = []
    for instruction, group in grouped.items():
        dropped = [row for row in group if _is_dropped(row, baseline, threshold, rule)]
        episodes = sum(len(row["episodes"]) for row in group)
        gone = sum(len(row["episodes"]) for row in dropped)
        objects = len(group[0]["score"]["objects"])
        lines.append((gone / episodes, len(dropped), len(group), gone, episodes,
                      objects, instruction))
    lines.sort(reverse=True)

    print(f"\n=== 指示文ごと（外れる割合の高い順）===")
    print(f"{'割合':>7} {'組':>9} {'本':>11} {'対象外':>6}  指示文")
    for share, dropped, total, gone, episodes, objects, instruction in lines:
        print(f"{share * 100:>6.1f}% {dropped:>4}/{total:<4} {gone:>5}/{episodes:<5} "
              f"{objects:>6}  {instruction[:58]}")


def _print_distribution(rows: list[dict], baseline: int) -> None:
    """いちばん動いた量の分布。**境目 1 mm が崖の上か谷底かを見る。**"""
    values = np.array([_worst(row, baseline)[1] for row in rows])
    print(f"\n=== いちばん動いた量の分布（基準コマ {baseline}・mm）===")
    quantiles = [50, 75, 85, 90, 95, 99, 100]
    printed = "  ".join(f"{q}%tile {np.percentile(values, q) * 1000:.2f}"
                        for q in quantiles)
    print(f"  {printed}")
    print(f"{'境目 (mm)':>10} {'外れる組':>9} {'外れる本':>9}")
    episodes_total = sum(len(row["episodes"]) for row in rows)
    for threshold_mm in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        dropped = [row for row in rows
                   if _worst(row, baseline)[1] > threshold_mm / 1000.0]
        episodes = sum(len(row["episodes"]) for row in dropped)
        print(f"{threshold_mm:>10.1f} {len(dropped):>9} {episodes:>6} "
              f"({episodes / episodes_total * 100:.1f}%)")


def _print_worst_objects(rows: list[dict], baseline: int, threshold: float) -> None:
    """外れる組で、犯人になっている物体を数える。"""
    counter: dict[str, int] = defaultdict(int)
    for row in rows:
        name, value = _worst(row, baseline)
        if value > threshold:
            counter[name] += 1
    print(f"\n=== 動かされている物体（多い順に 12 件）===")
    for name, count in sorted(counter.items(), key=lambda item: -item[1])[:12]:
        print(f"  {count:>4} 組  {name}")


def main() -> None:
    args = _parse_args()
    rows: list[dict] = json.loads(args.scores.read_text())
    episodes_total = sum(len(row["episodes"]) for row in rows)
    print(f"読んだ: {len(rows)} 組 / {episodes_total} 本\n")

    _print_rule_table(rows, args.baseline, args.threshold)
    _print_distribution(rows, args.baseline)
    _print_worst_objects(rows, args.baseline, args.threshold)
    _print_per_task(rows, args.baseline, args.threshold, args.rule)

    survivors = [row for row in rows
                 if not _is_dropped(row, args.baseline, args.threshold, args.rule)]

    if args.keep_out:
        keep = sorted(number for row in survivors for number in row["episodes"])
        args.keep_out.write_text(json.dumps({"episodes": keep}))
        instructions = {row["instruction"] for row in survivors}
        print(f"\n=== 残す {len(keep)} 本の番号を書いた: {args.keep_out} ===")
        print(f"  残る指示文 {len(instructions)} / "
              f"{len({row['instruction'] for row in rows})}")

    if args.boost_out:
        wanted = set(args.boost_instruction or [])
        unknown = wanted - {row["instruction"] for row in rows}
        if unknown:
            raise SystemExit(f"その指示文は判定表に無い: {sorted(unknown)}")
        boost = sorted(number for row in survivors
                       if row["instruction"] in wanted for number in row["episodes"])
        args.boost_out.write_text(json.dumps({"episodes": boost}))
        print(f"\n=== 痩せる課題で残った {len(boost)} 本の番号を書いた: {args.boost_out} ===")
        for instruction in sorted(wanted):
            count = sum(len(row["episodes"]) for row in survivors
                        if row["instruction"] == instruction)
            before = sum(len(row["episodes"]) for row in rows
                         if row["instruction"] == instruction)
            print(f"  {before:>4} → {count:>4} 本（3 倍で {count * 3}）  {instruction[:52]}")


if __name__ == "__main__":
    main()
