#!/usr/bin/env python3
"""接触の失点が、どの課題の群に、どれだけ集まっているかを 1 つの表にする。

**なぜ要るか。**[plan/24](../../../docs/plan/24_contact_avoidance_redesign.md) の
2.2 節は「丼の隣（上）に別の物がある 5 課題が、本数の 20% で失点の 65% を出す」
ことを示した。**では残りはどう並んでいるのか**——次にどこへ広げるかは
「接触でいくら落ちているか」と「そもそも届いていない余地」を**分けて**見ないと決まらない。
接触を消しても `done` が立たない課題に手を入れても、点は動かない。

**群の切り方は課題名ではなく `obj_of_interest` である。**同じ道具立て
（同じ丼・同じカゴ）なら 1 つの雛形で書けるので、**追加の手間がそこで決まる。**

| 群 | 中身 | なぜひとまとめか |
|---|---|---|
| **A** | 丼の隣・上に別の物がある 5 課題 | **いま作り直した所** |
| 1 | 残りの `akita_black_bowl_1` 課題 | **同じ丼**。`RimPickPlace` の計画を足すだけ |
| 2 | `basket_1` 課題 | **同じカゴ**。雛形 1 つで 12 家族 |
| 3 | それ以外 | 引き出し・電子レンジ・本棚…**機構が別** |

**課題名には摂動の接尾辞が付く**（`_view_45_0_100_0_0_initstate_0`・`_light_29`・
`_table_3`）。**剥がしてから数える**——剥がさないと 1 家族が 6 つに割れ、
「家族あたり何本か」が読めない。

**2 つの数を分けて出す。**どちらも**全体の本数**を分母にした百分率で、
合計すると「その群を完全に解いたときの全体の伸び」になる。

    接触の失点   done ∧ collided … **接触を消せば取れる**
    未達の余地   ¬done           … **接触を消しても取れない**

用法:
    python3 solution/tools/eval/tier_contact_loss.py \\
        work/final_eval/0910_105839_track2_full_local \\
        work/final_eval/0910_210156_track2_v2_base24k ...

Reads:
  - <評価ディレクトリ>/results/object_disp.jsonl
Generates:
  - 標準出力に表（群ごと・`--detail` で家族ごとも）
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

#: 2026-09-10 までに手元で回した評価のうち、`object_disp.jsonl` を持つ 6 回。
#: **既定でこれを読む。**plan/24 の 2.1 節と同じ母集団（1,154 本）である。
DEFAULT_RUNS = (
    "0910_105839_track2_full_local", "0910_210156_track2_v2_base24k",
    "0910_094945_track2_clean24k", "0828_224224_track2_trial1500_s1500",
    "0910_171657_track1_full_local_ep5", "0910_162003_track1_clean24k_ep5",
)

#: 群 A——`RimPickPlace` で作り直した 5 課題。
BOWL5 = frozenset({
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
})

#: 課題名の末尾に付く摂動の印。**繰り返し剥がす**（2 つ付くことがある）。
_SUFFIX = re.compile(r"_(view_[\d_]+_initstate_\d+|light_\d+|table_\d+|initstate_\d+)$")


def family(task: str) -> str:
    """摂動の接尾辞を剥がした課題名を返す。"""
    previous = None
    while previous != task:
        previous, task = task, _SUFFIX.sub("", task)
    return task


def tier(task_family: str, objects: set[str]) -> str:
    """1 本がどの群に属するかを返す。**判定は道具立てで行う。**"""
    if task_family in BOWL5:
        return "A 今回の 5 課題"
    if "akita_black_bowl_1" in objects:
        return "1 同じ丼"
    if "basket_1" in objects:
        return "2 カゴ"
    return "3 機構が別"


def load(runs: list[Path]) -> list[dict]:
    """評価ディレクトリの `object_disp.jsonl` を読み、群と家族を付けて返す。"""
    rows: list[dict] = []
    for run in runs:
        path = run / "results" / "object_disp.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"object_disp.jsonl が無い: {path}")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["family"] = family(row["task"])
            row["tier"] = tier(row["family"], set(row["obj_of_interest"]))
            rows.append(row)
    return rows


def _stats(rows: list[dict], total: int) -> tuple[int, int, float, float, float, float]:
    count = len(rows)
    done = sum(bool(row["done"]) for row in rows)
    success = sum(bool(row["success"]) for row in rows)
    return (len({row["family"] for row in rows}), count,
            100.0 * done / count, 100.0 * success / count,
            100.0 * (done - success) / total, 100.0 * (count - done) / total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, metavar="DIR",
                        help=f"評価ディレクトリ。省略すると既定の {len(DEFAULT_RUNS)} 回")
    parser.add_argument("--eval-root", type=Path, default=Path("work/final_eval"))
    parser.add_argument("--detail", action="store_true", help="家族ごとも出す")
    args = parser.parse_args()

    runs = args.runs or [args.eval_root / name for name in DEFAULT_RUNS]
    rows = load(runs)
    total = len(rows)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["tier"]].append(row)

    print(f"全 {total} 本・{len({row['family'] for row in rows})} 家族"
          f"（{len(runs)} 回の評価）\n")
    print("| 段 | 家族 | 本 | done | 成功 | 接触の失点 | 未達の余地 |")
    print("|---|---|---|---|---|---|---|")
    for name in sorted(grouped, key=lambda key: (key[0] != "A", key)):
        families, count, done, success, lost, missed = _stats(grouped[name], total)
        print(f"| {name} | {families} | {count} | {done:.1f}% | {success:.1f}% "
              f"| {lost:.1f} pt | {missed:.1f} pt |")

    if not args.detail:
        return
    for name in sorted(grouped, key=lambda key: (key[0] != "A", key)):
        print(f"\n--- {name} ---")
        for name_family in sorted({row["family"] for row in grouped[name]}):
            picked = [row for row in grouped[name] if row["family"] == name_family]
            count = len(picked)
            done = sum(bool(row["done"]) for row in picked)
            success = sum(bool(row["success"]) for row in picked)
            print(f"  {count:4d} done {100.0 * done / count:5.1f}% "
                  f"接触 {100.0 * (done - success) / count:5.1f}%  {name_family}")


if __name__ == "__main__":
    main()
