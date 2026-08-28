#!/usr/bin/env python3
"""失敗の原因を「タスクのせい」と「摂動のせい」に切り分ける。

**摂動カテゴリ別の成功率だけでは、学習の的が決まらない。**
84 件は 30 種類の元タスク × 7 カテゴリに散っているため、
**あるカテゴリが低いのは、そのカテゴリに難しいタスクが偏っているせいかもしれない。**
実際 8/10 の 80 件（元タスク 4 種類）では背景テクスチャが最弱に見えていたが、
84 件では上位だった（`docs/talk/open_items.md` の項目 7）。

ここでは 3 つを分けて数える。

- **元タスクごとの成功率** … タスクそのものの難しさ
- **同じ元タスクの中でのカテゴリ差** … 摂動そのものの難しさ
- **手本の長さとの関係** … `MAX_STEPS=300` に原理的に収まるか

元タスクの識別には CSV の `init_states_file` を使う（variant 名を切り出す必要がない。
1 タスク 1 ファイルで対応している）。手本の長さは `--demo-lengths` の JSON から引く。

    python3 solution/tools/analysis/decompose_failures.py outputs/0811_101824_base_n5_ep1_max300

Reads:
  - <output-dir>/eval.log
  - <output-dir>/VAL_TASKS.csv
  - solution/tools/analysis/demo_lengths.json（`--demo-lengths` で差し替え可）
Generates:
  - 標準出力のみ
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

_RESULT_RE = re.compile(
    r"\[結果\] (?P<task>\S+) ep(?P<episode>\d+): (?P<outcome>成功|衝突|未完了) "
    r"(?P<steps>\d+) steps"
)

# 元タスク名の先頭に付く場面の名前（`KITCHEN_SCENE3_` など）。
# **落とすと指示文そのものになる**ので、手本の長さを引く鍵に使える。
_SCENE_PREFIX_RE = re.compile(r"^[A-Z][A-Z_]*SCENE\d+_")

_DEFAULT_DEMO_LENGTHS = Path(__file__).resolve().parent / "demo_lengths.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="評価の出力先")
    parser.add_argument("--csv", type=Path, default=None,
                        help="検証用スイートの CSV。省略時は出力先の VAL_TASKS.csv")
    parser.add_argument("--demo-lengths", type=Path, default=_DEFAULT_DEMO_LENGTHS,
                        help="指示文 → 手本の長さ の JSON")
    return parser.parse_args()


def _load_outcomes(log_path: Path) -> dict[str, tuple[str, int]]:
    """タスク名 → (結末, ステップ数)。

    Parameters
    ----------
    log_path : Path
        評価ログ。

    Returns
    -------
    outcomes : dict[str, tuple[str, int]]
        結末は `成功` / `衝突` / `未完了` のいずれか。
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        found.group("task"): (found.group("outcome"), int(found.group("steps")))
        for found in _RESULT_RE.finditer(text)
    }


def _load_demo_lengths(path: Path) -> dict[str, int]:
    """指示文 → 手本のステップ数（中央値）。読めなければ空。

    JSON は `lerobot/libero_plus` の `meta/episodes/*.parquet` の `length` 列から
    作ったもので、40 個の指示文それぞれの `median` / `p90` / `max` を持つ。
    """
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: int(values["median"]) for name, values in payload.items()}


def _base_task(row: dict[str, str]) -> str:
    """元タスクの名前。`init_states_file` から拡張子と階層を落としたもの。"""
    return row["init_states_file"].replace(".pruned_init", "").split("/")[-1]


def _instruction_of(row: dict[str, str]) -> str:
    """元タスク名から、学習データ側の指示文を組み立てる。

    **CSV の `instruction` 列は使えない。** あれは variant が実際に受け取る文で、
    `Language Instructions` では言い換えられており、他のカテゴリでも
    綴りが学習データと違う（`Pick the akita black bowl …` 対
    `pick up the black bowl …`）。**元タスク名のほうが学習データと一致する。**
    """
    return _SCENE_PREFIX_RE.sub("", _base_task(row)).replace("_", " ")


def _rate(outcomes: list[str]) -> float:
    """成功率。"""
    return sum(1 for outcome in outcomes if outcome == "成功") / len(outcomes)


def _report_by_base_task(rows: list[dict[str, str]],
                         outcomes: dict[str, tuple[str, int]]) -> None:
    """元タスクごとの成功率。**タスクそのものの難しさを見る。**"""
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        if row["task_id"] in outcomes:
            grouped[_base_task(row)].append(outcomes[row["task_id"]][0])

    print("=== 元タスクごとの成功率（低い順）===")
    print(f"  {'成功率':>6} {'件数':>4}  タスク")
    for task, results in sorted(grouped.items(), key=lambda item: _rate(item[1])):
        print(f"  {_rate(results):>6.2f} {len(results):>4}  {task[:66]}")
    print(f"\n  元タスク {len(grouped)} 種類")


def _report_by_category(rows: list[dict[str, str]],
                        outcomes: dict[str, tuple[str, int]]) -> None:
    """カテゴリごとの成功率。**摂動そのものの難しさ、ただしタスクが混ざる。**"""
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        if row["task_id"] in outcomes:
            grouped[row["category"]].append(outcomes[row["task_id"]][0])

    print("\n=== 摂動カテゴリごとの成功率（低い順）===")
    print("  **元タスクが混ざっている。この表だけで的を決めない。**")
    for category, results in sorted(grouped.items(), key=lambda item: _rate(item[1])):
        print(f"  {_rate(results):>6.2f} {len(results):>4}  {category}")


def _report_within_task(rows: list[dict[str, str]],
                        outcomes: dict[str, tuple[str, int]]) -> None:
    """同じ元タスクの中でカテゴリ差を見る。**タスクの難しさを打ち消す。**

    元タスクごとに「そのタスクの平均」を引いた値を、カテゴリごとに平均する。
    **マイナスなら、その摂動はタスクの平均より難しい。**

    **variant が 1 本だけの元タスクは使えない。** そのタスクの平均がその 1 本自身
    なので、引くと必ず 0 になる。**混ぜると「差が無い」と読めてしまう**
    （物体が増える摂動は全件がこれに当たる）。
    """
    by_task: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for row in rows:
        if row["task_id"] in outcomes:
            by_task[_base_task(row)].append(
                (row["category"], outcomes[row["task_id"]][0])
            )

    deviations: dict[str, list[float]] = collections.defaultdict(list)
    dropped: dict[str, int] = collections.defaultdict(int)
    for pairs in by_task.values():
        if len(pairs) < 2:
            for category, _ in pairs:
                dropped[category] += 1
            continue
        mean = _rate([outcome for _, outcome in pairs])
        for category, outcome in pairs:
            deviations[category].append((1.0 if outcome == "成功" else 0.0) - mean)

    print("\n=== 元タスクの難しさを引いたあとのカテゴリ差 ===")
    print("  **同じ元タスクの中だけで比べている。")
    print("  マイナスが大きいほど、その摂動そのものが効いている。**")
    print(f"  {'ずれ':>6} {'使えた':>6} {'落とした':>8}  カテゴリ")
    for category, values in sorted(
        deviations.items(), key=lambda item: sum(item[1]) / len(item[1])
    ):
        print(f"  {sum(values) / len(values):>+6.2f} {len(values):>6} "
              f"{dropped.get(category, 0):>8}  {category}")
    for category, count in sorted(dropped.items()):
        if category not in deviations:
            print(f"  {'—':>6} {0:>6} {count:>8}  {category}"
                  "  **全件が単独。比べられない**")


def _report_horizon(rows: list[dict[str, str]], outcomes: dict[str, tuple[str, int]],
                    demo_lengths: dict[str, int], cap: int) -> None:
    """手本の長さと結末の関係を見る。**打ち切りが原理的なものかを判定する。**

    Parameters
    ----------
    cap : int
        ステップ数の上限（`MAX_STEPS`）。ログの最大ステップ数から決める。
    """
    if not demo_lengths:
        print("\n=== 手本の長さとの関係 ===\n  **JSON が無いので測れない**")
        return

    bands = (
        (0, cap // 2, "A 手本 <150"),
        (cap // 2, int(cap * 0.77), "B 手本 150〜230"),
        (int(cap * 0.77), cap, "C 手本 230〜300"),
        (cap, 10 ** 9, "**D 手本 >300（原理的に収まらない）**"),
    )
    counted: dict[str, list[int]] = {name: [0, 0, 0, 0] for _, _, name in bands}
    ratios: list[float] = []
    unmatched: set[str] = set()
    per_task: dict[tuple[int, str], list[int]] = collections.defaultdict(
        lambda: [0, 0, 0]
    )

    for row in rows:
        if row["task_id"] not in outcomes:
            continue
        instruction = _instruction_of(row)
        if instruction not in demo_lengths:
            unmatched.add(instruction)
            continue
        length = demo_lengths[instruction]
        outcome, steps = outcomes[row["task_id"]]
        for low, high, name in bands:
            if low <= length < high:
                slot = counted[name]
                slot[0] += 1
                slot[1] += outcome == "成功"
                slot[2] += outcome == "未完了"
                slot[3] += outcome == "衝突"
                break
        task = per_task[(length, instruction)]
        task[0] += 1
        task[1] += outcome == "成功"
        task[2] += outcome == "未完了"
        if outcome == "成功":
            ratios.append(steps / length)

    print("\n=== 手本の長さとの関係 ===")
    print(f"  上限 {cap} 歩。**手本より速く動くことはできない**ので、")
    print("  手本が上限に近いタスクは、少しの無駄でも打ち切りになる")
    if unmatched:
        print(f"  対応が付かなかった指示文 {len(unmatched)} 種"
              "（物体が増える摂動は元タスクが別物になるため）")
    print(f"\n  {'成功率':>6} {'未完了率':>8} {'衝突率':>6} {'件数':>5}  帯")
    for _, _, name in bands:
        total, success, timeout, collision = counted[name]
        if total:
            print(f"  {success / total:>6.2f} {timeout / total:>8.2f} "
                  f"{collision / total:>6.2f} {total:>5}  {name}")

    if ratios:
        ordered = sorted(ratios)
        print(f"\n  成功した回は手本の {statistics.median(ordered):.2f} 倍のステップ数"
              f"（n={len(ordered)}、4 分の 3 {ordered[3 * len(ordered) // 4]:.2f}、"
              f"最大 {ordered[-1]:.2f}）")

    print(f"\n  {'手本':>5} {'成功':>5} {'未完了':>6} {'件数':>5}  タスク（手本の長い順）")
    for (length, instruction), (total, success, timeout) in sorted(
        per_task.items(), key=lambda item: -item[0][0]
    ):
        print(f"  {length:>5} {success:>5} {timeout:>6} {total:>5}  {instruction[:58]}")


def _report_crosstab(rows: list[dict[str, str]],
                     outcomes: dict[str, tuple[str, int]]) -> None:
    """元タスク × カテゴリの表。**どこが空白かも見る。**"""
    categories = sorted({row["category"] for row in rows})
    table: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for row in rows:
        if row["task_id"] in outcomes:
            table[(_base_task(row), row["category"])].append(
                outcomes[row["task_id"]][0]
            )

    print("\n=== 元タスク × カテゴリ（○ 成功 / × 失敗 / 空白 該当なし）===")
    print("  " + "".join(f"{category[:9]:>10}" for category in categories) + "  タスク")
    for task in sorted({_base_task(row) for row in rows}):
        cells = []
        for category in categories:
            results = table.get((task, category))
            marks = "".join("○" if r == "成功" else "×" for r in results or [])
            cells.append(f"{marks:>10}")
        print("  " + "".join(cells) + f"  {task[:44]}")


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output_dir.resolve()
    csv_path: Path = args.csv or (output_dir / "VAL_TASKS.csv")

    outcomes = _load_outcomes(output_dir / "eval.log")
    if not outcomes:
        raise SystemExit(f"結果が 1 件も読めない: {output_dir / 'eval.log'}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    cap = max(steps for _, steps in outcomes.values())
    print(f"結果 {len(outcomes)} 件 / CSV {len(rows)} 行 / ステップ数の上限 {cap}\n")
    _report_by_base_task(rows, outcomes)
    _report_by_category(rows, outcomes)
    _report_within_task(rows, outcomes)
    _report_horizon(rows, outcomes, _load_demo_lengths(args.demo_lengths), cap)
    _report_crosstab(rows, outcomes)


if __name__ == "__main__":
    main()
