#!/usr/bin/env python3
"""2 つの実行を、同じタスクどうしで突き合わせて比べる。

**なぜ率ではなく対で見るのか。** 72 行 216 本の成功率は、**同じ重みを 2 回測るだけで
0.667 と 0.722 の間を動く**（2026-08-28 の実測）。**だから率の差 0.05 は何の証拠にもならない。**
**同じ `(タスク, エピソード番号)` は同じ初期状態である**（評価キットは初期状態をファイルから
順に読み、乱数を使わない）ので、**対にして差の符号を数えれば、率では消える差が残る。**

`compare_object_disp.py` は**物の変位**を対で比べる道具で、こちらは**成功率と歩数**を
タスク単位で比べる。変位の記録（`object_disp.jsonl`）が無い古い実行どうしでも使える。

    python3 compare_runs_by_task.py <基準の実行> <比べる実行>

**符号検定の p は「上下が偶然と区別できるか」だけを見る。**上が 11・下が 4 で p=0.06 なら、
**「動いた」と言うには足りない。**タスクは 72 あるが、条件ごとに分けると 12 しかない。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _results(target: Path) -> dict:
    if target.is_file():
        return json.loads(target.read_text())
    hits = sorted(target.glob("results/*.json"))
    if not hits:
        raise SystemExit(f"結果 JSON が見つからない: {target}")
    return json.loads(hits[0].read_text())


def _tasks(doc: dict) -> dict[str, dict]:
    out = {}
    for track in doc["tracks"]:
        for t in track["tasks"]:
            out[t["task_name"]] = t
    return out


def _labels(run_dir: Path) -> dict[str, str]:
    """実行ディレクトリに写した CSV から、タスク名 → 条件名の対応を作る。"""
    path = run_dir / "TASKS_CSV.csv"
    if not path.is_file():
        return {}
    labels = {}
    for row in csv.DictReader(path.open()):
        category = row.get("category", "")
        note = row.get("difficulty_level", "")
        labels[row["task_id"]] = ("Base" if category == "Base"
                                  else f"{category.split()[0]} {note}".strip())
    return labels


def _sign_p(up: int, down: int) -> float:
    """符号検定（両側）。上下が同じ確率で起きるという前提のもとでの p。"""
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", type=Path, help="基準の実行")
    ap.add_argument("other", type=Path, help="比べる実行")
    args = ap.parse_args()

    a, b = _tasks(_results(args.base)), _tasks(_results(args.other))
    labels = _labels(args.base) or _labels(args.other)
    shared = [k for k in a if k in b]
    if not shared:
        raise SystemExit("同じ名前のタスクが 1 つも無い（別の CSV で回した実行である）")

    print(f"== 基準 {args.base.name}")
    print(f"   比べる {args.other.name}")
    print(f"   対にできたタスク {len(shared)} / 基準 {len(a)} ・ 比べる側 {len(b)}")

    groups: dict[str, list[str]] = {}
    for k in shared:
        groups.setdefault(labels.get(k, "（条件不明）"), []).append(k)

    print("\n   成功率の対（↑ は比べる側が良い）")
    print("   %-18s %4s %4s %4s %7s" % ("条件", "↑", "↓", "=", "p"))
    tot_up = tot_down = 0
    for name in sorted(groups):
        up = down = same = 0
        for k in groups[name]:
            d = b[k]["success_rate"] - a[k]["success_rate"]
            up += d > 1e-9
            down += d < -1e-9
            same += abs(d) <= 1e-9
        tot_up += up
        tot_down += down
        print("   %-18s %4d %4d %4d %7.3f" % (name, up, down, same, _sign_p(up, down)))
    print("   %-18s %4d %4d %4d %7.3f" % ("合計", tot_up, tot_down,
                                          len(shared) - tot_up - tot_down,
                                          _sign_p(tot_up, tot_down)))

    # **歩数は、両方で 1 回以上成功したタスクだけを見る。**成功が 0 の回の歩数は
    # 打ち切りの 300 に張り付くので、混ぜると成功率の差を歩数として二重に数えてしまう。
    print("\n   歩数の対（両方で成功のあるタスクだけ。↑ は比べる側が歩数が多い＝悪い）")
    print("   %-18s %4s %4s %10s %7s" % ("条件", "↑", "↓", "中位の変化", "p"))
    for name in sorted(groups):
        up = down = 0
        diffs = []
        for k in groups[name]:
            if a[k]["success_rate"] <= 0 or b[k]["success_rate"] <= 0:
                continue
            sa = a[k]["metrics"]["avg_steps_to_success"]
            sb = b[k]["metrics"]["avg_steps_to_success"]
            if not sa or not sb:
                continue
            diffs.append((sb - sa) / sa)
            up += sb > sa
            down += sb < sa
        if not diffs:
            continue
        print("   %-18s %4d %4d %9.1f%% %7.3f"
              % (name, up, down, 100 * statistics.median(diffs), _sign_p(up, down)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
