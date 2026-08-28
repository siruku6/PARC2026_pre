#!/usr/bin/env python3
"""2 つの実行を、同じ初期状態のロールアウトどうしで突き合わせて比べる。

**何のためにあるか。** 12 行 36 本や 72 行 216 本では、成功率と衝突率が動かなくても
**中身は入れ替わっている。**2026-08-28 に `EMA 0.5` を測ったとき、成功率も衝突率も
基準と小数点以下まで一致したのに、**36 本のうち 16 本は衝突の有無が入れ替わっていた。**
**率だけ見ていると「何も起きなかった」と読んでしまう。**

**同じ `(タスク, エピソード番号)` は同じ初期状態である。**評価キットは初期状態を
ファイルから順に読み、乱数を使わない。**だから 2 つの実行の同じ番号どうしは、
設定だけが違う対である。**対にして引き算すると、率では消える差が残る。

**表は 3 つ出る。**

| 表 | 何が読めるか |
|---|---|
| 1 | **どれだけ入れ替わったか。**率が同じでも中身が違うことはここで分かる |
| 2 | **両方で衝突した対の変位の差。**同じ状況で接触が深くなったか浅くなったか |
| 3 | 片方だけで衝突した対の一覧（どのタスクが不安定か） |

**表 2 の「悪化した本数」が主役である。**中央値や平均は 1 本の外れ値で動くが、
**符号の数は動かない。**14 本のうち 10 本が悪化なら、偶然（7 本）から離れている。

    python3 compare_object_disp.py <基準の実行> <比べる実行>
    python3 compare_object_disp.py --metric l1 <基準の実行> <比べる実行>
    python3 compare_object_disp.py --tasks compe/val/COOKIE12.csv <72行の実行> <12行の実行>

Reads:
  - <実行ディレクトリ>/results/object_disp.jsonl（両方）
  - --tasks に渡した CSV の `task_id` 列（省略時は両方に共通するタスクすべて）
Generates:
  - 標準出力の Markdown（ファイルは書かない）
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# **中央値は `summarize_object_disp.py` と同じ規則で取る。**偶数本のときに
# 真ん中 2 つの平均を使う流儀と、上側を使う流儀がある。**道具どうしで違うと、
# 同じ実行に 2 つの中央値が付いて追えなくなる。**あちらは上側なのでこちらも上側にする。
METRIC_KEYS: dict[str, str] = {"l1": "max_disp_m", "l2": "max_disp_l2_m"}

RolloutKey = tuple[str, int]


def _resolve_jsonl(target: Path) -> Path:
    """指定が実行ディレクトリでもファイルでも、JSONL の実体へたどり着く。"""
    if target.is_file():
        return target
    for candidate in (target / "results" / "object_disp.jsonl",
                      target / "object_disp.jsonl"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"object_disp.jsonl が見つからない: {target}")


def _load_by_key(jsonl_path: Path) -> dict[RolloutKey, dict]:
    """JSONL を `(タスク名, エピソード番号)` で引ける形に読む。"""
    records: dict[RolloutKey, dict] = {}
    for line_number, line in enumerate(jsonl_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{jsonl_path}:{line_number} が読めない: {error}")
        records[(record["task"], int(record["episode"]))] = record
    return records


def _load_task_filter(tasks_csv: Path | None) -> set[str] | None:
    """CSV の `task_id` 列を読む。渡されなければ絞り込まない。"""
    if tasks_csv is None:
        return None
    if not tasks_csv.is_file():
        raise SystemExit(f"CSV が見つからない: {tasks_csv}")
    return {row["task_id"] for row in csv.DictReader(tasks_csv.open())}


def _max_displacement_mm(record: dict, metric: str) -> float:
    """そのロールアウトで最も動いた物体の変位を mm で返す。"""
    values = record.get(METRIC_KEYS[metric], {}).values()
    return max(values, default=0.0) * 1000.0


def _median(values: list[float]) -> float:
    """`summarize_object_disp.py` と同じ規則の中央値（偶数本なら上側）。"""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _print_flip_table(
    keys: list[RolloutKey], base: dict[RolloutKey, dict], other: dict[RolloutKey, dict]
) -> None:
    """表 1 —— 率が同じでも中身が入れ替わっていないかを出す。"""
    both = sum(1 for k in keys if base[k]["collided"] and other[k]["collided"])
    neither = sum(1 for k in keys if not base[k]["collided"] and not other[k]["collided"])
    only_base = sum(1 for k in keys if base[k]["collided"] and not other[k]["collided"])
    only_other = sum(1 for k in keys if not base[k]["collided"] and other[k]["collided"])
    print(f"\n## 1. どれだけ入れ替わったか（{len(keys)} 対）\n")
    print("| | 本数 |")
    print("|---|---:|")
    print(f"| 両方で衝突した | {both} |")
    print(f"| どちらでも衝突しなかった | {neither} |")
    print(f"| **基準だけで衝突した**（比べる側で消えた） | **{only_base}** |")
    print(f"| **比べる側だけで衝突した**（新しく出た） | **{only_other}** |")
    flipped = only_base + only_other
    print(f"\n**入れ替わったのは {flipped} / {len(keys)} 本である。**")
    if only_base == only_other and flipped:
        print(f"**衝突率は動いていないが、中身は {flipped} 本入れ替わっている——"
              "率だけ見て「何も起きなかった」と読まないこと。**")


def _print_paired_table(
    keys: list[RolloutKey], base: dict[RolloutKey, dict],
    other: dict[RolloutKey, dict], metric: str,
) -> None:
    """表 2 —— 両方で衝突した対だけを引き算する。"""
    pairs = [k for k in keys if base[k]["collided"] and other[k]["collided"]]
    print(f"\n## 2. 両方で衝突した対の変位の差（{metric.upper()}・{len(pairs)} 対）\n")
    if not pairs:
        print("両方で衝突した対が無い。")
        return
    differences = [_max_displacement_mm(other[k], metric)
                   - _max_displacement_mm(base[k], metric) for k in pairs]
    worse = sum(1 for d in differences if d > 0)
    print("| | 値 |")
    print("|---|---:|")
    print(f"| 差の中央値（比べる側 − 基準） | {_median(differences):+.2f} mm |")
    print(f"| 差の平均 | {sum(differences) / len(differences):+.2f} mm |")
    print(f"| **深くなった（悪化した）対** | **{worse} / {len(pairs)}** |")
    print(f"| 浅くなった対 | {len(pairs) - worse} / {len(pairs)} |")
    print("\n**符号の数を先に見ること。**中央値と平均は 1 本の外れ値で向きが変わるが、"
          "**何本が悪化したかは変わらない。**")


def _print_flip_detail_table(
    keys: list[RolloutKey], base: dict[RolloutKey, dict],
    other: dict[RolloutKey, dict], metric: str,
) -> None:
    """表 3 —— 片方だけで衝突した対を並べる。"""
    rows = [k for k in keys if base[k]["collided"] != other[k]["collided"]]
    print(f"\n## 3. 片方だけで衝突した対（{len(rows)} 本）\n")
    if not rows:
        print("入れ替わった対は無い。")
        return
    print("| タスク | ep | 基準 | 比べる側 |")
    print("|---|---:|---|---|")
    for key in sorted(rows):
        task, episode = key
        cells = []
        for record in (base[key], other[key]):
            cells.append(f"{_max_displacement_mm(record, metric):.2f} mm"
                         if record["collided"] else "衝突なし")
        # **タスク名は切り詰めない。**区別が付くのは末尾の `view_...` と
        # `initstate_...` なので、頭を残して切ると全部同じ行に見える。
        print(f"| {task} | {episode} | {cells[0]} | {cells[1]} |")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="基準の実行ディレクトリ")
    parser.add_argument("other", type=Path, help="比べる実行ディレクトリ")
    parser.add_argument("--metric", choices=["l1", "l2"], default="l2",
                        help="L1 はキットが衝突の判定に使う値、L2 は実距離（既定 l2）")
    parser.add_argument("--tasks", type=Path, default=None,
                        help="この CSV の task_id 列に載っている行だけを比べる")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base: dict[RolloutKey, dict] = _load_by_key(_resolve_jsonl(args.base))
    other: dict[RolloutKey, dict] = _load_by_key(_resolve_jsonl(args.other))
    task_filter: set[str] | None = _load_task_filter(args.tasks)

    keys: list[RolloutKey] = sorted(
        key for key in set(base) & set(other)
        if task_filter is None or key[0] in task_filter
    )
    if not keys:
        raise SystemExit("2 つの実行に共通するロールアウトが 1 本も無い")

    print(f"# 対応を取った比較 — 基準 {Path(args.base).name} ／ "
          f"比べる側 {Path(args.other).name}")
    print(f"\n- 共通するロールアウト: {len(keys)} 対"
          f"（基準 {len(base)} 本・比べる側 {len(other)} 本）")
    for label, records in (("基準", base), ("比べる側", other)):
        subset = [records[k] for k in keys]
        print(f"- {label}: 成功率 {sum(r['success'] for r in subset) / len(subset):.3f}"
              f"・衝突率 {sum(r['collided'] for r in subset) / len(subset):.3f}")

    _print_flip_table(keys, base, other)
    _print_paired_table(keys, base, other, args.metric)
    _print_flip_detail_table(keys, base, other, args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
