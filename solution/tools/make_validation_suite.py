#!/usr/bin/env python3
"""本番に近い「未知タスク」の検証用スイートを組み立てる。

配布された `compe/t1/T1_TASKS.csv` の 4 件は **example** であり、
本番は非公開タスクで採点される（`compe/t1/README.md`）。実際、この 4 件で
成功率 0.875 を出しても本番は 0.096 だった。**手元の評価が本番を予測できない。**

そこで LIBERO-plus の 10,030 タスク変種から、example と重ならないものを
選んで CSV にする。`compe/t1/register.py` は CSV を読んでスイートを組み立てる
だけなので、この CSV に差し替えれば評価パイプラインはそのまま動く。

選び方は example の構成に合わせる。example 4 件はすべて Background Textures か
Light Conditions に属し、運営 Q&A No.21 の「7 カテゴリすべてではない」と
合わせると、**視覚摂動の 2 カテゴリが本命**と読める。スイート（spatial /
object / goal / 10）は 4 つとも混ぜる。

**Colab のキット環境で実行すること。** LIBERO-plus の実体が要る。

    PYTHONPATH=$KIT_DIR/LIBERO-plus:$KIT_DIR:$KIT_DIR/compe \\
    $KIT_DIR/venv/bin/python make_validation_suite.py \\
        --libero-plus $KIT_DIR/LIBERO-plus --per-cell 3

Reads:
  - <libero-plus>/libero/libero/benchmark/task_classification.json
  - <libero-plus> 配下の bddl ファイル（指示文の取得）
  - compe/t1/T1_TASKS.csv（除外する example）
Generates:
  - compe/val/VAL_TASKS.csv（検証用スイートの定義）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

# example 4 件が属する 2 カテゴリ。本番もここが中心と読める。
_DEFAULT_CATEGORIES = ("Background Textures", "Light Conditions")

# 初期状態ファイル名を得るために剥がす接尾辞。
# **配布キットの register.py は `_light_*` と `_table_N` しか剥がさない。**
# 運営 Q&A によれば本番の実装は `_tb_` も剥がすため、こちらを広く取る。
_SUFFIX_RE = re.compile(
    r"_light_[^.]*|_table_\d+|_tb_\d+|_noise_\d+|_initstate_\d+|_add_\d+"
)

_LANGUAGE_RE = re.compile(r"\(:language\s+(.+?)\)", re.S)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--libero-plus", type=Path, required=True,
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--exclude", type=Path,
                        default=Path("compe/t1/T1_TASKS.csv"),
                        help="除外する example の CSV")
    parser.add_argument("--output", type=Path,
                        default=Path("compe/val/VAL_TASKS.csv"),
                        help="出力する CSV")
    parser.add_argument("--per-cell", type=int, default=3,
                        help="スイート × カテゴリ 1 組あたりの本数")
    parser.add_argument("--categories", type=str, nargs="*",
                        default=list(_DEFAULT_CATEGORIES),
                        help="対象の摂動カテゴリ")
    parser.add_argument("--init-states-root", type=Path, default=None,
                        help="初期状態ファイルの置き場。省略時は libero から取得する")
    return parser.parse_args()


def _load_classification(libero_plus: Path) -> list[dict]:
    """タスク変種の分類表を読む。

    Returns
    -------
    entries : list[dict]
        `id` / `name` / `category` / `difficulty_level` / `suite` を持つ辞書の列。
        鍵の名前は版によって違いうるため、見つからなければその場で落とす。
    """
    path = libero_plus / "libero/libero/benchmark/task_classification.json"
    if not path.is_file():
        raise FileNotFoundError(f"分類表が見つかりません: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else list(raw.values())
    if records and not isinstance(records[0], dict):
        # {category: [name, ...]} の形なら、平坦な辞書の列へ直す。
        records = [
            {"name": name, "category": category}
            for category, names in raw.items() for name in names
        ]

    missing = [key for key in ("name", "category") if key not in records[0]]
    if missing:
        raise KeyError(
            f"分類表に {missing} がありません。実際の鍵: {sorted(records[0])}"
        )
    return records


def _suite_of(entry: dict) -> str:
    """元のスイート名を返す。分類表に無ければタスク名から推測する。"""
    for key in ("suite", "problem_folder", "benchmark"):
        if entry.get(key):
            return str(entry[key])

    name = entry["name"]
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        if suite in name:
            return suite
    raise KeyError(f"スイートを決められません: {name}")


def _find_bddl(libero_plus: Path, task_id: str) -> Path | None:
    """タスク変種の bddl を探す。"""
    matches = list(libero_plus.rglob(f"{task_id}.bddl"))
    return matches[0] if matches else None


def _instruction_of(entry: dict, bddl: Path | None) -> str | None:
    """指示文を得る。分類表に無ければ bddl の `:language` を読む。"""
    for key in ("instruction", "language", "goal"):
        if entry.get(key):
            return str(entry[key]).strip().strip('"')

    if bddl is None:
        return None
    found = _LANGUAGE_RE.search(bddl.read_text(encoding="utf-8"))
    if not found:
        return None
    return " ".join(found.group(1).split()).strip('"')


def _init_states_path(
    task_id: str, suite: str, init_states_root: Path
) -> Path | None:
    """初期状態ファイルの場所を返す。無ければ None。

    `register.py` と同じく、接尾辞を剥がした名前で探す。剥がさない名前でも
    見つかることがあるため、両方試す。
    """
    stripped = _SUFFIX_RE.sub("", task_id)
    for stem in (stripped, task_id):
        candidate = init_states_root / suite / f"{stem}.pruned_init"
        if candidate.is_file():
            return candidate
    return None


def _resolve_init_states_root(explicit: Path | None) -> Path:
    """初期状態ファイルの置き場を決める。"""
    if explicit is not None:
        return explicit
    from libero.libero import get_libero_path
    return Path(get_libero_path("init_states"))


def _load_excluded(path: Path) -> set[str]:
    """example のタスク ID を読む。検証用スイートから外すため。"""
    if not path.is_file():
        return set()
    with path.open(newline="") as stream:
        return {row["task_id"] for row in csv.DictReader(stream)}


def _spread(items: list[dict], count: int) -> list[dict]:
    """並びの両端を含めて等間隔に count 件選ぶ。

    乱数を使わないので、**何度実行しても同じスイートになる。**
    条件を変えて比べるときに母集団が動くと比較にならない。
    """
    if count >= len(items):
        return list(items)
    if count <= 1:
        return [items[len(items) // 2]]
    positions = [round(rank * (len(items) - 1) / (count - 1)) for rank in range(count)]
    return [items[position] for position in positions]


def _select(
    entries: list[dict], categories: list[str], excluded: set[str], per_cell: int
) -> list[dict]:
    """スイート × カテゴリの組ごとに、等間隔で選ぶ。"""
    cells: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        if entry["category"] not in categories or entry["name"] in excluded:
            continue
        cells.setdefault((_suite_of(entry), entry["category"]), []).append(entry)

    selected: list[dict] = []
    for key in sorted(cells):
        group = sorted(cells[key], key=lambda item: item["name"])
        selected.extend(_spread(group, per_cell))
    return selected


def _build_rows(
    selected: list[dict], libero_plus: Path, init_states_root: Path
) -> tuple[list[dict], list[str]]:
    """CSV の行を作り、実体が無いものは落とす。

    Returns
    -------
    rows : list[dict]
        `T1_TASKS.csv` と同じ列を持つ行。
    skipped : list[str]
        落とした理由つきのタスク名。**黙って減らさない。**
    """
    rows: list[dict] = []
    skipped: list[str] = []

    for entry in selected:
        task_id = entry["name"]
        suite = _suite_of(entry)

        bddl = _find_bddl(libero_plus, task_id)
        if bddl is None:
            skipped.append(f"{task_id}: bddl が無い")
            continue

        instruction = _instruction_of(entry, bddl)
        if not instruction:
            skipped.append(f"{task_id}: 指示文を取れない")
            continue

        if _init_states_path(task_id, suite, init_states_root) is None:
            skipped.append(f"{task_id}: 初期状態ファイルが無い")
            continue

        rows.append({
            "task_num": len(rows) + 1,
            "task_id": task_id,
            "instruction": instruction,
            "libero_plus_id": entry.get("id", ""),
            "suite": suite,
            "category": entry["category"],
            "difficulty_level": entry.get("difficulty_level", ""),
        })
    return rows, skipped


def _report(rows: list[dict], skipped: list[str]) -> None:
    """選んだ結果を表で見せる。"""
    print(f"\n選んだタスク変種: {len(rows)} 件")
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["suite"], row["category"])
        counts[key] = counts.get(key, 0) + 1
    for (suite, category), count in sorted(counts.items()):
        print(f"  {suite:16s} {category:22s} {count}")

    if skipped:
        print(f"\n**落としたもの: {len(skipped)} 件**")
        for reason in skipped[:20]:
            print(f"  {reason}")
        if len(skipped) > 20:
            print(f"  ... 他 {len(skipped) - 20} 件")


def main() -> None:
    args = _parse_args()
    init_states_root = _resolve_init_states_root(args.init_states_root)
    print(f"[val] 初期状態: {init_states_root}")

    entries = _load_classification(args.libero_plus)
    print(f"[val] 分類表: {len(entries)} 件")

    excluded = _load_excluded(args.exclude)
    print(f"[val] 除外する example: {len(excluded)} 件")

    selected = _select(entries, args.categories, excluded, args.per_cell)
    rows, skipped = _build_rows(selected, args.libero_plus, init_states_root)
    if not rows:
        raise RuntimeError("1 件も選べませんでした。--libero-plus の指定を確認してください")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _report(rows, skipped)
    print(f"\n書き出した: {args.output}")
    print(f"1 タスクあたり 5 エピソードなら計 {len(rows) * 5} エピソード")


if __name__ == "__main__":
    main()
