#!/usr/bin/env python3
"""本選 Track2 の 4 カテゴリを 1 枚で測る、小さな探査用 CSV を作る。

**8/24 の本選開始前に「どのカテゴリでいくら落ちるか」を知るために作った。**
`EVAL_TASKS.csv` の 36 件は Track1 の 2 カテゴリ（背景・照明）だけで、
Track2 で加わる**カメラ位置・ロボット初期姿勢・物体配置・センサノイズ**が
1 件も入っていない。**つまり今の一覧では Track2 の弱点が見えない。**

作る形は「タスク 8 件 × （摂動 4 カテゴリ ＋ 摂動なし 1 件）＝ 40 行」である。

  - **摂動なしの行を同じタスクで並べて入れる。** これが無いと
    「タスクが難しいのか、摂動が効いたのか」を切り分けられない。
  - **行の順はタスクごとにまとめる**（タスク1の5行 → タスク2の5行 …）。
    途中で評価が落ちても、そこまでのタスクは 5 カテゴリぶんそろう。
  - **1 セルにつき 1 件しか測らないので、粒度は 8 件 = 12.5 ポイントである。**
    2 カテゴリの差が 1 件（12.5 pt）なら偶然と区別できない。**順位を見るための
    道具であって、値を報告するための道具ではない。**

変種の選び方:
  1. `EVAL_TASKS.csv` の `init_states_file` から元タスク名を取る
     （`instruction` では引けない。**あちらの指示文は運営の言い換えで、
     学習データの綴りと違う**ため、名前で引かないと 0 件になる）。
  2. スイート 4 つから均等にタスクを取る（既定は各 2 件）。
  3. そのタスク × カテゴリの変種のうち、**難易度が中央値のものを 1 件**取る。
     `id` の小さい順に取ると系統的にいちばん弱い摂動だけを測ってしまう。
  4. 乱数を使わない。同じ引数なら同じ CSV になる。

**出た CSV は、学習用の変種を作るときの除外一覧としてそのまま渡せる。**
`make_validation_suite.py --exclude` は `task_id` 列だけを読むので、

    python3 solution/tools/make_validation_suite.py \
        --exclude compe/val/TRACK2_PROBE.csv ...

とすれば、検証で測る変種が学習データに混ざることを防げる
（`docs/plan/07_track2_preparation.md` 2.2 節の評価スイート設計要件 2）。

使い方:

    python3 solution/tools/make_track2_probe.py --output compe/val/TRACK2_PROBE.csv

    # 回す（1 件 2 分ほど。40 行で 80 分ほど）
    EVAL_CSV=compe/val/TRACK2_PROBE.csv bash solution/tools/run_docker_eval.sh --detach

Reads:
  - compe/val/EVAL_TASKS.csv（測る元タスクの出どころ）
  - <libero-plus>/libero/libero/benchmark/task_classification.json
  - <libero-plus>/libero/libero/{bddl_files,init_files}/（実体の存在確認）
Generates:
  - <output>（`compe/t1/T1_TASKS.csv` と同じ列 ＋ `init_states_file`）
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# **本選 Track2 で加わると読んでいる 4 カテゴリ。**
# 運営が category 名まで確定させたわけではない（`docs/reference/05_*.md`）。
# 本選の課題文を読んだら、まずこの並びを直すこと。
_TRACK2_CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Objects Layout",
    "Sensor Noise",
)

# 摂動なしの行に入れるカテゴリ名。`compe/val/MOKA_BASE.csv` と同じ書き方にそろえる。
_BASE_CATEGORY = "Base"

_INIT_SUFFIX = ".pruned_init"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-tasks", type=Path,
                        default=_REPO_ROOT / "compe/val/EVAL_TASKS.csv",
                        help="元タスクの出どころ。36 件の評価用 CSV")
    parser.add_argument("--tasks", type=int, default=8,
                        help="測るタスク数。スイート 4 つに均等に割る")
    parser.add_argument("--categories", nargs="*", default=list(_TRACK2_CATEGORIES),
                        help="測る摂動の種類")
    parser.add_argument("--no-base", action="store_true",
                        help="摂動なしの行を入れない。**切り分けができなくなる**")
    parser.add_argument("--libero-plus", type=Path,
                        default=_REPO_ROOT / "work/LIBERO-plus",
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--bddl-root", type=Path, default=None,
                        help="bddl の置き場。既定は <libero-plus>/libero/libero/bddl_files")
    parser.add_argument("--init-states-root", type=Path, default=None,
                        help="初期状態の置き場。既定は <libero-plus>/libero/libero/init_files")
    parser.add_argument("--output", type=Path, required=True, help="出力する CSV")
    parser.add_argument("--dry-run", action="store_true", help="書かずに内容だけ出す")
    return parser.parse_args()


def _load_classification(libero_plus: Path) -> list[dict]:
    """変種の分類表を読み、スイート名を各件に写して返す。

    **変種そのものには `suite` 鍵が無く、スイートは外側の鍵にしかない。**
    `make_validation_suite._load_classification` と同じ扱いにそろえてある。
    """
    path = libero_plus / "libero/libero/benchmark/task_classification.json"
    if not path.is_file():
        raise SystemExit(f"[probe] 分類表が見つかりません: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"[probe] 分類表の形が想定外です: {type(raw).__name__}")
    return [
        {**record, "suite": record.get("suite", suite)}
        for suite, records in raw.items()
        for record in records
    ]


def _load_base_tasks(eval_tasks: Path) -> list[dict]:
    """評価用 CSV から元タスク名を取る。

    **`instruction` では引けない。** `EVAL_TASKS.csv` の指示文は運営の言い換え
    （`Pick the alphabet soup and ...`）で、学習データ・分類表の綴りと違う。
    名前は `init_states_file`（`<元タスク名>.pruned_init`）から取れば確実である。
    """
    with eval_tasks.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"[probe] 中身がありません: {eval_tasks}")

    seen: dict[str, dict] = {}
    for row in rows:
        init_file = row.get("init_states_file", "")
        if not init_file.endswith(_INIT_SUFFIX):
            raise SystemExit(
                f"[probe] 元タスク名が取れません: {init_file!r}"
                f"（`{_INIT_SUFFIX}` で終わる列を期待している）"
            )
        name = init_file[: -len(_INIT_SUFFIX)]
        seen.setdefault(name, {"name": name, "suite": row["suite"]})
    return list(seen.values())


def _pick_tasks(base_tasks: list[dict], count: int) -> list[dict]:
    """スイートに均等に割り、各スイートの中では等間隔に取る。

    **1 つのスイートに寄せて測ると、そのスイート特有の弱さを摂動の弱さと
    読み違える。** 余りは件数の多いスイートから 1 件ずつ足す。
    """
    by_suite: dict[str, list[dict]] = collections.defaultdict(list)
    for task in base_tasks:
        by_suite[task["suite"]].append(task)

    suites = sorted(by_suite, key=lambda name: (-len(by_suite[name]), name))
    quota = {name: count // len(suites) for name in suites}
    for name in suites[: count % len(suites)]:
        quota[name] += 1

    picked: list[dict] = []
    for name in sorted(suites):
        items = by_suite[name]
        take = min(quota[name], len(items))
        if take <= 1:
            picked.extend(items[:take])
            continue
        step = (len(items) - 1) / (take - 1)
        picked.extend(items[round(index * step)] for index in range(take))
    return picked


def _pick_variant(entries: list[dict], base: str, category: str) -> dict | None:
    """そのタスク × カテゴリの変種から、難易度が中央値の 1 件を選ぶ。

    **`id` 順の先頭を取ってはいけない。** 分類表は摂動の強さの順に並んでいる
    ことがあり、いちばん弱い摂動だけを 8 件測って「Track2 でも落ちない」と
    読み違える。中央値なら、そのカテゴリの典型的な強さを 1 件で代表できる。

    Returns
    -------
    variant : dict | None
        `name` / `suite` / `category` / `difficulty_level` / `id` を持つ辞書。
        1 件も無ければ None。
    """
    matched = [
        entry for entry in entries
        if entry["category"] == category and entry["name"].startswith(base)
    ]
    if not matched:
        return None
    levels = sorted(entry.get("difficulty_level", 0) for entry in matched)
    median = levels[(len(levels) - 1) // 2]
    at_median = [
        entry for entry in matched
        if entry.get("difficulty_level", 0) == median
    ]
    return min(at_median, key=lambda entry: entry["id"])


def _select(
    entries: list[dict],
    tasks: list[dict],
    categories: list[str],
    with_base: bool,
) -> tuple[list[dict], list[str]]:
    """`_build_rows` に渡す並びを作る。**タスクごとにまとめる。**

    Returns
    -------
    selected : list[dict]
    missing : list[str]
        変種が 1 件も無かった（タスク, カテゴリ）の説明。**黙って減らさない。**
    """
    selected: list[dict] = []
    missing: list[str] = []
    for task in tasks:
        if with_base:
            # **摂動なしの行。** 変種名ではなく元タスク名をそのまま入れる。
            # 分類表に無いので `difficulty_level` と `id` は空にする。
            selected.append({
                "name": task["name"], "suite": task["suite"],
                "category": _BASE_CATEGORY, "difficulty_level": "", "id": "",
            })
        for category in categories:
            variant = _pick_variant(entries, task["name"], category)
            if variant is None:
                missing.append(f"{task['name']} × {category}")
                continue
            selected.append({
                "name": variant["name"], "suite": variant["suite"],
                "category": category,
                "difficulty_level": variant.get("difficulty_level", ""),
                "id": variant["id"],
            })
    return selected, missing


def _report_design(rows: list[dict], categories: list[str]) -> None:
    """**測れる粒度を最後に書く。** 8 件で 1 件の差は 12.5 pt である。"""
    counts = collections.Counter(row["category"] for row in rows)
    print("\n[probe] カテゴリ別の件数と粒度")
    for category in [_BASE_CATEGORY, *categories]:
        count = counts.get(category, 0)
        if not count:
            continue
        print(f"[probe]   {category:22s} {count:2d} 件 → 1 件 = {100 / count:.1f} pt")
    levels = [row["difficulty_level"] for row in rows if row["difficulty_level"] != ""]
    if levels:
        print(f"[probe]   難易度: 中央 {statistics.median(map(int, levels)):.1f} / "
              f"範囲 {min(map(int, levels))}–{max(map(int, levels))}")


def main() -> None:
    args = _parse_args()
    categories = [name for name in args.categories if name]

    bddl_root = args.bddl_root or args.libero_plus / "libero/libero/bddl_files"
    init_states_root = (args.init_states_root
                        or args.libero_plus / "libero/libero/init_files")
    for root in (bddl_root, init_states_root):
        if not root.is_dir():
            raise SystemExit(f"[probe] 置き場が見つかりません: {root}")

    entries = _load_classification(args.libero_plus)
    base_tasks = _load_base_tasks(args.eval_tasks)
    print(f"[probe] 分類表 {len(entries)} 件 / 元タスク {len(base_tasks)} 件"
          f"（{args.eval_tasks}）")

    tasks = _pick_tasks(base_tasks, args.tasks)
    print(f"[probe] 測るタスク {len(tasks)} 件: "
          + ", ".join(f"{name} {count}"
                      for name, count in
                      sorted(collections.Counter(t["suite"] for t in tasks).items())))
    for task in tasks:
        print(f"[probe]   {task['suite']:16s} {task['name']}")

    selected, missing = _select(entries, tasks, categories, not args.no_base)
    if missing:
        print(f"\n[probe] **変種が無かった組 {len(missing)} 件**")
        for text in missing:
            print(f"[probe]   {text}")

    # **行の組み立ては `make_validation_suite` に任せる。** `init_states_file` の
    # 決め方（物体を増やす摂動だけ `libero_newobj/<suite>/` を見る等）を
    # 写すと必ずずれる。置き場を渡すかたちなら `libero` の import は起きない。
    from make_validation_suite import _build_rows, _index_init_state_stems, _report

    rows, skipped = _build_rows(
        selected, bddl_root, init_states_root,
        _index_init_state_stems(init_states_root),
    )
    _report(rows, skipped)
    if not rows:
        raise SystemExit("[probe] 1 件も載せられませんでした。上の理由を読むこと")
    if len(rows) < len(selected):
        raise SystemExit(
            f"[probe] **実体が無くて落ちた変種が {len(selected) - len(rows)} 件ある。**"
            "黙って減った一覧では、そのカテゴリの弱さが見えない"
        )
    _report_design(rows, categories)

    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[probe] 書き出し {args.output}（{len(rows)} 行 ≒ "
          f"{len(rows) * 2} 分）")


if __name__ == "__main__":
    main()
