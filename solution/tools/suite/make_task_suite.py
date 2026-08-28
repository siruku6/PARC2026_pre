#!/usr/bin/env python3
"""指示文を 1 つ指定して、そのタスクの variant だけの評価用 CSV を作る。

**36 件では答えられない問いがあるために作った。** scripted policy で手本を作った
`put both moka pots on the stove` は、**実演が 146 本すべて 341 ステップ以上**で
300 ステップの打ち切りでは解けないため、`make_eval_split.py` の 36 件に入らない。
**つまり「scripted policy を混ぜたら解けるようになったか」を測る場所が無い。**

**このスクリプトは、そのタスクの variant だけを並べた小さな一覧を作る。**
学習の前後で同じ一覧を回せば、**そのタスクだけの成否を直接比べられる。**

    # モカポットのタスクを 8 件（本番と同じ 2 カテゴリから）
    python3 solution/tools/suite/make_task_suite.py \
        --instruction "put both moka pots on the stove" \
        --count 8 --output compe/val/MOKA_TASKS.csv

    # 回す（1 件 2 分ほど）
    EVAL_CSV=compe/val/MOKA_TASKS.csv bash solution/tools/eval/run_docker_eval.sh --detach

**`make_validation_suite.py` は使えない。** あれは `libero` パッケージを
import して初期状態のパスを組み立てるので、**手元（macOS）では動かない。**
こちらは分類表と BDDL しか読まないので、**Docker が要らない。**

**選び方に乱数を使わない。** 分類表に並んでいる順で等間隔に取るので、
同じ引数なら同じ CSV になる。

Reads:
  - <libero-plus>/libero/libero/benchmark/task_classification.json
  - <libero-plus>/libero/libero/bddl_files/**/*.bddl（渡す指示文を取る）
Generates:
  - <output>（`compe/t1/T1_TASKS.csv` と同じ列。評価にそのまま渡せる）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruction_targets import _bddl_index, _resolve_bddl, language_of  # noqa: E402
from make_holdout_split import base_task, instruction  # noqa: E402

# **本番に含まれる摂動**（運営 Q&A No.21 の回答と、配布された 4 タスク）。
_PRODUCTION_CATEGORIES = ("Background Textures", "Light Conditions")

# `T1_TASKS.csv` と同じ列。`compe/t1/register.py` が読む。
_COLUMNS = ("task_num", "task_id", "instruction", "libero_plus_id",
            "suite", "category", "difficulty_level")


def _spread(items: list, count: int) -> list:
    """並び順を保ったまま、両端を含めて等間隔に count 個取る。"""
    if count >= len(items):
        return list(items)
    if count <= 1:
        return [items[len(items) // 2]]
    positions = [round(rank * (len(items) - 1) / (count - 1)) for rank in range(count)]
    return [items[position] for position in positions]


def collect_variants(libero_plus: Path, wanted: str,
                     categories: tuple[str, ...]) -> list[dict]:
    """指定した指示文の variant を、分類表から集める。

    Parameters
    ----------
    libero_plus : Path
        LIBERO-plus の clone 先。
    wanted : str
        学習データの書き方の指示文（例 `put both moka pots on the stove`）。
    categories : tuple[str, ...]
        残す摂動の種類。空なら全部。

    Returns
    -------
    variants : list[dict]
        `name`・`suite`・`category`・`difficulty_level`・`libero_plus_id`。
    """
    path = libero_plus / "libero/libero/benchmark/task_classification.json"
    if not path.is_file():
        raise FileNotFoundError(f"分類表がありません: {path}")

    found: list[dict] = []
    for suite, records in json.loads(path.read_text(encoding="utf-8")).items():
        for record in records:
            if instruction(base_task(record["name"])) != wanted:
                continue
            if categories and record["category"] not in categories:
                continue
            found.append({
                "name": record["name"],
                "suite": suite,
                "category": record["category"],
                "difficulty_level": record.get("difficulty_level", ""),
                "libero_plus_id": record["id"],
            })
    return found


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instruction", required=True,
                        help="学習データの書き方の指示文（完全一致）")
    parser.add_argument("--libero-plus", type=Path,
                        default=_REPO_ROOT / "work/LIBERO-plus",
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--count", type=int, default=8, help="作る件数")
    parser.add_argument("--categories", nargs="*", default=list(_PRODUCTION_CATEGORIES),
                        help="残す摂動の種類。空文字を 1 つ渡すと全部")
    parser.add_argument("--bddl-root", type=Path, default=None,
                        help="bddl の置き場。既定は <libero-plus>/libero/libero/bddl_files")
    parser.add_argument("--init-states-root", type=Path, default=None,
                        help="初期状態の置き場。既定は <libero-plus>/libero/libero/init_files")
    parser.add_argument("--output", type=Path, required=True, help="出力する CSV")
    parser.add_argument("--dry-run", action="store_true", help="書かずに内容だけ出す")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    categories = tuple(name for name in args.categories if name)
    variants = collect_variants(args.libero_plus, args.instruction, categories)
    if not variants:
        raise SystemExit(
            f"[suite] variant が 1 つも見つかりません: {args.instruction!r}"
            f"（摂動 {categories or '全部'}）。"
            "**指示文は学習データの綴りと完全一致でなければならない**"
        )

    chosen = _spread(sorted(variants, key=lambda item: item["name"]), args.count)

    # **行の組み立ては `make_validation_suite` に任せる。**
    # 自前で組んでいたら `init_states_file` が抜け、評価が接尾辞を正規表現で
    # 剥がして初期状態を探しにいって「初期状態が無い」で 8 件とも落ちた
    # （`KITCHEN_SCENE8_..._light_11.pruned_init` は存在しない。実在するのは
    # 接尾辞を剥がした `KITCHEN_SCENE8_....pruned_init` である）。
    # **置き場の解決は 1 か所に置く**（`make_eval_split.py` の 306 行と同じ方針）。
    #
    # `libero` パッケージの import は `_resolve_libero_path` の中だけなので、
    # **置き場を渡すかたちなら手元（macOS）でも動く。**
    from make_validation_suite import _build_rows, _index_init_state_stems, _report

    bddl_root = args.bddl_root or args.libero_plus / "libero/libero/bddl_files"
    init_states_root = (args.init_states_root
                        or args.libero_plus / "libero/libero/init_files")
    for root in (bddl_root, init_states_root):
        if not root.is_dir():
            raise SystemExit(f"[suite] 置き場が見つかりません: {root}")

    # `_build_rows` は `id` を読む。こちらは `libero_plus_id` で持っているので渡し直す。
    selected = [dict(item, id=item["libero_plus_id"]) for item in chosen]
    rows, skipped = _build_rows(
        selected, bddl_root, init_states_root,
        _index_init_state_stems(init_states_root),
    )

    print(f"[suite] {args.instruction!r}")
    print(f"[suite]   候補 {len(variants)} 件 → {len(chosen)} 件を選んだ")
    _report(rows, skipped)
    if not rows:
        raise SystemExit("[suite] 1 件も載せられませんでした。上の理由を読むこと")
    if len(rows) < len(chosen):
        raise SystemExit(
            f"[suite] **実体が無くて落ちた variant が {len(chosen) - len(rows)} 件ある。**"
            "黙って減った一覧は使わない。--count を増やして選び直すこと"
        )
    print(f"[suite]   渡る指示文: {rows[0]['instruction']!r}")
    for row in rows:
        print(f"[suite]   {row['category']:<20} {row['task_id']}")
        print(f"[suite]     初期状態 {row['suite']}/{row['init_states_file']}")

    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        # **列は `_build_rows` が決めたものをそのまま使う。**
        # `register.py` は `DictReader` で読むので、列が増えても困らない。
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[suite] 書き出し {args.output}")


if __name__ == "__main__":
    main()
