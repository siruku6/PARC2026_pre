#!/usr/bin/env python3
"""検証用スイートの CSV が、評価パイプラインで本当に開けるかを先に確かめる。

**評価を回さずに確かめる**ことが目的である。Colab で 82 分回してから
「4 行目で落ちていた」と分かるのは高すぎる。ここで落ちる行が分かれば、
アップロードする前に直せる。

配布キットと LIBERO-plus の**実際のコード**と同じ手順でパスを組み立てる。
真似ではなく、条件分岐まで写している（写した箇所は各関数に出典を書いた）。

    python3 solution/tools/check_validation_suite.py \\
        --csv work/VAL_TASKS.csv --libero-plus work/LIBERO-plus

`--patched` を付けると、`COLAB_EVAL.md` のセル H が当てるパッチ後の挙動で
確かめる（`parse_obj_of_interest` が `_view_` を切る）。**付けない場合は
配布キットそのままの挙動**なので、どこで落ちるかが分かる。

Reads:
  - <csv>（make_validation_suite.py が書き出したもの）
  - <libero-plus>/libero/libero/bddl_files/<suite>/*.bddl
  - <libero-plus>/libero/libero/init_files/<suite>/*.pruned_init
Generates:
  - 標準出力のみ（落ちる行と理由）
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="検証用スイートの CSV")
    parser.add_argument("--libero-plus", type=Path, required=True,
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--patched", action="store_true",
                        help="セル H のパッチを当てた状態で確かめる")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="先頭この件数だけ見る（--max-tasks と同じ意味）")
    return parser.parse_args()


def _bddl_path_from_register(row: dict, bddl_root: Path) -> Path:
    """`register.py` が組み立てる bddl のパスを再現する。

    出典: `compe/t1/register.py` の `register_t1`。
    `bddl_file=f"{task_id}.bddl"` と `problem_folder=suite` をそのまま結合し、
    `Benchmark.get_task_bddl_file_path` が
    `bddl_files/<problem_folder>/<bddl_file>` を返す。**`_view_` は切らない。**
    """
    return bddl_root / row["suite"] / f"{row['task_id']}.bddl"


def _bddl_path_for_env(bddl_path: Path) -> Path:
    """環境生成時に実際に開かれる bddl を返す。

    出典: `libero/libero/envs/env_wrapper.py` の `OffScreenRenderEnv.__init__`。
    `_view_` と `_initstate_` の両方を含むときだけ `_view_` で切る。
    """
    name = bddl_path.name
    if "_view_" in name and "_initstate_" in name:
        return bddl_path.with_name(name.split("_view_")[0] + ".bddl")
    return bddl_path


def _bddl_path_for_collision(bddl_path: Path, patched: bool) -> Path:
    """衝突判定が開く bddl を返す。

    出典: `pipeline/environment.py` の `parse_obj_of_interest`。
    **配布キットのままだと `_view_` を切らずに開くため、変種名の実体が
    無ければここで `FileNotFoundError` になる。** セル H のパッチを当てると
    `_view_` を切るようになる。
    """
    if not patched:
        return bddl_path
    if not bddl_path.is_file() and "_view_" in bddl_path.name:
        return bddl_path.with_name(bddl_path.name.split("_view_")[0] + ".bddl")
    return bddl_path


def _init_states_path(row: dict, init_root: Path) -> Path:
    """初期状態ファイルのパスを返す。

    出典: セル H のパッチ後の `register.py`。CSV の `init_states_file` 列を
    そのまま使う（列が無ければ従来どおり `<task_id>.pruned_init`）。
    """
    name = row.get("init_states_file") or f"{row['task_id']}.pruned_init"
    return init_root / row["suite"] / name


def _check_row(
    row: dict, bddl_root: Path, init_root: Path, patched: bool
) -> list[str]:
    """1 行について、開けないパスの理由を並べて返す。空なら通る。"""
    problems: list[str] = []
    declared = _bddl_path_from_register(row, bddl_root)

    for label, path in (
        ("環境生成の bddl", _bddl_path_for_env(declared)),
        ("衝突判定の bddl", _bddl_path_for_collision(declared, patched)),
        ("初期状態", _init_states_path(row, init_root)),
    ):
        if not path.is_file():
            problems.append(f"{label}が無い: {path.name}")
    return problems


def _report(rows: list[dict], failures: list[tuple[dict, list[str]]]) -> None:
    """結果をまとめて見せる。**落ちる最初の行を必ず名指しする。**"""
    print(f"\n確かめた行: {len(rows)}  /  落ちる行: {len(failures)}")

    if not failures:
        print("\n**全部開ける。** この CSV なら評価は最後まで走る。")
        return

    first_index = next(
        index for index, row in enumerate(rows, start=1)
        if any(row is failed for failed, _ in failures)
    )
    print(
        f"\n**最初に落ちるのは {first_index} 行目。**"
        f" 例外はトラック全体を止めるので、**{first_index - 1} タスクだけ走って終わる。**"
    )

    # 理由にはファイル名が入るので、**丸めてから数える。**
    # 丸める前に数えると全部 1 件になり、どこが多いのか分からない。
    grouped = collections.Counter(
        (row["category"], reason.split(":")[0])
        for row, reasons in failures for reason in reasons
    )
    print("\nカテゴリ × 理由")
    for (category, reason), count in sorted(grouped.items(), key=lambda p: -p[1]):
        print(f"  {count:4d}  {category:22s} {reason}")

    print("\n例:")
    for row, reasons in failures[:3]:
        print(f"  {row['category']:22s} {row['task_id'][-52:]}")
        for reason in reasons:
            print(f"      {reason}")


def main() -> None:
    args = _parse_args()
    bddl_root = args.libero_plus / "libero/libero/bddl_files"
    init_root = args.libero_plus / "libero/libero/init_files"
    for root in (bddl_root, init_root):
        if not root.is_dir():
            raise FileNotFoundError(f"置き場が見つかりません: {root}")

    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]

    state = "セル H のパッチ後" if args.patched else "配布キットのまま"
    print(f"[check] {args.csv}  ({len(rows)} 行)")
    print(f"[check] 前提: {state}")

    failures = [
        (row, problems)
        for row in rows
        if (problems := _check_row(row, bddl_root, init_root, args.patched))
    ]
    _report(rows, failures)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
