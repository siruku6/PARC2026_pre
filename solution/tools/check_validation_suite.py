#!/usr/bin/env python3
"""検証用スイートの CSV が、評価パイプラインで本当に開けるかを先に確かめる。

**評価を回さずに確かめる**ことが目的である。Colab で 82 分回してから
「4 行目で落ちていた」と分かるのは高すぎる。ここで落ちる行が分かれば、
アップロードする前に直せる。

配布キットと LIBERO-plus の**実際のコード**と同じ手順でパスを組み立てる。
真似ではなく、条件分岐まで写している（写した箇所は各関数に出典を書いた）。

    python3 solution/tools/check_validation_suite.py \\
        --csv work/VAL_TASKS.csv --libero-plus work/LIBERO-plus \\
        --kit-dir /workspace

**パッチが当たっているかは `--kit-dir` の実物を見て決める。** 旗で受け取ると
「当てたつもりで当たっていない」状態を信じてしまい、この道具の存在意義
（82 分の評価を無駄にしないこと）が消える。

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
import re
import sys
from pathlib import Path

# 隣に置かれた patch_kit_for_variants.py を読む。Colab では `/content/`、
# Docker では `/host/tools/` に 2 つとも並ぶが、**呼び出し元の作業ディレクトリは
# どちらでもない**ため、自分の場所を sys.path に入れてから import する。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_kit_for_variants import is_applied  # noqa: E402

# 配布キットの `compe/t1/register.py` が持つ接尾辞除去。**写しである。**
# パッチを当てていない状態を再現するためだけに使う。
_KIT_SUFFIX_RE = re.compile(r"_light_[^.]*|_table_\d+")

# 初期状態ファイルの拡張子と、物体を増やす摂動だけが置かれている階層。
# `make_validation_suite.py` と同じ値である。
_INIT_SUFFIX = ".pruned_init"
_NEWOBJ_DIR = "libero_newobj"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="検証用スイートの CSV")
    parser.add_argument("--libero-plus", type=Path, required=True,
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--kit-dir", type=Path, required=True,
                        help="配布キットの場所。**パッチが当たっているかを実物で見る**")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="先頭この件数だけ見る（--max-tasks と同じ意味）")
    parser.add_argument("--bddl-root", type=Path, default=None,
                        help="bddl の置き場。省略時は --libero-plus から組み立てる")
    parser.add_argument("--init-states-root", type=Path, default=None,
                        help="初期状態の置き場。省略時は --libero-plus から組み立てる")
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


def _init_states_path(row: dict, init_root: Path, patched: bool) -> Path:
    """初期状態ファイルのパスを返す。

    出典: `compe/t1/register.py` の `_init_states_for`。
    パッチ後は CSV の `init_states_file` 列をそのまま使う。**`/` を含む値は
    置き場からの相対パス**として扱い、スイート名を足さない（物体が増える
    摂動の初期状態は `libero_newobj/<suite>/` にある）。

    **パッチ前は** `<task_id>.pruned_init` から `_SUFFIX_RE` で接尾辞を剥がすので、
    `_view_` を含む変種では存在しない名前になる。ここを patched と
    無関係に CSV 列で計算してしまうと、「未パッチの世界」を再現できない。
    """
    if patched:
        name = row.get("init_states_file") or f"{row['task_id']}.pruned_init"
        return init_root / name if "/" in name else init_root / row["suite"] / name
    stripped = _KIT_SUFFIX_RE.sub("", row["task_id"])
    return init_root / row["suite"] / f"{stripped}.pruned_init"


def _count_bddl_objects(bddl_path: Path) -> int | None:
    """bddl の `(:objects ...)` に並ぶ物体の数を返す。読めなければ None。

    物体を増やす摂動を見分けるために使う。**名前だけでは足りない** ——
    `_add_` という綴りに頼らず、bddl と初期状態の食い違いそのものを見たい。
    """
    if not bddl_path.is_file():
        return None
    found = re.search(r"\(:objects(.*?)\n\s*\)", bddl_path.read_text(), re.S)
    if found is None:
        return None
    # `名前 - 型` の並びなので、`-` でない語だけ数える。
    return len([word for word in found.group(1).split() if word != "-"]) // 2


def _check_init_states_origin(row: dict, bddl_root: Path, init_root: Path,
                              patched: bool) -> str | None:
    """初期状態が、その変種のために作られたものかを確かめる。

    **見るのは 2 つ。順番に意味がある。**

    1. **専用の初期状態が実在するのに使っていないか。** 物体を増やす摂動
       （`_add_` / `_level`）は `libero_newobj/<suite>/` に変種名そのものの
       ファイルを持つ。元タスクの状態で代用すると、物体数が違えば落ち、
       **同じなら落ちずに摂動だけが消える**（後者のほうが厄介である）。
    2. **物体の数が合っているか。** 1 に当てはまらない摂動でも、
       置き場の取り違えで数が食い違えば評価の開始直後にこうなる。

           ValueError: could not broadcast input array
                       from shape (23,) into shape (49,)

    自由度そのものは読まない。初期状態は torch の pickle であり、
    **この道具は torch を要らないことが利点**なので中は開けない。

    Returns
    -------
    problem : str | None
        食い違いの説明。問題なければ None。
    """
    if not patched:
        return None
    name = row.get("init_states_file") or ""
    if "/" in name:
        return None   # 変種そのものの初期状態を指しているので合っている

    dedicated = f"{row['task_id']}{_INIT_SUFFIX}"
    if (init_root / _NEWOBJ_DIR / row["suite"] / dedicated).is_file():
        return (
            f"専用の初期状態があるのに使っていない: "
            f"{_NEWOBJ_DIR}/{row['suite']}/{dedicated} が実在する"
            f"（いまは {name} を指している）"
        )

    variant_bddl = _bddl_path_for_env(_bddl_path_from_register(row, bddl_root))
    origin_stem = name[: -len(_INIT_SUFFIX)] if name.endswith(_INIT_SUFFIX) else ""
    if not origin_stem or origin_stem == variant_bddl.stem:
        return None   # 同じ bddl を見ているので比べる必要が無い

    variant_count = _count_bddl_objects(variant_bddl)
    origin_count = _count_bddl_objects(bddl_root / row["suite"] / f"{origin_stem}.bddl")
    if variant_count is None or origin_count is None or variant_count == origin_count:
        return None

    return (
        f"物体の数が合わない: bddl {variant_bddl.name} は {variant_count} 個だが、"
        f"初期状態 {name} は {origin_count} 個のタスクのもの"
    )


def _check_row(
    row: dict, bddl_root: Path, init_root: Path, patched: bool
) -> list[str]:
    """1 行について、評価が落ちる理由を並べて返す。空なら通る。"""
    declared = _bddl_path_from_register(row, bddl_root)
    problems = [
        f"{label}が無い: {path.name}"
        for label, path in (
            ("環境生成の bddl", _bddl_path_for_env(declared)),
            ("衝突判定の bddl", _bddl_path_for_collision(declared, patched)),
            ("初期状態", _init_states_path(row, init_root, patched)),
        )
        if not path.is_file()
    ]
    mismatch = _check_init_states_origin(row, bddl_root, init_root, patched)
    if mismatch is not None:
        problems.append(mismatch)
    return problems


def _report(
    row_count: int, failures: list[tuple[int, dict, list[str]]]
) -> None:
    """結果をまとめて見せる。**落ちる最初の行を必ず名指しする。**"""
    print(f"\n確かめた行: {row_count}  /  落ちる行: {len(failures)}")

    if not failures:
        print("\n**全部開ける。** この CSV なら評価は最後まで走る。")
        return

    # failures は行の順に積んであるので、先頭が最初に落ちる行である。
    first_index = failures[0][0]
    print(
        f"\n**最初に落ちるのは {first_index} 行目。**"
        f" 例外はトラック全体を止めるので、**{first_index - 1} タスクだけ走って終わる。**"
    )

    # 理由にはファイル名が入るので、**丸めてから数える。**
    # 丸める前に数えると全部 1 件になり、どこが多いのか分からない。
    grouped = collections.Counter(
        (row["category"], reason.split(":")[0])
        for _, row, reasons in failures for reason in reasons
    )
    print("\nカテゴリ × 理由")
    for (category, reason), count in sorted(grouped.items(), key=lambda p: -p[1]):
        print(f"  {count:4d}  {category:22s} {reason}")

    print("\n例:")
    for _, row, reasons in failures[:3]:
        print(f"  {row['category']:22s} {row['task_id'][-52:]}")
        for reason in reasons:
            print(f"      {reason}")


def main() -> None:
    args = _parse_args()
    # **make_validation_suite.py と同じ置き場を見る。** 食い違うと、検査は
    # 通るのに評価が落ちるという最悪の形になる。あちらは libero の設定
    # （`~/.libero/config.yaml`）に従うので、明示指定を受け取れるようにしてある。
    bddl_root = args.bddl_root or args.libero_plus / "libero/libero/bddl_files"
    init_root = args.init_states_root or args.libero_plus / "libero/libero/init_files"
    for root in (bddl_root, init_root):
        if not root.is_dir():
            raise FileNotFoundError(f"置き場が見つかりません: {root}")

    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]

    # **旗で受け取らない。** キットの実物を見てどちらの世界かを決める。
    # 「当てたつもりで当たっていない」を検査側が信じると、この道具の
    # 存在意義（82 分を無駄にしないこと）が消える。
    patched = is_applied(args.kit_dir)
    print(f"[check] {args.csv}  ({len(rows)} 行)")
    print(f"[check] キット: {args.kit_dir}")
    print(f"[check] パッチ: {'当たっている' if patched else '**当たっていない**'}")

    failures = [
        (index, row, problems)
        for index, row in enumerate(rows, start=1)
        if (problems := _check_row(row, bddl_root, init_root, patched))
    ]
    _report(len(rows), failures)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
