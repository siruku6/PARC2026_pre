#!/usr/bin/env python3
"""学習から外すタスクの一覧を作る（検証専用にするため）。

**手元の検証が本番の代わりにならない原因は、件数ではなく、学習に使ったタスクを
検証にも使っていることである。** 追加学習した重みは手元 84 件で素より上に
見えたが（0.500 対 0.476）、採点環境では 8.8 倍悪かった（0.016 対 0.1409）。
**手元の数字は「覚えた度合い」を測っていた。**

**摂動版は元のタスクの実演データを共有する。** 検証に使う変種の id を学習から
抜いても、同じ指示文の別の摂動版が学習に残るので中身は見たままになる。
**だから指示文（タスク）単位で丸ごと分ける。**

    python3 solution/tools/suite/make_holdout_split.py            # 既定で 8 タスク外す
    python3 solution/tools/suite/make_holdout_split.py --extra 0  # 配布 4 タスクだけ外す
    python3 solution/tools/suite/make_holdout_split.py --dry-run  # 書かずに内容を見る

出力の CSV は 2 か所から読まれる。**同じ 1 枚を両方が読むことで、片方だけ
古くなる事故を防ぐ。**

| 読む側 | 何に使うか |
|---|---|
| `solution/train/train.py --holdout-tasks` | 学習から外す |
| `solution/tools/make_validation_suite.py --holdout` | 検証集合をここに限る |

Reads:
  - <libero-plus>/libero/libero/benchmark/task_classification.json
  - compe/t1/T1_TASKS.csv（配布された 4 タスク）
  - examples/smolvla_libero_spatial_lora.ipynb（**導出の自己検査に使う**）
Generates:
  - compe/val/HOLDOUT_TASKS.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# 摂動の接尾辞。**後ろから繰り返し剥がすと素のタスク名になる。**
# 3 種類ある。`_table_1` のように数字が離れている形、`_level1_sample2` のように
# 数字が語にくっついている形、`_moved` のように数字が付かない形。
_SUFFIX_PATTERNS = (
    re.compile(r"_(?:table|tb|light|noise|add|initstate|view|language)(?:_-?\d+)+$"),
    re.compile(r"_level\d+(?:_sample\d+)?$"),
    re.compile(r"_sample\d+$"),
    re.compile(r"_moved$"),
)

# **本番の評価に含まれるカテゴリ。** 運営 Q&A No.21 の回答は「7 カテゴリすべてが
# 含まれているわけではありません…『同一タスク・同一ドメインにおける、テクスチャ
# などの摂動』に該当すると考えられるものが含まれています」である。
# 配布された 4 タスクもこの 2 カテゴリだけでできている。
_PRODUCTION_CATEGORIES = ("Background Textures", "Light Conditions")

# 元タスク名の先頭に付く場面の名前（`KITCHEN_SCENE3_` など）。
# **落とすとデータセットの `task` 列と同じ文になる。**
# `solution/tools/analysis/decompose_failures.py` の `_instruction_of` が同じ処理をしており、
# そこに「CSV の instruction 列は学習データと綴りが違う」理由が書いてある。
_SCENE_PREFIX_RE = re.compile(r"^[A-Z][A-Z_]*SCENE\d+_")

# 導出の自己検査に使う、配布ノートブックの定数名。
_NOTEBOOK_TASK_LIST = "SPATIAL_TASK_NAMES"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--libero-plus", type=Path, default=_REPO_ROOT / "work/LIBERO-plus",
        help="LIBERO-plus の clone 先",
    )
    parser.add_argument(
        "--t1", type=Path, default=_REPO_ROOT / "compe/t1/T1_TASKS.csv",
        help="配布された 4 タスクの CSV",
    )
    parser.add_argument(
        "--notebook", type=Path,
        default=_REPO_ROOT / "examples/smolvla_libero_spatial_lora.ipynb",
        help="自己検査に使う配布ノートブック",
    )
    parser.add_argument(
        "--extra", type=int, default=4,
        help="配布 4 タスクに加えて外す本数（既定 4。合計 8 タスクになる）",
    )
    parser.add_argument(
        "--output", type=Path, default=_REPO_ROOT / "compe/val/HOLDOUT_TASKS.csv",
        help="出力する CSV",
    )
    parser.add_argument("--dry-run", action="store_true", help="書かずに内容だけ出す")
    return parser.parse_args()


def base_task(variant_name: str) -> str:
    """摂動版の名前から、素のタスク名を取り出す。

    Parameters
    ----------
    variant_name : str
        `task_classification.json` の `name`。
        例 `pick_up_the_black_bowl_..._on_the_plate_table_1`。

    Returns
    -------
    base : str
        接尾辞を剥がした名前。例 `pick_up_the_black_bowl_..._on_the_plate`。
    """
    while True:
        for pattern in _SUFFIX_PATTERNS:
            stripped = pattern.sub("", variant_name)
            if stripped != variant_name:
                variant_name = stripped
                break
        else:
            return variant_name


def instruction(name: str) -> str:
    """素のタスク名を、データセットの `task` 列と同じ書き方に直す。

    **場面の名前を落として、下線を空白に置き換える。**
    `libero_10` の名前は `KITCHEN_SCENE3_turn_on_the_stove_...` の形で、
    **場面の名前を残すとデータセットの `task` 列と一致しない。**
    これが正しいことは `_self_check` が配布ノートブックと突き合わせて確かめる。
    """
    return _SCENE_PREFIX_RE.sub("", name).replace("_", " ")


def _load_variants(libero_plus: Path) -> list[dict]:
    """摂動版の一覧を読む。

    Returns
    -------
    variants : list[dict]
        `category`・`suite`・`instruction` を持つ。
    """
    path = libero_plus / "libero/libero/benchmark/task_classification.json"
    if not path.is_file():
        raise FileNotFoundError(f"分類表が見つかりません: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"分類表の形が想定外です: {type(raw).__name__}")

    variants: list[dict] = []
    for suite, records in raw.items():
        for record in records:
            # **読まれる鍵だけ持つ。** 使わない列を運ぶと、
            # 何が判定に効いているのか分からなくなる。
            variants.append({
                "category": record["category"],
                "suite": suite,
                "instruction": instruction(base_task(record["name"])),
            })
    if not variants:
        raise RuntimeError(f"分類表が空です: {path}")
    return variants


def _self_check(variants: list[dict], notebook: Path) -> None:
    """素のタスク名の導出が正しいかを、配布ノートブックと突き合わせる。

    **配布ノートブックは信用できる一次資料である。** そこに
    `SPATIAL_TASK_NAMES` として libero_spatial の 10 タスクが
    データセットと同じ書き方で並んでいる。導出した 10 件がこれと
    完全一致しなければ、**接尾辞の剥がし方が LIBERO-plus の版と合っていない。**

    Raises
    ------
    RuntimeError
        一致しなかったとき。**黙って続けない。**
    """
    if not notebook.is_file():
        raise FileNotFoundError(f"自己検査に使うノートブックがありません: {notebook}")

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = ""
    for cell in payload.get("cells", []):
        text = "".join(cell.get("source", []))
        if _NOTEBOOK_TASK_LIST in text:
            source = text
            break
    if not source:
        raise RuntimeError(
            f"{notebook} に {_NOTEBOOK_TASK_LIST} が見つかりません"
        )

    trusted = set(re.findall(r'^\s*"([a-z][^"]+)",\s*$', source, re.M))
    derived = {
        entry["instruction"] for entry in variants if entry["suite"] == "libero_spatial"
    }
    if derived != trusted:
        raise RuntimeError(
            "素のタスク名の導出が配布ノートブックと一致しません。"
            f"不足 {sorted(trusted - derived)} / 余り {sorted(derived - trusted)}"
        )
    # **場面の名前が残っていないかを全スイートで見る。** ノートブックで
    # 突き合わせられるのは libero_spatial だけで、そこには `KITCHEN_SCENE3_` の
    # ような接頭辞が 1 つも無い。**libero_10 は全件に付いている**ので、
    # 落とし忘れをここで捕まえる。
    leftover = sorted({
        entry["instruction"] for entry in variants if "SCENE" in entry["instruction"]
    })
    if leftover:
        raise RuntimeError(
            "場面の名前が残った指示文があります"
            f"（{len(leftover)} 種、例 {leftover[0]!r}）。"
            "_SCENE_PREFIX_RE が LIBERO-plus の版と合っていません。"
        )
    print(f"[holdout] 自己検査: libero_spatial の {len(trusted)} 件が一致し、"
          "場面の名前も残っていない")


def _load_t1_bases(t1_csv: Path) -> list[dict]:
    """配布された 4 タスクの、素のタスク名を読む。

    **この 4 つは代理集合の土台である。** 4 タスク × 2 エピソード = 8 で
    採点環境のエピソード数と揃う。**だから学習から外す必要がある。**
    """
    if not t1_csv.is_file():
        raise FileNotFoundError(f"配布 4 タスクの CSV がありません: {t1_csv}")

    with t1_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"配布 4 タスクの CSV が空です: {t1_csv}")

    bases = []
    seen = set()
    for row in rows:
        base = base_task(row["task_id"])
        if base in seen:
            continue
        seen.add(base)
        bases.append({
            "instruction": instruction(base),
            "suite": row["suite"],
            "category": row["category"],
            "source": "t1",
        })
    return bases


def _pick_extra(variants: list[dict], already: set[str], count: int) -> list[dict]:
    """本番と同じ 2 カテゴリから、外すタスクを追加で選ぶ。

    **スイートを順に回して 1 つずつ取る**ので、1 つのスイートに偏らない。
    並べ替えは名前順で、乱数を使わない（**同じ入力なら同じ結果になる**）。

    Parameters
    ----------
    variants : list[dict]
        摂動版の一覧。
    already : set[str]
        すでに外すことが決まっている指示文。
    count : int
        追加で選ぶ本数。

    Returns
    -------
    picked : list[dict]
        選んだタスク。**足りなければ足りるだけ返す。**
    """
    if count <= 0:
        return []

    by_suite: dict[str, set[str]] = {}
    for entry in variants:
        if entry["category"] in _PRODUCTION_CATEGORIES \
                and entry["instruction"] not in already:
            by_suite.setdefault(entry["suite"], set()).add(entry["instruction"])

    # スイートごとの並びを横に並べ、上から順に取ると 1 巡ずつ回った形になる。
    # `zip_longest` が空きを `None` で埋めるので、落とせば本数の違いを吸収する。
    rounds = itertools.zip_longest(
        *(sorted(by_suite[suite]) for suite in sorted(by_suite))
    )
    order = [
        (suite, name)
        for row in rounds
        for suite, name in zip(sorted(by_suite), row)
        if name is not None
    ]

    picked: list[dict] = []
    for suite, name in order[:count]:
        picked.append({
            "instruction": name,
            "suite": suite,
            "category": "/".join(_PRODUCTION_CATEGORIES),
            "source": "extra",
        })
    return picked


def _report(variants: list[dict], holdout: list[dict]) -> None:
    """外すタスクと、学習に残る本数を出す。"""
    bases = {entry["instruction"] for entry in variants}
    held = {entry["instruction"] for entry in holdout}

    print(f"[holdout] 素のタスク 全 {len(bases)} 種 / 外す {len(held)} 種 "
          f"/ 学習に残る {len(bases) - len(held)} 種")
    for entry in holdout:
        print(f"[holdout]   {entry['source']:5s} {entry['suite']:15s} {entry['instruction']}")

    kept_variants = sum(1 for entry in variants if entry["instruction"] not in held)
    print(f"[holdout] 摂動版で数えると 全 {len(variants)} 件 / "
          f"学習に残る {kept_variants} 件")

    in_production = [
        entry for entry in variants
        if entry["instruction"] in held
        and entry["category"] in _PRODUCTION_CATEGORIES
    ]
    print(f"[holdout] 外したタスクのうち、本番と同じ 2 カテゴリの変種は "
          f"{len(in_production)} 件（検証に使える上限）")


def main() -> None:
    args = _parse_args()

    variants = _load_variants(args.libero_plus)
    _self_check(variants, args.notebook)

    holdout = _load_t1_bases(args.t1)
    holdout.extend(_pick_extra(
        variants, {entry["instruction"] for entry in holdout}, args.extra
    ))

    _report(variants, holdout)

    if args.dry_run:
        print("[holdout] --dry-run のため書き込んでいない")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["instruction", "suite", "category", "source"]
        )
        writer.writeheader()
        writer.writerows(holdout)
    print(f"[holdout] 書き出した: {args.output}")
    print("[holdout] 次に: train.py に --holdout-tasks で渡し、"
          "make_validation_suite.py に --holdout で渡す")


if __name__ == "__main__":
    main()
