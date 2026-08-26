#!/usr/bin/env python3
"""指示文の正規化に使う「寄せ先 40 種」と、評価で来る指示文の対応表を作る。

**評価が渡してくる指示文は、学習データの書き方と違う。** 配布された 4 タスク
（`compe/t1/T1_TASKS.csv`）を見ると、渡ってくるのは
`Pick the akita black bowl in the top layer of the wooden cabinet and place it
on the plate` で、学習データの `task` 列は
`pick up the black bowl in the top drawer of the wooden cabinet and place it on
the plate` である。**同じタスクなのに綴りが違う。**

**このずれを、学習データの 40 種のうち最も近いものへ寄せる**のが正規化である。
運営 Q&A No.29 が「小文字化、"Pick the"→"pick up the" のような言い換えの統一」を
名指しで許容し、No.31 が「学習データの指示文集合への埋め込み最近傍を用いた入力
正規化」まで許容している。**最近傍の結果をタスク ID として分岐に使わないこと。**

このスクリプトは 2 つを出す。

| 出すもの | 使い道 |
|---|---|
| **寄せ先 40 種** | `solution/policy_server.py` に貼る（`--emit python`） |
| **対応表** | 正規化の精度を測る（`--emit json` → `check_instruction_normalization.py`） |

    python3 solution/tools/suite/instruction_targets.py                  # 要点だけ
    python3 solution/tools/suite/instruction_targets.py --emit python    # 貼る形で出す
    python3 solution/tools/suite/instruction_targets.py --emit json \
        --output outputs/instruction_norm/pairs.json               # 対応表

**寄せ先は 2 つの独立な資料が一致することで確かめている。** 1 つは
`task_classification.json` の変種名から素のタスク名を取り出したもの
（`make_holdout_split.py` が配布ノートブックと突き合わせて自己検査している）、
もう 1 つは学習データのエピソードのメタデータ（`--episodes-parquet`）である。
**40 種にならなければ、その場で止まる。**

Reads:
  - <libero-plus>/libero/libero/benchmark/task_classification.json（変種 10,030 件）
  - <libero-plus>/libero/libero/bddl_files/**/*.bddl（`(:language ...)` が渡る文）
  - <episodes-parquet>（任意。あれば寄せ先を突き合わせる）
Generates:
  - 標準出力（`--emit summary|python`）
  - <output>（`--emit json`。delivered / target / category / suite の対応表）
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_holdout_split import (  # noqa: E402
    _SUFFIX_PATTERNS,
    base_task,
    instruction,
)

# BDDL の `(:language ...)`。**閉じ括弧までを 1 つ取る。**
_LANGUAGE_RE = re.compile(r"\(:language\s+([^)]*)\)")

# 学習データのタスク数。**この数にならなければ資料の版が変わっている。**
_EXPECTED_TARGETS = 40

# 学習データの `task` 列は小文字と空白だけでできている。
# エピソードのメタデータ（parquet）は文字列を素のまま持っているので、
# **pyarrow が無くても取り出せる。**
_TASK_LIKE_RE = re.compile(r"[a-z][a-z ]{14,}")
_TASK_VERBS = ("pick ", "put ", "open ", "close ", "turn ", "push ", "stack ")


def _bddl_index(libero_plus: Path) -> dict[str, Path]:
    """BDDL を「ファイル名（拡張子なし）→ 実体」で引けるようにする。

    **同じ名前が複数のディレクトリにある**（16,330 ファイルに対して 8,210 種）。
    `libero_mix` は他のスイートと同じ名前を持つため、**名前順で先に来たものを
    採る**。`(:language ...)` は同じ名前なら同じ文なので、どちらでも変わらない。
    """
    root = libero_plus / "libero/libero/bddl_files"
    if not root.is_dir():
        raise FileNotFoundError(f"BDDL の置き場がありません: {root}")

    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.bddl")):
        index.setdefault(path.stem, path)
    if not index:
        raise RuntimeError(f"BDDL が 1 つもありません: {root}")
    return index


def _resolve_bddl(index: dict[str, Path], variant_name: str) -> Path | None:
    """変種の名前から、実際に読まれる BDDL を割り出す。

    **変種名と BDDL の名前は一致しない。** 言い換えの変種は
    `..._language_1_view_0_0_100_0_0_initstate_0` という名前だが、BDDL は
    `..._language_1.bddl` である（`_view_...` と `_initstate_...` は
    カメラと初期姿勢の指定で、BDDL を作り分けない）。
    **観測ノイズ・カメラ・初期姿勢の 3 カテゴリは素の BDDL に落ちる**ので、
    渡る指示文は素のタスクの文になる。

    **接尾辞を後ろから 1 つずつ剥がし、そのたびに実在を見る。**
    これで 10,030 件すべてが解ける（`--emit summary` が件数を出す）。

    Parameters
    ----------
    index : dict[str, Path]
        `_bddl_index` の戻り値。
    variant_name : str
        `task_classification.json` の `name`。

    Returns
    -------
    path : Path | None
        見つかった BDDL。1 つも見つからなければ None。
    """
    current = variant_name
    while True:
        found = index.get(current)
        if found is not None:
            return found
        for pattern in _SUFFIX_PATTERNS:
            stripped = pattern.sub("", current)
            if stripped != current:
                current = stripped
                break
        else:
            return None


def language_of(path: Path) -> str | None:
    """BDDL から、評価が渡す指示文をそのまま取り出す。"""
    matched = _LANGUAGE_RE.findall(path.read_text(errors="replace"))
    if not matched:
        return None
    return " ".join(matched[0].split())


def build_pairs(libero_plus: Path) -> list[dict]:
    """変種 10,030 件それぞれについて「渡る文」と「寄せ先」を並べる。

    Returns
    -------
    pairs : list[dict]
        `delivered`（評価が渡す文）・`target`（学習データの書き方）・
        `category`（摂動の種類）・`suite`・`name`（変種名）を持つ。

    Raises
    ------
    RuntimeError
        BDDL が見つからない変種がある、または `(:language ...)` が無いとき。
        **黙って数を減らさない** —— 減ったまま精度を測ると良く見える。
    """
    classification = libero_plus / "libero/libero/benchmark/task_classification.json"
    if not classification.is_file():
        raise FileNotFoundError(f"分類表がありません: {classification}")

    index = _bddl_index(libero_plus)
    raw = json.loads(classification.read_text(encoding="utf-8"))

    pairs: list[dict] = []
    unresolved: list[str] = []
    for suite, records in raw.items():
        for record in records:
            path = _resolve_bddl(index, record["name"])
            delivered = language_of(path) if path is not None else None
            if delivered is None:
                unresolved.append(record["name"])
                continue
            pairs.append({
                "delivered": delivered,
                "target": instruction(base_task(record["name"])),
                "category": record["category"],
                "suite": suite,
                "name": record["name"],
            })

    if unresolved:
        raise RuntimeError(
            f"BDDL を割り出せない変種が {len(unresolved)} 件あります"
            f"（例 {unresolved[0]}）。_SUFFIX_PATTERNS が LIBERO-plus の版と"
            "合っていません"
        )
    return pairs


def training_instructions(pairs: list[dict]) -> tuple[str, ...]:
    """寄せ先（学習データの指示文）を名前順で返す。

    Raises
    ------
    RuntimeError
        40 種にならなかったとき。
    """
    targets = tuple(sorted({pair["target"] for pair in pairs}))
    if len(targets) != _EXPECTED_TARGETS:
        raise RuntimeError(
            f"寄せ先が {len(targets)} 種になりました（想定 {_EXPECTED_TARGETS} 種）。"
            "LIBERO-plus か学習データの版が変わっています"
        )
    return targets


def _targets_from_parquet(path: Path) -> set[str]:
    """学習データのメタデータから指示文を取り出す（突き合わせ用）。

    **pyarrow を要求しない。** parquet は文字列を素のまま持っているので、
    小文字と空白だけでできた語の並びを拾えば `task` 列の値が出てくる。
    `turn on the stove` のように**他の指示文の一部になっているものは
    拾えない**ので、包含関係にあるものは落として比べる。
    """
    blob = path.read_bytes().decode("latin-1")
    found = {
        match.group(0).strip()
        for match in _TASK_LIKE_RE.finditer(blob)
        if match.group(0).strip().startswith(_TASK_VERBS)
    }
    return {s for s in found if not any(s != other and s in other for other in found)}


def _print_summary(pairs: list[dict], targets: tuple[str, ...]) -> None:
    """何件を何に寄せることになるのかを出す。"""
    by_category = collections.Counter(pair["category"] for pair in pairs)
    delivered = {pair["delivered"]: pair["target"] for pair in pairs}
    ambiguous = collections.defaultdict(set)
    for pair in pairs:
        ambiguous[pair["delivered"]].add(pair["target"])
    clashes = {text: sorted(t) for text, t in ambiguous.items() if len(t) > 1}

    already = sum(
        1 for text, target in delivered.items() if " ".join(text.lower().split()) == target
    )

    print(f"[targets] 変種            {len(pairs)} 件")
    print(f"[targets] 寄せ先          {len(targets)} 種（学習データの指示文）")
    print(f"[targets] 渡ってくる文    {len(delivered)} 種")
    print(f"[targets]   うち寄せ先と一致（小文字化のみ） {already} 種")
    print(f"[targets]   寄せる必要がある               {len(delivered) - already} 種")
    print(f"[targets] 同じ文が 2 つの寄せ先を持つ       {len(clashes)} 種")
    for text, conflict in list(clashes.items())[:3]:
        print(f"[targets]   {text!r} -> {conflict}")
    print("[targets] 摂動の種類ごとの変種の数")
    for category, count in by_category.most_common():
        texts = {pair["delivered"] for pair in pairs if pair["category"] == category}
        print(f"[targets]   {category:<22} {count:>5} 件 / 文 {len(texts):>5} 種")


def _print_python(targets: tuple[str, ...]) -> None:
    """`policy_server.py` に貼る形で出す。"""
    print("# 寄せ先。**学習データ（lerobot/libero_plus）の指示文 40 種そのままである。**")
    print("# 出し方は solution/tools/suite/instruction_targets.py --emit python。")
    print("_TRAINING_INSTRUCTIONS = (")
    for target in targets:
        print(f'    "{target}",')
    print(")")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--libero-plus", type=Path, default=_REPO_ROOT / "work/LIBERO-plus",
        help="LIBERO-plus の clone 先",
    )
    parser.add_argument(
        "--episodes-parquet", type=Path,
        default=_REPO_ROOT / "outputs/tmp/libero_plus_episodes.parquet",
        help="学習データのエピソードのメタデータ（あれば寄せ先を突き合わせる）",
    )
    parser.add_argument(
        "--emit", choices=("summary", "python", "json"), default="summary",
        help="出す形（既定 summary）",
    )
    parser.add_argument(
        "--output", type=Path,
        default=_REPO_ROOT / "outputs/instruction_norm/pairs.json",
        help="--emit json のときの出力先",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pairs = build_pairs(args.libero_plus)
    targets = training_instructions(pairs)

    if args.episodes_parquet.is_file():
        from_data = _targets_from_parquet(args.episodes_parquet)
        # 包含関係で落ちるものがあるので、片側の差だけを見る。
        missing = from_data - set(targets)
        if missing:
            raise RuntimeError(
                "学習データにあって寄せ先に無い指示文があります: "
                f"{sorted(missing)[:3]}"
            )
        print(f"[targets] 突き合わせ      学習データの {len(from_data)} 種すべてが寄せ先にある")
    else:
        print(f"[targets] 突き合わせ      省略（{args.episodes_parquet} が無い）")

    if args.emit == "python":
        _print_python(targets)
        return
    if args.emit == "json":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"targets": list(targets), "pairs": pairs}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[targets] 書き出し        {args.output}（{len(pairs)} 件）")
        return
    _print_summary(pairs, targets)


if __name__ == "__main__":
    main()
