#!/usr/bin/env python3
"""採点器が実際に送る 16 文が、モデルに届くまでにどう書き換わるかを出す。

**提出物に入っている `InstructionNormalizer` そのものを呼ぶ。** 別に書き写した
実装ではないので、`solution/policy_server.py` のしきい値を変えればこの表も動く。

`check_instruction_normalization.py` とは見ている母集団が違う。あちらは
LIBERO-plus の 10,030 件（言い換えを含む）で**傾向**を測る。こちらは
**配布キットに入っている 16 文だけ**を、1 行ずつ人が読んで判定するために並べる。

    python3 solution/tools/suite/audit_grader_instructions.py           # 対応表
    python3 solution/tools/suite/audit_grader_instructions.py --sweep   # しきい値 3 点
    python3 solution/tools/suite/audit_grader_instructions.py --task-id  # 正解と突き合わせ

**16 文は配布の example であって、本番のタスク集合ではない**（`compe/t3/README.md`）。
それでも、採点器が送る文の言い方を直接示す一次資料はこれしかない。

読み方は [docs/reference/36_instruction_rewriting_table.md](../../../docs/reference/36_instruction_rewriting_table.md) にある。
"""

from __future__ import annotations

import argparse
import csv
import difflib
import math
import pathlib
import re
import sys
import types

# 採点器の課題定義。**提出物の外にある**ので、場所は引数で動かせる。
_DEFAULT_COMPE = pathlib.Path.home() / "src/b_PARC2026_final/compe"

# 比べるしきい値。**いま提出済みの zip は Track3 が 0.99・それ以外が 0.65 である。**
_SWEEP_THRESHOLDS: tuple[float, ...] = (0.50, 0.65, 0.99)

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
_LANGUAGE_RE = re.compile(r"^\s*\(:language\s+|\)\s*$")

# `task_id` の末尾に付く摂動の印。**落とすと素の LIBERO タスク名が残る。**
# 印の綴りは `compe/t1/register.py`・`compe/t2/register.py` の `_SUFFIX_RE` に
# 合わせてある（あちらは init_states を引くために同じことをしている）。
_PERTURBATION_SUFFIX_RE = re.compile(
    r"_(?:language|view|light)_[^.]*"
    r"|_(?:table|tb)_\d+"
    r"|_add_\d+"
    r"|_noise_\d+"
    r"|_moved_level\d+_sample\d+"
)


def _load_policy_server_namespace() -> dict[str, object]:
    """提出物の `InstructionNormalizer` を、重みも torch も無しで読み込む。

    `policy_server.py` は冒頭で msgpack や fastapi を import し、後半で方策本体を
    組み立てる。**`class MyPolicy` の手前までを実行すれば正規化器だけが手に入る**
    ので、入れ物だけの偽の module を差してから、その範囲を exec する。

    Returns
    -------
    namespace : dict[str, object]
        `InstructionNormalizer` と `_TRAINING_INSTRUCTIONS` を含む名前空間。
    """
    for name in ("msgpack", "fastapi", "uvicorn", "torch"):
        stub = types.ModuleType(name)
        stub.__getattr__ = lambda attribute: None  # type: ignore[attr-defined]
        sys.modules.setdefault(name, stub)
    for name in ("FastAPI", "Request", "Response"):
        setattr(sys.modules["fastapi"], name, object)

    source_path = _REPOSITORY_ROOT / "solution/policy_server.py"
    source = source_path.read_text()
    namespace: dict[str, object] = {"__name__": "policy_server_head"}
    head = source[: source.index("class MyPolicy")]
    exec(compile(head, str(source_path), "exec"), namespace)
    return namespace


def _grader_lines(compe: pathlib.Path) -> list[tuple[str, str]]:
    """採点器が送る 16 文を、出典（ファイル名と行番号）付きで集める。

    Track1・Track2 は CSV の `instruction` 列がそのまま `Task.language` になる
    （`compe/t1/register.py`）。Track3 は bddl の `(:language …)` である。

    Reads:
      - compe/t1/T1_TASKS.csv, compe/t2/T2_TASKS.csv
      - compe/t3/assets/bddl_files/libero_t3/*.bddl

    Returns
    -------
    lines : list[tuple[str, str]]
        `("T1_TASKS.csv:2", "Pick the akita …")` の並び。
    """
    lines: list[tuple[str, str]] = []
    for track in ("t1", "t2"):
        path = compe / track / f"T{track[1]}_TASKS.csv"
        with path.open() as handle:
            for number, row in enumerate(csv.DictReader(handle), start=2):
                if row.get("instruction"):
                    lines.append((f"{path.name}:{number}", row["instruction"]))

    folder = compe / "t3/assets/bddl_files/libero_t3"
    for path in sorted(folder.glob("*.bddl")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if ":language" in line:
                lines.append((f"{path.name}:{number}", _LANGUAGE_RE.sub("", line)))
    return lines


def _word_diff(before: str, after: str) -> str:
    """語の単位で、消えた語と入った語だけを並べる。

    文を丸ごと 2 つ並べても、どこが動いたかは目で追えない。
    **判定に要るのは入れ替わった語そのもの**なので、それだけを取り出す。
    """
    left, right = before.split(), after.split()
    parts: list[str] = []
    for tag, left_start, left_end, right_start, right_end in difflib.SequenceMatcher(
        a=left, b=right
    ).get_opcodes():
        if tag == "equal":
            continue
        gone = " ".join(left[left_start:left_end]) or "∅"
        came = " ".join(right[right_start:right_end]) or "∅"
        parts.append(f"{gone} -> {came}")
    return " / ".join(parts) if parts else "(語の入れ替えなし)"


def _verdict(score: float, margin: float, is_exact: bool, threshold: float) -> str:
    """そのしきい値でこの文が寄るかどうか。

    **`exact` はしきい値を通らない。** 学習データと同じ文は辞書引きで自分に
    写るだけなので、0.99 にしても素通りしない。
    """
    if is_exact:
        return "exact"
    if score >= threshold and margin >= 0.05:
        return "寄る"
    return "素通り"


def _print_table(namespace: dict[str, object], compe: pathlib.Path) -> None:
    """1 文ごとに、寄せ先の上位 2 件と入れ替わった語を出す。"""
    normalizer = namespace["InstructionNormalizer"]()  # type: ignore[operator]
    targets = namespace["_TRAINING_INSTRUCTIONS"]

    for where, raw in _grader_lines(compe):
        # **normalize() と同じ前処理を通してから測る。** 語の切り出しは小文字しか
        # 拾わないので、生の文のままだと `Pick` が `ick` になって点が狂う。
        cleaned: str = normalizer.canonical(raw)
        ranked = sorted(
            zip(normalizer._lexical_scores(cleaned), targets), reverse=True
        )
        output, how, score = normalizer.normalize(raw)[:3]
        margin: float = ranked[0][0] - ranked[1][0]

        print(f"{where}")
        print(f"  採点器 : {raw}")
        print(f"  モデル : {output}")
        print(f"  判定   : {how} {score:.3f}  (2 番 {ranked[1][0]:.3f} / 差 {margin:.3f})")
        print(f"  2 番手 : {ranked[1][1]}")
        print(f"  語の差 : {_word_diff(cleaned, output)}")
        print()


def _print_sweep(namespace: dict[str, object], compe: pathlib.Path) -> None:
    """しきい値 3 点で、16 文それぞれが寄るか素通りするかを並べる。"""
    normalizer = namespace["InstructionNormalizer"]()  # type: ignore[operator]
    targets = namespace["_TRAINING_INSTRUCTIONS"]

    header = "  ".join(f"{threshold:.2f}" for threshold in _SWEEP_THRESHOLDS)
    print(f"{'出典':<52} {'点':>6} {'差':>6}  {header}")
    changed = 0
    for where, raw in _grader_lines(compe):
        cleaned: str = normalizer.canonical(raw)
        ranked = sorted(
            zip(normalizer._lexical_scores(cleaned), targets), reverse=True
        )
        is_exact: bool = cleaned in normalizer._exact
        score: float = 1.0 if is_exact else ranked[0][0]
        margin: float = ranked[0][0] - ranked[1][0]
        verdicts = [
            _verdict(ranked[0][0], margin, is_exact, threshold)
            for threshold in _SWEEP_THRESHOLDS
        ]
        if len(set(verdicts)) > 1:
            changed += 1
        cells = "  ".join(f"{verdict:<4}" for verdict in verdicts)
        print(f"{where:<52} {score:>6.3f} {margin:>6.3f}  {cells}")
    print(f"\nしきい値で判定が変わる行: {changed} 行")


def _print_task_id_check(namespace: dict[str, object], compe: pathlib.Path) -> None:
    """寄せた先が、採点器が実際に読み込むタスクと同じものかを突き合わせる。

    **これが寄せ先の正しさの答えである。** Track1・Track2 の CSV は、送ってくる
    文（`instruction` 列）と、採点器が読み込む課題（`task_id` 列）を同じ行に
    持っている。`task_id` から摂動の印を落とせば素の LIBERO タスク名になるので、
    **寄せた先の文と 1 文字ずつ比べられる。**

    Track3 の bddl には `task_id` に当たる列が無いので、ここでは見ない。

    Reads:
      - compe/t1/T1_TASKS.csv, compe/t2/T2_TASKS.csv
    """
    normalizer = namespace["InstructionNormalizer"]()  # type: ignore[operator]
    agreed = 0
    total = 0
    for track in ("t1", "t2"):
        path = compe / track / f"T{track[1]}_TASKS.csv"
        with path.open() as handle:
            for number, row in enumerate(csv.DictReader(handle), start=2):
                if not row.get("instruction"):
                    continue
                total += 1
                truth = _PERTURBATION_SUFFIX_RE.sub("", row["task_id"]).replace("_", " ")
                output = normalizer.normalize(row["instruction"])[0]
                same = output == truth
                agreed += same
                print(f"{'一致  ' if same else '不一致'} {path.name}:{number}")
                print(f"    寄せた先 : {output}")
                print(f"    task_id  : {truth}")
    print(f"\n一致 {agreed} / {total}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compe",
        default=str(_DEFAULT_COMPE),
        help="採点器の課題定義が置いてある場所",
    )
    parser.add_argument("--sweep", action="store_true", help="しきい値 3 点で比べる")
    parser.add_argument(
        "--task-id", action="store_true", help="寄せ先を CSV の task_id と突き合わせる"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compe: pathlib.Path = pathlib.Path(args.compe).expanduser().resolve()
    if not compe.is_dir():
        raise SystemExit(f"課題定義が見つかりません: {compe}")
    namespace = _load_policy_server_namespace()
    if args.sweep:
        _print_sweep(namespace, compe)
    elif args.task_id:
        _print_task_id_check(namespace, compe)
    else:
        _print_table(namespace, compe)


if __name__ == "__main__":
    main()
