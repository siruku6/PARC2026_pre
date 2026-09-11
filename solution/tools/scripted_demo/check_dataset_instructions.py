#!/usr/bin/env python3
"""学習データの指示文が、**評価で実際に渡る文**と一致しているかを検査する。

**なぜ要るか。**`solution/policy_server.py` の `InstructionNormalizer` は、
採点器から来た文を**配布データの言い方へ寄せてから**モデルへ渡す。
**だから学習データ側は「寄せた先の言い方」で入っていなければ、評価で一度も引かれない。**
bddl の `:language` をそのまま `task` にして足すと、**41 番目の課題**になる。

    採点器が送る文   Pick the akita black bowl on the cookies box and place it on the plate
    正規化した文     pick up the black bowl on the cookie box and place it on the plate  ← モデルが見る
    自作データの文   pick the akita black bowl on the cookies box and place it on the plate  ← 誰も引かない

**自作したデータを足すときは必ず通すこと。**`record` が残すのは
`env.language_instruction`（bddl の書き方）なので、**既定では必ずずれる。**
直し方は `npz_to_lerobot.py build --normalize-instructions`。

**規則は「配布データの 40 種に合わせる」ではない。**正しくは
**「評価でモデルが実際に見る文に合わせる」**である。正規化はしきい値を超えたときしか
書き換えないので、**しきい値を超えない課題では採点器の文がそのまま届く。**

| 課題 | 採点器が送る文 | 正規化 | **学習データが持つべき文** |
|---|---|---|---|
| Track1・Track2 | bddl の言い方 | **寄る**（0.71〜1.00） | **配布データの言い方** |
| Track3（逆向き） | bddl の言い方 | **寄らない**（0.32〜0.40 < 0.65） | **bddl の言い方そのまま** |

**だから逆向きデータの 3 文が配布 40 種に無いのは正しい**（実測・2026-09-12。
`reverse_lerobot_v2` の 3 文は Track3 の bddl と完全一致）。
**`--grader-dir` を渡すと、この「実際に見る文」を作って突き合わせる。**

用法:
    python3 solution/tools/scripted_demo/check_dataset_instructions.py \\
        work/regen/lerobot_batch1 work/artifacts/reverse_lerobot_v2 \\
        --grader-dir ~/src/b_PARC2026_final/compe

Reads:
  - <データセット>/meta/tasks.parquet … `task` 列
  - <データセット>/meta/episodes/**/*.parquet … 課題ごとの本数
  - <--reference>/meta/tasks.parquet  … 配布データの言い方
  - <--grader-dir>/t1/T1_TASKS.csv・t2/T2_TASKS.csv … `instruction` 列
  - <--grader-dir>/t3/assets/bddl_files/libero_t3/*.bddl … `:language`
Generates:
  - 標準出力に、ずれている文と寄せ先
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suite"))


def grader_texts(compe: Path) -> list[str]:
    """採点器が方策へ送る文を集める（Track1・2 の CSV と Track3 の bddl）。"""
    import csv
    import re

    texts: list[str] = []
    for track in ("t1", "t2"):
        path = compe / track / f"T{track[1]}_TASKS.csv"
        if path.exists():
            with path.open() as handle:
                texts += [row["instruction"] for row in csv.DictReader(handle)
                          if row.get("instruction")]
    for path in sorted(compe.glob("t3/assets/bddl_files/**/*.bddl")):
        found = re.search(r"\(:language\s+(.+?)\)\s*$", path.read_text(),
                          re.MULTILINE | re.DOTALL)
        if found:
            texts.append(" ".join(found.group(1).split()))
    # **0 件で黙って進まない。**この道具はコンテナの中から呼ばれることがあり、
    # 配布キットは repo の外にあるので**マウントされていないと静かに 0 になる。**
    # そのまま進むと「基準に無い」と正しく言えなくなる。
    if not texts:
        raise FileNotFoundError(
            f"採点器の文が 1 つも読めない: {compe}\n"
            "  コンテナの中なら配布キットが見えていない。"
            "`--grader-texts FILE`（1 行 1 文）に書き出して渡すか、外で実行すること")
    return texts


def dataset_tasks(root: Path) -> list[str]:
    """`meta/tasks.parquet` の `task` 列を読む。"""
    import pyarrow.parquet as pq

    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        raise FileNotFoundError(f"tasks.parquet が無い: {path}")
    return [str(value) for value in pq.read_table(path)["task"].to_pylist()]


def episode_counts(root: Path) -> dict[str, int]:
    """課題ごとのエピソード本数。**「何種ずれたか」より「何本ずれたか」が効く。**"""
    import pyarrow.parquet as pq

    tasks = dataset_tasks(root)
    counts: dict[str, int] = {}
    for path in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"),
                                 recursive=True)):
        table = pq.read_table(path)
        names = table.column_names
        column = "tasks" if "tasks" in names else ("task_index" if "task_index" in names
                                                   else None)
        if column is None:
            continue
        for value in table[column].to_pylist():
            if isinstance(value, list):
                value = value[0] if value else None
            text = tasks[value] if isinstance(value, int) else str(value)
            counts[text] = counts.get(text, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", type=Path, metavar="DIR")
    parser.add_argument("--reference", type=Path,
                        default=Path("work/artifacts/libero_plus"),
                        help="配布データ（Track1・2 の寄せ先はここにある）")
    parser.add_argument("--grader-dir", type=Path, default=None, metavar="DIR",
                        help="配布キットの `compe`。渡すと**評価で実際に見る文**を作って"
                             "突き合わせる（Track3 のように寄らない文も基準に入る）")
    parser.add_argument("--grader-texts", type=Path, default=None, metavar="FILE",
                        help="採点器が送る文を 1 行 1 文で書いたファイル。"
                             "**配布キットが見えない所（コンテナの中）ではこちらを使う**")
    args = parser.parse_args()

    from check_instruction_normalization import _load_normalizer_class

    normalizer_class, _ = _load_normalizer_class()
    normalizer = normalizer_class()
    reference = set(dataset_tasks(args.reference))
    print(f"基準 {args.reference} … {len(reference)} 種")
    sent: list[str] = []
    if args.grader_dir is not None:
        sent += grader_texts(args.grader_dir)
    if args.grader_texts is not None:
        sent += [line.strip() for line in args.grader_texts.read_text().splitlines()
                 if line.strip()]
    if sent:
        visible = {normalizer.normalize(text)[0] for text in sent}
        outside = sorted(visible - reference)
        print(f"＋ 採点器が送る {len(sent)} 文を正規化したもの … {len(visible)} 種"
              f"（うち配布に無い {len(outside)} 種）")
        for text in outside:
            print(f"    寄らずに素通りする: {text}")
        reference |= visible
    print()

    worst = 0
    for root in args.datasets:
        try:
            tasks = dataset_tasks(root)
        except FileNotFoundError as error:
            print(f"## {root}\n  読めない: {error}\n")
            continue
        counts = episode_counts(root)
        total = sum(counts.values()) or 1
        stray = [task for task in tasks if task not in reference]
        stray_episodes = sum(counts.get(task, 0) for task in stray)
        mark = "OK" if not stray else "ずれている"
        print(f"## {root}\n  {len(tasks)} 種 / {total} 本 … **{mark}** "
              f"（一致しない {len(stray)} 種・{stray_episodes} 本"
              f"＝{100.0 * stray_episodes / total:.0f}%）")
        for task in stray:
            normalized, matched, score, *_ = (list(normalizer.normalize(task)) + [None])[:4]
            fixed = normalized in reference
            print(f"    {counts.get(task, 0):5d} 本  {task}")
            print(f"           → {normalized}")
            if fixed:
                verdict = "基準にある。`--normalize-instructions` で直る"
            elif matched == "unchanged":
                verdict = ("**寄らない文である。**採点器がこの文をそのまま送るなら正しい"
                           "（Track3 の逆向きがこれ）。`--grader-dir` で確かめること")
            else:
                verdict = "**寄せ先が基準に無い。この手本は誰も引かない**"
            print(f"              寄せ方 {matched} {score:.2f} … {verdict}")
        worst = max(worst, len(stray))
        print()
    sys.exit(1 if worst else 0)


if __name__ == "__main__":
    main()
