#!/usr/bin/env python3
"""提出した版の点が docs/ の何ファイルで何回書かれているかを数える。

提出結果の記述は実験台帳（`docs/reference/19_weights_catalog.md`）1 か所に寄せる
決まりにした（2026-09-01）。**この道具は、その作業の前後を数字で比べるためにある。**

使い方::

    python3 solution/tools/analysis/count_score_mentions.py            # 全部の版
    python3 solution/tools/analysis/count_score_mentions.py 5.0.1      # 版を指定
    python3 solution/tools/analysis/count_score_mentions.py --files    # ファイル別に並べる

数えているのは**点の文字列**（`0.492` など）であって、文脈は見ていない。
帰結が 2 つある。

* **0 にはならない。**台帳の行と「いまどこにいるか」の 1 数字は残すからである。
  **見るのは「何ファイルに散っているか」で、目安は 3 ファイル以下**
  （台帳・横断分析の `reference/26`・その版を出した日の `talk/`）。
* **偶然の一致が混ざる。**たとえば `docs/reference/08_appearance_spread.md` の `0.208` は
  5.1.0 の点とは無関係な別の測定値である。**残った箇所は中身を見て確かめる。**
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

# 版 → その版だけを指す点の文字列。出典は
# docs/reference/19_weights_catalog.md の「本選の提出と点」と「一覧」である。
SCORES: dict[str, list[str]] = {
    "3.6.0": ["0.184"],
    "3.9.0": ["0.264"],
    "4.2.0": ["0.370", "0.154", "0.485"],
    "4.3.0": ["0.305", "0.207"],
    "4.5.0": ["0.136", "0.035"],
    "4.6.0": [],  # Track3 は 0.000 なので、数字では区別できない
    "4.7.0": ["0.354"],
    "4.8.0": ["0.161"],
    "5.0.1": ["0.492", "0.233"],
    "5.1.0": ["0.332", "0.208"],
    "6.1.0": ["0.397", "0.223"],
}

# 数えない場所。
#   report/         … 提出済み・社外に出した成果物なので書き換えない
#   archive         … 退避済み
#   talk/tmp/       … 一時ファイル
#   inter_sessions/ … セッション間の依頼。**閉じたら消す一時ファイル**であり、
#                     他のディレクトリから指してはいけないと決めてある
#                     （docs/talk/inter_sessions/README.md）。だから正本の候補にならない
SKIP = (
    "docs/report/",
    "archive",
    "docs/talk/tmp/",
    "docs/plan/archive/",
    "docs/talk/inter_sessions/",
)


def is_skipped(path: pathlib.Path) -> bool:
    text = str(path)
    return any(part in text for part in SKIP)


def count(docs: pathlib.Path, numbers: list[str]) -> collections.Counter:
    """版の点の文字列が、どのファイルに何回出るかを数える。"""
    hits: collections.Counter = collections.Counter()
    for path in sorted(docs.rglob("*.md")):
        if is_skipped(path):
            continue
        body = path.read_text(errors="replace")
        total = sum(body.count(number) for number in numbers)
        if total:
            hits[str(path)] = total
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="*", help="数える版（省略すると全部）")
    parser.add_argument("--docs", type=pathlib.Path, default=pathlib.Path("docs"))
    parser.add_argument("--files", action="store_true", help="ファイル別に並べる")
    args = parser.parse_args()

    if not args.docs.is_dir():
        print(f"docs が無い: {args.docs}（リポジトリの直下で打つ）", file=sys.stderr)
        return 1

    versions = args.version or list(SCORES)
    unknown = [v for v in versions if v not in SCORES]
    if unknown:
        print(f"知らない版: {', '.join(unknown)}", file=sys.stderr)
        print(f"知っているのは: {', '.join(SCORES)}", file=sys.stderr)
        return 1

    print("版      ファイル数  のべ回数  点")
    for version in versions:
        numbers = SCORES[version]
        if not numbers:
            print(f"{version:7s}      —         —     数字で区別できない（0.000）")
            continue
        hits = count(args.docs, numbers)
        mark = "  ✅" if len(hits) <= 3 else "  ← 散っている"
        print(
            f"{version:7s} {len(hits):8d} {sum(hits.values()):9d}  "
            f"{' / '.join(numbers)}{mark}"
        )
        if args.files:
            for path, total in hits.most_common():
                print(f"          {total:3d}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
