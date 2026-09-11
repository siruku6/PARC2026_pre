#!/usr/bin/env python3
"""配布データから、**作り直した課題のエピソードを外した**残し一覧を作る。

**なぜ「外すだけ」ではないのか。**汚れている手本を外すだけの手（手 F）は
606 ロールアウトで否定されている——**引き算は汚れを消すが、正解を教えない**
（`docs/plan/23_data_regeneration_design.md` の 2 節）。ここで外すのは
**代わりに作り直した手本を入れる課題だけ**である。

**汚れている本だけでなく、その課題の全部を外す。**理由は、**清潔に見える本も
余裕が数 mm しかない掴み方をしている**ためである——`next_to_the_cookie_box` の
人の手本は指がクッキー箱から 2 mm の所を通しており、5/20 は 1 mm 未満で
収まるが、**方策はその精度を再現できない**（評価での箱の変位は中央 17.3 mm。
`docs/plan/24_contact_avoidance_redesign.md` の 3 節）。

用法:
    python3 solution/tools/regen/make_replace_keep.py \\
        --root work/artifacts/libero_plus \\
        --out work/artifacts/libero_states/keep_without_bowl5.json

Reads:
  - <root>/meta/episodes/**/*.parquet（`episode_index` と `tasks`）
Generates:
  - <out> … `{"episodes": [番号, …]}`。`contact_clean/rebuild_dataset.py` へ渡す
  - 標準出力 … 課題ごとの外した本数
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPLACED = (
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
)
"""作り直した 5 課題の指示文。**`RimPickPlace` で手本を作った課題と一致させる。**"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="FILE")
    parser.add_argument("--task", action="append", metavar="TEXT",
                        help="外す課題の指示文。**省くと既定の 5 課題**")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import pyarrow.parquet as pq

    drop = set(args.task or REPLACED)
    files = sorted(glob.glob(str(args.root / "meta" / "episodes" / "**" / "*.parquet"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"episodes の parquet が無い: {args.root}")

    keep: list[int] = []
    dropped: dict[str, int] = {}
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "tasks"])
        for index, tasks in zip(table.column("episode_index").to_pylist(),
                                table.column("tasks").to_pylist()):
            # `tasks` は 1 本につき 1 つだが、版によって list で入る。
            text = tasks[0] if isinstance(tasks, list) else tasks
            if text in drop:
                dropped[text] = dropped.get(text, 0) + 1
            else:
                keep.append(int(index))

    keep.sort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"episodes": keep}))
    total = len(keep) + sum(dropped.values())
    print(f"全 {total} 本  →  残す {len(keep)} 本・外す {sum(dropped.values())} 本")
    for text in sorted(drop):
        print(f"  {dropped.get(text, 0):>5} 本  {text}")
    if len(drop) != len(dropped):
        missing = sorted(drop - set(dropped))
        print(f"\n!! 指示文が見つからない課題がある（綴りを確かめること）: {missing}")
    print(f"\n=== {args.out} ===")


if __name__ == "__main__":
    main()
