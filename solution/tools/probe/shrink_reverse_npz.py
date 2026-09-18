#!/usr/bin/env python3
"""逆向きの記録 npz を、`divergence.py` が実際に読む分だけに縮める。

**なぜ要るか。** `work/probe_rot2` は 23 GB あるが、その大半は
**1 度も読まれない画像**である。`divergence.py` が npz から読むのは

    state・action・instruction・steps ＋ front の先頭 1 コマだけ

で、**`wrist` は 1 度も読まない**（`grep 'recorded\\["' divergence.py` で確かめられる）。
**縮めても探りの結果は 1 ビットも変わらない。**

    python3 solution/tools/probe/shrink_reverse_npz.py \\
        work/probe_rot2/book/reverse_npz_book work/probe_rot2_small/book
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KEEP = ("state", "action", "success", "steps", "instruction")


def shrink(source: Path, destination: Path) -> tuple[int, int, int]:
    """`source` の npz を `destination` へ書き直し、(本数, 元のバイト, 後のバイト) を返す。"""
    destination.mkdir(parents=True, exist_ok=True)
    before = after = count = 0
    for path in sorted(source.glob("*.npz")):
        with np.load(path, allow_pickle=True) as recorded:
            payload = {key: recorded[key] for key in KEEP if key in recorded}
            # **先頭 1 コマだけ残す。**`front[0]` は記録と再生の画が合っているかの
            # 確認（`frame0_absdiff`）にしか使われない。
            if "front" in recorded:
                payload["front"] = recorded["front"][:1]
        target = destination / path.name
        np.savez_compressed(target, **payload)
        before += path.stat().st_size
        after += target.stat().st_size
        count += 1
    return count, before, after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="元の npz ディレクトリ")
    parser.add_argument("destination", type=Path, help="書き出し先")
    args = parser.parse_args()

    count, before, after = shrink(args.source, args.destination)
    print(f"{args.source} -> {args.destination}")
    print(f"  {count} 本  {before / 1e9:.2f} GB -> {after / 1e9:.3f} GB"
          f"（{before / after:.0f} 分の 1）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
