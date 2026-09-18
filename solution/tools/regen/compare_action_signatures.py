#!/usr/bin/env python3
"""2 つのデータセットの行動列を、エピソードごとの sha256 で突き合わせる。

**数が合うだけでは足りないときに使う。**本数・コマ数・課題数が同じでも、
中身が同じとは限らない。行動列の指紋を並べれば、1 本ずつ同じかが分かる
（`build_l90clean_manifest.py` が埋め戻しの出どころを決めたときと同じ方法）。

    IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/compare_action_signatures.py \\
        --a work/artifacts/l90clean_published \\
        --b work/artifacts/libero_combined_bowl5_initpose_l90clean_rebuilt

**parquet を 1 ファイルずつ読む。**両方を丸ごとメモリに載せると 8 GB では
足りない（21,642 本・343 万コマ × 2）。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def signatures(root: Path) -> dict[int, str]:
    """エピソード番号 → 行動列の sha256。"""
    digests: dict[int, hashlib._Hash] = {}
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise SystemExit(f"parquet が無い: {root}/data")
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "action"])
        episodes = np.asarray(table["episode_index"])
        actions = np.stack(table["action"].to_numpy(zero_copy_only=False))
        # **番号ごとに順番を保ったまま足す。**1 本が複数ファイルに跨ることは
        # 無い前提だが、跨っても順番さえ保てば同じ指紋になる。
        for episode in np.unique(episodes):
            rows = np.ascontiguousarray(actions[episodes == episode], dtype=np.float32)
            digests.setdefault(int(episode), hashlib.sha256()).update(rows.tobytes())
    return {index: digest.hexdigest() for index, digest in digests.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()

    a, b = signatures(args.a), signatures(args.b)
    shared = sorted(set(a) & set(b))
    same = sum(a[i] == b[i] for i in shared)

    print(f"  A {len(a)} 本 / B {len(b)} 本")
    if set(a) != set(b):
        print(f"  **番号が食い違う: A だけ {len(set(a) - set(b))} 本・B だけ {len(set(b) - set(a))} 本**")
    print(f"行動列が一致した本: {same} / {len(shared)}")

    if same != len(shared) or set(a) != set(b):
        differ = [i for i in shared if a[i] != b[i]][:20]
        if differ:
            print(f"  違う番号（先頭 20）: {differ}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
