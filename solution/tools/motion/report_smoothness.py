"""作ったデモの指令が配布データと同じ形をしているかを、表 1 つで出す。

**028 の 6 節の 4 条件だけでは足りない。**張り付きを潰した先にできる
「停止の塊」「立ち上がりの遅さ」を見ないと、**別の 1 点への積み替え**を
見逃す（`smoothness.py` の説明）。この道具はその両方を並べて出す。

    python3 solution/tools/motion/report_smoothness.py \\
        --dataset work/artifacts/libero_plus \\
        --npz "028版=work/probe_028/L/npz/*.npz" \\
        --npz "Ruckig=work/probe_ruckig/npz_rk_a/*.npz"

Reads:
  - `--dataset` の LeRobot parquet（基準にする配布データ）
  - `--npz` の glob（`action` を持つ npz。`名前=glob` の形）
Generates:
  - 標準出力に表 1 つ
"""
from __future__ import annotations

import argparse
import glob as globmodule
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion.smoothness import format_table


def _load_npz(pattern: str) -> list[np.ndarray]:
    """glob に当たる npz の `action` を読む。"""
    files = sorted(globmodule.glob(pattern))
    if not files:
        raise FileNotFoundError(f"npz が無い: {pattern}")
    return [np.load(path)["action"].astype(np.float64) for path in files]


def _load_dataset(dataset: Path, count: int) -> list[np.ndarray]:
    """配布データから無作為に `count` 本の `action` を読む。"""
    import pyarrow.parquet as pq

    files = sorted(globmodule.glob(str(dataset / "data" / "chunk-*" / "file-*.parquet")))
    table = pq.read_table(files[0], columns=["action", "episode_index"])
    actions = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float64)
    index = table["episode_index"].to_numpy()
    unique = np.unique(index)
    picked = np.random.default_rng(0).choice(unique, min(count, len(unique)),
                                             replace=False)
    return [actions[index == episode] for episode in picked]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="work/artifacts/libero_plus",
                        help="基準にする配布データ（空文字で読まない）")
    parser.add_argument("--episodes", type=int, default=200,
                        help="配布データから取る本数")
    parser.add_argument("--npz", action="append", default=[], metavar="名前=glob",
                        help="比べる npz の組。何度でも指定できる")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    named: dict[str, list[np.ndarray]] = {}
    if args.dataset:
        named["配布"] = _load_dataset(Path(args.dataset), args.episodes)
    for entry in args.npz:
        name, _, pattern = entry.partition("=")
        named[name] = _load_npz(pattern)

    print(format_table(named))


if __name__ == "__main__":
    main()
