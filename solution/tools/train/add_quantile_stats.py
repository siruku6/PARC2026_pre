#!/usr/bin/env python3
"""LeRobot v3.0 のデータセットの `meta/stats.json` に q01/q10/q50/q90/q99 を足す。

**なぜ要るか。** MolmoAct2 は `observation.state` と `action` を QUANTILES で正規化する
（`policies/molmoact2/configuration_molmoact2.py` の `normalization_mapping`）。
π0.5 向けに組んだデータセットには min/max/mean/std しか入っていないので、
そのまま学習を起こすと正規化の段で落ちる。

**なぜ公式のスクリプトを使わないか。** 同梱 LeRobot の
`scripts/augment_dataset_quantile_stats.py` は全エピソードを `LeRobotDataset` 経由で読み、
**動画を 1 コマずつ復号する**（`process_single_episode`）。21,642 本・343 万コマでは日をまたぐ。
**MolmoAct2 は画像を IDENTITY で通す**ので、画像の分位点は使われない。
このスクリプトは `data/**/*.parquet` の数値列だけを読み、動画には触らない。

使い方（Docker のイメージの中で打つ。素の python には pyarrow も lerobot も無い）:

    python solution/tools/train/add_quantile_stats.py --root /data

Reads:
    - <root>/data/**/*.parquet の `action` と `observation.state`
Generates:
    - <root>/meta/stats.json（足りない分位点があるときだけ書く。初回は .bak に控える）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets import DEFAULT_QUANTILES

# MolmoAct2 が QUANTILES で正規化する列。画像は IDENTITY なので要らない。
KEYS = ["action", "observation.state"]


def collect_columns(root: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """データセット全体を 1 周して、列ごとに (コマ数, 次元) の配列を返す。"""
    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise SystemExit(f"**{root}/data に parquet が無い**")
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in files:
        table = pq.read_table(path, columns=keys)
        for key in keys:
            # 固定長 list 列は、行ごとの配列に展開せず、バッファのまま (行, 次元) に見る
            column = table.column(key).combine_chunks()
            chunks[key].append(column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), -1))
    return {key: np.concatenate(parts) for key, parts in chunks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="データセットの場所")
    args = parser.parse_args()

    stats_path = args.root / "meta" / "stats.json"
    original = stats_path.read_text()
    stats = json.loads(original)

    missing = []
    for key in KEYS:
        if key not in stats:
            raise SystemExit(f"**{stats_path} に {key} が無い**")
        if "q01" in stats[key]:
            print(f"  {key}: 既にある")
        else:
            missing.append(key)
    if not missing:
        print("  足すものは無い（stats.json は触っていない）")
        return

    for key, data in collect_columns(args.root, missing).items():
        values = np.quantile(data, DEFAULT_QUANTILES, axis=0, overwrite_input=True)
        for quantile, row in zip(DEFAULT_QUANTILES, values, strict=True):
            stats[key][f"q{int(quantile * 100):02d}"] = row.tolist()
        print(f"  {key}: {data.shape[0]:,} コマ × {data.shape[1]} 次元から出した")
        print(f"    q01 {np.round(values[0], 4).tolist()}")
        print(f"    q99 {np.round(values[-1], 4).tolist()}")

    backup = stats_path.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(original)
        print(f"  控えた: {backup}")
    stats_path.write_text(json.dumps(stats))
    print(f"  書いた: {stats_path}")


if __name__ == "__main__":
    main()
