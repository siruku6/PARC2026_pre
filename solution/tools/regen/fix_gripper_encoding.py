#!/usr/bin/env python3
"""指の指令 `action[6]` を {0,1} の書き方から ±1（robosuite の書き方）に直す。

**なぜ要るか。**配布の `libero_combined_20hz` は 2 つのデータの足し算で、指の書き方が違う
（`docs/reference/25_dataset_inventory.md` の 8.7 節）。

| 部分 | 開く | 閉じる |
|---|---|---|
| `libero_plus`（後ろ 14,347 本） | −1 | +1 |
| 標準スイートの元データ（先頭 5,186 本） | **1** | **0** |

シミュレータは符号しか見ない（正で閉じる・0 で動かない）ので、{0,1} のまま学習すると
「開く」に逆向きの正解が付き、方策は掴み損ねる（`work/regen/videos_gripper_mix/`・9/14）。

**直し方は `a → 1 − 2a`**（1 → −1 開く、0 → +1 閉じる）。

**どのエピソードを直すかは「負の値が 1 つでも出るか」で決める。**番号の範囲で分けてはいけない
——開いたまま終わる手本は +1 だけ・−1 だけになり、範囲の境目をまたいで散らばっている
（8.7 節の注意）。±1 系で「+1 だけ」の手本はこのデータに 1 本も無いことを確かめてあるので、
**負の値が無いエピソードは {0,1} 系とみなしてよい。**

**書き換えるのは 3 か所。**動画には触らない。

1. `data/**/*.parquet` の `action` 列
2. `meta/episodes/**/*.parquet` の `stats/action/*`（エピソードごとの統計。merge がこれを足し合わせる）
3. `meta/stats.json` の `action`（2 から組み直す）

**その場で書き換える。**`rebuild_dataset.py` の出力（data と meta は新しいファイル、動画だけハードリンク）
に対して打つこと。**展開した配布データそのものに打つと、元が壊れる**——だから data と meta が
ハードリンクになっていたら止める。

用法:
    python3 solution/tools/regen/fix_gripper_encoding.py --root work/regen/combined/head

Reads / Generates:
  - <root>/data/**/*.parquet・<root>/meta/episodes/**/*.parquet・<root>/meta/stats.json（書き換える）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, metavar="DIR")
    return parser.parse_args()


def _refuse_hardlinks(files: list[Path]) -> None:
    linked = [f for f in files if f.stat().st_nlink > 1]
    if linked:
        raise SystemExit(f"ハードリンクのファイルがある（元のデータまで書き換わる）: {linked[0]} ほか {len(linked) - 1} 個")


def _column(table: pa.Table, name: str) -> np.ndarray:
    return np.stack(table.column(name).to_numpy(zero_copy_only=False)).astype(np.float32)


def main() -> None:
    args = _parse_args()
    data_files = sorted((args.root / "data").rglob("*.parquet"))
    meta_files = sorted((args.root / "meta" / "episodes").rglob("*.parquet"))
    _refuse_hardlinks(data_files + meta_files)

    # 1. data：エピソードごとに負の値の有無を見て、{0,1} 系だけ直す。
    fixed: set[int] = set()
    per_episode: dict[int, np.ndarray] = {}
    for path in data_files:
        table = pq.read_table(path)
        action = _column(table, "action")
        episode = table.column("episode_index").to_numpy()
        for index in np.unique(episode):
            rows = episode == index
            if not (action[rows, 6] < 0).any():
                values = set(np.unique(action[rows, 6]).tolist())
                if not values <= {0.0, 1.0}:
                    raise SystemExit(f"エピソード {index} の指の値が {{0,1}} でも ±1 でもない: {sorted(values)[:5]}")
                action[rows, 6] = 1.0 - 2.0 * action[rows, 6]
                fixed.add(int(index))
            per_episode[int(index)] = action[rows]
        column = pa.array(list(action), type=table.schema.field("action").type)
        table = table.set_column(table.schema.get_field_index("action"), "action", column)
        pq.write_table(table, path)

    # 2. meta/episodes：直したエピソードの `stats/action/*` を実データから計算し直す。
    for path in meta_files:
        table = pq.read_table(path)
        indices = table.column("episode_index").to_pylist()
        updates = {key: table.column(f"stats/action/{key}").to_pylist()
                   for key in ("min", "max", "mean", "std")}
        for row, index in enumerate(indices):
            if index in fixed:
                values = per_episode[index].astype(np.float64)
                updates["min"][row] = values.min(axis=0).tolist()
                updates["max"][row] = values.max(axis=0).tolist()
                updates["mean"][row] = values.mean(axis=0).tolist()
                updates["std"][row] = values.std(axis=0).tolist()
        for key, rows in updates.items():
            name = f"stats/action/{key}"
            position = table.schema.get_field_index(name)
            table = table.set_column(position, name, pa.array(rows, type=table.schema.field(name).type))
        pq.write_table(table, path)

    # 3. stats.json：エピソードごとの統計から組み直す（`rebuild_dataset._write_stats` と同じ式）。
    table = pa.concat_tables([pq.read_table(p) for p in meta_files])
    counts = np.array([np.asarray(x, dtype=np.float64).reshape(-1)[0]
                       for x in table.column("stats/action/count").to_pylist()])
    means = np.array(table.column("stats/action/mean").to_pylist(), dtype=np.float64)
    stds = np.array(table.column("stats/action/std").to_pylist(), dtype=np.float64)
    total = counts.sum()
    mean = (means * counts[:, None]).sum(axis=0) / total
    variance = ((stds ** 2 + (means - mean) ** 2) * counts[:, None]).sum(axis=0) / total
    stats_path = args.root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"].update({
        "min": np.array(table.column("stats/action/min").to_pylist()).min(axis=0).tolist(),
        "max": np.array(table.column("stats/action/max").to_pylist()).max(axis=0).tolist(),
        "mean": mean.tolist(), "std": np.sqrt(variance).tolist()})
    stats_path.write_text(json.dumps(stats, indent=4))

    # 門：全エピソードの指の値が ±1 だけになったか。
    left = sorted({float(v) for values in per_episode.values() for v in np.unique(values[:, 6])})
    print(f"直したエピソード {len(fixed)} / {len(per_episode)} 本・指の値に残った種類 {left}")
    if not set(left) <= {-1.0, 1.0}:
        raise SystemExit("**±1 以外の値が残った**")
    print(f"action の平均 {np.round(mean, 4).tolist()}")


if __name__ == "__main__":
    main()
