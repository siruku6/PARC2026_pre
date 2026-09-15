#!/usr/bin/env python3
"""配布データ 14,347 本を「元デモ」の組にまとめ、LIBERO 本家のデモと対応づける。

**なぜ要るか。**接触が起きたかどうかは配布データからは判定できない
（物体の位置が 1 列も無い）。本家の hdf5 には MuJoCo の状態が丸ごと入っていて
物体の姿勢が読めるので、**配布のエピソードが本家のどのデモなのかが分かれば、
判定を本家側で 1 回やって配布側へ配れる。**

**配布の 14,347 本は、1,681 個の物理を見た目だけ変えて焼き直したものである**
（行動列も状態列も、組の中では 1 ビットも違わない。2026-09-09 に実測）。
**だから対応づけも判定も 1,681 回で済む。**

**突き合わせは「前 6 次元がすべて 0 のコマを両側から落としてから」行う。**
配布側は `no_noops` の処理でそういうコマが落ちているが、**落ち方が完全に揃って
いない**（1 課題 50 デモ中 1 つだけ残っていた例がある）。片側にだけ掛けると
その 1 つを取り逃がす。

用法（ホスト側。`.venv` の python で打つ）:
    .venv/bin/python solution/tools/contact_clean/map_demos.py \\
        --dataset work/artifacts/libero_plus \\
        --states work/artifacts/libero_states \\
        --out work/artifacts/libero_states/demo_map.json

Reads:
  - <dataset>/data/chunk-*/file-*.parquet, <dataset>/meta/tasks.parquet
  - <states>/index.json, <states>/<suite>/<task>_demo.npz
Generates:
  - <out> … 組ごとの「配布のエピソード番号 ↔ 本家のデモ」の対応表
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, metavar="DIR",
                        help="配布データセット（work/artifacts/libero_plus）")
    parser.add_argument("--states", type=Path, required=True, metavar="DIR",
                        help="fetch_libero_states.py の出力先")
    parser.add_argument("--out", type=Path, required=True, metavar="JSON",
                        help="対応表の書き出し先")
    return parser.parse_args()


def _drop_noop_frames(actions: np.ndarray) -> np.ndarray:
    """前 6 次元がすべて 0 のコマを落とす。**突き合わせの前に両側へ掛ける。**"""
    return actions[~np.all(actions[:, :6] == 0.0, axis=1)]


def _fingerprint(actions: np.ndarray) -> str:
    """行動列そのものを鍵にする。**丸めない**——一致は完全一致で見る。"""
    return hashlib.md5(_drop_noop_frames(actions).astype(np.float32).tobytes()).hexdigest()


def _load_distributed_clusters(dataset: Path) -> tuple[dict[str, dict], dict[int, str]]:
    """配布データを読み、同じ行動列のエピソードを 1 つの組にまとめる。

    Returns
    -------
    clusters : dict[str, dict]
        鍵は行動列の指紋。値は指示文・エピソード番号の一覧・コマ数。
    instructions : dict[int, str]
        `task_index` から指示文への対応。

    Reads:
      - <dataset>/data/chunk-*/file-*.parquet
      - <dataset>/meta/tasks.parquet
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(dataset / "data" / "chunk-*" / "file-*.parquet")))
    table = pa.concat_tables([pq.read_table(path, columns=["action", "episode_index", "task_index"])
                              for path in files])
    episode_index = np.asarray(table["episode_index"])
    task_index = np.asarray(table["task_index"])
    action = np.asarray(table["action"].to_numpy(zero_copy_only=False).tolist(), dtype=np.float32)

    order = np.argsort(episode_index, kind="stable")
    episode_index, task_index, action = episode_index[order], task_index[order], action[order]
    bounds = np.searchsorted(episode_index, np.arange(int(episode_index.max()) + 2))

    task_table = pq.read_table(dataset / "meta" / "tasks.parquet").to_pydict()
    instructions: dict[int, str] = dict(zip(task_table["task_index"], task_table["task"]))

    clusters: dict[str, dict] = {}
    for number in range(int(episode_index.max()) + 1):
        start, stop = bounds[number], bounds[number + 1]
        key = _fingerprint(action[start:stop])
        entry = clusters.setdefault(key, {
            "instruction": instructions[int(task_index[start])],
            "task_index": int(task_index[start]),
            "episodes": [],
            "frames": int(stop - start),
        })
        entry["episodes"].append(number)
    return clusters, instructions


def _load_source_demos(states_dir: Path) -> dict[str, dict]:
    """本家から抜いた npz を読み、デモごとに行動列の指紋を作る。

    Reads:
      - <states_dir>/index.json
      - <states_dir>/<suite>/<task>_demo.npz
    """
    index = json.loads((states_dir / "index.json").read_text())
    demos: dict[str, dict] = {}
    collisions = 0
    for repo_path, summary in index.items():
        npz_path = states_dir / repo_path.replace(".hdf5", ".npz")
        with np.load(npz_path) as bundle:
            names = sorted({name.split("/")[0] for name in bundle.files},
                           key=lambda name: int(name.split("_")[1]))
            for name in names:
                key = _fingerprint(bundle[f"{name}/actions"])
                if key in demos:
                    collisions += 1
                demos[key] = {"source": repo_path, "demo": name,
                              "instruction": summary["language_instruction"],
                              "frames": int(bundle[f"{name}/actions"].shape[0]),
                              "state_dim": summary["state_dim"]}
    if collisions:
        print(f"  **本家の中で行動列が重なったデモが {collisions} 件あった**")
    return demos


def _report(clusters: dict[str, dict], demos: dict[str, dict]) -> list[dict]:
    """対応づけの結果を印字し、1 組 1 行の表を返す。"""
    matched, unmatched = [], []
    for key, entry in clusters.items():
        source = demos.get(key)
        row = dict(entry, fingerprint=key)
        if source is None:
            unmatched.append(row)
        else:
            row.update(source_file=source["source"], source_demo=source["demo"],
                       source_frames=source["frames"], state_dim=source["state_dim"])
            matched.append(row)

    episodes_matched = sum(len(row["episodes"]) for row in matched)
    episodes_total = sum(len(row["episodes"]) for row in clusters.values())
    print(f"=== 対応づけ ===")
    print(f"  組        {len(matched)} / {len(clusters)}  "
          f"({len(matched) / len(clusters) * 100:.1f}%)")
    print(f"  エピソード {episodes_matched} / {episodes_total}  "
          f"({episodes_matched / episodes_total * 100:.1f}%)")
    print(f"  本家のデモ {len(demos)}  うち使われた {len({r['fingerprint'] for r in matched})}")

    if unmatched:
        by_task: collections.Counter = collections.Counter(
            row["instruction"] for row in unmatched)
        print(f"\n  **対応が付かなかった {len(unmatched)} 組**（指示文ごと・多い順に 10 行）")
        for instruction, count in by_task.most_common(10):
            print(f"    {count:>3d} 組  {instruction[:66]}")

    # 指示文の食い違いも見る。**一致しても指示文が違うなら、対応づけを疑う。**
    mismatched = [row for row in matched
                  if row["instruction"].strip().lower()
                  != demos[row["fingerprint"]]["instruction"].strip().lower()]
    print(f"\n  行動列は一致したが指示文が違う組: {len(mismatched)}")
    for row in mismatched[:5]:
        print(f"    配布『{row['instruction'][:40]}』 ／ 本家『"
              f"{demos[row['fingerprint']]['instruction'][:40]}』")
    return matched + unmatched


def main() -> None:
    args = _parse_args()
    print("=== 配布データを組にまとめる ===")
    clusters, _ = _load_distributed_clusters(args.dataset)
    episodes = sum(len(entry["episodes"]) for entry in clusters.values())
    print(f"  {episodes} 本 → {len(clusters)} 組")

    print("=== 本家のデモを読む ===")
    demos = _load_source_demos(args.states)
    print(f"  {len(demos)} デモ")

    rows = _report(clusters, demos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"\n=== 書いた: {args.out} ===")


if __name__ == "__main__":
    main()
