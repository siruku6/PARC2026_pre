#!/usr/bin/env python3
"""017: 1 エピソードの `action` と `observation.state` をまとめて npz に抜く。

ホスト側で走る（pyarrow が要る。コンテナには入っていない）。
[host_extract_actions.py](host_extract_actions.py) は action だけを抜くが、
**再生の答え合わせには基準の state 列が要る**ので、両方を同じ 1 回の読み出しで
抜いて同じ npz に入れる（同じ parquet を 2 回読まないため）。

出力 npz のキー: `actions_<ep>` (T,7) / `states_<ep>` (T,8) / `length_<ep>`。

    python3 host_extract_episode.py --dataset-root <展開先> --task-index 92 \
        --out work/probe_017/ep_data.npz [--episodes 4061,4062]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow.parquet as pq

STATE_COL = "observation.state"


def read_meta_episodes(root: str):
    d = os.path.join(root, "meta", "episodes", "chunk-000")
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".parquet"):
            rows.extend(pq.read_table(os.path.join(d, fn)).to_pylist())
    return rows


def task_text(root: str, task_index: int) -> str:
    for r in pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pylist():
        if r.get("task_index") == task_index:
            return r["__index_level_0__"]
    raise SystemExit(f"task_index {task_index} が tasks.parquet に無い")


def extract(path: str, ep_id: int):
    """1 エピソードの action と state を frame_index 順で返す。"""
    tbl = pq.read_table(
        path, columns=["episode_index", "frame_index", "action", STATE_COL]
    ).to_pydict()
    order = sorted(
        (i for i, e in enumerate(tbl["episode_index"]) if e == ep_id),
        key=lambda i: tbl["frame_index"][i],
    )
    actions = np.array([tbl["action"][i] for i in order], dtype=np.float32)
    states = np.array([tbl[STATE_COL][i] for i in order], dtype=np.float32)
    return actions, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--task-index", type=int, default=92)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--max-episodes", type=int, default=3)
    args = ap.parse_args()

    meta = read_meta_episodes(args.dataset_root)
    text = task_text(args.dataset_root, args.task_index)
    matches = [r for r in meta if text in (r.get("tasks") or [])]
    print(f"task {args.task_index} ({text!r}): {len(matches)} エピソード")

    by_id = {int(r["episode_index"]): r for r in matches}
    if args.episodes.strip():
        ep_ids = [int(x) for x in args.episodes.split(",") if x.strip()]
        missing = [e for e in ep_ids if e not in by_id]
        if missing:
            raise SystemExit(f"このタスクに無いエピソード: {missing}")
    else:
        ep_ids = [int(r["episode_index"]) for r in matches[: args.max_episodes]]
    print(f"抜くエピソード: {ep_ids}")

    out = {"task_text": np.array(text)}
    for e in ep_ids:
        row = by_id[e]
        path = os.path.join(
            args.dataset_root, "data",
            f"chunk-{int(row['data/chunk_index']):03d}",
            f"file-{int(row['data/file_index']):03d}.parquet")
        actions, states = extract(path, e)
        if len(actions) != int(row["length"]):
            print(f"  warn: ep {e}: {len(actions)} 行 vs meta length {row['length']}")
        out[f"actions_{e}"] = actions
        out[f"states_{e}"] = states
        out[f"length_{e}"] = len(actions)
        print(f"  ep {e}: action {actions.shape} / state {states.shape}  "
              f"eef_pos 初={np.round(states[0, :3], 4)} 終={np.round(states[-1, :3], 4)}")
    np.savez(args.out, **out)
    print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()
