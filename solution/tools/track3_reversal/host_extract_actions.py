#!/usr/bin/env python3
"""ホスト側で tar 展開先から action 列を抽出する（parc2026-final 画像内での
pyarrow 依存を無くすため）。pyenv の python3（pyarrow 25 入りのもの）で実行する。

    python3 host_extract_actions.py \
      --dataset-root /home/ubuntu/src/train_data/libero_combined_20hz_extracted \
      --task-text "pick up the book on the left and place it on top of the shelf" \
      --out /tmp/t3c1_actions.npz \
      [--episodes 123,456,789]   # 未指定なら先頭 N 本（既定 3）

出力 npz のキー: actions_<ep> (T,7) float32 / length_<ep>。
reversal_probe.py の --actions-npz へ渡す。

`meta/episodes` の `tasks` 列は**指示文のリスト**（task_index ではない）なので、
タスクの絞込みは `meta/tasks.parquet` の task_index→指示文 変換を挟む。
また、v3.0 の meta には `is_success` 列が無く、成功エピソードの絞り込みは
できず（`--episodes` で明示するか先頭 N 本になる）。
エピソードが載る data ファイルは `data/chunk_index` / `data/file_index` から
特定できるため、全 data parquet の走査はしない。
"""

import argparse
import os

import numpy as np
import pyarrow.parquet as pq


def _read_meta_episodes(dataset_root: str):
    meta_dir = os.path.join(dataset_root, "meta", "episodes", "chunk-000")
    rows = []
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".parquet"):
            continue
        rows.extend(pq.read_table(os.path.join(meta_dir, fn)).to_pylist())
    return rows


def _task_text(dataset_root: str, task_index: int) -> str:
    """tasks.parquet から task_index の指示文を取る。"""
    for r in pq.read_table(os.path.join(dataset_root, "meta", "tasks.parquet")).to_pylist():
        if r.get("task_index") == task_index:
            return r["__index_level_0__"]
    raise SystemExit(f"task_index {task_index} が tasks.parquet に無い")


def _pick_data_file(dataset_root: str, row: dict) -> str:
    """meta 行からそのエピソードが載る data parquet のパスを解決する。"""
    chunk = f"chunk-{int(row['data/chunk_index']):03d}"
    fn = f"file-{int(row['data/file_index']):03d}.parquet"
    path = os.path.join(dataset_root, "data", chunk, fn)
    if not os.path.exists(path):
        raise SystemExit(f"meta が見る data ファイルが無い: {path}")
    return path


def _extract_from_file(path: str, ep_id: int, length: int) -> np.ndarray:
    """1 つの data parquet から指定エピソードの action 列を抽出する。

    v3.0 では 1 エピソードのフレームは 1 つの data ファイルに収まる
    （meta は 1 エピソード = 1 ファイルで記録する）。
    """
    b = pq.read_table(
        path, columns=["episode_index", "frame_index", "action"]
    ).to_pydict()
    frames = b["frame_index"]
    keep = sorted(
        (i for i, e in enumerate(b["episode_index"]) if e == ep_id),
        key=lambda i: frames[i],
    )
    actions = np.array([b["action"][i] for i in keep], dtype=np.float32)
    if len(actions) != length:
        print(
            f"warn: episode {ep_id}: {len(actions)} frames vs meta length {length}"
        )
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument(
        "--task-index", type=int, default=92,
        help="tasks.parquet の task_index（既定 92 = 黄本の順方向 STUDY_SCENE4）",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="", help="コンマ区切り（未指定なら先頭 N 本）")
    ap.add_argument("--max-episodes", type=int, default=3)
    args = ap.parse_args()

    meta = _read_meta_episodes(args.dataset_root)
    text = _task_text(args.dataset_root, args.task_index)
    matches = [
        r for r in meta
        if text in (r.get("tasks") or [])
    ]
    print(f"task {args.task_index} ({text!r}): {len(matches)} episodes", flush=True)

    # 1) 候補エピソード
    if args.episodes.strip():
        by_id = {int(r["episode_index"]): r for r in matches}
        ep_ids = [int(x) for x in args.episodes.split(",") if x.strip()]
        for e in ep_ids:
            if e not in by_id:
                raise SystemExit(f"episode {e} がこのタスクに無い")
    else:
        ep_ids = [int(r["episode_index"]) for r in matches[: args.max_episodes]]
        for r in matches[: args.max_episodes]:
            print(
                f"episode {r['episode_index']}: length={r['length']} "
                f"data file chunk-{r['data/chunk_index']:03d}/"
                f"file-{r['data/file_index']:03d}"
            )
    print(f"selected episodes: {ep_ids}", flush=True)

    # 2) 各エピソードが載る data ファイルだけを読む
    by_id = {int(r["episode_index"]): r for r in matches}
    out = {}
    for e in ep_ids:
        row = by_id[e]
        path = _pick_data_file(args.dataset_root, row)
        print(f"reading {path}", flush=True)
        actions = _extract_from_file(path, e, int(row["length"]))
        print(f"episode {e}: {len(actions)} action frames, last={actions[-1]!r}")
        out[f"actions_{e}"] = actions
        out[f"length_{e}"] = len(actions)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()