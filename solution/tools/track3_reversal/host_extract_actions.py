#!/usr/bin/env python3
"""ホスト側で tar 展開先から action 列を抽出する（parc2026-final 画像内での
pyarrow 依存を無くすため）。pyenv の python3（pyarrow 25 入りのもの）で実行する。

    python3 host_extract_actions.py \
      --dataset-root /home/ubuntu/src/train_data/libero_combined_20hz_extracted \
      --task-index 92 \
      --out /tmp/t3c1_actions.npz \
      [--episodes 123,456,789]   # 未指定なら成功フラグ付きを先頭 3 本

出力 npz のキー: actions_<ep> (T,7) float32 / length_<ep>。
reversal_probe.py の --actions-npz へ渡す。
"""

import argparse
import os

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--task-index", type=int, default=92)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="", help="コンマ区切り（未指定なら成功 3 本）")
    ap.add_argument("--max-episodes", type=int, default=3)
    args = ap.parse_args()

    # 1) 候補エピソード
    lens = {}
    if args.episodes.strip():
        ep_ids = [int(x) for x in args.episodes.split(",") if x.strip()]
    else:
        eps = []
        meta_dir = os.path.join(args.dataset_root, "meta", "episodes", "chunk-000")
        for fn in sorted(os.listdir(meta_dir)):
            if not fn.endswith(".parquet"):
                continue
            for r in pq.read_table(os.path.join(meta_dir, fn)).to_pylist():
                if r.get("task_index") == args.task_index and r.get("is_success"):
                    eps.append((int(r["episode_index"]), int(r["length"])))
                if len(eps) >= args.max_episodes:
                    break
            if len(eps) >= args.max_episodes:
                break
        ep_ids = [e for e, _ in eps]
        lens = {e: l for e, l in eps}
        for e, l in eps:
            print(f"episode {e}: length={l}")
        lens = {e: l for e, l in eps}
    print(f"selected episodes: {ep_ids}", flush=True)

    # 2) data parquet を全走査して各 episode の action 列を取る
    #    全 chunk 全部開くと重いので、まず meta のエピソード長から
    #    どの file に載るか推定できない（v3.0 は frame 連番で切るため）→ 全走査。
    #    data 全体は数 GB 程度。一度読む。
    data_dir = os.path.join(args.dataset_root, "data")
    found = {}
    for chunk in sorted(os.listdir(data_dir)):
        cdir = os.path.join(data_dir, chunk)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".parquet"):
                continue
            path = os.path.join(cdir, fn)
            print(f"reading {path}", flush=True)
            table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
            for batch in table.to_batches():
                b = batch.to_pydict()
                eps_col = b["episode_index"]
                for i, e in enumerate(eps_col):
                    if e in ep_ids and e not in found:
                        found[e] = {"actions": [], "frames": []}
                for e in set(eps_col) & set(ep_ids):
                    keep = [
                        i for i, ee in enumerate(eps_col) if ee == e
                    ]
                    keep.sort(key=lambda i: b["frame_index"][i])
                    for i in keep:
                        found[e]["actions"].append(b["action"][i])
                        found[e]["frames"].append(b["frame_index"][i])
            # 全部そろったら打ち切り
            if set(found) >= set(ep_ids):
                break
        if set(found) >= set(ep_ids):
            break

    # 3) 検証・保存
    out = {}
    for e in ep_ids:
        if e not in found:
            raise SystemExit(f"episode {e} not found in data parquet")
        rec = found[e]
        # frame 番号で重複を取り除く（跨ぎ file の重複防止）
        seen, uniq = set(), []
        for f, a in zip(rec["frames"], rec["actions"]):
            if f not in seen:
                seen.add(f)
                uniq.append(a)
        actions = np.array(uniq, dtype=np.float32)
        expected = lens.get(e)
        if expected is not None and len(actions) != expected:
            print(f"warn: episode {e}: {len(actions)} frames vs meta length {expected}")
        else:
            print(f"episode {e}: {len(actions)} action frames, last={actions[-1]!r}")
        out[f"actions_{e}"] = actions
        out[f"length_{e}"] = len(actions)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()