#!/usr/bin/env python3
"""LIBERO 本家の hdf5 から、物体の姿勢が入った `states` だけを抜き出して置く。

**なぜ要るか。**学習に使っている `lerobot/libero_plus` には手先と指しか入って
いない（`observation.state` は 8 次元）。**物体がどこにあるかが 1 列も無いので、
「対象外の物体に触れたか」はそのデータからは判定できない。**LIBERO 本家の
hdf5 には MuJoCo の状態がまるごと（`states`・79〜のベクトル）記録されていて、
そこから全物体の姿勢が読める。**再生をやり直す必要が無く、記録そのものを読む。**

**hdf5 は画像込みで 33.8 GB あるが、要るのは `states` と `actions` だけである。**
1 課題ぶんを落とす → 抜き出す → hdf5 を消す、を 40 回まわす。**使うディスクは
どの時点でも 2.1 GB を超えない。**

用法（ホスト側。`.venv` の python で打つ）:
    .venv/bin/python solution/tools/contact_clean/fetch_libero_states.py \\
        --out work/artifacts/libero_states

Reads:
  - Hugging Face `yifengzhu-hf/LIBERO-datasets` の `<suite>/<task>_demo.hdf5`
Generates:
  - <out>/<suite>/<task>_demo.npz … デモごとの `states` と `actions`
  - <out>/index.json … 課題ごとのデモ数・コマ数・bddl の名前
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, metavar="DIR",
                        help="抜き出した npz の置き場")
    parser.add_argument("--suites", nargs="*", default=list(SUITES), metavar="NAME",
                        help=f"落とすスイート。既定は {' '.join(SUITES)}")
    parser.add_argument("--keep-hdf5", action="store_true",
                        help="hdf5 を消さずに残す（**既定は消す。**33.8 GB ある）")
    parser.add_argument("--hdf5-dir", type=Path, default=None, metavar="DIR",
                        help="hdf5 の一時置き場。既定は <out>/_hdf5")
    return parser.parse_args()


def _list_task_files(suites: list[str]) -> list[tuple[str, int]]:
    """落とす hdf5 のパスと大きさを、小さい順に並べて返す。

    Returns
    -------
    files : list[tuple[str, int]]
        `(リポジトリ内のパス, バイト数)` の一覧。
    """
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(HF_REPO_ID, repo_type="dataset", recursive=True)
    files = [(entry.path, entry.size) for entry in entries
             if getattr(entry, "size", None) and entry.path.split("/")[0] in suites]
    return sorted(files, key=lambda item: item[1])


def _extract_one(hdf5_path: Path, npz_path: Path) -> dict:
    """1 課題ぶんの hdf5 から、デモごとの `states` と `actions` を npz へ写す。

    **画像（`obs/agentview_rgb` ほか）は写さない。**接触の判定に要らず、
    大きさの 99% を占めるためである。

    Parameters
    ----------
    hdf5_path : Path
        落としてきた `<task>_demo.hdf5`。
    npz_path : Path
        書き出し先。

    Returns
    -------
    summary : dict
        デモ数・コマ数・bddl のファイル名。`index.json` に積む。

    Reads:
      - <hdf5_path>
    Generates:
      - <npz_path>
    """
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        data = handle["data"]
        demo_names: list[str] = sorted(data.keys(), key=lambda name: int(name.split("_")[1]))
        arrays: dict[str, np.ndarray] = {}
        total_frames = 0
        for name in demo_names:
            states = np.asarray(data[name]["states"], dtype=np.float64)
            actions = np.asarray(data[name]["actions"], dtype=np.float32)
            arrays[f"{name}/states"] = states
            arrays[f"{name}/actions"] = actions
            total_frames += int(actions.shape[0])
        bddl_name: str = str(data.attrs["bddl_file_name"])
        problem = json.loads(data.attrs["problem_info"])

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    return {"demos": len(demo_names), "frames": total_frames,
            "bddl_file_name": bddl_name,
            "language_instruction": problem["language_instruction"],
            "state_dim": int(arrays[f"{demo_names[0]}/states"].shape[1])}


def _fetch_and_extract(repo_path: str, out_dir: Path, hdf5_dir: Path,
                       keep_hdf5: bool) -> dict:
    """1 課題ぶんを落として抜き出し、hdf5 を片付ける。"""
    from huggingface_hub import hf_hub_download

    local = Path(hf_hub_download(HF_REPO_ID, repo_path, repo_type="dataset",
                                 local_dir=hdf5_dir))
    try:
        summary = _extract_one(local, out_dir / repo_path.replace(".hdf5", ".npz"))
    finally:
        if not keep_hdf5:
            local.unlink(missing_ok=True)
    return summary


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out.resolve()
    hdf5_dir: Path = (args.hdf5_dir or out_dir / "_hdf5").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_task_files(args.suites)
    total_bytes = sum(size for _, size in files)
    print(f"=== {len(files)} 課題・{total_bytes / 1e9:.2f} GB を順に落として抜き出す ===")

    index: dict[str, dict] = {}
    index_path = out_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        print(f"  すでに {len(index)} 課題ぶんある — その分は飛ばす")

    for number, (repo_path, size) in enumerate(files, start=1):
        label = f"[{number}/{len(files)}] {repo_path}"
        if repo_path in index:
            print(f"{label}  飛ばす")
            continue
        started = time.time()
        summary = _fetch_and_extract(repo_path, out_dir, hdf5_dir, args.keep_hdf5)
        index[repo_path] = summary
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        print(f"{label}  {summary['demos']} デモ / {summary['frames']} コマ / "
              f"状態 {summary['state_dim']} 次元  ({time.time() - started:.1f}s)")

    if not args.keep_hdf5 and hdf5_dir.exists():
        shutil.rmtree(hdf5_dir, ignore_errors=True)

    demos = sum(item["demos"] for item in index.values())
    frames = sum(item["frames"] for item in index.values())
    print(f"\n=== 出来上がり: {out_dir} ===")
    print(f"  {len(index)} 課題 / {demos} デモ / {frames} コマ")


if __name__ == "__main__":
    main()
