"""学習データ `lerobot/libero_plus` に、どの摂動がどれだけ入っているかを測る。

**variant 名はメタデータのどの列にも残っていない**（docs/reference/06 の
「variant 名はどの列にも無い」節）ので、中身から読むしかない。この道具は 2 つを出す。

1. **カメラ視点** — 1 タスクぶんのエピソードから等間隔に抜き、各エピソードの
   1 枚目の front 画像を 1 枚のタイル画像に並べる。**別のカメラ位置かどうかは
   目視で数える**（画素の統計では照明・背景の変化と区別が付かない）。
2. **ロボット初期姿勢** — 各エピソードのフレーム 0 の `observation.state` の
   先頭 3 つが手先の位置である（docs/reference/10「state は LeRobot 側が
   8 次元に組む」）。**タスクごとの中央値からの距離**を集計する。
   タスクが違えば場面も違うので、必ずタスク内で比べる。

使い方（`camera` は動画を読むので av が要る。`pose` は parquet だけで足りる）:

    python3 measure_train_data_perturbation.py pose  --root ~/.cache/huggingface/lerobot/lerobot/libero_plus
    python3 measure_train_data_perturbation.py camera --root ... --task "turn on the stove" \
        --count 32 --output docs/figures/libero_plus_camera_32ep.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _episode_meta(root: Path) -> pd.DataFrame:
    """meta/episodes/**/*.parquet を 1 枚にまとめる。tasks は要素 1 の配列で入っている。"""
    files = sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"meta/episodes が空である: {root}")
    cols = [
        "episode_index", "tasks",
        "videos/observation.images.front/chunk_index",
        "videos/observation.images.front/file_index",
        "videos/observation.images.front/from_timestamp",
    ]
    meta = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    meta["task"] = meta["tasks"].apply(lambda v: v[0] if not isinstance(v, str) else v)
    return meta


def measure_pose(root: Path) -> None:
    """フレーム 0 の手先位置が、タスクの中央値からどれだけ散っているかを出す。"""
    meta = _episode_meta(root)[["episode_index", "task"]]
    frames = []
    for f in sorted(glob.glob(str(root / "data/chunk-*/file-*.parquet"))):
        d = pd.read_parquet(f, columns=["observation.state", "frame_index", "episode_index"])
        frames.append(d[d["frame_index"] == 0])
    first = pd.concat(frames, ignore_index=True)
    state = np.stack(first["observation.state"].values)
    first = first[["episode_index"]].copy()
    first[["x", "y", "z"]] = state[:, :3]
    first = first.merge(meta, on="episode_index")

    rows = []
    for task, group in first.groupby("task"):
        pos = group[["x", "y", "z"]].to_numpy()
        dist = np.linalg.norm(pos - np.median(pos, axis=0), axis=1)
        rows.append({
            "task": task, "n": len(group),
            "p50_m": np.percentile(dist, 50), "p90_m": np.percentile(dist, 90),
            "max_m": dist.max(),
            "over_1cm": (dist > 0.01).mean(), "over_3cm": (dist > 0.03).mean(),
        })
    table = pd.DataFrame(rows)
    print(f"エピソード {len(first)} 本・{len(table)} タスク")
    print("タスクごとの『手先がタスク中央値からどれだけ離れているか』（m）:")
    print(table[["p50_m", "p90_m", "max_m", "over_1cm", "over_3cm"]]
          .describe().loc[["min", "50%", "max"]].round(4).to_string())
    print(f"\n1 cm 超の割合（全タスク平均）: {100 * table['over_1cm'].mean():.1f}%")
    print(f"3 cm 超の割合（全タスク平均）: {100 * table['over_3cm'].mean():.1f}%")


def measure_camera(root: Path, task: str, count: int, output: Path) -> None:
    """1 タスクぶんのエピソードから等間隔に抜き、1 枚目の front 画像を並べる。"""
    import av
    from PIL import Image

    meta = _episode_meta(root)
    sel = meta[meta["task"] == task].sort_values("episode_index")
    if sel.empty:
        raise SystemExit(f"そのタスクは無い: {task}")
    sel = sel.iloc[:: max(1, len(sel) // count)][:count]
    print(f"{task}: 全 {len(meta[meta['task'] == task])} 本から {len(sel)} 本を抜く")

    tiles = []
    for _, row in sel.iterrows():
        chunk = int(row["videos/observation.images.front/chunk_index"])
        index = int(row["videos/observation.images.front/file_index"])
        start = float(row["videos/observation.images.front/from_timestamp"])
        path = root / f"videos/observation.images.front/chunk-{chunk:03d}/file-{index:03d}.mp4"
        container = av.open(str(path))
        stream = container.streams.video[0]
        container.seek(int(start / float(stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) >= start - 1e-3:
                tiles.append(frame.to_ndarray(format="rgb24"))
                break
        container.close()

    side = 160
    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (side * cols, side * rows))
    for i, tile in enumerate(tiles):
        sheet.paste(Image.fromarray(tile).resize((side, side)), (side * (i % cols), side * (i // cols)))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"書き出した: {output}（{cols} 列 × {rows} 行）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("pose", "camera"))
    parser.add_argument("--root", type=Path, required=True,
                        help="LeRobot データセットの root（meta/ と data/ がある階層）")
    parser.add_argument("--task", default="turn on the stove", help="camera のときの対象タスク")
    parser.add_argument("--count", type=int, default=32, help="camera のときに抜く本数")
    parser.add_argument("--output", type=Path, default=Path("libero_plus_camera.png"))
    args = parser.parse_args()

    if args.what == "pose":
        measure_pose(args.root)
    else:
        measure_camera(args.root, args.task, args.count, args.output)


if __name__ == "__main__":
    main()
