#!/usr/bin/env python3
"""動画ファイルの通し番号の桁数を、lerobot が焼き込んでいる 3 桁に合わせる。

**なぜ要るか。**`aggregate_datasets`（`merge` の中身）は**データセット自身の
`video_path` を見ない。**lerobot v0.6.0 に焼き込まれた

    CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"

だけを使って読みに行く（`datasets/aggregate.py` の `aggregate_videos`）。
ところが**配布の `libero_combined_20hz` は 6 桁**である

    video_path = videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4

ので、`file-000000.mp4` が実物なのに `file-000.mp4` を探して
`av.error.FileNotFoundError` で止まる。**2026-09-07 に踏んだ。**

**やること。**実物の隣に 3 桁の名前で symlink を張るだけである。

- **info.json は書き換えない。**だから**このデータセットは今までどおり
  6 桁の書式で読める**（`LeRobotDataset` は info.json の書式を使う）。
  **学習に直接使う道はまったく壊れない。**
- `shutil.copy` は symlink をたどって中身を写すので、**混ぜた先には実体が入る。**
- **出力側は 3 桁で書かれ、出力の info.json も 3 桁を宣言する**ので、
  混ぜた結果は自己整合する。

番号が 1000 以上あると 3 桁に収まらないので、そのときは止める
（`chunks_size` が 1000 なので、本来そこで chunk が繰り上がる）。

使い方:

    python3 link_video_index_width.py --root /opt/dlami/nvme/data/libero_combined_20hz
    python3 link_video_index_width.py --root ... --undo      # 張った symlink を消す
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

NAME = re.compile(r"file-(\d+)\.mp4")


def iter_chunks(root: Path):
    """`videos/<key>/chunk-*/` を順に返す。"""
    videos = root / "videos"
    if not videos.is_dir():
        raise SystemExit(f"{videos} が無い。--root がデータセットの根を指していない")
    for key in sorted(p for p in videos.iterdir() if p.is_dir()):
        for chunk in sorted(p for p in key.iterdir() if p.is_dir()):
            yield key.name, chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, metavar="DIR",
                        help="データセットの根（meta / data / videos がある所）")
    parser.add_argument("--undo", action="store_true",
                        help="張った symlink を消す（実物には触らない）")
    args = parser.parse_args()

    made = removed = already = 0
    for key, chunk in iter_chunks(args.root):
        for path in sorted(chunk.glob("file-*.mp4")):
            match = NAME.fullmatch(path.name)
            if match is None:
                continue
            index = int(match.group(1))
            short = chunk / f"file-{index:03d}.mp4"

            if args.undo:
                if short != path and short.is_symlink():
                    short.unlink()
                    removed += 1
                continue

            if short == path:
                already += 1          # 既に 3 桁。触らない
                continue
            if index > 999:
                raise SystemExit(
                    f"3 桁に収まらない番号がある: {path}\n"
                    "  **chunk が繰り上がっていない。symlink では直せない。**")
            if short.is_symlink() or short.exists():
                already += 1
                continue
            # **相対名で張る。**データセットごと移しても壊れない
            short.symlink_to(path.name)
            made += 1

    if args.undo:
        print(f"=== symlink を {removed} 本消した（実物は触っていない）===")
        return
    print(f"=== 3 桁の symlink を {made} 本張った"
          f"（既にあった／不要だったもの {already} 本）===")
    if made:
        print("  **info.json は書き換えていない。**"
              "このデータセットは今までどおり読める")


if __name__ == "__main__":
    main()
