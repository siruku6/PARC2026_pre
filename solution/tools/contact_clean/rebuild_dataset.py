#!/usr/bin/env python3
"""残すエピソードの一覧から、**動画を作り直さずに**データセットを組み直す。

**なぜ lerobot の `split_dataset` を使わないのか。**あれは残すコマだけを
**AV1 で再エンコードし直す**（`dataset_tools._keep_episodes_from_video_with_av`）。
実測で 90 本に数十分かかり、**12,216 本では 100 時間を超える。**

**そこを避けられるのは、LeRobot v3 が「1 本の mp4 に何十本ものエピソードを詰め、
エピソードごとの開始時刻で切り出して読む」形だからである。**読む側は
`dataset_reader` の `shifted_query_ts = from_timestamp + ts` で場所を決める。
**だから動画のファイルはそのまま置いておき、メタデータだけ組み直せばよい。**
動画はハードリンクで置くので、**ディスクは 1 バイトも増えない。**

**残す一覧に同じ番号を 2 度書いてよい。**その回数だけコピーが作られる。
`--dataset.episodes` では同じことができない（集合として扱われるので 1 度しか
採られない。`reference/29` の 7.1 節）。**痩せた課題を 3 倍に戻すのはこの口を使う。**

用法:
    .venv/bin/python solution/tools/contact_clean/rebuild_dataset.py \\
        --root work/artifacts/libero_plus --output work/artifacts/libero_plus_clean \\
        --keep-file work/artifacts/libero_states/keep_cleaned.json \\
        --keep-file work/artifacts/libero_states/keep_boost.json \\
        --keep-file work/artifacts/libero_states/keep_boost.json

Reads:
  - <root>/meta/info.json, meta/tasks.parquet, meta/episodes/*.parquet
  - <root>/data/chunk-*/file-*.parquet
  - <keep-file> … `{"episodes": [番号, …]}`。複数渡すと順に繋ぐ
Generates:
  - <output>/meta/{info.json,tasks.parquet,stats.json,episodes/...}
  - <output>/data/chunk-*/file-*.parquet
  - <output>/videos/... … 元のファイルへのハードリンク
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_STAT_KEYS = ("min", "max", "mean", "std", "count")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, metavar="DIR")
    parser.add_argument("--output", type=Path, required=True, metavar="DIR")
    parser.add_argument("--keep-file", type=Path, action="append", required=True,
                        metavar="JSON", help="残す番号の一覧。複数渡すと順に繋ぐ")
    parser.add_argument("--copy-videos", action="store_true",
                        help="ハードリンクではなく複製する（**別の入れ物へ運ぶとき**）")
    return parser.parse_args()


def _load_plan(keep_files: list[Path]) -> list[int]:
    """残すエピソード番号を、渡された順に繋いだ一覧にする。**重複はそのまま残す。**"""
    plan: list[int] = []
    for path in keep_files:
        plan.extend(int(x) for x in json.loads(path.read_text())["episodes"])
    return plan


def _load_source_frames(root: Path) -> tuple[pa.Table, np.ndarray]:
    """データ本体を 1 つの表にまとめ、エピソード番号の境目を返す。

    Returns
    -------
    table : pa.Table
        `episode_index` の昇順に並べ直した全コマ。
    bounds : np.ndarray
        `bounds[i]` から `bounds[i+1]` がエピソード `i` の行。
    """
    files = sorted((root / "data").rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in files])
    episode_index = np.asarray(table["episode_index"])
    order = np.argsort(episode_index, kind="stable")
    table = table.take(pa.array(order))
    episode_index = episode_index[order]
    bounds = np.searchsorted(episode_index, np.arange(int(episode_index.max()) + 2))
    return table, bounds


def _remap_tasks(source_tasks: pa.Table, used: set[int]):
    """使った課題だけを残し、`task_index` を 0 から振り直す。

    **振り直さずに元の番号を残すと `total_tasks` と実際の数が食い違う。**

    **指示文は列ではなく pandas の索引として書く。**lerobot の `load_tasks` が
    `tasks.index.name = "task"` を当てて読むので、列で書くと指示文が引けず、
    学習データの `task` が数字のまま出てくる（9/9 に 1 度そうなった）。

    **読むときも列の名前に頼らない。**指示文の列は `lerobot/libero_plus` では `task`、
    配布の `libero_combined_20hz` では `__index_level_0__` である（どちらも pandas の索引として
    書かれている）。`to_pandas()` で索引に戻してから読めば、どちらでも同じに読める。
    """
    import pandas as pd

    source = source_tasks.to_pandas()
    pairs = sorted((int(index), str(task)) for index, task
                   in zip(source["task_index"], source.index)
                   if int(index) in used)
    mapping = {old: new for new, (old, _) in enumerate(pairs)}
    frame = pd.DataFrame({"task_index": [mapping[old] for old, _ in pairs]},
                         index=pd.Index([task for _, task in pairs], name="task"))
    return frame, mapping


def _build_data_files(table: pa.Table, bounds: np.ndarray, plan: list[int],
                      task_map: dict[int, int], output: Path, info: dict
                      ) -> tuple[list[dict], int]:
    """コマを並べ直して parquet に書き、エピソードごとの置き場を返す。

    **`index` は データセット全体で 0 から連番**であり、`episode_index` も
    振り直す。**元の番号をそのまま残すと、番号で引く仕組みがすべて壊れる。**

    Returns
    -------
    placement : list[dict]
        各エピソードの `data/chunk_index`・`data/file_index`・行の範囲。
    total_frames : int
    """
    limit_bytes = int(info.get("data_files_size_in_mb", 100)) * 1024 * 1024
    chunks_size = int(info.get("chunks_size", 1000))
    columns = table.column_names

    placement: list[dict] = []
    chunk_index, file_index, written_bytes = 0, 0, 0
    buffer: list[pa.Table] = []
    global_row = 0

    def flush() -> None:
        nonlocal buffer, written_bytes, file_index, chunk_index
        if not buffer:
            return
        path = output / info["data_path"].format(chunk_index=chunk_index, file_index=file_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(buffer), path)
        buffer, written_bytes = [], 0
        file_index += 1
        if file_index >= chunks_size:
            chunk_index, file_index = chunk_index + 1, 0

    for new_episode, source_episode in enumerate(plan):
        start, stop = int(bounds[source_episode]), int(bounds[source_episode + 1])
        length = stop - start
        piece = table.slice(start, length)
        piece = piece.set_column(columns.index("episode_index"), "episode_index",
                                 pa.array(np.full(length, new_episode, dtype=np.int64)))
        piece = piece.set_column(columns.index("index"), "index",
                                 pa.array(np.arange(global_row, global_row + length,
                                                    dtype=np.int64)))
        piece = piece.set_column(
            columns.index("task_index"), "task_index",
            pa.array([task_map[int(x)] for x in piece["task_index"]], pa.int64()))

        piece_bytes = piece.nbytes
        if buffer and written_bytes + piece_bytes > limit_bytes:
            flush()
        placement.append({"data/chunk_index": chunk_index, "data/file_index": file_index,
                          "dataset_from_index": global_row,
                          "dataset_to_index": global_row + length})
        buffer.append(piece)
        written_bytes += piece_bytes
        global_row += length
    flush()
    return placement, global_row


def _build_episode_metadata(source_meta: pa.Table, plan: list[int], placement: list[dict],
                            output: Path, info: dict) -> None:
    """エピソードのメタデータを組み直す。

    **`videos/…` の 4 列は元の値をそのまま写す。**動画のファイルは触っていないので、
    そこを書き換えると読めなくなる。**per-episode の `stats/…` もそのまま写す**
    （中身のコマは 1 つも変えていない）。
    """
    columns = source_meta.column_names
    rows = source_meta.take(pa.array(plan, pa.int64()))
    arrays, names = [], []
    for name in columns:
        if name == "episode_index":
            arrays.append(pa.array(np.arange(len(plan), dtype=np.int64)))
        elif name in placement[0]:
            arrays.append(pa.array([entry[name] for entry in placement], pa.int64()))
        else:
            arrays.append(rows[name])
        names.append(name)
    path = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays, names=names), path)


def _link_videos(root: Path, output: Path, copy: bool) -> int:
    """動画をそのまま置く。**既定はハードリンクなのでディスクは増えない。**"""
    count = 0
    for source in sorted((root / "videos").rglob("*.mp4")):
        target = output / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(source, target)
        else:
            target.hardlink_to(source.resolve())
        count += 1
    return count


def _write_stats(source_meta: pa.Table, plan: list[int], output: Path) -> None:
    """`meta/stats.json` を、残したエピソードの統計から組み直す。

    **画像の統計はコマを読み直さずに済む**——エピソードごとの統計が
    `meta/episodes` に入っているので、それを重み付きで合成する
    （lerobot の `aggregate_stats` と同じ式）。
    """
    rows = source_meta.take(pa.array(plan, pa.int64())).to_pydict()
    features = sorted({name.split("/", 1)[1].rsplit("/", 1)[0]
                       for name in source_meta.column_names if name.startswith("stats/")})
    stats: dict[str, dict] = {}
    for feature in features:
        counts = np.array([np.asarray(x).reshape(-1)[0]
                           for x in rows[f"stats/{feature}/count"]], dtype=np.float64)
        means = np.array([np.asarray(x, dtype=np.float64) for x in rows[f"stats/{feature}/mean"]])
        stds = np.array([np.asarray(x, dtype=np.float64) for x in rows[f"stats/{feature}/std"]])
        mins = np.array([np.asarray(x, dtype=np.float64) for x in rows[f"stats/{feature}/min"]])
        maxs = np.array([np.asarray(x, dtype=np.float64) for x in rows[f"stats/{feature}/max"]])
        weight = counts.reshape((-1,) + (1,) * (means.ndim - 1))
        total = counts.sum()
        mean = (means * weight).sum(axis=0) / total
        # 群をまたぐ分散は「群内の分散 ＋ 群平均のばらつき」で足す（並列アルゴリズム）
        variance = ((stds ** 2 + (means - mean) ** 2) * weight).sum(axis=0) / total
        stats[feature] = {"min": mins.min(axis=0).tolist(), "max": maxs.max(axis=0).tolist(),
                          "mean": mean.tolist(), "std": np.sqrt(variance).tolist(),
                          "count": [int(total)]}
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))


def main() -> None:
    args = _parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"出力先が既にある: {output}")

    info = json.loads((root / "meta/info.json").read_text())
    plan = _load_plan(args.keep_file)
    print(f"=== {root} の {info['total_episodes']} 本から {len(plan)} 本を組む ===")
    print(f"  実体 {len(set(plan))} 本・コピーで増えた分 {len(plan) - len(set(plan))} 本")

    source_meta = pq.read_table(sorted((root / "meta/episodes").rglob("*.parquet"))[0])
    source_meta = source_meta.take(pa.array(np.argsort(np.asarray(source_meta["episode_index"]),
                                                       kind="stable")))
    table, bounds = _load_source_frames(root)

    used = {int(x) for x in
            pa.concat_tables([table.slice(int(bounds[e]), 1) for e in sorted(set(plan))])["task_index"]}
    tasks, task_map = _remap_tasks(pq.read_table(root / "meta/tasks.parquet"), used)

    (output / "meta").mkdir(parents=True)
    placement, total_frames = _build_data_files(table, bounds, plan, task_map, output, info)
    _build_episode_metadata(source_meta, plan, placement, output, info)
    tasks.to_parquet(output / "meta" / "tasks.parquet")
    _write_stats(source_meta, plan, output)

    videos = _link_videos(root, output, args.copy_videos)
    info = dict(info, total_episodes=len(plan), total_frames=total_frames,
                total_tasks=len(tasks), splits={"train": f"0:{len(plan)}"})
    (output / "meta/info.json").write_text(json.dumps(info, indent=4))

    print(f"  課題 {len(tasks)} 種 / コマ {total_frames:,} / 動画 {videos} ファイル"
          f"（{'複製' if args.copy_videos else 'ハードリンク'}）")
    print(f"=== 出来上がり: {output} ===")


if __name__ == "__main__":
    main()
