#!/usr/bin/env python3
"""作った mp4 を「どれを開くべきか」の順に並べ、コマ並べ画像を作る。

**40 本の動画をあてもなく開くと時間が溶ける。** 打ち切りに張り付いた回
（`未完了`）を先頭に、ステップ数の多い順で並べる。**そこが本番で起きていることに
いちばん近い。** さらに 1 本ずつ 8 コマの静止画に畳むので、**動画を再生せずに
当たりを付けられる。**

**使うのは動画のファイル名とコマの中身だけ**である。評価ログが手元に無くても
読めるようにするため、`[結果]` 行には依存しない。結末・タスク名・ステップ数は
[rollout_recorder.py](rollout_recorder.py) が付けたファイル名から取り、
静止か往復かは動画に焼かれた文字から目で読む。

    python3 solution/tools/eval/summarize_rollout_videos.py work/videos

    # コマ並べ画像を作らない（一覧だけ見る）
    python3 solution/tools/eval/summarize_rollout_videos.py work/videos --no-sheets

Reads:
  - <dir>/*.mp4（rollout_recorder.py が書いたもの）
Generates:
  - 標準出力に一覧
  - <dir>/sheets/<動画名>.png（`--no-sheets` で抑制）
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

# ファイル名の形： <結末>_<タスク名>_ep<N>_<ステップ数>steps.mp4
_NAME_RE = re.compile(r"^(?P<outcome>[^_]+)_(?P<task>.+)_ep(?P<ep>\d+)_(?P<steps>\d+)steps$")

# コマ並べ画像に何コマ並べるか。4 列 × 2 行で、空き枠が出ない数にしている。
_SHEET_FRAMES = 8
_SHEET_COLUMNS = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="mp4 の置き場")
    parser.add_argument("--no-sheets", action="store_true",
                        help="コマ並べ画像を作らない")
    return parser.parse_args()


def _describe(path: Path) -> dict:
    """1 本の mp4 から、並べ替えに使う情報を取り出す。"""
    import cv2

    match = _NAME_RE.match(path.stem)
    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    return {
        "path": path,
        "outcome": match.group("outcome") if match else "（不明）",
        "task": match.group("task") if match else path.stem,
        "episode": int(match.group("ep")) if match else -1,
        "steps": int(match.group("steps")) if match else frames,
        "frames": frames,
        "readable": frames > 0,
    }


def _write_sheet(entry: dict, output_dir: Path) -> Path | None:
    """1 本を等間隔 8 コマの静止画に畳む。"""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(entry["path"]))
    total = entry["frames"]
    if total <= 0:
        capture.release()
        return None

    picks = [int(round(i * (total - 1) / (_SHEET_FRAMES - 1))) for i in range(_SHEET_FRAMES)]
    tiles = []
    for index in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        read, frame = capture.read()
        if read:
            tiles.append(frame)
    capture.release()

    if not tiles:
        return None

    # 行に足りない分は黒で埋める。**形を揃えないと hstack が落ちる。**
    while len(tiles) % _SHEET_COLUMNS:
        tiles.append(np.zeros_like(tiles[0]))

    rows = [
        np.hstack(tiles[i:i + _SHEET_COLUMNS])
        for i in range(0, len(tiles), _SHEET_COLUMNS)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{entry['path'].stem}.png"
    cv2.imwrite(str(destination), np.vstack(rows))
    return destination


def main() -> None:
    args = _parse_args()
    directory: Path = args.directory
    if not directory.is_dir():
        raise SystemExit(f"エラー: {directory} がありません")

    videos = sorted(directory.glob("*.mp4"))
    if not videos:
        raise SystemExit(
            f"{directory} に mp4 がありません。\n"
            "  PARC_RECORD_DIR を設定して評価を回しましたか？"
        )

    entries = [_describe(path) for path in videos]

    # **打ち切りに張り付いた回を先頭に置く。** そこが見たいものである。
    entries.sort(key=lambda e: (e["outcome"] != "未完了", -e["steps"], e["task"]))

    counts = Counter(entry["outcome"] for entry in entries)
    print(f"{directory} の mp4 {len(entries)} 本")
    print("  " + "  ".join(f"{key} {value}" for key, value in counts.most_common()))

    broken = [entry for entry in entries if not entry["readable"]]
    if broken:
        print(f"\n**読めないファイルが {len(broken)} 件**")
        for entry in broken:
            print(f"  {entry['path'].name}")

    print(f"\n{'結末':6s} {'歩':>4s}  タスク")
    for entry in entries:
        print(f"  {entry['outcome']:6s} {entry['steps']:4d}  "
              f"{entry['task'][:64]} ep{entry['episode']}")

    if args.no_sheets:
        return

    print(f"\nコマ並べ画像（{_SHEET_FRAMES} コマ）")
    made = 0
    for entry in entries:
        destination = _write_sheet(entry, directory / "sheets")
        if destination is not None:
            made += 1
    print(f"  {made} 枚を {directory / 'sheets'} に作った")
    print("\n**まず 未完了 のコマ並べを上から見る。** 手先が動いていないなら"
          "停滞の検知（項目 9）、物を掴めていないなら学習側の話になる。")


if __name__ == "__main__":
    main()
