#!/usr/bin/env python3
"""元の手本と、台本で作った軌道を、同じ画面に並べて見る。

**数字だけでは「なぜ 400 ステップかかるのか」が分からない。** 元の手本は人が
遠隔操作したもので、迷い・行き過ぎ・持ち直しがそのまま入っている。台本で
作った軌道はそれが無い。**並べて初めて、削れたステップ数の中身が見える。**

左が元の手本（`lerobot/libero_plus` の実フレーム）、右が台本で作った軌道
（[scripted_demo.py](scripted_demo.py) の `record` が書いた npz）である。
**どちらも 180 度回した向き**で、学習時にモデルへ入るのと同じ絵になる。

    # 1. 台本の軌道を書き出す（先にこちらを済ませる）
    MUJOCO_GL=egl python3 solution/tools/scripted_demo.py record \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \\
        --episodes 0 --output /out/scripted

    # 2. 並べて動画にする
    python3 solution/tools/visualize_demo.py \\
        --generated /out/scripted/KITCHEN_SCENE8_..._ep000.npz \\
        --task "put both moka pots on the stove" \\
        --episodes-parquet outputs/tmp/libero_plus_episodes.parquet \\
        --output /out/compare

元の手本は 1 本ずつ切り出さない。**データセットは 1 ファイルに何百本も
繋げて入れてある**（1 ファイル 209 MB）ので、`meta/episodes` の
`from_timestamp`〜`to_timestamp` でその区間だけ読む。

**色の並びは RGB のまま持ち回り、書き出す直前に 1 回だけ入れ替える。**
途中で入れ替えると、打ち消し合う変換が生まれるか、片方の出力だけ赤と青が
逆になる（実際に 1 枚絵だけ反転していた）。

Reads:
  - <generated>.npz（`scripted_demo.py record` の出力）
  - <episodes-parquet>（`lerobot/libero_plus` の `meta/episodes/*.parquet`）
  - https://huggingface.co/datasets/<repo>/resolve/<revision>/videos/...
Generates:
  - <output>/original_ep<n>.mp4 / generated_ep<n>.mp4 / compare_ep<n>.mp4
  - <output>/sheet_ep<n>.png（コマを並べた 1 枚絵）
  - <output>/paths_ep<n>.png（手先の軌跡の図）
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple

import numpy as np

# npz の取り決めは書き手と同じファイルに置いてある。**写しを作らない。**
# `scripted_demo` は lerobot を import しないので、ここから読んで安全である。
#
# **標準入力から流し込んで実行すると `__file__` が使えない**
# （`docker exec -i ... python - <引数> < この file` の形）。その場合は
# 道具の置き場を PYTHONPATH に入れて渡す。
if Path(__file__).suffix == ".py":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from scripted_demo import SavedEpisode  # noqa: E402
except ModuleNotFoundError as error:  # pragma: no cover - 起動の仕方の問題
    raise SystemExit(
        f"{error}\n  scripted_demo.py が import できない。"
        "**道具の置き場を PYTHONPATH に入れて実行すること**\n"
        "  例: PYTHONPATH=/path/to/solution/tools python - check ... < npz_to_lerobot.py"
    ) from error

# **学習に使った版に固定する。** `main` から取ると、上流が再アップロードした
# 瞬間に「元の手本」が別の版のコマになり、しかも気づけない。
# 版は train.py / check_holdout_episodes.py と同じもの。
DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
_HF_VIDEO_URL = ("https://huggingface.co/datasets/{repo}/resolve/{revision}/"
                 "videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4")

# 並べるときの 1 枚あたりの大きさ。**128 px のままでは何が起きているか読めない。**
_PANEL_SIZE = 384
_LABEL_HEIGHT = 44


class Original(NamedTuple):
    """元の手本 1 本の在り処。**添字で持つと列を足したとき全部ずれる。**"""

    episode: int
    length: int
    chunk: int
    file: int
    start_sec: float
    end_sec: float


def _list_originals(parquet_path: Path, task: str) -> list[Original]:
    """その指示文の手本を、短い順に並べて返す。"""
    import pyarrow.parquet as pq

    prefix = "videos/observation.images.front"
    columns = ["episode_index", "tasks", "length",
               f"{prefix}/chunk_index", f"{prefix}/file_index",
               f"{prefix}/from_timestamp", f"{prefix}/to_timestamp"]
    table = pq.read_table(parquet_path, columns=columns).to_pydict()
    rows = [
        Original(table["episode_index"][i], table["length"][i],
                 table[f"{prefix}/chunk_index"][i],
                 table[f"{prefix}/file_index"][i],
                 table[f"{prefix}/from_timestamp"][i],
                 table[f"{prefix}/to_timestamp"][i])
        for i in range(len(table["length"]))
        # **指示文は完全一致で引く。** `in` で部分一致させると別タスクを拾う。
        if task in {str(cell) for cell in table["tasks"][i]}
    ]
    if not rows:
        raise SystemExit(f"その指示文の手本が無い: {task}")
    return sorted(rows, key=lambda row: row.length)


def _video_url(repo: str, row: Original) -> str:
    """手本の動画ファイルの場所。"""
    return _HF_VIDEO_URL.format(repo=repo, revision=DATASET_REVISION,
                                key="observation.images.front",
                                chunk=row.chunk, file=row.file)


def _open_video(url: str) -> tuple[Any, Any]:
    """遠隔の mp4 を開く。

    復号には PyAV を使う。**データセットの動画は AV1 で、OpenCV に付いてくる
    ffmpeg では復号できない**（`Failed to get pixel format` で 0 コマになる）。
    PyAV は libdav1d を同梱しているので、そのまま読める。

        pip install av
    """
    import av

    container = av.open(url)
    return container, container.streams.video[0]


def _read_span(container: Any, stream: Any, start_sec: float, end_sec: float,
               count: int | None = None) -> np.ndarray:
    """開いてある動画から、一区間だけを読む。

    `count` を渡したらその枚数でちょうど止める。**終了時刻だけで切ると
    次のエピソードの 1 コマ目が紛れ込む**（境界がコマの中心と一致しない）。
    """
    container.seek(int(start_sec / stream.time_base), stream=stream)
    frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        if frame.time is None or frame.time < start_sec - 1e-6:
            continue
        if frame.time > end_sec + 1e-6:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
        if count is not None and len(frames) >= count:
            break
    return np.stack(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)


def _sharpness_by_episode(repo: str, rows: list[Original]) -> dict[int, float]:
    """候補の 1 コマ目の鮮明さ。**摂動の強さを見分けるのに使う。**

    候補は同じ mp4 に入っていることが多いので、**ファイルごとに 1 回だけ
    開いて、その中で seek する。** 1 本ずつ開き直すと HTTPS の接続と
    ヘッダ読みを候補数ぶん繰り返すことになる（10 本で 10 回）。
    """
    import cv2

    grouped: dict[tuple[int, int], list[Original]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row.chunk, row.file)].append(row)

    scores: dict[int, float] = {}
    for key, group in grouped.items():
        container, stream = _open_video(_video_url(repo, group[0]))
        try:
            for row in sorted(group, key=lambda item: item.start_sec):
                frames = _read_span(container, stream, row.start_sec,
                                    row.start_sec + 0.01, count=1)
                if not len(frames):
                    continue
                gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
                scores[row.episode] = float(
                    cv2.Laplacian(gray, cv2.CV_64F).var())
        finally:
            container.close()
    return scores


def _choose_original(repo: str, rows: list[Original], rank: str,
                     explicit: int | None, candidates: int = 10) -> Original:
    """比べる手本を 1 本決める。

    Parameters
    ----------
    rank : str
        `clean` … 短いほうから `candidates` 本を見て、**いちばん鮮明な 1 本**。
        学習データには Sensor Noise（ぼかし）を掛けた回が混ざっており、
        **最短の回がたまたまぼけていると、見た目の比較にならない。**
        `shortest` … 最短の 1 本（人の最速）。
        `median` … 中央値の 1 本（人の並）。
    explicit : int | None
        エピソード番号を直接指定する場合。`rank` より優先する。
    """
    if explicit is not None:
        for row in rows:
            if row.episode == explicit:
                return row
        raise SystemExit(f"その番号の手本が無い: ep{explicit}")
    if rank == "shortest":
        return rows[0]
    if rank == "median":
        return rows[len(rows) // 2]

    shortlist = rows[:candidates]
    scores = _sharpness_by_episode(repo, shortlist)
    for row in shortlist:
        print(f"    ep{row.episode:<6d} {row.length:>4d} 歩  "
              f"鮮明さ {scores.get(row.episode, 0.0):8.1f}")
    return max(shortlist, key=lambda row: scores.get(row.episode, 0.0))


def _panel(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """1 枚に見出しを付けて、決まった大きさに揃える。**RGB のまま返す。**"""
    import cv2

    image = cv2.resize(np.ascontiguousarray(frame), (_PANEL_SIZE, _PANEL_SIZE),
                       interpolation=cv2.INTER_NEAREST)
    banner = np.zeros((_LABEL_HEIGHT, _PANEL_SIZE, 3), np.uint8)
    cv2.putText(banner, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, subtitle, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 220, 140), 1, cv2.LINE_AA)
    return np.vstack([banner, image])


def _write_video(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    """RGB のコマ列を mp4 に書き出す。**制御が 20 Hz なので fps 20 が実時間。**

    1 コマずつ受け取って書く。**動画 1 本ぶんを list で抱えると、
    並べた絵では 400 コマ × 約 1 MB で 400 MB 近くが同時に生きる。**

    書き出しの口が開けたかを確かめる。**確かめないと、コーデックが無い環境で
    0 バイトの mp4 を「書けた」と表示して終わる。**
    """
    import cv2

    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise SystemExit(f"書くコマが無い: {path}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise SystemExit(f"mp4 の書き出しを開けない（コーデックが無い）: {path}")
    written = 0
    for frame in (first, *iterator):
        writer.write(cv2.cvtColor(np.ascontiguousarray(frame),
                                  cv2.COLOR_RGB2BGR))
        written += 1
    writer.release()
    print(f"  → {path}（{written} コマ）")


def _compose(original: np.ndarray, generated: np.ndarray, original_steps: int,
             generated_steps: int) -> Iterator[np.ndarray]:
    """左右に並べたコマを 1 枚ずつ返す。

    **短いほうは最後のコマで埋めて、長さを揃える。** 埋めずに切ると
    「台本のほうが短い」ことが画面から消えてしまう。
    """
    for index in range(max(len(original), len(generated))):
        left = _panel(
            original[min(index, len(original) - 1)],
            f"moto no tehon (human)  {original_steps} steps",
            f"step {min(index, original_steps - 1) + 1}"
            + ("" if index < len(original) else "  (owari)"))
        right = _panel(
            generated[min(index, len(generated) - 1)],
            f"daihon (scripted)  {generated_steps} steps",
            f"step {min(index, generated_steps - 1) + 1}"
            + ("" if index < len(generated) else "  (owari)"))
        divider = np.full((left.shape[0], 4, 3), 60, np.uint8)
        yield np.hstack([left, divider, right])


def _contact_sheet(original: np.ndarray, generated: np.ndarray,
                   columns: int = 10) -> np.ndarray:
    """等間隔に抜いたコマを 2 段に並べた 1 枚絵。**RGB のまま返す。**

    **動画より先にこれを見る。** 掴み損ね・行き過ぎ・ぶつけは、静止画を
    並べたほうが早く見つかる。上段が元の手本、下段が台本である。
    """
    def _row(frames: np.ndarray, label: str) -> np.ndarray:
        picked = np.linspace(0, len(frames) - 1, columns).astype(int)
        return np.hstack([
            _panel(frames[index], label if position == 0 else "",
                   f"step {index + 1}/{len(frames)}")
            for position, index in enumerate(picked)
        ])

    return np.vstack([_row(original, "moto no tehon (human)"),
                      _row(generated, "daihon (scripted)")])


def _write_image(path: Path, frame: np.ndarray) -> None:
    """RGB の 1 枚絵を png に書き出す。**入れ替えはここだけ。**"""
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(frame),
                                        cv2.COLOR_RGB2BGR))
    print(f"  → {path}")


def _plot_paths(episodes: list[tuple[str, SavedEpisode]], destination: Path
                ) -> None:
    """手先の軌跡を 3 面で描く。**ガタつきは動画より図のほうが早く見つかる。**

    左から「上から見た経路」「高さの推移」「1 ステップごとの向きの変化」。
    3 枚目が肝で、**向きが1 ステップごとに何度折れているかがそのままガタつき**である。
    直す前は中央値 6.7 度、直したあとは 2.1 度だった。

    Parameters
    ----------
    episodes : list[tuple[str, SavedEpisode]]
        （見出し, 読み込んだ 1 本）の並び。2 本渡すと重ねて描く。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2a9d8f", "#d1495b", "#457b9d")
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for (label, episode), color in zip(episodes, colors):
        track = np.asarray(episode.state)[:, :3]
        axes[0].plot(track[:, 0], track[:, 1], color=color, lw=1.4,
                     label=f"{label}  {episode.steps} steps")
        axes[1].plot(np.arange(len(track)), track[:, 2], color=color, lw=1.4)
        turn = _heading_change(track)
        axes[2].plot(turn, color=color, lw=1.0, alpha=0.85,
                     label=f"{label}  median {np.median(turn):.1f} deg")

    axes[0].set_title("hand path, seen from above (x-y)")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal"); axes[0].legend()
    axes[1].set_title("height over time (z)")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("z [m]")
    axes[2].set_title("direction change per step")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("degrees"); axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)
    print(f"  → {destination}")


def _heading_change(track: np.ndarray) -> np.ndarray:
    """1 ステップごとに進む向きが何度折れたか。**これがそのままガタつきである。**"""
    vectors = np.diff(track, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    moving = lengths > 1e-4
    unit = vectors[moving] / lengths[moving][:, None]
    return np.degrees(np.arccos(
        np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)))


def _report_actions(original_length: int, generated: SavedEpisode) -> None:
    """ステップ数と行動の癖を並べて印字する。

    **台本の行動は振り切れている**（速く動かすため）。元の手本は人の操作なので
    中間の値が多い。**この違いは学習に効きうる**ので数字で出しておく。
    """
    actions = np.asarray(generated.action)
    saturated = float(np.mean(np.abs(actions[:, :3]) > 0.99))
    print("\n--- ステップ数と行動 ---")
    print(f"  元の手本   {original_length:4d} 歩")
    print(f"  台本       {generated.steps:4d} 歩 "
          f"（{generated.steps / original_length:.2f} 倍）")
    print(f"  台本の行動のうち、位置の指令が振り切れている割合 {saturated:.1%}")
    print(f"  台本の行動の絶対値の平均 {np.abs(actions[:, :3]).mean():.3f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generated", type=Path, required=True,
                        help="scripted_demo.py record が書いた npz")
    parser.add_argument("--task", required=True,
                        help="指示文（元の手本を引くのに使う）")
    parser.add_argument("--episodes-parquet", type=Path, required=True,
                        help="lerobot/libero_plus の meta/episodes の parquet")
    parser.add_argument("--repo", default="lerobot/libero_plus",
                        help="元データの HuggingFace リポジトリ")
    parser.add_argument("--rank", choices=("clean", "shortest", "median"),
                        default="clean",
                        help="元の手本のどれを使うか。既定の clean は「短いほうから "
                             "何本かのうち、いちばん鮮明な 1 本」。**学習データには "
                             "ぼかしを掛けた回が混ざっており、最短の回が"
                             "たまたまぼけていると見た目の比較にならない**")
    parser.add_argument("--original-episode", type=int, default=None,
                        help="元の手本のエピソード番号を直接指定する（--rank より優先）")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="比べたいもう 1 本の npz（直す前の軌道など）。"
                             "渡すと経路の図に重ねて描く")
    parser.add_argument("--output", type=Path, default=Path("outputs/compare"),
                        help="書き出し先")
    parser.add_argument("--fps", type=int, default=20,
                        help="mp4 のコマ速度（既定 20＝実時間）")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    generated = SavedEpisode(args.generated)
    print(f"台本の軌道: {generated.steps} 歩  成功 {generated.success}")
    print(f"指示文: {generated.instruction}")

    rows = _list_originals(args.episodes_parquet, args.task)
    print(f"元の手本 {len(rows)} 本  最短 {rows[0].length} 歩 "
          f"最長 {rows[-1].length} 歩"
          f"  300 ステップ以下 {sum(1 for row in rows if row.length <= 300)} 本")
    chosen = _choose_original(args.repo, rows, args.rank, args.original_episode)
    print(f"  → ep{chosen.episode}（{chosen.length} 歩）"
          f" chunk-{chosen.chunk:03d}/file-{chosen.file:03d}"
          f" {chosen.start_sec:.2f}〜{chosen.end_sec:.2f} 秒")

    container, stream = _open_video(_video_url(args.repo, chosen))
    try:
        original = _read_span(container, stream, chosen.start_sec,
                              chosen.end_sec, chosen.length)
    finally:
        container.close()
    print(f"  {len(original)} コマ読めた（期待 {chosen.length} コマ）")

    front = generated.front
    stem = f"ep{chosen.episode}_vs_{args.generated.stem}"
    _write_video(args.output / f"original_{stem}.mp4", original, args.fps)
    _write_video(args.output / f"generated_{stem}.mp4", front, args.fps)
    _write_video(args.output / f"compare_{stem}.mp4",
                 _compose(original, front, chosen.length, generated.steps),
                 args.fps)
    _write_image(args.output / f"sheet_{stem}.png",
                 _contact_sheet(original, front))

    tracks = [("scripted", generated)]
    if args.baseline is not None:
        tracks.append(("baseline", SavedEpisode(args.baseline)))
    _plot_paths(tracks, args.output / f"paths_{stem}.png")

    _report_actions(chosen.length, generated)


if __name__ == "__main__":
    main()
