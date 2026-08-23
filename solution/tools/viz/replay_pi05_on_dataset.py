#!/usr/bin/env python3
"""学習データの実フレームを π0.5 に流し、正解の行動と並べた動画を作る。

[infer_pi05_once.py](infer_pi05_once.py) は**ダミーの真っ黒な画像**で「通るか」
だけを見る道具である。こちらは**本物のフレーム**を入れて、
**今の重みが何をしようとしているのか**を見る。

    左: front カメラ  |  右: wrist カメラ  |  下: 7 次元の行動を GT と並べた棒

**シミュレータは回さない**（teacher forcing）。各時点の入力は必ず学習データの
実フレームなので、**「その時点の判断が合っているか」だけ**が出る。誤差が積み
上がって別の場所へ行ってしまう分は含まれない——実機で失敗する原因が
「判断」なのか「積み上がり」なのかを切り分けるための見方である
（[visualize_actions.py](visualize_actions.py) と同じ考え方。あちらは提出物の
`MyPolicy`＝SmolVLA を読み、図を 1 枚描く。こちらは**π0.5 のチェックポイントを
直接読み**、動画を作る）。

## 2 つの見方

| `--mode` | 何をするか | 推論の回数 |
|---|---|---|
| `chunk`（既定） | ある 1 フレームから**先 50 ステップぶんを一気に予測**し、その先の正解と並べる | **1 回** |
| `stream` | 本番と同じく `n_action_steps`（10）歩ごとに予測し直す | フレーム数 ÷ 10 |

**CPU では 1 推論に 3 分以上かかる**（このマシンで 195 秒／1 コア、
`infer_pi05_once.py` 実測）。まず `chunk` で 1 回だけ走らせて中身を見て、
GPU が空いてから `stream` に進むのが安い。

## 使い方

学習に使ったイメージの中で動かす（ホストに torch も lerobot も無い）。

    # 実データ 1 フレームから先 50 ステップを予測して動画にする（推論 1 回・約 10 分）
    docker run --rm -it --memory=13g --cpus=1 \\
        --user "$(id -u):$(id -g)" \\
        -v "$PWD:/repo" -v "$HOME/.cache/huggingface:/hf_home" \\
        parc2026-train-pi05 \\
        python /repo/solution/tools/viz/replay_pi05_on_dataset.py \\
            /repo/work/train/pi05_full_local_old/checkpoints/000500/pretrained_model \\
            --task "pick up the black bowl" --output /repo/work/replay

    # 素の LIBERO（lerobot/libero）で見る。**カメラの列名が違う**
    ... --dataset-root /hf_home/lerobot/lerobot/libero --dataset-cameras image,image2

Reads:
  - <weights>/config.json, model.safetensors, policy_pre/postprocessor*
  - <dataset-root>/meta/episodes/**.parquet（エピソードの区間と指示文）
  - <dataset-root>/data/chunk-XXX/file-YYY.parquet（正解の行動と状態）
  - <dataset-root>/videos/observation.images.<カメラ>/chunk-XXX/file-YYY.mp4
Generates:
  - <output>/replay_ep<N>_<mode>.mp4（左右のカメラ＋行動の棒）
  - <output>/replay_ep<N>_<mode>.csv（GT と予測の生の値。描き直しに使える）
  - 標準出力に次元ごとの誤差
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# infer_pi05_once の読み込み処理をそのまま使う（**写しを作らない**）。
# train_pi05.py が train.py から関数を借りているのと同じ形である。
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_pi05_once import load_policy, resolve_device  # noqa: E402

# 方策側の受け口の名前。データセット側の列名（`--dataset-cameras`）を
# **この順で**ここへ写す。素の LIBERO は image / image2 なので違う。
_POLICY_CAMERAS = ("observation.images.front", "observation.images.wrist")

# 画像を [0, 1] に直すための割る数（policy_server.py と同じ）。
_IMAGE_SCALE = 255.0

# 行動 7 次元の名前。6 番目はバンバン制御のグリッパで性質が違う。
_ACTION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "grip")

# 動画の下に付ける棒グラフの寸法（画素）。
_ROW_HEIGHT = 26
_PANEL_MARGIN = 14
_LABEL_WIDTH = 54
_HEADER_HEIGHT = 52

# 色（RGB のまま持ち回り、**書き出す直前に 1 回だけ BGR へ入れ替える**。
# 途中で入れ替えると片方だけ赤と青が逆になる——visualize_demo.py の教訓）。
_COLOR_TRUTH = (70, 200, 90)
_COLOR_PREDICTED = (250, 140, 40)
_COLOR_TEXT = (235, 235, 235)
_COLOR_BACKGROUND = (28, 28, 32)
_COLOR_MARK = (240, 80, 80)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", type=Path,
                        help="チェックポイントの pretrained_model ディレクトリ")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="LeRobot 形式のデータセット（既定は "
                             "$HF_HOME/lerobot/lerobot/libero_plus）")
    parser.add_argument("--dataset-cameras", type=str, default="front,wrist",
                        help="データセット側のカメラ列名を front,wrist の順で。"
                             "素の LIBERO は image,image2")
    parser.add_argument("--episode", type=int, default=None,
                        help="エピソード番号。省略時は --task から選ぶ")
    parser.add_argument("--task", type=str, default=None,
                        help="指示文の前方一致。最初に当たったエピソードを使う")
    parser.add_argument("--mode", type=str, default="chunk",
                        choices=("chunk", "stream"),
                        help="chunk（既定）は推論 1 回で先 50 ステップを予測。"
                             "stream は本番と同じく 10 ステップごとに予測し直す")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="エピソードの何フレーム目から始めるか")
    parser.add_argument("--frames", type=int, default=0,
                        help="何フレームぶん見るか（0 なら --mode の既定）")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "cpu", "cuda"))
    parser.add_argument("--load-mode", type=str, default="lowmem",
                        choices=("lowmem", "builtin"),
                        help="infer_pi05_once.py と同じ意味。"
                             "builtin は 18GB 級を要求する")
    parser.add_argument("--num-inference-steps", type=int, default=0,
                        help="flow matching の刻み数。0 なら設定のまま（10）")
    parser.add_argument("--dry-run", action="store_true",
                        help="**方策を読まずに**データの取り出しと動画作りだけ試す"
                             "（予測は 0 で埋める）。数秒で終わるので、"
                             "3 分×N の推論を始める前の配管の確認に使う")
    parser.add_argument("--output", type=Path, default=Path("/out/replay"),
                        help="動画と csv の置き場")
    parser.add_argument("--fps", type=int, default=10,
                        help="動画の速さ。データは 20 fps なので既定は半分の速さ")
    parser.add_argument("--scale", type=int, default=3,
                        help="128〜256px の画を何倍に伸ばすか")
    return parser.parse_args()


def _default_dataset_root() -> Path:
    """`$HF_HOME` から libero_plus の置き場を組み立てる。

    LeRobot は Hub のキャッシュ（`hub/datasets--*`）とは別に
    `$HF_HOME/lerobot/<repo_id>/` へ展開する。**動画と行動が揃っているのは
    後者だけ**で、`hub/` 側にはメタデータしか落ちていない場合がある。
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "lerobot" / "lerobot" / "libero_plus"


def _load_episode_row(dataset_root: Path, episode: int | None, task: str | None):
    """見るエピソードを 1 本選び、その 1 行を返す。

    Parameters
    ----------
    dataset_root : Path
        データセットの根。
    episode : int | None
        エピソード番号。`None` なら `task` で選ぶ。
    task : str | None
        指示文の前方一致。両方 `None` なら先頭のエピソード。

    Returns
    -------
    row : pandas.Series
        `meta/episodes` の 1 行（区間・動画のファイル番号・長さ・指示文）。

    Reads:
      - <dataset_root>/meta/episodes/**/*.parquet
    """
    import pandas as pd

    files = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"エラー: {dataset_root} に meta/episodes がありません")
    episodes = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    episodes["task"] = episodes["tasks"].apply(
        lambda cell: cell if isinstance(cell, str) else str(cell[0])
    )

    if episode is not None:
        matched = episodes[episodes["episode_index"] == episode]
        if matched.empty:
            raise SystemExit(f"エラー: エピソード {episode} がありません")
        return matched.iloc[0]

    if task is not None:
        matched = episodes[episodes["task"].str.startswith(task.lower())]
        if matched.empty:
            raise SystemExit(f"エラー: 指示文が前方一致しません: {task!r}")
        return matched.iloc[0]

    return episodes.iloc[0]


def _load_ground_truth(dataset_root: Path, row, start: int,
                       count: int) -> tuple[np.ndarray, np.ndarray]:
    """正解の行動と状態を、選んだ区間だけ取り出す。

    Returns
    -------
    truth : np.ndarray
        `(T, 7)` の正解行動。
    states : np.ndarray
        `(T, 8)` の状態（方策への入力）。

    Reads:
      - <dataset_root>/data/chunk-XXX/file-YYY.parquet
    """
    import pandas as pd

    path = (dataset_root / "data"
            / f"chunk-{int(row['data/chunk_index']):03d}"
            / f"file-{int(row['data/file_index']):03d}.parquet")
    # **1 ファイルに何百本ものエピソードが入っている**（54 MB）ので、
    # 読む前に絞る。列も 4 つだけ取る。
    table = pd.read_parquet(
        path,
        columns=["episode_index", "frame_index", "action", "observation.state"],
        filters=[("episode_index", "==", int(row["episode_index"]))],
    )
    episode = table.sort_values("frame_index").iloc[start:start + count]

    def stack(column: str, width: int) -> np.ndarray:
        return np.stack([np.asarray(value, dtype=np.float32).reshape(width)
                         for value in episode[column].to_numpy()])

    return stack("action", 7), stack("observation.state", 8)


def _decode_frames(dataset_root: Path, camera: str, row, skip: int,
                   count: int) -> list[np.ndarray]:
    """1 つのカメラの動画から、選んだ区間のフレームを取り出す。

    **データセットの動画は 1 ファイルに何百本ものエピソードを繋げて入れて
    ある。** そのため `from_timestamp` へ跳んでから数えて切り出す。

    Parameters
    ----------
    camera : str
        データセット側の列名（`front` / `image` など）。
    skip : int
        エピソードの先頭から捨てるフレーム数（`--start-frame`）。
    count : int
        取り出す枚数。

    Returns
    -------
    frames : list[np.ndarray]
        `(H, W, 3)` uint8 の RGB。

    Reads:
      - <dataset_root>/videos/observation.images.<camera>/chunk-XXX/file-YYY.mp4
    """
    import av

    prefix = f"videos/observation.images.{camera}"
    path = (dataset_root / prefix
            / f"chunk-{int(row[f'{prefix}/chunk_index']):03d}"
            / f"file-{int(row[f'{prefix}/file_index']):03d}.mp4")
    start_sec = float(row[f"{prefix}/from_timestamp"])

    container = av.open(str(path))
    stream = container.streams.video[0]
    container.seek(int(start_sec / stream.time_base), stream=stream)

    frames: list[np.ndarray] = []
    seen = 0
    for frame in container.decode(stream):
        if float(frame.pts * stream.time_base) < start_sec - 1e-6:
            continue
        seen += 1
        if seen <= skip:
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= count:
            break
    container.close()
    return frames


def _build_batch(images: list[np.ndarray], state: np.ndarray, task: str,
                 preprocessor, torch_module) -> dict:
    """1 フレームの観測を、方策が食べられる batch に直す。

    **`LiberoProcessorStep` は通さない。** あれは評価環境の生画像（上下逆）を
    学習データの向きへ回すためのものである。ここでの入力は学習データその
    ものなので、もう一度回すと逆さになる。
    """
    def to_image(image: np.ndarray):
        tensor = torch_module.from_numpy(np.ascontiguousarray(image))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(dtype=torch_module.float32) / _IMAGE_SCALE

    batch = {key: to_image(image) for key, image in zip(_POLICY_CAMERAS, images)}
    batch["observation.state"] = torch_module.from_numpy(
        np.ascontiguousarray(state)).to(dtype=torch_module.float32).unsqueeze(0)
    batch["task"] = [task]
    return preprocessor(batch)


def _run_replay(policy, preprocessor, postprocessor, frames: dict[str, list],
                states: np.ndarray, task: str,
                torch_module) -> tuple[np.ndarray, list[float]]:
    """実フレームを順に流し、各歩の行動と所要時間を返す。

    `select_action` は内部にキューを持つので、**素直に 1 ステップずつ呼べば本番と
    同じ切り替わり方**になる。重い推論が走るのは `n_action_steps` 歩ごとの
    1 回だけで、残りは作り置きを出すだけである（そこは 0.01 秒で終わる）。

    Returns
    -------
    predicted : np.ndarray
        `(T, 7)` の予測行動。
    seconds : list[float]
        各歩の所要時間。**大きい値の歩が推論が走った歩**である。
    """
    policy.reset()
    cameras = list(frames)
    total = len(states)

    predicted: list[np.ndarray] = []
    seconds: list[float] = []
    for index in range(total):
        images = [frames[camera][index] for camera in cameras]
        batch = _build_batch(images, states[index], task, preprocessor, torch_module)
        started = time.perf_counter()
        with torch_module.no_grad():
            action = policy.select_action(batch)
        action = postprocessor(action)
        elapsed = time.perf_counter() - started

        predicted.append(np.asarray(action[0].detach().to("cpu").numpy(),
                                    dtype=np.float32).reshape(7))
        seconds.append(elapsed)
        mark = "  ← 推論" if elapsed > 1.0 else ""
        print(f"  {index + 1:>3}/{total}  {elapsed:>7.2f} 秒{mark}", flush=True)

    return np.stack(predicted), seconds


def _draw_bars(canvas: np.ndarray, truth: np.ndarray, predicted: np.ndarray,
               limit: float, top: int, width: int) -> None:
    """7 次元ぶんの棒を、正解と予測で 2 本ずつ描く。

    棒は中央から左右に伸びる（行動は符号付きの増分だからである）。
    上が正解（緑）、下が予測（橙）。
    """
    import cv2

    center = _LABEL_WIDTH + (width - _LABEL_WIDTH - _PANEL_MARGIN) // 2
    half = (width - _LABEL_WIDTH - _PANEL_MARGIN) // 2 - 2

    for dimension, name in enumerate(_ACTION_NAMES):
        row_top = top + dimension * _ROW_HEIGHT
        cv2.putText(canvas, name, (6, row_top + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, _COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.line(canvas, (center, row_top + 3), (center, row_top + _ROW_HEIGHT - 3),
                 (90, 90, 96), 1)
        for offset, value, color in ((4, float(truth[dimension]), _COLOR_TRUTH),
                                     (14, float(predicted[dimension]),
                                      _COLOR_PREDICTED)):
            length = int(max(-1.0, min(1.0, value / limit)) * half)
            cv2.rectangle(canvas, (center, row_top + offset),
                          (center + length, row_top + offset + 7), color, -1)
        cv2.putText(canvas, f"{truth[dimension]:+.2f} / {predicted[dimension]:+.2f}",
                    (center + half - 96, row_top + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, _COLOR_TEXT, 1, cv2.LINE_AA)


def _compose_frame(images: list[np.ndarray], truth: np.ndarray,
                   predicted: np.ndarray, limit: float, index: int, task: str,
                   inferred: bool, scale: int) -> np.ndarray:
    """左右のカメラと棒グラフを 1 枚に組む（RGB のまま返す）。"""
    import cv2

    strip = np.concatenate(images, axis=1)
    strip = cv2.resize(strip, (strip.shape[1] * scale, strip.shape[0] * scale),
                       interpolation=cv2.INTER_NEAREST)
    width = strip.shape[1]
    panel_height = _HEADER_HEIGHT + len(_ACTION_NAMES) * _ROW_HEIGHT + _PANEL_MARGIN
    panel = np.full((panel_height, width, 3), _COLOR_BACKGROUND, dtype=np.uint8)

    cv2.putText(panel, f"step {index}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, _COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, "GT (green) / pi05 (orange)", (94, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_TEXT, 1, cv2.LINE_AA)
    if inferred:
        cv2.putText(panel, "* inference here", (width - 150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_MARK, 1, cv2.LINE_AA)
    cv2.putText(panel, task[:88], (8, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, _COLOR_TEXT, 1, cv2.LINE_AA)
    _draw_bars(panel, truth, predicted, limit, _HEADER_HEIGHT, width)

    return np.concatenate([strip, panel], axis=0)


def _write_video(destination: Path, images: dict[str, list], truth: np.ndarray,
                 predicted: np.ndarray, seconds: list[float], task: str,
                 start_frame: int, fps: int, scale: int) -> None:
    """組んだ絵を mp4 に書き出す。

    **BGR への入れ替えは書き出す直前の 1 回だけ**にしてある。
    """
    import cv2

    limit = max(0.05, float(np.abs(np.concatenate([truth, predicted])).max()))
    cameras = list(images)
    writer = None
    destination.parent.mkdir(parents=True, exist_ok=True)

    for index in range(len(truth)):
        frame = _compose_frame([images[camera][index] for camera in cameras],
                               truth[index], predicted[index], limit,
                               start_frame + index, task,
                               seconds[index] > 1.0, scale)
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(destination),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (width, height))
        writer.write(frame[:, :, ::-1])
    if writer is not None:
        writer.release()


def _write_csv(destination: Path, truth: np.ndarray, predicted: np.ndarray,
               seconds: list[float], start_frame: int) -> None:
    """GT と予測の生の値を残す（描き直しに推論をやり直さなくて済む）。"""
    header = ("frame,seconds,"
              + ",".join(f"gt_{name}" for name in _ACTION_NAMES) + ","
              + ",".join(f"pi05_{name}" for name in _ACTION_NAMES))
    lines = [header]
    for index in range(len(truth)):
        values = list(truth[index]) + list(predicted[index])
        lines.append(f"{start_frame + index},{seconds[index]:.3f},"
                     + ",".join(f"{value:.6f}" for value in values))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(truth: np.ndarray, predicted: np.ndarray,
            title: str | None = None) -> None:
    """次元ごとの誤差と相関を出す。

    **`visualize_actions.py` の同名の関数と同じ書式**にしてある（あちらは
    matplotlib を import するため、学習イメージからは読み込めない）。

    Parameters
    ----------
    title : str | None
        表の上に出す見出し。区間を分けて 2 回呼ぶときに使う。
    """
    print()
    if title is not None:
        print(f"  --- {title}（{len(truth)} 歩） ---")
    print("  次元   平均絶対誤差   GT の標準偏差   誤差/標準偏差   相関")
    for dimension, name in enumerate(_ACTION_NAMES):
        error = float(np.abs(predicted[:, dimension] - truth[:, dimension]).mean())
        deviation = float(truth[:, dimension].std())
        ratio = error / deviation if deviation > 1e-8 else float("nan")
        if deviation > 1e-8 and predicted[:, dimension].std() > 1e-8:
            correlation = float(np.corrcoef(truth[:, dimension],
                                            predicted[:, dimension])[0, 1])
        else:
            correlation = float("nan")
        print("  %-5s %11.4f %14.4f %14.2f %8.3f"
              % (name, error, deviation, ratio, correlation))


def _print_report_notes() -> None:
    """表の読み方を最後に 1 度だけ出す。"""
    print()
    print("  相関は**向きが合っているか**、誤差/標準偏差 は**大きさが合っているか**を見る。")
    print("  相関が高いのに誤差/標準偏差 が 1 前後なら、行き先は合っていて力加減が足りない。")
    print("  GT の標準偏差がほぼ 0 の軸（動いていない軸）は、相関を見ても意味がない。")


def _resolve_length(mode: str, frames: int, chunk_size: int,
                    n_action_steps: int, available: int) -> int:
    """何フレームぶん流すかを決める。

    `chunk` は「1 回の推論で作った列を全部見る」ので `chunk_size`、
    `stream` は既定で 30 ステップ（推論 3 回）にしてある。
    """
    if frames > 0:
        return min(frames, available)
    default = chunk_size if mode == "chunk" else max(n_action_steps * 3, 30)
    return min(default, available)


def _chunk_params_from_config(weights: Path) -> tuple[int, int]:
    """`chunk_size` と `n_action_steps` をチェックポイント自身から読む。

    **`--dry-run` 用**である。方策を組み立てずに（5 分かからずに）同じ値が
    要るときに使う。出典は `config.json` そのものなので、方策から取るのと
    同じ値になる。

    Reads:
      - <weights>/config.json
    """
    config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    return int(config["chunk_size"]), int(config["n_action_steps"])


def _print_run_header(weights: Path, dataset_root: Path, row, task: str,
                      device: str, mode: str, length: int,
                      n_action_steps: int, dry_run: bool) -> None:
    """何をどの設定で流すのか、走る前に 1 か所へまとめて出す。"""
    inferences = 0 if dry_run else -(-length // n_action_steps)
    print(f"[replay] 重み         {weights}")
    print(f"[replay] データ       {dataset_root}")
    print(f"[replay] エピソード   {int(row['episode_index'])}"
          f"（全 {int(row['length'])} フレーム）")
    print(f"[replay] 指示文       {task!r}")
    print(f"[replay] device       {device}")
    print(f"[replay] mode         {mode}"
          f"（n_action_steps={n_action_steps}、{length} フレーム）")
    if dry_run:
        print("[replay] **--dry-run なので推論は走らない**")
    else:
        print(f"[replay] **推論は {inferences} 回走る。**"
              "CPU なら 1 回 3 分前後かかる")


def main() -> None:
    args = _parse_args()
    weights: Path = args.weights.resolve()
    if not (weights / "config.json").is_file():
        raise SystemExit(f"エラー: {weights} に config.json がありません")
    dataset_root: Path = (args.dataset_root or _default_dataset_root()).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"エラー: データセットがありません: {dataset_root}")

    # 採点環境と同じ「外へ出ない」条件で試す。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    row = _load_episode_row(dataset_root, args.episode, args.task)
    task = str(row["task"])

    torch = None
    policy = preprocessor = postprocessor = None
    loading_seconds = 0.0
    if args.dry_run:
        device = args.device
        chunk_size, production_steps = _chunk_params_from_config(weights)
    else:
        import torch  # noqa: PLC0415  （--dry-run では torch を触らない）

        device = resolve_device(args.device)
        print("[replay] 方策を読み込み中（CPU では 5 分ほどかかる）")
        loading_started = time.perf_counter()
        policy, preprocessor, postprocessor, tally = load_policy(
            weights, device, load_mode=args.load_mode,
            num_inference_steps=args.num_inference_steps,
        )
        loading_seconds = time.perf_counter() - loading_started
        if tally is not None and tally["missing_parameters"] > 0:
            raise SystemExit(f"エラー: 写されなかった重みが "
                             f"{tally['missing_parameters']} 個あります")
        chunk_size = int(policy.config.chunk_size)
        production_steps = int(policy.config.n_action_steps)
    n_action_steps = production_steps

    # chunk では作り置きの列を 50 ステップぜんぶ使いたいので、その場で伸ばす。
    # **`_run_replay` の中の `policy.reset()` より前に書き換えること。**
    # `reset` が `deque(maxlen=n_action_steps)` を作り、`select_action` は
    # `[:, :n_action_steps]` で切るので、後から変えても 10 ステップしか残らない
    # （lerobot v0.6.0 の PI05Policy.reset / select_action 実測）。
    if args.mode == "chunk":
        n_action_steps = chunk_size
        if policy is not None:
            policy.config.n_action_steps = chunk_size

    available = int(row["length"]) - args.start_frame
    if available <= 0:
        raise SystemExit(f"エラー: --start-frame が長さ {int(row['length'])} を"
                         "超えています")
    length = _resolve_length(args.mode, args.frames, chunk_size,
                             n_action_steps, available)

    _print_run_header(weights, dataset_root, row, task, device, args.mode,
                      length, n_action_steps, args.dry_run)
    if not args.dry_run:
        print(f"[replay] 読み込み     {loading_seconds:.1f} 秒（{args.load_mode}）")

    truth, states = _load_ground_truth(dataset_root, row, args.start_frame, length)
    frames = {camera: _decode_frames(dataset_root, camera, row,
                                     args.start_frame, length)
              for camera in (name.strip() for name in args.dataset_cameras.split(","))}
    length = min(length, len(truth), *(len(images) for images in frames.values()))
    truth, states = truth[:length], states[:length]
    frames = {camera: images[:length] for camera, images in frames.items()}
    print(f"[replay] 取り出した   {length} フレーム "
          f"({' / '.join(f'{k}:{len(v)}' for k, v in frames.items())})")

    if args.dry_run:
        print("[replay] **--dry-run: 予測は 0 で埋める。**動画の配管だけを見る")
        predicted = np.zeros_like(truth)
        seconds = [0.0] * length
    else:
        predicted, seconds = _run_replay(policy, preprocessor, postprocessor,
                                         frames, states, task, torch)

    stem = f"replay_ep{int(row['episode_index'])}_{args.mode}"
    video = args.output / f"{stem}.mp4"
    _write_video(video, frames, truth, predicted, seconds, task,
                 args.start_frame, args.fps, args.scale)
    _write_csv(args.output / f"{stem}.csv", truth, predicted, seconds,
               args.start_frame)

    # **本番が使うのは先頭 n_action_steps 歩だけ**なので、chunk では
    # そこと外挿の部分を分けて数える（混ぜると外挿の崩れに引きずられる）。
    if args.mode == "chunk" and length > production_steps:
        _report(truth[:production_steps], predicted[:production_steps],
                f"本番が実際に使う先頭 {production_steps} 歩")
        _report(truth[production_steps:], predicted[production_steps:],
                f"{production_steps + 1} 歩目以降（作り置きの外挿。本番では使わない）")
    else:
        _report(truth, predicted)
    _print_report_notes()
    print()
    print(f"  動画: {video}")
    print(f"  数値: {args.output / (stem + '.csv')}")


if __name__ == "__main__":
    main()
