#!/usr/bin/env python3
"""学習データの正解行動と、方策の出力を重ねて描く。

第 7 回講義資料「可視化ファースト」の 2 番目と 4 番目に相当する。

- **GT action vs policy action の重ね書き**：方策が各時点で正しい行動を
  出せているかを見る。成功率だけでは「惜しかった」と「何もできていない」を
  区別できない
- **chunk 境界の可視化**：50 ステップごとの継ぎ目で軌道が跳んでいないかを見る

学習データの観測をそのまま入力する（teacher forcing）。シミュレータを
回さないため、**誤差の蓄積を含まない「その時点での正しさ」**が測れる。
実環境で失敗する原因が「各時点の判断」なのか「誤差の蓄積」なのかを
切り分けられる。

コンテナ内で実行すること。

    /opt/subvenv2/bin/python visualize_actions.py --episode 2
    /opt/subvenv2/bin/python visualize_actions.py --task "put the bowl on the stove"

Reads:
  - lerobot/libero_plus のメタデータ・動画・行動（HF キャッシュ）
  - /sub/policy_server.py（MyPolicy）
Generates:
  - <output>/actions_ep<N>.png（重ね書きの図）
  - 標準出力に次元ごとの誤差
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import av
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SNAPSHOT = Path(
    "/tmp/hf_probe/hub/datasets--lerobot--libero_plus/snapshots/"
    "f3f49f426d75030177b18778374005bc12ccd588"
)
_SERVER_PATH = Path("/sub/policy_server.py")

# 行動 7 次元の内訳。6 次元目はバンバン制御のグリッパで、性質が違う。
_DIMENSION_GROUPS = (
    ("EEF translation (delta)", (0, 1, 2), ("dx", "dy", "dz")),
    ("EEF rotation (delta)", (3, 4, 5), ("drx", "dry", "drz")),
    ("Gripper", (6,), ("grip",)),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=int, default=None,
                        help="エピソード番号。省略時は --task から選ぶ")
    parser.add_argument("--task", type=str, default=None,
                        help="タスクの指示文。前方一致で最初のエピソードを使う")
    parser.add_argument("--max-frames", type=int, default=150,
                        help="描画するフレーム数の上限")
    parser.add_argument("--output", type=Path, default=Path("/sub/vis"),
                        help="図の出力先")
    return parser.parse_args()


def _load_episode_row(episode: int | None, task: str | None) -> pd.Series:
    """描画対象のエピソードを 1 本選ぶ。

    動画は chunk/file に分かれているため、**file-000 に入っているもの**に
    限定する。他のファイルを引くと追加のダウンロードが発生する。
    """
    episodes = pd.read_parquet(_SNAPSHOT / "meta/episodes/chunk-000/file-000.parquet")
    episodes["task"] = episodes["tasks"].apply(
        lambda cell: cell if isinstance(cell, str) else str(cell[0])
    )
    available = episodes[
        (episodes["videos/observation.images.front/file_index"] == 0)
        & (episodes["videos/observation.images.wrist/file_index"] == 0)
    ]

    if episode is not None:
        matched = available[available["episode_index"] == episode]
        if matched.empty:
            raise ValueError(f"エピソード {episode} は file-000 に含まれていません")
        return matched.iloc[0]

    if task is not None:
        matched = available[available["task"].str.startswith(task.lower())]
        if matched.empty:
            raise ValueError(f"該当するタスクがありません: {task}")
        return matched.iloc[0]

    return available.iloc[0]


def _decode_frames(camera: str, start_sec: float, count: int) -> list[np.ndarray]:
    """指定カメラの動画から、開始秒以降の連続フレームを取り出す。"""
    path = str(_SNAPSHOT / f"videos/observation.images.{camera}/chunk-000/file-000.mp4")
    container = av.open(path)
    stream = container.streams.video[0]
    container.seek(int(start_sec / stream.time_base), stream=stream)

    frames = []
    for frame in container.decode(stream):
        if float(frame.pts * stream.time_base) >= start_sec - 1e-6:
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= count:
                break
    container.close()
    return frames


def _load_policy():
    """提出物の MyPolicy をそのまま読み込む。"""
    spec = importlib.util.spec_from_file_location("policy_server", _SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyPolicy()


def _predict_sequence(policy, frames_front, frames_wrist, states, task: str) -> np.ndarray:
    """観測列を順に流し、方策の行動列を得る。

    `select_action` は内部キューを持つため、**素直に 1 ステップずつ呼べば
    本番と同じ chunk の切り替わり方**になる。学習データの観測をそのまま
    入れるので、誤差の蓄積は起きない（teacher forcing）。

    Parameters
    ----------
    policy : MyPolicy
        読み込み済みの方策。
    frames_front, frames_wrist : list[np.ndarray]
        (H, W, 3) uint8 のフレーム列。
    states : np.ndarray
        (T, 8) の状態列。
    task : str
        言語指示。

    Returns
    -------
    actions : np.ndarray
        (T, 7) の行動列。
    """
    torch = policy._torch
    policy.reset(instruction=task)

    def to_tensor(image: np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        return tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0

    predicted = []
    for index in range(len(frames_front)):
        observation = {
            "observation.images.front": to_tensor(frames_front[index]),
            "observation.images.wrist": to_tensor(frames_wrist[index]),
            "observation.state": torch.tensor(states[index]).unsqueeze(0),
            "task": [task],
        }
        batch = policy._preprocessor(observation)
        with torch.no_grad():
            action = policy._policy.select_action(batch)
        action = policy._postprocessor(action)
        predicted.append(
            np.asarray(action[0].detach().cpu().numpy(), dtype=np.float32).reshape(7)
        )
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(frames_front)} フレーム")

    return np.stack(predicted)


def _plot(truth: np.ndarray, predicted: np.ndarray, chunk_size: int,
          title: str, destination: Path) -> None:
    """正解と予測を重ねて描き、chunk 境界を縦線で示す。"""
    figure, axes = plt.subplots(
        len(_DIMENSION_GROUPS) + 1, 1, figsize=(13, 12), sharex=True
    )
    steps = np.arange(len(truth))
    boundaries = [b for b in range(0, len(truth), chunk_size) if b > 0]

    for axis, (label, dims, names) in zip(axes, _DIMENSION_GROUPS):
        for dim, name in zip(dims, names):
            color = f"C{dim}"
            axis.plot(steps, truth[:, dim], color=color, linewidth=1.6,
                      label=f"{name} GT")
            axis.plot(steps, predicted[:, dim], color=color, linewidth=1.2,
                      linestyle="--", alpha=0.85, label=f"{name} policy")
        axis.set_ylabel(label, fontsize=9)
        axis.legend(fontsize=7, ncol=len(dims) * 2, loc="upper right")
        axis.grid(alpha=0.25)
        for boundary in boundaries:
            axis.axvline(boundary, color="black", alpha=0.25, linewidth=0.8)

    # 差分を積み上げて軌道の乖離を見る。1 ステップごとの誤差は小さくても、
    # 積み上がると別の場所へ行く。
    axis = axes[-1]
    for dim, name in zip((0, 1, 2), ("x", "y", "z")):
        axis.plot(steps, np.cumsum(truth[:, dim]), color=f"C{dim}",
                  linewidth=1.6, label=f"{name} GT")
        axis.plot(steps, np.cumsum(predicted[:, dim]), color=f"C{dim}",
                  linewidth=1.2, linestyle="--", alpha=0.85, label=f"{name} policy")
    axis.set_ylabel("Cumulative translation", fontsize=9)
    axis.set_xlabel(f"step  (vertical lines = chunk boundaries, every {chunk_size})")
    axis.legend(fontsize=7, ncol=6, loc="upper left")
    axis.grid(alpha=0.25)

    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110)
    plt.close(figure)


def _report(truth: np.ndarray, predicted: np.ndarray) -> None:
    """次元ごとの誤差と相関を出す。"""
    names = ("dx", "dy", "dz", "drx", "dry", "drz", "grip")
    print()
    print("  次元   平均絶対誤差   GT の標準偏差   誤差/標準偏差   相関")
    for dim, name in enumerate(names):
        error = float(np.abs(predicted[:, dim] - truth[:, dim]).mean())
        deviation = float(truth[:, dim].std())
        ratio = error / deviation if deviation > 1e-8 else float("nan")
        correlation = float(np.corrcoef(truth[:, dim], predicted[:, dim])[0, 1])
        print("  %-5s %11.4f %14.4f %14.2f %8.3f"
              % (name, error, deviation, ratio, correlation))

    print()
    print("  誤差/標準偏差 が 1 を超える次元は、その軸で予測が役に立っていない。")
    print("  相関が 0 付近なら、GT の動きをまったく追えていない。")


def main() -> None:
    args = _parse_args()

    row = _load_episode_row(args.episode, args.task)
    episode_index = int(row["episode_index"])
    task = str(row["task"])
    length = min(int(row["length"]), args.max_frames)
    print(f"エピソード {episode_index} / {task}")
    print(f"  {length} フレームを描画（全 {int(row['length'])}）")

    data = pd.read_parquet(_SNAPSHOT / "data/chunk-000/file-000.parquet")
    episode = data[data["episode_index"] == episode_index].sort_values("frame_index")
    truth = np.stack([
        np.asarray(value, dtype=np.float32).reshape(7)
        for value in episode["action"].to_numpy()[:length]
    ])
    states = np.stack([
        np.asarray(value, dtype=np.float32).reshape(8)
        for value in episode["observation.state"].to_numpy()[:length]
    ])

    frames_front = _decode_frames(
        "front", float(row["videos/observation.images.front/from_timestamp"]), length)
    frames_wrist = _decode_frames(
        "wrist", float(row["videos/observation.images.wrist/from_timestamp"]), length)
    length = min(length, len(frames_front), len(frames_wrist))
    truth, states = truth[:length], states[:length]

    print("方策を読み込み中")
    policy = _load_policy()
    chunk_size = int(policy._n_action_steps)
    print(f"  n_action_steps={chunk_size}, samples={policy._samples}")

    print("推論中")
    predicted = _predict_sequence(
        policy, frames_front[:length], frames_wrist[:length], states, task)

    destination = args.output / f"actions_ep{episode_index}.png"
    _plot(truth, predicted, chunk_size,
          f"episode {episode_index}  |  {task[:70]}", destination)
    _report(truth, predicted)
    print()
    print(f"図: {destination}")


if __name__ == "__main__":
    main()
