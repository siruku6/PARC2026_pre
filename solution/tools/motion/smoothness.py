"""自作デモの指令が、配布データと同じ形をしているかを測る物差し。

**なぜ要るか。** 028 は「上限ちょうどのコマが何 % か」の 4 つで合否を見ていた。
それは**張り付きだけ**の物差しで、**張り付きを潰した先にできた別の歪み**を
見ない。実測（2026-09-11・配布 200 本と 028 版の逆向き 3 本）では、
4 つの条件をすべて満たしたデータに次の差が残っていた:

    停止の塊（|u_trans| < 0.02 が続く区間）  配布 0.8 個/本 ／ 028 版 6.7 個/本
    |u| が 0.2 に達するまで（先頭）          配布 1 コマ    ／ 028 版 8 コマ
    並進 |u| の中央値                        配布 0.63      ／ 028 版 0.47

**停止の塊と立ち上がりの遅さは「勾配が急に切り替わる点」そのものである。**
指令が 0 に落ちてまた立ち上がるたび、方策はその点で別の答えを出す必要がある。
`format_table()` はこの 3 つを含めて、配布との差を 1 つの表にして返す。

**符号反転率は載せるが、単独で読まないこと。**配布データは成分が**厳密に 0**
のコマが多い（並進 15.8%・回転 35.1%）ので、しきい値なしで数えると
「配布は反転しない」と誤って読める。`sign_flip_gated` は両側の大きさが
0.02 を超えるコマだけを数える。
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np


def _episode_list(episodes: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(x, dtype=np.float64) for x in episodes if len(x) >= 3]


def stop_clusters(episodes: list[np.ndarray], threshold: float = 0.02
                  ) -> tuple[float, float]:
    """止まっている区間の数と、止まっているコマの割合を返す。

    Returns
    -------
    clusters_per_episode : float
        1 本あたりの停止の塊の数。**配布データは 0.8 個**（掴む所だけ）。
    stopped_ratio : float
        並進指令のノルムが `threshold` を下回るコマの割合。
    """
    episodes = _episode_list(episodes)
    clusters, stopped, total = 0, 0, 0
    for actions in episodes:
        quiet = np.linalg.norm(actions[:, :3], axis=1) < threshold
        stopped += int(quiet.sum())
        total += len(quiet)
        clusters += int(np.sum(quiet[1:] & ~quiet[:-1]) + (1 if quiet[0] else 0))
    return clusters / max(len(episodes), 1), stopped / max(total, 1)


def onset_steps(episodes: list[np.ndarray], threshold: float = 0.2) -> float:
    """エピソードの頭で、並進指令が `threshold` に達するまでのコマ数の中央値。"""
    episodes = _episode_list(episodes)
    steps = []
    for actions in episodes:
        moving = np.linalg.norm(actions[:, :3], axis=1) > threshold
        steps.append(int(np.argmax(moving)) if moving.any() else len(actions))
    return float(np.median(steps)) if steps else float("nan")


def sign_flip_gated(episodes: list[np.ndarray], columns: range,
                    threshold: float = 0.02) -> float:
    """両側の大きさが `threshold` を超えるコマだけを数えた符号反転率 [%]。"""
    episodes = _episode_list(episodes)
    flips = counted = 0
    for actions in episodes:
        for column in columns:
            series = actions[:, column]
            usable = (np.abs(series[:-1]) > threshold) & (np.abs(series[1:]) > threshold)
            flips += int(np.sum(usable & (np.sign(series[:-1]) * np.sign(series[1:]) < 0)))
            counted += int(usable.sum())
    return flips / max(counted, 1) * 100.0


def saturation(episodes: list[np.ndarray],
               stacked: np.ndarray | None = None) -> dict[str, float]:
    """028 の 6 節の 4 条件のうち、指令に関する 3 つ [%]。

    `stacked` に連結済みの配列を渡せば、そこを作り直さない（配布データの
    117 万行では 65 MB の確保が 1 回浮く）。
    """
    if stacked is None:
        stacked = np.concatenate(_episode_list(episodes))
    translation, rotation = stacked[:, :3], stacked[:, 3:6]
    return {
        "trans_at_one": float((np.abs(translation) >= 0.9999).any(1).mean() * 100),
        "trans_over_envelope": float((np.abs(translation) >= 0.9374).any(1).mean() * 100),
        "rot_at_limit": float((np.abs(rotation) >= 0.3749).any(1).mean() * 100),
    }


def describe(episodes: list[np.ndarray]) -> dict[str, float]:
    """1 組のデータについて、比較に使う数をまとめて出す。"""
    episodes = _episode_list(episodes)
    stacked = np.concatenate(episodes)
    first = np.concatenate([np.diff(x[:, :6], axis=0) for x in episodes])
    clusters, stopped = stop_clusters(episodes)
    lengths = [len(x) for x in episodes]
    return {
        **saturation(episodes, stacked),
        "trans_median": float(np.median(np.abs(stacked[:, :3]).max(1))),
        "rot_median": float(np.median(np.abs(stacked[:, 3:6]).max(1))),
        "delta_p95": float(np.percentile(np.abs(first[:, :3]).max(1), 95)),
        "stop_clusters_per_episode": clusters,
        "stopped_ratio": stopped * 100,
        "onset_steps": onset_steps(episodes),
        "sign_flip_trans": sign_flip_gated(episodes, range(0, 3)),
        "sign_flip_rot": sign_flip_gated(episodes, range(3, 6)),
        "length_median": float(np.median(lengths)),
        "length_max": float(np.max(lengths)),
    }


#: `describe()` の各項目の、表に出す名前と桁数。
ROWS: list[tuple[str, str, int]] = [
    ("trans_at_one", "① 並進 ±1.000 ちょうど [%]", 2),
    ("trans_over_envelope", "② 並進 ±0.9375 以上 [%]", 2),
    ("rot_at_limit", "③ 回転 ±0.375 ちょうど [%]", 2),
    ("length_median", "④ コマ数 中央", 0),
    ("length_max", "④ コマ数 最長", 0),
    ("trans_median", "並進 |u| 中央", 3),
    ("rot_median", "回転 |u| 中央", 3),
    ("delta_p95", "|Δu| 並進 p95", 3),
    ("stop_clusters_per_episode", "停止の塊 [個/本]", 1),
    ("stopped_ratio", "停止コマ [%]", 2),
    ("onset_steps", "立ち上がり [コマ]", 0),
    ("sign_flip_trans", "符号反転 並進 [%]", 2),
    ("sign_flip_rot", "符号反転 回転 [%]", 2),
]


def format_table(named: dict[str, list[np.ndarray]]) -> str:
    """複数の組を 1 つの表にして返す。**列の順は渡した順である。**

    Parameters
    ----------
    named : dict
        「名前 → エピソードごとの `action` (T, >=6) の列」。先頭に配布データを
        置くと、以降の列がそこからどれだけ外れているかを読みやすい。

    Returns
    -------
    str
        Markdown の表 1 つ（改行込み）。
    """
    described = {name: describe(episodes) for name, episodes in named.items()}
    names = list(described)
    width = max(11, max(len(name) for name in names))
    lines = ["| 見るもの                     | "
             + " | ".join(f"{name:>{width}}" for name in names) + " |",
             "|" + "-" * 30 + "|" + "|".join(["-" * (width + 2)] * len(names)) + "|"]
    for key, label, digits in ROWS:
        cells = " | ".join(f"{described[name][key]:>{width}.{digits}f}" for name in names)
        lines.append(f"| {label:<28} | {cells} |")
    return "\n".join(lines)


def load_dataset_episodes(dataset: Path | str, count: int,
                          seed: int = 0) -> list[np.ndarray]:
    """配布データ（LeRobot 形式）から `count` 本ぶんの `action` 列を無作為に読む。

    **比べる基準はいつも同じ引き方で取る。**`compare_waypoint_profiles.py` と
    `report_smoothness.py` が別々に引いていると、片方の引き方を直したときに
    **両方とも「配布」と名乗ったまま違う母集団を指す。**

    **`action` 列は選んだ本のぶんしか読まない。**先に `episode_index` だけを
    読んで本を選び、当たった行だけを取り出す（全 117 万行を float64 に広げると
    65 MB になるが、200 本なら 1.8 MB で済む）。

    Parameters
    ----------
    dataset : path-like
        LeRobot のデータセットディレクトリ（`data/chunk-*/file-*.parquet` を持つ）。
    count : int
        取り出す本数。データセットの本数より多ければ全部返す。
    seed : int
        抽選の種。既定 0。

    Returns
    -------
    list of np.ndarray
        エピソードごとの `action` (T, 7)。

    Reads:
      - <dataset>/data/chunk-*/file-*.parquet
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(Path(dataset) / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"parquet が無い: {dataset}")

    # **全ファイルを見る。**先頭 1 つだけだと、後ろのファイルに居る本が
    # 一度も当たらない（配布データは 2 ファイルに分かれている）。
    indices = [pq.read_table(path, columns=["episode_index"])["episode_index"].to_numpy()
               for path in files]
    available = np.unique(np.concatenate(indices))
    picked = np.random.default_rng(seed).choice(
        available, min(count, len(available)), replace=False)
    wanted = set(picked.tolist())

    episodes: dict[int, list[np.ndarray]] = {episode: [] for episode in wanted}
    for path, index in zip(files, indices):
        keep = np.isin(index, picked)
        if not keep.any():
            continue
        table = pq.read_table(path, columns=["action"])
        actions = np.stack(table["action"].to_numpy(zero_copy_only=False))[keep]
        for episode in np.unique(index[keep]):
            episodes[int(episode)].append(actions[index[keep] == episode])
    return [np.concatenate(parts).astype(np.float64)
            for parts in episodes.values() if parts]
