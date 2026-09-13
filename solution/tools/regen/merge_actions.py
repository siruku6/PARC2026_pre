#!/usr/bin/env python3
"""人の手本の指令を、上限を超えない範囲で隣どうし 1 歩にまとめて、手本を短くする。

**なぜ要るか。**`put both moka pots on the stove` の人の手本は 146 本すべてが 341 歩以上で、
採点の打ち切り 300 歩に 1 本も入らない。**止まっているのではなく、上限の 4 割程度で
ゆっくり動いている**（1 コマの指令の大きさ中央 0.37・上限 0.9375 に届くコマ 0%）ので、
止まった所を削っても縮まない。

**やり方。**`action` の前 6 つは「その 1 歩で手先をどれだけ動かすか」の指令である。
**隣の 2 コマの指令を足した 1 歩は、おおよそ 2 コマぶんを 1 歩で動く。**
並進 0.9375・回転 0.375 を超えない所まで足していき、超える手前で 1 歩として区切る。

    元の手本   x 指令  0.30  0.32  0.35  0.33  0.31
    まとめる          └──── 0.62 ────┘ └──── 0.68 ────┘ …

**指の開閉の直後 `dwell` コマはまとめない。**指が閉じきるのに時間が要り、
ここを縮めると掴み損ねる。

**これは近似である。**腕は指令に 3〜4 コマ遅れて付いてくるので、大きな指令を
1 歩で出しても同じ経路を通るとは限らない。**だから必ずシムで撃ち直して、
ゴールと接触の門を通った本だけを使う**（試作では 50 本中 24 本が通った）。
"""
from __future__ import annotations

import numpy as np

LIMIT_POS = 0.9375
"""並進の上限（配布データの端・`motion/envelope.py` の `limit_pos`）。"""

LIMIT_ROT = 0.375
"""回転の上限（配布データの端）。

**`motion/envelope.py` の `limit_rot`（0.30）ではない。**あちらは自分で書く台本を
配布データの内側に収めるための、意図して狭い値である。ここで見たいのは
**配布データそのものの端**——人の手本をまとめて、配布データの外へ出さない、という判定である。
"""


def merge_adjacent(actions: np.ndarray, scale: float = 1.0, dwell: int = 8) -> np.ndarray:
    """隣どうしの指令を、上限の `scale` 倍を超えない所まで 1 歩にまとめる。

    Parameters
    ----------
    actions : np.ndarray
        (N, 7)。前 3 つが並進、次の 3 つが回転、最後が指。
    scale : float
        上限の何倍までまとめるか。**1.0 で上限いっぱい**（試作では 0.8 だと 300 歩に入らなかった）。
    dwell : int
        指の開閉の直後、まとめずに残すコマ数。

    Returns
    -------
    np.ndarray
        (M, 7)。M ≤ N。指の値はまとめた区間の最後のコマのものを使う。
    """
    actions = np.asarray(actions, dtype=np.float64)
    # **指の開閉の直後 `dwell` コマは、まとめずにそのまま残す区間である。**
    # 先に印を付けてしまえば、まとめる側は「保留中の 1 コマ」だけを持てばよい
    # （数え上げながらまとめると、区間の出入りと保留の状態が絡まる）。
    frozen = np.zeros(len(actions), dtype=bool)
    for start in np.flatnonzero(np.diff(actions[:, 6])) + 1:
        frozen[start:start + dwell] = True
    merged: list[np.ndarray] = []
    pending: np.ndarray | None = None
    for row, hold in zip(actions, frozen):
        if hold:
            if pending is not None:
                merged.append(pending)
                pending = None
            merged.append(row.copy())
            continue
        if pending is None:
            pending = row.copy()
            continue
        trans, rot = pending[:3] + row[:3], pending[3:6] + row[3:6]
        if (np.abs(trans).max() > LIMIT_POS * scale
                or np.abs(rot).max() > LIMIT_ROT * scale):
            merged.append(pending)
            pending = row.copy()
        else:
            pending[:3], pending[3:6], pending[6] = trans, rot, row[6]
    if pending is not None:
        merged.append(pending)
    return np.asarray(merged, dtype=np.float32)
