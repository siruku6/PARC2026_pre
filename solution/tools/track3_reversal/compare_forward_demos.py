#!/usr/bin/env python3
"""同じ指示文の順方向デモが**互いにどれだけ違うか**を数える。

**「順方向デモを増やす」に意味があるかは、デモ同士が違っているかで決まる。**
同じ動きが 44 本あるだけなら、増やしても掴む姿勢の網羅は広がらない
（[reference/30 の 6 節](../../../docs/reference/30_reverse_grasp_pose.md)）。

シムを回さずに `observation.state`（`[eef_pos(3), 軸角(3), 指(2)]`）だけで測る。
出すのは 4 つ。

| 値 | 何が分かるか |
|---|---|
| コマ数の散らばり | 動きの長さがどれだけ違うか |
| **腕の初期姿勢の散らばり** | ロボット初期姿勢の摂動が入っているか |
| **掴んだ瞬間の手の位置の散らばり** | 物の置き場所がどれだけ振れているか |
| **掴んだ瞬間の手の向きの散らばり** | **掴む姿勢の網羅**。赤マグで効いたのはここ（[reference/30 の 6 節]） |
| **経路同士の距離** | 動きそのものがどれだけ違うか（長さを揃えて突き合わせる） |

    python3 compare_forward_demos.py work/probe_020/all_mug.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reversal_probe import decode_gripper                        # noqa: E402


def clamp_frame(grip, states):
    """順方向で**実際に物を挟み終えた**コマ（指の開き幅から決める）。"""
    w = states[:, 6] - states[:, 7]
    lo = int(np.argmax(grip > 0))
    hi = len(grip) - 1 - int(np.argmax(grip[::-1] > 0))
    hold = float(np.median(w[lo:hi + 1]))
    return lo + int(np.argmax(w[lo:hi + 1] <= hold + 0.003))


def spread(v):
    """位置の散らばり[mm]。**軸ごとの幅と、重心からの距離の中央値**で出す。"""
    v = np.asarray(v)
    return (np.round((v.max(0) - v.min(0)) * 1000, 1),
            round(float(np.median(np.linalg.norm(v - v.mean(0), axis=1))) * 1000, 1))


def rot_spread(aas):
    """向きの散らばり[deg]。**全ペアの相対回転角**で出す。

    軸角ベクトルの成分ごとの差は相対回転ではないので使えない。
    回転行列の合成から角度を出す（`R_a @ R_b^T` の軸角の大きさ）。
    """
    def mat(aa):
        aa = np.asarray(aa, dtype=np.float64)
        th = float(np.linalg.norm(aa))
        if th < 1e-9:
            return np.eye(3)
        k = aa / th
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * (kx @ kx)

    r = [mat(a) for a in aas]
    out = []
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            c = (np.trace(r[i] @ r[j].T) - 1.0) / 2.0
            out.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.asarray(out)


def resample(path, n=60):
    """経路を弧長で n 点に取り直す。**長さの違いを除いて形だけ比べる**ため。"""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if d[-1] <= 0:
        return np.repeat(path[:1], n, axis=0)
    return np.stack([np.interp(np.linspace(0, d[-1], n), d, path[:, k])
                     for k in range(path.shape[1])], axis=1)


def main() -> None:
    for arg in sys.argv[1:]:
        z = np.load(arg, allow_pickle=True)
        eps = sorted(int(k.split("_")[1]) for k in z if k.startswith("actions_"))
        lens, starts, grasps, gaas, paths = [], [], [], [], []
        for ep in eps:
            st = z[f"states_{ep}"].astype(np.float64)
            a = decode_gripper(z[f"actions_{ep}"].astype(np.float32), st)
            lens.append(len(st))
            starts.append(st[0, :3])
            j = clamp_frame(a[:, 6], st)
            grasps.append(st[j, :3])
            gaas.append(st[j, 3:6])
            paths.append(resample(st[:, :3]))
        lens = np.asarray(lens)
        p = np.stack(paths)
        # 経路同士の距離: 弧長で揃えた点列の平均距離（全ペア）
        n = len(p)
        dist = [float(np.linalg.norm(p[i] - p[j], axis=1).mean())
                for i in range(n) for j in range(i + 1, n)]
        dist = np.asarray(dist) * 1000
        arc = np.asarray([np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
                          for q in p]) * 1000
        print(f"\n=== {Path(arg).name}  デモ {n} 本 ===")
        print(f"  コマ数        : 最小 {lens.min()} / 中央 {int(np.median(lens))} "
              f"/ 最大 {lens.max()}")
        w, m = spread(starts)
        print(f"  腕の初期姿勢  : 軸ごとの幅 {w} mm / 重心からの距離 中央 {m} mm")
        w, m = spread(grasps)
        print(f"  掴んだ瞬間の手: 軸ごとの幅 {w} mm / 重心からの距離 中央 {m} mm")
        rs = rot_spread(gaas)
        print(f"  掴んだ瞬間の向き: 全ペアの相対回転角 最小 {rs.min():.1f} / "
              f"中央 {np.median(rs):.1f} / 最大 {rs.max():.1f} °")
        print(f"  経路の全長    : 中央 {np.median(arc):.0f} mm")
        print(f"  経路同士の距離: 最小 {dist.min():.1f} / 中央 "
              f"{np.median(dist):.1f} / 最大 {dist.max():.1f} mm  "
              f"（全長の {np.median(dist) / np.median(arc) * 100:.1f}%）")


if __name__ == "__main__":
    main()
