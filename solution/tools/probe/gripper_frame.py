#!/usr/bin/env python3
"""指の向きと当たり判定の寸法を測る。**掴む向きを選ぶ前に必要な数字。**

**なぜ要るか。**「掴む対象は Z 軸まわりに回転対称だから、空いている方位から
指を入れれば当てずに掴める」という筋を実装するには、**指がどの軸に並んでいて、
どれだけ張り出しているか**を知らないと向きを計算できない。
robosuite の `robot0_eef_quat` から手首の回転は読めるが、**指がその回転の
どの軸に並ぶかは実装依存**なので、パッドの実座標から測る。

出すもの:
  - 指のパッド 2 つの世界座標と、その並ぶ向き（方位角）
  - 手先の回転行列の各軸が世界のどちらを向いているか
  - 掴む対象・邪魔になる物体・置き先の当たり判定の寸法
  - **`action[5]` を 1 歩入れたときに方位角が何度回るか**（制御の利き）

用法（コンテナの中で打つ）:
    bash solution/tools/regen/run_in_container.sh \\
        solution/tools/probe/gripper_frame.py \\
        --task pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate \\
        --suite libero_spatial
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, metavar="NAME")
    parser.add_argument("--suite", default="libero_spatial", metavar="NAME")
    parser.add_argument("--layout", type=int, default=0, metavar="N")
    return parser.parse_args()


def _azimuth(vector: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def main() -> None:
    args = _parse_args()
    import scripted_demo as sd
    from libero.libero import get_libero_path
    from robosuite.utils.transform_utils import quat2mat

    libero_root = Path(get_libero_path("bddl_files")).parent
    env = sd._make_env(sd._bddl_path(libero_root, args.suite, args.task), False)
    try:
        env.seed(0)
        init_states = sd._init_states(libero_root, args.suite, args.task)
        obs = sd.reset_to(env, init_states[args.layout])
        sim = env.env.sim

        pads = [i for i in range(sim.model.ngeom)
                if (sim.model.geom_id2name(i) or "").endswith("_pad_collision")]
        print("--- 指のパッド ---")
        for i in pads:
            print(f"  {sim.model.geom_id2name(i):<34} "
                  f"{np.round(sim.data.geom_xpos[i], 4)}  "
                  f"size {np.round(sim.model.geom_size[i], 4)}")
        if len(pads) == 2:
            axis = sim.data.geom_xpos[pads[1]] - sim.data.geom_xpos[pads[0]]
            print(f"  指が並ぶ向き 方位 {_azimuth(axis):.1f}度  "
                  f"間隔 {100 * np.linalg.norm(axis):.2f} cm")

        matrix = quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
        print("\n--- 手先の回転（列が局所軸の世界向き）---")
        for label, column in zip("xyz", matrix.T):
            print(f"  局所 {label} 軸 → {np.round(column, 3)}  方位 {_azimuth(column):>7.1f}度")
        print(f"  手先 {np.round(obs['robot0_eef_pos'], 4)}")

        print("\n--- 当たり判定の寸法（掴む高さの帯にある物）---")
        for name, body_id in env.env.obj_body_id.items():
            geoms = [i for i in range(sim.model.ngeom)
                     if sim.model.geom_contype[i]
                     and (sim.model.geom_id2name(i) or "").startswith(name)]
            if not geoms:
                continue
            low = min(float(sim.data.geom_xpos[i][2] - sim.model.geom_size[i][2])
                      for i in geoms)
            high = max(float(sim.data.geom_xpos[i][2] + sim.model.geom_size[i][2])
                       for i in geoms)
            widest = max(float(2 * max(sim.model.geom_size[i][0],
                                       sim.model.geom_size[i][1]))
                         for i in geoms)
            origin = np.asarray(sim.data.body_xpos[body_id])
            print(f"  {name:<34} 原点 {np.round(origin, 4)}  "
                  f"当たる高さ {low:.4f}〜{high:.4f}  最大幅 {100 * widest:.1f} cm")

        # **物体どうしの隙間。**「最大幅」は geom 1 つの幅なので、
        # **物体全体の張り出しは原点から測り直す**（本体が小さな箱の集まりで
        # できている物では、geom 1 つの幅は全体の幅にならない）。
        print("\n--- 物体の張り出しと、隙間 ---")
        radius: dict[str, float] = {}
        centre: dict[str, np.ndarray] = {}
        for name, body_id in env.env.obj_body_id.items():
            geoms = [i for i in range(sim.model.ngeom)
                     if sim.model.geom_contype[i]
                     and (sim.model.geom_id2name(i) or "").startswith(name)]
            if not geoms:
                continue
            origin = np.asarray(sim.data.body_xpos[body_id])
            radius[name] = max(
                float(np.linalg.norm(sim.data.geom_xpos[i][:2] - origin[:2])
                      + max(sim.model.geom_size[i][0], sim.model.geom_size[i][1]))
                for i in geoms)
            centre[name] = origin
            print(f"  {name:<34} 水平の張り出し {100 * radius[name]:>5.2f} cm")
        target = "akita_black_bowl_1" if "akita_black_bowl_1" in radius else \
            next(iter(radius))
        print(f"\n  {target} から見た隙間（面と面の間）:")
        for name in radius:
            if name == target:
                continue
            gap = float(np.linalg.norm(centre[name][:2] - centre[target][:2])) \
                - radius[name] - radius[target]
            azimuth = _azimuth(centre[name][:2] - centre[target][:2])
            print(f"    {name:<34} 隙間 {100 * gap:>6.2f} cm  方位 {azimuth:>7.1f}度")

        # **制御の利き。**`action[5]` を 1 歩入れて、指の方位が何度回るか測る。
        before = sim.data.geom_xpos[pads[1]] - sim.data.geom_xpos[pads[0]] \
            if len(pads) == 2 else np.zeros(3)
        for command in (0.25, 0.5, 1.0):
            env2 = sd._make_env(sd._bddl_path(libero_root, args.suite, args.task), False)
            env2.seed(0)
            sd.reset_to(env2, init_states[args.layout])
            sim2 = env2.env.sim
            pads2 = [i for i in range(sim2.model.ngeom)
                     if (sim2.model.geom_id2name(i) or "").endswith("_pad_collision")]
            action = np.zeros(7)
            action[5] = command
            for _ in range(5):
                env2.env.step(action)
            after = sim2.data.geom_xpos[pads2[1]] - sim2.data.geom_xpos[pads2[0]]
            turned = _azimuth(after) - _azimuth(before)
            turned = (turned + 180) % 360 - 180
            print(f"\n  action[5]={command:>4} を 5 歩 → 指の方位が {turned:+.1f}度 回った"
                  f"（1 歩あたり {turned / 5:+.1f}度）")
            env2.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()
