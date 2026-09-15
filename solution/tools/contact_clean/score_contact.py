#!/usr/bin/env python3
"""記録された MuJoCo の状態から、デモが「対象外の物体」に触れたかを数える。

**再生はしない。**LIBERO 本家の hdf5 に入っている `states`（`sim.get_state()` を
平坦にしたもの）を 1 コマずつ env に流し込み、`forward` だけ掛けて物体の位置と
接触対を読む。**物理を 1 歩も進めないので、再生の誤りが入らない。**

**採点側と同じ数え方に揃える**（`pipeline/rollout.py`）。

  - 対象外の物体 ＝ 観測の `<名前>_pos` のうち、`robot0` で始まらず、
    `_to_robot0_eef_pos` で終わらず、bddl の `:obj_of_interest` に無いもの
  - ずれの大きさ ＝ **3 軸の絶対値の和**（ユークリッド距離ではない）
  - 失格の境目 ＝ **0.001 m**

**基準のコマをどこに置くかは、この道具では決めない。**物体は少し浮いた位置に
置かれていて最初の数コマで落ちるので（`002` の実測で、摂動なしでも 111 mm 動いたと
出た）、基準を 0 コマ目に置くと全部が接触扱いになる。**0・5・10・20 コマ目の
4 通りで測って全部書き出す。**どれを採るかは分布を見てから決める。

用法（`parc2026_verify` の中で打つ。描画しないので EGL は要らない）:
    python score_contact.py --states /tmp/contact_clean/libero_states \\
        --map /tmp/contact_clean/demo_map.json \\
        --bddl-root /workspace/LIBERO-plus/libero/libero/bddl_files \\
        --out /tmp/contact_clean/contact_scores.json

Reads:
  - <states>/index.json, <states>/<suite>/<課題>_demo.npz
  - <map> … map_demos.py が書いた対応表
  - <bddl-root>/<suite>/<課題>.bddl
Generates:
  - <out> … 1 デモ 1 行の判定表
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASELINE_FRAMES = (0, 5, 10, 20)
COLLISION_THRESHOLD_M = 0.001


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--map", type=Path, required=True, metavar="JSON")
    parser.add_argument("--bddl-root", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="JSON")
    parser.add_argument("--limit-tasks", type=int, default=0, metavar="N",
                        help="先頭 N 課題だけ見る（試し打ち用。0 なら全部）")
    return parser.parse_args()


def _parse_obj_of_interest(bddl_file: Path) -> set[str]:
    """bddl の `:obj_of_interest` を読む。**`pipeline/environment.py` と同じ実装。**"""
    text = bddl_file.read_text()
    found = re.search(r"\(:obj_of_interest(.*?)\)\s*\(:init", text, re.S)
    return set(found.group(1).split()) if found else set()


def _make_env(bddl_file: Path):
    """採点と同じ物理の env を、描画なしで作る。**`reversal_probe.make_env` と同じ。**"""
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(bddl_file_name=str(bddl_file), use_camera_obs=False,
                      has_offscreen_renderer=False, has_renderer=False,
                      control_freq=20, horizon=2000, ignore_done=True)


def _nontarget_names(observation: dict, obj_of_interest: set[str]) -> list[str]:
    """採点側と同じ絞り方で「対象外の物体」の名前を出す。"""
    return sorted(key[:-4] for key in observation
                  if key.endswith("_pos")
                  and not key.startswith("robot0")
                  and not key.endswith("_to_robot0_eef_pos")
                  and key[:-4] not in obj_of_interest)


def _body_geom_groups(sim, names: list[str], obj_of_interest: set[str]
                      ) -> tuple[dict[str, int], dict[int, str], set[int]]:
    """物体の名前から body 番号を引き、geom 番号から所属を引けるようにする。

    Returns
    -------
    body_of : dict[str, int]
        対象外の物体の名前 → body 番号。
    owner_of_geom : dict[int, str]
        geom 番号 → 対象外の物体の名前。接触対の判定に使う。
    mover_geoms : set[int]
        腕・指と、`:obj_of_interest` の物体の geom 番号。**触れる側**である。
    """
    body_of: dict[str, int] = {}
    owner_of_geom: dict[int, str] = {}
    mover_geoms: set[int] = set()

    interest_bodies = {f"{name}_main" for name in obj_of_interest}
    target_bodies = {f"{name}_main": name for name in names}

    for geom_id in range(sim.model.ngeom):
        body_id = int(sim.model.geom_bodyid[geom_id])
        body_name = sim.model.body_id2name(body_id) or ""
        if body_name in target_bodies:
            owner_of_geom[geom_id] = target_bodies[body_name]
        elif body_name.startswith(("robot0", "gripper0")) or body_name in interest_bodies:
            mover_geoms.add(geom_id)

    for body_name, name in target_bodies.items():
        body_of[name] = sim.model.body_name2id(body_name)
    return body_of, owner_of_geom, mover_geoms


def _score_one_demo(sim, states: np.ndarray, body_of: dict[str, int],
                    owner_of_geom: dict[int, str], mover_geoms: set[int]) -> dict:
    """1 デモぶんの状態列を流し込み、ずれの最大と、触れた相手を集める。

    Parameters
    ----------
    states : np.ndarray
        `(コマ数, 状態の次元)`。本家 hdf5 の `states` そのもの。

    Returns
    -------
    score : dict
        基準コマごとの「いちばん動いた物体とその量」、触れた物体の名前、
        触れ始めたコマ。
    """
    order = sorted(body_of)
    body_ids = [body_of[name] for name in order]
    positions = np.empty((len(states), len(order), 3), dtype=np.float64)
    touched: dict[str, int] = {}

    for frame, state in enumerate(states):
        sim.set_state_from_flattened(state)
        sim.forward()
        positions[frame] = sim.data.body_xpos[body_ids]
        for contact_index in range(sim.data.ncon):
            contact = sim.data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            name = owner_of_geom.get(first) or owner_of_geom.get(second)
            other = second if owner_of_geom.get(first) else first
            if name is not None and other in mover_geoms and name not in touched:
                touched[name] = frame

    displacement: dict[str, dict[str, float]] = {}
    for baseline in BASELINE_FRAMES:
        if baseline >= len(states):
            continue
        l1 = np.abs(positions[baseline:] - positions[baseline]).sum(axis=2).max(axis=0)
        displacement[str(baseline)] = {name: float(value) for name, value in zip(order, l1)}

    return {"objects": order, "displacement_l1_m": displacement,
            "touched": touched, "frames": int(len(states))}


def _score_one_task(source_file: str, demo_names: list[str], states_dir: Path,
                    bddl_root: Path, bddl_name: str) -> dict[str, dict]:
    """1 課題ぶんの env を 1 度だけ作り、その課題のデモを順に見る。"""
    suite = source_file.split("/")[0]
    bddl_file = bddl_root / suite / Path(bddl_name).name
    obj_of_interest = _parse_obj_of_interest(bddl_file)

    env = _make_env(bddl_file)
    try:
        observation = env.reset()
        names = _nontarget_names(observation, obj_of_interest)
        sim = env.env.sim
        body_of, owner_of_geom, mover_geoms = _body_geom_groups(sim, names, obj_of_interest)

        scores: dict[str, dict] = {}
        with np.load(states_dir / source_file.replace(".hdf5", ".npz")) as bundle:
            for demo in demo_names:
                states = bundle[f"{demo}/states"]
                if states.shape[1] != len(sim.get_state().flatten()):
                    raise SystemExit(
                        f"状態の次元が合わない: {source_file} {demo} "
                        f"{states.shape[1]} 対 {len(sim.get_state().flatten())}")
                scores[demo] = _score_one_demo(sim, states, body_of,
                                               owner_of_geom, mover_geoms)
        for score in scores.values():
            score["obj_of_interest"] = sorted(obj_of_interest)
    finally:
        env.close()
    return scores


def _summarize(rows: list[dict]) -> None:
    """外れる組の数を、基準コマと物差しごとに印字する。"""
    total_clusters = len(rows)
    total_episodes = sum(len(row["episodes"]) for row in rows)
    print(f"\n=== 判定 ({total_clusters} 組 / {total_episodes} 本) ===")
    print(f"{'基準コマ':>8} {'1mm 超の組':>12} {'消える本':>10} {'割合':>8}")
    for baseline in BASELINE_FRAMES:
        hit = [row for row in rows
               if max(row["score"]["displacement_l1_m"].get(str(baseline), {0: 0.0}).values(),
                      default=0.0) > COLLISION_THRESHOLD_M]
        episodes = sum(len(row["episodes"]) for row in hit)
        print(f"{baseline:>8} {len(hit):>12} {episodes:>10} "
              f"{episodes / total_episodes * 100:>7.1f}%")
    touched = [row for row in rows if row["score"]["touched"]]
    episodes = sum(len(row["episodes"]) for row in touched)
    print(f"{'触れた':>8} {len(touched):>12} {episodes:>10} "
          f"{episodes / total_episodes * 100:>7.1f}%")


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")

    demo_map = json.loads(args.map.read_text())
    index = json.loads((args.states / "index.json").read_text())

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in demo_map:
        if "source_file" in row:
            by_task[row["source_file"]].append(row)

    task_files = sorted(by_task)
    if args.limit_tasks:
        task_files = task_files[:args.limit_tasks]
    print(f"=== {len(task_files)} 課題 / "
          f"{sum(len(by_task[name]) for name in task_files)} 組 ===")

    rows: list[dict] = []
    for number, source_file in enumerate(task_files, start=1):
        started = time.time()
        entries = by_task[source_file]
        scores = _score_one_task(source_file, [row["source_demo"] for row in entries],
                                 args.states, args.bddl_root,
                                 index[source_file]["bddl_file_name"])
        for entry in entries:
            rows.append(dict(entry, score=scores[entry["source_demo"]]))
        print(f"[{number}/{len(task_files)}] {source_file}  "
              f"{len(entries)} 組  ({time.time() - started:.1f}s)", flush=True)

    args.out.write_text(json.dumps(rows, ensure_ascii=False))
    _summarize(rows)
    print(f"\n=== 書いた: {args.out} ===")


if __name__ == "__main__":
    main()
