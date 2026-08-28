#!/usr/bin/env python3
"""評価用 CSV の各行が、方策を動かす前の時点で成り立つかを確かめる。

**姿勢によっては初手から物に触れていたり、腕が机に埋まっていたりする。**
そういう行を混ぜたまま評価を回すと、崩れ方の表に「そもそも解けない場面」が
混ざる。方策を呼ぶ前に、採点ハーネスと同じ手順（reset →
`set_state_from_flattened` → 空撃ち 10 ステップ）をここで踏んでおく。

同時に、**その姿勢が手先を何 cm 動かすかも測る。** 同じ CSV の中に
`category` が `Base` の行（摂動なし）が入っている前提で、行の並びを
「Base 行 → その元タスクの摂動行」というまとまりとして読み、
まとまりごとの Base 行との手先の距離を出す。

判定は 3 つで、どれかに当たった行は「通らない」として印を付ける:

  - 環境が作れない、または空撃ちの途中で例外が出る
  - 空撃ちのあと手先の位置に NaN が出る
  - **空撃ちを終えたあと、さらに 10 ステップ何もしない間に物体が動く**
    採点側は空撃ち 10 ステップのあとの位置を衝突判定の基準にする
    （`pipeline/rollout.py`）。そこからさらに何もせずに物体が 1 mm 動くなら、
    腕がその物に当たったまま止まっている。**方策が何をしても衝突扱いになる行**である。

**空撃ちの前後で比べてはいけない。** 物体は少し浮いた位置に置かれるので、
空撃ちの 10 ステップの間に落ちて 10 cm 動く。これは摂動と関係がない。

**描画は要らない。** `_make_env(with_images=False)` なので `MUJOCO_GL` は不問。

使い方（コンテナの中で回す）:

    docker cp compe/val/TRACK2_ROBOT_LADDER60.csv parc2026_verify:/workspace/
    docker cp solution/tools/probe/check_init_pose_validity.py parc2026_verify:/workspace/
    docker exec parc2026_verify /workspace/venv/bin/python \\
        /workspace/check_init_pose_validity.py \\
        --tasks-csv /workspace/TRACK2_ROBOT_LADDER60.csv \\
        --libero-root /workspace/LIBERO-plus/libero/libero \\
        --output /workspace/init_pose_validity.json

Reads:
  - --tasks-csv に渡した CSV
  - <libero-root>/bddl_files/<suite>/<元タスク>.bddl
  - <libero-root>/init_files/<suite>/<元タスク>.pruned_init
Generates:
  - --output に渡した JSON（1 行 1 件・手先のずれと判定）
  - 標準出力の要約
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_VARIANT_RE = re.compile(r"_view_\d+_\d+_\d+_\d+_\d+_initstate_\d+$")
# 採点ハーネスと同じ本数（pipeline/environment.py が init_states[0:n] を使う）。
_DEFAULT_EPISODES = 3
# 物体が「動いた」とみなす幅。採点側の閾値（pipeline/total_score.py）に合わせる。
_MOVE_THRESHOLD_M = 0.001


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks-csv", type=Path, required=True)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=_DEFAULT_EPISODES)
    return p.parse_args()


def _base_task(task_id: str) -> str:
    """変種名から元タスク名を取る。`.bddl` は変種名より前の部分にしか無い。"""
    return _VARIANT_RE.sub("", task_id)


def _object_positions(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """観測から物体の位置だけを抜く。採点側の衝突判定と同じ選び方。"""
    return {
        k[:-4]: np.asarray(observation[k], dtype=float).copy()
        for k in observation
        if k.endswith("_pos")
        and not k.startswith("robot0")
        and not k.endswith("_to_robot0_eef_pos")
    }


def _robot_object_contacts(sim: Any) -> list[str]:
    """腕・指と、ロボット以外の物体との接触を名前で返す。

    物体が動いていなくても、**当たったまま釣り合っている**姿勢はここに出る。
    机（`table`）との接触は台の上に物が載っているだけなので数えない。
    """
    names = []
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        g1 = sim.model.geom_id2name(contact.geom1) or ""
        g2 = sim.model.geom_id2name(contact.geom2) or ""
        is_robot = [g.startswith(("robot0", "gripper0")) for g in (g1, g2)]
        if sum(is_robot) != 1:
            continue
        other = g2 if is_robot[0] else g1
        if other.startswith(("table", "floor", "wall")):
            continue
        names.append(other)
    return sorted(set(names))


def _measure_row(row: dict, libero_root: Path, n_episodes: int,
                 helpers: dict) -> dict:
    """1 行について、空撃ち後の手先の位置と物体の動きをエピソードごとに測る。"""
    task_id = row["task_id"]
    base_task = _base_task(task_id)
    bddl = helpers["_bddl_path"](libero_root, row["suite"], task_id)
    init_states = helpers["_init_states"](libero_root, row["suite"], base_task)

    env = helpers["_make_env"](bddl, False)
    try:
        episodes = []
        for ep in range(min(n_episodes, len(init_states))):
            env.reset()
            sim = env.env.sim
            sim.set_state_from_flattened(init_states[ep])
            sim.forward()
            observation: dict[str, Any] = {}
            for _ in range(10):
                observation, _, _, _ = env.env.step(np.zeros(7))
            # 採点側が衝突判定の基準に取るのと同じ時点。
            before = _object_positions(observation)
            contacts = _robot_object_contacts(env.env.sim)
            for _ in range(10):
                observation, _, _, _ = env.env.step(np.zeros(7))
            after = _object_positions(observation)
            moved = {
                name: float(np.sum(np.abs(after[name] - before[name])))
                for name in after
                if name in before
            }
            eef = np.asarray(observation["robot0_eef_pos"], dtype=float)
            episodes.append({
                "episode": ep,
                "eef": eef.tolist(),
                "eef_nan": bool(np.any(~np.isfinite(eef))),
                "max_object_move_m": max(moved.values()) if moved else 0.0,
                "robot_object_contacts": contacts,
                "moved_objects": sorted(
                    n for n, d in moved.items() if d > _MOVE_THRESHOLD_M
                ),
            })
        return {"episodes": episodes, "error": None}
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    sys.path.insert(0, str(Path("/workspace/solution/tools/scripted_demo")))
    import scripted_demo  # noqa: E402

    helpers = {
        "_bddl_path": scripted_demo._bddl_path,
        "_init_states": scripted_demo._init_states,
        "_make_env": scripted_demo._make_env,
    }

    with open(args.tasks_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    results: list[dict] = []
    base_eef: dict[tuple[str, int], np.ndarray] = {}
    started = time.time()

    for i, row in enumerate(rows, start=1):
        record = {
            "task_id": row["task_id"],
            "base_task": _base_task(row["task_id"]),
            "suite": row["suite"],
            "category": row["category"],
            "difficulty_level": row["difficulty_level"],
        }
        try:
            record.update(_measure_row(row, args.libero_root, args.episodes, helpers))
        except Exception as exc:  # noqa: BLE001
            record.update({"episodes": [], "error": f"{type(exc).__name__}: {exc}"})

        if row["category"] == "Base":
            for ep in record["episodes"]:
                base_eef[(record["base_task"], ep["episode"])] = np.asarray(ep["eef"])

        for ep in record["episodes"]:
            key = (record["base_task"], ep["episode"])
            ref = base_eef.get(key)
            ep["eef_shift_cm"] = (
                None if ref is None
                else float(np.linalg.norm(np.asarray(ep["eef"]) - ref) * 100)
            )

        shifts = [e["eef_shift_cm"] for e in record["episodes"]
                  if e.get("eef_shift_cm") is not None]
        record["eef_shift_cm"] = statistics.median(shifts) if shifts else None
        record["max_object_move_m"] = max(
            (e["max_object_move_m"] for e in record["episodes"]), default=0.0)
        record["robot_object_contacts"] = sorted(
            {c for e in record["episodes"] for c in e["robot_object_contacts"]})
        record["ok"] = bool(
            record["error"] is None
            and record["episodes"]
            and not any(e["eef_nan"] for e in record["episodes"])
            and record["max_object_move_m"] <= _MOVE_THRESHOLD_M
        )
        results.append(record)

        shift = record["eef_shift_cm"]
        print(f"[{i}/{len(rows)}] {'OK ' if record['ok'] else 'NG '}"
              f"{record['difficulty_level'] or 'Base':>4s} "
              f"ずれ {'--' if shift is None else f'{shift:5.2f} cm'} "
              f"物体 {record['max_object_move_m']*1000:5.2f} mm "
              f"接触 {len(record['robot_object_contacts'])}  "
              f"{record['task_id'][:64]}"
              + (f"  {record['error']}" if record["error"] else ""), flush=True)

    args.output.write_text(json.dumps(results, indent=2))

    print(f"\n所要 {time.time() - started:.0f} 秒", flush=True)
    ng = [r for r in results if not r["ok"]]
    print(f"通らなかった行: {len(ng)} / {len(results)}")
    for r in ng:
        why = r["error"] or f"物体が {r['max_object_move_m'] * 1000:.1f} mm 動いた"
        print(f"  {r['difficulty_level'] or 'Base'} {r['task_id']}  {why}")

    print("\n段ごとの手先のずれ（cm・Base 行との距離の中央値）")
    by_level: dict[str, list[float]] = {}
    for r in results:
        if r["category"] == "Base" or r["eef_shift_cm"] is None:
            continue
        by_level.setdefault(r["difficulty_level"], []).append(r["eef_shift_cm"])
    for level in sorted(by_level):
        v = sorted(by_level[level])
        print(f"  {level}  n={len(v):2d}  中央値 {statistics.median(v):5.2f}  "
              f"最小 {v[0]:5.2f}  最大 {v[-1]:5.2f}")


if __name__ == "__main__":
    main()
