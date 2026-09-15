#!/usr/bin/env python3
"""接触が**どの局面で**起きているかを、人の手本の再生で測る。

**なぜ要るか。**評価の `object_disp.jsonl` は 1 本につき「どの物体が何 mm
動いたか」しか残さない。**そのため「掴む前にぶつけた」のか「運搬中にぶつけた」
のか「置くときにぶつけた」のかが分からず、打つ手が決まらない。**
接触を避ける手は局面ごとに全く違う——掴む前なら**近づく向き**、運搬中なら
**運ぶ高さと経路**、置くときなら**置く位置合わせ**である。

**測り方。**人の手本の行動列をそのまま再生し、**採点と同じ規則**（対象外の物体の
L1 変位 1 mm）で、局面ごとに変位がどれだけ増えたかを積む。局面は手本の
グリッパ指令と対象物の高さから切る。

| 記号 | 局面 | 切り方 |
|---|---|---|
| `A_approach` | 掴みに行くまで | 閉じる指令が出る前 |
| `B_grasp` | 掴んで持ち上げるまで | 閉じる指令から、対象が 5 cm 上がるまで |
| `C_transport` | 運搬 | 上がってから、下ろし始めるまで |
| `D_place` | 下ろして離すまで | 下ろし始めから、開く指令まで |
| `E_retreat` | 離れる | 開く指令のあと |

**同時に幾何も測る。**対象物のまわりを見て、**どの方位に何があるか**と
**空いている方位がどれだけ広いか**を出す。掴む対象が Z 軸まわりに回転対称なら、
**空いている方位から指を入れれば当てずに掴める**——それが成り立つかを数字で見る。

用法（コンテナの中で打つ）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/contact_clean/contact_phase.py \\
        --states work/artifacts/libero_states \\
        --out work/regen/contact_phase.json --demos 10

Reads:
  - <states>/index.json・<states>/<suite>/<task>_demo.npz
Generates:
  - <out> … 局面ごとの変位の増分と、対象物まわりの幾何
  - 標準出力 … 家族ごとの内訳
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))

HOT_TASKS = (
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
)
"""既定で測る課題。**評価で失点の 65% を出している家族**と、比較用の対照。"""

PHASES = ("A_approach", "B_grasp", "C_transport", "D_place", "E_retreat")


def _partner_kind(name: str, target: str) -> str:
    """接触の相手を、打つ手が変わる単位に丸める。

    **「何が当てたか」で手が変わる。**指なら**掴む向き**、腕なら**近づく経路**、
    運んでいる対象物なら**運ぶ高さと置く位置**である。
    """
    if "finger" in name or "pad" in name or "gripper" in name:
        return "gripper_finger"
    if name.startswith("robot0") or name.startswith("gripper0"):
        return "robot_arm"
    if name.startswith(target):
        return "carried_object"
    if name.startswith("table") or "floor" in name:
        return "table"
    return "other_object"


def _contacts(sim, watched_prefixes: tuple[str, ...], target: str
              ) -> dict[str, set[str]]:
    """いま当たっている組のうち、**対象外の物体が絡むものだけ**を拾う。

    **変位を測るだけでは「何が当てたか」が分からない。**MuJoCo の接触表を
    そのまま読めば、指か腕か運んでいる物かが 1 歩ごとに分かる。
    """
    found: dict[str, set[str]] = {}
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        names = []
        for geom in (contact.geom1, contact.geom2):
            try:
                names.append(sim.model.geom_id2name(int(geom)) or "")
            except Exception:
                names.append("")
        for first, second in (names, names[::-1]):
            hit = next((p for p in watched_prefixes if first.startswith(p)), None)
            if hit is not None:
                found.setdefault(hit, set()).add(_partner_kind(second, target))
    return found


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="FILE")
    parser.add_argument("--task", action="append", metavar="NAME",
                        help="測る課題。**省くと既定の一覧**")
    parser.add_argument("--demos", type=int, default=50, metavar="N")
    parser.add_argument("--layout", type=int, default=0, metavar="N")
    return parser.parse_args()


def _phase_bounds(actions: np.ndarray, target_z: np.ndarray) -> dict[str, tuple[int, int]]:
    """手本の指令と対象物の高さから、局面の境目を決める。"""
    grip = actions[:, 6]
    steps = len(actions)
    close = int(np.argmax(grip > 0.0)) if np.any(grip > 0.0) else steps
    after = grip[close:]
    open_at = close + int(np.argmax(after < 0.0)) if np.any(after < 0.0) else steps
    z0 = float(target_z[0])
    lifted = np.nonzero(target_z > z0 + 0.05)[0]
    lift_done = int(lifted[0]) if len(lifted) else close
    # **下ろし始め**は、対象がいちばん高かった所を最後に離れた歩とする。
    top = float(target_z.max())
    near_top = np.nonzero(target_z > top - 0.02)[0]
    descend = int(near_top[-1]) if len(near_top) else open_at
    bounds = [0, close, max(lift_done, close), max(descend, lift_done, close),
              max(open_at, descend), steps]
    bounds = [min(b, steps) for b in bounds]
    for i in range(1, len(bounds)):
        bounds[i] = max(bounds[i], bounds[i - 1])
    return {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(PHASES)}


def _geometry(rollout, target: str) -> dict:
    """対象物のまわりの方位を測る。**空いている方位の広さが要点。**"""
    sim = rollout.env.env.sim
    target_pos = np.asarray(rollout.obs[f"{target}_pos"], dtype=np.float64)
    try:
        base = np.asarray(sim.data.body_xpos[sim.model.body_name2id("robot0_base")],
                          dtype=np.float64)
    except Exception:            # 名前が違う版でも幾何の残りは使える
        base = np.zeros(3)

    # **物体全体の張り出しを原点から測る。**本体が小さな箱の集まりでできて
    # いる物では、geom 1 つの幅は全体の幅にならない。
    extent: dict[str, float] = {}
    origin: dict[str, np.ndarray] = {}
    for name, body_id in rollout.env.env.obj_body_id.items():
        geoms = [i for i in range(sim.model.ngeom)
                 if sim.model.geom_contype[i]
                 and (sim.model.geom_id2name(i) or "").startswith(name)]
        if not geoms:
            continue
        centre = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
        origin[name] = centre
        extent[name] = max(
            float(np.linalg.norm(sim.data.geom_xpos[i][:2] - centre[:2])
                  + max(sim.model.geom_size[i][0], sim.model.geom_size[i][1]))
            for i in geoms)
    gaps = {}
    for name in extent:
        if name == target:
            continue
        distance = float(np.linalg.norm(origin[name][:2] - origin[target][:2]))
        gaps[name] = {
            "gap_cm": round(100 * (distance - extent[name] - extent.get(target, 0.0)), 2),
            "azimuth_deg": round(float(np.degrees(np.arctan2(
                *(origin[name][:2] - origin[target][:2])[::-1]))), 1)}

    blocked: list[dict] = []
    for i in range(sim.model.ngeom):
        if not sim.model.geom_contype[i]:
            continue
        name = sim.model.geom_id2name(i) or ""
        if not name or name.startswith("robot0") or name.startswith("gripper"):
            continue
        if name.startswith(target):
            continue
        pos = np.asarray(sim.data.geom_xpos[i], dtype=np.float64)
        # **掴む高さの帯だけを見る。**床や天板は方位を塞がない。
        if pos[2] < target_pos[2] - 0.03 or pos[2] > target_pos[2] + 0.12:
            continue
        delta = pos[:2] - target_pos[:2]
        distance = float(np.linalg.norm(delta))
        if distance > 0.20 or distance < 1e-6:
            continue
        radius = float(max(sim.model.geom_size[i][0], sim.model.geom_size[i][1]))
        half = float(np.degrees(np.arctan2(min(radius, distance * 0.999), distance)))
        blocked.append({"geom": name, "distance_cm": round(100 * distance, 2),
                        "azimuth_deg": round(float(np.degrees(np.arctan2(delta[1], delta[0]))), 1),
                        "half_width_deg": round(half, 1)})

    # 1 度刻みで塞がれている方位を塗り、**空いている連続区間のいちばん広い所**を出す。
    free = np.ones(360, dtype=bool)
    for item in blocked:
        centre = int(round(item["azimuth_deg"])) % 360
        span = int(np.ceil(item["half_width_deg"]))
        for d in range(-span, span + 1):
            free[(centre + d) % 360] = False
    widest, start, run, run_start = 0, 0, 0, 0
    doubled = np.concatenate([free, free])
    for d in range(720):
        if doubled[d]:
            if run == 0:
                run_start = d
            run += 1
            if run > widest:
                widest, start = run, run_start
        else:
            run = 0
    widest = min(widest, 360)
    to_base = np.degrees(np.arctan2(*(base[:2] - target_pos[:2])[::-1]))
    return {"target_pos": [round(float(v), 4) for v in target_pos],
            "target_extent_cm": round(100 * extent.get(target, 0.0), 2),
            "gaps": gaps,
            "blocked": sorted(blocked, key=lambda x: x["distance_cm"])[:8],
            "free_span_deg": int(widest),
            "free_centre_deg": round(float((start + widest / 2) % 360), 1),
            "azimuth_to_robot_base_deg": round(float(to_base), 1)}


def _one_demo(sd, env, bundle, demo: int) -> dict | None:
    """1 本を再生する。**環境は課題ごとに 1 つ作って使い回す**（生成に 2 秒かかる）。"""
    if True:
        env.seed(0)
        actions = np.asarray(bundle[f"demo_{demo}/actions"], dtype=np.float32)
        states = bundle[f"demo_{demo}/states"]
        rollout = sd._Rollout(env, states[0], len(actions) + 20,
                              record_images=False)
        interest = list(env.obj_of_interest)
        target = interest[0]
        geometry = _geometry(rollout, target)
        watched = list(rollout._watched_keys)
        origin = rollout._watched_origin.copy()

        prefixes = tuple(k[:-4] for k in watched)
        eef_path = []
        sim = env.env.sim
        target_z, disp = [], []
        touched: dict[str, dict[str, list[int]]] = {}
        for action in actions:
            if not rollout._take_step(action):
                break
            positions = np.stack([rollout.obs[k] for k in watched]) \
                if watched else np.zeros((0, 3))
            disp.append(np.abs(positions - origin).sum(axis=1))
            target_z.append(float(rollout.obs[f"{target}_pos"][2]))
            eef_path.append(np.asarray(rollout.obs["robot0_eef_pos"],
                                       dtype=np.float64).copy())
            for obj, kinds in _contacts(sim, prefixes, target).items():
                for kind in kinds:
                    touched.setdefault(obj, {}).setdefault(kind, []).append(
                        len(disp) - 1)
        if not disp:
            return None
        disp = np.stack(disp)                       # (steps, objects)
        bounds = _phase_bounds(actions[:len(disp)], np.asarray(target_z))
        # **掴む瞬間、手先が対象のどちら側に居たか。**丼は Z 軸まわりに
        # 回転対称なので、**この向きは自由に選べる量である。**
        grasp_at = min(bounds["B_grasp"][0], len(eef_path) - 1)
        delta = eef_path[grasp_at][:2] - np.asarray(geometry["target_pos"])[:2]
        grasp = {"azimuth_deg": round(float(np.degrees(np.arctan2(delta[1], delta[0]))), 1),
                 "radius_cm": round(100 * float(np.linalg.norm(delta)), 2),
                 "step": int(grasp_at)}

        # **局面ごとの増分。**その局面の終わりまでの最大値の差を取る。
        running = np.maximum.accumulate(disp, axis=0)
        per_phase = {}
        for name, (low, high) in bounds.items():
            before = running[low - 1] if low > 0 else np.zeros(disp.shape[1])
            after = running[min(high, len(running)) - 1] if high > 0 else before
            per_phase[name] = {watched[j][:-4]: round(1000.0 * float(after[j] - before[j]), 3)
                               for j in range(disp.shape[1])}
        total = {watched[j][:-4]: round(1000.0 * float(running[-1][j]), 3)
                 for j in range(disp.shape[1])}
        worst = max(total, key=total.get) if total else None
        first = None
        if worst is not None and total[worst] > 1.0:
            step = int(np.argmax(disp[:, watched.index(worst + "_pos")] > 0.001))
            for name, (low, high) in bounds.items():
                if low <= step < high:
                    first = name
                    break
        return {"demo": demo, "steps": int(len(disp)), "target": target,
                "obj_of_interest": interest,
                "bounds": {k: list(v) for k, v in bounds.items()},
                "per_phase_mm": per_phase, "total_mm": total,
                "worst": worst, "worst_mm": total.get(worst, 0.0),
                "touched_by": {obj: {kind: [len(steps), int(steps[0])]
                                     for kind, steps in kinds.items()}
                               for obj, kinds in touched.items()},
                "first_violation_phase": first, "grasp": grasp,
                "geometry": geometry}


def main() -> None:
    args = _parse_args()
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    index = json.loads((args.states / "index.json").read_text())
    suite_of = {k.split("/")[1].replace("_demo.hdf5", ""): k.split("/")[0]
                for k in index}
    tasks = args.task or list(HOT_TASKS)
    out: dict[str, list[dict]] = {}

    for task in tasks:
        suite = suite_of.get(task)
        if suite is None:
            print(f"!! {task}: libero_states に無い", flush=True)
            continue
        npz = args.states / suite / f"{task}_demo.npz"
        print(f"=== {task} ===", flush=True)
        rows = []
        env = sd._make_env(sd._bddl_path(libero_root, suite, task), False)
        try:
            with np.load(npz, allow_pickle=False) as bundle:
                count = len({k.split('/')[0] for k in bundle.files})
                for demo in range(min(args.demos, count)):
                    row = _one_demo(sd, env, bundle, demo)
                    if row:
                        rows.append(row)
        finally:
            env.close()
        out[task] = rows
        if not rows:
            continue
        geometry = rows[0]["geometry"]
        print(f"  対象 {rows[0]['target']}  動かしてよい {rows[0]['obj_of_interest']}")
        print(f"  空いている方位 {geometry['free_span_deg']}度"
              f"（中心 {geometry['free_centre_deg']}度・ロボット基部は"
              f" {geometry['azimuth_to_robot_base_deg']}度）")
        near = sorted(geometry["gaps"].items(), key=lambda x: x[1]["gap_cm"])[:3]
        print(f"  対象の張り出し {geometry['target_extent_cm']:.2f} cm"
              f"  **指の開き 4.81 cm なので縁をまたいで掴むしかない**")
        for name, gap in near:
            print(f"    隙間 {name:<34}{gap['gap_cm']:>6.2f} cm"
                  f"  方位 {gap['azimuth_deg']:>7.1f}度")
        blocked_az = near[0][1]["azimuth_deg"] if near else 0.0
        offsets = []
        for row in rows:
            difference = abs(row["grasp"]["azimuth_deg"] - blocked_az)
            offsets.append(min(difference, 360 - difference))
        print(f"  掴んだ向き: いちばん近い障害物の方位から"
              f" 中央 {np.median(offsets):.0f}度 離れている"
              f"（半径 中央 {np.median([r['grasp']['radius_cm'] for r in rows]):.2f} cm）")
        dirty_off = [o for o, r in zip(offsets, rows) if r["worst_mm"] > 1.0]
        clean_off = [o for o, r in zip(offsets, rows) if r["worst_mm"] <= 1.0]
        if dirty_off and clean_off:
            print(f"    1 mm 超の手本 {np.median(dirty_off):.0f}度 /"
                  f" 収まった手本 {np.median(clean_off):.0f}度"
                  f"  **離れているほうが清潔なら、向きが効いている証拠**")
        for item in geometry["blocked"][:4]:
            print(f"    塞ぐもの {item['geom']:<34} {item['distance_cm']:>5.1f}cm"
                  f"  方位 {item['azimuth_deg']:>6.1f}度 ±{item['half_width_deg']:.0f}度")
        dirty = [r for r in rows if r["worst_mm"] > 1.0]
        print(f"  1 mm を超えた手本 {len(dirty)}/{len(rows)}")
        share = {p: 0.0 for p in PHASES}
        for row in dirty:
            for p in PHASES:
                share[p] += max(0.0, row["per_phase_mm"][p].get(row["worst"], 0.0))
        grand = sum(share.values()) or 1.0
        print(f"  {'局面':<14}{'変位の増分':>12}{'割合':>8}   最初に超えた本数")
        for p in PHASES:
            firsts = sum(1 for r in dirty if r["first_violation_phase"] == p)
            print(f"  {p:<14}{share[p]:>10.1f}mm{100 * share[p] / grand:>7.1f}%"
                  f"   {firsts:>3}")
        by_kind: dict[str, int] = {}
        for row in dirty:
            for kind in row.get("touched_by", {}).get(row["worst"], {}):
                by_kind[kind] = by_kind.get(kind, 0) + 1
        if by_kind:
            print("  **何が当てたか**（1 mm を超えた手本のうち、その相手が触れた本数）")
            for kind, count in sorted(by_kind.items(), key=lambda x: -x[1]):
                print(f"    {kind:<18}{count:>3} / {len(dirty)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== {args.out} ===")


if __name__ == "__main__":
    main()
