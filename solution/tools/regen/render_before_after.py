#!/usr/bin/env python3
"""**手を加える前の元データ**と、**今回の作り直し**を、同じ画面に並べる。

**なぜ要るか。**成否の数（元データ 141/250 → 作り直し 246/250）だけでは、
**何がどう変わったのかが目で確かめられない。**この道具は 1 つの (課題, デモ)
について、**まったく同じ初期状態から** 2 通りを走らせて並べる。

| 列 | 中身 |
|---|---|
| **before** | **配布データの人の手本を、記録された行動のまま再生したもの。**手を加える前 |
| **after** | **`RimPickPlace` で作り直したもの。**方位を選んで縁をまたぐ |

**同じ初期状態を使う。**人の手本が持っている `states[0]` を両方に渡すので、
物の置き場所も腕の出発姿勢も完全に同じ場面での比較になる。

**画面に焼く数字。**その歩までに対象外の物体が動いた量（mm・1 mm を超えると
赤で `OVER 1mm`）・ゴールが立った歩・採点と同じ規則での採否。

用法（コンテナの中で打つ）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/render_before_after.py \\
        --states work/artifacts/libero_states --out work/regen/videos_ba --demos 3

Reads:
  - <states>/index.json・<states>/<suite>/<task>_demo.npz
Generates:
  - <out>/<課題>__demo<番号>.mp4 … 2 列 × 2 段（front / wrist）
  - <out>/summary.json … 焼いた数値
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))

from replay_yield import _replay_one  # noqa: E402

PANEL = 256
HEADER = 22
FPS = 20
MAX_STEPS = 300

TASKS = (
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
)
"""既定で並べる 5 課題。**評価で失点の 65% を出している家族である。**"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--task", action="append", metavar="NAME",
                        help="並べる課題。**省くと既定の 5 課題**")
    parser.add_argument("--demos", type=int, default=3, metavar="N",
                        help="1 課題あたりの本数（先頭から）")
    parser.add_argument("--pick", type=Path, default=None, metavar="FILE",
                        help="`{課題: [デモ番号]}` の JSON。**渡すとこちらが優先。**"
                             "元データが汚れている本を選んで並べるために使う")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    return parser.parse_args()


def _annotate(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]
              ) -> np.ndarray:
    """コマの下に数行を焼く。**026 の `render_contact_videos.py` と同じ作り。**"""
    import cv2

    canvas = np.ascontiguousarray(frame)
    for number, (text, color) in enumerate(lines):
        y = canvas.shape[0] - 6 - 14 * (len(lines) - 1 - number)
        cv2.putText(canvas, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    color, 1, cv2.LINE_AA)
    return canvas


def _header_band(width: int, text: str) -> np.ndarray:
    import cv2

    band = np.full((HEADER, width, 3), 24, dtype=np.uint8)
    cv2.putText(band, text, (5, HEADER - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return band


def _view(images: np.ndarray) -> np.ndarray:
    """**画像は学習データの向き（180 度回し）で溜まっている。**見るために戻す。"""
    return np.asarray(images)[:, ::-1, ::-1]


def _before(sd, env, init_state: np.ndarray, actions: np.ndarray,
            ee_pos0: np.ndarray | None, seed: int) -> dict:
    """人の手本を、記録された行動のまま再生する。"""
    result, rollout = _replay_one(sd, env, init_state, actions, seed,
                                  "open_loop", ee_pos0, record_images=True)
    rollout.attach_images()
    record = rollout.record
    return {"front": _view(record.front), "wrist": _view(record.wrist),
            "disp": [1000.0 * d for d in record.displacements],
            "goal_step": result["goal_step"],
            "worst_mm": 1000.0 * record.max_displacement,
            "steps": record.steps}


def _after(sd, env, init_state: np.ndarray, task: str) -> dict:
    """`RimPickPlace` で作り直したものを走らせる。"""
    env.seed(0)
    record = sd.run_episode(env, init_state, sd.TASK_PLANS[task], MAX_STEPS,
                            record_images=True)
    goal = record.steps if record.goal_reached else 0
    return {"front": _view(record.front), "wrist": _view(record.wrist),
            "disp": [1000.0 * d for d in record.displacements],
            "goal_step": goal,
            "worst_mm": 1000.0 * record.max_displacement,
            "steps": record.steps,
            "objects": {k: round(1000.0 * v, 3)
                        for k, v in record.object_displacements.items()}}


def _verdict(run: dict) -> tuple[str, bool]:
    """採点と同じ規則。**ゴール到達 ∧ 対象外の物体が 1 mm 未満。**"""
    reached = 0 < run["goal_step"] <= MAX_STEPS
    clean = run["worst_mm"] <= 1.0
    if reached and clean:
        return "KEEP", True
    why = []
    if not reached:
        why.append("no goal")
    if not clean:
        why.append("contact")
    return "DROP: " + ",".join(why), False


def main() -> None:
    args = _parse_args()
    import cv2
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    index = json.loads((args.states / "index.json").read_text())
    suite_of = {k.split("/")[1].replace("_demo.hdf5", ""): k.split("/")[0]
                for k in index}
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []

    picked: dict[str, list[int]] = {}
    if args.pick is not None:
        picked = {k: list(v) for k, v in json.loads(args.pick.read_text()).items()}
    for task in (args.task or list(picked) or list(TASKS)):
        suite = suite_of[task]
        npz = args.states / suite / f"{task}_demo.npz"
        env = sd._make_env(sd._bddl_path(libero_root, suite, task), True)
        print(f"=== {task} ===", flush=True)
        try:
            with np.load(npz, allow_pickle=False) as bundle:
                for demo in picked.get(task, list(range(args.demos))):
                    actions = bundle[f"demo_{demo}/actions"]
                    states = bundle[f"demo_{demo}/states"]
                    key = f"demo_{demo}/ee_pos"
                    ee_pos0 = bundle[key][0] if key in bundle.files else None
                    runs = [
                        {"label": "before (distributed demo)",
                         **_before(sd, env, states[0], actions, ee_pos0, args.seed)},
                        {"label": "after (RimPickPlace)",
                         **_after(sd, env, states[0], task)},
                    ]
                    for run in runs:
                        run["verdict"], run["kept"] = _verdict(run)
                        print(f"  {run['label']:<26} 歩 {run['steps']:>3}  "
                              f"goal {run['goal_step']:>3}  "
                              f"接触 {run['worst_mm']:>8.2f}mm  "
                              f"{run['verdict']}", flush=True)

                    length = max(len(r["front"]) for r in runs)
                    width = PANEL * len(runs)
                    path = args.out / f"{task}__demo{demo:03d}.mp4"
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                        (width, HEADER + 2 * PANEL))
                    bands = np.concatenate(
                        [_header_band(PANEL, r["label"]) for r in runs], axis=1)
                    for step in range(length):
                        row_front, row_wrist = [], []
                        for run in runs:
                            here = min(step, len(run["front"]) - 1)
                            disp = run["disp"][min(here, len(run["disp"]) - 1)]
                            over = disp > 1.0
                            goal = 0 < run["goal_step"] <= here + 1
                            lines = [
                                (f"nontarget {disp:7.2f}mm"
                                 + ("  OVER 1mm" if over else ""),
                                 (60, 60, 255) if over else (235, 235, 235)),
                                (f"step {here + 1}" + ("  GOAL" if goal else "")
                                 + f"   {run['verdict']}",
                                 (80, 220, 80) if run["kept"] else (60, 60, 255)),
                            ]
                            row_front.append(_annotate(run["front"][here], lines))
                            row_wrist.append(
                                np.ascontiguousarray(run["wrist"][here]))
                        frame = np.concatenate([
                            bands,
                            np.concatenate(row_front, axis=1),
                            np.concatenate(row_wrist, axis=1)], axis=0)
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                    print(f"  → {path.name}  {length} コマ  "
                          f"{path.stat().st_size / 1e6:.1f} MB", flush=True)
                    summary.append({
                        "task": task, "demo": demo, "video": path.name,
                        "runs": [{k: v for k, v in r.items()
                                  if k not in ("front", "wrist", "disp")}
                                 for r in runs]})
        finally:
            env.close()

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\n=== {args.out} ===")


if __name__ == "__main__":
    main()
