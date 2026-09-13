#!/usr/bin/env python3
"""**学習データに入れる予定の動き**を、元の人の手本と横に並べて動画にする。

**なぜ要るか。**採否は表の数字だけで決めない。腕が画面の中でどう動いたか——
寄せる区間で暴れていないか、短くした手本が雑に見えないか——は動画でしか分からない。

**1 本の動画に、同じ課題・同じ物体配置の列を横に並べる。**列の中身は `kind` で選ぶ。

| `kind` | 中身 | 作り方 |
|---|---|---|
| `human` | **元の動き。**人の手本を摂動なしでそのまま撃つ | `replay_yield._replay_one(open_loop)` |
| `prealign` | 腕の初期姿勢を振り、手本の出発点へ寄せてから手本を流す | `_replay_one(prealign, planner)` |
| `merged` | 人の手本の指令を隣どうしまとめて短くしたもの | `merge_actions.merge_adjacent` |
| `scripted` | 台本の手本（**物体配置は人の手本と同じ**） | `scripted_demo.run_episode` |

**すべてその場でシムを回して撮る。**画面に焼くもの（英字）——列の見出しに採否とゴールの歩、
各コマに区間（`PREALIGN`／`REPLAY`）・対象外の物体が動いた量（1 mm 超で赤）・歩数・
**300 歩を超えたら `OVER 300`**。短い列は最後のコマで止めて待つ。

用法（コンテナの中で打つ。`run_in_container.sh` が包む）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/render_training_pairs.py \\
        --spec work/regen/videos_training/spec.json \\
        --states work/artifacts/libero_states --out work/regen/videos_training

Reads:
  - <spec> … `[{"name": …, "task": …, "demo": 0, "columns": [{"kind": "human"}, …]}]`
  - <states>/index.json・<states>/<suite>/<課題>_demo.npz
Generates:
  - <out>/<name>.mp4 … 列を横に並べた 1 本
  - <out>/<name>.json … 列ごとの数値（画像は除く）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_actions import merge_adjacent  # noqa: E402
from render_before_after import (FPS, HEADER, PANEL, _annotate,  # noqa: E402
                                 _header_band, _view)
from replay_yield import (MAX_STEPS, _replay_one, _variant,  # noqa: E402
                          passes_gates, suite_of)

IMAGES = ("front", "wrist", "disp")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True, metavar="FILE")
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--only", nargs="*", default=None, metavar="NAME",
                        help="spec のうち撃つ名前を絞る（並列に分けて流すため）")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    return parser.parse_args()


def _replayed(sd, libero_root: Path, suite: str, task: str, column: dict,
              init_state: np.ndarray, actions: np.ndarray, ee_pos0: np.ndarray,
              seed: int) -> dict:
    """`human`・`prealign`・`merged` の列を撃つ。"""
    kind = column["kind"]
    variant = _variant(task, column["initstate"]) if kind == "prealign" else task
    sequence = (merge_adjacent(actions, column.get("scale", 1.0), column.get("dwell", 8))
                if kind == "merged" else actions)
    method = "prealign" if kind == "prealign" else "open_loop"
    env = sd._make_env(sd._bddl_path(libero_root, suite, variant), True)
    try:
        result, rollout = _replay_one(sd, env, init_state, sequence, seed, method, ee_pos0,
                                      record_images=True,
                                      planner=column.get("planner", "proportional"))
        rollout.attach_images()
    finally:
        env.close()
    gates = passes_gates(result)
    # 摂動なしの列は手先がずれていないので、被覆の門は見ない。
    keep = gates["keep"] if kind == "prealign" else gates["reached"] and gates["clean"]
    record = rollout.record
    return {"front": _view(record.front), "wrist": _view(record.wrist),
            "disp": [1000.0 * d for d in record.displacements],
            "goal_step": result["goal_step"], "steps": result["steps"],
            "prealign_steps": result["prealign_steps"], "keep": bool(keep),
            "length": int(len(sequence)), "eef_offset_cm": result["eef_offset_cm"],
            "disp_at_goal_mm": result["disp_at_goal_mm"]}


def _scripted(sd, libero_root: Path, suite: str, task: str, column: dict,
              init_state: np.ndarray) -> dict:
    """`scripted` の列を撃つ。**置き場所は人の手本の 0 コマ目と同じ。**

    `column["plan"]` を渡すと `TaskPlan` の項目をその列だけ上書きする
    （例 `{"command_envelope": true, "cruise": 0.85}`）。**台本の既定を書き換える前に、
    採るかどうかを動画で決めるためである。**
    """
    import dataclasses

    plan = dataclasses.replace(sd._resolve_plan(task), **column.get("plan", {}))
    # **掴む順番と置く場所も列ごとに変えられる。**`TaskPlan.steps` は物の並びなので JSON では
    # 書けない——順番は `reverse_steps`、置く場所は `place_offsets`（物の名前 → [x, y] m）で渡す。
    if column.get("reverse_steps"):
        plan = dataclasses.replace(plan, steps=list(reversed(plan.steps)))
    offsets = column.get("place_offsets", {})
    if offsets:
        plan = dataclasses.replace(plan, steps=[
            dataclasses.replace(step, place_offset=tuple(offsets[step.target]))
            if step.target in offsets else step for step in plan.steps])
    env = sd._make_env(sd._bddl_path(libero_root, suite, task), True)
    try:
        record = sd.run_episode(env, init_state, plan, MAX_STEPS, record_images=True,
                                glide=column.get("glide"))
    finally:
        env.close()
    return {"front": _view(record.front), "wrist": _view(record.wrist),
            "disp": [1000.0 * d for d in record.displacements],
            "goal_step": record.steps if record.goal_reached else -1,
            "steps": int(record.steps), "prealign_steps": 0,
            "keep": bool(record.success), "length": int(record.steps),
            "eef_offset_cm": 0.0,
            "disp_at_goal_mm": round(1000.0 * float(record.max_displacement), 3)}


def _header(column: dict, run: dict) -> str:
    """列の見出し。**1 列 256 px に収まるよう 32 文字前後に抑える**（超えると採否が切れて見えない）。"""
    kind = column["kind"]
    if kind == "prealign":
        what = f"init{column['initstate']} {run['eef_offset_cm']:.1f}cm"
    elif kind == "merged":
        what = f"merged {run['length']}st"
    elif kind == "scripted":
        what = "script" + ("+ruckig" if column.get("glide") else "")
    else:
        what = "original"
    goal = "no goal" if run["goal_step"] < 0 else f"goal {run['goal_step']}"
    return f"{what} | {goal} | {'KEEP' if run['keep'] else 'DROP'}"


def _panel(run: dict, step: int) -> tuple[np.ndarray, np.ndarray]:
    here = min(step, len(run["front"]) - 1)
    disp = run["disp"][min(here, len(run["disp"]) - 1)] if run["disp"] else 0.0
    goal = 0 < run["goal_step"] <= here + 1
    lines = []
    if run["prealign_steps"]:
        aligning = here < run["prealign_steps"]
        lines.append(("PREALIGN" if aligning else "REPLAY",
                      (80, 200, 255) if aligning else (200, 200, 200)))
    lines.append((f"nontarget {disp:6.2f}mm" + ("  OVER 1mm" if disp > 1.0 else ""),
                  (255, 60, 60) if disp > 1.0 else (235, 235, 235)))
    over = here + 1 > MAX_STEPS
    lines.append((f"step {here + 1}/{run['steps']}" + ("  GOAL" if goal else "")
                  + ("  OVER 300" if over else ""),
                  (255, 60, 60) if over else ((80, 220, 80) if goal else (235, 235, 235))))
    return _annotate(run["front"][here], lines), np.ascontiguousarray(run["wrist"][here])


def _write(path: Path, bands: list[np.ndarray], runs: list[dict]) -> int:
    import cv2

    width = PANEL * len(runs)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (width, HEADER + 2 * PANEL))
    length = max(len(run["front"]) for run in runs)
    top = np.concatenate(bands, axis=1)
    for step in range(length):
        panels = [_panel(run, step) for run in runs]
        frame = np.concatenate([top,
                                np.concatenate([p[0] for p in panels], axis=1),
                                np.concatenate([p[1] for p in panels], axis=1)])
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return length


def main() -> None:
    args = _parse_args()
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    suites = suite_of(args.states)
    args.out.mkdir(parents=True, exist_ok=True)
    for item in json.loads(args.spec.read_text()):
        if args.only and item["name"] not in args.only:
            continue
        task, demo = item["task"], int(item["demo"])
        suite = suites[task]
        print(f"=== {item['name']} ({task} demo{demo}) ===", flush=True)
        with np.load(args.states / suite / f"{task}_demo.npz", allow_pickle=False) as bundle:
            init_state = bundle[f"demo_{demo}/states"][0]
            actions = bundle[f"demo_{demo}/actions"]
            ee_pos0 = bundle[f"demo_{demo}/ee_pos"][0]
        runs, bands = [], []
        for column in item["columns"]:
            if column["kind"] == "scripted":
                run = _scripted(sd, libero_root, suite, task, column, init_state)
            else:
                run = _replayed(sd, libero_root, suite, task, column, init_state,
                                actions, ee_pos0, args.seed)
            header = _header(column, run)
            print(f"  {header}", flush=True)
            runs.append(run)
            bands.append(_header_band(PANEL, header))
        length = _write(args.out / f"{item['name']}.mp4", bands, runs)
        (args.out / f"{item['name']}.json").write_text(json.dumps(
            {"task": task, "demo": demo, "frames": length,
             "columns": [{**column, **{k: v for k, v in run.items() if k not in IMAGES}}
                         for column, run in zip(item["columns"], runs)]},
            indent=2, ensure_ascii=False, default=float))
        print(f"  → {item['name']}.mp4 {length} コマ", flush=True)
        del runs


if __name__ == "__main__":
    main()
