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
    台本の列は `plan` で `TaskPlan` の項目を上書きできる（掴む順番と置き場所は
    その中の `steps`。`scripted_demo.steps_from_json`）
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
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_actions import merge_adjacent  # noqa: E402
from render_before_after import IMAGES, _view, _write_columns  # noqa: E402
from replay_yield import (MAX_STEPS, _replay_one, _variant,  # noqa: E402
                          passes_gates, suite_of)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True, metavar="FILE")
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--only", nargs="*", default=None, metavar="NAME",
                        help="spec のうち撃つ名前を絞る（並列に分けて流すため）")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    return parser.parse_args()


def _column(record, *, goal_step: int, keep: bool, length: int | None = None,
            prealign_steps: int = 0, eef_offset_cm: float = 0.0,
            disp_at_goal_mm: float | None = None) -> dict:
    """1 列ぶんの結果。**`_panel` と `_header` が読む形はここだけで決める。**

    `length` は「送った行動の本数」で、`steps` と違うのは短くした列だけである
    （`merged`。見出しに出す）。
    """
    return {"front": _view(record.front), "wrist": _view(record.wrist),
            "disp": [1000.0 * d for d in record.displacements],
            "goal_step": int(goal_step), "steps": int(record.steps),
            "length": int(record.steps if length is None else length),
            "prealign_steps": int(prealign_steps), "keep": bool(keep),
            "eef_offset_cm": float(eef_offset_cm),
            "disp_at_goal_mm": disp_at_goal_mm}


def _replayed(sd, env, column: dict, init_state: np.ndarray,
              actions: np.ndarray, ee_pos0: np.ndarray, seed: int) -> dict:
    """`human`・`prealign`・`merged` の列を撃つ。"""
    kind = column["kind"]
    merge_keys = {key: column[key] for key in ("scale", "dwell") if key in column}
    sequence = merge_adjacent(actions, **merge_keys) if kind == "merged" else actions
    method = "prealign" if kind == "prealign" else "open_loop"
    result, rollout = _replay_one(sd, env, init_state, sequence, seed, method, ee_pos0,
                                  record_images=True,
                                  planner=column.get("planner", "proportional"))
    rollout.attach_images()
    gates = passes_gates(result)
    # 摂動なしの列は手先がずれていないので、被覆の門は見ない。
    keep = gates["keep"] if kind == "prealign" else gates["reached"] and gates["clean"]
    return _column(rollout.record, goal_step=result["goal_step"], keep=keep,
                   length=len(sequence), prealign_steps=result["prealign_steps"],
                   eef_offset_cm=result["eef_offset_cm"],
                   disp_at_goal_mm=result["disp_at_goal_mm"])


def _scripted(sd, env, task: str, column: dict, init_state: np.ndarray) -> dict:
    """`scripted` の列を撃つ。**置き場所は人の手本の 0 コマ目と同じ。**

    `column["plan"]` を渡すと `TaskPlan` の項目をその列だけ上書きする
    （例 `{"command_envelope": true, "cruise": 0.85}`）。**台本の既定を書き換える前に、
    採るかどうかを動画で決めるためである。**

    **掴む順番と置く場所も、`plan` の中の `steps` で列ごとに変えられる**
    （`sd.steps_from_json`）。`steps` は**やらせたい順に並べた差分**で、
    書いた項目だけが既定を上書きする。

        "plan": {"cruise": 0.90,
                 "steps": [{"target": "moka_pot_2", "place_offset": [-0.045, -0.035]},
                           {"target": "moka_pot_1", "place_offset": [0.030, 0.035]}]}
    """
    import dataclasses

    overrides = dict(column.get("plan", {}))
    plan = sd._resolve_plan(task)
    if "steps" in overrides:
        overrides["steps"] = sd.steps_from_json(plan.steps, overrides["steps"])
    plan = dataclasses.replace(plan, **overrides)
    record = sd.run_episode(env, init_state, plan, MAX_STEPS, record_images=True,
                            glide=column.get("glide"))
    return {**_column(record, goal_step=record.goal_step, keep=record.success,
                      disp_at_goal_mm=(None if record.disp_at_goal_m is None else
                                       round(1000.0 * record.disp_at_goal_m, 3))),
            # **見出しに出すのは「実際に効いた」値である。**課題の既定で Ruckig が
            # 入るようになったので、spec に書いていなくても入っていることがある。
            "glide": bool(plan.glide if column.get("glide") is None else column["glide"])}


def _header(column: dict, run: dict) -> str:
    """列の見出し。**1 列 256 px に収まるよう 32 文字前後に抑える**（超えると採否が切れて見えない）。"""
    kind = column["kind"]
    if kind == "prealign":
        what = f"init{column['initstate']} {run['eef_offset_cm']:.1f}cm"
    elif kind == "merged":
        what = f"merged {run['length']}st"
    elif kind == "scripted":
        what = "script" + ("+ruckig" if run.get("glide") else "")
    else:
        what = "original"
    goal = "no goal" if run["goal_step"] < 0 else f"goal {run['goal_step']}"
    return f"{what} | {goal} | {'KEEP' if run['keep'] else 'DROP'}"


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
        runs, headers = [], []
        # **同じ bddl の列は env を使い回す。**摂動を掛ける `prealign` の列だけ
        # 別の bddl になる——ほかの列は 1 本ぶん作り直すだけ無駄である。
        envs: dict[str, Any] = {}
        try:
            for column in item["columns"]:
                variant = (_variant(task, column["initstate"])
                           if column["kind"] == "prealign" else task)
                if variant not in envs:
                    envs[variant] = sd._make_env(
                        sd._bddl_path(libero_root, suite, variant), True)
                env = envs[variant]
                if column["kind"] == "scripted":
                    run = _scripted(sd, env, task, column, init_state)
                else:
                    run = _replayed(sd, env, column, init_state,
                                    actions, ee_pos0, args.seed)
                headers.append(_header(column, run))
                print(f"  {headers[-1]}", flush=True)
                runs.append(run)
        finally:
            for env in envs.values():
                env.close()
        length = _write_columns(args.out / f"{item['name']}.mp4", headers, runs)
        (args.out / f"{item['name']}.json").write_text(json.dumps(
            {"task": task, "demo": demo, "frames": length,
             "columns": [{**column, **{k: v for k, v in run.items() if k not in IMAGES}}
                         for column, run in zip(item["columns"], runs)]},
            indent=2, ensure_ascii=False, default=float))
        print(f"  → {item['name']}.mp4 {length} コマ", flush=True)
        del runs


if __name__ == "__main__":
    main()
