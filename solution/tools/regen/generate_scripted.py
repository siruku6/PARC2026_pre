#!/usr/bin/env python3
"""**台本で作る課題**の手本を、初期姿勢と外観を振って作る。合格した回だけ npz で書き出す。

**なぜ `generate_perturbed.py` と別なのか。**あちらは人の手本を撃ち直す。
丼 5 課題は土台（`libero_plus_bowl5`）がもう**台本の掴み方（縁をまたぐ `RimPickPlace`）**に
入れ替わっているので、人の手本を撃ち直して足すと**同じ場面に掴み方の正解が 2 つできる。**
モカポット 2 つは人の手本が 1 本も 300 歩に入らない。**だからこの 6 課題は台本で作る。**

**台本は閉ループ**（物体の位置を見て動く）なので、腕の初期姿勢がずれても同じ台本で撃てる。

| 課題 | 置き場所の出どころ | 理由 |
|---|---|---|
| 丼 5 課題 | `.pruned_init`（採点器と同じ 50 通り） | `on_the_ramekin` は人の手本の置き場所では 0/50 しか通らない（`run_scripted_bowls.sh`） |
| モカポット 2 つ | 人の手本の置き場所（50 通り） | 台本を人の順番・置き場所に揃えて 50/50 を確かめたのがこちら（034 の 9 節） |

**本数・デモ（置き場所）ごとの上限・外観・塊の割合**は `generate_perturbed.py` と同じ仕組み
（`_claim`・`appearance.draw`・`--quota-fraction`）。**ゴールの 10 歩後で切る**（`POST_GOAL_STEPS`）。

用法（コンテナの中で打つ）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/generate_scripted.py \\
        --states work/artifacts/libero_states --out-npz work/regen/npz_scripted_v2/s01 \\
        --quota work/regen/quota_4000.json --quota-fraction 0.5 \\
        --initstates $(python3 -c "import json;print(*json.load(open('work/regen/initstates_48.json'))['initstates'])") \\
        --appearance plain=25,light=25,background=25,camera=25 --per-demo-cap 3 --workers 12

Reads:
  - <quota> … `quota`（課題ごとの本数）と `scripted`（この道具が作る課題）
  - <states>/<suite>/<task>_demo.npz（モカポットの置き場所）・コンテナの `.pruned_init`
Generates:
  - <out-npz>/<課題>__layout<番号>__init<番号>__scripted.npz … `npz_to_lerobot build` が読む形
  - <out-npz>/manifest.jsonl … 1 回 1 行（落ちた回も残す）
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))

from appearance import draw as draw_appearance, parse_weights  # noqa: E402
from generate_perturbed import _claim, _release  # noqa: E402
from replay_yield import MAX_STEPS, POST_GOAL_STEPS, _variant, suite_of  # noqa: E402

FROM_DEMOS = ("KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",)
"""人の手本の置き場所で撃つ課題。**それ以外は `.pruned_init`。**"""

UNPERTURBED = ("KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",)
"""**腕の初期姿勢を振らない課題**（外観だけ振る）。

モカポット 2 つは初期姿勢を振ると 300 歩に入らない——9/14 の試し撃ちで
初期姿勢 285 は 50 置き場所中 0 本（全部 301 歩前後で打ち切り）、372 は 2 本中 1 本（292 歩）。
振らなければ 50/50・中央 269 歩（034 の 9 節）。**単位の番号は外観の引きにだけ使う。**
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out-npz", type=Path, required=True, metavar="DIR")
    parser.add_argument("--quota", type=Path, required=True, metavar="JSON")
    parser.add_argument("--quota-fraction", type=float, default=1.0, metavar="F")
    parser.add_argument("--initstates", type=int, nargs="+", required=True, metavar="N")
    parser.add_argument("--appearance", default="off", metavar="WEIGHTS")
    parser.add_argument("--per-demo-cap", type=int, default=0, metavar="N",
                        help="同じ (課題, 置き場所) を何本まで採るか。下限で、本数の多い課題は引き上げる")
    parser.add_argument("--skip", type=Path, default=None, metavar="FILE")
    parser.add_argument("--workers", type=int, default=12, metavar="N")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    return parser.parse_args()


def _run_item(item: dict, args: argparse.Namespace, shared: tuple) -> list[dict]:
    """(課題, 初期姿勢) 1 単位を撃つ。**env はここで 1 回だけ作る。**"""
    if item["budget"] <= 0:
        return []
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    task = item["task"]
    plan = sd._resolve_plan(task)
    from_demos = args.states if task in FROM_DEMOS else None
    layouts = sd._init_states(libero_root, plan.suite, task, from_demos)
    look = draw_appearance(task, f"{args.seed}:{task}:{item['initstate']}",
                           parse_weights(args.appearance))
    initstate = 0 if task in UNPERTURBED else item["initstate"]
    variant = _variant(task, initstate, look)
    env = sd._make_env(sd._bddl_path(libero_root, plan.suite, variant), with_images=True)
    rows, written = [], 0
    try:
        for layout in item["layouts"]:
            if written >= item["budget"]:
                break
            claim = _claim(shared, task, layout)
            if claim == "task_full":
                break
            if claim == "demo_full":
                continue
            started = time.perf_counter()
            env.seed(args.seed)
            record = sd.run_episode(env, layouts[layout], plan, MAX_STEPS,
                                    record_images=True, post_goal_steps=POST_GOAL_STEPS)
            keep = bool(record.success and 0 < record.goal_step <= MAX_STEPS)
            row = {"task": task, "demo": layout, "initstate": initstate, "unit": item["initstate"],
                   "method": "scripted", "init_source": "demos" if from_demos else "pruned",
                   "keep": keep, "goal_step": record.goal_step, "steps": record.steps,
                   "collided": bool(record.collided),
                   "disp_full_mm": round(1000.0 * record.max_displacement, 3),
                   "appearance": look["kind"], "variant": variant,
                   "seconds": round(time.perf_counter() - started, 2)}
            if keep:
                # **単位の番号も名前に入れる。**初期姿勢を振らない課題は init が全部 0 なので、
                # 入れないと同じ置き場所の別の外観が上書きし合う。
                destination = (args.out_npz / f"{task}__layout{layout:03d}"
                               f"__init{initstate:03d}__unit{item['initstate']:03d}__scripted.npz")
                np.savez(destination, state=np.stack(record.states),
                         action=np.stack(record.actions), front=record.front,
                         wrist=record.wrist, success=True, steps=record.steps,
                         instruction=env.language_instruction, source_task=task,
                         source_demo=layout, initstate=initstate,
                         appearance=look["kind"], variant=variant, method="scripted",
                         init_source=row["init_source"], goal_step=record.goal_step,
                         displacement_m=np.asarray(record.displacements, dtype=np.float32))
                row["npz"] = destination.name
                written += 1
            else:
                _release(shared, task, layout)
            rows.append(row)
            # 画像を手放してから次を撃つ（`generate_perturbed._run_item` と同じ理由）。
            record = None
    finally:
        env.close()
    return rows


def _worker(queue: "mp.Queue", results: "mp.Queue", args: argparse.Namespace,
            shared: tuple) -> None:
    while True:
        item = queue.get()
        if item is None:
            break
        try:
            results.put(_run_item(item, args, shared))
        except Exception as error:  # 1 単位の失敗で全体を止めない
            results.put([{"task": item["task"], "initstate": item["initstate"],
                          "keep": False, "error": repr(error)}])


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.out_npz.mkdir(parents=True, exist_ok=True)
    quota_file = json.loads(args.quota.read_text())
    tasks = [task for task in quota_file.get("scripted", ())
             if quota_file["quota"].get(task, 0) > 0]
    suite_of(args.states)  # 置き場の形が違えばここで落とす

    before, demo_before = Counter(), Counter()
    if args.skip and args.skip.exists():
        for line in args.skip.open():
            if line.strip() and (row := json.loads(line)).get("keep"):
                before[row["task"]] += 1
                demo_before[f"{row['task']}#{row['demo']}"] += 1
    allowed = {task: math.ceil(quota_file["quota"][task] * args.quota_fraction)
               for task in tasks}
    remaining = {task: max(0, allowed[task] - before[task]) for task in tasks}
    layout_count = 50
    demo_caps = {task: (max(args.per_demo_cap, math.ceil(1.5 * allowed[task] / layout_count))
                        if args.per_demo_cap else 0) for task in tasks}

    rng = random.Random(args.seed)
    items = []
    for initstate in args.initstates:
        for task in tasks:
            layouts = list(range(layout_count))
            random.Random(f"{args.seed}:{task}").shuffle(layouts)
            shift = (initstate * 7919) % layout_count
            items.append({"task": task, "initstate": initstate,
                          "layouts": layouts[shift:] + layouts[:shift],
                          "budget": (0 if remaining[task] == 0 else
                                     max(1, math.ceil(2 * remaining[task] / len(args.initstates))))})
    rng.shuffle(items)
    print(f"=== 台本で作る {len(tasks)} 課題 / 初期姿勢 {len(args.initstates)} 個 / "
          f"残り {sum(remaining.values())} 本 / 並列 {args.workers} ===")
    for task in tasks:
        print(f"  {task[:60]:<62} 累計の上限 {allowed[task]:>4}（これまで {before[task]}）"
              f"・置き場所ごと {demo_caps[task]}")

    manager = mp.Manager()
    shared = (manager.Lock(), manager.dict(dict(before)), manager.dict(dict(demo_before)),
              allowed, demo_caps)
    queue, results = mp.Queue(), mp.Queue()
    for item in items:
        queue.put(item)
    for _ in range(args.workers):
        queue.put(None)
    workers = [mp.Process(target=_worker, args=(queue, results, args, shared))
               for _ in range(args.workers)]
    for process in workers:
        process.start()

    written, attempts, started = 0, 0, time.time()
    with (args.out_npz / "manifest.jsonl").open("a") as handle:
        for done_count in range(1, len(items) + 1):
            batch = results.get()
            for row in batch:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                attempts += 1
                written += bool(row.get("keep"))
            handle.flush()
            if batch:
                print(f"[{done_count}/{len(items)}] {batch[0]['task'][:40]} init={batch[0]['initstate']}"
                      f" — **{written} 本** / {attempts} 回 / {(time.time() - started) / 60:.1f} 分",
                      flush=True)
    for process in workers:
        process.join(timeout=10)
    print(f"\n=== 書き出し {written} 本 / 撃った回 {attempts} ===")
    print("END_OF_SCRIPTED")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
