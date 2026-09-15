#!/usr/bin/env python3
"""**道 A**：300 コマを超えて外した人の手本を、指令をまとめて短くし、外観を振って撮り直す。

**なぜ要るか。**300 コマを超える手本は採点の打ち切りに入らないので土台から外した
（`keep_replace_v2.json`）。外した本数ぶんは別の手本の撃ち直しで埋めているが、
**その組（置き場所と人の動き方）は戻らない。**指令を隣どうし 1 歩にまとめると
（`merge_actions.merge_adjacent`）、多くの組が 300 歩以内にゴールする（034 の 9.4.4 節）。

**どの版で撮るか。**下見（描画なし）で門を通った版のうち、**まとめ方の弱い順**に撃つ——
`orig`（まとめない。撃ち直すと人の記録より早くゴールが立つ組がある）→ `m80`（上限の 8 割）→
`m100`（上限いっぱい）。**速くした手本は少ないほどよい**（その課題だけ動きの速さの分布が変わる）。
シムは同じ設定でも毎回わずかに揺らぐので、1 本ごとに門を見直し、落ちたら次の版を撃つ。

**本数は配布データでその組が持っていた本数と同じ**（`demo_map.json` の `episodes`）。
1 本ごとに外観だけを引き直す（物理は変わらない）。**腕の初期姿勢は振らない**
——配布データの手本と同じ、素の初期姿勢の置き換えである。

門は `reached`（300 歩以内にゴール）と `clean`（その時点で対象外の物体が 1 mm 以内）の 2 つ。
**`covered`（手先のずれ 2 cm 以上）は見ない。**初期姿勢を振らないので必ず 0 cm になる。

用法（コンテナの中で打つ）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/generate_merged.py --out-npz work/regen/npz_merged_v2 \\
        --appearance plain=25,light=25,background=25,camera=25 --workers 4

Reads:
  - <long-rows>（`work/regen/compress/merge_long.py` の出力）・<moka-rows>（`merge_pilot.py` の出力）
  - <map>・<scores>（組 → 本家のデモ番号と本数）・<states>/<suite>/<課題>_demo.npz
Generates:
  - <out-npz>/<課題>__demo<番号>__init000__copy<番号>__merge_<版>.npz … `npz_to_lerobot build` が読む形
  - <out-npz>/manifest.jsonl … 1 回 1 行（落ちた回も残す）
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))

from appearance import draw as draw_appearance, parse_weights  # noqa: E402
from merge_actions import merge_adjacent  # noqa: E402
from replay_yield import _replay_one, _variant, passes_gates, suite_of  # noqa: E402

PREFERENCE = (("orig", None), ("m80", 0.8), ("m100", 1.0))
"""撃つ順（版の名前, まとめる上限の倍率）。**まとめ方の弱い順。**"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, default=Path("work/artifacts/libero_states"))
    parser.add_argument("--map", type=Path, default=Path("work/regen/contact_v2/demo_map.json"))
    parser.add_argument("--scores", type=Path,
                        default=Path("work/regen/contact_v2/contact_scores.json"))
    parser.add_argument("--long-rows", type=Path,
                        default=Path("work/regen/compress/merge_long.jsonl"))
    parser.add_argument("--moka-rows", type=Path,
                        default=Path("work/regen/compress/moka2_pilot.jsonl"))
    parser.add_argument("--out-npz", type=Path, required=True, metavar="DIR")
    parser.add_argument("--appearance", default="off", metavar="WEIGHTS")
    parser.add_argument("--workers", type=int, default=4, metavar="N")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="試し撃ち用。先頭 N 本だけ撃つ（0 で全部）")
    return parser.parse_args()


def _plan(args: argparse.Namespace) -> list[dict]:
    """下見で通った組ごとに、配布データでの本数だけ 1 本ずつの仕事を並べる。"""
    passed: dict[tuple[str, int], set[str]] = defaultdict(set)
    for line in args.long_rows.open():
        if line.strip() and (row := json.loads(line))["keep"]:
            passed[(row["task"], int(str(row["demo"]).removeprefix("demo_")))].add(row["variant"])
    for line in args.moka_rows.open():
        if line.strip() and (row := json.loads(line))["keep"]:
            passed[(row["task"], int(row["demo"]))].add(row["variant"])

    demo_of = {row["fingerprint"]: int(row["source_demo"].removeprefix("demo_"))
               for row in json.loads(args.scores.read_text())}
    copies: Counter = Counter()
    for group in json.loads(args.map.read_text()):
        task = group["source_file"].split("/")[1].replace("_demo.hdf5", "")
        copies[(task, demo_of[group["fingerprint"]])] += len(group["episodes"])

    suites = suite_of(args.states)
    jobs = []
    for (task, demo), variants in sorted(passed.items()):
        order = [(name, scale) for name, scale in PREFERENCE if name in variants]
        for copy in range(copies[(task, demo)]):
            jobs.append({"task": task, "suite": suites[task], "demo": demo, "copy": copy,
                         "order": order})
    return jobs


def _run_job(job: dict, args: argparse.Namespace) -> list[dict]:
    """1 本ぶん。外観を引き、通る版が出るまで弱い順に撃つ。"""
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    task, demo = job["task"], job["demo"]
    look = draw_appearance(task, f"{args.seed}:{task}:merge:{demo}:{job['copy']}",
                           parse_weights(args.appearance))
    variant = _variant(task, 0, look)
    npz = args.states / job["suite"] / f"{task}_demo.npz"
    with np.load(npz, allow_pickle=False) as bundle:
        actions = bundle[f"demo_{demo}/actions"]
        init_state = bundle[f"demo_{demo}/states"][0]
    env = sd._make_env(sd._bddl_path(libero_root, job["suite"], variant), with_images=True)
    rows = []
    try:
        for name, scale in job["order"]:
            started = time.perf_counter()
            sequence = actions if scale is None else merge_adjacent(actions, scale)
            result, rollout = _replay_one(sd, env, init_state, sequence, args.seed,
                                          "open_loop", None, record_images=True)
            gates = passes_gates(result)
            keep = gates["reached"] and gates["clean"]
            row = {"task": task, "demo": demo, "copy": job["copy"], "initstate": 0,
                   "method": f"merge_{name}", "raw_length": int(len(actions)),
                   "length": int(len(sequence)), "reached": gates["reached"],
                   "clean": gates["clean"], "keep": keep,
                   "appearance": look["kind"], "variant": variant,
                   "seconds": round(time.perf_counter() - started, 2),
                   **{k: v for k, v in result.items()
                      if k in ("goal_step", "disp_at_goal_mm", "disp_full_mm", "steps")}}
            if keep:
                rollout.attach_images()
                record = rollout.record
                destination = (args.out_npz / f"{task}__demo{demo:03d}__init000"
                                f"__copy{job['copy']:02d}__merge_{name}.npz")
                # 鍵は `generate_perturbed._write_npz` と揃える（`npz_to_lerobot build` が読む形）。
                np.savez(destination, state=np.stack(record.states),
                         action=np.stack(record.actions), front=record.front,
                         wrist=record.wrist, success=True, steps=record.steps,
                         instruction=env.language_instruction, source_task=task,
                         source_demo=demo, initstate=0, appearance=look["kind"],
                         variant=variant, method=f"merge_{name}", planner="none",
                         prealign_steps=0,
                         displacement_m=np.asarray(record.displacements, dtype=np.float32),
                         disp_at_goal_mm=result["disp_at_goal_mm"],
                         goal_step=result["goal_step"])
                row["npz"] = destination.name
            rows.append(row)
            # 画像を手放してから次を撃つ（`generate_perturbed._run_item` と同じ理由）。
            result = rollout = None
            if keep:
                break
    finally:
        env.close()
    return rows


def _worker(job: dict) -> list[dict]:
    try:
        return _run_job(job, _worker.args)
    except Exception as error:  # 1 本の失敗で全体を止めない
        return [{"task": job["task"], "demo": job["demo"], "copy": job["copy"],
                 "keep": False, "error": repr(error)}]


def _init_worker(args: argparse.Namespace) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _worker.args = args


def main() -> None:
    args = _parse_args()
    args.out_npz.mkdir(parents=True, exist_ok=True)
    jobs = _plan(args)
    if args.limit:
        jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]
    per_task = Counter(job["task"] for job in jobs)
    print(f"=== 道 A {len(jobs)} 本 / {len(per_task)} 課題 / 並列 {args.workers} ===")
    for task, count in sorted(per_task.items()):
        print(f"  {task[:60]:<62} {count:>4} 本")

    written, attempts, started = Counter(), 0, time.time()
    with (args.out_npz / "manifest.jsonl").open("a") as handle, \
            mp.get_context("spawn").Pool(args.workers, _init_worker, (args,)) as pool:
        for done, rows in enumerate(pool.imap_unordered(_worker, jobs), start=1):
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                attempts += 1
                if row.get("keep"):
                    written[row["method"]] += 1
            handle.flush()
            print(f"[{done}/{len(jobs)}] {rows[-1]['task'][:40]} demo{rows[-1]['demo']}"
                  f" — {rows[-1].get('method', 'error')} {'KEEP' if rows[-1].get('keep') else 'drop'}"
                  f" / 書き出し {sum(written.values())} 本 / {attempts} 回"
                  f" / {(time.time() - started) / 60:.1f} 分", flush=True)
    print(f"\n=== 書き出し {sum(written.values())} 本 / {len(jobs)} 本 / 撃った回 {attempts} ===")
    print(f"  版ごと: {dict(written)}")
    print("END_OF_MERGED")


if __name__ == "__main__":
    main()
