#!/usr/bin/env python3
"""ロボット初期姿勢を振った手本を作る。**合格した回だけを npz で書き出す。**

**なぜ要るか。**学習データは手先のずれ 1.0 cm までしか覆っていないのに、Track2 の
採点では 2.4〜6.5 cm が出る。**その帯では成功率 0.833 → 0.535・衝突率 0.139 → 0.319** に
崩れる（`docs/plan/23_data_regeneration_design.md` の 2 節）。**帯の中の手本を足すには
新しい軌道が要る**ので、本家のデモを摂動環境で撃ち直し、**採点と同じ規則で合格したものを
学習データにする。**歩留まりは S0 で実測した——`open_loop` で 34.5%、
`prealign` を足すと 52.5%（同 10 節）。

**3 つの門をすべて通った回だけ書き出す。**

| 門 | 中身 |
|---|---|
| ゴール | 300 歩以内に bddl の goal が立つ |
| 接触 | **その時点で対象外の物体が 1 mm（3 軸の絶対値の和）を超えて動いていない** |
| **被覆** | **方策が引き継ぐ時点の手先のずれが `--min-offset-cm` 以上** |

**3 番目が要る理由。**摂動しても腕が元の経路に戻ってしまえば、作っても覆う範囲は
増えない。**「作ったのに被覆が増えていない」という静かな失敗を止める門である**
（同 5 節の門 6）。

**画像は溜めたうえで、合格した回だけ書き出す。**描画は 1 歩の 6% しか足さないので
（同 3.1 節）、先に選別してから描き直すより安い。**ただし npz は 1 本 50〜100 MB
あるので、`--target` は空きディスクから決めること。**

用法（コンテナの中で打つ。`run_in_container.sh` が包む）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/generate_perturbed.py \\
        --states work/artifacts/libero_states \\
        --out-npz work/regen/npz_batch1 --target 1000 --workers 14

Reads:
  - <states>/index.json … 課題ごとの bddl 名と指示文
  - <states>/<suite>/<task>_demo.npz … デモごとの `states`・`actions`・`ee_pos`
Generates:
  - <out-npz>/<課題>__demo<番号>__init<番号>__<撃ち方>.npz … `npz_to_lerobot build` が読む形
  - <out-npz>/manifest.jsonl … 1 ロールアウト 1 行（落ちた回も残す）
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))

from replay_yield import (MIN_OFFSET_CM, _replay_one, _variant,  # noqa: E402
                          passes_gates)

DEAD_TASKS = (
    # 手本が 146 本すべて 341 歩以上で、300 歩の打ち切りに入らない。
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    # 掴む相手が対象外のラメキンに載っている。**採点の規則では解けない。**
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
)
"""S0 で 40 本中 0 本だった課題。**撃っても 1 本も出ないので最初から外す。**"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out-npz", type=Path, required=True, metavar="DIR")
    parser.add_argument("--target", type=int, default=1000, metavar="N",
                        help="書き出す本数の目標。**達したら止める**")
    parser.add_argument("--per-task-cap", type=int, default=0, metavar="N",
                        help="1 課題あたりの上限。既定は target // 課題数 × 2")
    parser.add_argument("--method", choices=("open_loop", "prealign"),
                        default="open_loop",
                        help="**既定の `open_loop` はずれを最後まで抱える**ので被覆になる。"
                             "`prealign` は歩留まりの穴埋め用")
    parser.add_argument("--prealign-planner", choices=("proportional", "glide"),
                        default="proportional",
                        help="`prealign` の寄せ方。**`glide` は Ruckig**（比べた結果は "
                             "`prealign_ruckig_probe.py`）。`open_loop` では効かない")
    parser.add_argument("--initstates", type=int, nargs="+", metavar="N",
                        default=[282, 232, 389, 285, 242, 129,
                                 273, 392, 243, 432, 483, 372],
                        help="使う初期姿勢の番号。**既定は `screen_initstates.py` が "
                             "測った手先のずれで層化して選んだ 12 個**"
                             "（2.2/2.5/2.9/3.2/3.6/3.9/4.3/4.9/5.6/6.4/7.5/9.7 cm）。"
                             "**番号を倍率で選んではいけない**——`initstate_170` は "
                             "倍率 0.2 だが 0.88 cm しか動かず、被覆の門で全部落ちる")
    parser.add_argument("--min-offset-cm", type=float, default=MIN_OFFSET_CM, metavar="CM",
                        help="引き継ぎ時の手先のずれの下限。**これを割る回は捨てる**")
    parser.add_argument("--demos-per-task", type=int, default=0, metavar="N",
                        help="1 課題から使うデモの本数。0 で全部（50 本）")
    parser.add_argument("--skip", type=Path, default=None, metavar="FILE",
                        help="既に撃った manifest。**同じ組を撃ち直さない**")
    parser.add_argument("--workers", type=int, default=14, metavar="N")
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="`env.seed()` と、組を並べ替える種")
    return parser.parse_args()


def _plan(args: argparse.Namespace) -> tuple[list[dict], dict[str, dict]]:
    """(課題, デモ, 摂動) の組を並べる。**課題ごとにまとまるように並べる。**

    env の作り直しは 1.1 秒かかるので、**同じ (課題, 摂動) を続けて撃つ**。
    そのうえで課題の出る順は種で混ぜ、**目標に達して止めたときに
    特定の課題だけが入ることを避ける。**
    """
    index = json.loads((args.states / "index.json").read_text())
    tasks: dict[str, dict] = {}
    for repo_path, summary in sorted(index.items()):
        suite, filename = repo_path.split("/")
        stem = filename.replace("_demo.hdf5", "")
        if stem in DEAD_TASKS:
            continue
        npz = args.states / suite / filename.replace(".hdf5", ".npz")
        if npz.exists():
            # **鍵を `demos` にしてはいけない。**下で `**info` を展開するときに
            # 「撃つデモの一覧」を本数の int で上書きしてしまう。
            tasks[stem] = {"suite": suite, "npz": str(npz),
                           "demo_count": int(summary["demos"]),
                           "instruction": summary["language_instruction"]}

    rng = random.Random(args.seed)
    order = sorted(tasks)
    rng.shuffle(order)

    items: list[dict] = []
    for round_index, initstate in enumerate(args.initstates):
        for task in order:
            info = tasks[task]
            count = info["demo_count"]
            demos = list(range(count))
            if args.demos_per_task and args.demos_per_task < count:
                picked = np.linspace(0, count - 1, args.demos_per_task)
                demos = sorted({int(value) for value in picked.round()})
            rng.shuffle(demos)
            items.append({"task": task, "initstate": initstate, "demos": demos,
                          "round": round_index, **info})
    # **単位そのものを混ぜる。**目標に達して止めたときに、摂動の段が
    # 早い順の数段に偏らないようにするため（段を外側に回すと、止めた時点で
    # 大きいずれの段が 1 本も入らない）。env の作り直しは単位ごとなので、
    # 混ぜても費用は変わらない。
    rng.shuffle(items)
    return items, tasks


def _run_item(item: dict, args: argparse.Namespace, budget: int) -> list[dict]:
    """(課題, 摂動) 1 単位を撃ち、合格した回を npz へ書く。

    `budget` はこの単位で書き出してよい本数。**0 なら撃たない。**
    """
    if budget <= 0:
        return []
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    bddl = sd._bddl_path(libero_root, item["suite"],
                         _variant(item["task"], item["initstate"]))
    rows: list[dict] = []
    written = 0
    # **画像を溜めるので env は描画つきで作る。**
    env = sd._make_env(bddl, with_images=True)
    try:
        with np.load(item["npz"], allow_pickle=False) as bundle:
            for demo in item["demos"]:
                if written >= budget:
                    break
                actions = bundle[f"demo_{demo}/actions"]
                states = bundle[f"demo_{demo}/states"]
                key = f"demo_{demo}/ee_pos"
                ee_pos0 = bundle[key][0] if key in bundle.files else None
                started = time.perf_counter()
                result, rollout = _replay_one(
                    sd, env, states[0], actions, args.seed, args.method,
                    ee_pos0, record_images=True, planner=args.prealign_planner)

                gates = passes_gates(result, args.min_offset_cm)
                keep = gates["keep"]
                row = {"task": item["task"], "suite": item["suite"],
                       "demo": demo, "initstate": item["initstate"],
                       "method": args.method, **gates,
                       "seconds": round(time.perf_counter() - started, 2),
                       **{k: v for k, v in result.items() if k != "eef0"}}
                if keep:
                    rollout.attach_images()
                    record = rollout.record
                    destination = (args.out_npz /
                                   f"{item['task']}__demo{demo:03d}"
                                   f"__init{item['initstate']:03d}"
                                   f"__{args.method}.npz")
                    # **圧縮しない。** 動画へ入れ直すだけの中継物である
                    # （`scripted_demo.command_record` と同じ判断）。
                    np.savez(
                        destination,
                        state=np.stack(record.states),
                        action=np.stack(record.actions),
                        front=record.front,
                        wrist=record.wrist,
                        success=True,
                        steps=record.steps,
                        instruction=env.language_instruction,
                        # **素性を残す。** 後から「どの摂動段を何本入れたか」を
                        # 数え直せないと、混合比を測れない。
                        source_task=item["task"],
                        source_demo=demo,
                        initstate=item["initstate"],
                        method=args.method,
                        planner=args.prealign_planner,
                        prealign_steps=result["prealign_steps"],
                        # **1 コマごとの対象外の変位 [m]。**動画で「いつ当たったか」を
                        # 焼くのに要る。後から足すには撃ち直すしかない。
                        displacement_m=np.asarray(record.displacements, dtype=np.float32),
                        eef_offset_cm=result["eef_offset_cm"],
                        disp_at_goal_mm=result["disp_at_goal_mm"],
                        goal_step=result["goal_step"],
                    )
                    row["npz"] = destination.name
                    written += 1
                rows.append(row)
    finally:
        env.close()
    return rows


def _worker(queue: "mp.Queue", results: "mp.Queue", args: argparse.Namespace,
            budget: int) -> None:
    while True:
        item = queue.get()
        if item is None:
            break
        try:
            results.put(_run_item(item, args, budget))
        except Exception as error:  # 1 単位の失敗で全体を止めない
            results.put([{"task": item["task"], "initstate": item["initstate"],
                          "keep": False, "error": repr(error)}])


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args.out_npz.mkdir(parents=True, exist_ok=True)

    items, tasks = _plan(args)
    cap = args.per_task_cap or max(1, 2 * args.target // max(1, len(tasks)))
    # 1 単位に許す本数。**摂動段の数だけ回るので、段ごとに上限を割る。**
    per_item = max(1, cap // max(1, len(args.initstates)))

    done: set[tuple] = set()
    if args.skip and args.skip.exists():
        for line in args.skip.open():
            row = json.loads(line)
            done.add((row.get("task"), row.get("demo"), row.get("initstate"),
                      row.get("method")))
        print(f"すでに撃った組 {len(done)} 件は飛ばす")

    print(f"=== 課題 {len(tasks)} / 摂動 {len(args.initstates)} 段 / "
          f"撃ち方 {args.method} / 目標 {args.target} 本 ===")
    print(f"  1 課題の上限 {cap} 本（1 単位あたり {per_item} 本）・"
          f"手先のずれ {args.min_offset_cm} cm 以上・並列 {args.workers}")

    queue: "mp.Queue" = mp.Queue()
    results: "mp.Queue" = mp.Queue()
    for item in items:
        item["demos"] = [d for d in item["demos"]
                         if (item["task"], d, item["initstate"],
                             args.method) not in done]
        queue.put(item)
    for _ in range(args.workers):
        queue.put(None)
    workers = [mp.Process(target=_worker, args=(queue, results, args, per_item))
               for _ in range(args.workers)]
    for process in workers:
        process.start()

    manifest = args.out_npz / "manifest.jsonl"
    written, attempts = 0, 0
    per_task: dict[str, int] = {}
    started = time.time()
    with manifest.open("a") as handle:
        for done_count in range(1, len(items) + 1):
            batch = results.get()
            for row in batch:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                attempts += 1
                if row.get("keep"):
                    written += 1
                    per_task[row["task"]] = per_task.get(row["task"], 0) + 1
            handle.flush()
            elapsed = time.time() - started
            print(f"[{done_count}/{len(items)}] {batch[0].get('task', '?')[:44]} "
                  f"init={batch[0].get('initstate')} — "
                  f"**{written} 本** / {attempts} 回 / {elapsed / 60:.1f} 分")
            if written >= args.target:
                print(f"\n目標 {args.target} 本に達した。止める。")
                break

    # **子を先に落としてから待つ。** queue に残った仕事は捨てる。
    for process in workers:
        process.terminate()
    for process in workers:
        process.join(timeout=10)

    print(f"\n=== 書き出し {written} 本 / 撃った回 {attempts} "
          f"（歩留まり {100 * written / max(1, attempts):.1f}%） ===")
    print(f"  {args.out_npz}")
    if per_task:
        counts = sorted(per_task.items(), key=lambda item: item[1])
        print(f"  課題ごと: 少ない順 {counts[:3]} / 多い順 {counts[-3:]}")
        print(f"  1 本も出なかった課題: "
              f"{sorted(set(tasks) - set(per_task))}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
