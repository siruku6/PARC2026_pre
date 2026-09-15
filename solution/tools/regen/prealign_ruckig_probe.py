#!/usr/bin/env python3
"""**摂動した初期姿勢から元のデモへ繋ぐ区間**を、Ruckig 有り／無しで比べる。

**何を比べるのか。**摂動下再生には撃ち方が 2 通りある
（`docs/plan/23_data_regeneration_design.md` の 10.2 節）。そのうち `prealign` は
**先に腕をデモの 0 コマ目の手先へ寄せてから**記録された行動を流す。
**この「寄せる区間」が比較の対象である。**

| 版 | 寄せ方 | 実体 |
|---|---|---|
| **無し** | **比例制御**——毎歩「目標 − いまの姿勢」を出す | `replay_yield._prealign(planner="proportional")` |
| **有り** | **通過点を先に作って追う**——躍度の上限を守った S 字 | `replay_yield._prealign(planner="glide")` |

**撃ち方も採否も、生成（`generate_perturbed.py`）と同じ関数を呼ぶ。**
ここで測った差は、そのまま `--prealign-planner` を切り替えたときの差である。

**見る数は 4 つである。**

| # | 見るもの | どちらが良いか |
|---|---|---|
| 1 | **歩留まり**——3 つの門（`replay_yield.passes_gates`）を通る割合 | 高いほう |
| 2 | **寄せるのに使った歩数**と、**寄せ終わった時の残り誤差** | 少なく・小さいほう |
| 3 | **指令の形**（`motion/report_smoothness.py` に渡す npz） | 配布データに近いほう |
| 4 | **被覆が保たれているか**——引き継ぎ時の手先のずれ | 両者同じはず（寄せる前に測るので） |

用法（コンテナの中で打つ。`run_in_container.sh` が包む）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/prealign_ruckig_probe.py \\
        --states work/artifacts/libero_states --out work/regen/prealign_ruckig

Reads:
  - <states>/index.json・<states>/<suite>/<課題>_demo.npz
Generates:
  - <out>/rows.jsonl … 1 行 1 ロールアウト（版・課題・デモ・摂動・門の通過）
  - <out>/npz_{with,without}/<課題>__demo…__{seg,full}.npz … `action` 列
    （`seg` は寄せる区間だけ・`full` は 1 本まるごと）
  - <out>/summary.md … 表にした比べ
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_yield import (MIN_OFFSET_CM, _replay_one, _variant,  # noqa: E402
                          _work_items, passes_gates)

PLANNERS = {"without": ("proportional", "Ruckig 無し"),
            "with": ("glide", "Ruckig 有り")}
"""ファイル名に使う札 → (`planner`, 表の見出し)。"""

KNOBS = ("MOTION_GLIDE_GAIN", "MOTION_GLIDE_LAG", "MOTION_GLIDE_EXIT_SPEED")
"""**Ruckig の摘み。**振ったときにどの値で撃ったのかが後から分かるよう、行に写す。"""

# **既定の 4 段は、手先のずれで層化した 12 個から両端と中を取ったものである。**
DEFAULT_INITSTATES = (282, 242, 432, 372)   # 2.17 / 3.58 / 6.45 / 9.67 cm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--initstates", type=int, nargs="+",
                        default=list(DEFAULT_INITSTATES), metavar="N")
    parser.add_argument("--demos-per-task", type=int, default=1, metavar="N",
                        help="1 つの (課題, 摂動) から何本撃つか（等間隔に抜く）")
    parser.add_argument("--tasks", nargs="*", default=None, metavar="NAME",
                        help="課題を絞る（既定は全部）。**下見に使う**")
    parser.add_argument("--min-offset-cm", type=float, default=MIN_OFFSET_CM,
                        metavar="CM")
    parser.add_argument("--planners", nargs="+", default=list(PLANNERS),
                        choices=list(PLANNERS), metavar="LABEL",
                        help="撃つ版。**摘みを振るときは `with` だけにする**"
                             "——`without` は摘みで 1 歩も変わらないので、"
                             "条件ごとに撃ち直すと時間が倍になる")
    parser.add_argument("--workers", type=int, default=14, metavar="N")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    return parser.parse_args()


def _run_item(job: tuple[dict, argparse.Namespace]) -> list[dict]:
    """(課題, 摂動) 1 単位を、両方の寄せ方で撃つ。**env は 1 回だけ作る。**"""
    item, args = job
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    env = sd._make_env(sd._bddl_path(libero_root, item["suite"],
                                     _variant(item["task"], item["initstate"])),
                       with_images=False)
    rows: list[dict] = []
    try:
        with np.load(item["npz"], allow_pickle=False) as bundle:
            for demo in item["demos"]:
                states = bundle[f"demo_{demo}/states"]
                actions = bundle[f"demo_{demo}/actions"]
                ee_pos0 = bundle[f"demo_{demo}/ee_pos"][0]
                stem = f"{item['task']}__demo{demo:03d}__init{item['initstate']:03d}"
                for label in args.planners:
                    planner = PLANNERS[label][0]
                    result, rollout = _replay_one(sd, env, states[0], actions,
                                                  args.seed, "prealign", ee_pos0,
                                                  planner=planner)
                    full = np.asarray(rollout.record.actions, dtype=np.float32)
                    for suffix, action in (("seg", full[:result["prealign_steps"]]),
                                           ("full", full)):
                        np.savez_compressed(
                            args.out / f"npz_{label}" / f"{stem}__{suffix}.npz",
                            action=action)
                    rows.append({"variant": label, "task": item["task"],
                                 "demo": demo, "initstate": item["initstate"],
                                 **passes_gates(result, args.min_offset_cm),
                                 **{k: v for k, v in result.items() if k != "eef0"},
                                 **{knob: os.environ.get(knob) for knob in KNOBS}})
    finally:
        env.close()
    return rows


def _table(rows: list[dict]) -> str:
    """版ごとの表を組む。"""
    lines = ["| 版 | 本数 | 歩留まり | 寄せ歩数 中央（計画） | 残り誤差 中央 [mm] "
             "| 届いた | 接触が 1 mm 未満 | 全歩数 中央 |",
             "|---|---|---|---|---|---|---|---|"]
    for label, (_, title) in PLANNERS.items():
        sub = [r for r in rows if r["variant"] == label]
        if not sub:
            continue
        lines.append(
            f"| {title} | {len(sub)} "
            f"| **{sum(r['keep'] for r in sub) / len(sub) * 100:.1f}%** "
            f"| {np.median([r['prealign_steps'] for r in sub]):.0f} "
            f"（{np.median([r['prealign_plan_steps'] for r in sub]):.0f}） "
            f"| {np.nanmedian([r['prealign_residual_mm'] for r in sub]):.1f} "
            f"| {sum(r['reached'] for r in sub) / len(sub) * 100:.1f}% "
            f"| {sum(r['reached'] and r['clean'] for r in sub) / len(sub) * 100:.1f}% "
            f"| {np.median([r['steps'] for r in sub]):.0f} |")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    items = _work_items(args.states, args.demos_per_task, args.initstates, args.tasks)
    for label in args.planners:
        (args.out / f"npz_{label}").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with (args.out / "rows.jsonl").open("w") as handle, mp.Pool(args.workers) as pool:
        for done, batch in enumerate(
                pool.imap_unordered(_run_item, [(item, args) for item in items]), 1):
            rows.extend(batch)
            for row in batch:
                handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            handle.flush()
            print(f"[{done}/{len(items)}] {batch[0]['task'][:48]} "
                  f"init{batch[0]['initstate']} — "
                  + " / ".join(f"{r['variant']} {'KEEP' if r['keep'] else 'DROP'}"
                               for r in batch), flush=True)

    (args.out / "summary.md").write_text(_table(rows) + "\n")
    print("\n" + _table(rows))
    print(f"\n=== {args.out} ===")


if __name__ == "__main__":
    main()
