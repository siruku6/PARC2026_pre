#!/usr/bin/env python3
"""本家のデモを摂動環境で撃ち直し、**学習データとして使える本数**を数える。

**なぜ要るか。**Track2 の衝突は「汚い手本を真似た」からではなく、**手本が 1 本も
無い範囲（手先のずれ 2〜6 cm）で崩れている**ためだと見ている
（`docs/plan/23_data_regeneration_design.md` の 2 節）。**その範囲の手本を足すには
新しい軌道が要る**——腕が別の場所から始まるので、記録した行動列をそのまま
再生できるとは限らない。**この道具は「そのまま撃ったら何本使えるか」を数える。**

**判定は採点側と同じ規則である**——ゴール到達 かつ **対象外の物体が 1 mm
（3 軸の絶対値の和）を超えて動いていない**。判定そのものは
`scripted_demo._Rollout` を借りているので、採点ハーネスと 1 か所で揃う。

**撃ち方は 2 通り測る。**

| 方法 | 中身 |
|---|---|
| `open_loop` | 記録された行動をそのまま流す。**新しい制御は 1 行も入らない** |
| `prealign` | **先に腕を、デモの 0 コマ目の手先へ寄せてから**流す |

**`prealign` が要る理由は、摂動の効き方にある。**採点ハーネスは
`reset` → `set_state_from_flattened` → **ゼロ行動 10 歩**の順に進めるが、
`.pruned_init` の状態には腕の関節角も入っているので、**腕はいったんデモと同じ姿勢に戻る。**
摂動が効くのはそのあとの 10 歩で、**OSC が握ったままの目標（摂動した姿勢）へ腕が流れる。**
つまり方策が引き継ぐ時点で腕は 2〜7 cm ずれている。**`prealign` はそのずれを
「戻す区間」として先頭に足す**——足した区間そのものが、学習データに 0.2% しか無い
「ずれた所から戻る」手本になる。

用法（コンテナの中で打つ。`run_in_container.sh` が包む）:
    python solution/tools/regen/replay_yield.py \\
        --states work/artifacts/libero_states \\
        --out work/regen/replay_yield.jsonl \\
        --demos-per-task 10 --initstates 0 150 250 350 450 --workers 14

Reads:
  - <states>/index.json … 課題ごとの bddl 名と指示文
  - <states>/<suite>/<task>_demo.npz … デモごとの `states` と `actions`
Generates:
  - <out> … 1 ロールアウト 1 行の JSONL
  - 標準出力 … 摂動ごと・課題ごとの歩留まりの表
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

# `scripted_demo` は環境の作り方と採点と同じ衝突判定を持っている。**写さずに借りる。**
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))

PREALIGN_STEPS = 40
"""`prealign` で、デモの 0 コマ目の手先へ寄せるのに許す歩数の上限。"""

PREALIGN_TOLERANCE = 0.006
"""`prealign` を打ち切る距離 [m]。"""

MAX_EXTRA_STEPS = 60
"""記録された行動を撃ち切ったあと、ゴール判定のために足す歩数。

**摂動下では同じ行動列でも到着が遅れる。**撃ち切った時点で届いていなくても、
数十歩そのまま止めて待つと物が落ち着いてゴールが立つことがある。
"""

MAX_STEPS = 300
"""採点の打ち切り。**寄せる区間の歩数もこの中に数える**（学習データに入るのは寄せた区間ごとである）。"""

MIN_OFFSET_CM = 2.0
"""被覆の門。引き継ぎ時の手先のずれがこれを割る回は、腕が元の経路に戻っているので捨てる。"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR",
                        help="`fetch_libero_states.py` が書いた置き場")
    parser.add_argument("--out", type=Path, required=True, metavar="FILE",
                        help="1 ロールアウト 1 行の JSONL の書き出し先")
    parser.add_argument("--demos-per-task", type=int, default=10, metavar="N",
                        help="1 課題から何本のデモを撃つか（等間隔に抜く）。0 で全部")
    parser.add_argument("--initstates", type=int, nargs="+", metavar="N",
                        default=[0, 150, 250, 350, 450],
                        help="ロボット初期姿勢の番号。**0 は摂動なし**")
    parser.add_argument("--tasks", nargs="*", default=None, metavar="NAME",
                        help="課題を絞る（既定は全部）。bddl の stem で指定する")
    parser.add_argument("--workers", type=int, default=14, metavar="N",
                        help="並列数。**物理は 16 までほぼ線形に伸びる**")
    parser.add_argument("--method", choices=("open_loop", "prealign"),
                        default="open_loop",
                        help="撃ち方。`prealign` は先に腕をデモの 0 コマ目へ寄せる")
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="`env.seed()` に渡す値。**reset のたびに引き直される "
                             "fixture の位置を、実行をまたいで揃えるために要る**")
    return parser.parse_args()


def _work_items(states_dir: Path, demos_per_task: int, initstates: list[int],
                tasks: list[str] | None) -> list[dict]:
    """(課題, 摂動) の単位で仕事を並べる。**env は 1 単位につき 1 回だけ作る。**"""
    index = json.loads((states_dir / "index.json").read_text())
    items: list[dict] = []
    for repo_path, summary in sorted(index.items()):
        suite, filename = repo_path.split("/")
        stem = filename.replace("_demo.hdf5", "")
        if tasks and stem not in tasks:
            continue
        npz = states_dir / suite / filename.replace(".hdf5", ".npz")
        if not npz.exists():
            continue
        count = int(summary["demos"])
        if demos_per_task and demos_per_task < count:
            picked = np.linspace(0, count - 1, demos_per_task).round().astype(int)
            demos = sorted({int(value) for value in picked})
        else:
            demos = list(range(count))
        for initstate in initstates:
            items.append({"suite": suite, "task": stem, "npz": str(npz),
                          "demos": demos, "initstate": initstate,
                          "instruction": summary["language_instruction"]})
    return items


def _variant(task: str, initstate: int) -> str:
    """摂動を当てた bddl の名前。**0 なら素の課題名のままである。**"""
    if initstate == 0:
        return task
    return f"{task}_view_0_0_100_0_0_initstate_{initstate}"


def _prealign(sd, rollout, target: np.ndarray, gripper: float,
              planner: str = "proportional") -> int:
    """デモの 0 コマ目の手先へ腕を寄せる。

    | `planner` | 寄せ方 |
    |---|---|
    | `proportional` | **比例制御。**毎歩「目標 − いまの姿勢」を `_velocity_command` で出す |
    | `glide` | **Ruckig。**躍度の上限を守った通過点を先に作って追う（`_Rollout.glide`）。**封筒つきの `_Rollout` が要る** |

    `_velocity_command` を借りるのは、**行動ラベルの大きさを配布データに寄せる**
    ためである。誤差に比例させただけの指令は小さすぎて這い、
    `action` 列の標準偏差が 3 分の 1 に痩せる（`docs/plan/23` の 4.2 節）。

    **`glide` の区間は 2 つに割れる。**Ruckig の計画を配る前半と、配り終えたあとに
    残りを詰める後半（比例制御）である（`_Rollout.glide`）。**指令の作り方が違うので、
    歪みを見るときは分けて数える**——歩数は `rollout.glide_plan_steps` に入る。

    Returns
    -------
    steps : int
        寄せるのに使った歩数。
    """
    used = rollout.record.steps
    rollout.glide_plan_steps = 0
    if planner == "glide":
        # **出口の速さは 0 にする。**寄せた先はデモの 0 コマ目で、人が静止した状態から
        # 始めている。速度を残して渡すと、記録された行動が想定していない初速が乗る。
        rollout.glide("prealign", [np.asarray(target, dtype=np.float64)], gripper,
                      exit_ratio=0.0, tolerance=PREALIGN_TOLERANCE,
                      budget=PREALIGN_STEPS)
        return rollout.record.steps - used
    for _ in range(PREALIGN_STEPS):
        delta = target - rollout.eef_position
        if float(np.linalg.norm(delta)) < PREALIGN_TOLERANCE:
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = rollout._velocity_command(delta, sd.BRAKE_WITHIN_STOP)
        action[6] = gripper
        if not rollout._take_step(action):
            break
    return rollout.record.steps - used


def _replay_one(sd, env, init_state: np.ndarray, actions: np.ndarray,
                seed: int, method: str, ee_pos0: np.ndarray | None,
                record_images: bool = False,
                planner: str = "proportional") -> tuple[dict, object]:
    """記録された行動を 1 本撃ち、採点と同じ規則で成否を付ける。

    **変位は 2 つ出す。**採点は**ゴールが立った時点で打ち切る**ので、
    採否に効くのは `disp_at_goal_mm` のほうである。`disp_full_mm` は撃ち切るまでの
    最大で、**学習データとして全コマ入れるなら**こちらを見る。

    Parameters
    ----------
    record_images : bool
        観測画像も溜めるか。**描画は 1 歩の 6% しか足さない**ので、
        生成では溜めたうえで**合格した回だけ書き出す**のがいちばん安い。
    planner : str
        `method="prealign"` のときの寄せ方（`_prealign`）。`open_loop` では効かない。

    Returns
    -------
    result : dict
        `goal_step`（届かなければ -1）・`disp_at_goal_mm`・`disp_full_mm`・`steps`・
        `prealign_steps`・`prealign_plan_steps`・`prealign_residual_mm`・
        `eef_offset_cm`・`eef0`。
    rollout : scripted_demo._Rollout
        書き出しに使う。画像は `attach_images()` のあと `record.front` にある。
    """
    env.seed(seed)
    budget = int(actions.shape[0]) + MAX_EXTRA_STEPS
    aligning = method == "prealign" and ee_pos0 is not None
    if method == "prealign":
        budget += PREALIGN_STEPS
    envelope = None
    if aligning and planner == "glide":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from motion.envelope import envelope_from_environment

        envelope = envelope_from_environment()
    rollout = sd._Rollout(env, init_state, budget, record_images=record_images,
                          envelope=envelope, glide=envelope is not None)
    eef0 = rollout.eef_position.copy()
    prealign_steps, residual_mm = 0, float("nan")
    if aligning:
        target = np.asarray(ee_pos0, dtype=np.float64)
        prealign_steps = _prealign(sd, rollout, target, float(actions[0][6]), planner)
        residual_mm = 1000.0 * float(np.linalg.norm(target - rollout.eef_position))
        # **封筒は寄せる区間にだけ当てる。**記録された行動に当てると 7.79% のコマが
        # 削られ、撃ち直す手本そのものが変わる。
        rollout._envelope = None
    goal_step = -1
    disp_at_goal = float("nan")
    for action in actions:
        if not rollout._take_step(np.asarray(action, dtype=np.float32)):
            break
        if goal_step < 0 and bool(env.check_success()):
            goal_step = rollout.record.steps
            disp_at_goal = float(rollout.record.max_displacement)
    # 撃ち切っても届いていなければ、止めたまま少し待つ。
    while goal_step < 0 and rollout.record.steps < budget:
        if not rollout._take_step(np.zeros(7, dtype=np.float32)):
            break
        if bool(env.check_success()):
            goal_step = rollout.record.steps
            disp_at_goal = float(rollout.record.max_displacement)
    offset = (float("nan") if ee_pos0 is None
              else float(np.linalg.norm(np.asarray(ee_pos0) - eef0)))
    return {"goal_step": goal_step,
            "disp_at_goal_mm": (None if goal_step < 0
                                else round(1000.0 * disp_at_goal, 3)),
            "disp_full_mm": round(1000.0 * float(rollout.record.max_displacement), 3),
            "steps": int(rollout.record.steps),
            "prealign_steps": prealign_steps,
            # **寄せる区間のうち、Ruckig の計画を配ったぶん。**残りは比例制御の詰めで、
            # 比例制御だけで寄せた回（`planner="proportional"`）では 0 である。
            "prealign_plan_steps": int(rollout.glide_plan_steps),
            "prealign_residual_mm": round(residual_mm, 3),
            # **方策が引き継ぐ時点で腕がどれだけずれているか [cm]。**
            # ここが 2 cm を割ると、作っても被覆は増えない（`docs/plan/23` の 5 節の門 6）。
            "eef_offset_cm": round(100.0 * offset, 3),
            "eef0": [round(float(value), 5) for value in eef0]}, rollout


def passes_gates(result: dict, min_offset_cm: float = MIN_OFFSET_CM) -> dict[str, bool]:
    """生成の 3 つの門を判定する。**採否の規則はここ 1 か所だけに置く。**

    | 門 | 中身 |
    |---|---|
    | `reached` | 採点の打ち切り（`MAX_STEPS`）以内にゴールが立った |
    | `clean` | ゴール時点で対象外の物体が 1 mm を超えて動いていない |
    | `covered` | 引き継ぎ時の手先のずれが `min_offset_cm` 以上 |
    """
    reached = 0 < result["goal_step"] <= MAX_STEPS
    clean = (result["disp_at_goal_mm"] or 0.0) <= 1.0
    covered = result["eef_offset_cm"] >= min_offset_cm
    return {"reached": bool(reached), "clean": bool(clean), "covered": bool(covered),
            "keep": bool(reached and clean and covered)}


def suite_of(states_dir: Path) -> dict[str, str]:
    """課題名 → suite。`index.json` の鍵（`<suite>/<課題>_demo.hdf5`）から引く。"""
    index = json.loads((states_dir / "index.json").read_text())
    return {key.split("/")[1].replace("_demo.hdf5", ""): key.split("/")[0]
            for key in index}

def _run_item(item: dict, seed: int, method: str) -> list[dict]:
    """(課題, 摂動) 1 単位ぶんを撃つ。**env はここで 1 回だけ作る。**"""
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    bddl = sd._bddl_path(libero_root, item["suite"], _variant(item["task"],
                                                             item["initstate"]))
    rows: list[dict] = []
    with np.load(item["npz"], allow_pickle=False) as bundle:
        env = sd._make_env(bddl, with_images=False)
        try:
            for demo in item["demos"]:
                states = bundle[f"demo_{demo}/states"]
                actions = bundle[f"demo_{demo}/actions"]
                key = f"demo_{demo}/ee_pos"
                ee_pos0 = bundle[key][0] if key in bundle.files else None
                started = time.perf_counter()
                result, _ = _replay_one(sd, env, states[0], actions, seed,
                                        method, ee_pos0)
                rows.append({"suite": item["suite"], "task": item["task"],
                             "instruction": item["instruction"],
                             "demo": demo, "initstate": item["initstate"],
                             "demo_steps": int(actions.shape[0]),
                             "seconds": round(time.perf_counter() - started, 2),
                             **result})
        finally:
            env.close()
    return rows


def _worker(queue: "mp.Queue", results: "mp.Queue", seed: int,
            method: str) -> None:
    while True:
        item = queue.get()
        if item is None:
            break
        try:
            results.put(_run_item(item, seed, method))
        except Exception as error:  # 1 課題の失敗で全体を止めない
            results.put([{"suite": item["suite"], "task": item["task"],
                          "initstate": item["initstate"], "error": repr(error)}])


def _summarise(rows: list[dict], initstates: list[int]) -> None:
    """摂動ごとの歩留まりと、課題ごとの内訳を印字する。"""
    def reached(row: dict) -> bool:
        """採点と同じ——**300 歩以内にゴールが立ったか。**"""
        return 0 < row.get("goal_step", -1) <= 300

    def usable(row: dict) -> bool:
        """**採点と同じ規則で成功と数えられるか。**変位はゴール時点のものを見る。"""
        return reached(row) and (row.get("disp_at_goal_mm") or 0.0) <= 1.0

    print()
    header = (f"{'initstate':>10} {'本数':>6} {'300 歩以内':>11} {'1mm 未満':>10} "
              f"{'使える':>8} {'割合':>7} {'変位の中央':>12} {'手先ずれ':>10}")
    print(header)
    print("-" * 84)
    for initstate in initstates:
        group = [row for row in rows if row.get("initstate") == initstate
                 and "error" not in row]
        if not group:
            continue
        goals = sum(1 for row in group if reached(row))
        cleans = sum(1 for row in group
                     if reached(row) and (row.get("disp_at_goal_mm") or 0.0) <= 1.0)
        good = sum(1 for row in group if usable(row))
        median = float(np.median([row["disp_full_mm"] for row in group]))
        offsets = [row["eef_offset_cm"] for row in group
                   if row.get("eef_offset_cm") == row.get("eef_offset_cm")]
        offset = float(np.median(offsets)) if offsets else float("nan")
        print(f"{initstate:>10} {len(group):>6} {goals:>11} {cleans:>10} "
              f"{good:>8} {100 * good / len(group):>6.1f}% {median:>11.2f}mm "
              f"{offset:>8.2f}cm")

    errors = [row for row in rows if "error" in row]
    if errors:
        print(f"\n落ちた単位 {len(errors)} 件（先頭 3 件）")
        for row in errors[:3]:
            print(f"  {row['task']} initstate={row['initstate']}: {row['error'][:160]}")


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    items = _work_items(args.states, args.demos_per_task, args.initstates, args.tasks)
    total = sum(len(item["demos"]) for item in items)
    print(f"=== {len(items)} 単位 / {total} ロールアウト / "
          f"撃ち方 {args.method} / 並列 {args.workers} ===")

    queue: "mp.Queue" = mp.Queue()
    results: "mp.Queue" = mp.Queue()
    for item in items:
        queue.put(item)
    for _ in range(args.workers):
        queue.put(None)
    workers = [mp.Process(target=_worker,
                          args=(queue, results, args.seed, args.method))
               for _ in range(args.workers)]
    for process in workers:
        process.start()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.time()
    with args.out.open("w") as handle:
        for done in range(1, len(items) + 1):
            batch = results.get()
            rows.extend(batch)
            for row in batch:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.time() - started
            print(f"[{done}/{len(items)}] {batch[0].get('task', '?')} "
                  f"initstate={batch[0].get('initstate')} — "
                  f"{len(rows)} 本 / {elapsed / 60:.1f} 分 "
                  f"(残り {elapsed / done * (len(items) - done) / 60:.1f} 分)")
    for process in workers:
        process.join()

    _summarise(rows, args.initstates)
    print(f"\n=== 生の結果: {args.out} ===")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
