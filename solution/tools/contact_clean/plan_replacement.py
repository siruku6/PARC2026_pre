"""置き換えたあと、課題ごとに本数がどう変わるかを出す。

**4,000 本は「足す」のではなく「置き換えに回す」**（ユーザ・9/14）。
外す本（接触・300 コマ超）を課題ごとに数え、**生成した本をどう配ると
どの課題が痩せるか**を見る。

配り方は 2 通り並べる。
  flat   … 課題ごとに同じ本数（4,000 ÷ 課題数 ≒ 105）。033 に今書いてある案
  need   … **外す本数に比例して配る**（痩せる課題へ厚く）

用法（ホスト・シム不要）:
    python3 solution/tools/contact_clean/plan_replacement.py \
        --map work/regen/contact_v2/demo_map.json \
        --scores work/regen/contact_v2/contact_scores.json --budget 4000 \
        --done pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate ...

Reads:
  - <map> … `map_demos.py` が書いた対応表（`frames` から 300 コマ超も数える）
  - <scores> … `score_contact.py` が書いた判定表
Generates:
  - 標準出力 … 課題ごとの「今／外す／残る／配る／置き換え後」と、配り方 2 通りの要約
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

BASELINE = 10
"""接触を測る基準コマ。**026 の 2 節で決めた**（0 だと置いた物の落下を拾う）。"""

THRESHOLD_MM = 1.0
"""対象外の物体が動いてよい量。採点と同じ。"""

MAX_FRAMES = 300
"""採点の打ち切り。**これを超える手本は 1 本も成功できない。**"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, required=True, metavar="JSON")
    parser.add_argument("--scores", type=Path, required=True, metavar="JSON")
    parser.add_argument("--budget", type=int, default=4000, metavar="N",
                        help="生成する本数（置き換えも新規もこの中から出す）")
    parser.add_argument("--done", nargs="*", default=None, metavar="TASK",
                        help="**もう置き換え済みの課題**（土台が `libero_plus_bowl5` なら丼 5 課題）。"
                             "外す側から抜き、配る側にも入れない")
    parser.add_argument("--share-done", action="store_true",
                        help="**余りは `--done` の課題にも配る**（外す側からは抜いたまま）。"
                             "丼 5 課題は台本が通るので、余りを受け取ってよい（ユーザ・9/14）")
    parser.add_argument("--scripted", nargs="*", default=None, metavar="TASK",
                        help="**人の手本を撃ち直さず、台本で作る課題。**`--quota-out` に書くだけで、"
                             "本数の計算は変えない（`generate_perturbed.py` はこれを飛ばす）")
    parser.add_argument("--quota-out", type=Path, default=None, metavar="JSON",
                        help="**need の配り方を課題ごとの本数として書く**"
                             "（`generate_perturbed.py --quota` が読む）")
    return parser.parse_args()


def _worst_mm(score: dict) -> float:
    """基準コマから、対象外の物体がいちばん動いた量 [mm]。"""
    return 1000.0 * max(score["displacement_l1_m"][str(BASELINE)].values(), default=0.0)


def main() -> None:
    args = _parse_args()
    groups = json.loads(args.map.read_text())
    scores = {row["fingerprint"]: row for row in json.loads(args.scores.read_text())}

    total: dict[str, int] = defaultdict(int)
    drop: dict[str, int] = defaultdict(int)
    why: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [接触, 300 超]
    missing = 0
    for group in groups:
        task = group["source_file"].split("/")[1].replace("_demo.hdf5", "")
        count = len(group["episodes"])
        total[task] += count
        score = scores.get(group["fingerprint"])
        if score is None:
            missing += 1
            continue
        touched = _worst_mm(score["score"]) > THRESHOLD_MM
        too_long = group["frames"] > MAX_FRAMES
        if touched:
            why[task][0] += count
        if too_long:
            why[task][1] += count
        if touched or too_long:
            drop[task] += count

    if missing:
        print(f"**判定が引けない組が {missing} 個ある**（対応表と判定表が食い違っている）")

    # **もう直した課題は触らない。**配布データでは接触していても、土台では
    # 台本の手本に入れ替わっている（`libero_plus_bowl5`）。
    done = set(args.done or ())
    for task in done:
        drop[task] = 0
    tasks = sorted(total)
    share_tasks = [task for task in tasks if args.share_done or task not in done]
    dropped_total = sum(drop.values())
    print(f"課題 {len(tasks)}・全 {sum(total.values())} 本")
    print(f"**外す本 {dropped_total} 本**"
          f"（{100 * dropped_total / sum(total.values()):.1f}%。"
          f"接触 {sum(v[0] for v in why.values())}・300 コマ超 {sum(v[1] for v in why.values())}）")

    # 配り方 2 通り。**flat は課題ごとに同じ本数、need は外す本数に比例。**
    flat = {task: (args.budget // len(share_tasks) if task in share_tasks else 0)
            for task in tasks}
    # **痩せる課題を作らないように配る**——まず「外した数だけ埋め戻す」に必要な分を置き、
    # 余りは残りの課題へ均等に。
    need = {task: drop[task] for task in tasks}
    spare = args.budget - sum(need.values())
    if spare > 0:
        for task in share_tasks:
            need[task] += spare // len(share_tasks)

    print(f"\n{'課題':<52} {'今':>5} {'外す':>5} {'残':>5} "
          f"{'flat':>5} {'→計':>6} {'need':>5} {'→計':>6}")
    for task in sorted(tasks, key=lambda t: drop[t] - flat[t], reverse=True):
        rest = total[task] - drop[task]
        print(f"{task[:50]:<52} {total[task]:>5} {drop[task]:>5} {rest:>5} "
              f"{flat[task]:>5} {rest + flat[task]:>6} "
              f"{need[task]:>5} {rest + need[task]:>6}")

    if args.quota_out:
        # **need を採る**（ユーザ・9/14）。まず外した数だけ埋め戻し、余りを均等に配る。
        args.quota_out.write_text(json.dumps(
            {"quota": {task: need[task] for task in tasks if need[task] > 0},
             "dropped": {task: drop[task] for task in tasks if drop[task] > 0},
             "budget": args.budget, "done": sorted(done),
             "scripted": sorted(args.scripted or ())}, indent=1))
        print(f"\n=== 課題ごとの本数を書いた: {args.quota_out}"
              f"（{sum(need.values())} 本・{sum(v > 0 for v in need.values())} 課題）===")

    for label, share in (("flat", flat), ("need", need)):
        shrink = [t for t in tasks if total[t] - drop[t] + share[t] < total[t]]
        empty = [t for t in tasks if total[t] - drop[t] + share[t] == 0]
        after = sum(total[t] - drop[t] + share[t] for t in tasks)
        print(f"\n**{label}**: 置き換え後 {after} 本"
              f"（今より {after - sum(total.values()):+d}）  "
              f"**減る課題 {len(shrink)}**・消える課題 {len(empty)}")
        for task in sorted(shrink, key=lambda t: total[t] - drop[t] + share[t] - total[t])[:6]:
            print(f"    {task[:50]:<52} {total[task]:>5} → "
                  f"{total[task] - drop[task] + share[task]:>5}")


if __name__ == "__main__":
    main()
