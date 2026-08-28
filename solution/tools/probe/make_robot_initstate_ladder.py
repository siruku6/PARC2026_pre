#!/usr/bin/env python3
"""ロボット初期姿勢の摂動を、**運営が使う難易度の段**に沿って測る CSV を作る。

**なぜ難易度で並べるのか。** 8/28 に倍率（0.1〜0.5）で段を作って測ったところ
ほぼ 0 の差しか出ず、8/24 の「+47.8%・12 件中 6 件を破壊」と食い違った。
その後の実測で、**倍率は難しさの目安になっていない**ことが分かっている——
倍率 0.4 の手先のずれは 1.03〜9.84 cm で、倍率 0.1 の上端（2.08 cm）と重なる。

一方、`task_classification.json` には運営（LIBERO-plus）自身が付けた
`difficulty_level`（1〜5）があり、**同じ initstate 番号でもタスクによって
難易度が違う**（450 番号のうち 421 番号が複数の難易度にまたがる）。
つまりこの列は「番号の強さ」ではなく「そのタスクでその姿勢がどれだけ効くか」を
測ったものである。本選の配布 CSV（`compe/t2/T2_TASKS.csv`）も
この列で L3〜L5 を選んでいる。**測る段は難易度で作る。**

作る形は「タスク 12 件 × （摂動なし 1 行 ＋ L2・L3・L4・L5 の 4 行）＝ 60 行」である。

  - **摂動なしの行を同じ回に入れる。** 回し直しのばらつきが ±6.5% あるので、
    前の回の Base と比べると差が埋もれる。
  - **行の順はタスクごとにまとめる**（タスク 1 の 5 行 → タスク 2 の 5 行 …）。
    `summarize_probe_by_category.py` が 5 行区切りで Base と対にできる。
  - **initstate 番号は 60 行すべてで重複させない。** 8/28 は 1 段につき
    1 番号を 12 タスクへ当てたので、向きが 3 通りしか入っていなかった。
    番号が違えば摂動の向きが違うので、これで 48 通りの向きが入る。

variant の選び方:
  1. `TRACK2_VAL72.csv` の `category` が `Base` の 12 行を元タスクとする。
     これは 8/24 に摂動なしで 3 回とも 300 歩未満で完走したタスクである。
  2. `task_classification.json` から、その元タスクの
     `Robot Initial States` variant を難易度ごとに集める。
  3. 難易度ごとに 1 件ずつ選ぶ。**選び方は固定の種の擬似乱数**で、
     同じ引数なら同じ CSV になる。
  4. その難易度の variant を持たないタスクは、**代わりに一つ上の難易度から
     もう 1 件取る**（`--fallback-level`）。落とすと段の本数が減るため。

使い方:

    python3 solution/tools/probe/make_robot_initstate_ladder.py \
        --output compe/val/TRACK2_ROBOT_LADDER60.csv

    # 回す（1 ロールアウト 18 秒・60 行 × 3 本 = 180 本で 55 分ほど）
    TASKS_CSV=compe/val/TRACK2_ROBOT_LADDER60.csv TRACK_EPISODES=3 MUJOCO_GL=egl \
      PARC_N_ACTION_STEPS=16 PARC_EMA_ALPHA=0.3 LABEL=robot_ladder \
      solution/tools/eval/run_final_eval.sh track2

Reads:
  - compe/val/TRACK2_VAL72.csv（元タスク 12 件の出どころ）
  - work/LIBERO-plus/libero/libero/benchmark/task_classification.json（variant と難易度）
Generates:
  - --output に渡した CSV
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CLASSIFICATION = _REPO / "work/LIBERO-plus/libero/libero/benchmark/task_classification.json"
_BASE_CSV = _REPO / "compe/val/TRACK2_VAL72.csv"
_CATEGORY = "Robot Initial States"
# variant 名の形。`_view_<方位角>_<仰角>_<拡大率>_<レンズ回転>_<レンズ上下>_initstate_<番号>`。
_VARIANT_RE = re.compile(r"_view_0_0_100_0_0_initstate_(\d+)$")
_COLUMNS = [
    "task_num", "task_id", "instruction", "libero_plus_id",
    "suite", "category", "difficulty_level",
    "init_states_file", "n_nontarget_objects",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="書き出す CSV")
    p.add_argument("--levels", default="2,3,4,5",
                   help="測る難易度の段（既定: 2,3,4,5）")
    p.add_argument("--fallback-level", type=int, default=5,
                   help="その段の variant が無いタスクで代わりに取る段（既定: 5）")
    p.add_argument("--seed", type=int, default=20260828, help="選ぶときの種")
    return p.parse_args()


def _load_base_tasks() -> list[dict]:
    """摂動なしの 12 行を、CSV に書かれた順のまま返す。"""
    with open(_BASE_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["category"] == "Base"]
    if not rows:
        raise SystemExit(f"{_BASE_CSV} に category=Base の行が無い")
    return rows


def _load_variants() -> dict[tuple[str, str], dict[int, list[tuple[int, int]]]]:
    """(スイート, 元タスク名) -> 難易度 -> [(initstate 番号, 公式 id)] を返す。"""
    with open(_CLASSIFICATION) as f:
        classification = json.load(f)
    out: dict[tuple[str, str], dict[int, list[tuple[int, int]]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for suite, entries in classification.items():
        for entry in entries:
            if entry["category"] != _CATEGORY:
                continue
            m = _VARIANT_RE.search(entry["name"])
            if m is None:
                continue
            base_task = entry["name"][: m.start()]
            out[(suite, base_task)][entry["difficulty_level"]].append(
                (int(m.group(1)), entry["id"])
            )
    return out


def _pick(candidates: list[tuple[int, int]], used: set[int],
          rng: random.Random) -> tuple[int, int] | None:
    """まだ使っていない initstate 番号を 1 件選ぶ。"""
    free = [c for c in candidates if c[0] not in used]
    if not free:
        return None
    return rng.choice(sorted(free))


def main() -> None:
    args = _parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    rng = random.Random(args.seed)

    base_tasks = _load_base_tasks()
    variants = _load_variants()

    rows: list[dict] = []
    used_initstates: set[int] = set()
    substituted: list[str] = []

    for base in base_tasks:
        key = (base["suite"], base["task_id"])
        by_level = variants.get(key)
        if by_level is None:
            raise SystemExit(f"公式一覧に {_CATEGORY} の variant が無い: {key}")

        rows.append({
            "task_num": 0,
            "task_id": base["task_id"],
            "instruction": base["instruction"],
            "libero_plus_id": "",
            "suite": base["suite"],
            "category": "Base",
            "difficulty_level": "",
            "init_states_file": base["init_states_file"],
            "n_nontarget_objects": base["n_nontarget_objects"],
        })

        for level in levels:
            picked = _pick(by_level.get(level, []), used_initstates, rng)
            actual_level = level
            if picked is None:
                picked = _pick(by_level.get(args.fallback_level, []),
                               used_initstates, rng)
                actual_level = args.fallback_level
                substituted.append(f"{base['task_id']} の L{level} → L{actual_level}")
            if picked is None:
                raise SystemExit(
                    f"{base['task_id']} の L{level} を埋められない"
                    f"（L{args.fallback_level} も空）")
            initstate, plus_id = picked
            used_initstates.add(initstate)
            rows.append({
                "task_num": 0,
                "task_id": f"{base['task_id']}_view_0_0_100_0_0_initstate_{initstate}",
                "instruction": base["instruction"],
                "libero_plus_id": plus_id,
                "suite": base["suite"],
                "category": _CATEGORY,
                "difficulty_level": f"L{actual_level}",
                "init_states_file": base["init_states_file"],
                "n_nontarget_objects": base["n_nontarget_objects"],
            })

    for i, row in enumerate(rows, start=1):
        row["task_num"] = i

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_variants = sum(1 for r in rows if r["category"] == _CATEGORY)
    print(f"{args.output} に {len(rows)} 行"
          f"（Base {len(base_tasks)} 行・摂動 {n_variants} 行）", file=sys.stderr)
    print(f"initstate 番号は {len(used_initstates)} 通りで重複なし", file=sys.stderr)
    by_level = collections.Counter(r["difficulty_level"] for r in rows if r["difficulty_level"])
    print("段ごとの行数: " + " ".join(f"{k}={v}" for k, v in sorted(by_level.items())),
          file=sys.stderr)
    for line in substituted:
        print(f"  段を差し替えた: {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
