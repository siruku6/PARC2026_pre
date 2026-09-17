#!/usr/bin/env python3
"""手元評価の実行を評価セット（CSV）ごとに束ね、候補を 4 つの数で並べる。

**なぜ要るか。** `score_composite.py` は 1 実行の点を出し、`decision_score.py` は
2 つの集合を混ぜて 1 位を選ぶ。**「どの評価セットで、どの候補が、どう負けたのか」を
並べて読む道具が無かった。**提出物を選ぶ最後の判断には、合計点 1 つではなく
**「着いたか」「当てたか」「なめらかか」を分けた表**が要る。

出す 4 つの数は次のとおりである。

| 名前 | 意味 |
|---|---|
| 成功率 | ゴールに着き、かつ対象外の物を 1 mm 以上動かさなかった回の割合 |
| ゴール到達率 | ゴール条件を満たした回の割合（衝突は問わない） |
| 方策の衝突率 | **方策のせいで**対象外の物を 1 mm 以上動かした回の割合 |
| 滑らかさ | 運営の式の smooth metrics（0〜1。全回成功でも 0.49 付近が実力の上限） |

**「方策の衝突率」は素の衝突率と違う。** 場面の準備中にまだ落ちている物があり、
その変位が「動いた」と数えられる回がある。**この回はどの重みでも同じ値になり、
学習でも推論でも直せない**（同じ課題・同じ物体で、全実行・全エピソードを通して
変位が 1 ビットも変わらないことで見分ける）。素の衝突率はこれを含むので、
**重みを比べる物差しとしては使えない。**

    python3 solution/tools/eval/compare_candidates.py            # 全 CSV
    python3 solution/tools/eval/compare_candidates.py --csv TRACK2_NOISE84.csv
    python3 solution/tools/eval/compare_candidates.py --markdown

Reads:
  - work/final_eval/*/results/object_disp.jsonl（1 回ごとの成否・到達・変位）
  - work/final_eval/*/results/server_8000.json（タスクごとのなめらかさ 5 指標）
  - work/final_eval/*/TASKS_CSV.csv・eval_conditions.json（どの評価セットの実行か）
Generates:
  - 標準出力のみ
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_composite import DEFAULT_WEIGHTS, smooth_components, smooth_metrics  # noqa: E402

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_FINAL_EVAL = _REPOSITORY_ROOT / "work" / "final_eval"
_VAL_DIR = _REPOSITORY_ROOT / "compe" / "val"
_COLLISION_M = 0.001


@dataclass
class Run:
    """1 回の評価実行。"""

    directory: Path
    csv_name: str
    episodes: list[dict]
    smooth: float | None
    renderer: str
    n_episodes_per_row: int


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _known_csv_digests() -> dict[str, str]:
    return {_digest(p): p.name for p in sorted(_VAL_DIR.glob("*.csv"))}


def _csv_of(directory: Path, digests: dict[str, str]) -> str | None:
    """この実行がどの評価セットを回したかを、CSV の中身で決める。

    `eval_conditions.json` の名前は当てにしない——同じ名前で中身が差し替わりうる。
    """
    copied = directory / "TASKS_CSV.csv"
    if copied.is_file():
        name = digests.get(_digest(copied))
        if name:
            return name
        return f"（未登録の CSV・{sum(1 for _ in copied.open()) - 1} 行）"
    return None


def _renderer_of(directory: Path) -> str:
    conditions = directory / "eval_conditions.json"
    if conditions.is_file():
        try:
            return str(json.loads(conditions.read_text()).get("mujoco_gl", "?"))
        except json.JSONDecodeError:
            return "?"
    return "記録なし"


def _smooth_of(directory: Path, weights: dict[str, float]) -> float | None:
    """タスクごとの smooth metrics を平均する（成功率では重み付けしない）。"""
    result = directory / "results" / "server_8000.json"
    if not result.is_file():
        return None
    payload = json.loads(result.read_text())
    values = []
    for track in payload.get("tracks", []):
        for task in track.get("tasks", []):
            metrics = task.get("metrics")
            if metrics:
                values.append(smooth_metrics(smooth_components(metrics), weights))
    return sum(values) / len(values) if values else None


def _load_runs(weights: dict[str, float]) -> list[Run]:
    digests = _known_csv_digests()
    runs = []
    for directory in sorted(_FINAL_EVAL.glob("*/")):
        disp = directory / "results" / "object_disp.jsonl"
        if not disp.is_file():
            continue
        episodes = []
        for line in disp.read_text(errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not episodes:
            continue
        csv_name = _csv_of(directory, digests)
        if csv_name is None:
            continue
        per_row = max(int(e.get("episode", 0)) for e in episodes) + 1
        runs.append(
            Run(
                directory=directory,
                csv_name=csv_name,
                episodes=episodes,
                smooth=_smooth_of(directory, weights),
                renderer=_renderer_of(directory),
                n_episodes_per_row=per_row,
            )
        )
    return runs


def _scene_artifacts(runs: list[Run]) -> set[tuple[str, str]]:
    """どの実行・どのエピソードでも変位が 1 ビットも変わらない（課題, 物体）を拾う。

    場面の準備中に落ちている物である。**重みを変えても動かないので、
    方策の衝突とは分けて数える。**
    """
    seen: dict[tuple[str, str], set[float]] = defaultdict(set)
    runs_seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for run in runs:
        for episode in run.episodes:
            for name, value in (episode.get("max_disp_m") or {}).items():
                if value > _COLLISION_M:
                    key = (episode["task"], name)
                    seen[key].add(round(float(value), 6))
                    runs_seen[key].add(run.directory.name)
    return {
        key
        for key, values in seen.items()
        if len(values) == 1 and len(runs_seen[key]) >= 2
    }


def _rates(run: Run, artifacts: set[tuple[str, str]]) -> dict[str, float]:
    total = len(run.episodes)
    success = sum(1 for e in run.episodes if e.get("success"))
    done = sum(1 for e in run.episodes if e.get("done"))
    collided = sum(1 for e in run.episodes if e.get("collided"))
    policy = 0
    for episode in run.episodes:
        movers = [
            name
            for name, value in (episode.get("max_disp_m") or {}).items()
            if value > _COLLISION_M and (episode["task"], name) not in artifacts
        ]
        if movers:
            policy += 1
    return {
        "n": total,
        "success": success / total,
        "done": done / total,
        "collided": collided / total,
        "policy_collision": policy / total,
        "smooth": run.smooth if run.smooth is not None else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--csv", help="この評価セットだけ出す（例 TRACK2_NOISE84.csv）")
    parser.add_argument("--markdown", action="store_true", help="Markdown の表で出す")
    parser.add_argument("--min-runs", type=int, default=1, help="この数未満の実行しかない CSV は出さない")
    args = parser.parse_args()

    runs = _load_runs(DEFAULT_WEIGHTS)
    artifacts = _scene_artifacts(runs)
    by_csv: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_csv[run.csv_name].append(run)

    if artifacts:
        print(f"# 場面の落下物として除いた（課題, 物体）: {len(artifacts)} 組")
        for task, obj in sorted(artifacts)[:12]:
            print(f"#   {task} / {obj}")
        if len(artifacts) > 12:
            print(f"#   …ほか {len(artifacts) - 12} 組")
        print()

    for csv_name in sorted(by_csv):
        if args.csv and csv_name != args.csv:
            continue
        group = by_csv[csv_name]
        if len(group) < args.min_runs:
            continue
        print(f"## {csv_name}")
        if args.markdown:
            print("| 実行 | 本数 | 描画 | 成功率 | ゴール到達率 | 方策の衝突率 | 素の衝突率 | 滑らかさ |")
            print("|---|---|---|---|---|---|---|---|")
        else:
            print(f"{'実行':58} {'本':>4} {'描画':>7} {'成功':>7} {'到達':>7} {'方策衝突':>8} {'素衝突':>7} {'滑らか':>7}")
        for run in sorted(group, key=lambda r: r.directory.name):
            r = _rates(run, artifacts)
            if args.markdown:
                print(
                    f"| `{run.directory.name}` | {r['n']} | {run.renderer} | {r['success']:.3f} "
                    f"| {r['done']:.3f} | {r['policy_collision']:.3f} | {r['collided']:.3f} | {r['smooth']:.3f} |"
                )
            else:
                print(
                    f"{run.directory.name:58} {r['n']:>4} {run.renderer:>7} {r['success']:>7.3f} "
                    f"{r['done']:>7.3f} {r['policy_collision']:>8.3f} {r['collided']:>7.3f} {r['smooth']:>7.3f}"
                )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
