#!/usr/bin/env python3
"""Read per-episode step counts out of a PARC grading log.

The grading log is the only measuring instrument for the hidden 8-episode set.
It prints just one number ("総合スコア"), but the HTTP access lines carry more:
the harness calls ``POST /reset`` once per episode and ``POST /act`` once per
simulation step, so counting ``/act`` between ``/reset`` lines recovers the exact
step count of every episode.

That matters because ``success = bool(done) and not collided`` and the rollout
loop breaks on ``done`` (``b_PARC2026_final/pipeline/rollout.py``). An episode at
the step cap therefore never reached the goal, and an episode below the cap did
reach it -- so the step counts split the score into "how many episodes were lost"
and "how much the smoothness term moved", which the score alone cannot do.

Usage
-----
    # one log
    python3 solution/tools/analysis/parse_grading_log.py \
        b_PARC2026_final/outputs/submission_logs/track1/260830_pi0.5_6.1.0.txt

    # two logs side by side (baseline first)
    python3 solution/tools/analysis/parse_grading_log.py \
        b_PARC2026_final/outputs/submission_logs/track1/260829_pi0.5_5.0.1.txt \
        b_PARC2026_final/outputs/submission_logs/track1/260830_pi0.5_6.1.0.txt

The episode order is the order the harness ran them, and it is stable across runs
of the same track, so column i of one log pairs with column i of the other.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

RESET = re.compile(r'"POST /reset')
ACT = re.compile(r'"POST /act')
SCORE = re.compile(r"(track\d):\s*総合スコア\s*([0-9.]+)")
SETTINGS = re.compile(r"\[parc\] policy_server\s+(\S+)\s+(.*)")
STEP_CAP = 300


@dataclass
class Run:
    path: Path
    version: str = "?"
    settings: str = ""
    track: str = "?"
    score: float | None = None
    steps: list[int] = field(default_factory=list)

    @property
    def finished(self) -> list[bool]:
        """Episodes that reached the goal (below the cap), i.e. done=True."""
        return [s < STEP_CAP for s in self.steps]

    @property
    def n_finished(self) -> int:
        return sum(self.finished)


def parse(path: Path) -> Run:
    run = Run(path=path)
    pending = None  # steps of the episode currently being counted
    for line in path.read_text(errors="replace").splitlines():
        if m := SETTINGS.search(line):
            run.version, run.settings = m.group(1), m.group(2).strip()
        elif m := SCORE.search(line):
            run.track, run.score = m.group(1), float(m.group(2))
        elif RESET.search(line):
            if pending is not None:
                run.steps.append(pending)
            pending = 0
        elif ACT.search(line) and pending is not None:
            pending += 1
    if pending is not None:
        run.steps.append(pending)
    return run


def show_one(run: Run) -> None:
    print(f"== {run.path}")
    print(f"   版 {run.version} / {run.track} / 総合スコア {run.score}")
    print(f"   設定 {run.settings}")
    cells = [f"{s}{'*' if s >= STEP_CAP else ''}" for s in run.steps]
    print(f"   歩数 {' '.join(cells)}   (* は打ち切り＝目標未達)")
    print(f"   目標に届いた本数 {run.n_finished} / {len(run.steps)}")
    if run.score is not None and run.n_finished:
        per = run.score * len(run.steps) / run.n_finished
        print(f"   届いた 1 本あたりの success*(1-collision)*smooth {per:.4f}")


def show_pair(a: Run, b: Run) -> None:
    if len(a.steps) != len(b.steps):
        print(f"! エピソード数が違う: {len(a.steps)} vs {len(b.steps)}", file=sys.stderr)
    print()
    print(f"== 対で見る（基準 {a.version} → {b.version}）")
    print(f"{'ep':>3} {a.version:>10} {b.version:>10} {'差':>8}  読み")
    for i, (x, y) in enumerate(zip(a.steps, b.steps), 1):
        if x >= STEP_CAP and y < STEP_CAP:
            note = "拾った（打ち切り → 到達）"
        elif x < STEP_CAP and y >= STEP_CAP:
            note = "落とした（到達 → 打ち切り）"
        elif x >= STEP_CAP:
            note = "どちらも打ち切り"
        else:
            note = f"到達どうし {(y - x) / x:+.1%}"
        print(f"{i:>3} {x:>10} {y:>10} {y - x:>+8}  {note}")
    print(f"{'到達':>3} {a.n_finished:>10} {b.n_finished:>10} {b.n_finished - a.n_finished:>+8}")
    if a.score is not None and b.score is not None:
        print(f"{'点':>3} {a.score:>10.3f} {b.score:>10.3f} {b.score - a.score:>+8.3f}")
        n = len(a.steps)
        if b.n_finished:
            expect = a.score * b.n_finished / a.n_finished if a.n_finished else 0.0
            print(
                f"    到達本数の変化だけで説明できる点は {expect:.3f}、"
                f"残り {b.score - expect:+.3f} がなめらかさ側の動きである"
            )
        print(f"    参考: 8 本なら 1 本 = {a.score / max(a.n_finished, 1):.3f} 点")
        _ = n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", type=Path, help="採点ログ（2 本渡すと対で並べる）")
    args = ap.parse_args()

    runs = [parse(p) for p in args.logs]
    for run in runs:
        show_one(run)
    if len(runs) == 2:
        show_pair(runs[0], runs[1])


if __name__ == "__main__":
    main()
