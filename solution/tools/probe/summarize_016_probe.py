#!/usr/bin/env python3
"""016: 順方向 vs 逆向きの評価結果を 1 つの表にまとめる。

`run_reverse_probe_eval.sh` の出力ディレクトリを読み、タスクごとに
順方向（`<stem>`）と逆向き（`<stem>_rev`）を 1 組として並べる。

読む先:
  results/object_disp.jsonl  … エピソードごとの success / collided / total_steps
                                （差し替え版 rollout が書く。1 行 = 1 エピソード）
  results/*.json             … 採点器の task ごとの success_rate / collision_rate
  server.log                 … `POST /act` の総数（方策が呼ばれた回数）

用法:
    python3 summarize_016_probe.py work/final_eval/<出力ディレクトリ>
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REV_SUFFIX = "_rev"


def load_episodes(out_dir: Path) -> dict[str, list[dict]]:
    """object_disp.jsonl を task 名ごとのエピソード列にまとめる。"""
    path = out_dir / "results" / "object_disp.jsonl"
    by_task: dict[str, list[dict]] = {}
    if not path.exists():
        return by_task
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_task.setdefault(rec["task"], []).append(rec)
    for eps in by_task.values():
        eps.sort(key=lambda r: r.get("episode", 0))
    return by_task


def load_scorer(out_dir: Path) -> dict[str, dict]:
    """採点器の json から task ごとの success_rate / collision_rate を拾う。"""
    out: dict[str, dict] = {}
    for path in sorted((out_dir / "results").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for track in data.get("tracks", []):
            for task in track.get("tasks", []):
                out[task["task_name"]] = task
    return out


def count_post_act(out_dir: Path) -> int | None:
    """server.log の `POST /act` 行数（＝方策が呼ばれた回数）。"""
    path = out_dir / "server.log"
    if not path.exists():
        return None
    return sum(line.count("POST /act")
               for line in path.read_text(encoding="utf-8", errors="replace").splitlines())


def fmt_row(label: str, eps: list[dict], scored: dict | None) -> str:
    n = len(eps)
    succ = sum(1 for e in eps if e.get("success"))
    steps = [e.get("total_steps") for e in eps if e.get("total_steps") is not None]
    coll = sum(1 for e in eps if e.get("collided"))
    med = f"{statistics.median(steps):.0f}" if steps else "-"
    steps_txt = f"{med} ({', '.join(str(s) for s in steps)})" if steps else "-"
    rate = ""
    if scored is not None:
        rate = f"{scored.get('success_rate', 0.0):.2f}"
    return f"| {label} | {succ}/{n} | {steps_txt} | {coll}/{n} | {rate} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    by_task = load_episodes(out_dir)
    scored = load_scorer(out_dir)
    names = set(by_task) | set(scored)
    stems = sorted({n[: -len(REV_SUFFIX)] if n.endswith(REV_SUFFIX) else n
                    for n in names})

    print(f"# 016 逆向きプロブ: {out_dir}")
    post = count_post_act(out_dir)
    if post is not None:
        print(f"POST /act 合計: {post}")
    print()
    print("| タスク / 向き | 成功 | ステップ数 中央 (各本) | 衝突 | 採点器 success_rate |")
    print("|---|---|---|---|---|")
    for stem in stems:
        for suffix, arrow in (("", "順方向"), (REV_SUFFIX, "逆向き")):
            name = stem + suffix
            if name not in names:
                continue
            short = stem[:46] + ("…" if len(stem) > 46 else "")
            print(fmt_row(f"{short} / **{arrow}**",
                          by_task.get(name, []), scored.get(name)))
    print()

    # go / no-go: 逆向きで 1 本でも着いたタスクがいくつあるか
    rev_ok = [s for s in stems
              if any(e.get("success") for e in by_task.get(s + REV_SUFFIX, []))]
    fwd_ok = [s for s in stems
              if any(e.get("success") for e in by_task.get(s, []))]
    print(f"順方向で 1 本以上成功したタスク: {len(fwd_ok)}/{len(stems)}  {fwd_ok}")
    print(f"逆向きで 1 本以上成功したタスク: {len(rev_ok)}/{len(stems)}  {rev_ok}")


if __name__ == "__main__":
    main()
