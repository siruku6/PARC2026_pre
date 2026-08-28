#!/usr/bin/env python3
"""variant ごとに、scripted policy（`scripted_demo.py`）が通るかどうかを数えて表に残す。

**同じ検査は 2026-08-26 に一度やっているが、通った番号が保存されていない。**
[plan/12](../../../docs/plan/12_tier2_recording_spec.md) 4.2 節は「variant 78 件のうち
3 回とも通ったのは 30」という**個数だけ**を残していて、どの `initstate` が通ったのかが
残っていない。**だから 8/28 に「手先のずれ 2 cm 以上から variant 12 件を選び直す」と決めた
ときに、選び直す母集団が引けなかった。**このスクリプトは同じ検査を、
**variant 1 件 1 行の JSONL** に残す形でやり直すためのものである。

やっていることは variant 1 件につき `scripted_demo.py verify` を 1 回呼ぶだけである。
**描画しないので GPU は要らず、variant 1 件 13 秒ほどで終わる。**variant ごとに独立なので
`--workers` で並列にできる。

**scripted policy が通ることは、学習データを作れることと同義である。** 通らない variant は
`record` に回しても失敗の山しか出ない（[plan/12](../../../docs/plan/12_tier2_recording_spec.md)
4.1 節）。**通る variant の一覧が、Tier2 で作れる範囲そのものである。**

使い方（コンテナの中で回す）:

    docker exec -i parc2026_verify bash -c 'cat > /workspace/scan_scripted_pass.py' \\
        < solution/tools/probe/scan_scripted_pass.py
    docker exec parc2026_verify /workspace/venv/bin/python \\
        /workspace/scan_scripted_pass.py \\
        --tasks-csv /workspace/tier2_variants.csv \\
        --libero-root /workspace/LIBERO-plus/libero/libero \\
        --workers 6 --output /workspace/scripted_pass.jsonl

Reads:
  - --tasks-csv に渡した CSV（`task_id`・`suite`・`category`・`difficulty_level`）
  - <libero-root>/bddl_files/<suite>/<variant>.bddl
Generates:
  - --output に渡した JSONL（variant 1 件 1 行）
  - 標準出力の要約
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_VARIANT_RE = re.compile(r"_view_\d+_\d+_\d+_\d+_\d+_initstate_(\d+)$")
_SUCCESS_RE = re.compile(r"成功 (\d+)/(\d+)")
_STEPS_RE = re.compile(r"ステップ数 中央値 (\d+) 最小 (\d+) 最大 (\d+)")
_SCRIPT = "solution/tools/scripted_demo/scripted_demo.py"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks-csv", type=Path, required=True)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", default="0,1,2",
                   help="初期配置の番号（既定 0,1,2。plan/12 4.2 節と同じ）")
    p.add_argument("--workers", type=int, default=4,
                   help="同時に走らせる本数。1 本 1 コアを使う")
    p.add_argument("--workdir", type=Path, default=Path("/workspace"),
                   help="scripted_demo.py を相対で引く起点")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def _base_task(task_id: str) -> str:
    return _VARIANT_RE.sub("", task_id)


def _verify_one(row: dict, args: argparse.Namespace) -> dict:
    """variant 1 件ぶん `verify` を呼び、成功数とステップ数を拾う。"""
    task_id = row["task_id"]
    command = [
        args.python, _SCRIPT, "verify",
        "--task", _base_task(task_id),
        "--variant", task_id,
        "--episodes", args.episodes,
        "--libero-root", str(args.libero_root),
    ]
    started = time.time()
    completed = subprocess.run(command, cwd=args.workdir, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    success = _SUCCESS_RE.search(output)
    steps = _STEPS_RE.search(output)
    match = _VARIANT_RE.search(task_id)
    return {
        "task_id": task_id,
        "base_task": _base_task(task_id),
        "initstate": int(match.group(1)) if match else None,
        "difficulty_level": row.get("difficulty_level", ""),
        "n_success": int(success.group(1)) if success else None,
        "n_episodes": int(success.group(2)) if success else None,
        "steps_median": int(steps.group(1)) if steps else None,
        "steps_max": int(steps.group(3)) if steps else None,
        "returncode": completed.returncode,
        "seconds": round(time.time() - started, 1),
        "tail": "" if success else output[-400:],
    }


def main() -> None:
    args = _parse_args()
    with open(args.tasks_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["category"] != "Base"]

    print(f"{len(rows)} variant を {args.workers} 並列で検査する", flush=True)
    started = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(lambda r: _verify_one(r, args), rows), start=1):
            results.append(record)
            passed = record["n_success"]
            total = record["n_episodes"]
            verdict = "検査が落ちた" if passed is None else f"{passed}/{total}"
            print(f"[{i}/{len(rows)}] {verdict:>6s} "
                  f"{record['difficulty_level']:>3s} "
                  f"歩数 {record['steps_median'] or 0:3d}  "
                  f"{record['task_id'][-26:]}", flush=True)

    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    print(f"\n所要 {time.time() - started:.0f} 秒 → {args.output}")

    full = [r for r in results if r["n_success"] == r["n_episodes"] and r["n_episodes"]]
    broken = [r for r in results if r["n_success"] is None]
    print(f"3 回とも通った variant: {len(full)} / {len(results)}"
          f"（検査そのものが落ちた: {len(broken)}）")
    by_task = collections.Counter(r["base_task"] for r in full)
    for task, count in sorted(by_task.items()):
        total = sum(1 for r in results if r["base_task"] == task)
        print(f"  {task[:52]:52s} {count}/{total}")
    for r in broken:
        print(f"  検査が落ちた: {r['task_id']}  {r['tail'][-160:]}")


if __name__ == "__main__":
    main()
