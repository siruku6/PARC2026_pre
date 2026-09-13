#!/usr/bin/env python3
"""**寄せる区間の Ruckig 有り／無し**を、同じ場面から 1 本ずつ動画にする。

**なぜ要るか。**`prealign_ruckig_probe.py` は歩留まりと歩数の表を出すが、
**腕が寄せる区間でどう動いて、そのあと記録された行動をどう辿ったのかは、
表からは見えない。**この道具はそれを目で見るためのものである。

**2 本は 1 か所だけ違う。**同じ課題・同じデモ・同じ初期姿勢から始め、
**先頭の「寄せる区間」の作り方だけを変える**（`replay_yield._prealign` の `planner`）。
そのあと流す記録された行動は 1 ビットも同じ。

| ファイル | 寄せ方 |
|---|---|
| `<組>__without.mp4` | **比例制御**（`planner="proportional"`） |
| `<組>__with.mp4` | **Ruckig**（`planner="glide"`） |

画面に焼くもの——**区間**（`PREALIGN` ＝寄せている／`REPLAY` ＝記録どおり流している）・
歩数・対象外の物体が動いた量（1 mm を超えると赤）・ゴールが立った歩・採否。

用法（コンテナの中で打つ。`run_in_container.sh` が包む）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/render_prealign_pairs.py \\
        --pairs work/regen/prealign_video_pairs.json \\
        --states work/artifacts/libero_states --out work/regen/videos_prealign

Reads:
  - <pairs> … `[{"task":…, "demo":…, "initstate":…}]`
  - <states>/index.json・<states>/<suite>/<課題>_demo.npz
Generates:
  - <out>/<番号>_<課題>__demo<番号>__init<番号>__{without,with}.mp4
  - <out>/summary.json … 焼いた数値
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_before_after import (FPS, HEADER, PANEL, _annotate,  # noqa: E402
                                 _header_band, _view)
from replay_yield import (MIN_OFFSET_CM, _replay_one, _variant,  # noqa: E402
                          passes_gates, suite_of)

PLANNERS = (("without", "proportional"), ("with", "glide"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, required=True, metavar="FILE")
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--min-offset-cm", type=float, default=MIN_OFFSET_CM,
                        metavar="CM")
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="`env.seed()`。**比較の表と同じ 0 にしてある**")
    return parser.parse_args()


def _write(path: Path, record, result: dict, header: str) -> int:
    """1 本を動画にする（上段 front・下段 wrist）。"""
    import cv2

    front, wrist = _view(record.front), _view(record.wrist)
    disp_mm = [1000.0 * d for d in record.displacements]
    band = _header_band(PANEL, header)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                            (PANEL, HEADER + 2 * PANEL))
    for step in range(len(front)):
        here = disp_mm[min(step, len(disp_mm) - 1)] if disp_mm else 0.0
        aligning = step < result["prealign_steps"]
        goal = 0 < result["goal_step"] <= step + 1
        lines = [
            ("PREALIGN" if aligning else "REPLAY",
             (80, 200, 255) if aligning else (200, 200, 200)),
            (f"nontarget {here:6.2f}mm" + ("  OVER" if here > 1.0 else ""),
             (60, 60, 255) if here > 1.0 else (235, 235, 235)),
            (f"step {step + 1}/{len(front)}" + ("  GOAL" if goal else ""),
             (80, 220, 80) if goal else (235, 235, 235)),
        ]
        frame = np.concatenate([band, _annotate(front[step], lines),
                                np.ascontiguousarray(wrist[step])], axis=0)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return len(front)


def main() -> None:
    args = _parse_args()
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    suites = suite_of(args.states)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    for number, pair in enumerate(json.loads(args.pairs.read_text()), 1):
        task, demo, initstate = pair["task"], int(pair["demo"]), int(pair["initstate"])
        suite = suites[task]
        stem = f"{number:02d}_{task}__demo{demo:03d}__init{initstate:03d}"
        print(f"=== {stem} ===", flush=True)
        env = sd._make_env(sd._bddl_path(libero_root, suite, _variant(task, initstate)),
                           with_images=True)
        record = {"stem": stem, "task": task, "demo": demo, "initstate": initstate}
        try:
            with np.load(args.states / suite / f"{task}_demo.npz",
                         allow_pickle=False) as bundle:
                states = bundle[f"demo_{demo}/states"]
                actions = bundle[f"demo_{demo}/actions"]
                ee_pos0 = bundle[f"demo_{demo}/ee_pos"][0]
                for label, planner in PLANNERS:
                    result, rollout = _replay_one(sd, env, states[0], actions,
                                                  args.seed, "prealign", ee_pos0,
                                                  record_images=True, planner=planner)
                    rollout.attach_images()
                    gates = passes_gates(result, args.min_offset_cm)
                    verdict = "KEEP" if gates["keep"] else "DROP"
                    path = args.out / f"{stem}__{label}.mp4"
                    length = _write(path, rollout.record, result, f"{planner}  {verdict}")
                    # **画像は 1 本 100 MB を超える。**次の版を撃つ前に手放す。
                    del rollout
                    print(f"  {label:>7}: 寄せ {result['prealign_steps']:>2} 歩"
                          f" 残り {result['prealign_residual_mm']:5.1f} mm"
                          f" goal {result['goal_step']:>4} {verdict}"
                          f" → {path.name} {length} コマ", flush=True)
                    record[label] = {**gates,
                                     **{k: v for k, v in result.items() if k != "eef0"}}
        finally:
            env.close()
        summary.append(record)

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\n=== {args.out} ===")


if __name__ == "__main__":
    main()
