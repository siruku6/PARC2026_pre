#!/usr/bin/env python3
"""生成の前と後を、同じ画面に並べた動画にする。

**なぜ要るか。**数字（歩留まり 16.7%・手先のずれ 中央 3.84 cm）だけでは、
**作ったデータが狙いどおりのものかを目で確かめられない。**この道具は 1 つの
(課題, デモ) について 3 通りを並べる。

| 列 | 中身 |
|---|---|
| **before** | **摂動なしで元デモを再生したもの。**配布データに入っているのと同じ動き |
| **after (open_loop)** | **摂動した環境で同じ行動列をそのまま流したもの。**腕はずれたまま進む |
| **after (prealign)** | **先にデモの 0 コマ目へ寄せてから流したもの。**先頭に「戻す」区間が入る |

**画面に焼く数字は 3 つ。**引き継ぎ時の手先のずれ（cm）・その歩までの対象外の
物体のずれ（mm・1 mm を超えると赤）・ゴールが立った歩。**採点と同じ規則で
採用したか却下したかも焼く。**

**この動画で確かめられるのは「作ったデータが狙いどおりか」までである。**
点が上がるかどうかは学習して評価しないと分からない。

用法（コンテナの中で打つ）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/render_examples.py \\
        --states work/artifacts/libero_states --out work/regen/videos \\
        --case pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate:32:372

Reads:
  - <states>/index.json・<states>/<suite>/<task>_demo.npz
Generates:
  - <out>/<課題>__demo<番号>__init<番号>.mp4 … 3 列 × 2 段（front / wrist）
  - <out>/summary.json … 焼いた数値
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))

from replay_yield import _replay_one, _variant  # noqa: E402

PANEL = 256
HEADER = 22
FPS = 20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--case", action="append", required=True,
                        metavar="TASK:DEMO:INITSTATE",
                        help="並べる組。**何度でも渡せる**")
    parser.add_argument("--min-offset-cm", type=float, default=2.0, metavar="CM")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    return parser.parse_args()


def _annotate(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]
              ) -> np.ndarray:
    """コマの下に数行を焼く。**026 の `render_contact_videos.py` と同じ作り。**"""
    import cv2

    canvas = np.ascontiguousarray(frame)
    for number, (text, color) in enumerate(lines):
        y = canvas.shape[0] - 6 - 14 * (len(lines) - 1 - number)
        cv2.putText(canvas, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    color, 1, cv2.LINE_AA)
    return canvas


def _header(width: int, text: str) -> np.ndarray:
    import cv2

    band = np.full((HEADER, width, 3), 24, dtype=np.uint8)
    cv2.putText(band, text, (5, HEADER - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return band


def _run(sd, libero_root: Path, suite: str, task: str, demo: int,
         initstate: int, method: str, bundle, seed: int) -> dict:
    """1 通りを走らせ、画像と焼く数値を返す。"""
    bddl = sd._bddl_path(libero_root, suite, _variant(task, initstate))
    env = sd._make_env(bddl, with_images=True)
    try:
        actions = bundle[f"demo_{demo}/actions"]
        states = bundle[f"demo_{demo}/states"]
        key = f"demo_{demo}/ee_pos"
        ee_pos0 = bundle[key][0] if key in bundle.files else None
        result, rollout = _replay_one(sd, env, states[0], actions, seed, method,
                                      ee_pos0, record_images=True)
        rollout.attach_images()
        record = rollout.record
        # **画像は学習データの向き（180 度回し）で溜まっている。**見るために戻す。
        front = np.asarray(record.front)[:, ::-1, ::-1]
        wrist = np.asarray(record.wrist)[:, ::-1, ::-1]
        return {"front": front, "wrist": wrist,
                "disp": [1000.0 * d for d in record.displacements],
                **result}
    finally:
        env.close()


def _verdict(run: dict, min_offset: float) -> tuple[str, bool]:
    reached = 0 < run["goal_step"] <= 300
    clean = (run["disp_at_goal_mm"] or 0.0) <= 1.0
    covered = run["eef_offset_cm"] >= min_offset
    if reached and clean and covered:
        return "KEEP", True
    why = []
    if not reached:
        why.append("no goal")
    if not clean:
        why.append("contact")
    if not covered:
        why.append("offset<%.0fcm" % min_offset)
    return "DROP: " + ",".join(why), False


def main() -> None:
    args = _parse_args()
    import cv2
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    index = json.loads((args.states / "index.json").read_text())
    suite_of = {k.split("/")[1].replace("_demo.hdf5", ""): k.split("/")[0]
                for k in index}
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []

    for case in args.case:
        task, demo_text, init_text = case.rsplit(":", 2)
        demo, initstate = int(demo_text), int(init_text)
        suite = suite_of[task]
        npz = args.states / suite / f"{task}_demo.npz"
        print(f"=== {task} demo={demo} initstate={initstate} ===", flush=True)

        columns = [("before (no perturb)", 0, "open_loop"),
                   (f"after open_loop", initstate, "open_loop"),
                   (f"after prealign", initstate, "prealign")]
        runs = []
        with np.load(npz, allow_pickle=False) as bundle:
            for label, init, method in columns:
                run = _run(sd, libero_root, suite, task, demo, init, method,
                           bundle, args.seed)
                verdict, kept = _verdict(run, args.min_offset_cm if init else 0.0)
                run.update(label=label, verdict=verdict, kept=kept,
                           initstate=init, method=method)
                runs.append(run)
                print(f"  {label:<22} 歩数 {run['steps']:>3}  "
                      f"ずれ {run['eef_offset_cm']:>5.2f}cm  "
                      f"goal {run['goal_step']:>4}  "
                      f"接触 {run['disp_full_mm']:>7.2f}mm  {verdict}", flush=True)

        length = max(len(r["front"]) for r in runs)
        width = PANEL * len(runs)
        path = args.out / f"{task}__demo{demo:03d}__init{initstate:03d}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (width, HEADER + 2 * PANEL))
        bands = np.concatenate([_header(PANEL, r["label"]) for r in runs], axis=1)
        for step in range(length):
            row_front, row_wrist = [], []
            for run in runs:
                last = len(run["front"]) - 1
                index_here = min(step, last)          # 短い列は最後のコマで止める
                disp = run["disp"][min(index_here, len(run["disp"]) - 1)]
                over = disp > 1.0
                goal = 0 < run["goal_step"] <= index_here + 1
                lines = [
                    (f"init {run['initstate']} off {run['eef_offset_cm']:.2f}cm",
                     (235, 235, 235)),
                    (f"nontarget {disp:6.2f}mm" + ("  OVER 1mm" if over else ""),
                     (60, 60, 255) if over else (235, 235, 235)),
                    (f"step {index_here + 1}" + ("  GOAL" if goal else "")
                     + f"   {run['verdict']}",
                     (80, 220, 80) if run["kept"] else (60, 60, 255)),
                ]
                row_front.append(_annotate(run["front"][index_here], lines))
                row_wrist.append(np.ascontiguousarray(run["wrist"][index_here]))
            frame = np.concatenate([
                bands,
                np.concatenate(row_front, axis=1),
                np.concatenate(row_wrist, axis=1)], axis=0)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        size = path.stat().st_size / 1e6
        print(f"  → {path.name}  {length} コマ  {size:.1f} MB", flush=True)
        summary.append({"task": task, "demo": demo, "initstate": initstate,
                        "video": path.name,
                        "runs": [{k: v for k, v in r.items()
                                  if k not in ("front", "wrist", "disp")}
                                 for r in runs]})

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\n=== {args.out} ===")


if __name__ == "__main__":
    main()
