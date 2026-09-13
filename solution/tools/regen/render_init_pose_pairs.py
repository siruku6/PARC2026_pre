#!/usr/bin/env python3
"""摂動下再生の 1 本と、**摂動を加える前の同じ手本**を、動画で並べる。

**なぜ要るか。**摂動下再生（`docs/plan/23_data_regeneration_design.md` の 4.1 節の (A)）は
**配布データの行動列をそのまま、腕の初期姿勢だけ変えた環境で撃ち直したもの**である。
数値では「手先が 2〜11 cm・向きが 10〜34 度ずれている」と出ているが、
**その腕が画面の中で何をしているのかを誰も見ていない。**この道具はそれを目で見るためのものである。

| 列 | 中身 | どこから来るか |
|---|---|---|
| **before** | **摂動なし**（`initstate` 0）で同じ手本を再生したもの | **その場でシムを回す** |
| **after** | **摂動あり**で撃ち直したもの＝学習データそのもの | **npz の画像をそのまま読む**（再生し直さない） |

**after を撃ち直さないのは、学習に入る画そのものを見たいからである。**
シムを回し直すと、乱数や版の違いで別のものを見ることになる。

**書き出すのは 1 組 3 本である**（コマを 1 回だけ組んで 3 本へ同時に書く）。

| ファイル | 中身 |
|---|---|
| `<組>__before.mp4` | 摂動なしだけ（front 上・wrist 下） |
| `<組>__after.mp4` | 摂動ありだけ（同じ並び） |
| `<組>__pair.mp4` | **2 つを左右に並べたもの。**比べるのはこれを見る |

画面に焼く数字は、その歩までに**対象外の物体が動いた量**（mm・1 mm を超えると赤）・
歩数・ゴールが立った歩である。**npz が 1 コマごとの変位（`displacement_m`）を持っていれば
それを、持っていなければゴール時点の値を全コマに出す**（9/12 以前に作った npz は後者）。

用法（コンテナの中で打つ。`run_in_container.sh` が包む）:
    CHOWN_PATH="$PWD/work/regen" bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/render_init_pose_pairs.py \\
        --pairs work/regen/pairs_10.json \\
        --states work/artifacts/libero_states \\
        --npz-dir work/regen/npz_batch1 \\
        --out work/regen/videos_init_pose

Reads:
  - <pairs> … `[{"task":…, "suite":…, "demo":…, "initstate":…, "npz":…}]`
  - <states>/index.json・<states>/<suite>/<課題>_demo.npz
  - <npz-dir>/<npz> … 摂動下再生の 1 本（画像つき）
Generates:
  - <out>/<課題>__demo<番号>__init<番号>__{before,after,pair}.mp4
  - <out>/summary.json … 焼いた数値
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_before_after import (IMAGES, _before, _view,  # noqa: E402
                                 _write_columns)
from replay_yield import suite_of  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, required=True, metavar="FILE")
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--npz-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="`env.seed()` に渡す。**生成時と同じ 0 にしてある**")
    return parser.parse_args()


def _load_after(path: Path) -> dict:
    """摂動下再生の 1 本を読む。**撃ち直さない。**"""
    with np.load(path, allow_pickle=False) as bundle:
        if "displacement_m" in bundle.files:
            disp = [1000.0 * d for d in bundle["displacement_m"]]
        else:
            disp = [float(bundle["disp_at_goal_mm"])]
        return {
            "label": f"after  init{int(bundle['initstate'])}"
                     f"  {float(bundle['eef_offset_cm']):.1f}cm",
            "front": _view(bundle["front"]), "wrist": _view(bundle["wrist"]),
            "disp": disp,
            "goal_step": int(bundle["goal_step"]),
            "steps": int(bundle["steps"]),
            "eef_offset_cm": float(bundle["eef_offset_cm"]),
        }


def _run_before(sd, env, init_state: np.ndarray, actions: np.ndarray,
                ee_pos0: np.ndarray, seed: int) -> dict:
    """摂動なしで同じ手本を再生する。"""
    run = _before(sd, env, init_state, actions, ee_pos0, seed)
    return {"label": f"before  init0  {run['eef_offset_cm']:.1f}cm", **run}


def _write_all(prefix: Path, before: dict, after: dict) -> int:
    """before・after・pair の 3 本を書く。**焼く中身は `_panel` 1 か所で決まる。**"""
    for name, runs in (("before", [before]), ("after", [after]),
                       ("pair", [before, after])):
        length = _write_columns(Path(f"{prefix}__{name}.mp4"),
                                [run["label"] for run in runs], runs)
    return length


def main() -> None:
    args = _parse_args()
    import scripted_demo as sd
    from libero.libero import get_libero_path

    libero_root = Path(get_libero_path("bddl_files")).parent
    suites = suite_of(args.states)
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(args.pairs.read_text())
    summary = []
    for number, pair in enumerate(pairs, 1):
        task = pair["task"]
        demo = int(pair["demo"])
        suite = pair.get("suite") or suites[task]
        stem = f"{task}__demo{demo:03d}__init{int(pair['initstate']):03d}"
        print(f"=== [{number}/{len(pairs)}] {stem} ===", flush=True)

        after = _load_after(args.npz_dir / pair["npz"])
        env = sd._make_env(sd._bddl_path(libero_root, suite, task), True)
        try:
            with np.load(args.states / suite / f"{task}_demo.npz",
                         allow_pickle=False) as bundle:
                before = _run_before(sd, env,
                                     bundle[f"demo_{demo}/states"][0],
                                     bundle[f"demo_{demo}/actions"],
                                     bundle[f"demo_{demo}/ee_pos"][0], args.seed)
        finally:
            env.close()

        length = _write_all(args.out / stem, before, after)
        print(f"  → {stem}__{{before,after,pair}}.mp4  {length} コマ", flush=True)
        for run in (before, after):
            print(f"  {run['label']:<28} 歩 {run['steps']:>3}  "
                  f"goal {run['goal_step']:>4}", flush=True)
        summary.append({"stem": stem, "task": task, "demo": demo,
                        "initstate": int(pair["initstate"]),
                        "pos_cm": pair.get("pos_cm"),
                        "rot_deg": pair.get("rot_deg"),
                        **{side: {k: v for k, v in run.items() if k not in IMAGES}
                           for side, run in (("before", before), ("after", after))}})

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\n=== {args.out} ===")


if __name__ == "__main__":
    main()
