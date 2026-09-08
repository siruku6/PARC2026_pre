#!/usr/bin/env python3
"""初期観測を、方策サーバーが受け取る形そのままで npz に落とす。

**配布 T3（既定）でも、`compe/val` の CSV で挙げた課題でも取れる。**
後者は「いま成功している課題」を対照に使うためで、
**成功している課題で色に反応していなければ、成功は「候補が 1 つだから」である。**

**推論だけで調べたいことがあるときに使う。**ロールアウトを回さずに、
「同じ画に、指示文だけ変えて入れたら出力がどう変わるか」を測れる。

観測のキーは `policy_server.py` の `_build_batch` が読むものに揃えてある——
`agentview_image` / `robot0_eye_in_hand_image` / `robot0_eef_pos` /
`robot0_eef_quat` / `robot0_gripper_qpos`。**画は 180 度回す前の生である**
（回すのは `LiberoProcessorStep` の仕事なので、ここでやると二重になる）。

シムのイメージ（`parc2026-final`）の中で動かすこと。lerobot は要らない。

    bash solution/tools/probe/run_dump_t3_obs.sh work/probe_color/t3_obs.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

T3_ASSETS = Path("/workspace/compe/t3/assets")
# **課題の並びは評価と同じにする**（`bddl_files/libero_t3` の名前順）。
CAMERAS = ("agentview", "robot0_eye_in_hand")


def instruction_of(bddl: Path) -> str:
    """bddl の `(:language …)` を 1 行の文へ直す。"""
    text = bddl.read_text()
    head = text.index("(:language")
    tail = text.index(")", head)
    return " ".join(text[head + len("(:language"):tail].split())


LIBERO = Path("/workspace/LIBERO-plus/libero/libero")


def csv_jobs(path: Path, only: list[str] | None):
    """`compe/val` の CSV から (名前, bddl, init のディレクトリ) を作る。"""
    import csv as _csv

    for row in _csv.DictReader(path.open()):
        task_id = row["task_id"]
        if only and task_id not in only:
            continue
        suite = row.get("suite") or "libero_90"
        yield (task_id,
               LIBERO / "bddl_files" / suite / f"{task_id}.bddl",
               LIBERO / "init_files" / suite)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", type=Path)
    ap.add_argument("--init-index", type=int, default=0,
                    help="init は 1 課題 50 行ある。そのどれを使うか")
    ap.add_argument("--csv", type=Path, default=None,
                    help="`compe/val/*.csv` を渡すと T3 ではなくその課題を取る"
                         "（列 `task_id` / `suite` / `init_states_file` を読む）")
    ap.add_argument("--only", nargs="*", default=None,
                    help="CSV のうち `task_id` がこれに一致するものだけ")
    ap.add_argument("--camera-size", type=int, default=128,
                    help="評価パイプラインが渡す大きさ（規定 128）")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace")
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    if args.csv:
        jobs = list(csv_jobs(args.csv, args.only))
    else:
        bddls = sorted((T3_ASSETS / "bddl_files/libero_t3").glob("*.bddl"))
        if not bddls:
            sys.exit(f"**{T3_ASSETS} に bddl が無い。**シムのイメージの中で動かすこと")
        jobs = [(b.stem, b, T3_ASSETS / "init_files/libero_t3") for b in bddls]

    out: dict[str, np.ndarray] = {}
    for stem, bddl, init_dir in jobs:
        if not bddl.exists():
            print(f"  - {stem}: bddl が無い（{bddl}）。飛ばす"); continue
        inits = init_dir / f"{stem}.pruned_init"
        if not inits.exists():
            inits = next(init_dir.glob(f"{stem}*"), None)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                 camera_names=list(CAMERAS),
                                 camera_heights=args.camera_size,
                                 camera_widths=args.camera_size)
        env.reset()
        if inits is not None:
            import torch  # LIBERO の init は torch の保存形式である
            states = torch.load(str(inits), weights_only=False)
            obs = env.set_init_state(states[args.init_index])
        else:
            obs = env.reset()
        for key in ("agentview_image", "robot0_eye_in_hand_image",
                    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            out[f"{stem}/{key}"] = np.asarray(obs[key])
        out[f"{stem}/instruction"] = np.asarray(instruction_of(bddl))
        # **物の位置も全部残す。**「どの物へ手を伸ばしたか」を推論だけで判定するのに使う
        # （行動列を積んだ変位の向きと、手先から各物への向きを比べる）。
        objects = [k[:-4] for k in obs
                   if k.endswith("_pos") and not k.startswith("robot0")]
        out[f"{stem}/objects"] = np.asarray(objects)
        for name in objects:
            out[f"{stem}/obj/{name}"] = np.asarray(obs[f"{name}_pos"])
        print(f"    物 {len(objects)} 個: {', '.join(objects)}")
        print(f"  {stem}\n    {instruction_of(bddl)}")
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"\n{len(jobs)} 課題を {args.output} へ書いた")


if __name__ == "__main__":
    main()
