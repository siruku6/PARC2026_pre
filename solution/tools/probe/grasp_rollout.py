#!/usr/bin/env python3
"""邪魔な物を消した場面で方策を 300 歩回し、**掴めるかどうか**を測る。

## なぜこの測りが要るか

動画では「違う物の上で漂う」が T3 の 4 課題すべてで起きている。
**これが「候補の間で迷っている」なのか「掴む力を失った」なのかで、次の手が正反対になる。**

**変えるのは候補物体の数だけである。**場面・指示文・カメラはそのままにして、
`obj_of_interest` 以外の物を bddl から消す。

| 結果 | 読み | 次に打つ手 |
|---|---|---|
| 掴める | **迷っていた。**把持は残っている | 色で見分けるデータに全振りする |
| 掴まないが近づく | **接近の最後か把持そのもの**が壊れている | 逆向きデータの掴み方を見直す |
| 近づかない | **場面の理解が届いていない** | `base` からのやり直しの値打ちが上がる |

設計は `docs/plan/22_grasp_and_color_next_moves.md` の 3.1 節。

## init ファイルは要らない

`OffScreenRenderEnv(bddl_file_name=...)` は **bddl だけで env を作れる**
（`env.reset()` が `:init` 領域から位置をサンプルする）。
**配布 init を使うと物の数が合わず `set_state_from_flattened` が落ちる**ので、
物を消す実験ではこちらを使う。

## 掴めたかの判定

**`success` では判定しない。**目標は「机の決まった場所へ置く」なので、
掴めても置き場所を外すと 0 になる。**掴む段だけを切り出す。**

| 出す数 | 判定 |
|---|---|
| `lift_mm` | 対象の z が初期値からどれだけ上がったか。**30 mm 超えたら「持ち上げた」** |
| `target_moved_mm` | 対象が水平にどれだけ動いたか |
| `others_moved_mm` | **対象以外**が動いた最大量（1 mm 超で採点は失格になる） |
| `grip_close_steps` | 指を閉じる指令を出した歩数 |
| `min_dist_mm` | 手先が対象へいちばん近づいた距離。**掴みに行ったかどうか** |

## 使い方

    bash solution/tools/probe/run_grasp_rollout.sh <重み> [出力ディレクトリ]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

T3_ASSETS = Path("/workspace/compe/t3/assets")
CAMERAS = ("agentview", "robot0_eye_in_hand")
OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image",
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
LIFT_THRESHOLD_MM = 30.0


def instruction_of(bddl: Path) -> str:
    text = bddl.read_text()
    match = re.search(r"\(:language\s+(.*?)\)\s*\n", text, re.S)
    return " ".join(match.group(1).split()) if match else ""


def objects_of(text: str) -> list[str]:
    block = re.search(r"\(:objects(.*?)\n  \)", text, re.S)
    names = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        left = line.split("-")[0]
        names.extend(left.split())
    return names


def interest_of(text: str) -> list[str]:
    block = re.search(r"\(:obj_of_interest(.*?)\n  \)", text, re.S)
    return block.group(1).split() if block else []


def strip_bddl(source: Path, destination: Path) -> tuple[list[str], list[str]]:
    """`obj_of_interest` 以外の物を消した bddl を書く。

    **消すのは `(:objects)` の宣言と `(:init)` の述語だけ**である。
    領域の宣言は残しても害が無い（使われないだけ）ので触らない。
    """
    text = source.read_text()
    keep = set(interest_of(text))
    removed = [name for name in objects_of(text) if name not in keep]

    def drop_from_objects(match: re.Match) -> str:
        lines = []
        for line in match.group(1).splitlines():
            names = line.strip().split("-")[0].split()
            if names and all(n in removed for n in names):
                continue
            lines.append(line)
        return "(:objects" + "\n".join(lines) + "\n  )"

    text = re.sub(r"\(:objects(.*?)\n  \)", drop_from_objects, text, flags=re.S)
    kept_init = [line for line in text.splitlines()
                 if not (line.strip().startswith("(On ")
                         and line.strip().split()[1] in removed)]
    text = "\n".join(kept_init)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return sorted(keep), removed


def horizontal_mm(now: np.ndarray, start: np.ndarray) -> float:
    return float(np.linalg.norm(now[:2] - start[:2]) * 1000)


def rollout(env, server: str, instruction: str, target: str,
            steps: int, session) -> dict:
    """1 本回して、掴む段の数を返す。"""
    import msgpack

    session.post(f"{server}/reset", json={"instruction": instruction}, timeout=60)
    observation = env.reset()
    start = {name[:-4]: np.asarray(observation[name]).copy()
             for name in observation if name.endswith("_pos")
             and not name.startswith("robot0")}

    lift_mm = 0.0
    others_mm = 0.0
    grip_close = 0
    min_dist_mm = float("inf")
    done = False
    for _ in range(steps):
        payload = {key: {"data": np.ascontiguousarray(observation[key]).tobytes(),
                         "dtype": str(np.asarray(observation[key]).dtype),
                         "shape": list(np.asarray(observation[key]).shape)}
                   for key in OBS_KEYS}
        reply = session.post(f"{server}/act",
                             data=msgpack.packb(payload, use_bin_type=True),
                             headers={"Content-Type": "application/x-msgpack"},
                             timeout=300)
        action = np.frombuffer(msgpack.unpackb(reply.content, raw=False)["data"],
                               dtype=np.float32).copy()
        if action[6] > 0:
            grip_close += 1
        observation, _reward, done, _info = env.step(action.tolist())

        eef = np.asarray(observation["robot0_eef_pos"])
        if f"{target}_pos" in observation:
            now = np.asarray(observation[f"{target}_pos"])
            lift_mm = max(lift_mm, float((now[2] - start[target][2]) * 1000))
            min_dist_mm = min(min_dist_mm, float(np.linalg.norm(now - eef) * 1000))
        for name, origin in start.items():
            if name == target:
                continue
            others_mm = max(others_mm, horizontal_mm(
                np.asarray(observation[f"{name}_pos"]), origin))
        if done:
            break

    target_now = np.asarray(observation.get(f"{target}_pos", start.get(target, np.zeros(3))))
    return {
        "lift_mm": round(lift_mm, 1),
        "target_moved_mm": round(horizontal_mm(target_now, start[target]), 1)
        if target in start else None,
        "others_moved_mm": round(others_mm, 1),
        "grip_close_steps": grip_close,
        "min_dist_mm": round(min_dist_mm, 1) if min_dist_mm < float("inf") else None,
        "done": bool(done),
        "lifted": bool(lift_mm > LIFT_THRESHOLD_MM),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://policy:8000")
    ap.add_argument("--out", type=Path, default=Path("/out/grasp_rollout.json"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--camera-size", type=int, default=256)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="bddl の stem。省略すると 4 課題すべて")
    ap.add_argument("--modes", nargs="*", default=["full", "stripped"],
                    choices=("full", "stripped"),
                    help="full=配布のまま／stripped=邪魔な物を消す")
    args = ap.parse_args()

    import requests
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    bddls = sorted((T3_ASSETS / "bddl_files/libero_t3").glob("*.bddl"))
    if args.tasks:
        bddls = [b for b in bddls if b.stem in args.tasks]
    if not bddls:
        raise SystemExit(f"**{T3_ASSETS} に bddl が無い。**シムのイメージの中で動かすこと")

    session = requests.Session()
    results = []
    for bddl in bddls:
        text = bddl.read_text()
        interest = interest_of(text)
        # **対象は `obj_of_interest` のうち、置き場所（皿・トレイ）でないもの。**
        targets = [n for n in interest
                   if not n.startswith(("plate", "wooden_tray", "basket"))]
        target = targets[0] if targets else interest[0]
        instruction = instruction_of(bddl)

        for mode in args.modes:
            if mode == "stripped":
                path = Path("/out/stripped") / bddl.name
                keep, removed = strip_bddl(bddl, path)
                print(f"\n=== {bddl.stem}  [{mode}]  残す {keep} / 消す {removed}")
            else:
                path = bddl
                print(f"\n=== {bddl.stem}  [{mode}]  物 {len(objects_of(text))} 個")
            print(f"    対象 {target}\n    {instruction}")

            env = OffScreenRenderEnv(bddl_file_name=str(path),
                                     camera_names=list(CAMERAS),
                                     camera_heights=args.camera_size,
                                     camera_widths=args.camera_size)
            try:
                for episode in range(args.episodes):
                    row = rollout(env, args.server, instruction, target,
                                  args.steps, session)
                    row.update({"task": bddl.stem, "mode": mode,
                                "episode": episode, "target": target})
                    results.append(row)
                    print(f"    ep{episode}: 持ち上げ {row['lift_mm']:6.1f} mm"
                          f"  最接近 {row['min_dist_mm']}"
                          f"  閉じる {row['grip_close_steps']:3d}"
                          f"  他が動いた {row['others_moved_mm']:6.1f} mm"
                          f"  done {row['done']}"
                          f"  {'**持ち上げた**' if row['lifted'] else ''}", flush=True)
            finally:
                env.close()

    print()
    print("=" * 96)
    print(f"  {'課題':44}{'配置':>10}{'持ち上げ':>10}{'最接近':>9}{'閉じる':>8}")
    for row in results:
        print(f"  {row['task'][:44]:44}{row['mode']:>10}"
              f"{row['lift_mm']:>10.1f}{row['min_dist_mm'] or -1:>9.1f}"
              f"{row['grip_close_steps']:>8}")
    lifted = sum(1 for r in results if r["lifted"])
    print(f"\n  持ち上げた: {lifted} / {len(results)} 本")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"  書いた: {args.out}")


if __name__ == "__main__":
    main()
