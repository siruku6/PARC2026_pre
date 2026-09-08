#!/usr/bin/env python3
"""邪魔な物を消した場面で方策を 300 歩回し、**掴めるかどうか**を測る。

## なぜこの測りが要るか

動画では「違う物の上で漂う」が T3 の 4 課題すべてで起きている。
**これが「候補の間で迷っている」なのか「掴む力を失った」なのかで、次の手が正反対になる。**

**変えるのは候補物体の数だけである。**場面・指示文・カメラ・乱数はそのままにして、
**対象と競合する物だけ**を bddl から消す。

| 結果 | 読み | 次に打つ手 |
|---|---|---|
| 掴める | **迷っていた。**把持は残っている | 色で見分けるデータに全振りする |
| 掴まないが近づく | **接近の最後か把持そのもの**が壊れている | 逆向きデータの掴み方を見直す |
| 近づかない | **場面の理解が届いていない** | `base` からのやり直しの値打ちが上がる |

設計は `docs/plan/22_grasp_and_color_next_moves.md` の 3.1 節。

## 何を消し、何を残すか —— **bddl の述語から決める**（名前の当て推量をしない）

| 残すもの | なぜ |
|---|---|
| **`:goal` が名指しする物**（＝運ぶ対象） | これが無いと課題が成立しない。**two mugs は 2 つある** |
| **対象を支えている物**（`:init` の `(on 対象 X)` の X） | 消すと対象が空中から落ちる |
| **目的地の領域を持つ物** | goal の領域の `target` を引く |
| **指示文が名指しする物**（`--keep-landmarks` のとき） | 缶の「ケチャップの前に置く」・two mugs の「赤いマグの左」 |

**残りが「競合」である。**それだけを消す。

**既定では目印も消す。**T3 の 3 課題は指示文が競合そのものを名指ししている
（赤マグの「**他の 2 つのマグ**の後ろ」・黄本の「**黄色い**本」）ので、
**目印を残すと 1 つも消えず、実験が成立しない。**
消した物が指示文に出ていたら**警告を出す**——判定は `lift_mm`（掴んだか）で行い
`success` では行わないので、目印が欠けても掴む段の測りは成り立つ。

**消したあと、env が本当にその物だけになっているかを確かめる。**
LIBERO の `:objects` は**同じ型の物を 1 行にまとめて書く**
（`plate_1 plate_2 - plate`）ので、行ごとに消すと**片方だけ残る。**
**9/9 にこれを踏んだ**——`yellow_book_1` が残ったまま「消した」と表示していた。

## init ファイルは要らない。ただし乱数は評価と揃える

`make_eval_env`（`_016_libero_env.py`）で開く。**据え置き（棚・caddy・コンロ）の位置は
`qpos` に入らず `reset()` の乱数で決まる**ので、キットと同じ `seed` で開かないと
別の場面を測ることになる。**配布 init は物の数が変わると使えない**（`set_state_from_flattened`
が落ちる）ので、`reset()` に `:init` 領域からサンプルさせる。

## 掴めたかの判定

**`success` では判定しない。**目標は「机の決まった場所へ置く」なので、
掴めても置き場所を外すと 0 になる。**掴む段だけを切り出す。**

| 出す数 | 判定 |
|---|---|
| `lift_mm` | 対象の z が初期値からどれだけ上がったか。**30 mm 超えたら「持ち上げた」** |
| `target_moved_mm` | 対象が水平にどれだけ動いたか |
| `others_moved_mm` | **`:goal` に出てこない物**が動いた最大量。**キットと同じ 3 軸の L1**（1 mm 超で失格） |
| `grip_close_steps` | 指を閉じる指令を出した歩数 |
| `min_dist_mm` | 手先が対象へいちばん近づいた距離。**掴みに行ったかどうか** |

## 使い方

    bash solution/tools/probe/run_grasp_rollout.sh <重み> [出力ディレクトリ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dump_t3_obs import CAMERAS, T3_ASSETS, instruction_of  # noqa: E402

LIFT_THRESHOLD_MM = 30.0
# **`:goal` で「物を置く」ことを表す述語。**LIBERO のパーサは小文字で返す。
# 一覧は `track3_reversal/gripper_shift_sweep.py` の `PLACEMENT` と同じ。
PLACEMENT = {"on", "in", "at", "infrontof", "behind",
             "totheleftof", "totherightof", "ontopof"}


def parse_bddl(path: Path) -> dict:
    """LIBERO 自身のパーサで bddl を読む。**自前の正規表現を持たない。**"""
    from libero.libero.envs import bddl_utils

    return bddl_utils.robosuite_parse_problem(str(path))


def names_of(problem: dict) -> list[str]:
    """`:objects` に宣言されている物の名前を全部返す（型ではなく実体の名前）。"""
    return [name for names in problem["objects"].values() for name in names]


def targets_of(problem: dict) -> list[str]:
    """運ぶ対象を `:goal` の「置く系」述語から取る。**1 つとは限らない。**"""
    declared = set(names_of(problem))
    return [state[1] for state in problem["goal_state"]
            if len(state) == 3 and state[0] in PLACEMENT and state[1] in declared]


def keep_set(problem: dict, targets: list[str],
             keep_landmarks: bool = False) -> set[str]:
    """消してはいけない物の集合。**述語から決める。**

    `keep_landmarks` を立てると、指示文が名指しする物も残す。**既定は消す**——
    T3 の 3 課題は指示文が競合そのものを名指ししていて、残すと 1 つも消えない。
    """
    declared = set(names_of(problem))
    keep = set(targets)

    # 対象を支えている物（`(on 対象 X)` の X が物なら残す）と、目的地の領域を持つ物。
    for state in list(problem["initial_state"]) + list(problem["goal_state"]):
        if len(state) != 3 or state[1] not in keep:
            continue
        place = state[2]
        if place in declared:
            keep.add(place)
        owner = (problem["regions"].get(place) or {}).get("target")
        if owner in declared:
            keep.add(owner)

    if keep_landmarks:
        for name in declared:
            if named_in_instruction(problem, name):
                keep.add(name)
    return keep


def named_in_instruction(problem: dict, name: str) -> bool:
    """物の名前の語が指示文に出てくるか（`red_coffee_mug_1` → `red`/`coffee`/`mug`）。"""
    words = {word.strip(",.").lower() for word in problem["language_instruction"]}
    return bool(set(name.rsplit("_", 1)[0].split("_")) & words)


def strip_bddl(source: Path, destination: Path, keep: set[str]) -> list[str]:
    """`keep` 以外の物を消した bddl を書き、消した名前を返す。

    **同じ型の物が 1 行に並ぶ宣言**（`plate_1 plate_2 - plate`）を扱うため、
    行を消すのではなく**行の中の名前を選び直す。**
    """
    text = source.read_text()
    removed = [name for name in names_of(parse_bddl(source)) if name not in keep]
    dropped = set(removed)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        words = stripped.split()
        # `:init` / `:goal` の述語。第 1 引数が消す物なら行ごと落とす。
        if len(words) >= 2 and words[0].startswith("(") and words[1] in dropped:
            continue
        # `:objects` の宣言。**残す名前だけを書き直す。**
        if " - " in stripped and not stripped.startswith("("):
            names, _, kind = stripped.partition(" - ")
            survivors = [n for n in names.split() if n not in dropped]
            if not survivors:
                continue
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}{' '.join(survivors)} - {kind}")
            continue
        lines.append(line)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return removed


def object_positions(observation: dict) -> dict[str, np.ndarray]:
    """物の位置だけを取る。**`*_to_robot0_eef_pos` は相対ベクトルなので外す。**"""
    return {key[:-4]: np.asarray(observation[key]).copy()
            for key in observation
            if key.endswith("_pos") and not key.startswith("robot0")
            and not key.endswith("_to_robot0_eef_pos")}


def displacement_mm(now: np.ndarray, start: np.ndarray) -> float:
    """キットの衝突判定と同じ **3 軸の L1**（`pipeline/rollout.py`）。"""
    return float(np.abs(now - start).sum() * 1000)


def rollout(env, client, instruction: str, targets: list[str],
            steps: int, episode: int) -> list[dict]:
    """1 本回して、対象ごとに掴む段の数を返す。"""
    client.reset(instruction=instruction, seed=episode)
    observation = env.reset()
    start = object_positions(observation)
    tracked = [name for name in targets if name in start]

    lift_mm = {name: 0.0 for name in tracked}
    min_dist_mm = {name: float("inf") for name in tracked}
    others_mm = 0.0
    grip_close = 0
    done = False
    for _ in range(steps):
        action = client.get_action(observation)
        if action[6] > 0:
            grip_close += 1
        observation, _reward, done, _info = env.step(action.tolist())

        eef = np.asarray(observation["robot0_eef_pos"])
        now = object_positions(observation)
        for name in tracked:
            lift_mm[name] = max(lift_mm[name],
                                float((now[name][2] - start[name][2]) * 1000))
            min_dist_mm[name] = min(min_dist_mm[name],
                                    float(np.linalg.norm(now[name] - eef) * 1000))
        for name, origin in start.items():
            if name in targets:
                continue
            others_mm = max(others_mm, displacement_mm(now[name], origin))
        if done:
            break

    final = object_positions(observation)
    return [{
        "target": name,
        "lift_mm": round(lift_mm[name], 1),
        "target_moved_mm": round(displacement_mm(final[name], start[name]), 1),
        "others_moved_mm": round(others_mm, 1),
        "grip_close_steps": grip_close,
        "min_dist_mm": round(min_dist_mm[name], 1),
        "done": bool(done),
        "lifted": bool(lift_mm[name] > LIFT_THRESHOLD_MM),
    } for name in tracked]


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
                    help="full=配布のまま／stripped=競合する物を消す")
    ap.add_argument("--keep-landmarks", action="store_true",
                    help="指示文が名指しする物を残す（**既定は消す。**残すと T3 では"
                         "1 つも消えない）")
    args = ap.parse_args()

    from pipeline.remote_policy import RemotePolicyClient

    from _016_libero_env import make_eval_env

    bddls = sorted((T3_ASSETS / "bddl_files/libero_t3").glob("*.bddl"))
    if args.tasks:
        bddls = [b for b in bddls if b.stem in args.tasks]
    if not bddls:
        raise SystemExit(f"**{T3_ASSETS} に bddl が無い。**シムのイメージの中で動かすこと")

    client = RemotePolicyClient(args.server, timeout_sec=300.0)
    client.wait_for_server(timeout=900.0)

    results = []
    for bddl in bddls:
        problem = parse_bddl(bddl)
        targets = targets_of(problem)
        keep = keep_set(problem, targets, args.keep_landmarks)
        instruction = instruction_of(bddl)

        for mode in args.modes:
            if mode == "stripped":
                path = Path("/out/stripped") / bddl.name
                removed = strip_bddl(bddl, path, keep)
                landmarks = [n for n in removed if named_in_instruction(problem, n)]
                print(f"\n=== {bddl.stem}  [{mode}]"
                      f"\n    残す {sorted(keep)}\n    消す {removed}")
                if landmarks:
                    print(f"    **指示文が名指ししている物を消した: {landmarks}**"
                          f"（判定は lift_mm で行うので掴む段の測りは成り立つ）")
                if not removed:
                    print("    **1 つも消えていない。**full と同じ条件である")
            else:
                path = bddl
                print(f"\n=== {bddl.stem}  [{mode}]  物 {len(names_of(problem))} 個")
            print(f"    対象 {targets}\n    {instruction}")

            env = make_eval_env(str(path), camera_size=args.camera_size,
                                camera_names=list(CAMERAS))
            try:
                # **消したはずの物が env に残っていないかを確かめる。**
                # 宣言が 1 行にまとまっていると、行ごとに消すと片方だけ残る（9/9 に踏んだ）。
                present = set(object_positions(env.reset()))
                if mode == "stripped" and not present <= keep:
                    raise SystemExit(f"**消し損ねた: {sorted(present - keep)}**"
                                     f"（{path} を見ること）")
                for episode in range(args.episodes):
                    for row in rollout(env, client, instruction, targets,
                                       args.steps, episode):
                        row.update({"task": bddl.stem, "mode": mode, "episode": episode})
                        results.append(row)
                        print(f"    ep{episode} {row['target']:22s}"
                              f" 持ち上げ {row['lift_mm']:6.1f} mm"
                              f"  最接近 {row['min_dist_mm']:6.1f}"
                              f"  閉じる {row['grip_close_steps']:3d}"
                              f"  他が動いた {row['others_moved_mm']:6.1f} mm"
                              f"  done {row['done']}"
                              f"  {'**持ち上げた**' if row['lifted'] else ''}", flush=True)
            finally:
                env.close()

    print()
    print("=" * 100)
    print(f"  {'課題':40}{'配置':>10}{'対象':>24}{'持ち上げ':>10}{'最接近':>9}{'閉じる':>8}")
    for row in results:
        print(f"  {row['task'][:40]:40}{row['mode']:>10}{row['target']:>24}"
              f"{row['lift_mm']:>10.1f}{row['min_dist_mm']:>9.1f}"
              f"{row['grip_close_steps']:>8}")
    lifted = sum(1 for r in results if r["lifted"])
    print(f"\n  持ち上げた: {lifted} / {len(results)} 本")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"  書いた: {args.out}")


if __name__ == "__main__":
    main()
