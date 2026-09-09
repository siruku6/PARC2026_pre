#!/usr/bin/env python3
"""**学習データそのものと同じ初期状態から方策を回し、軌跡が離れる場所を出す。**

## なぜこの測りが要るか

[3.1](../../../docs/plan/22_grasp_and_color_next_moves.md#31-邪魔な物を消した場面でロールアウト)（手 A）で
「迷いではない」が消え、[3.2](../../../docs/plan/22_grasp_and_color_next_moves.md#32-教師強制の再現テスト)（手 B）で
「学習が入っていない」が消えた。**残ったのは誤差の蓄積で、その内訳が 2 つある。**

| 内訳 | この測りでどう見えるか |
|---|---|
| **(c-1) データがそう教えた**（v1 は接近区間の手首回転が封筒を外れている） | **経路には乗っているが、掴む手前で進まなくなる。**回転の誤差が序盤で立ち上がる |
| **(c-2) ただ流されている** | **序盤から経路を外れ、離れ方が単調に増える** |

**この 2 つは「離れ始める場所」で分かれる。**だから場所を測る。

## なぜ比べられるのか —— **逆向きデータは T3 の配布初期状態から作った**

生成（`track3_reversal/gripper_shift_sweep.py`）は `t3_inits[i]` を復元して回している。
**評価が使う初期状態と同じものである。**npz の名前に `_i<番号>` が入っているので、
**その番号の初期状態からモデルを回せば、1 歩目から同じ状況で並べられる。**

**空撃ちの回数も揃っている**——生成側の `reset_to` も 10 歩の空撃ちを踏んでから
記録を始めるので、記録の 0 歩目とここの 0 歩目は同じ約束である。

## 出す数

| 数 | 定義 | 何が読めるか |
|---|---|---|
| `lag_mm[t]` | 同じ歩数の手先の距離 | **遅れ＋外れの合計** |
| `off_mm[t]` | **記録された経路のいちばん近い点**までの距離 | **経路を外れた量だけ**（遅れを除く） |
| **`spread_mm`** | **同じ初期状態の記録どうし**を同じ式で比べた `off_mm` の中央値 | **物差し。**モデルの外れ量がこれと同じ桁なら「外れている」とは言えない |
| `reach_mm[s]` | 記録の s 番目の点へ、モデルが**一度でも**近づけた最小距離 | 経路のどの区間を通れたか |
| `tracked_30mm` | `reach_mm` が 30 mm 以内で続いた**先頭からの割合** | **経路の何割まで辿れたか**（前へしか進まない読み） |
| `rot_deg[t]` | 同じ歩数の手首姿勢の角度差 | **(c-1) の回転が本当にずれているか** |
| `rot_cmd_p95` | モデルが出した回転指令の 95 パーセンタイル | 配布の封筒 ±0.375 を外れているか |
| `frame0_absdiff` | 記録の 1 コマ目とここの 1 コマ目の画素差 | **場面が本当に同じか**（据え置きは `qpos` に入らない） |

**`off_mm` と `tracked_30mm` を分けるのが要点である。**同じ歩数で比べるだけだと、
**「経路は正しいが遅い」と「経路を外れた」が混ざる。**

**「いちばん近い点の番号」を進み具合として使ってはいけない**（9/9 に踏んだ）。
逆向きの経路は**掴んで戻ってくる往復**なので、腕が出発点の近くに居るだけで
「経路の終点に居る」と出る（黄本で 5 歩目に 1.00 と出た）。
**先頭から続けて辿れた長さで測る。**

## 使い方

    bash solution/tools/probe/run_divergence.sh <重み> [出力ディレクトリ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _016_libero_env import SETTLE_STEPS, make_eval_env  # noqa: E402
from dump_t3_obs import CAMERAS, T3_ASSETS, instruction_of  # noqa: E402
from grasp_rollout import (load_inits, object_positions, parse_bddl,  # noqa: E402
                           targets_of)

# 配布データの 1 コマあたりの回転指令が収まる範囲（`reference/30` の 8 節）。
# v1 の逆向きデータはここを外して ±1 のクリップに張り付いていた。
ROT_ENVELOPE = 0.375
# `lag_mm` / `off_mm` がこれを超えた歩数を「離れた場所」として報告する。
DIVERGE_MM = (20.0, 50.0, 100.0)
# 「経路を辿れた」と見なす許容（`tracked_<N>mm`）。
TRACK_MM = (30.0, 50.0)


def quat_of(axisangle: np.ndarray) -> np.ndarray:
    from robosuite.utils import transform_utils as T

    return np.asarray(T.axisangle2quat(np.asarray(axisangle, dtype=np.float64)))


def angle_between_deg(aa_a: np.ndarray, aa_b: np.ndarray) -> float:
    """2 つの軸角の間の回転角（度）。**符号の違う同じ姿勢を同一視する。**"""
    qa, qb = quat_of(aa_a), quat_of(aa_b)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def first_over(values: np.ndarray, threshold: float) -> int | None:
    """`values` が初めて `threshold` を超える番号。超えなければ `None`。"""
    hits = np.flatnonzero(np.asarray(values) > threshold)
    return int(hits[0]) if hits.size else None


def tracked_prefix(reach_mm: np.ndarray, tolerance: float) -> float:
    """**先頭から続けて `tolerance` 以内で辿れた長さ**を、経路全体に対する割合で返す。

    往復する経路でも意味が壊れない読みである（`argmin` の番号は使えない）。
    """
    over = np.flatnonzero(np.asarray(reach_mm) > tolerance)
    if not over.size:
        return 1.0
    return float(over[0]) / max(len(reach_mm) - 1, 1)


def off_path_mm(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """`points` の各点から `path` のいちばん近い点までの距離（mm）。"""
    gaps = np.linalg.norm(np.asarray(points)[:, None, :]
                          - np.asarray(path)[None, :, :], axis=2)
    return gaps.min(axis=1) * 1000


def path_spread_mm(candidates: list) -> float | None:
    """**記録同士のばらつき。**同じ初期状態の記録を互いに比べた `off_mm` の中央値。

    **これが物差しである。**同じ課題の正しい経路どうしでも、元にした順方向デモが
    違えば復路が違うので 0 にはならない。**モデルの外れ量がこれと同じ桁なら
    「外れている」とは言えない。**1 本しか無ければ測れないので `None` を返す。
    """
    paths = [r["state"][:, :3].astype(np.float64) for r in candidates]
    if len(paths) < 2:
        return None
    values = [np.median(off_path_mm(paths[i], paths[j]))
              for i in range(len(paths)) for j in range(len(paths)) if i != j]
    return round(float(np.median(values)), 1)


def recorded_episodes(directory: Path, count: int) -> list[tuple[int, list[Path]]]:
    """`rev_ep<番号>_i<初期状態>_k+00.npz` を初期状態の番号でまとめる。

    **同じ初期状態の記録を全部返す。**元にした順方向デモが違うと復路が違うので、
    1 本だけと比べると「**別の本に乗っていた**」を「外れた」と数えてしまう。
    モデルを回すのは初期状態ごとに 1 回で、比べる相手を増やすだけである。
    """
    grouped: dict[int, list[Path]] = {}
    for path in sorted(directory.glob("rev_ep*_i*.npz")):
        marker = path.stem.split("_i")[-1].split("_")[0]
        if not marker.isdigit():
            continue
        grouped.setdefault(int(marker), []).append(path)
    return sorted(grouped.items())[:count]


def model_rollout(env, client, instruction: str, init_state, steps: int,
                  seed: int, target: str) -> dict:
    """1 本回して手先の軌跡・姿勢・出した行動を返す。**手 A と同じ約束で回す。**"""
    client.reset(instruction=instruction, seed=seed)
    env.reset()
    observation = env.set_init_state(init_state)
    for _ in range(SETTLE_STEPS):
        observation, _reward, _done, _info = env.step([0, 0, 0, 0, 0, 0, -1])

    frame0 = np.asarray(observation["agentview_image"])[::-1, ::-1].copy()
    eef, axisangle, actions = [], [], []
    target_dist_mm, done = [], False
    for _ in range(steps):
        action = np.asarray(client.get_action(observation), dtype=np.float64)
        actions.append(action)
        eef.append(np.asarray(observation["robot0_eef_pos"], dtype=np.float64))
        axisangle.append(eef_axisangle_of(observation))
        positions = object_positions(observation)
        target_dist_mm.append(
            float(np.linalg.norm(positions[target] - eef[-1]) * 1000)
            if target in positions else float("nan"))
        observation, _reward, done, _info = env.step(action.tolist())
        if done:
            break
    return {
        "eef": np.asarray(eef),
        "axisangle": np.asarray(axisangle),
        "actions": np.asarray(actions),
        "target_dist_mm": np.asarray(target_dist_mm),
        "frame0": frame0,
        "done": bool(done),
    }


def eef_axisangle_of(observation: dict) -> np.ndarray:
    """観測の手先姿勢を軸角で返す。**記録の `state[:, 3:6]` と同じ作り方。**"""
    from robosuite.utils import transform_utils as T

    quat = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
    return np.asarray(T.quat2axisangle(quat), dtype=np.float64)


def compare(recorded: dict, run: dict) -> dict:
    """記録とモデルの軌跡を並べて、離れる場所を出す。"""
    rec_eef = recorded["state"][:, :3].astype(np.float64)
    rec_aa = recorded["state"][:, 3:6].astype(np.float64)
    rec_actions = recorded["action"].astype(np.float64)
    model_eef, model_aa = run["eef"], run["axisangle"]

    overlap = min(len(rec_eef), len(model_eef))
    lag_mm = np.linalg.norm(model_eef[:overlap] - rec_eef[:overlap], axis=1) * 1000
    rot_deg = np.array([angle_between_deg(model_aa[i], rec_aa[i])
                        for i in range(overlap)])

    # **経路のいちばん近い点。**遅れを除いて「外れた量」だけを取り出す。
    gaps = np.linalg.norm(model_eef[:, None, :] - rec_eef[None, :, :], axis=2)
    off_mm = gaps.min(axis=1) * 1000
    # 記録の各点へ、モデルが**一度でも**近づけた距離。経路のどこで切れたかが出る。
    reach_mm = gaps.min(axis=0) * 1000

    head = max(int(round(0.1 * len(run["actions"]))), 1)   # 先頭 10%
    rot_cmd = np.abs(run["actions"][:, 3:6]).max(axis=1)
    rec_rot_cmd = np.abs(rec_actions[:, 3:6]).max(axis=1)
    rec_close = first_over(rec_actions[:, 6], 0.0)

    return {
        "n_recorded": int(len(rec_eef)),
        "n_model": int(len(model_eef)),
        "done": run["done"],
        "diverge_lag": {f"{mm:g}mm": first_over(lag_mm, mm) for mm in DIVERGE_MM},
        "diverge_off": {f"{mm:g}mm": first_over(off_mm, mm) for mm in DIVERGE_MM},
        "lag_mm_final": round(float(lag_mm[-1]), 1),
        "off_mm_median": round(float(np.median(off_mm)), 1),
        "off_mm_max": round(float(off_mm.max()), 1),
        # **経路の何割まで辿れたか。**1.0 なら最後まで辿れている。
        **{f"tracked_{mm:g}mm": round(tracked_prefix(reach_mm, mm), 3)
           for mm in TRACK_MM},
        "rot_deg_median": round(float(np.median(rot_deg)), 1),
        "rot_deg_head": round(float(np.median(rot_deg[:head])), 1),
        "rot_deg_max": round(float(rot_deg.max()), 1),
        "rot_cmd_p95": round(float(np.percentile(rot_cmd, 95)), 3),
        "rot_cmd_over_envelope": round(float((rot_cmd > ROT_ENVELOPE).mean()), 3),
        "rot_cmd_over_envelope_head": round(
            float((rot_cmd[:head] > ROT_ENVELOPE).mean()), 3),
        "recorded_rot_cmd_p95": round(float(np.percentile(rec_rot_cmd, 95)), 3),
        "recorded_close_step": rec_close,
        # **記録が指を閉じた瞬間の手先へ、モデルはどれだけ近づけたか。**
        # ここが遠いなら、掴む姿勢に一度も入っていない。
        "reach_at_close_mm": (None if rec_close is None
                              else round(float(reach_mm[rec_close]), 1)),
        "target_dist_min_mm": round(float(np.nanmin(run["target_dist_mm"])), 1),
        "grip_close_steps": int((run["actions"][:, 6] > 0).sum()),
        "traces": {"lag_mm": lag_mm, "off_mm": off_mm, "reach_mm": reach_mm,
                   "rot_deg": rot_deg},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://policy:8000")
    ap.add_argument("--out", type=Path, default=Path("/out/divergence.json"))
    ap.add_argument("--pair", action="append", default=[], metavar="STEM=DIR",
                    help="bddl の stem と、その課題の逆向き npz のディレクトリ")
    ap.add_argument("--episodes", type=int, default=2,
                    help="1 課題あたりに並べる本数（初期状態の番号ごとに 1 本）")
    ap.add_argument("--tail-steps", type=int, default=50,
                    help="記録の長さに足して回す歩数（**止まったのか遅いのかを見る**）")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--camera-size", type=int, default=256)
    args = ap.parse_args()

    if not args.pair:
        raise SystemExit("**--pair STEM=DIR を 1 つ以上渡すこと**")

    from pipeline.remote_policy import RemotePolicyClient

    client = RemotePolicyClient(args.server, timeout_sec=300.0)
    client.wait_for_server(timeout=900.0)

    results, traces = [], {}
    for pair in args.pair:
        stem, _, directory = pair.partition("=")
        bddl = T3_ASSETS / "bddl_files/libero_t3" / f"{stem}.bddl"
        if not bddl.is_file():
            raise SystemExit(f"**bddl が無い: {bddl}**")
        episodes = recorded_episodes(Path(directory), args.episodes)
        if not episodes:
            raise SystemExit(f"**{directory} に rev_ep*_i*.npz が無い**")

        target = targets_of(parse_bddl(bddl))[0]
        inits = load_inits(stem)
        # **指示文は 1 つしか要らない。**逆向きデータは T3 の env で録ったので、
        # 記録に入っている文は bddl の `:language` そのものである（9/9 に確かめた）。
        instruction = instruction_of(bddl)
        env = make_eval_env(str(bddl), camera_size=args.camera_size,
                            camera_names=list(CAMERAS))
        try:
            for init_index, paths in episodes:
                candidates = [np.load(path, allow_pickle=True) for path in paths]
                for recorded in candidates:
                    if str(recorded["instruction"]) != instruction:
                        print(f"  **記録の文が bddl と違う**"
                              f"\n    記録 {recorded['instruction']}"
                              f"\n    bddl {instruction}", flush=True)
                longest = max(int(r["steps"]) for r in candidates)
                steps = min(longest + args.tail_steps, args.max_steps)
                print(f"\n=== {stem}  init #{init_index}  記録 {len(paths)} 本"
                      f"（最長 {longest} コマ）→ {steps} 歩回す", flush=True)
                run = model_rollout(env, client, instruction,
                                    inits[init_index], steps, init_index, target)
                # **モデルにいちばん有利な本を代表にする。**どれかに乗っていれば
                # 「経路に乗っていた」と見なす。
                rows = [compare(recorded, run) for recorded in candidates]
                best = int(np.argmin([r["off_mm_median"] for r in rows]))
                row, recorded = rows[best], candidates[best]
                row["off_mm_median_all"] = [r["off_mm_median"] for r in rows]
                row["spread_mm"] = path_spread_mm(candidates)
                row["frame0_absdiff"] = round(float(np.abs(
                    run["frame0"].astype(np.int16)
                    - recorded["front"][0].astype(np.int16)).mean()), 2)
                traces[f"{stem}__i{init_index}"] = row.pop("traces")
                row.update({"task": stem, "init_index": init_index,
                            "recorded_npz": paths[best].name, "target": target})
                results.append(row)
                print(f"     経路を外れた量 中央 {row['off_mm_median']:6.1f} mm"
                      f"（記録同士は {row['spread_mm']}）"
                      f" / 最大 {row['off_mm_max']:6.1f}"
                      f"   50 mm 外れた歩数 {row['diverge_off']['50mm']}"
                      f"   辿れた割合 30mm {row['tracked_30mm']:.2f}"
                      f" / 50mm {row['tracked_50mm']:.2f}", flush=True)
                print(f"     手首の角度差 中央 {row['rot_deg_median']:5.1f}°"
                      f" / 先頭 10% {row['rot_deg_head']:5.1f}°"
                      f"   回転指令 p95 {row['rot_cmd_p95']:.3f}"
                      f"（記録 {row['recorded_rot_cmd_p95']:.3f}"
                      f"・封筒 {ROT_ENVELOPE}）", flush=True)
                print(f"     対象への最接近 {row['target_dist_min_mm']:6.1f} mm"
                      f"   掴む瞬間の姿勢まで {row['reach_at_close_mm']} mm"
                      f"   閉じる {row['grip_close_steps']:3d}"
                      f"   1 コマ目の画素差 {row['frame0_absdiff']:5.2f}",
                      flush=True)
        finally:
            env.close()

    print()
    print("=" * 108)
    print(f"  {'課題':30}{'init':>5}{'外れ中央':>10}{'記録同士':>10}"
          f"{'50mm歩':>8}{'辿れた30':>10}{'角度中央':>9}{'最接近':>8}{'掴む姿勢':>10}")
    for row in results:
        print(f"  {row['task'][:30]:30}{row['init_index']:>5}"
              f"{row['off_mm_median']:>10.1f}{str(row['spread_mm']):>10}"
              f"{str(row['diverge_off']['50mm']):>8}{row['tracked_30mm']:>10.2f}"
              f"{row['rot_deg_median']:>9.1f}{row['target_dist_min_mm']:>8.1f}"
              f"{str(row['reach_at_close_mm']):>10}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    np.savez(args.out.with_suffix(".npz"),
             **{f"{key}__{name}": value
                for key, row in traces.items() for name, value in row.items()})
    print(f"\n  書いた: {args.out} と {args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
