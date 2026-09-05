#!/usr/bin/env python3
"""Track3 試作(006 やること1)・パス①: 時間反転プロトタイプ。

黄色い本タスク（配布 T3 「棚から黄色い本を下ろす」）について、
順方向デモ（tar `libero_combined_20hz` の task 92 =
`STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf`）を
このシミュレータで順再生し、最終状態を捕まえた上で、
逆方向（T3 の初期状態から反転 action を再生）を T3 の :goal 判定に通す。

実行先は parc2026-final 画像内（/workspace に LIBERO-plus が入っている）。
CPU 完結（描画不要）。GPU は不要。

action 列はホスト側（host_extract_actions.py）が npz として抽出し、
本スクリプトはそれを受け取るだけ（--actions-npz）。
2 段構成は run_reversal_probe.sh が回す。
"""

import argparse
import json
import os
import pickle
import zipfile

import numpy as np


# --------------------------------------------------------------------------
# 初期状態の読み込み（.pruned_init = zip の中の data.pkl = 平坦化シム状態のリスト）
# --------------------------------------------------------------------------

def load_pruned_init(path: str):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            pick = [n for n in names if n.endswith("data.pkl")]
            if not pick:
                raise SystemExit(f"no data.pkl inside {path}: {names}")
            blob = z.read(pick[0])
    else:
        blob = open(path, "rb").read()
    states = pickle.loads(blob)
    states = [np.asarray(s, dtype=np.float32) for s in states]
    return states


# --------------------------------------------------------------------------
# 反転 action の生成
# --------------------------------------------------------------------------

def build_reversed_actions(
    actions: np.ndarray, gripper_mode: str = "keep", skip_gripper: bool = False
) -> np.ndarray:
    """順方向 action 列 (T, 7) を時間反転する。

    データ規約: use_relative_actions=False（絶対 EEF 姿勢 6 次元 + gripper 1 次元）。
    絶対姿勢の反転は意味が通る（同じ空間経路を戻る）。
    gripper は開閉の絶対値なので、そのまま反転すると「掴む」が「放す」に
    ならない。gripper_mode で扱いを選ぶ:

      keep   — action_6 をそのまま反転列に乗せる（デフォルト・最初に試す）
      invert — action_6 を 1 - x で反転する（掴み/放しの切替を補正する）
      skip_gripper=True — action_6 を 0（掴んだまま）にする
    """
    rev = np.ascontiguousarray(actions[::-1].copy())
    if skip_gripper:
        rev[:, 6] = 0.0
    elif gripper_mode == "invert":
        rev[:, 6] = 1.0 - rev[:, 6]
    return rev


# --------------------------------------------------------------------------
# 環境の生成と再生
# --------------------------------------------------------------------------

def make_env(bddl_file: str, control_freq: int = 20):
    """評価と同じ ControlEnv を作る（scripted_demo._make_env と同じ経路）。"""
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=bddl_file,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
        control_freq=control_freq,
        horizon=2000,
        ignore_done=True,
    )


def replay(env, init_state, actions, max_steps=None):
    """初期状態から action 列を再生する。最終シム状態（平坦化）と step 数を返す。

    scripted_demo.reset_to と同じ手順:
    set_state_from_flattened → forward → qvel ゼロの空撃ち 10 ステップ。
    step は ControlEnv の内部 env に対して行う（ndarray action を受理する）。
    """
    inner = env.env
    sim = inner.sim
    sim.set_state_from_flattened(init_state)
    sim.forward()
    for _ in range(10):
        sim.data.qvel[:] = 0
        inner.step(np.zeros(7))
    n = len(actions) if max_steps is None else min(len(actions), max_steps)
    for i in range(n):
        inner.step(np.asarray(actions[i], dtype=np.float32))
    return sim.get_state().flatten(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--actions-npz",
        required=True,
        help="ホスト側で action 列を抽出した npz (host_extract_actions.py が作る)。"
        "キー: actions_<ep> (T,7) と length_<ep>。",
    )
    ap.add_argument("--forward-dddl", required=True)
    ap.add_argument("--forward-inits", required=True)
    ap.add_argument("--t3-dddl", required=True)
    ap.add_argument("--t3-inits", required=True)
    ap.add_argument("--out", default="/out")
    ap.add_argument(
        "--gripper-mode",
        choices=["keep", "invert"],
        default="keep",
        help="反転時の gripper 軸(action_6)の扱い。keep=そのまま、invert=1-x で反転。"
        "データは use_relative_actions=False（絶対姿勢）で、gripper は開閉の絶対値なので"
        "単純反転では意味が狂う可能性がある。両方試すのが安全。",
    )
    ap.add_argument(
        "--skip-gripper",
        action="store_true",
        help="反転再生時に action_6 を 0（掴んだまま）にする。"
        "『放す瞬間の位置をそのまま辿る』解釈。",
    )
    args = ap.parse_args()

    # 1) action 列の npz を 1 回だけ読み、候補エピソードを決定
    z = np.load(args.actions_npz)
    ep_ids = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("actions_"))
    print(f"candidate episodes: {ep_ids}", flush=True)

    # 2) 環境（順方向用 / T3 用）を各 1 回だけ作って使い回す
    print("creating forward env ...", flush=True)
    env_fwd = make_env(args.forward_dddl)
    print("forward env ready", flush=True)
    print("creating t3 env ...", flush=True)
    env_t3 = make_env(args.t3_dddl)
    print("t3 env ready", flush=True)

    fwd_inits = load_pruned_init(args.forward_inits)
    t3_inits = load_pruned_init(args.t3_inits)
    print(f"forward inits: {len(fwd_inits)}, t3 inits: {len(t3_inits)}", flush=True)

    results = []
    for ep in ep_ids:
        rec = {"episode_index": ep}
        print(f"=== episode {ep} ===", flush=True)

        actions = z[f"actions_{ep}"].astype(np.float32)
        rec["n_actions"] = len(actions)
        print(f"  actions: {actions.shape}", flush=True)

        # 2a) 順再生: task92 の初期状態から → 棚上面まで行くはず
        fwd_state, n_steps = replay(env_fwd, fwd_inits[0], actions)
        env_fwd_env_ok = bool(env_fwd.check_success())
        rec["forward"] = {
            "n_steps": n_steps,
            "success_at_end": env_fwd_env_ok,
        }
        # 最終シム状態をバイトで残す（後で .pruned_init に書き出せる）
        with open(os.path.join(args.out, f"fwd_final_ep{ep}.npz"), "wb") as f:
            np.savez(f, state=fwd_state)
        print(
            f"  forward replay: {n_steps} steps, success={env_fwd_env_ok}",
            flush=True,
        )
        if not env_fwd_env_ok:
            print("  !! forward demo itself does not reach its goal in our sim", flush=True)

        # 2b) 逆再生: T3 の初期状態（＝順方向デモのゴールに近い配置）から、
        #     action 列を反転して再生 → T3 の :goal（机上 left region）を満たすか
        rev_actions = build_reversed_actions(
            actions, gripper_mode=args.gripper_mode, skip_gripper=args.skip_gripper
        )
        t3_state, n_steps_r = replay(env_t3, t3_inits[0], rev_actions)
        t3_ok = bool(env_t3.check_success())
        rec["reverse"] = {
            "gripper_mode": args.gripper_mode,
            "skip_gripper": args.skip_gripper,
            "n_steps": n_steps_r,
            "success_at_end": t3_ok,
        }
        with open(os.path.join(args.out, f"rev_final_ep{ep}.npz"), "wb") as f:
            np.savez(f, state=t3_state)
        print(
            f"  reverse replay: {n_steps_r} steps, success={t3_ok} "
            f"(gripper={args.gripper_mode}, skip_gripper={args.skip_gripper})",
            flush=True,
        )
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)

    out_path = os.path.join(args.out, "reversal_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"wrote {out_path}", flush=True)

    # go/no-go サマリー
    ok = [r for r in results if r["reverse"]["success_at_end"]]
    print(
        f"\n=== SUMMARY: {len(ok)}/{len(results)} reversed replays satisfy T3 :goal ===",
        flush=True,
    )


if __name__ == "__main__":
    main()