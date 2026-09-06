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

def decode_gripper(actions: np.ndarray, states: np.ndarray | None = None) -> np.ndarray:
    """`action[6]` を robosuite の規約（-1=開く / +1=閉じる）へ直す。

    **`libero_combined_20hz` は gripper の符号化が 2 種類混ざっている。**
    実測（全 19,533 本を走査）:

      - エピソード 0〜5047（4,827 本）: `action[6] ∈ {0, 1}` で **1=開く・0=閉じる**
      - エピソード 5186〜19532（14,347 本）: `action[6] ∈ {-1, +1}` で
        **-1=開く・+1=閉じる**（robosuite そのまま）

    指の開き幅（`observation.state[6] - state[7]`）と突き合わせて確かめた。
    例（ep 4061・0/1 系）: f0-52 は `+1` で幅 0.078（開いている）、f53-132 は `0` で
    幅 0.024 まで閉じ、f133 以降 `+1` で 0.079 へ開く。
    例（ep 5195・±1 系）: f0 は `-1` で幅 0.078、f34 で `+1` にして 0.006 まで閉じる。

    **task 92（黄本の順方向）の 44 本は全部 0/1 系である。**
    生のまま robosuite へ流すと、`+1` が「閉じる」に、`0` が「そのまま」になり、
    **掴む・放すが完全に入れ替わる**（本を掴めないまま腕だけ動く）。

    変換: 0/1 系は `a → 1 - 2a`（1→-1 開く、0→+1 閉じる）。±1 系はそのまま。
    判定は「負の値があるか」で行い、`states` があれば開き幅と符号が合うか検算する。
    """
    out = np.array(actions, dtype=np.float32, copy=True)
    g = out[:, 6]
    if float(g.min()) < 0.0:
        encoding = "pm1"           # robosuite そのまま
    else:
        encoding = "zero_one"
        out[:, 6] = 1.0 - 2.0 * g

    if states is not None and len(states) == len(out):
        # 検算: 「閉じる指令(+1)」の直後に開き幅が縮んでいるはず
        width = np.asarray(states)[:, 6] - np.asarray(states)[:, 7]
        dw = np.diff(width)
        cmd = out[:-1, 6]
        moving = np.abs(dw) > 1e-4
        if moving.sum() >= 5:
            corr = float(np.corrcoef(cmd[moving], -dw[moving])[0, 1])
            if corr < 0:
                print(f"  !! gripper 復号({encoding}) が開き幅と逆相関 ({corr:+.3f})。"
                      f" 規約の判定を疑うこと", flush=True)
            else:
                print(f"  gripper 符号化={encoding}（開き幅との相関 {corr:+.3f}）",
                      flush=True)
    else:
        print(f"  gripper 符号化={encoding}", flush=True)
    return out


def build_reversed_actions(
    actions: np.ndarray, gripper_mode: str = "keep", skip_gripper: bool = False
) -> np.ndarray:
    """順方向 action 列 (T, 7) を時間反転する。

    **データ規約（コンテナで実測）: OSC_POSE・`control_delta=True`・
    `input_max/min=±1`・`output_max=[0.05,0.05,0.05,0.5,0.5,0.5]`。**
    つまり `action[:6]` は「**今の姿勢からの相対差分**を ±1 に正規化した指令値」で、
    絶対姿勢ではない（この docstring は以前 `use_relative_actions=False`＝絶対姿勢と
    書いていたが、実測と食い違っていた。[017](../../../docs/talk/inter_sessions/017_action_convention_reconciliation.md)）。

    相対差分なので、**同じ経路を戻るには順序を逆にするだけでは足りず、符号も反転する**。
    順方向が差分 d0,d1,...,d(T-1) で x0 → xT へ動くなら、戻る経路は
    -d(T-1), -d(T-2), ..., -d0 である。順序だけ逆にすると、同じ向きの動きを
    バラバラな順で撃つことになり、元の場所には戻らない。

    gripper（`action[6]`）は差分ではなく開閉の指令なので**符号は反転しない**。
    `decode_gripper` で robosuite 規約（-1=開く/+1=閉じる）に直してから渡すこと。

      keep   — 並べ替えるだけ（**既定・これが正しい**）。pick&place の開閉は
               「開く→閉じる→開く」で時間反転しても同じ形になり、掴む位置は
               順方向で放した位置＝物が今ある場所に来る。
      invert — `-x` で開閉を入れ替える（切り分け用。時間反転では通常不要）。
               以前は `1 - x` だったが、robosuite 規約の ±1 に対しては {0,2} を返し
               指令範囲の外だった。
      skip_gripper=True — `action[6]` を +1（掴んだまま）にする。

    開閉指令だけを姿勢 6 次元に対して前倒しするには [shift_gripper](#) を使う。
    """
    rev = np.ascontiguousarray(actions[::-1].copy())
    rev[:, :6] = -rev[:, :6]          # 相対差分なので符号を反転して経路を戻る
    if skip_gripper:
        rev[:, 6] = 1.0               # +1 = 閉じたまま（robosuite 規約）
    elif gripper_mode == "invert":
        rev[:, 6] = -rev[:, 6]
    return rev


def shift_gripper(gripper_cmd: np.ndarray, shift: int) -> np.ndarray:
    """gripper 指令列を `shift` フレームだけ**前倒し**する（負なら遅らせる）。

    **なぜ要るか。** 逆向き再生では、腕とグリッパの遅れが**どちらも打ち消されない**。

      - 腕: OSC は 1 指令で目標の 2 割ほどしか進まない。実測すると逆向き再生の eef は
        基準経路を**ちょうど 3 フレーム遅れて**なぞる（`eef[i]` を `ref[i-3]` と
        突き合わせると残差 中央 27〜30 mm → 3〜6 mm）。
      - 指: `PandaGripper.format_action` は
        `current_action += [-1,+1] * 0.01 * sign(a6)` を**サブステップ毎**に回す
        積分器で、1 指令あたり 0.25 しか進まない。全開↔全閉の 2.0 を渡るのに
        **8 指令 = 0.4 秒**かかる。

    遅れを時間反転すると「先読み」になるが、それは実現できない。つまり反転列では
    遅れが打ち消されず**二重に効く**。オフラインで指令を前倒しするのが、その
    実現できない先読みを与える唯一の手段である。効かせるべき量は
    （指のストローク）−（腕の遅れ）の向きの差なので、掃いて決める。

    端は最初／最後の値で埋める。`action[6]` は切り替え点 2 つだけの矩形波
    （開く→閉じる→開く）なので、端の値の延長が素直な外挿になる。
    """
    g = np.asarray(gripper_cmd, dtype=np.float32)
    idx = np.clip(np.arange(len(g)) + int(shift), 0, len(g) - 1)
    return g[idx]


def reverse_reference_path(states: np.ndarray) -> np.ndarray:
    """順方向の state 列から「逆向きに辿るべき基準経路」を作る。

    `observation.state` は `[eef_pos(3), 軸角(3), 指(2)]`。時間反転は**姿勢の列を
    逆順に並べるだけ**で意味が通る（014 の往復テストで残差 中央 0.6〜0.9 mm）。
    差分を作り直すのは再生時（`rollout_tracking`）で、そこで**今の姿勢から次の
    基準姿勢へ向かう差分**を撃つ。

    open-loop（`build_reversed_actions`）との違い:
    記録された指令を反転して撃つだけだと、OSC の遅れ（1 ステップで指令の 2 割しか
    到達しない）と接触の非可逆性のぶん**誤差が溜まり続ける**。基準経路を持って
    毎ステップ引き直せば、溜まらない。
    """
    return np.asarray(states, dtype=np.float32)[::-1].copy()


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
    # `set_state_from_flattened` は qpos/qvel しか戻さない。OSC コントローラの
    # 内部目標（goal_pos/goal_ori）は前の再生のまま残るので、先に reset を通す。
    # これが無いと、同じ初期状態から始めても前の再生に結果が引きずられる。
    env.reset()
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
        help="反転時の gripper 軸(action_6)の扱い。keep=並べ替えのみ（既定・これが正しい）、"
        "invert=-x で開閉を入れ替える（切り分け用）。"
        "action は相対差分の指令値なので、姿勢 6 次元は常に符号反転される。",
    )
    ap.add_argument(
        "--skip-gripper",
        action="store_true",
        help="反転再生時に action_6 を +1（掴んだまま）にする。"
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

        # 生データの gripper 符号化はエピソードによって違う。robosuite 規約へ直す。
        raw_actions = z[f"actions_{ep}"].astype(np.float32)
        states = z[f"states_{ep}"] if f"states_{ep}" in z.files else None
        actions = decode_gripper(raw_actions, states)
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