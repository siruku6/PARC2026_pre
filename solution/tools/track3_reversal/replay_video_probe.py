#!/usr/bin/env python3
"""017: 順再生と逆再生を**動画付きで**回し、フレーム毎のずれを出す。

[reversal_probe.py](reversal_probe.py) は成否しか返さないので、
「逆向きに腕を振ったのか、順方向をやり直したのか」「本を運んだのか弾き落としたのか」
が読めない。ここは同じ再生を**描画付き**で回し、次の 3 つを残す:

  1. `<出力>/videos/*.mp4` — 順再生・逆再生の動画（agentview）
  2. `<出力>/replay_metrics.json` — フレーム毎の eef_pos ずれ（基準はデータの
     `observation.state[:, :3]`）と、発散し始めたフレーム番号
  3. 標準出力の表

**逆向き action の作り方は `reversal_probe.build_reversed_actions` をそのまま呼ぶ。**
そうしておけば、この 1 か所を直すだけで前後の動画が比べられる。

初期状態は `.pruned_init` の**全候補から、デモの 1 フレーム目の eef_pos に
いちばん近いものを選ぶ**。デモは特定の初期状態から録られているので、`[0]` 決め打ちだと
「初期状態不一致」で序盤から発散し、規約の話と混ざる（006 の候補 (c)）。

用法（コンテナ内。`/app` に track3_reversal を丸ごと mount する）:
    python /app/replay_video_probe.py --ep-npz /out/ep_data.npz \\
        --forward-bddl <task92.bddl> --forward-inits <task92.pruned_init> \\
        --t3-bddl <t3.bddl> --t3-inits <t3.pruned_init> --out /out
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from reversal_probe import (build_reversed_actions, decode_gripper,
                            load_pruned_init, reverse_reference_path)

CAMERA = os.environ.get("PROBE_CAMERA", "agentview")
SETTLE_STEPS = 10          # reversal_probe.replay と同じ空撃ち回数


def make_render_env(bddl_file: str, control_freq: int = 20, camera_size: int = 256):
    """`reversal_probe.make_env` と同じ物理で、描画だけ足した env。

    `use_camera_obs=True` は観測に画を足すだけで物理には触れないので、
    再生の結果は `reversal_probe` と同じになる。
    """
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=[CAMERA],
        camera_heights=camera_size,
        camera_widths=camera_size,
        control_freq=control_freq,
        horizon=100000,
        ignore_done=True,
    )


def fixture_snapshot(env):
    """据え置き(fixture)の配置（位置と向き）を写し取る。`reset_to` に渡して固定する。

    棚・机などの fixture の位置は `qpos` ではなく **`model.body_pos`** に入るので
    `set_state_from_flattened` では復らず、**`reset()` のたびに引き直される**。
    `reset()` の前後で model の数値フィールドを全件比較すると、変わるのは
    `body_pos` だけで、振れ幅は最大 18.7 mm あった。

    **比較するときは必ず固定すること。**これを渡さないと、同じ初期状態・同じ
    action を流しても本の着地が 2 か所に割れる（実測 35 mm）。固定すれば
    同一プロセス内の繰り返しはビット一致する。

    比べるときの注意: `model` は `hard_reset=True` のたびに**作り直される**ので、
    写し取った直後のオブジェクトを持ち回して比べても差は出ない（1 度これで
    「変化なし」と誤判定した）。毎回 `env.env.sim.model` を取り直すこと。

    なお**プロセスをまたいで揃えるには `env.seed()` も要る**。固定するのは
    「1 回目の reset が引いた配置」なので、その 1 回目が run ごとに違えば
    固定先も違う。016 の [_016_libero_env.py](../probe/_016_libero_env.py) と
    同じ `seed=42` を使う。
    """
    sim = env.env.sim
    # 位置だけでなく**向き**も引き直される。片方だけ固定すると差が残る。
    return (np.asarray(sim.model.body_pos).copy(),
            np.asarray(sim.model.body_quat).copy())


def reset_to(env, init_state, body_pos=None):
    """初期状態へ入れる。**先に `env.reset()` を通すこと。**

    `set_state_from_flattened` は qpos/qvel しか戻さない。OSC コントローラが持つ
    内部の目標姿勢（`goal_pos` / `goal_ori`）は**前の再生の続きのまま残る**ので、
    reset を挟まないと「同じ初期状態から始めたのに前回の再生に依存して結果が変わる」。
    実際、これを入れる前は同じ初期状態選びが run ごとに違う番号を返していた。

    `body_pos` に [fixture_snapshot](#) の戻り値を渡すと、`reset()` が引き直した
    fixture の配置を上書きして固定する。掃引のように**条件だけを変えて比べる**
    ときは必ず渡すこと。
    """
    env.reset()                      # コントローラの内部状態まで初期化する
    inner = env.env
    sim = inner.sim
    if body_pos is not None:
        # reset が引き直した fixture の姿勢を戻す
        sim.model.body_pos[:], sim.model.body_quat[:] = body_pos
    sim.set_state_from_flattened(np.asarray(init_state, dtype=np.float32))
    sim.forward()
    obs = None
    for _ in range(SETTLE_STEPS):
        sim.data.qvel[:] = 0
        obs, _, _, _ = inner.step(np.zeros(7))
    return obs


def pick_init(env, inits, ref_eef_pos, body_pos=None):
    """デモ 1 フレーム目の eef_pos にいちばん近い初期状態を選ぶ。

    デモは特定の初期状態から録られている。`inits[0]` 決め打ちで再生すると
    「初期状態が違うから序盤で発散した」のか「action の規約が違う」のかが
    分離できない（006 の候補 (c) と (b) の混線）。
    """
    best, best_d, dists = None, None, []
    for i, st in enumerate(inits):
        obs = reset_to(env, st, body_pos)
        d = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - ref_eef_pos))
        dists.append(d)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best, best_d, dists


def rollout(env, init_state, actions, video_path=None, ref_states=None,
            body_pos=None, track_obj=None):
    """初期状態から action 列を再生する。動画と、フレーム毎の eef_pos を返す。

    `track_obj` に物体名を渡すと、その姿勢も毎コマ残す（`obj_pos` / `obj_quat`）。
    逆向きの掴む姿勢を [grasp_pose_from_forward](#) で作るのに要る。
    """
    from robosuite.utils import transform_utils as T

    obs = reset_to(env, init_state, body_pos)
    frames, eef, aa = [], [], []

    obj_p, obj_q = [], []

    def record():
        if video_path is not None:
            frames.append(obs[f"{CAMERA}_image"][::-1])
        eef.append(np.asarray(obs["robot0_eef_pos"]).copy())
        # 逆向きの掴む姿勢をここから作るので、向きも残す（軸角。state 列と同じ形）
        aa.append(np.asarray(T.quat2axisangle(
            np.asarray(obs["robot0_eef_quat"], dtype=np.float64))))
        if track_obj is not None:
            # 掴む瞬間の「本から見た手の位置」を作るために、物体の姿勢も残す
            obj_p.append(np.asarray(obs[f"{track_obj}_pos"]).copy())
            obj_q.append(np.asarray(obs[f"{track_obj}_quat"]).copy())

    record()
    inner = env.env
    for a in actions:
        obs, _, _, _ = inner.step(np.asarray(a, dtype=np.float32))
        record()

    if video_path is not None:
        write_video(frames, video_path)
    eef = np.asarray(eef)

    out = {"n_steps": len(actions), "eef_pos": eef, "eef_aa": np.asarray(aa)}
    if track_obj is not None:
        out["obj_pos"] = np.asarray(obj_p)
        out["obj_quat"] = np.asarray(obj_q)
    if ref_states is not None:
        # 再生 t ステップ後の位置を、データの state[t] と突き合わせる。
        # データの state[t] は「action[t] を撃つ**前**の姿勢」なので、
        # 再生側も同じ並び（撃つ前）で比べる。
        n = min(len(ref_states), len(eef))
        err = np.linalg.norm(eef[:n] - np.asarray(ref_states)[:n, :3], axis=1)
        out["err_m"] = err
    return out


def grasp_pose_from_forward(fwd, j, obj_pos_now, obj_quat_now):
    """逆向きの「掴む姿勢」を、**本の今の姿勢に合わせて**作る。

    **なぜ基準経路の姿勢をそのまま使えないか。** 順再生はデモと同じ場所へ本を
    置くとは限らない（実測で 25〜35 mm ずれた）。さらに**放した後に本が動く**
    ——指が離れた瞬間に落ちたり滑ったりする。基準経路の姿勢へ腕を 3 mm で
    寄せても、本がそこに無ければ空を掴む（ep4062・4063 がこれだった）。

    そこで順再生の**掴んでいた時刻 `j`** で「本から見た手の姿勢」を測り、
    それを**今の本の姿勢に掛け直す**。本がどこへ動いていても掴みに行ける。

    `obs[..._quat]` は robosuite 規約の (x, y, z, w)。`transform_utils` も同じ。
    """
    from robosuite.utils import transform_utils as T

    r_obj_then = T.quat2mat(np.asarray(fwd["obj_quat"][j], dtype=np.float64))
    r_eef_then = T.quat2mat(T.axisangle2quat(
        np.asarray(fwd["eef_aa"][j], dtype=np.float64)))
    # 本の座標系で見た手（位置と向き）
    rel_pos = r_obj_then.T @ (np.asarray(fwd["eef_pos"][j], dtype=np.float64)
                              - np.asarray(fwd["obj_pos"][j], dtype=np.float64))
    r_rel = r_obj_then.T @ r_eef_then

    r_obj_now = T.quat2mat(np.asarray(obj_quat_now, dtype=np.float64))
    pos = np.asarray(obj_pos_now, dtype=np.float64) + r_obj_now @ rel_pos
    aa = T.quat2axisangle(T.mat2quat(r_obj_now @ r_rel))
    return np.concatenate([pos, np.asarray(aa)])


def legacy_prepare(actions, gripper_mode="keep", skip_gripper=False):
    """**修正前**の作り方をそのまま再現する（前後を見比べるためだけに残す）。

    修正前は 2 つ間違えていた:
      1. `action[6]` を復号せずに流していた（task 92 は 0/1 系なので開閉が入れ替わる）
      2. 反転で順序を変えるだけで**符号を反転していなかった**（相対差分なのに）
    `--legacy` を付けるとこちらが使われ、`before` の動画が再現できる。
    """
    fwd = np.asarray(actions, dtype=np.float32)          # 復号しない
    rev = np.ascontiguousarray(fwd[::-1].copy())          # 符号は反転しない
    if skip_gripper:
        rev[:, 6] = 0.0
    elif gripper_mode == "invert":
        rev[:, 6] = 1.0 - rev[:, 6]
    return fwd, rev


OUT_MAX_POS = np.array([0.05, 0.05, 0.05], dtype=np.float64)
OUT_MAX_ROT = np.array([0.5, 0.5, 0.5], dtype=np.float64)


def rollout_tracking(env, init_state, ref_states, gripper_cmd, video_path=None,
                     gain=1.0, body_pos=None):
    """基準経路（逆順の state 列）を**閉ループで追いかける**再生。

    記録された指令を反転して撃つ open-loop と違い、毎ステップ
    「**今の姿勢から次の基準姿勢へ向かう差分**」を作って撃つので、OSC の遅れぶんの
    誤差が溜まらない。差分は `output_max` で正規化して ±1 に収める
    （`control_delta=True` なので、これがそのまま指令の単位になる）。

    **回転は world 合成（`R_target @ R_now^T`）の軸角で出す。**
    軸角ベクトルの成分ごとの引き算は相対回転ではないので使えない
    （014 の実測でも body 合成・成分引き算は相関が崩れている）。

    `gripper_cmd` は robosuite 規約に直した `action[6]` を逆順にしたもの。
    """
    obs = reset_to(env, init_state, body_pos)
    frames, eef = [], []
    if video_path is not None:
        frames.append(obs[f"{CAMERA}_image"][::-1])
    eef.append(np.asarray(obs["robot0_eef_pos"]).copy())

    inner = env.env
    n = len(ref_states) - 1
    for i in range(n):
        a = osc_command(obs, ref_states[i + 1], gripper_cmd[i], gain)
        obs, _, _, _ = inner.step(a)
        if video_path is not None:
            frames.append(obs[f"{CAMERA}_image"][::-1])
        eef.append(np.asarray(obs["robot0_eef_pos"]).copy())

    if video_path is not None:
        write_video(frames, video_path)
    eef = np.asarray(eef)
    err = np.linalg.norm(eef - np.asarray(ref_states)[:len(eef), :3], axis=1)
    return {"n_steps": n, "eef_pos": eef, "err_m": err}


def osc_command(obs, target, grip, gain=1.0):
    """「今の姿勢 → 目標姿勢」を OSC の指令 (7,) に直す。

    **回転は world 合成（`R_target @ R_now^T`）の軸角で出す。**軸角ベクトルの
    成分ごとの引き算は相対回転ではないので使えない（014 の実測）。
    差分は `output_max` で割って ±1 に収める（`control_delta=True` なので
    これがそのまま指令の単位になる）。
    """
    from robosuite.utils import transform_utils as T

    target = np.asarray(target, dtype=np.float64)
    cur_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    dpos = np.clip(gain * (target[:3] - cur_pos) / OUT_MAX_POS, -1.0, 1.0)
    r_tgt = T.quat2mat(T.axisangle2quat(target[3:6]))
    r_cur = T.quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    aa_rel = T.quat2axisangle(T.mat2quat(r_tgt @ r_cur.T))
    drot = np.clip(gain * np.asarray(aa_rel) / OUT_MAX_ROT, -1.0, 1.0)
    return np.concatenate([dpos, drot, [grip]]).astype(np.float32)


def rollout_phased(env, init_state, ref_states, gripper_cmd, video_path=None,
                   gain=1.0, body_pos=None, tol=0.005, max_approach=60,
                   grip_hold=10, grasp_pose=None):
    """掴むまでは**時刻表を捨てて**姿勢で判定し、掴んでから時刻逆向きに戻る。

    **なぜ要るか。** 時刻をそのまま逆にすると、掴む指令が出る時刻は
    「デモが本を放したフレーム」で決まってしまう。デモが終わり際に放していると
    （ep4062 は全 146 コマ中 f139）、逆向きでは腕が掴む姿勢へ戻るのに 6 コマしか
    与えられない。腕は基準経路に 3 コマ遅れるので**間に合わない**。
    前倒し（`shift_gripper`）はこの猶予を削る向きなので、この型の ep では効かない。

    そこで 3 段に分ける:

      A 接近 — 基準経路の**掴む姿勢**へ、グリッパを開いたまま、**着くまで**寄せる
               （何コマかかってもよい。`tol` 以内に入るか `max_approach` で打ち切り）
      B 把持 — その場に留めたまま閉じる。指のストロークは 1 指令 0.25 進む積分器
               なので全閉まで 8 指令。`grip_hold` はその余裕を見た値
      C 復路 — 掴んだ後は**記録どおりの時刻逆向き**で基準経路を追う

    A と B で入るコマは**デモに無い余分なコマ**だが、学習データを作るのが目的なので
    問題にならない。指令も観測も動画も同じ列に入る。

    **`grasp_pose` には「自分の順再生が実際に本を放した姿勢」を渡すこと。**
    基準経路（デモの state 列）の掴む姿勢をそのまま使うと、**本がそこに無い**
    ことがある——順再生はデモと同じ場所へ本を置くとは限らず、実測で 25〜35 mm
    ずれた。腕は基準の姿勢へ 3〜4 mm で着くのに空を掴む、という失敗になる。
    順再生の eef 列から取れば、本が今どこにあっても必ず整合する。
    省略時は `ref_states[close_i]`（＝デモの姿勢）を使う。

    戻り値の `n_extra` が A+B で足したコマ数。
    """
    obs = reset_to(env, init_state, body_pos)
    inner = env.env
    frames, eef, acts = [], [], []
    if video_path is not None:
        frames.append(obs[f"{CAMERA}_image"][::-1])
    eef.append(np.asarray(obs["robot0_eef_pos"]).copy())

    def step(a):
        nonlocal obs
        obs, _, _, _ = inner.step(a)
        acts.append(np.asarray(a).copy())
        if video_path is not None:
            frames.append(obs[f"{CAMERA}_image"][::-1])
        eef.append(np.asarray(obs["robot0_eef_pos"]).copy())

    grip = np.asarray(gripper_cmd)
    closes = np.where(grip > 0)[0]
    close_i = int(closes[0]) if len(closes) else 0
    grasp_pose = np.asarray(
        ref_states[close_i] if grasp_pose is None else grasp_pose,
        dtype=np.float64)

    # A: 掴む姿勢へ、開いたまま、着くまで
    n_app = 0
    for _ in range(max_approach):
        d = float(np.linalg.norm(
            np.asarray(obs["robot0_eef_pos"]) - grasp_pose[:3]))
        if d < tol:
            break
        step(osc_command(obs, grasp_pose, -1.0, gain))   # -1 = 開く
        n_app += 1
    approach_err = float(np.linalg.norm(
        np.asarray(obs["robot0_eef_pos"]) - grasp_pose[:3]))

    # B: その場で閉じる（指が閉じ切るまで姿勢は保持）
    for _ in range(grip_hold):
        step(osc_command(obs, grasp_pose, 1.0, gain))    # +1 = 閉じる

    # C: 掴んだ以降は記録どおりの時刻逆向き
    for i in range(close_i, len(ref_states) - 1):
        step(osc_command(obs, ref_states[i + 1], grip[i], gain))

    if video_path is not None:
        write_video(frames, video_path)
    eef = np.asarray(eef)
    return {"n_steps": len(acts), "eef_pos": eef, "actions": np.asarray(acts),
            "n_extra": n_app + grip_hold, "n_approach": n_app,
            "approach_err_mm": round(approach_err * 1000, 2),
            "close_i": close_i,
            "err_m": np.array([approach_err])}


def write_video(frames, path, fps=20):
    """mp4 を書く。コンテナに ffmpeg が無いので cv2 で書く。"""
    import cv2

    if not frames:
        return
    h, w = frames[0].shape[:2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"   動画: {path}  ({len(frames)} コマ)", flush=True)


def divergence_frame(err, thresh=0.02):
    """ずれが `thresh`(m) を超えて以後戻らない最初のフレーム番号。"""
    over = np.where(err > thresh)[0]
    if len(over) == 0:
        return None
    return int(over[0])


def summarize(tag, res, thresh=0.02):
    err = res.get("err_m")
    if err is None:
        return {"tag": tag, "n_steps": res["n_steps"]}
    d = divergence_frame(err, thresh)
    out = {
        "tag": tag,
        "n_steps": res["n_steps"],
        "err_median_mm": round(float(np.median(err)) * 1000, 2),
        "err_max_mm": round(float(err.max()) * 1000, 2),
        "err_final_mm": round(float(err[-1]) * 1000, 2),
        f"first_frame_over_{int(thresh * 1000)}mm": d,
    }
    print(f"   {tag}: ずれ 中央 {out['err_median_mm']} mm / 最大 "
          f"{out['err_max_mm']} mm / 最終 {out['err_final_mm']} mm / "
          f"{int(thresh * 1000)}mm 超え開始 = {d}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-npz", required=True)
    ap.add_argument("--forward-bddl", required=True)
    ap.add_argument("--forward-inits", required=True)
    ap.add_argument("--t3-bddl", required=True)
    ap.add_argument("--t3-inits", required=True)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--label", default="before", help="動画名に付ける印（before/after）")
    ap.add_argument("--gripper-mode", choices=["keep", "invert"], default="keep")
    ap.add_argument("--skip-gripper", action="store_true")
    ap.add_argument("--episodes", default="", help="コンマ区切り。既定は npz の全部")
    ap.add_argument("--legacy", action="store_true",
                    help="修正前の作り方で回す（gripper 復号なし・符号反転なし）")
    args = ap.parse_args()

    z = np.load(args.ep_npz, allow_pickle=True)
    eps = ([int(x) for x in args.episodes.split(",") if x.strip()]
           if args.episodes.strip()
           else sorted(int(k.split("_")[1]) for k in z.files
                       if k.startswith("actions_")))
    out_dir = Path(args.out)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    print("順方向 env を作る ...", flush=True)
    env_fwd = make_render_env(args.forward_bddl)
    print("T3 env を作る ...", flush=True)
    env_t3 = make_render_env(args.t3_bddl)
    fwd_inits = load_pruned_init(args.forward_inits)
    t3_inits = load_pruned_init(args.t3_inits)
    print(f"初期状態: forward {len(fwd_inits)} 本 / t3 {len(t3_inits)} 本", flush=True)

    report = []
    for ep in eps:
        raw_actions = z[f"actions_{ep}"].astype(np.float32)
        states = z[f"states_{ep}"].astype(np.float32)
        print(f"\n=== ep {ep}  action {raw_actions.shape} ===", flush=True)
        if args.legacy:
            actions, rev_actions_pre = legacy_prepare(
                raw_actions, args.gripper_mode, args.skip_gripper)
            print("  [legacy] gripper 復号なし・符号反転なしで回す", flush=True)
        else:
            # 生データの gripper 符号化はエピソードによって違う。robosuite 規約へ直す。
            actions = decode_gripper(raw_actions, states)
            rev_actions_pre = None
        rec = {"episode": ep, "n_actions": int(len(actions)),
               "label": args.label, "gripper_mode": args.gripper_mode,
               "skip_gripper": bool(args.skip_gripper)}

        # 1) デモの 1 フレーム目に合う初期状態を選ぶ（候補 (c) を潰す）
        idx, dist, dists = pick_init(env_fwd, fwd_inits, states[0, :3])
        rec["init_index"] = idx
        rec["init_match_mm"] = round(dist * 1000, 2)
        rec["init_match_worst_mm"] = round(max(dists) * 1000, 2)
        print(f"   初期状態: #{idx} を採用（デモ 1 コマ目との差 {dist * 1000:.1f} mm／"
              f"最悪の候補 {max(dists) * 1000:.1f} mm）", flush=True)

        # 2) 順再生（記録された action をそのまま流す）
        fwd = rollout(env_fwd, fwd_inits[idx], actions,
                      vid_dir / f"{args.label}_ep{ep}_forward.mp4", states)
        rec["forward"] = summarize("順再生", fwd)
        rec["forward"]["success"] = bool(env_fwd.check_success())
        print(f"   順再生 :goal 判定 = {rec['forward']['success']}", flush=True)
        fwd_final = env_fwd.env.sim.get_state().flatten()

        # 3) 逆再生 — 逆向き action の作り方は reversal_probe のものをそのまま使う
        rev_actions = (rev_actions_pre if rev_actions_pre is not None
                       else build_reversed_actions(
                           actions, gripper_mode=args.gripper_mode,
                           skip_gripper=args.skip_gripper))

        # 3a) 順再生の最終状態から（＝順方向の終わりを実際に作ってから戻す）
        #     ここは**順方向 env** なので `check_success()` は「順方向の goal
        #     （本が棚の上）」を見ている。**逆再生が成功したなら False になる**
        #     （棚から下ろしたのだから）。True なら本は棚に残ったまま＝戻せていない。
        r1 = rollout(env_fwd, fwd_final, rev_actions,
                     vid_dir / f"{args.label}_ep{ep}_reverse_from_fwdfinal.mp4")
        rec["reverse_from_forward_final"] = {
            "mode": "negate(open-loop)",
            "n_steps": r1["n_steps"],
            "forward_goal_still_true": bool(env_fwd.check_success()),
            "undone": not bool(env_fwd.check_success()),
            "back_to_start_mm": round(float(np.linalg.norm(
                r1["eef_pos"][-1] - fwd["eef_pos"][0])) * 1000, 2),
            "eef_travel_m": round(float(np.linalg.norm(
                r1["eef_pos"][-1] - r1["eef_pos"][0])), 4),
        }
        print(f"   逆再生(open-loop・順の最終状態から): 出発点との差 "
              f"{rec['reverse_from_forward_final']['back_to_start_mm']} mm  "
              f"棚から下ろせた={rec['reverse_from_forward_final']['undone']}",
              flush=True)

        # 3b) 同じ状況を**基準経路の閉ループ追従**で戻す（本命の作り方）
        ref = reverse_reference_path(states)
        grip_rev = actions[::-1, 6].copy()
        r3 = rollout_tracking(env_fwd, fwd_final, ref, grip_rev,
                              vid_dir / f"{args.label}_ep{ep}_reverse_track.mp4")
        rec["reverse_track_from_forward_final"] = {
            "mode": "track(closed-loop)",
            "n_steps": r3["n_steps"],
            "forward_goal_still_true": bool(env_fwd.check_success()),
            "undone": not bool(env_fwd.check_success()),
            "err_median_mm": round(float(np.median(r3["err_m"])) * 1000, 2),
            "err_max_mm": round(float(r3["err_m"].max()) * 1000, 2),
            "back_to_start_mm": round(float(np.linalg.norm(
                r3["eef_pos"][-1] - fwd["eef_pos"][0])) * 1000, 2),
        }
        print(f"   逆再生(閉ループ追従): 基準経路とのずれ 中央 "
              f"{rec['reverse_track_from_forward_final']['err_median_mm']} mm / "
              f"出発点との差 {rec['reverse_track_from_forward_final']['back_to_start_mm']} mm"
              f"  棚から下ろせた={rec['reverse_track_from_forward_final']['undone']}",
              flush=True)

        # 3c) T3 の配布初期状態から（＝本番の Track3 の状況）。
        #     こちらは **T3 env** なので `check_success()` は T3 の goal
        #     （本が机の左領域に戻る）＝ True が成功。
        r2 = rollout(env_t3, t3_inits[0], rev_actions,
                     vid_dir / f"{args.label}_ep{ep}_reverse_t3.mp4")
        rec["reverse_from_t3_init"] = {
            "mode": "negate(open-loop)",
            "n_steps": r2["n_steps"],
            "t3_goal": bool(env_t3.check_success()),
            "eef_travel_m": round(float(np.linalg.norm(
                r2["eef_pos"][-1] - r2["eef_pos"][0])), 4),
        }
        print(f"   逆再生(T3 初期状態から): T3 :goal="
              f"{rec['reverse_from_t3_init']['t3_goal']}  "
              f"移動 {rec['reverse_from_t3_init']['eef_travel_m']} m", flush=True)

        # 3d) 本命: **T3 の配布初期状態から、閉ループ追従で戻す**。
        #     T3 は task 92 と同じ scene・同じ物体で、`:init` と `:goal` が
        #     task 92 のちょうど裏返し。ただし**腕は home 姿勢から始まる**ので、
        #     順方向の最終姿勢を前提にした open-loop の反転列は最初から合わない。
        #     閉ループなら「今どこに居ても基準経路へ寄せる」ので姿勢差を吸収できる。
        r4 = rollout_tracking(env_t3, t3_inits[0], ref, grip_rev,
                              vid_dir / f"{args.label}_ep{ep}_reverse_t3_track.mp4")
        rec["reverse_track_from_t3_init"] = {
            "mode": "track(closed-loop)",
            "n_steps": r4["n_steps"],
            "t3_goal": bool(env_t3.check_success()),
            "err_median_mm": round(float(np.median(r4["err_m"])) * 1000, 2),
            "err_max_mm": round(float(r4["err_m"].max()) * 1000, 2),
        }
        print(f"   逆再生(閉ループ・T3 初期状態から): T3 :goal="
              f"{rec['reverse_track_from_t3_init']['t3_goal']}  "
              f"基準経路とのずれ 中央 "
              f"{rec['reverse_track_from_t3_init']['err_median_mm']} mm", flush=True)

        np.savez(out_dir / f"{args.label}_ep{ep}_traces.npz",
                 forward_eef=fwd["eef_pos"], forward_err=fwd["err_m"],
                 ref_state=states,
                 rev_fwdfinal_eef=r1["eef_pos"], rev_t3_eef=r2["eef_pos"],
                 rev_track_eef=r3["eef_pos"], rev_track_err=r3["err_m"],
                 rev_t3_track_eef=r4["eef_pos"], rev_t3_track_err=r4["err_m"])
        report.append(rec)

    path = out_dir / f"replay_metrics_{args.label}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n書き出し: {path}", flush=True)


if __name__ == "__main__":
    main()
