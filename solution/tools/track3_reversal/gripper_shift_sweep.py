#!/usr/bin/env python3
"""017 続き: 逆向き再生で **gripper 指令だけを時間シフト**して掃く。

**なぜこれを掃くのか。** 逆向き再生の動画で、腕が本から離れて宙に浮いてから
宙でグリッパを閉じている。実測すると、閉じる指令が出るフレームで eef は
基準経路より **109〜162 mm 高い**（open-loop）。原因は
[shift_gripper](reversal_probe.py) の docstring のとおり、腕（3 フレーム）と
指（8 フレーム）の遅れが時間反転で打ち消されないことにある。
**姿勢 6 次元と開閉 1 次元の相対タイミングを変える**のが素直な補償になる。

**測る土俵は「順再生の最終状態から戻す」に固定する。**T3 の配布初期状態は本が
立っており、順方向デモが置いた姿勢と違う。そこで測るとシフトの効きと姿勢差が
混ざるので、まず**物の姿勢が定義上ぴったり一致している**こちらで k を決める。

出力: `<out>/gripper_shift_sweep.json` と標準出力の表。
描画しない `ControlEnv` で回す（掃引を速くするため。物理は同じ）。

用法（コンテナ内。`/app` に track3_reversal を丸ごと mount する）:
    python /app/gripper_shift_sweep.py --ep-npz /out/ep_data.npz \\
        --forward-bddl <task92.bddl> --forward-inits <task92.pruned_init> \\
        --t3-bddl <t3.bddl> --shifts -8,-6,-4,-2,0,2,4,6,8 --out /out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from replay_video_probe import (fixture_snapshot, grasp_pose_from_forward,
                                pick_init, reset_to, rollout, rollout_phased,
                                rollout_tracking)
from reversal_probe import (decode_gripper, load_pruned_init, make_env,
                            reverse_reference_path, shift_gripper)


def goal_of(bddl_path):
    """bddl の :goal から 1 つ目の (predicate, 物体, 領域) を返す（小文字名）。"""
    from libero.libero.envs import bddl_utils as BDDLUtils

    for st in BDDLUtils.robosuite_parse_problem(str(bddl_path))["goal_state"]:
        if len(st) == 3:
            return tuple(st)
    raise SystemExit(f"{bddl_path} の goal から 3 項の述語を取れない")


def t3_goal_env(bddl_path):
    """**逆向き goal を判定するためだけ**の T3 env を作る。

    **同じ region 名でも、定義ファイルが違えば広さが違う。**
    `study_table_yellow_book_left_init_region` は
    順方向 libero_90 では `:ranges (-0.060 -0.26 -0.04 -0.24)`＝**±10 mm**
    （物を置く初期配置用の極小領域）だが、
    配布 T3 では `:ranges (-0.125 -0.325 0.025 -0.175)`＝**±75 mm（15 cm 角）**である。
    T3 の goal を順方向 env の中で評価すると、7.5 倍狭い箱で採点することになる。
    **判定は必ず T3 の env で行うこと。**
    """
    return make_env(str(bddl_path))


def eval_in_t3(env_t3, obj, pred, region, pos, quat):
    """物体を T3 env の中の `pos`/`quat` へ置いて goal 述語を判定する。

    物体は free joint なので `qpos[adr : adr+7]` が xyz + quat（mocap ではない）。
    `On(物, 領域site)` は幾何だけで決まる（領域の親が `study_table`＝アリーナの机で
    `get_object` が None を返すため、接触条件は掛からない）ので、姿勢を差し込んで
    `forward()` すれば判定できる。物理を進める必要はない。
    """
    sim = env_t3.env.sim
    adr = sim.model.jnt_qposadr[sim.model.joint_name2id(f"{obj}_joint0")]
    sim.data.qpos[adr:adr + 3] = pos
    sim.data.qpos[adr + 3:adr + 7] = quat
    sim.forward()
    return bool(env_t3.env._eval_predicate([pred, obj, region]))


def body_xpos(env, name):
    """物体の body 位置。`body_xpos` は**生きたビュー**なので必ず複製する。"""
    return np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id[name]]).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-npz", required=True)
    ap.add_argument("--forward-bddl", required=True)
    ap.add_argument("--forward-inits", required=True)
    ap.add_argument("--t3-bddl", required=True,
                    help="逆向き goal の述語を借りるためだけに読む（env は作らない）")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--episodes", default="", help="コンマ区切り。既定は npz の全部")
    ap.add_argument("--shifts", default="-8,-6,-4,-2,0,2,4,6,8",
                    help="掃く前倒し量（フレーム）。正=グリッパを早める")
    ap.add_argument("--mode", choices=["shift", "phased"], default="shift",
                    help="shift=時刻をずらすだけ / phased=掴むまで姿勢で判定して"
                         "から時刻逆向き（ideas/016 の案。shift は無視される）")
    ap.add_argument("--demo-grasp-pose", dest="grasp_from_forward",
                    action="store_false", default=True,
                    help="phased で掴む姿勢にデモの state を使う（既定は自分の"
                         "順再生が本を放した eef。切り分け用）")
    ap.add_argument("--seed", type=int, default=42,
                    help="016 の評価パリティと同じ既定値。fixture 配置を決める")
    ap.add_argument("--repeats", type=int, default=1,
                    help="1 エピソードを何回繰り返すか。**シムは接触で分岐する**"
                         "（同じ入力でも本の着地が 2 か所に割れ、順再生の着地が"
                         "逆向きの成否を決める）ので、1 回の結果では判断できない")
    ap.add_argument("--render-shifts", default="",
                    help="この shift だけ動画も残す（コンマ区切り）。描画は物理を変えない")
    args = ap.parse_args()

    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    render = {int(x) for x in args.render_shifts.split(",") if x.strip()}
    z = np.load(args.ep_npz, allow_pickle=True)
    eps = ([int(x) for x in args.episodes.split(",") if x.strip()]
           if args.episodes.strip()
           else sorted(int(k.split("_")[1]) for k in z.files
                       if k.startswith("actions_")))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fwd_pred, obj, fwd_region = goal_of(args.forward_bddl)
    rev_pred, rev_obj, rev_region = goal_of(args.t3_bddl)
    print(f"順方向 goal: {fwd_pred} {obj} -> {fwd_region}", flush=True)
    print(f"逆向き goal: {rev_pred} {rev_obj} -> {rev_region}", flush=True)

    vid_dir = out_dir / "videos"
    if render:
        # 描画は物理の後に走るだけなので、掃引の結果は描画の有無で変わらない。
        from replay_video_probe import make_render_env
        print("env を作る（描画あり）...", flush=True)
        env = make_render_env(args.forward_bddl)
        vid_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("env を作る（描画なし）...", flush=True)
        env = make_env(args.forward_bddl)
    # 棚などの fixture は `reset()` のたびに `model.body_pos` を引き直される
    # （実測 最大 18.7 mm）。掃引では **shift 以外を 1 ミリも動かしたくない**ので、
    # seed を立てた上で 1 回目の配置を写し取り、以後すべての reset で上書きする。
    # seed が無いと「固定はされるが run ごとに違う配置に固定される」ことになり、
    # プロセスをまたいだ比較ができない（実測で順再生の着地が 35 mm 割れた）。
    env.seed(args.seed)
    env.reset()
    fixtures = fixture_snapshot(env)
    inits = load_pruned_init(args.forward_inits)
    print(f"初期状態 {len(inits)} 本", flush=True)
    print("T3 env を作る（goal 判定専用）...", flush=True)
    env_t3 = t3_goal_env(args.t3_bddl)
    env_t3.reset()
    t3_half = np.asarray(env_t3.env.object_sites_dict[rev_region].size)
    print(f"T3 の {rev_region} の箱 半分 = "
          f"{np.round(t3_half[:2] * 1000, 1)} mm", flush=True)

    report = []
    for ep, rep in ((e, r) for e in eps for r in range(args.repeats)):
        actions = decode_gripper(z[f"actions_{ep}"].astype(np.float32),
                                 z[f"states_{ep}"])
        states = z[f"states_{ep}"].astype(np.float32)

        # 1) デモ 1 コマ目に合う初期状態を選び、順再生して「戻すべき終点」を作る
        idx, dist, _ = pick_init(env, inits, states[0, :3], fixtures)
        obs0 = reset_to(env, inits[idx], fixtures)
        book_start = np.asarray(obs0[f"{obj}_pos"]).copy()   # 逆向きが目指す位置
        fwd = rollout(env, inits[idx], actions,
                      vid_dir / f"{args.mode}_ep{ep}_r{rep}_forward.mp4"
                      if render else None,
                      states, body_pos=fixtures, track_obj=obj)
        fwd_ok = bool(env.check_success())
        fwd_final = env.env.sim.get_state().flatten()
        book_fwd_end = body_xpos(env, obj)
        book_quat_fwd_end = np.asarray(fwd["obj_quat"][-1]).copy()
        print(f"\n=== ep {ep} (回 {rep + 1}/{args.repeats})  T={len(actions)}  "
              f"初期状態 #{idx} ({dist*1000:.1f} mm) ===", flush=True)
        print(f"  順再生: ずれ 中央 {np.median(fwd['err_m'])*1000:.1f} mm  "
              f":goal={fwd_ok}  本 {np.round(book_start,3)} -> "
              f"{np.round(book_fwd_end,3)}", flush=True)
        if not fwd_ok:
            print("  !! 順再生が goal に届いていない。この ep の逆向きは参考値", flush=True)

        region_pos = np.asarray(env.env.sim.data.get_site_xpos(rev_region)).copy()
        region_half = np.asarray(env.env.object_sites_dict[rev_region].size).copy()
        ref = reverse_reference_path(states)
        grip_rev = actions[::-1, 6].copy()
        closes = np.where(grip_rev > 0)[0]
        close_i = int(closes[0]) if len(closes) else 0

        rows = []
        for k in shifts:
            vid = (vid_dir / f"{args.mode}_ep{ep}_r{rep}_rev_k{k:+03d}.mp4"
                   if k in render else None)
            if args.mode == "phased":
                # 掴む姿勢は**自分の順再生が本を放した瞬間の eef**（デモの姿勢では
                # なく）。逆向きの close_i は順方向の frame T-1-close_i に当たる。
                gp = None
                if args.grasp_from_forward:
                    j = min(max(len(actions) - 1 - close_i, 0),
                            len(fwd["eef_pos"]) - 1)
                    gp = grasp_pose_from_forward(
                        fwd, j, book_fwd_end, book_quat_fwd_end)
                r = rollout_phased(env, fwd_final, ref,
                                   shift_gripper(grip_rev, k), vid,
                                   body_pos=fixtures, grasp_pose=gp)
            else:
                r = rollout_tracking(env, fwd_final, ref,
                                     shift_gripper(grip_rev, k), vid,
                                     body_pos=fixtures)
            book_end = body_xpos(env, obj)
            book_quat = np.asarray(
                env.env.sim.data.body_xquat[env.env.obj_body_id[obj]]).copy()
            row = {
                "shift": k,
                # 本が「順方向の出発点」へどれだけ戻ったか。これが本命の指標。
                "book_to_start_mm": round(float(np.linalg.norm(
                    book_end - book_start)) * 1000, 1),
                # 掴めずに置き去りなら、本は順方向の終点に残る
                "book_from_fwd_end_mm": round(float(np.linalg.norm(
                    book_end - book_fwd_end)) * 1000, 1),
                "book_z": round(float(book_end[2]), 3),
                "forward_goal_still_true": bool(env.check_success()),
                # **T3 の env で判定する。**順方向 env の同名 region は 7.5 倍狭い。
                "reverse_goal": eval_in_t3(env_t3, rev_obj, rev_pred, rev_region,
                                           book_end, book_quat),
                "reverse_goal_in_fwd_env": bool(env.env._eval_predicate(
                    [rev_pred, rev_obj, rev_region])),
                # `On(物, 領域site)` は TargetZone.under() で判定される。
                # 「領域の箱の中に居るか」を xy と z に分けて出す（落ちた理由を読むため）。
                "region_dxy_mm": [round(float(v) * 1000, 1) for v in
                                  (book_end - region_pos)[:2]],
                "region_half_mm": [round(float(v) * 1000, 1)
                                   for v in t3_half[:2]],
                "region_half_fwd_mm": [round(float(v) * 1000, 1)
                                       for v in region_half[:2]],
                "err_median_mm": round(float(np.median(r["err_m"])) * 1000, 1),
                "n_steps": int(r["n_steps"]),
                **({"n_extra": int(r["n_extra"]),
                    "n_approach": int(r["n_approach"]),
                    "approach_err_mm": r["approach_err_mm"]}
                   if "n_extra" in r else {}),
                "back_to_start_mm": round(float(np.linalg.norm(
                    r["eef_pos"][-1] - fwd["eef_pos"][0])) * 1000, 1),
            }
            row["undone"] = not row["forward_goal_still_true"]
            rows.append(row)
            extra = (f"  接近 {row['n_approach']} コマで {row['approach_err_mm']} mm"
                     if "n_approach" in row else
                     f"  腕ずれ {row['err_median_mm']:.1f} mm")
            print(f"  shift {k:+3d}: 本→出発点 {row['book_to_start_mm']:7.1f} mm  "
                  f"本→順の終点 {row['book_from_fwd_end_mm']:7.1f} mm  "
                  f"z={row['book_z']:.3f}  棚から下ろせた={row['undone']}  "
                  f"逆向き goal={row['reverse_goal']}{extra}", flush=True)

        best = min(rows, key=lambda r: r["book_to_start_mm"])
        print(f"  --> 最良 shift={best['shift']:+d} "
              f"(本→出発点 {best['book_to_start_mm']} mm)", flush=True)
        report.append({"episode": ep, "repeat": rep, "n_actions": int(len(actions)),
                       "init_index": idx, "forward_goal": fwd_ok,
                       "book_start": [round(float(x), 4) for x in book_start],
                       "book_forward_end": [round(float(x), 4) for x in book_fwd_end],
                       "shifts": rows, "best_shift": best["shift"]})

    path = out_dir / "gripper_shift_sweep.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n書き出し: {path}", flush=True)


if __name__ == "__main__":
    main()
