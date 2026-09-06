#!/usr/bin/env python3
"""017 続き: 逆向き再生の作り方を掃いて、配布 T3 の goal に通るかを測る。

**選べる軸は直交している。**初期状態をどこから取るか（`--init-source`）と、
軌道をどう作るか（`--mode`）と、掴む姿勢を順方向のどのフレームから測るか
（`--grasp-frame`）は互いに独立で、エピソードを見分ける分岐は書かない。

**実測でいちばん良かった組み合わせ**（3 本すべてで逆向き goal が立つ）:

    --mode phased --init-source t3 --grasp-frame clamped --standoff 0.06

既定は従来の組み合わせ（`shift` / `forward-final` / `release` / standoff 無し）
のままにしてある。過去に測った数字をそのまま再現できるようにするためで、
**新しく回すなら上の組み合わせを使うこと。**

出力: `<out>/gripper_shift_sweep.json` と標準出力の表。
描画しない `ControlEnv` で回す（掃引を速くするため。物理は同じ）。

用法（コンテナ内。`/app` に track3_reversal を丸ごと mount する）:
    python /app/gripper_shift_sweep.py --ep-npz /out/ep_data.npz \\
        --forward-bddl <task92.bddl> --forward-inits <task92.pruned_init> \\
        --t3-bddl <t3.bddl> --t3-inits <t3.pruned_init> --out /out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from replay_video_probe import (episode_ids, grasp_pose_from_forward,
                                make_render_env, pick_init, pin_fixtures,
                                probe_init_eef, reset_to, rollout,
                                rollout_phased, rollout_tracking)
from reversal_probe import (decode_gripper, load_pruned_init, make_env,
                            reverse_reference_path, shift_gripper)

# bddl の `:goal` で「物を置く」ことを表す述語。パーサが小文字化した名前で来る。
PLACEMENT = {"on", "in", "at", "infrontof", "behind",
             "totheleftof", "totherightof", "ontopof"}


def goal_of(bddl_path):
    """bddl の :goal から「置く系」の (predicate, 物体, 領域) を返す（小文字名）。

    置く系だけを拾う。`(And (Turnon stove) (On pot region))` のように別の述語が
    先に来ることがあり、それを取ると逆向きの向きを取り違える。
    """
    from libero.libero.envs import bddl_utils as BDDLUtils

    for st in BDDLUtils.robosuite_parse_problem(str(bddl_path))["goal_state"]:
        if len(st) == 3 and st[0] in PLACEMENT:
            return tuple(st)
    raise SystemExit(f"{bddl_path} の goal から「置く系」の述語を取れない")


def eval_in_t3(env_t3, obj, pred, region, pos, quat):
    """物体を T3 env の中の `pos`/`quat` へ置いて goal 述語を判定する。

    **同じ region 名でも、定義ファイルが違えば広さが違う。**
    `study_table_yellow_book_left_init_region` は順方向 libero_90 では
    `:ranges (-0.060 -0.26 -0.04 -0.24)`＝**±10 mm**（物を置く初期配置用の
    極小領域）だが、配布 T3 では `:ranges (-0.125 -0.325 0.025 -0.175)`＝
    **±75 mm（15 cm 角）**である。順方向 env の中で T3 の goal を評価すると
    7.5 倍狭い箱で採点することになる。**判定は必ず T3 の env で行う。**

    物体は free joint なので `qpos[adr : adr+7]` が xyz + quat（mocap ではない）。
    `qpos` のクォータニオンは **(w, x, y, z)** なので、robosuite 規約で受け取って
    ここで並べ替える。
    `On(物, 領域site)` は幾何だけで決まる（領域の親が `study_table`＝アリーナの机で
    `get_object` が None を返すため、接触条件は掛からない）ので、姿勢を差し込んで
    `forward()` すれば判定できる。物理を進める必要はない。
    """
    from robosuite.utils import transform_utils as T

    sim = env_t3.env.sim
    adr = sim.model.jnt_qposadr[sim.model.joint_name2id(f"{obj}_joint0")]
    sim.data.qpos[adr:adr + 3] = pos
    # 受け取るのは robosuite 規約 (x,y,z,w)。qpos は MuJoCo 規約 (w,x,y,z)。
    sim.data.qpos[adr + 3:adr + 7] = T.convert_quat(np.asarray(quat), to="wxyz")
    sim.forward()
    return bool(env_t3.env._eval_predicate([pred, obj, region]))


def clamp_frame(gripper_cmd, states):
    """順方向で**実際に物を挟み終えた**フレームを、指の開き幅から決める。

    **「閉じる指令が出たフレーム」は掴む前である。** デモは指令を出したあとも
    腕を進めながら掴むので、指令の瞬間の姿勢で腕を止めて閉じると空を掴む
    （実測: 指令の瞬間、指の中点は本から 131.7 mm 離れていた）。

    運搬中の開き幅の中央値まで縮んだ最初のフレームを採る。実測では閉じ指令の
    8〜17 コマ後で、そのときの幅は本の厚み（23.6 mm）とよく合う。
    """
    w = gripper_width(states)
    g = np.asarray(gripper_cmd)
    t_close = int(np.argmax(g > 0))
    t_open = len(g) - 1 - int(np.argmax(g[::-1] > 0))     # 最後に閉じている frame
    w_hold = float(np.median(w[t_close:t_open + 1]))
    return t_close + int(np.argmax(w[t_close:t_open + 1] <= w_hold + 0.003))


def gripper_width(states):
    """`observation.state` から指の開き幅を出す（`state[6] - state[7]`）。"""
    st = np.asarray(states)
    return st[:, 6] - st[:, 7]


def object_is_upright(quat, thin_axis_local=0):
    """物が「立っている」か（挟むべき最短辺の軸が水平を向いているか）を返す。

    **寝ているか立っているかで、掴み方の前提が変わる。**寝ていると最短辺が
    鉛直を向き、下側の指を物の下へ入れる必要が出る（棚の天板からの浮きは
    3 mm しかないので、**縁からはみ出した部分でしか掴めない**）。立っていれば
    最短辺が水平なので、横から普通に挟める。

    **エピソード番号ではなく、この測った値で分岐させること。**そうすれば
    隠しタスクにも同じ規則が効く。
    """
    from robosuite.utils import transform_utils as T

    thin = T.quat2mat(np.asarray(quat, dtype=np.float64))[:, thin_axis_local]
    return abs(float(thin[2])) < 0.5


def body_pose(env, name):
    """物体の body の位置と向きを **robosuite 規約 (x, y, z, w)** で返す。

    **クォータニオンの並びが 2 種類あることに注意。**MuJoCo の
    `sim.data.body_xquat` と `qpos` は **(w, x, y, z)** だが、robosuite の観測
    （`obs["<物体>_quat"]`）と `transform_utils` は **(x, y, z, w)** である。
    ここは後者へ揃えて返すので、`qpos` へ書き戻すときだけ並べ替えること
    （[eval_in_t3](#) がやっている）。

    `body_xpos` などは**生きたビュー**なので必ず複製する。
    """
    from robosuite.utils import transform_utils as T

    data, bid = env.env.sim.data, env.env.obj_body_id[name]
    return (np.asarray(data.body_xpos[bid]).copy(),
            T.convert_quat(np.asarray(data.body_xquat[bid]).copy(), to="xyzw"))


def region_box(env, region):
    """region site の中心と、箱の半サイズを返す。**必ず判定する env から取る。**"""
    return (np.asarray(env.env.sim.data.get_site_xpos(region)).copy(),
            np.asarray(env.env.object_sites_dict[region].size).copy())


def grasp_source_frame(kind, actions, states, close_i):
    """掴む姿勢の相対オフセットを、順方向のどのフレームから測るかを決める。"""
    if kind == "clamped":
        return clamp_frame(actions[:, 6], states)
    if kind == "grasp":
        return int(np.argmax(actions[:, 6] > 0))
    return len(actions) - 1 - close_i          # release: 放す直前


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-npz", required=True)
    ap.add_argument("--forward-bddl", required=True)
    ap.add_argument("--forward-inits", required=True)
    ap.add_argument("--t3-bddl", required=True,
                    help="逆向き goal の述語と、--init-source t3 のときの env に使う")
    ap.add_argument("--t3-inits", default="",
                    help="配布 T3 の .pruned_init。--init-source t3 に必須")
    ap.add_argument("--init-source", choices=["forward-final", "t3"],
                    default="forward-final",
                    help="逆向きエピソードの初期状態をどこから取るか。"
                         "forward-final=順再生の最終状態（既定。腕は棚の上・本は寝ている）/ "
                         "t3=配布 T3 の init（本番と同じ。本は立ち・腕は home）")
    ap.add_argument("--t3-init-index", type=int, default=0,
                    help="--init-source t3 のとき使う init の行番号")
    ap.add_argument("--grasp-frame", choices=["release", "grasp", "clamped"],
                    default="release",
                    help="掴む姿勢の相対オフセットを順方向のどのフレームから測るか。"
                         "release=放す直前（既定）/ grasp=閉じる指令の瞬間"
                         "（**まだ掴んでいない**ので通常は不可）/ "
                         "clamped=実際に挟み終えた瞬間（立っている物にはこれ）")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--episodes", default="", help="コンマ区切り。既定は npz の全部")
    ap.add_argument("--shifts", default="-8,-6,-4,-2,0,2,4,6,8",
                    help="掃く前倒し量（フレーム）。正=グリッパを早める")
    ap.add_argument("--mode", choices=["shift", "phased"], default="shift",
                    help="shift=グリッパ指令の時刻をずらすだけ / "
                         "phased=掴むまで姿勢で判定してから時刻逆向き。"
                         "**phased でも --shifts は効く**——ずらした列から"
                         "「掴む時刻」を決めるので、復路に入る位置が変わる")
    ap.add_argument("--standoff", type=float, default=0.0,
                    help="掴む姿勢の手前（グリッパの進入軸に沿って）に置く待機点の"
                         "距離[m]。0 で無効。物を押さずに近づくために使う")
    ap.add_argument("--seed", type=int, default=42,
                    help="016 の評価パリティと同じ既定値。fixture 配置を決める")
    ap.add_argument("--render-shifts", default="",
                    help="この shift だけ動画も残す（コンマ区切り）。描画は物理を変えない")
    return ap.parse_args()


def build_envs(args, render, rev_region):
    """順方向 env と T3 env を作る。**fixture は両方とも固定する。**

    `--init-source t3` のときは T3 env が再生の舞台にもなるので、描画の要否は
    そちらにも効く。fixture の固定は [pin_fixtures](replay_video_probe.py) が
    面倒を見るので、以降どの関数にも配って回らなくてよい。
    """
    use_t3 = args.init_source == "t3"
    make = make_render_env if render else make_env
    print(f"env を作る（描画{'あり' if render else 'なし'}）...", flush=True)
    env = pin_fixtures(make(args.forward_bddl), args.seed)
    env_t3 = pin_fixtures(make(args.t3_bddl) if use_t3 and render
                          else make_env(args.t3_bddl), args.seed)
    env_t3.reset()
    _, t3_half = region_box(env_t3, rev_region)
    print(f"T3 の {rev_region} の箱 半分 = "
          f"{np.round(t3_half[:2] * 1000, 1)} mm", flush=True)
    return env, env_t3, use_t3


def measure(rev_env, obj, rev_region, r, rev_goal, book_rev_start):
    """1 本の逆向き再生から、**その再生を回した env だけで測れる**値を出す。

    順方向 env の値（本の出発点など）と混ぜないこと。`--init-source t3` では
    逆向きを別の env で回すので、順方向側の座標と引き算しても意味が無い。
    """
    book_end, _ = body_pose(rev_env, obj)
    region_pos, region_half = region_box(rev_env, rev_region)
    row = {
        # 本が「逆向きの出発点」からどれだけ動いたか（掴んで運べたかの目安）
        "book_moved_mm": round(float(np.linalg.norm(
            book_end - book_rev_start)) * 1000, 1),
        "book_z": round(float(book_end[2]), 3),
        "reverse_goal": rev_goal,
        # `On(物, 領域site)` は TargetZone.under() で判定される。
        # 「領域の箱の中に居るか」を xy に分けて出す（落ちた理由を読むため）。
        "region_dxy_mm": [round(float(v) * 1000, 1)
                          for v in (book_end - region_pos)[:2]],
        "region_half_mm": [round(float(v) * 1000, 1) for v in region_half[:2]],
        "n_steps": int(r["n_steps"]),
    }
    if "n_extra" in r:                          # phased のときだけ付く
        row.update(n_extra=int(r["n_extra"]), n_approach=int(r["n_approach"]),
                   approach_err_mm=r["approach_err_mm"])
    else:
        row["err_median_mm"] = round(float(np.median(r["err_m"])) * 1000, 1)
    return row


def main():
    args = parse_args()
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]
    render = {int(x) for x in args.render_shifts.split(",") if x.strip()}
    z = np.load(args.ep_npz, allow_pickle=True)
    eps = episode_ids(z, args.episodes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = out_dir / "videos"
    if render:
        vid_dir.mkdir(parents=True, exist_ok=True)

    fwd_pred, obj, fwd_region = goal_of(args.forward_bddl)
    rev_pred, rev_obj, rev_region = goal_of(args.t3_bddl)
    print(f"順方向 goal: {fwd_pred} {obj} -> {fwd_region}", flush=True)
    print(f"逆向き goal: {rev_pred} {rev_obj} -> {rev_region}", flush=True)

    env, env_t3, use_t3 = build_envs(args, render, rev_region)
    inits = load_pruned_init(args.forward_inits)
    # 50 本ぶんの reset は 25 秒かかるが、結果はエピソードに依らない。1 度だけ測る。
    probed = probe_init_eef(env, inits)
    print(f"初期状態 {len(inits)} 本を下見した", flush=True)
    t3_inits = None
    if use_t3:
        if not args.t3_inits:
            raise SystemExit("--init-source t3 には --t3-inits が要る")
        t3_inits = load_pruned_init(args.t3_inits)
        print(f"T3 の初期状態 {len(t3_inits)} 本。#{args.t3_init_index} を使う",
              flush=True)

    report = []
    for ep in eps:
        actions = decode_gripper(z[f"actions_{ep}"].astype(np.float32),
                                 z[f"states_{ep}"])
        states = z[f"states_{ep}"].astype(np.float32)

        # 1) デモ 1 コマ目に合う初期状態を選び、順再生して「戻すべき終点」を作る
        idx, dist, _ = pick_init(env, inits, states[0, :3], probed)
        book_start = np.asarray(reset_to(env, inits[idx])[f"{obj}_pos"]).copy()
        fwd = rollout(env, inits[idx], actions,
                      vid_dir / f"{args.mode}_ep{ep}_forward.mp4" if render else None,
                      states, track_obj=obj)
        fwd_ok = bool(env.check_success())
        fwd_final = env.env.sim.get_state().flatten()
        book_fwd_end, book_quat_fwd_end = body_pose(env, obj)
        print(f"\n=== ep {ep}  T={len(actions)}  初期状態 #{idx} "
              f"({dist * 1000:.1f} mm) ===", flush=True)
        print(f"  順再生: ずれ 中央 {np.median(fwd['err_m']) * 1000:.1f} mm  "
              f":goal={fwd_ok}  本 {np.round(book_start, 3)} -> "
              f"{np.round(book_fwd_end, 3)}", flush=True)
        if not fwd_ok:
            print("  !! 順再生が goal に届いていない。この ep の逆向きは参考値",
                  flush=True)

        ref = reverse_reference_path(states)
        grip_rev = actions[::-1, 6].copy()
        closes = np.where(grip_rev > 0)[0]
        close_i = int(closes[0]) if len(closes) else 0
        j = min(max(grasp_source_frame(args.grasp_frame, actions, states, close_i),
                    0), len(fwd["eef_pos"]) - 1)

        # 2) 逆向きを回す env と初期状態を決める（軌道の作り方とは独立の軸）
        if use_t3:
            rev_env = env_t3
            rev_init = t3_inits[args.t3_init_index]
            obs_t3 = reset_to(rev_env, rev_init)
            book_rev_start = np.asarray(obs_t3[f"{obj}_pos"]).copy()
            book_rev_quat = np.asarray(obs_t3[f"{obj}_quat"]).copy()
            print(f"  T3 初期状態: 本 {np.round(book_rev_start, 3)}  "
                  f"腕 {np.round(obs_t3['robot0_eef_pos'], 3)}", flush=True)
        else:
            rev_env = env
            rev_init = fwd_final
            book_rev_start, book_rev_quat = book_fwd_end, book_quat_fwd_end
        upright = object_is_upright(book_rev_quat)
        print(f"  逆向きの初期状態: 本は{'立っている' if upright else '寝ている'}"
              f"／掴む姿勢は順方向 f{j} から取る", flush=True)

        # 掴む姿勢は**本の座標系で測った相対姿勢を、今の本の姿勢に掛け直して**作る。
        # 本がどの向きに置かれていても狙える。shift に依らないのでここで 1 度だけ。
        grasp_pose = grasp_pose_from_forward(fwd, j, book_rev_start, book_rev_quat)

        rows = []
        for k in shifts:
            vid = (vid_dir / f"{args.mode}_ep{ep}_rev_k{k:+03d}.mp4"
                   if k in render else None)
            grip = shift_gripper(grip_rev, k)
            if args.mode == "phased":
                r = rollout_phased(rev_env, rev_init, ref, grip, vid,
                                   grasp_pose=grasp_pose, standoff=args.standoff)
            else:
                r = rollout_tracking(rev_env, rev_init, ref, grip, vid)
            # **判定は逆向きを回した env で行う。**T3 env で回したならそのまま
            # `check_success()` が T3 の goal。順方向 env で回したなら、本の姿勢を
            # T3 env へ差し込んで判定する（順方向 env の同名 region は 7.5 倍狭い）。
            if use_t3:
                rev_goal = bool(rev_env.check_success())
            else:
                rev_goal = eval_in_t3(env_t3, rev_obj, rev_pred, rev_region,
                                      *body_pose(rev_env, obj))
            row = {"shift": k,
                   **measure(rev_env, obj, rev_region, r, rev_goal, book_rev_start)}
            if not use_t3:
                # 順方向と同じ env で回したときだけ、順方向側の座標と比べられる
                book_end, _ = body_pose(rev_env, obj)
                row["book_to_start_mm"] = round(float(np.linalg.norm(
                    book_end - book_start)) * 1000, 1)
                row["forward_goal_undone"] = not bool(env.check_success())
            rows.append(row)
            extra = (f"  接近 {row['n_approach']} コマで {row['approach_err_mm']} mm"
                     if "n_approach" in row else
                     f"  腕ずれ {row['err_median_mm']:.1f} mm")
            print(f"  shift {k:+3d}: 本の移動 {row['book_moved_mm']:7.1f} mm  "
                  f"領域dxy {row['region_dxy_mm']} (箱±{row['region_half_mm'][0]:.0f})  "
                  f"z={row['book_z']:.3f}  逆向き goal={row['reverse_goal']}{extra}",
                  flush=True)

        # 最良の選び方: goal が立ったものを優先し、その中で領域中心に近いもの
        best = min(rows, key=lambda r: (not r["reverse_goal"],
                                        float(np.hypot(*r["region_dxy_mm"]))))
        print(f"  --> 最良 shift={best['shift']:+d} "
              f"(逆向き goal={best['reverse_goal']}, "
              f"領域中心まで {np.hypot(*best['region_dxy_mm']):.1f} mm)", flush=True)
        report.append({"episode": ep, "n_actions": int(len(actions)),
                       "init_index": idx, "forward_goal": fwd_ok,
                       "object_upright": upright, "grasp_frame_index": j,
                       "book_start": [round(float(x), 4) for x in book_start],
                       "book_forward_end": [round(float(x), 4)
                                            for x in book_fwd_end],
                       "shifts": rows, "best_shift": best["shift"]})

    path = out_dir / "gripper_shift_sweep.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n書き出し: {path}", flush=True)


if __name__ == "__main__":
    main()
