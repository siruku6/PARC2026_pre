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
既定は描画しない `ControlEnv` で回す（掃引を速くするため。物理は同じ）。

**`--write-npz <dir>` を付けると、逆向き goal が立った回を学習データの中継物
（npz）として書く。**鍵は `scripted_demo.SavedEpisode` に合わせてあるので、
そこから先は [npz_to_lerobot.py](../scripted_demo/npz_to_lerobot.py) の
`build` / `inspect` / `merge` がそのまま使える。要るものの一覧は
`docs/reference/29_training_data_requirements.md`。

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

from replay_video_probe import (DATASET_CAMERAS, episode_ids, fwd_view,
                                grasp_backoff, standing_axis, top_down_grasp,
                                grasp_pose_from_forward, make_render_env,
                                pick_init, pin_fixtures, probe_init_eef,
                                reset_to, rollout, rollout_phased,
                                rollout_tracking)
from reversal_probe import (decode_gripper, load_pruned_init, make_env,
                            reverse_reference_path, shift_gripper)

# bddl の `:goal` で「物を置く」ことを表す述語。パーサが小文字化した名前で来る。
PLACEMENT = {"on", "in", "at", "infrontof", "behind",
             "totheleftof", "totherightof", "ontopof"}


def goals_of(bddl_path):
    """bddl の :goal から「置く系」の (predicate, 物体, 領域) を**全部**返す。

    置く系だけを拾う。`(And (Turnon stove) (On pot region))` のように別の述語が
    先に来ることがあり、それを取ると逆向きの向きを取り違える。

    **1 つとは限らない。**two mugs の goal は
    `(And (On porcelain_mug_1 領域) (On white_yellow_mug_1 領域))` で、
    **2 つの物をそれぞれ別の領域へ**置くことを求める。
    """
    from libero.libero.envs import bddl_utils as BDDLUtils

    out = [tuple(st) for st
           in BDDLUtils.robosuite_parse_problem(str(bddl_path))["goal_state"]
           if len(st) == 3 and st[0] in PLACEMENT]
    if not out:
        raise SystemExit(f"{bddl_path} の goal から「置く系」の述語を取れない")
    return out


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


def clamp_frame(gripper_cmd, states, lo=None, hi=None):
    """順方向で**実際に物を挟み終えた**フレームを、指の開き幅から決める。

    **「閉じる指令が出たフレーム」は掴む前である。** デモは指令を出したあとも
    腕を進めながら掴むので、指令の瞬間の姿勢で腕を止めて閉じると空を掴む
    （実測: 指令の瞬間、指の中点は本から 131.7 mm 離れていた）。

    運搬中の開き幅の中央値まで縮んだ最初のフレームを採る。実測では閉じ指令の
    8〜17 コマ後で、そのときの幅は本の厚み（23.6 mm）とよく合う。
    """
    w = gripper_width(states)
    g = np.asarray(gripper_cmd)
    if lo is None:
        lo = int(np.argmax(g > 0))
    if hi is None:
        hi = len(g) - 1 - int(np.argmax(g[::-1] > 0))     # 最後に閉じている frame
    w_hold = float(np.median(w[lo:hi + 1]))
    return lo + int(np.argmax(w[lo:hi + 1] <= w_hold + 0.003))


def grasp_cycles(gripper_cmd, states, obj_pos, min_frames=4, min_move=0.02):
    """順方向デモを「掴んで放す」単位に切り分け、各回で**動いた物**を割り当てる。

    **物の名前は書かない。**閉じている区間で最も動いた物をその回の相手とする。
    測って決めるので、隠しタスクにも同じ規則が効く。

    捨てるものが 2 つある。**指が一瞬しか閉じない区間**（`min_frames`）と、
    **何も動かしていない区間**（`min_move`）である。どちらも掴み損ねで、
    そこから掴む姿勢を測ると値が壊れる（[reference/30 の 5 節]）。

    実測: two mugs の順方向デモ 20 本のうち 19 本が**ちょうど 2 回**、
    1 本が 3 回（掴み直し）だった。
    """
    g = np.asarray(gripper_cmd) > 0
    edge = np.diff(np.concatenate([[False], g, [False]]).astype(int))
    out = []
    for c, o in zip(np.where(edge == 1)[0], np.where(edge == -1)[0] - 1):
        if o - c < min_frames:
            continue
        moved = {n: float(np.linalg.norm(p[o] - p[c])) for n, p in obj_pos.items()}
        name, dist = max(moved.items(), key=lambda kv: kv[1])
        if dist < min_move:
            continue
        out.append({"close": int(c), "open": int(o), "obj": name,
                    "clamp": clamp_frame(gripper_cmd, states, int(c), int(o)),
                    "moved_mm": round(dist * 1000, 1)})
    return out


def gripper_width(states):
    """`observation.state` から指の開き幅を出す（`state[6] - state[7]`）。"""
    st = np.asarray(states)
    return st[:, 6] - st[:, 7]


def rot_angle_deg(quat_a, quat_b):
    """2 つの向きの間の回転角[deg]（どちらも robosuite 規約 (x,y,z,w)）。

    **接近が発散したときに真っ先に見る値である。**OSC の回転指令は 1 コマ
    0.5 rad で頭打ちなので、要求角が大きいと腕が回りながら遠回りし、位置の
    誤差がかえって開く。「狙う位置は正しいのに届かない」はこれで起きる。
    """
    from robosuite.utils import transform_utils as T

    r = T.quat2mat(np.asarray(quat_a)) @ T.quat2mat(np.asarray(quat_b)).T
    return float(np.degrees(np.linalg.norm(T.quat2axisangle(T.mat2quat(r)))))


def object_yaw_deg(quat):
    """物の鉛直軸まわりの向き[deg]。**配布 init が振っている主な軸である。**

    失敗した init を「位置のせい」と「向きのせい」に切り分けるために出す。
    """
    from robosuite.utils import transform_utils as T

    return float(np.degrees(T.mat2euler(T.quat2mat(np.asarray(quat)))[2]))


def index_spec(spec, n):
    """`"0"` / `"0,2,5"` / `"0-49"` / `"all"` を行番号の並びにする。

    `.pruned_init` は 1 タスクにつき **50 行**入っている（1 行ではない）。
    既定で `#0` しか使わないと「初期状態が 1 通りだけの学習データ」になるので、
    範囲で掃けるようにしてある。
    """
    s = spec.strip()
    if not s or s == "all":
        return list(range(n))
    out = []
    for part in s.split(","):
        if not part.strip():
            continue
        a, _, b = part.strip().partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    bad = sorted({i for i in out if not 0 <= i < n})
    if bad:
        raise SystemExit(f"init の行番号が範囲外: {bad}（0..{n - 1}）")
    return out


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


def write_episode_npz(path, env, r, success):
    """1 本を `scripted_demo.SavedEpisode` の鍵で書く。

    **形式を手で起こさない。**この鍵で書けば
    [npz_to_lerobot.py](../scripted_demo/npz_to_lerobot.py) の
    `build` / `inspect` / `align` / `merge` がそのまま使える。取り決めの置き場は
    [scripted_demo.py](../scripted_demo/scripted_demo.py) の `SavedEpisode`。

    `success` を書いておくと `build` が失敗した回を**画を伸長する前に**捨てる。
    圧縮しない——動画へ再符号化するだけの中継物なので、時間のほうが高い。
    """
    n = len(r["actions"])
    if len(r["states"]) != n:
        raise SystemExit(f"state {len(r['states'])} と action {n} の長さが違う")
    np.savez(path, state=r["states"], action=r["actions"],
             front=r["images"]["front"], wrist=r["images"]["wrist"],
             success=bool(success), steps=n,
             instruction=env.language_instruction)


def all_object_pos(env):
    """シーンの物すべての位置。**掴む物以外が動いたら、腕か荷物が押している。**

    配布 T3 の採点は「置けたか」だけを見るが、皿やトレイがずれる経路は**接触で
    動かしている**ので、学習データとしては入れたくない。1 本ごとにこの差分を
    見れば、擦っている回をその場で弾ける。
    """
    return {n: np.asarray(env.env.sim.data.body_xpos[b]).copy()
            for n, b in env.env.obj_body_id.items()}


def site_frame(env, site):
    """site の world 位置と回転行列。**env をまたいで姿勢を移す基準に使う。**"""
    sim = env.env.sim
    return (np.asarray(sim.data.get_site_xpos(site)).copy(),
            np.asarray(sim.data.get_site_xmat(site)).reshape(3, 3).copy())


def object_qpos(env, name):
    """free joint の qpos 7 要素（xyz + quat **(w,x,y,z)**）をそのまま返す。

    `body_pose` と違って**並べ替えない**。`.pruned_init` へ書き戻す値なので、
    MuJoCo の並びのまま持ち回るほうが取り違えが起きない。
    """
    sim = env.env.sim
    adr = sim.model.jnt_qposadr[sim.model.joint_name2id(f"{name}_joint0")]
    return np.asarray(sim.data.qpos[adr:adr + 7]).copy()


def rebase_qpos(qpos7, frame_from, frame_to):
    """基準 site で測った姿勢を、別の env の同じ site の座標系へ移す。

    **座標をそのままコピーしてはいけない。**同じ名前の棚でも env が違えば配置の
    抽選が違い、実測で **21.7 mm** ずれていた（順方向 `(-0.198, 0.271)` に対し
    配布 T3 は `(-0.210, 0.289)`）。はみ出し量は数 mm で掴めるかどうかが変わる
    ので、このずれは結論を変える。
    """
    from robosuite.utils import transform_utils as T

    p_a, r_a = frame_from
    p_b, r_b = frame_to
    r_rel = r_b @ r_a.T
    pos = p_b + r_rel @ (np.asarray(qpos7[:3], dtype=np.float64) - p_a)
    # qpos の quat は (w,x,y,z)、transform_utils は (x,y,z,w)。
    r_obj = T.quat2mat(T.convert_quat(np.asarray(qpos7[3:7]), to="xyzw"))
    return np.concatenate(
        [pos, T.convert_quat(T.mat2quat(r_rel @ r_obj), to="wxyz")])


def with_object_qpos(env, init_state, name, qpos7):
    """flatten 済みシム状態の、その物の姿勢だけ差し替えた複製を返す。

    `set_state_from_flattened` が読む並びは `[time] + qpos + qvel` なので、
    qpos の先頭が index 1 になる。
    """
    sim = env.env.sim
    adr = sim.model.jnt_qposadr[sim.model.joint_name2id(f"{name}_joint0")]
    st = np.asarray(init_state, dtype=np.float64).copy()
    if len(st) != 1 + sim.model.nq + sim.model.nv:
        raise SystemExit(f"init の長さ {len(st)} が env（{sim.model.nq}+"
                         f"{sim.model.nv}+1）と合わない")
    st[1 + adr:1 + adr + 7] = qpos7
    return st


def region_box(env, region):
    """region site の中心と、箱の半サイズを返す。**必ず判定する env から取る。**"""
    return (np.asarray(env.env.sim.data.get_site_xpos(region)).copy(),
            np.asarray(env.env.object_sites_dict[region].size).copy())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-npz", required=True)
    ap.add_argument("--forward-bddl", required=True)
    ap.add_argument("--forward-inits", required=True)
    ap.add_argument("--t3-bddl", required=True,
                    help="逆向き goal の述語と、--init-source t3 のときの env に使う")
    ap.add_argument("--t3-inits", default="",
                    help="配布 T3 の .pruned_init。--init-source t3 に必須")
    ap.add_argument("--init-source",
                    choices=["forward-final", "t3", "t3-lying"],
                    default="forward-final",
                    help="逆向きエピソードの初期状態をどこから取るか。"
                         "forward-final=順再生の最終状態（既定。腕は棚の上・本は寝ている）/ "
                         "t3=配布 T3 の init（本番と同じ。本は立ち・腕は home）/ "
                         "t3-lying=T3 の init に**順再生が残した本の姿勢だけ**を"
                         "差し込む（腕は home のまま・本は寝ている）")
    ap.add_argument("--t3-init-index", default="0",
                    help="--init-source t3 のとき使う init の行番号。"
                         "`0` / `0,2,5` / `0-49` / `all`（配布は 50 行ある）")
    ap.add_argument("--grasp-frame", choices=["release", "grasp", "clamped"],
                    default="release",
                    help="掴む姿勢の相対オフセットを順方向のどのフレームから測るか。"
                         "release=放す直前（既定）/ grasp=閉じる指令の瞬間"
                         "（**まだ掴んでいない**ので通常は不可）/ "
                         "clamped=実際に挟み終えた瞬間（立っている物にはこれ）")
    ap.add_argument("--trace-release", type=int, default=0,
                    help="放す前後のこのコマ数を 1 行ずつ出す。**落下の原因を"
                         "「傾いて放した」「触れずに放した」に切り分けるため**")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--episodes", default="", help="コンマ区切り。既定は npz の全部")
    ap.add_argument("--shifts", default="-8,-6,-4,-2,0,2,4,6,8",
                    help="掃く前倒し量（フレーム）。正=グリッパを早める")
    ap.add_argument("--mode", choices=["shift", "phased"], default="shift",
                    help="shift=グリッパ指令の時刻をずらすだけ / "
                         "phased=掴むまで姿勢で判定してから時刻逆向き。"
                         "**phased でも --shifts は効く**——ずらした列から"
                         "「掴む時刻」を決めるので、復路に入る位置が変わる")
    ap.add_argument("--grasp-style", choices=["transfer", "top-down"],
                    default="transfer",
                    help="掴む姿勢の作り方。transfer=順方向デモの相対姿勢を今の"
                         "物の向きに掛け直す（既定）/ top-down=真上から降りて"
                         "直径を挟む姿勢を物の形から組み立てる。**物が順方向デモ"
                         "に無い向きで置かれているときは transfer が使えない**"
                         "（配布 T3 の缶はトレイの中で横倒しだが、順方向デモは"
                         "立っている缶しか掴んでいない）")
    ap.add_argument("--object-symmetry", type=int, default=1,
                    help="物のローカル z 軸まわりで**掴み方として等価**な回転の数。"
                         "1=折り返さない（既定。向きが効く物）/ 2=180 度で等価"
                         "（本）/ 0=連続対称（缶）。等価な向きのうち手首を"
                         "いちばん回さずに済むものを選ぶ")
    ap.add_argument("--max-approach", type=int, default=60,
                    help="接近の各段に許すコマ数の上限。**距離はタスクで違う**——"
                         "本は 30〜41 コマで着くが、腕の home から遠い場面では"
                         "足りない。上げても着いた回の結果は変わらない")
    ap.add_argument("--rot-limit", type=float, default=None,
                    help="1 コマに出してよい回転指令の上限（既定 None ＝ ±1）。"
                         "**配布データの回転は ±0.375 に収まっている**のにこちらは"
                         "毎コマ ±1 を出しており、混ぜて学習すると全ステップが"
                         "勾配クリップに当たる（reference/30 の 8 節）。"
                         "**0.375 にすると |z| の最大が 25.0 → 9.4 になり、"
                         "配布データ自身の裾と同じになる。**缶 20 本の実測では"
                         "コマ数がむしろ 11% 減り、接近誤差も 4.11 → 3.12 mm に"
                         "なった（成分ごとの clip をやめて軸を保つため）。"
                         "`--max-approach` は 200 に上げておくこと")
    ap.add_argument("--rot-tol-deg", type=float, default=5.0,
                    help="`--rot-limit` を効かせたとき、接近を終える向きの許容[度]。"
                         "**位置だけで打ち切ると向きが合わないまま指を閉じる**")
    ap.add_argument("--lift", type=float, default=0.0,
                    help="掴んだ直後に**真上へ**上がる高さ[m]。0 で無効。"
                         "復路の頭は物を置いた高さなので、そのまま追うと荷物を"
                         "下ろしたまま横へ引きずり、**下の皿やトレイを動かす**。"
                         "上がってから、復路が同じ高さに来る点で合流する")
    ap.add_argument("--release", choices=["schedule", "contact"],
                    default="schedule",
                    help="放す時刻の決め方。schedule=記録どおりの開き指令の時刻"
                         "（既定。**順方向の閉じ指令は挟む 8〜17 コマ前に出ている**"
                         "ので、反転すると手が戻り始めてから開いて物を落とす）/ "
                         "contact=順方向が実際に挟み終えたコマまで復路を辿り、"
                         "そこから触れるまで下ろして放す")
    ap.add_argument("--level-before-release", action="store_true",
                    help="放す前に、荷物の**立つ軸が鉛直になるよう手を回す**。"
                         "傾いたまま置くと、放したあとに荷物が自分で起き上がる"
                         "（赤マグは皿の上で 10.5° 傾いており、起き上がりで重心が"
                         "7.5 mm 下がっていた）。--release contact と併せて使う")
    ap.add_argument("--settle", type=int, default=20,
                    help="--release contact のとき、触れるまで 5 mm ずつ下ろす"
                         "コマ数の上限")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="**この長さを超えた回は学習データに書かない。**本番の "
                         "`max_steps_per_episode` が 300 なので、それより長い"
                         "動かし方を教えても必ず打ち切られる。0 で無制限")
    ap.add_argument("--max-tail", type=int, default=-1,
                    help="最後に放したあと手を引く区間の上限コマ数。**-1 で全部"
                         "辿る（既定）・0 で落とす。**この区間は腕が初期姿勢へ"
                         "戻るだけで goal 自体には効かないが、`chunk_size=50` の"
                         "教師として release 手前 50 コマぶんの答えを穴埋めせず"
                         "持たせるのに要る（29_training_data_requirements.md）。"
                         "`--max-steps` の 300 枠には数えない——本番の rollout は"
                         "`done` で打ち切られ、tail は一度も実行されない"
                         "（実測でこの区間は 44〜92 コマある）")
    ap.add_argument("--standoff", type=float, default=0.0,
                    help="掴む姿勢の手前（グリッパの進入軸に沿って）に置く待機点の"
                         "距離[m]。0 で無効。物を押さずに近づくために使う")
    ap.add_argument("--seed", type=int, default=42,
                    help="016 の評価パリティと同じ既定値。fixture 配置を決める")
    ap.add_argument("--render-shifts", default="",
                    help="この shift だけ動画も残す（コンマ区切り）。描画は物理を変えない")
    ap.add_argument("--write-npz", default="",
                    help="逆向き goal が立った回を、この directory へ学習データの"
                         "中継物（npz）として書く。**--mode phased のときだけ**")
    return ap.parse_args()


def build_envs(args, render, rev_region):
    """順方向 env と T3 env を作る。**fixture は両方とも固定する。**

    `--init-source t3` のときは T3 env が再生の舞台にもなるので、描画の要否は
    そちらにも効く。fixture の固定は [pin_fixtures](replay_video_probe.py) が
    面倒を見るので、以降どの関数にも配って回らなくてよい。

    **`--write-npz` のときは両方とも学習用の 2 台で作る。**逆向きを回す env が
    どちらになるかは `--init-source` で変わるので、片方だけ描画にすると
    「画の無い観測から画を取ろうとした」で落ちる。
    """
    use_t3 = args.init_source.startswith("t3")
    cams = list(DATASET_CAMERAS.values()) if args.write_npz else None

    def make(bddl):
        if args.write_npz:
            return make_render_env(bddl, cameras=cams)
        return make_render_env(bddl) if render else make_env(bddl)

    print(f"env を作る（{'学習用 2 台' if args.write_npz else ''}"
          f"描画{'あり' if render or args.write_npz else 'なし'}）...", flush=True)
    env = pin_fixtures(make(args.forward_bddl), args.seed)
    env_t3 = pin_fixtures(make(args.t3_bddl) if use_t3 or args.write_npz
                          else make_env(args.t3_bddl), args.seed)
    env_t3.reset()
    _, t3_half = region_box(env_t3, rev_region)
    print(f"T3 の {rev_region} の箱 半分 = "
          f"{np.round(t3_half[:2] * 1000, 1)} mm", flush=True)
    return env, env_t3, use_t3


def print_release_trace(r, n):
    """放す前後を 1 コマずつ出す。**落下の原因を切り分けるための唯一の窓である。**

    見るのは 3 列。`grip` が負に変わったコマが放した瞬間で、そこで
    `touch` が空なら**触れずに放している**、`tilt` が大きいなら**傾けたまま
    放している**。`z` の落ち方はその結果でしかない。

    掴んで置くのが複数回ある回（two mugs）は、**サイクルごとに**出す。
    """
    for k, name in enumerate(r.get("release_objs", [])):
        tr, a6 = r["trace"][name], r["actions"][:, 6]
        t0 = max(int(r["release_frames"][k]) - 3, 0)
        print(f"    [{k + 1}/{len(r['release_objs'])}] {name}", flush=True)
        for t in range(t0, min(t0 + n, len(tr["z"]))):
            print(f"      t={t:3d} grip={a6[t]:+.0f} z={tr['z'][t] * 1000:7.1f} mm "
                  f"tilt={tr['tilt'][t]:5.1f}° touch={tr['touch'][t]}", flush=True)


def measure(rev_env, rev_goals, r, rev_goal, starts, obj_pos_start=None):
    """1 本の逆向き再生から、**その再生を回した env だけで測れる**値を出す。

    順方向 env の値（本の出発点など）と混ぜないこと。`--init-source t3` では
    逆向きを別の env で回すので、順方向側の座標と引き算しても意味が無い。

    **物は 1 つとは限らない**（two mugs）。物ごとに測ったうえで、代表値は
    **いちばん領域中心から遠い物**のものを出す。1 つのときは同じ値になる。
    """
    per = {}
    for _, name, region in rev_goals:
        end, _ = body_pose(rev_env, name)
        region_pos, region_half = region_box(rev_env, region)
        per[name] = {
            "moved_mm": round(float(np.linalg.norm(
                end - starts[name])) * 1000, 1),
            "z": round(float(end[2]), 3),
            # `On(物, 領域site)` は TargetZone.under() で判定される。
            # 「領域の箱の中に居るか」を xy に分けて出す（落ちた理由を読むため）。
            "region_dxy_mm": [round(float(v) * 1000, 1)
                              for v in (end - region_pos)[:2]],
            "region_half_mm": [round(float(v) * 1000, 1) for v in region_half[:2]],
        }
    worst = max(per, key=lambda n: float(np.hypot(*per[n]["region_dxy_mm"])))
    row = {
        # 物が「逆向きの出発点」からどれだけ動いたか（掴んで運べたかの目安）
        "book_moved_mm": per[worst]["moved_mm"],
        "book_z": per[worst]["z"],
        "reverse_goal": rev_goal,
        "region_dxy_mm": per[worst]["region_dxy_mm"],
        "region_half_mm": per[worst]["region_half_mm"],
        "n_steps": int(r["n_steps"]),
    }
    if len(per) > 1:
        row["per_object"] = per
    if obj_pos_start is not None:
        now = all_object_pos(rev_env)
        handled = {name for _, name, _ in rev_goals}
        moved = {n: round(float(np.linalg.norm(now[n] - p)) * 1000, 1)
                 for n, p in obj_pos_start.items() if n not in handled}
        # **1 mm 未満は数えない。**シムの整定でこの程度は常に動く。
        row["disturb_mm"] = {n: v for n, v in moved.items() if v >= 1.0}
        row["disturb_max_mm"] = max(moved.values(), default=0.0)
    # **放した瞬間と終わりの傾き。**「置いたあとに倒れていないか」は goal では
    # 見えない——`On(物, 領域)` は箱の中に居るかだけを見るので、**倒れたまま
    # 箱の中に居れば ○ になる。**学習データとしては倒れた回を入れたくないので、
    # 数字で残す（`--level-before-release` を付けるかの判断もこれで決める）。
    if "trace" in r and r.get("release_frames"):
        tilts = {}
        for k, name in enumerate(r.get("release_objs", [])):
            tr = r["trace"][name]
            t0 = min(int(r["release_frames"][k]), len(tr["tilt"]) - 1)
            tilts[name] = {"release": round(float(tr["tilt"][t0]), 1),
                           "end": round(float(tr["tilt"][-1]), 1)}
        row["tilt_deg"] = tilts
        row["tilt_end_max_deg"] = round(
            max(v["end"] for v in tilts.values()), 1)
    if "n_extra" in r:                          # phased のときだけ付く
        row.update(n_extra=int(r["n_extra"]), n_approach=int(r["n_approach"]),
                   n_standoff=int(r["n_standoff"]), n_reach=int(r["n_reach"]),
                   n_lift=int(r["n_lift"]), n_floor=int(r["n_floor"]),
                   n_backoff=int(r.get("n_backoff", 0)),
                   n_settle=int(r["n_settle"]), n_cycles=int(r["n_cycles"]),
                   n_path=int(r["n_path"]), n_tail=int(r["n_tail"]),
                   per_cycle=r.get("per_cycle"),
                   drop_mm=r.get("drop_mm"),
                   approach_err_mm=r["approach_err_mm"],
                   approach_err_mm_all=r.get("approach_err_mm_all"))
    else:
        row["err_median_mm"] = round(float(np.median(r["err_m"])) * 1000, 1)
    return row


def settle_frames(r, tol_mm=0.5):
    """**放してから物が動かなくなるまでの猶予コマ数**をサイクルごとに返す。

    最後の姿勢を基準に、そこから `tol_mm` 以内に入って**戻らなくなる**最初の
    コマを探す。腕が初期姿勢へ戻る区間は学習データに要らないが、**置いた物が
    落ち着く前に切ると、動いている途中で終わったデータになる。**その境目を
    測るための値である。
    """
    out = {}
    for k, name in enumerate(r.get("release_objs", [])):
        pos = np.asarray(r["trace"][name]["pos"])
        t0 = int(r["release_frames"][k])
        d = np.linalg.norm(pos[t0:] - pos[-1], axis=1) * 1000
        moving = np.where(d > tol_mm)[0]
        out[name] = {"frames": int(moving[-1] + 1) if len(moving) else 0,
                     # k コマ残したときに「まだ動く量」[mm]。切る所を決める値。
                     "left_mm": {k: round(float(d[k]), 2)
                                 for k in (0, 5, 10, 15, 20, 30) if k < len(d)}}
    return out


def pose_now(env, obs, name, use_obs):
    """逆向き env のいまの物の姿勢（robosuite 規約 (x,y,z,w)）。"""
    if use_obs:
        return (np.asarray(obs[f"{name}_pos"]).copy(),
                np.asarray(obs[f"{name}_quat"]).copy())
    return body_pose(env, name)


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
    npz_dir = Path(args.write_npz) if args.write_npz else None
    if npz_dir is not None:
        if args.mode != "phased":
            raise SystemExit("--write-npz は --mode phased のときだけ使える")
        npz_dir.mkdir(parents=True, exist_ok=True)

    fwd_goals, rev_goals = goals_of(args.forward_bddl), goals_of(args.t3_bddl)
    fwd_pred, obj, fwd_region = fwd_goals[0]
    rev_pred, rev_obj, rev_region = rev_goals[0]
    # **動かす物は 1 つとは限らない**（two mugs は 2 つ）。順再生では全部を追う。
    watch = [o for _, o, _ in rev_goals]
    for pred, o, reg in fwd_goals:
        print(f"順方向 goal: {pred} {o} -> {reg}", flush=True)
    for pred, o, reg in rev_goals:
        print(f"逆向き goal: {pred} {o} -> {reg}", flush=True)

    env, env_t3, use_t3 = build_envs(args, render, rev_region)
    # 順方向 goal の領域 site（＝本を置いた棚）を、env をまたぐ姿勢の基準にする。
    # 同じ fixture が両方の env にあるので、タスク名を書かずに対応が取れる。
    # **t3-lying のときだけ要る。**順方向 goal の相手が site とは限らないので
    # （マグは `plate_1` という物が相手）、要らないときに引くと落ちる。
    fwd_frame = t3_frame = None
    if args.init_source == "t3-lying":
        fwd_frame, t3_frame = (site_frame(env, fwd_region),
                               site_frame(env_t3, fwd_region))
        print(f"基準 site {fwd_region}: 順方向 {np.round(fwd_frame[0], 4)} / "
              f"T3 {np.round(t3_frame[0], 4)}  "
              f"ずれ {np.linalg.norm(t3_frame[0] - fwd_frame[0]) * 1000:.1f} mm",
              flush=True)
    inits = load_pruned_init(args.forward_inits)
    # 50 本ぶんの reset は 25 秒かかるが、結果はエピソードに依らない。1 度だけ測る。
    probed = probe_init_eef(env, inits)
    print(f"初期状態 {len(inits)} 本を下見した", flush=True)
    t3_inits, t3_rows = None, [None]      # forward-final では init は 1 通り
    if use_t3:
        if not args.t3_inits:
            raise SystemExit("--init-source t3 には --t3-inits が要る")
        t3_inits = load_pruned_init(args.t3_inits)
        t3_rows = index_spec(args.t3_init_index, len(t3_inits))
        print(f"T3 の初期状態 {len(t3_inits)} 本のうち {len(t3_rows)} 本を使う "
              f"(#{t3_rows[0]}..#{t3_rows[-1]})", flush=True)

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
                      states, track_obj=watch)
        fwd_ok = bool(env.check_success())
        fwd_final = env.env.sim.get_state().flatten()
        book_fwd_end, book_quat_fwd_end = body_pose(env, obj)
        # 順再生が本を残した姿勢。t3-lying はこれを T3 の init へ差し込む。
        book_fwd_qpos = object_qpos(env, obj)
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
        # **順方向デモを「掴んで放す」単位に切り分ける。**単体タスクは 1 つ、
        # two mugs は 2 つになる。どの物を相手にした回かは**動いた量で決める。**
        fwd_cycles = grasp_cycles(actions[:, 6], states,
                                  {n: fwd["obj"][n]["pos"] for n in watch})
        for c in fwd_cycles:
            print(f"  掴んで放す f{c['close']}→f{c['open']}"
                  f"（挟んだのは f{c['clamp']}）: {c['obj']} が "
                  f"{c['moved_mm']:.0f} mm 動いた", flush=True)
        # **goal が求める物を 1 つでも動かせていないデモは使えない。**掴む姿勢は
        # 再生結果から測るので、動かしていない物の値は物理的な意味を持たない
        # （[reference/30 の 5 節]）。two mugs では「2 つとも動いたか」になる。
        missing = [n for n in watch if n not in {c["obj"] for c in fwd_cycles}]
        if missing:
            print(f"  !! 順再生が {', '.join(missing)} を動かせていない。"
                  f"この ep は飛ばす", flush=True)
            report.append({"episode": ep, "n_actions": int(len(actions)),
                           "init_index": idx, "forward_goal": fwd_ok,
                           "skipped": f"順再生が動かさなかった: {missing}",
                           "skipped_objs": missing,
                           "n_goal": 0, "n_trials": 0, "shifts": []})
            continue
        T = len(actions)
        # 逆向きは**順方向の最後のサイクルから**辿る。
        rev_cycles = []
        for c in reversed(fwd_cycles):
            # 掴む姿勢を測るコマ。clamped=実際に挟んだ / grasp=閉じ指令 /
            # release=放す直前。**1 サイクルのときは従来と同じ値になる。**
            j = {"clamped": c["clamp"], "grasp": c["close"],
                 "release": c["open"]}[args.grasp_frame]
            rev_cycles.append({
                "obj": c["obj"], "j": min(max(j, 0), len(fwd["eef_pos"]) - 1),
                # 復路の入口は「順方向が放したコマ」の反転
                "start_i": T - 1 - c["open"],
                # 放す位置は「順方向が実際に挟み終えたコマ」の反転。**開き指令の
                # 時刻ではない**——それは挟む 8〜17 コマ前なので、反転すると空中
                # で放す。
                "release_i": (T - 1 - c["clamp"]
                              if args.release == "contact" else None),
                # **置いた先で鉛直を向くべき軸は、順方向の初期姿勢から測る。**
                # 逆向きはそこへ戻す動きなので、いま置かれている向きではない。
                "place_axis": standing_axis(fwd["obj"][c["obj"]]["quat"][0]),
                "level": args.level_before_release})

        # 2) 逆向きを回す env と初期状態を決める（軌道の作り方とは独立の軸）。
        #    **順再生（1)）は init に依らないので、外側でちょうど 1 回だけ回す。**
        rev_env = env_t3 if use_t3 else env
        rows = []
        for t3_i in t3_rows:
            if use_t3:
                rev_init = t3_inits[t3_i]
                if args.init_source == "t3-lying":
                    rev_init = with_object_qpos(
                        rev_env, rev_init, obj,
                        rebase_qpos(book_fwd_qpos, fwd_frame, t3_frame))
                obs_rev = reset_to(rev_env, rev_init)
            else:
                rev_init, obs_rev = fwd_final, None
            # **この時点の env は逆向きの初期状態そのもの。**ここで全部の物の
            # 位置を控えておき、終わりと引き算して「押して動かした量」を出す。
            obj_pos_start = all_object_pos(rev_env)
            starts = {n: pose_now(rev_env, obs_rev, n, use_t3)[0] for n in watch}
            book_rev_start, book_rev_quat = pose_now(
                rev_env, obs_rev, rev_cycles[0]["obj"], use_t3)
            upright = object_is_upright(book_rev_quat)
            yaw = object_yaw_deg(book_rev_quat)

            # 掴む姿勢は**物の座標系で測った相対姿勢を、今の物の姿勢に掛け直して**
            # 作る。どの向きに置かれていても狙える。shift に依らないので 1 度だけ。
            # **サイクルごとに引き直す。**two mugs は物が違えば掴む所も違う。
            from robosuite.utils import transform_utils as T

            start_eef = (np.asarray(obs_rev["robot0_eef_pos"]) if use_t3
                         else np.asarray(fwd["eef_pos"][-1]))
            start_quat = (np.asarray(obs_rev["robot0_eef_quat"]) if use_t3
                          else T.axisangle2quat(fwd["eef_aa"][-1]))
            cycles = []
            for cy in rev_cycles:
                name, j = cy["obj"], cy["j"]
                pos_now, quat_now = pose_now(rev_env, obs_rev, name, use_t3)
                fv = fwd_view(fwd, name)
                if args.grasp_style == "top-down":
                    grasp_pose = top_down_grasp(
                        pos_now, quat_now, cy["place_axis"],
                        grasp_backoff(fv, j), ref_quat=start_quat)
                else:
                    grasp_pose = grasp_pose_from_forward(
                        fv, j, pos_now, quat_now, symmetry=args.object_symmetry)
                cycles.append({**cy, "grasp_pose": grasp_pose})
                # 手首を何度回す必要があるか／物が順方向のときから何度回っているか
                d_wrist = rot_angle_deg(T.axisangle2quat(grasp_pose[3:6]),
                                        start_quat)
                d_obj = rot_angle_deg(quat_now, fv["obj_quat"][j])
                print(f"  init #{t3_i} {name} {np.round(pos_now, 3)} / "
                      f"腕 {np.round(start_eef, 3)} → 掴む先 "
                      f"{np.round(grasp_pose[:3], 3)}（"
                      f"{np.linalg.norm(grasp_pose[:3] - start_eef) * 1000:.0f} mm 先"
                      f"・手首 {d_wrist:.0f}° 回す／物は順方向 f{j} から "
                      f"{d_obj:.0f}° 回っている）", flush=True)

            for k in shifts:
                tag = f"ep{ep}" + (f"_i{t3_i}" if use_t3 else "")
                vid = (vid_dir / f"{args.mode}_{tag}_rev_k{k:+03d}.mp4"
                       if k in render else None)
                grip = shift_gripper(grip_rev, k)
                if args.mode == "phased":
                    r = rollout_phased(rev_env, rev_init, ref, grip, vid,
                                       cycles=cycles,
                                       standoff=args.standoff,
                                       max_approach=args.max_approach,
                                       rot_limit=args.rot_limit,
                                       rot_tol_deg=args.rot_tol_deg,
                                       lift=args.lift, settle=args.settle,
                                       max_tail=args.max_tail,
                                       capture=bool(args.write_npz))
                else:
                    r = rollout_tracking(rev_env, rev_init, ref, grip, vid,
                                        rot_limit=args.rot_limit)
                # **判定は逆向きを回した env で行う。**T3 env で回したならそのまま
                # `check_success()` が T3 の goal。順方向 env で回したなら、本の姿勢を
                # T3 env へ差し込んで判定する（順方向 env の同名 region は 7.5 倍狭い）。
                if use_t3:
                    rev_goal = bool(rev_env.check_success())
                else:
                    rev_goal = eval_in_t3(env_t3, rev_obj, rev_pred, rev_region,
                                          *body_pose(rev_env, obj))
                row = {"t3_init": t3_i, "shift": k, "object_upright": upright,
                       "book_init_yaw_deg": round(yaw, 1),
                       "book_init_xy": [round(float(v), 4)
                                        for v in book_rev_start[:2]],
                       **measure(rev_env, rev_goals, r, rev_goal,
                                 starts, obj_pos_start)}
                if not use_t3:
                    # 順方向と同じ env で回したときだけ、順方向側の座標と比べられる
                    book_end, _ = body_pose(rev_env, obj)
                    row["book_to_start_mm"] = round(float(np.linalg.norm(
                        book_end - book_start)) * 1000, 1)
                    row["forward_goal_undone"] = not bool(env.check_success())
                rows.append(row)
                # **300 は本番の `max_steps_per_episode`——release までに goal に
                # 届く必要のある長さである。**tail（放したあと初期姿勢へ戻る区間）
                # は本番の rollout では `done` で打ち切られて一度も実行されない
                # （`pipeline/rollout.py` の `if done: break`）ので、300 の枠は
                # release までの長さ（`n_steps - n_tail`）にだけ掛ける。tail は
                # 300 を超えて残ってよい——むしろ `chunk_size=50` の教師として、
                # release 手前 50 コマぶんの答えを穴埋めせず持たせるために要る
                # （`docs/reference/29_training_data_requirements.md` 7.3）。
                n_pre_tail = int(r["n_steps"]) - int(r["n_tail"])
                too_long = (args.max_steps > 0 and n_pre_tail > args.max_steps)
                row["too_long"] = too_long
                row["n_pre_tail"] = n_pre_tail
                if args.write_npz and rev_goal and too_long:
                    print(f"    !! release まで {n_pre_tail} コマは打ち切り "
                          f"{args.max_steps} を超えるので書かない"
                          f"（全体 {r['n_steps']} コマ、tail {r['n_tail']} コマ含む）",
                          flush=True)
                if args.write_npz and rev_goal and not too_long:
                    # **通った回だけ書く。**落ちた回は学習に入れない方針なので、
                    # 100 MB 級の npz をディスクに残す意味が無い。
                    write_episode_npz(npz_dir / f"rev_{tag}_k{k:+03d}.npz",
                                      rev_env, r, rev_goal)
                    row["npz"] = f"rev_{tag}_k{k:+03d}.npz"
                extra = (f"  接近 {row['n_approach']:2d} コマで "
                         f"{row['approach_err_mm']:6.2f} mm"
                         if "n_approach" in row else
                         f"  腕ずれ {row['err_median_mm']:.1f} mm")
                if "trace" in r:
                    row["settle_frames"] = settle_frames(r)
                if row.get("drop_mm") is not None:
                    extra += f"  落下 {row['drop_mm']:5.1f} mm"
                if "tilt_end_max_deg" in row:
                    rel = max(v["release"] for v in row["tilt_deg"].values())
                    extra += (f"  傾き 放す{rel:5.1f}°→終{row['tilt_end_max_deg']:5.1f}°")
                # **`"obj_z"` を見ていたので、この窓は黙って閉じていた**（9/11 に直した）。
                # `rollout_phased` はその名前の値をもう返しておらず、`--trace-release`
                # は何を渡しても 1 行も出さないままだった。見るのは `trace` である。
                if args.trace_release and "trace" in r:
                    print_release_trace(r, args.trace_release)
                if "disturb_max_mm" in row:
                    extra += (f"  他の物 {row['disturb_max_mm']:5.1f} mm "
                              f"{row['disturb_mm'] or ''}")
                head = (f"  init #{t3_i:2d} yaw {yaw:+6.1f}°" if use_t3
                        else "  forward-final")
                print(f"{head} shift {k:+3d}: "
                      f"本の移動 {row['book_moved_mm']:7.1f} mm  "
                      f"領域dxy {row['region_dxy_mm']} "
                      f"(箱±{row['region_half_mm'][0]:.0f})  "
                      f"z={row['book_z']:.3f}  "
                      f"goal={'○' if rev_goal else '×'}{extra}", flush=True)

        # 最良の選び方: goal が立ったものを優先し、その中で領域中心に近いもの
        best = min(rows, key=lambda r: (not r["reverse_goal"],
                                        float(np.hypot(*r["region_dxy_mm"]))))
        n_ok = sum(r["reverse_goal"] for r in rows)
        print(f"  --> {n_ok}/{len(rows)} 本で逆向き goal。最良 "
              f"init #{best['t3_init']} shift {best['shift']:+d} "
              f"(goal={best['reverse_goal']}, "
              f"領域中心まで {np.hypot(*best['region_dxy_mm']):.1f} mm)", flush=True)
        report.append({"episode": ep, "n_actions": int(len(actions)),
                       "init_index": idx, "forward_goal": fwd_ok,
                       "grasp_frame_index": j, "n_goal": int(n_ok),
                       "n_trials": len(rows),
                       "book_start": [round(float(x), 4) for x in book_start],
                       "book_forward_end": [round(float(x), 4)
                                            for x in book_fwd_end],
                       "shifts": rows, "best_shift": best["shift"]})

    print("\n=== まとめ ===", flush=True)
    for rec in report:
        print(f"  ep {rec['episode']}: {rec['n_goal']}/{rec['n_trials']} 本成功",
              flush=True)
    path = out_dir / "gripper_shift_sweep.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n書き出し: {path}", flush=True)


if __name__ == "__main__":
    main()
