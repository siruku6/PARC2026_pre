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

# 学習データの 2 台と、シムの観測の鍵の対応。**この 2 台以外は要らない**
# （`lerobot/libero_plus` のカメラは `front` と `wrist` の 2 つだけ）。
DATASET_CAMERAS = {"front": "agentview", "wrist": "robot0_eye_in_hand"}


def make_render_env(bddl_file: str, control_freq: int = 20,
                    camera_size: int = 256, cameras=None):
    """`reversal_probe.make_env` と同じ物理で、描画だけ足した env。

    `use_camera_obs=True` は観測に画を足すだけで物理には触れないので、
    再生の結果は `reversal_probe` と同じになる。

    `cameras` を渡すとその台数ぶん観測に画が乗る。学習データを書くときは
    `DATASET_CAMERAS.values()` を渡すこと（動画を見るだけなら 1 台でよい）。
    """
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=list(cameras) if cameras else [CAMERA],
        camera_heights=camera_size,
        camera_widths=camera_size,
        control_freq=control_freq,
        horizon=100000,
        ignore_done=True,
    )


def obs_state8(obs):
    """`observation.state`（8 次元）を観測から作る。

    `[eef_pos(3), 軸角(3), 指 qpos(2)]` で**生スケール**。評価時に
    `LiberoProcessorStep` が組むものと同じ並びである（reference/10 の 2 節）。
    """
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        eef_axisangle(obs).astype(np.float32),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ]).astype(np.float32)


def dataset_image(obs, camera):
    """観測の画を、**学習データと同じ向き**にして返す。

    `[::-1, ::-1]` の 180 度回転である。`scripted_demo` が npz へ書くときと
    同じで、評価時は `LiberoProcessorStep` が同じ回転を入れる（reference/10）。
    **向きを揃えないと、学習した見え方と評価の見え方が食い違う。**
    動画用の `[::-1]`（上下だけ）とは別物なので、混ぜないこと。
    """
    return np.asarray(obs[f"{camera}_image"])[::-1, ::-1].copy()


def pin_fixtures(env, seed):
    """`reset()` のたびに fixture の配置を戻す env にして返す。

    棚・机などの fixture の位置と向きは `qpos` ではなく **`model.body_pos` /
    `model.body_quat`** に入るので `set_state_from_flattened` では復らず、
    **`reset()` のたびに引き直される**（実測 最大 18.7 mm）。固定しないと、
    同じ初期状態・同じ action を流しても本の着地が 2 か所に割れる（実測 35 mm）。
    固定すれば同一プロセス内の繰り返しはビット一致する。

    **`seed` も要る。**固定するのは「1 回目の reset が引いた配置」なので、
    その 1 回目が run ごとに違えばプロセスをまたいだ比較ができない。016 の
    [_016_libero_env.py](../probe/_016_libero_env.py) と同じ `seed=42` を使う。

    **包む形にしているのは、呼び出し側へ配って回ると必ず忘れるからである。**
    以前は配置を写し取って `reset_to` などへ引数で配っていたが、同じファイルの
    `main()` が既に渡し忘れており、黙って 35 mm ずれた数字が出ていた。

    注意: `model` は `hard_reset=True` のたびに**作り直される**ので、写し取った
    オブジェクトを持ち回して比べても差は出ない（1 度これで「変化なし」と
    誤判定した）。毎回 `env.env.sim.model` を取り直すこと。
    """

    class PinnedEnv:
        """fixture を固定した env。それ以外の属性は元の env へ素通しする。"""

        def __init__(self, inner_env):
            inner_env.seed(seed)
            inner_env.reset()
            model = inner_env.env.sim.model
            self._env = inner_env
            self._pins = (np.asarray(model.body_pos).copy(),
                          np.asarray(model.body_quat).copy())

        def reset(self, *a, **kw):
            obs = self._env.reset(*a, **kw)
            model = self._env.env.sim.model      # reset で作り直されている
            model.body_pos[:], model.body_quat[:] = self._pins
            return obs

        def __getattr__(self, name):
            return getattr(self._env, name)

    return PinnedEnv(env)


def reset_to(env, init_state):
    """初期状態へ入れる。**先に `env.reset()` を通すこと。**

    `set_state_from_flattened` は qpos/qvel しか戻さない。OSC コントローラが持つ
    内部の目標姿勢（`goal_pos` / `goal_ori`）は**前の再生の続きのまま残る**ので、
    reset を挟まないと「同じ初期状態から始めたのに前回の再生に依存して結果が変わる」。
    実際、これを入れる前は同じ初期状態選びが run ごとに違う番号を返していた。

    fixture の配置を揃えたい比較では、env を [pin_fixtures](#) で包んでおくこと。
    """
    env.reset()                      # コントローラの内部状態まで初期化する
    inner = env.env
    sim = inner.sim
    sim.set_state_from_flattened(np.asarray(init_state, dtype=np.float32))
    sim.forward()
    obs = None
    for _ in range(SETTLE_STEPS):
        sim.data.qvel[:] = 0
        obs, _, _, _ = inner.step(np.zeros(7))
    return obs


def probe_init_eef(env, inits):
    """各初期状態を実際に復元して eef 位置を測り、`(N, 3)` で返す。

    **高い。**1 本ごとに `reset()` が要るので 50 本で 25 秒かかる（`reset` は
    `hard_reset=True` でモデルごと作り直すため）。**結果はエピソードに依らない**
    ので、複数エピソードを回すなら 1 度測って使い回すこと。

    復元だけで済ませる近道（`reset()` を飛ばして `set_state_from_flattened`）は
    使えない。実測で eef が最大 27 mm ずれ、選ばれる初期状態が変わった。
    """
    return np.array([np.asarray(reset_to(env, st)["robot0_eef_pos"])
                     for st in inits])


def pick_init(env, inits, ref_eef_pos, probed=None):
    """デモ 1 フレーム目の eef_pos にいちばん近い初期状態を選ぶ。

    デモは特定の初期状態から録られている。`inits[0]` 決め打ちで再生すると
    「初期状態が違うから序盤で発散した」のか「action の規約が違う」のかが
    分離できない（006 の候補 (c) と (b) の混線）。

    `probed` に [probe_init_eef](#) の結果を渡すと、測り直さずに選ぶだけになる。
    """
    if probed is None:
        probed = probe_init_eef(env, inits)
    d = np.linalg.norm(np.asarray(probed) - np.asarray(ref_eef_pos), axis=1)
    i = int(np.argmin(d))
    return i, float(d[i]), d.tolist()


def rollout(env, init_state, actions, video_path=None, ref_states=None,
            track_obj=None):
    """初期状態から action 列を再生する。動画と、フレーム毎の eef_pos を返す。

    `track_obj` に物体名（または名前の並び）を渡すと、その姿勢も毎コマ残す。
    逆向きの掴む姿勢を [grasp_pose_from_forward](#) で作るのに要る。
    **並びで渡すと `obj[名前]` に全部入る**（two mugs のように 1 本のデモが
    2 つの物を動かす場合に要る）。`obj_pos` / `obj_quat` は先頭の物のもの。
    """
    obs = reset_to(env, init_state)
    frames, eef, aa = [], [], []

    names = ([track_obj] if isinstance(track_obj, str)
             else list(track_obj or []))
    obj_p = {n: [] for n in names}
    obj_q = {n: [] for n in names}

    def record():
        if video_path is not None:
            frames.append(obs[f"{CAMERA}_image"][::-1])
        eef.append(np.asarray(obs["robot0_eef_pos"]).copy())
        # 逆向きの掴む姿勢をここから作るので、向きも残す（軸角。state 列と同じ形）
        aa.append(eef_axisangle(obs))
        for name in names:
            # 掴む瞬間の「本から見た手の位置」を作るために、物体の姿勢も残す
            obj_p[name].append(np.asarray(obs[f"{name}_pos"]).copy())
            obj_q[name].append(np.asarray(obs[f"{name}_quat"]).copy())

    record()
    inner = env.env
    for a in actions:
        obs, _, _, _ = inner.step(np.asarray(a, dtype=np.float32))
        record()

    if video_path is not None:
        write_video(frames, video_path)
    eef = np.asarray(eef)

    out = {"n_steps": len(actions), "eef_pos": eef, "eef_aa": np.asarray(aa)}
    if names:
        out["obj"] = {n: {"pos": np.asarray(obj_p[n]),
                          "quat": np.asarray(obj_q[n])} for n in names}
        out["obj_pos"] = out["obj"][names[0]]["pos"]
        out["obj_quat"] = out["obj"][names[0]]["quat"]
    if ref_states is not None:
        # 再生 t ステップ後の位置を、データの state[t] と突き合わせる。
        # データの state[t] は「action[t] を撃つ**前**の姿勢」なので、
        # 再生側も同じ並び（撃つ前）で比べる。
        n = min(len(ref_states), len(eef))
        err = np.linalg.norm(eef[:n] - np.asarray(ref_states)[:n, :3], axis=1)
        out["err_m"] = err
    return out


def _rz(theta):
    """物のローカル z 軸まわりの回転行列。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def symmetry_rotations(order, n_continuous=72):
    """**掴み方として等価な**、物のローカル z 軸まわりの回転を並べる。

    | `order` | 意味 | 例 |
    |---|---|---|
    | 1 | 折り返さない（向きがそのまま効く） | 既定 |
    | 2 | 180 度で等価 | 本——最短辺が同じ直線に乗るので、どちらから挟んでも同じ |
    | 0 | 連続対称 | 缶——軸まわりのどの角度でも同じ |

    連続は 5 度刻みで並べる。残差は最大 2.5 度で、OSC の 1 コマ 0.5 rad から
    見れば無視できる。
    """
    n = n_continuous if order == 0 else max(int(order), 1)
    return [_rz(2 * np.pi * k / n) for k in range(n)]


def fwd_view(fwd, name):
    """順再生の結果を、**その物を追っていたことにして**見せる。

    `grasp_pose_from_forward` も `grasp_backoff` も `fwd["obj_pos"]` を見るので、
    物ごとに掴む姿勢を引き直すには、その物の列に差し替えた見え方が要る。
    """
    return {**fwd, "obj_pos": fwd["obj"][name]["pos"],
            "obj_quat": fwd["obj"][name]["quat"]}


def grasp_pose_from_forward(fwd, j, obj_pos_now, obj_quat_now, symmetry=1):
    """逆向きの「掴む姿勢」を、**本の今の姿勢に合わせて**作る。

    **なぜ基準経路の姿勢をそのまま使えないか。** 順再生はデモと同じ場所へ本を
    置くとは限らない（実測で 25〜35 mm ずれた）。さらに**放した後に本が動く**
    ——指が離れた瞬間に落ちたり滑ったりする。基準経路の姿勢へ腕を 3 mm で
    寄せても、本がそこに無ければ空を掴む（ep4062・4063 がこれだった）。

    そこで順再生の**掴んでいた時刻 `j`** で「本から見た手の姿勢」を測り、
    それを**今の本の姿勢に掛け直す**。本がどこへ動いていても掴みに行ける。

    **`symmetry` を渡すと、等価な掴み方のうち「順方向デモの世界座標での掴む向き」に
    いちばん近いものを選ぶ。**物の回転をそのまま手首へ写すのは、向きが効く物（本）
    には正しいが、回転体には有害である。実測: 配布 T3 の缶は順方向デモの掴む瞬間から
    **166〜179 度**回っており、そのまま写すと手首に 167〜176 度の回転を要求して
    腕が発散した（240 コマかけて誤差が 319 mm → 451 mm と**開いた**）。

    **近さを測る相手は「今の手首の向き」ではない。**そちらで選ぶと、手首は回らずに
    済む代わりに**上から掴む向きが失われ**、横向きのままトレイへ押し当てて終わった
    （実測でそうなった）。順方向デモの向きは、その場面で実際に掴めた向きである。
    そこからの差だけを詰め、物の自転は打ち消す。物が動いていなければ選ばれるのは
    恒等回転なので、`symmetry=1` の従来どおりの結果と一致する。

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
    obj_pos_now = np.asarray(obj_pos_now, dtype=np.float64)
    cands = [(obj_pos_now + r_obj_now @ (r_s @ rel_pos), r_obj_now @ r_s @ r_rel)
             for r_s in symmetry_rotations(symmetry)]
    if len(cands) > 1:
        cands.sort(key=lambda c: float(np.linalg.norm(
            T.quat2axisangle(T.mat2quat(c[1] @ r_eef_then.T)))))
    pos, r_best = cands[0]
    return np.concatenate([pos, np.asarray(T.quat2axisangle(T.mat2quat(r_best)))])


def level_pose(obs, obj, axis_local, clearance=0.0):
    """荷物の**立つ軸を鉛直へ**向け直す手の姿勢を返す（軸角 6 要素）。

    **手の原点まわりに回すと荷物が振れる。**荷物は手から離れた所にあるので、
    **荷物を中心に手を回す**（位置も一緒に回す）。`clearance` を渡すと、
    回す前にその高さだけ持ち上げる——触れたまま回すと机に擦り付けるため。

    傾いたまま置くと、放したあとに荷物が自分で起き上がる。実測では赤マグが
    10.5° 傾いたまま置かれ、起き上がりで重心が 7.5 mm 下がっていた。
    """
    from robosuite.utils import transform_utils as T

    p_obj = np.asarray(obs[f"{obj}_pos"], dtype=np.float64)
    axis = T.quat2mat(np.asarray(obs[f"{obj}_quat"],
                                 dtype=np.float64))[:, int(axis_local)]
    if axis[2] < 0:
        axis = -axis                      # 立つ軸は上下どちら向きでも同じ
    up = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, up)
    n = float(np.linalg.norm(v))
    r_fix = (np.eye(3) if n < 1e-8 else
             T.quat2mat(T.axisangle2quat(
                 v / n * float(np.arccos(np.clip(axis @ up, -1.0, 1.0))))))
    p_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    r_eef = T.quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    pos = p_obj + r_fix @ (p_eef - p_obj) + np.array([0.0, 0.0, clearance])
    return np.concatenate([pos, np.asarray(T.quat2axisangle(
        T.mat2quat(r_fix @ r_eef)))])


def tilt_deg(quat, axis_local):
    """物の「立つ軸」が鉛直から何度ずれているか。**傾けて放していないかを見る。**"""
    from robosuite.utils import transform_utils as T

    v = T.quat2mat(np.asarray(quat, dtype=np.float64))[:, int(axis_local)]
    return float(np.degrees(np.arccos(min(1.0, abs(float(v[2]))))))


def contacting_bodies(env, obj):
    """`obj` に**いま触れている他の物**の body 名。手と自分自身は除く。

    **「置いた」を時刻ではなく接触で決めるために使う。**順方向の閉じ指令は
    実際に挟む 8〜17 コマ前に出ている（[clamp_frame](gripper_shift_sweep.py)）
    ので、その時刻をそのまま反転して放すと、手はもう戻り始めていて**物を
    落とす**。触れた所で止めて放せば落ちない。

    body 名で見分ける。物の geom は `<物>_main` に付き、手の geom は
    `robot0_` / `gripper0_` で始まる。
    """
    sim = env.env.sim

    def body_of(geom):
        return sim.model.body_id2name(sim.model.geom_bodyid[geom])

    def is_own(name):
        return name == obj or name.startswith(obj + "_")

    hit = set()
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        a, b = body_of(c.geom1), body_of(c.geom2)
        for x, y in ((a, b), (b, a)):
            if is_own(x) and not (is_own(y) or y.startswith("robot")
                                  or y.startswith("gripper")):
                hit.add(y)
    return hit


def standing_axis(quat):
    """物が置かれているとき、**鉛直を向いているローカル軸**の番号（0=x,1=y,2=z）。

    回転体の対称軸はこの軸である。順方向デモの初期姿勢から測るので、**物ごとに
    手で決めない**。実測: 本はローカル z、`alphabet_soup_1` はローカル y。
    """
    from robosuite.utils import transform_utils as T

    r = T.quat2mat(np.asarray(quat, dtype=np.float64))
    return int(np.argmax(np.abs(r[2, :3])))


def grasp_backoff(fwd, j):
    """順方向デモが、掴む点から**手をどれだけ引いた位置**に居たか [m]。

    進入方向に沿って測る。`top_down_grasp` が手の高さを決めるのに使う。
    """
    from robosuite.utils import transform_utils as T

    r_eef = T.quat2mat(T.axisangle2quat(
        np.asarray(fwd["eef_aa"][j], dtype=np.float64)))
    rel = (np.asarray(fwd["eef_pos"][j], dtype=np.float64)
           - np.asarray(fwd["obj_pos"][j], dtype=np.float64))
    return float(-rel @ r_eef[:, 2])


def top_down_grasp(obj_pos, obj_quat, axis_local, back_offset, ref_quat=None):
    """**真上から降りて直径を挟む**掴む姿勢を、物の形から組み立てて返す。

    順方向デモの掴み方が移せないときの受け皿である。実測: 配布 T3 の缶は
    トレイの中で**横倒し**（対称軸が水平）なのに、順方向デモは**立っている缶を
    軸に沿って真上から**掴んでいる。缶の対称軸まわりに回しても軸の向きは
    変わらないので、移した進入方向は「円い端面へ真横から」のままで、トレイの
    壁に阻まれる（実測 240 コマかけて 60 mm 手前で停止）。**真上からの握りは
    順方向デモに 1 コマも無いので、ここで作るしかない。**

    握り自体は順方向と同じである。実測で缶は直径 62.4 mm の真円柱、指の開きは
    最大 80 mm、順方向デモが実際に挟んだ幅は 63 mm。**横倒しでも真上から軸に
    直交して挟めば、挟む幅も触る面（円筒側面）も順方向と同一**なので、滑る
    理由がない。違うのは軸のどちら向きから寄るかだけである。

    `axis_local` は `standing_axis` が返す対称軸の番号、`back_offset` は
    `grasp_backoff` が返す手を引く距離 [m]。`ref_quat` に今の手首を渡すと、
    手首を回す向きが近いほうの指の向きを採る（180 度回しても同じ握りなので）。
    """
    from robosuite.utils import transform_utils as T

    r_obj = T.quat2mat(np.asarray(obj_quat, dtype=np.float64))
    axis = r_obj[:, int(axis_local)]
    z_g = np.array([0.0, 0.0, -1.0])               # 真上から降りる
    y_g = np.cross(axis, z_g)                      # 軸に直交して挟む
    if np.linalg.norm(y_g) < 1e-6:
        # 物が立っている＝軸が鉛直。挟む方位は対称性で自由なので手首に合わせる。
        y_g = (np.array([0.0, 1.0, 0.0]) if ref_quat is None else
               T.quat2mat(np.asarray(ref_quat, dtype=np.float64))[:, 1])
        y_g = y_g - z_g * float(y_g @ z_g)
    y_g = y_g / np.linalg.norm(y_g)
    if ref_quat is not None:
        y_ref = T.quat2mat(np.asarray(ref_quat, dtype=np.float64))[:, 1]
        if y_g @ y_ref < 0:                        # 指の向きは 180 度で等価
            y_g = -y_g
    r_g = np.column_stack([np.cross(y_g, z_g), y_g, z_g])
    pos = np.asarray(obj_pos, dtype=np.float64) - z_g * back_offset
    return np.concatenate([pos, np.asarray(T.quat2axisangle(T.mat2quat(r_g)))])


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
                     gain=1.0):
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
    obs = reset_to(env, init_state)
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


def eef_axisangle(obs):
    """obs の eef クォータニオンを軸角にする（`observation.state[3:6]` と同じ形）。"""
    from robosuite.utils import transform_utils as T

    return np.asarray(T.quat2axisangle(
        np.asarray(obs["robot0_eef_quat"], dtype=np.float64)))


def rollout_phased(env, init_state, ref_states, gripper_cmd, video_path=None,
                   gain=1.0, tol=0.005, max_approach=60, grip_hold=10,
                   grasp_pose=None, standoff=0.0, capture=False,
                   lift=0.0, release_i=None, settle=0, obj=None,
                   place_axis=None, level=False, cycles=None, max_tail=-1,
                   level_tol=1.5):
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
      L 上昇 — `lift` を渡すと、掴んだ直後に**真上へだけ**上がる（後述）
      C 復路 — 掴んだ後は**記録どおりの時刻逆向き**で基準経路を追う
      D 設置 — `release_i` を渡すと、そこで復路を止めて**触れるまで下ろしてから**
               放す（後述）。放したあとは残りの復路で手を引く

    A と B で入るコマは**デモに無い余分なコマ**だが、学習データを作るのが目的なので
    問題にならない。指令も観測も動画も同じ列に入る。

    **`grasp_pose` には「自分の順再生が実際に本を放した姿勢」を渡すこと。**
    基準経路（デモの state 列）の掴む姿勢をそのまま使うと、**本がそこに無い**
    ことがある——順再生はデモと同じ場所へ本を置くとは限らず、実測で 25〜35 mm
    ずれた。腕は基準の姿勢へ 3〜4 mm で着くのに空を掴む、という失敗になる。
    順再生の eef 列から取れば、本が今どこにあっても必ず整合する。
    省略時は `ref_states[close_i]`（＝デモの姿勢）に落ちるが、それは上記の
    失敗そのものなので、切り分け以外では必ず渡すこと。

    **`standoff` を渡すと、A 段の前に「掴む姿勢の手前」へ寄る段を足す。**
    手前の向きは**グリッパ自身の進入軸**（eef ローカル z の逆向き）に取る。
    真上ではない——寝た物の最短辺は鉛直を向くので、真上から降ろすと天面に当たる。
    進入軸に沿って下がれば、最後の一手が「まっすぐ差し込む」動きになり、
    物を横から押さずに指の間へ入れられる。

    戻り値の `n_extra` が A+B で足したコマ数。

    **`lift` を渡すと、掴んだ直後に真上へ上がってから復路へ合流する。**
    復路の頭は「順方向が物を置いた高さ」＝配布 T3 が物を置いた場所のすぐ上だが、
    逆向きの掴む場所は init の抽選で**横にずれている**。そのまま復路を追うと、
    掴んだ物を下ろしたまま横へ引きずり、**下の皿やトレイを動かす**。真上へ逃げて
    から、復路が同じ高さまで上がってくる点で合流すれば擦らない。飛ばした頭の
    コマ数は `n_skipped` に返る。

    **`release_i` を渡すと、そこで復路を止めて「触れたら放す」に切り替える。**
    渡す値は**順方向が実際に挟み終えたコマ**を反転した位置
    （`len(ref_states) - 1 - clamp_frame(...)`）である。記録どおりの開き指令の
    時刻で放すと、その時点で手は既に戻り始めていて**物を空中で落とす**。
    `settle` コマまで 5 mm ずつ下ろし、`obj` に触れる物が出たらそこで止めて開く。

    **`cycles` を渡すと「掴んで置く」を何回でも回す。**渡さなければ引数から
    1 サイクルを組み立てるので、単体タスクの呼び方は変わらない。1 サイクルは
    `{"grasp_pose", "start_i", "release_i", "obj", "place_axis", "level"}` で、
    **逆向きに辿る順**（＝順方向の最後のサイクルが先頭）に並べて渡す。
    two mugs のように 1 本のデモが 2 回掴んで放す場合、**サイクルごとに
    「その物の今の姿勢へ接近して掴む → その区間の復路を辿る → 触れたら置く」**を
    回す。物がどこに置き直されていても、掴む姿勢はサイクルごとに引き直される。

    **`max_tail` は「放したあとに手を引く区間」の上限コマ数である**（`-1` で
    全部辿る・`0` で落とす）。この区間は**腕が初期姿勢へ戻るだけ**で、最後に
    放した時点で goal は既に立っているので採点に効かず、学習データにも要らない。
    **本番は 300 コマで打ち切られる**ので、長い回はここを落とすのがいちばん安い。

    **`capture=True` で学習データの中身も残す。**`actions` (T,7) と
    `states` (T,8) が返り、画も要るなら env を `DATASET_CAMERAS` の 2 台で
    作っておくこと（`images` が付く）。**画は観測に乗っている台ぶんしか取れない。**
    """
    obs = reset_to(env, init_state)
    inner = env.env
    frames, eef, acts, states8 = [], [], [], []
    cams = {k: [] for k in DATASET_CAMERAS} if capture else {}
    grip = np.asarray(gripper_cmd)
    ref = np.asarray(ref_states, dtype=np.float64)

    # **掴んで置く単位（サイクル）に揃える。**単体タスクは 1 サイクルで、
    # そのときは引数から組み立てる（呼ぶ側は今までどおりで動く）。
    if cycles is None:
        closes = np.where(grip > 0)[0]
        i0 = int(closes[0]) if len(closes) else 0
        cycles = [{"grasp_pose": ref[i0] if grasp_pose is None else grasp_pose,
                   "start_i": i0, "release_i": release_i, "obj": obj,
                   "place_axis": place_axis, "level": level}]
    cycles = [dict(c) for c in cycles]
    for c in cycles:
        c["grasp_pose"] = np.asarray(c["grasp_pose"], dtype=np.float64)
        c.setdefault("obj", None)
        c.setdefault("level", False)
        c.setdefault("release_i", None)
        if c.get("place_axis") is None and c["obj"] is not None:
            c["place_axis"] = standing_axis(obs[f"{c['obj']}_quat"])

    # 追いかける物。**サイクルごとに違う**ので、名前ごとに列を持つ。
    axis_of = {c["obj"]: int(c["place_axis"]) for c in cycles if c["obj"]}
    trace = {n: {"pos": [], "z": [], "tilt": [], "touch": []} for n in axis_of}

    if video_path is not None:
        frames.append(obs[f"{CAMERA}_image"][::-1])
    eef.append(np.asarray(obs["robot0_eef_pos"]).copy())

    def step(a):
        nonlocal obs
        # **撃つ前の観測を記録する。** 学習データは `state[t]` と `action[t]` が
        # 同じ時刻（撃つ前）で対になる規約なので、撃った後を記録すると 1 コマずれる。
        acts.append(np.asarray(a, dtype=np.float32).copy())
        if capture:
            states8.append(obs_state8(obs))
            for key, cam in DATASET_CAMERAS.items():
                cams[key].append(dataset_image(obs, cam))
        obs, _, _, _ = inner.step(a)
        if video_path is not None:
            frames.append(obs[f"{CAMERA}_image"][::-1])
        eef.append(np.asarray(obs["robot0_eef_pos"]).copy())
        for name, t in trace.items():
            t["pos"].append(np.asarray(obs[f"{name}_pos"]).copy())
            t["z"].append(float(obs[f"{name}_pos"][2]))
            t["tilt"].append(tilt_deg(obs[f"{name}_quat"], axis_of[name]))
            t["touch"].append(sorted(contacting_bodies(env, name)))

    def move_to(target, reach, gripcmd=-1.0):
        """`target` へ `reach` 以内に入るまで寄る（既定は開いたまま）。コマ数を返す。"""
        n = 0
        for _ in range(max_approach):
            if float(np.linalg.norm(
                    np.asarray(obs["robot0_eef_pos"]) - target[:3])) < reach:
                break
            step(osc_command(obs, target, gripcmd, gain))   # -1 = 開く / +1 = 閉じる
            n += 1
        return n

    n = {"standoff": 0, "reach": 0, "grip": 0, "lift": 0, "floor": 0,
         "settle": 0, "path": 0, "tail": 0}
    approach_err, opened_at, placed, last_stop = [], [], [], None
    per_cycle = []

    for cyc in cycles:
        gp = cyc["grasp_pose"]

        # A0: 掴む姿勢の手前（進入軸に沿って standoff だけ下がった点）へ先に寄る。
        #     直進で掴む姿勢へ向かうと、途中で物を横から押してしまう。
        if standoff > 0:
            from robosuite.utils import transform_utils as T

            r_g = T.quat2mat(T.axisangle2quat(gp[3:6]))
            n["standoff"] += move_to(
                np.concatenate([gp[:3] - r_g[:, 2] * standoff, gp[3:6]]), tol * 3)

        # A: 掴む姿勢へ、開いたまま、着くまで
        n["reach"] += move_to(gp, tol)
        approach_err.append(round(float(np.linalg.norm(
            np.asarray(obs["robot0_eef_pos"]) - gp[:3])) * 1000, 2))

        # B: その場で閉じる（指が閉じ切るまで姿勢は保持）
        for _ in range(grip_hold):
            step(osc_command(obs, gp, 1.0, gain))            # +1 = 閉じる
        n["grip"] += grip_hold

        # L: 掴んだら**まず真上へ**。横へ動かすと、下の皿やトレイを擦って動かす。
        z_safe = None
        if lift > 0:
            up = np.concatenate([np.asarray(obs["robot0_eef_pos"]) + [0, 0, lift],
                                 gp[3:6]])
            n["lift"] += move_to(up, tol * 2, 1.0)
            z_safe = float(up[2])

        # C: このサイクルの復路。ただし頭は**上げた高さより下へ潜らせない。**
        #    **コマを飛ばさずに高さだけ床を張る。**飛ばすと姿勢の変化まで飛んで、
        #    OSC の回転が追いつかずに荷物を傾けたまま運ぶ（本タスクで実測）。
        i0 = int(cyc["start_i"])
        rel = cyc["release_i"]
        stop_i = (len(ref) - 1) if rel is None else max(int(rel), i0)
        i_join = i0
        if z_safe is not None and stop_i > i0:
            z_path = ref[i0:stop_i, 2]
            above = np.where(z_path >= z_safe)[0]
            # 復路が一度も上がってこないなら、いちばん高い所までを床の対象にする
            i_join = i0 + int(above[0] if len(above) else np.argmax(z_path))
        for i in range(i0, stop_i):
            tgt = ref[i + 1].copy()
            if i < i_join and tgt[2] < z_safe:
                tgt[2] = z_safe
                n["floor"] += 1
            step(osc_command(obs, tgt, grip[i], gain))
        n["path"] += stop_i - i0
        per_cycle.append({"obj": cyc["obj"], "n_path": stop_i - i0})
        last_stop = stop_i
        if rel is None:
            continue

        # D: **触れた所で放す。**記録どおりの時刻で開くと、手はもう戻り始めていて
        #    物を落とす。触れる物が出るまで 5 mm ずつ下ろしてから開く。
        hold = np.concatenate([np.asarray(obs["robot0_eef_pos"]),
                               eef_axisangle(obs)])
        if cyc["level"] and cyc["obj"] is not None:
            # **下ろす前に水平へ直す。**少し浮かせてから回さないと机に擦る。
            # **水平になったら止める。**回す量は 10° 前後で数コマしか要らないのに
            # 固定で待つと、本番の打ち切り 300 コマをそのぶん食う。
            hold = level_pose(obs, cyc["obj"], cyc["place_axis"], clearance=0.02)
            for _ in range(grip_hold):
                step(osc_command(obs, hold, 1.0, gain))
                n["settle"] += 1
                if tilt_deg(obs[f"{cyc['obj']}_quat"], cyc["place_axis"]) < level_tol:
                    break
        for _ in range(settle):
            if cyc["obj"] is not None and contacting_bodies(env, cyc["obj"]):
                break
            hold[2] -= 0.005
            step(osc_command(obs, hold, 1.0, gain))
            n["settle"] += 1
        hold[:3] = np.asarray(obs["robot0_eef_pos"])         # 止めた所で開く
        opened_at.append(len(acts))
        placed.append(cyc["obj"])
        for _ in range(grip_hold):
            step(osc_command(obs, hold, -1.0, gain))
        n["grip"] += grip_hold

    # 放したあとの残りの復路は**腕が初期姿勢へ戻るだけ**の区間である。最後に
    # 放した時点で goal は既に立っているので採点に効かず、**学習データにも要ら
    # ない。**本番は 300 コマで打ち切られるので、two mugs ではここ（実測 44〜92
    # コマ）を落とすのがいちばん安い。`max_tail` は 0 で落とす・`-1` で全部辿る。
    if opened_at and last_stop is not None:
        end_i = (len(ref) - 1 if max_tail < 0
                 else min(len(ref) - 1, last_stop + max_tail))
        for i in range(last_stop, end_i):
            step(osc_command(obs, ref[i + 1], -1.0, gain))
            n["tail"] += 1

    if video_path is not None:
        write_video(frames, video_path)
    out = {"n_steps": len(acts), "eef_pos": np.asarray(eef),
           "n_cycles": len(cycles),
           "n_extra": (n["standoff"] + n["reach"] + n["grip"] + n["lift"]
                       + n["settle"]),
           "n_approach": n["standoff"] + n["reach"],
           # **どちらの段で足りなかったかを分けて残す。**片方が上限に張り付いた
           # ときだけ `max_approach` を上げればよく、闇雲に伸ばさずに済む。
           "n_standoff": n["standoff"], "n_reach": n["reach"],
           "n_lift": n["lift"], "n_floor": n["floor"], "n_settle": n["settle"],
           # **どの段がコマを食っているかを段ごとに残す。**打ち切り 300 に
           # 収める作業は、この内訳を見ないと当てずっぽうになる。
           "n_path": n["path"], "n_tail": n["tail"], "per_cycle": per_cycle,
           # サイクルごとの残り誤差。**代表値は最悪のもの**を出す。
           "approach_err_mm": max(approach_err) if approach_err else 0.0,
           "approach_err_mm_all": approach_err,
           "actions": np.asarray(acts, dtype=np.float32)}
    if trace:
        out["trace"] = trace
        out["release_frames"] = opened_at
        out["release_objs"] = placed
        # **放してから終わりまでに物が落ちた高さ。**空中で放していれば大きい。
        # 指令の後も指は 8 指令ぶん開き続けるので、その間に手が持ち上げた高さも
        # 落ちる高さに入る。**開き指令以降の最高点**から終わりを引く。
        drops = {}
        for k, name in enumerate(placed):
            z = trace[name]["z"]
            tail = z[max(min(opened_at[k] - 1, len(z) - 1), 0):]
            drops[name] = round((max(tail) - z[-1]) * 1000, 1)
        out["drop_mm_by"] = drops
        out["drop_mm"] = max(drops.values()) if drops else 0.0
    if capture:
        out["states"] = np.asarray(states8, dtype=np.float32)
        out["images"] = {k: np.asarray(v, dtype=np.uint8)
                         for k, v in cams.items()}
    return out

def episode_ids(npz, spec=""):
    """npz にある `actions_<ep>` から、回すエピソード番号を決める。"""
    if spec.strip():
        return [int(x) for x in spec.split(",") if x.strip()]
    return sorted(int(k.split("_")[1]) for k in npz.files
                  if k.startswith("actions_"))


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
    eps = episode_ids(z, args.episodes)
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
