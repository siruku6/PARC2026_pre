#!/usr/bin/env python3
"""016 推奨方式: 順方向タスクから「逆向きタスク」を作る。

コンテナ（parc2026-final）の中で走る。次の 3 つを書く:

  1. `<stem>_rev.bddl` — 逆向きのタスク定義
       - goal   = 対象物 A を「A の初期領域(..._init_region)」へ置く
       - 指示文 = 「take A off B and put it back」の形
       - obj_of_interest = A ＋ 据え置き(fixture)
       - `:init` は順方向のまま。**bddl の `:init` は初期配置の「定義」で、実際の
         初期 sim 状態は下の 2 が支配する**ため、書き換える必要が無い。

  2. `<stem>_rev.pruned_init` — 逆向きの初期状態（**評価のエピソード数ぶんの行**）
       順方向の初期配置から対象物 A を「順方向の着地先」へ移し、整定させた
       flattened sim state。`--episodes` 行ぶん作り、**行 k はエピソード k の
       fixture 配置の上で作る**（なぜそれが要るかは
       [_016_libero_env.py](_016_libero_env.py) の docstring）。

  3. `<stem>_rev.placement.json` — どの置き方を採ったかの記録（切り分け用）

置き方は 1 通りに決め打ちせず候補を掃き、**LIBERO 本体の predicate が真かつ静定した
最初の 1 つ**だけを採る。「A が着地先に在る」と言えない初期状態で評価すると、
逆向きが通らない原因を重みとタスク定義に分離できないため。

用法（ホストから docker run 経由で。`/app` へ probe ディレクトリを丸ごと mount する）:
    docker run --rm \
        -v <host_in>:/in -v <host_out>:/out \
        -v <repo>/solution/tools/probe:/app:ro \
        -e LIBERO_ROOT=/workspace/LIBERO-plus -e MUJOCO_GL=egl \
        parc2026-final python /app/_016_make_reverse.py --in /in --out /out --stem <タスク名>

`--in` には順方向の <stem>.bddl と <stem>.pruned_init が要る（extract が書いたもの）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from _016_libero_env import (eval_predicate, load_flat_state, make_eval_env,
                             region_site_xpos, settle_zero)

# 対象を「置く」系の predicate（逆向きの goal になる）
PLACEMENT = {"On", "In", "At", "InFrontOf", "Behind",
             "ToTheLeftOf", "ToTheRightOf", "OnTopOf"}


# ---------------------------------------------------------------- bddl パース
def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            toks.append(c); i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            toks.append(text[i:j]); i = j
    return toks


def parse_sexpr(toks: list[str]):
    pos = 0

    def rd():
        nonlocal pos
        if pos >= len(toks):
            return None
        t = toks[pos]
        if t == "(":
            pos += 1
            out = []
            while pos < len(toks) and toks[pos] != ")":
                out.append(rd())
            pos += 1
            return out
        if t == ")":
            raise ValueError("unbalanced )")
        pos += 1
        return t

    return rd()


def sections(root):
    out: dict = {}
    for child in root if isinstance(root, list) else []:
        if isinstance(child, list) and child and isinstance(child[0], str):
            key = child[0].lstrip(":")
            if key in ("regions", "objects", "init", "goal", "fixtures",
                       "obj_of_interest", "language"):
                out[key] = child[1:]
    return out


def leaf_preds(nodes):
    out = []

    def rec(n):
        if isinstance(n, list):
            if n and isinstance(n[0], str) and n[0][0:1].isupper() and len(n) >= 2:
                out.append((n[0], list(n[1:])))
            for c in n:
                rec(c)

    for n in nodes:
        rec(n)
    return out


def find_fixture_of_region(regions, region_name):
    """region の `:target`（据え置き fixture）を返す。

    region は参照時に `<target>_<name>` になる（例 `desk_caddy_1_back_contain_region`）が、
    定義名は `back_contain_region` なので、定義名が参照名の末尾に当たるものを探す。
    """
    for r in regions:
        if not (isinstance(r, list) and r and isinstance(r[0], str)):
            continue
        name = r[0]
        if name == region_name or region_name.endswith("_" + name) or region_name.endswith(name):
            for sub in r[1:]:
                if isinstance(sub, list) and sub and sub[0] == ":target":
                    return sub[1]
    return None


# ---------------------------------------------------------------- 逆向きの判定
def analyze_forward(bddl_text: str):
    """順方向 bddl から、逆向きを作るのに要る 3 要素を抽出する。

    - a_obj        : 順方向で「対象物」を掴んで運ぶ物体（goal の置く系が掛かる物体）
    - a_init_region: A が初期に置かれているテーブル上の region（= 逆向きの goal 領域）
    - a_goal_region: 順方向で A を置く region（= 逆向きの初期配置の場所）

    逆向きの形:
      goal = (On a_obj a_init_region)   … A を元のテーブル region へ戻す
      init = A が a_goal_region にある   … 順方向の着地先から始める
      obj_of_interest = [a_obj, a_goal の fixture]
    """
    root = parse_sexpr(tokenize(bddl_text))
    sec = sections(root)
    goal = leaf_preds(sec.get("goal", []))
    init = leaf_preds(sec.get("init", []))
    regions = sec.get("regions", [])

    # goal の置く系 predicate（A を置く先）。Turnon 等の非置く系は外す。
    a_obj, a_goal_region, fwd_goal_pred = None, None, None
    for pred, args in goal:
        if pred in PLACEMENT and len(args) >= 2:
            a_obj, a_goal_region, fwd_goal_pred = args[0], args[1], pred
            break
    if a_obj is None:
        raise SystemExit(f"goal に置く系 predicate が見つからない: {goal}")

    # init で A がどの region にいるか（テーブル上の ..._init_region を狙う）
    a_init_region = None
    for pred, args in init:
        if args and args[0] == a_obj and len(args) >= 2 and isinstance(args[1], str):
            a_init_region = args[1]
            break
    if a_init_region is None:
        raise SystemExit(f"init で {a_obj} の初期 region を特定できない")

    # a_goal_region の target = A を置く fixture（据え置き）
    a_goal_fixture = find_fixture_of_region(regions, a_goal_region)

    obj_of_interest = [a_obj]
    # 据え置き = 順方向の着地先の fixture（キット自带 4 件と同じ形）
    if a_goal_fixture and a_goal_fixture not in obj_of_interest:
        obj_of_interest.append(a_goal_fixture)

    return {
        "language": (sec.get("language", ["?"])[0] if sec.get("language") else "?"),
        "a_obj": a_obj,
        "a_init_region": a_init_region,
        "a_goal_region": a_goal_region,
        "a_goal_fixture": a_goal_fixture,
        "fwd_goal_pred": fwd_goal_pred,
        "obj_of_interest": obj_of_interest,
        "objects": [a[1] if isinstance(a, list) and len(a) >= 2 else a
                    for a in sec.get("objects", [])],
    }


def human_label(region_or_obj: str) -> str:
    """`black_book_1` / `desk_caddy_1` → `black book` / `desk caddy` のような呼称。"""
    s = re.sub(r"_(init_region|region)$", "", region_or_obj)
    s = re.sub(r"_(contain_region|cook_region)$", "compartment", s)
    s = re.sub(r"_\d+$", "", s)  # 個体番号 `_1` を落とす
    return s.replace("_", " ")


def rewrite_language(info) -> str:
    """逆向きの指示文。「take A out of/off <fixture> and put it back on the table」。

    fixture がテーブル上の region（containing 系）か据え置きかによって
    「out of」か「off」を使い分ける。
    """
    a_word = human_label(info["a_obj"])
    fx = info["a_goal_fixture"] or ""
    fx_word = human_label(fx) if fx else "its original spot"
    verb = "out of" if "contain" in (info["a_goal_region"] or "") else "off"
    return (f"take the {a_word} {verb} the {fx_word} "
            f"and put it back on the table")


# ---------------------------------------------------------------- 逆向き bddl 生成
def build_reverse_bddl(src_text: str, info) -> str:
    """源 bddl の :goal を「A を A_init_region へ」、:language を逆向きに書き換える。

    行ベースで置換する（構造パースからの再発生成は、未知の predicate や
    fixture の書き方を壊しやすい）。置換対象は :goal / :language /
    :obj_of_interest の 3 節だけ。
    """
    lines = src_text.splitlines()
    out = []
    state = None  # 'goal' | 'obj' | None
    goal_done = False
    obj_done = False
    lang_done = False
    # 逆向き goal = A を「A の初期テーブル region」へ置く（On）
    new_goal_inner = f"    (And (On {info['a_obj']} {info['a_init_region']}))"
    new_lang = rewrite_language(info)

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if s.startswith("(:goal"):
            out.append("  (:goal")
            out.append(new_goal_inner)
            out.append("  )")
            # 元の :goal 節を丸ごと読み飛ばす。`(:goal` の 1 個ぶんを既に
            # 消費しているので depth は 1 から始める（0 から始めると本体 1 行で
            # 閉じたと誤判定し、元の閉じ ")" が残って括弧が 1 個余る）。
            i += 1
            depth = 1
            while i < len(lines):
                depth += lines[i].count("(") - lines[i].count(")")
                i += 1
                if depth <= 0:
                    break
            goal_done = True
            continue

        if s.startswith("(:obj_of_interest"):
            out.append("  (:obj_of_interest")
            for o in info["obj_of_interest"]:
                out.append(f"    {o}")
            out.append("  )")
            i += 1
            # 元節をスキップ（閉じ ) まで）
            depth = 1
            while i < len(lines):
                depth += lines[i].count("(") - lines[i].count(")")
                i += 1
                if depth <= 0:
                    break
            obj_done = True
            continue

        if s.startswith("(:language"):
            out.append(f'  (:language {new_lang})')
            i += 1
            lang_done = True
            continue

        out.append(line)
        i += 1

    text = "\n".join(out) + "\n"
    if not (goal_done and obj_done and lang_done):
        raise SystemExit(f"置換漏れ: goal={goal_done} obj={obj_done} lang={lang_done}")
    return text


# ---------------------------------------------------------------- 逆向き初期状態
def _find_free_joint(sim, a_obj):
    """物体 A の自由関節（free joint）名と qpos/dof の先頭アドレスを返す。

    LIBERO/robosuite の物体は **mocap ではなく free joint** で置かれている
    （`black_book_1` → 関節 `black_book_1_joint0`、`jnt_type==0`）。
    位置を変えるには mocap ではなくこの関節の qpos（xyz＋quat の 7 要素）を書く。
    """
    want = f"{a_obj}_joint0"
    fallback = None
    for j in range(sim.model.njnt):
        nm = sim.model.joint_id2name(j) or ""
        if int(sim.model.jnt_type[j]) != 0:   # 0 = free
            continue
        if nm == want:
            return nm, int(sim.model.jnt_qposadr[j]), int(sim.model.jnt_dofadr[j])
        if fallback is None and nm.startswith(a_obj):
            fallback = (nm, int(sim.model.jnt_qposadr[j]), int(sim.model.jnt_dofadr[j]))
    if fallback is not None:
        return fallback
    raise SystemExit(f"{a_obj} の free joint が見つからない（{want} も前方一致も無し）")


def _quat_mul(q1, q2):
    """mujoco 順（w, x, y, z）の四元数の積。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _axis_quat(axis, deg):
    """軸まわり回転の四元数（w, x, y, z）。"""
    h = np.deg2rad(deg) / 2.0
    v = np.zeros(3)
    v["xyz".index(axis)] = np.sin(h)
    return np.array([np.cos(h), v[0], v[1], v[2]])


def _yaw_quat(deg):
    """world z 軸まわりのヨー角だけの四元数（w, x, y, z）。"""
    return _axis_quat("z", deg)


# 逆向き初期状態の置き方候補。順に試し、**本物の predicate が真になった最初の 1 つ**を採る。
#
# `in_box`（`In` の実体）は「物体の body 中心が site 中心まわりの軸並行箱に入るか」
# なので、容器（caddy の仕切りなど）では **A の姿勢が箱の細い軸に揃っているか**で
# 入る／弾かれるが決まる。テーブル上の姿勢を相対回転させるだけでは揃わないため、
# **world 基準の絶対ヨー**（`abs`）を先に試し、駄目なら元姿勢の相対回転（`rel`）へ落ちる。
def _placement_candidates():
    cands = []
    # 容器（caddy の仕切りなど）は「縁の上に乗っただけ」でも `in_box` が真になり得る。
    # 縁より下から始めれば必ず中に納まるので、**小さい／負の余裕を先に試す**。
    for margin in (0.0, -0.01, -0.02, 0.01, 0.03, 0.06):
        for yaw in (0, 90, 180, 270):
            cands.append((margin, ("abs", yaw)))
    for margin in (0.03, 0.06, 0.10, 0.15):
        cands.append((margin, ("orig", 0)))
        for axis in ("x", "y"):
            for deg in (90, -90):
                cands.append((margin, ("rel", (axis, deg))))
    return cands


PLACEMENT_CANDIDATES = _placement_candidates()


def _candidate_quat(mode, base_quat):
    kind, arg = mode
    if kind == "orig":
        return base_quat, "orig"
    if kind == "abs":
        return _yaw_quat(arg), f"absyaw{arg}"
    axis, deg = arg
    return _quat_mul(_axis_quat(axis, deg), base_quat), f"rel{axis}{deg}"


def make_reverse_inits(bddl_path, fwd_states, a_obj, a_goal_region, fwd_pred,
                       settle=80, stable_steps=60):
    """エピソードごとに 1 行ずつ、逆向きの初期状態（flattened sim state）を作る。

    `fwd_states[k]` = 評価がエピソード k で使う順方向の初期状態。
    評価と同じ「env を 1 個作って seed し、エピソードごとに `reset()`」の並びを
    再現するので、行 k は**エピソード k の fixture 配置**の上で作られる。

    各行について `PLACEMENT_CANDIDATES` を順に試し、順方向 goal の置く系
    predicate が真かつ静定した最初の 1 つを採る。1 行でも作れなければ落とす。
    """
    # 候補の掃き出しは画を 1 枚も使わないので描画を切る（`use_camera_obs=False`）。
    # 候補を何十通りも試して数千ステップ回るので `long_horizon`。
    env = make_eval_env(bddl_path, use_camera_obs=False, long_horizon=True)
    rows, diags = [], []
    for ep, fwd_state in enumerate(fwd_states):
        env.reset()                      # 評価と同じ回数だけ reset を進める
        sim = env.env.sim
        obs = load_flat_state(env, fwd_state)

        jnt_name, qadr, dadr = _find_free_joint(sim, a_obj)
        site_xyz = region_site_xpos(env, a_goal_region)
        # sim.data.* は生きたビューなので、基準値は必ず copy して持つ
        base_flat = np.asarray(sim.get_state().flatten()).copy()
        a_xyz_fwd = np.asarray(obs[f"{a_obj}_pos"]).copy()
        base_quat = np.asarray(sim.data.qpos[qadr + 3:qadr + 7]).copy()
        print(f"   [ep{ep}] 着地先 {a_goal_region} 中心={np.round(site_xyz, 4)} "
              f"／ A の順方向初期={np.round(a_xyz_fwd, 4)}")

        chosen = None
        tried = []
        for margin, mode in PLACEMENT_CANDIDATES:
            sim.set_state_from_flattened(base_flat)
            sim.forward()
            quat, quat_label = _candidate_quat(mode, base_quat)
            sim.data.qpos[qadr:qadr + 3] = [site_xyz[0], site_xyz[1],
                                            site_xyz[2] + margin]
            sim.data.qpos[qadr + 3:qadr + 7] = quat
            sim.data.qvel[dadr:dadr + 6] = 0.0
            sim.forward()
            obs = settle_zero(env, settle)

            label = f"margin={margin} quat={quat_label}"
            if not eval_predicate(env, fwd_pred, a_obj, a_goal_region):
                tried.append((label, "predicate=False"))
                continue

            # 静定チェック: さらに回しても predicate が保たれ、A が動かないことを要る。
            # 縁に引っ掛かっただけの姿勢（評価開始直後に転げ落ちる）を弾くため。
            a_pos = np.asarray(obs[f"{a_obj}_pos"]).copy()
            obs = settle_zero(env, stable_steps)
            still = eval_predicate(env, fwd_pred, a_obj, a_goal_region)
            a_pos2 = np.asarray(obs[f"{a_obj}_pos"]).copy()
            drift = float(np.linalg.norm(a_pos2 - a_pos))
            tried.append((label, f"predicate=True still={still} drift={drift:.4f}"))
            if still and drift < 0.01:
                chosen = {
                    "flat": np.asarray(sim.get_state().flatten()).copy(),
                    "a_pos": a_pos2, "placement": label, "drift": drift,
                }
                print(f"   [ep{ep}] 採用 {label}: {fwd_pred}=True 静定 "
                      f"drift={drift:.4f} A={np.round(a_pos2, 4)}")
                break

        if chosen is None:
            raise SystemExit(
                f"[ep{ep}] {a_obj} を {a_goal_region} へ静定して置ける候補が無い"
                f"（{fwd_pred} が真かつ静定、を満たす候補ゼロ）。試した候補: {tried}")
        rows.append(chosen["flat"].astype(np.float32))
        diags.append({"episode": ep, "placement": chosen["placement"],
                      "a_pos": np.round(chosen["a_pos"], 4).tolist(),
                      "drift": chosen["drift"],
                      "site_xyz": np.round(site_xyz, 4).tolist(),
                      "region": a_goal_region, "joint": jnt_name})
    env.close()
    return np.stack(rows), diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--settle", type=int, default=80)
    ap.add_argument("--episodes", type=int, default=2,
                    help="評価のエピソード数。この数だけ逆向き初期状態の行を作る")
    args = ap.parse_args()

    in_dir = Path(args.indir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bddl = (in_dir / f"{args.stem}.bddl").read_text(encoding="utf-8")
    init_path = in_dir / f"{args.stem}.pruned_init"
    # .pruned_init は複数初期状態（N 行）が入る。評価ハーネスはエピソード k で
    # k 行目を使う（`get_perturbed_init_states` の `init_states[ep_id]`）ので、
    # 逆向きも先頭 `--episodes` 行ぶんを 1 行ずつ作り直す。
    _raw = torch.load(str(init_path), weights_only=False)

    info = analyze_forward(bddl)
    print(f"== {args.stem}")
    print(f"   対象物 A: {info['a_obj']}")
    print(f"   順方向: A を {info['a_goal_region']} に置く (fixture={info['a_goal_fixture']})")
    print(f"   逆向き goal: (On {info['a_obj']} {info['a_init_region']})")
    print(f"   逆向き初期: A が {info['a_goal_region']} にある")
    print(f"   obj_of_interest: {info['obj_of_interest']}")

    # 1. 逆向き bddl
    rev_bddl = build_reverse_bddl(bddl, info)
    rev_bddl_path = out_dir / f"{args.stem}_rev.bddl"
    rev_bddl_path.write_text(rev_bddl, encoding="utf-8")
    # 書いたものを LIBERO のパーサに通し、壊れた bddl を下流（検証・評価）へ流さない
    from libero.libero.envs import bddl_utils as BDDLUtils
    BDDLUtils.get_problem_info(str(rev_bddl_path))
    BDDLUtils.robosuite_parse_problem(str(rev_bddl_path))
    print(f"   逆向き bddl 書き出し（パース確認済み）: {rev_bddl_path}")

    # 2. 逆向き初期状態: A を順方向の着地先（info['a_goal_region']）へ置く。
    #    評価が使うぶんだけ行を作る（エピソード k の行は、エピソード k の
    #    fixture 配置の上で作られる）。
    rev_rows, diags = make_reverse_inits(
        in_dir / f"{args.stem}.bddl", _raw[:args.episodes],
        info["a_obj"], info["a_goal_region"], info["fwd_goal_pred"].lower(),
        settle=args.settle)
    rev_init_path = out_dir / f"{args.stem}_rev.pruned_init"
    torch.save(torch.from_numpy(rev_rows), str(rev_init_path))
    print(f"   逆向き初期状態書き出し: {rev_init_path}  ({rev_rows.shape[0]} 行)")
    (out_dir / f"{args.stem}_rev.placement.json").write_text(
        json.dumps(diags, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()