#!/usr/bin/env python3
"""配布 T3 の指示文が、**init 50 行すべてで成り立つか**を幾何で確かめる。

**指示文は bddl に 1 つしか無い**（`env.language_instruction`）。ところが
「左の」「後ろに」のような**相対表現は、基準になる物の位置が init ごとに変わる**。
目標領域は bddl 固定なので、**領域の中に置いたときに相対表現が成り立たない init
が混ざり得る。**それを数えるのがこの道具である。

**LIBERO に相対位置の述語は無い**（`VALIDATE_PREDICATE_FN_DICT` は
`on` / `in` / `up` / `open` / `close` / `turnon` / `turnoff` だけ）。だから自分で
測るしかない。**向きの規約は推測せず、名前の付いた領域で固定する:**

  - `plate_left_region` は y = -0.30、`plate_right_region` は y = +0.30
  - `study_table_yellow_book_left_init_region` も y = -0.25

  → **左 = y が小さい / 右 = y が大きい。**

前後（`behind` / `in front of`）の軸は x だが、**向きは指示文そのものから
逆算する**しかない。両方の符号で数えて出すので、どちらが整合するかが分かる。

    python /app/check_instruction_coverage.py --t3-bddl ... --t3-inits ...
"""
from __future__ import annotations

import argparse

import numpy as np

from replay_video_probe import pin_fixtures, reset_to
from reversal_probe import load_pruned_init, make_env

PLACE = {"on", "in", "at", "infrontof", "behind", "totheleftof",
         "totherightof", "ontopof"}


def goals_of(bddl_path):
    from libero.libero.envs import bddl_utils as BDDLUtils

    return [tuple(st) for st
            in BDDLUtils.robosuite_parse_problem(str(bddl_path))["goal_state"]
            if len(st) == 3 and st[0] in PLACE]


def region_box(env, region):
    """領域の中心と半サイズ。**置ける範囲の端でも成り立つか**を見るために使う。"""
    return (np.asarray(env.env.sim.data.get_site_xpos(region)).copy(),
            np.asarray(env.env.object_sites_dict[region].size).copy())


def obj_xy(env, name):
    data, bid = env.env.sim.data, env.env.obj_body_id[name]
    return np.asarray(data.body_xpos[bid][:2]).copy()


# 指示文に出てくる相対表現 → (見る軸, 目標が基準より小さい側か)
# 左右は名前付き領域で固定できる。前後は符号を両方出す。
PHRASES = {
    "to the left of": (1, True, "左"),
    "to the right of": (1, False, "右"),
    "behind": (0, None, "後ろ"),
    "in front of": (0, None, "手前"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3-bddl", required=True)
    ap.add_argument("--t3-inits", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    env = pin_fixtures(make_env(args.t3_bddl), args.seed)
    rows = load_pruned_init(args.t3_inits)
    goals = goals_of(args.t3_bddl)
    env.reset()
    instr = env.language_instruction
    print(f"\n=== {instr}")
    hits = [(p, *PHRASES[p]) for p in PHRASES if p in instr.lower()]
    if not hits:
        print(f"  相対表現なし → **指示文は init に依らない**（goal: {goals}）")
        return
    others = [n for n in env.env.obj_body_id if n not in {g[1] for g in goals}]
    print(f"  相対表現: {[h[0] for h in hits]}  基準になり得る物: {others}")

    # **余裕[mm]で出す。**「成り立つ／崩れる」の 2 値だと、置き場所が少し
    # ずれただけで嘘になるのか、大きく余っているのかが分からない。
    # 余裕 = 目標領域の中心と基準の物の差（指示文が成り立つ向きを正とする）。
    marg = {}
    half = {}
    for row in rows:
        reset_to(env, row)
        for _, obj, region in goals:
            c, h = region_box(env, region)
            for phrase, axis, lower, _ in hits:
                half[(obj, axis)] = float(h[axis])
                for base in others:
                    b = obj_xy(env, base)[axis]
                    d = float(b - c[axis])          # 正 = 目標が基準より小さい側
                    marg.setdefault((phrase, obj, base, axis), []).append(d)
    print(f"  init {len(rows)} 行の余裕[mm]（正 = 指示文が成り立つ向き。"
          f"**目標領域の半サイズと比べること**）:")
    for key in sorted(marg):
        phrase, obj, base, axis = key
        v = np.asarray(marg[key]) * 1000
        # どちらの符号で成り立つかは、中心での多数決で決める
        sign = 1.0 if (v > 0).mean() >= 0.5 else -1.0
        v = v * sign
        h = half[(obj, axis)] * 1000
        n_bad = int((v <= 0).sum())
        mark = ("○" if n_bad == 0 else f"!! {n_bad} 行で符号が逆")
        room = "領域の端でも安全" if v.min() > h else f"領域の半サイズ {h:.0f} mm より狭い"
        print(f"    「{phrase} {base}」 {obj}: 余裕 最小 {v.min():6.1f} / "
              f"中央 {np.median(v):6.1f} / 最大 {v.max():6.1f} mm  {mark}  {room}")


if __name__ == "__main__":
    main()
