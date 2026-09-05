#!/usr/bin/env python3
"""016: 逆向きタスク定義の妥当性検証（GPU の方策は使わない・シムのみ）。

評価コンテナの中で走り、逆向きタスクが「正しく逆向きに設定されている」ことを
確かめる。**ここが通らなければ、評価で出る「成功」は偽物になる。**

3 つのチェック（**すべて LIBERO 本体の predicate で判定する**。距離のしきい値による
近似は使わない——採点器と同じ判定でなければ、「縁に乗っただけ」を「着地先に在る」と
通してしまう）:

  (a) 逆向き bddl が読めて scene が立つ。
  (b) 逆向き初期状態を復元した t=0 で、逆向きの goal（A が a_init_region に載る）が
      **未達**である。← 偽 Yes 防止の核心。t=0 で達していれば、腕が何もせずとも
      成功に見える。
  (c) 同じ t=0 の状態を**順方向 bddl の env** に載せると、順方向 goal の置く系
      predicate（`in` / `on`）が**真**である。＝ A が確かに「順方向の着地先」に在る。
      ここが偽なら、逆向きが通らない原因を「重み」と「初期配置の作り損ね」に
      分離できない。

**評価が実際に使う行だけ**（`--episodes` 行）を、評価と同じ `seed` → `reset()` の
並びで見る（なぜ並びが要るかは [_016_libero_env.py](_016_libero_env.py) の docstring）。

用法（`/app` へ probe ディレクトリを丸ごと mount する）:
    docker run --rm -v <in>:/in -v <out>:/out \\
        -v <repo>/solution/tools/probe:/app:ro \\
        -e LIBERO_ROOT=/workspace/LIBERO-plus \\
        parc2026-final python /app/_016_verify_reverse.py --in /in --out /out
`/in` には 1 タスクごとに <stem>.bddl / <stem>_rev.bddl / <stem>_rev.pruned_init が並ぶ。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _016_libero_env import eval_predicate, load_flat_state, make_eval_env

# 「置く」系 predicate。LIBERO のパーサが小文字化した名前で持つ。
PLACEMENT = {"on", "in", "at", "infrontof", "behind",
             "totheleftof", "totherightof", "ontopof"}


def placement_goal(bddl_path):
    """bddl の goal から「置く系」の (predicate, A, region) を返す。

    自前で s 式をパースせず **LIBERO のパーサ**（`robosuite_parse_problem`）を使う。
    返る `goal_state` は `['on', 'moka_pot_1', 'kitchen_table_moka_pot_init_region']`
    の形で、`_eval_predicate` がそのまま取れる小文字名になっている。

    goal が `(And (Turnon stove) (On pot region))` のように置く系以外を含む場合は、
    置く系の 1 つ目を返す（逆向きの向きを決めるのは物を動かす条件だけ）。
    """
    from libero.libero.envs import bddl_utils as BDDLUtils

    parsed = BDDLUtils.robosuite_parse_problem(str(bddl_path))
    for state in parsed["goal_state"]:
        if len(state) == 3 and state[0] in PLACEMENT:
            return tuple(state)
    return None


def verify_one(in_dir: Path, stem: str, episodes: int):
    """逆向き初期状態を、評価と同じ reset 並びで検証する。

    見るのは**評価が実際に使う行だけ**（先頭 `episodes` 行）。キット由来の
    初期状態は 50 行入っているが、評価は `init_states[ep_id]` で先頭から
    `--n-episodes` 行しか引かないので、残りを検証しても時間を捨てるだけになる。
    """
    rev_bddl_path = in_dir / f"{stem}_rev.bddl"
    fwd_bddl_path = in_dir / f"{stem}.bddl"
    rev_init = in_dir / f"{stem}_rev.pruned_init"
    if not (rev_bddl_path.exists() and rev_init.exists()):
        return {"stem": stem, "error": "reverse asset 欠け"}

    rev_states = np.asarray(torch.load(str(rev_init), weights_only=False),
                            dtype=np.float32)

    rev_goal = placement_goal(rev_bddl_path)
    fwd_goal = placement_goal(fwd_bddl_path)
    if rev_goal is None or fwd_goal is None:
        return {"stem": stem, "error": "goal の置く系 predicate を特定できない"}
    rev_pred, a, rev_region = rev_goal
    fwd_pred, _, fwd_region = fwd_goal

    # 逆向き env と順方向 env を 1 個ずつ作り、行ごとに **両方を同じ回数 reset** する
    # （reset の回数が fixture の配置を決めるため、片方だけ進めると揃わない）。
    rev_env, fwd_env = make_eval_env(rev_bddl_path), make_eval_env(fwd_bddl_path)
    checked = []
    for ep in range(min(episodes, rev_states.shape[0])):
        rev_env.reset()
        obs = load_flat_state(rev_env, rev_states[ep])
        rev_goal_t0 = eval_predicate(rev_env, rev_pred, a, rev_region)   # (b)

        fwd_env.reset()
        load_flat_state(fwd_env, rev_states[ep])
        a_at_fwd_goal = eval_predicate(fwd_env, fwd_pred, a, fwd_region)  # (c)

        checked.append({
            "episode": ep,
            "a_pos": [round(float(x), 3) for x in obs[f"{a}_pos"]],
            "check_b_t0_not_satisfied": not rev_goal_t0,   # True を期待
            "check_c_a_at_forward_goal": a_at_fwd_goal,    # True を期待
            "valid": (not rev_goal_t0) and a_at_fwd_goal,
        })
    rev_env.close()
    fwd_env.close()

    return {
        "stem": stem,
        "reverse_goal": f"{rev_pred} {a} -> {rev_region}",
        "forward_goal": f"{fwd_pred} {a} -> {fwd_region}",
        "check_a_loadable": True,
        "episodes": checked,
        "valid": bool(checked) and all(e["valid"] for e in checked),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--episodes", type=int, default=2,
                    help="評価のエピソード数。この数だけ先頭の行を検証する")
    args = ap.parse_args()
    in_dir = Path(args.indir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted({p.stem.replace("_rev", "") for p in in_dir.glob("*_rev.bddl")})
    report = []
    for stem in stems:
        try:
            r = verify_one(in_dir, stem, args.episodes)
        except Exception as e:  # noqa: BLE001 — 1 タスクの失敗で他を止めない
            r = {"stem": stem, "error": repr(e)}
        report.append(r)
        print(f"[{'OK ' if r.get('valid') else '!!!'}] {stem}")
        print(f"       逆向き goal: {r.get('reverse_goal')}")
        for e in r.get("episodes", []):
            print(f"       ep{e['episode']}: A={e['a_pos']}  "
                  f"(b) 逆向き goal 未達={e['check_b_t0_not_satisfied']}  "
                  f"(c) A が着地先に在る={e['check_c_a_at_forward_goal']}")
        if r.get("error"):
            print(f"       error: {r['error']}")

    (out_dir / "verify_reverse.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = [r["stem"] for r in report if not r.get("valid")]
    print(f"\n検証: {len(report) - len(bad)}/{len(report)} が妥当"
          + (f"  不合格: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
