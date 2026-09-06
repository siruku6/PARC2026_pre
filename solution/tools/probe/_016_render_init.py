#!/usr/bin/env python3
"""016: 順方向／逆向きの初期状態を 1 枚ずつ描画し、目で確かめられるようにする。

逆向きの初期状態は「A を順方向の着地先へ置き直した」合成状態なので、
predicate が真でも**見た目が破綻していない**ことは別に確かめたい
（縁に引っ掛かる・めり込む・宙に浮く）。評価に GPU を使う前の関門。

用法（コンテナ内）:
    python /app/_016_render_init.py --in /in --out /out --stem <タスク名>
`/in` に <stem>.bddl / <stem>.pruned_init / <stem>_rev.bddl / <stem>_rev.pruned_init。
`/out` に <stem>_fwd_init.png / <stem>_rev_init.png を書く。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch

from _016_libero_env import load_flat_state, make_eval_env

CAMERAS = ("agentview", "frontview")


def render_one(bddl_path: Path, init_path: Path, out_stem: Path, camera: str,
               episodes: int):
    """エピソードごとの初期状態を、評価と同じ seed・reset 並びで 1 枚ずつ書く。

    なぜ seed と reset の回数を揃える必要があるかは
    [_016_libero_env.py](_016_libero_env.py) の docstring。
    """
    # 目視用なので評価より大きく撮る（カメラの設定は配置の乱数に触れない）
    env = make_eval_env(bddl_path, camera_size=256, camera_names=[camera])
    states = np.asarray(torch.load(str(init_path), weights_only=False),
                        dtype=np.float32)
    for ep in range(min(episodes, states.shape[0])):
        env.reset()
        obs = load_flat_state(env, states[ep])
        png = out_stem.with_name(f"{out_stem.name}_ep{ep}.png")
        # robosuite の画は上下反転で返る
        imageio.imwrite(str(png), obs[f"{camera}_image"][::-1])
        print(f"   描画: {png}")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--camera", default="agentview", choices=CAMERAS)
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.indir), Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, bddl, init in (
        ("fwd", f"{args.stem}.bddl", f"{args.stem}.pruned_init"),
        ("rev", f"{args.stem}_rev.bddl", f"{args.stem}_rev.pruned_init"),
    ):
        render_one(in_dir / bddl, in_dir / init,
                   out_dir / f"{args.stem}_{tag}_init", args.camera,
                   args.episodes)


if __name__ == "__main__":
    main()
