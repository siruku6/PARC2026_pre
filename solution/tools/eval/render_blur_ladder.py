#!/usr/bin/env python3
"""採点側と同じ 128x128 の agentview に、`env_wrapper.py` の劣化関数をそのまま当てて
強度 1〜10 の梯子を描く。

**なぜ要るか。**運営が公開した Track2 の摂動図（`docs/materials/notifications/`）は
`noise_26`（zoom blur）と `noise_19`（gaussian blur）の 2 枚しか見せない。
**学習時の blur をどの強さに合わせるかを決めるには、強度と見え方の対応が要る。**

**劣化は agentview だけ・毎コマ・128x128 の描画に対して掛かる**
（`env_wrapper.py` の `step`/`reset`。手首カメラは無傷）。
`noise_N` と強度の対応も同じ所から取っている。

    1〜10  motion_blur   severity = N
    11〜20 gaussian_blur severity = N-10   （sigma = severity）
    21〜30 zoom_blur     severity = N-20
    31〜40 fog           severity = N-30
    41〜50 glass_blur    severity = N-40

**使い方**（評価イメージの中で走らせる。skimage・scipy・wand が要る）:

    IMAGE=parc2026-final bash solution/tools/regen/run_in_container.sh \
        solution/tools/eval/render_blur_ladder.py --out work/blur_ladder
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 採点側の既定。`b_PARC2026_final/pipeline/config.py` の `LIBERO_EVAL_CAMERA`
CAMERA = int(os.environ.get("LIBERO_EVAL_CAMERA", "128"))

# 運営の公開図に出ている 2 課題（`compe/t2/T2_TASKS.csv` の 1 行目・2 行目）。
# **摂動の付いていない素の bddl を描く。**劣化はあとから当てる
TASKS = [
    ("task01_orange_juice", "libero_object",
     "pick_up_the_orange_juice_and_place_it_in_the_basket"),
    ("task02_cream_cheese", "libero_goal",
     "put_the_cream_cheese_in_the_bowl"),
]

FAMILIES = ["motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("work/blur_ladder"))
    p.add_argument("--families", nargs="*", default=["gaussian_blur", "zoom_blur", "motion_blur"],
                   choices=FAMILIES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--upscale", type=int, default=2,
                   help="貼り合わせるときの拡大率。128 は小さすぎて目で見にくい")
    return p.parse_args()


def _load_corruptions():
    """`env_wrapper.py` の劣化関数を、書き写さずにそのまま使う。"""
    from libero.libero.envs import env_wrapper as ew
    return {name: getattr(ew, name) for name in FAMILIES}


def _render_clean(suite: str, task_stem: str, seed: int) -> np.ndarray:
    """採点と同じ引数で環境を作り、reset 直後の agentview を返す。"""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), suite, f"{task_stem}.bddl")
    if not os.path.exists(bddl):
        raise FileNotFoundError(bddl)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=CAMERA,
                             camera_widths=CAMERA,
                             control_freq=int(os.environ.get("LIBERO_CONTROL_FREQ", "20")))
    try:
        env.seed(seed)
        obs = env.reset()
        img = obs["agentview_image"]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # LIBERO の描画は上下反転で返る（採点側も同じものを受ける）
        return np.ascontiguousarray(img)
    finally:
        env.close()


def _to_uint8(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _grid(tiles: list[np.ndarray], labels: list[str], cols: int, scale: int) -> np.ndarray:
    """横 cols 枚で並べ、下に強度の帯を付ける。"""
    import cv2

    h, w = tiles[0].shape[:2]
    h, w = h * scale, w * scale
    band = 18
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (h + band), cols * w, 3), 255, np.uint8)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, cols)
        big = cv2.resize(_to_uint8(t), (w, h), interpolation=cv2.INTER_NEAREST)
        y, x = r * (h + band), c * w
        sheet[y:y + h, x:x + w] = big
        cv2.putText(sheet, lab, (x + 3, y + h + 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import cv2

    corrupt = _load_corruptions()
    from PIL import Image

    # `noise_N` の対応。強度は 1 つの族で 1〜10
    offset = {f: i * 10 for i, f in enumerate(FAMILIES)}

    for label, suite, stem in TASKS:
        clean = _render_clean(suite, stem, args.seed)
        cv2.imwrite(str(args.out / f"{label}_clean.png"), clean[..., ::-1])
        print(f"[clean] {label} {clean.shape}", flush=True)

        for fam in args.families:
            fn = corrupt[fam]
            tiles = [clean]
            labels = [f"clean (noise_0)"]
            for sev in range(1, 11):
                np.random.seed(args.seed)   # motion_blur は角度に乱数を使う
                out = _to_uint8(fn(Image.fromarray(clean), severity=sev))
                tiles.append(out)
                labels.append(f"sev {sev} = noise_{offset[fam] + sev}")
                cv2.imwrite(str(args.out / f"{label}_{fam}_sev{sev:02d}.png"), out[..., ::-1])
            sheet = _grid(tiles, labels, cols=6, scale=args.upscale)
            path = args.out / f"LADDER_{label}_{fam}.png"
            cv2.imwrite(str(path), sheet[..., ::-1])
            print(f"[ladder] {path}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
