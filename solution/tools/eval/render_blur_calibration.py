#!/usr/bin/env python3
"""採点側の劣化（128x128 で掛かる）と、学習側に入れる劣化（256x256 に掛ける）を
**同じ見え方に揃えるための倍率**を目で確かめる。

**なぜ倍率が要るか。**採点側は agentview を **128x128 で描画してから** `sigma =
severity` の gaussian blur を掛ける（`env_wrapper.py`）。学習データ
（`lerobot/libero_plus`）の画像は **256x256** である。**ぼけの強さは画素で決まるので、
同じ見え方にするには sigma を 2 倍にする必要がある。**

zoom_blur は倍率（zoom factor）で書かれているので**解像度に依らない**——
そのまま同じ強度を使えばよい。motion_blur は radius が画素なので 2 倍が要る。

出すもの: 上段が採点側の真値（128 で劣化 → 256 へ拡大）、下段が学習側の候補
（256 のまま cv2 の gaussian）。**左右で同じ列が同じ強度である。**

    IMAGE=parc2026-final bash solution/tools/regen/run_in_container.sh \
        solution/tools/eval/render_blur_calibration.py --out work/blur_ladder
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

TASK = ("task02_cream_cheese", "libero_goal", "put_the_cream_cheese_in_the_bowl")
SEVERITIES = [3, 6, 9]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("work/blur_ladder"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _render(suite: str, stem: str, camera: int, seed: int) -> np.ndarray:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), suite, f"{stem}.bddl")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera,
                             camera_widths=camera, control_freq=20)
    try:
        env.seed(seed)
        obs = env.reset()
        img = obs["agentview_image"]
        return img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import cv2
    from PIL import Image
    from libero.libero.envs import env_wrapper as ew

    label, suite, stem = TASK
    clean128 = _render(suite, stem, 128, args.seed)
    clean256 = _render(suite, stem, 256, args.seed)
    cv2.imwrite(str(args.out / f"{label}_clean256.png"), clean256[..., ::-1])

    rows: list[list[np.ndarray]] = [[], []]
    labels: list[list[str]] = [[], []]

    rows[0].append(cv2.resize(clean128, (256, 256), interpolation=cv2.INTER_NEAREST))
    labels[0].append("採点 128 clean")
    rows[1].append(clean256)
    labels[1].append("学習 256 clean")

    for sev in SEVERITIES:
        truth = np.clip(ew.gaussian_blur(Image.fromarray(clean128), severity=sev), 0, 255)
        truth = truth.astype(np.uint8)
        rows[0].append(cv2.resize(truth, (256, 256), interpolation=cv2.INTER_NEAREST))
        labels[0].append(f"noise_{10 + sev}  sigma {sev} @128")

        ours = cv2.GaussianBlur(clean256, (0, 0), sigmaX=2.0 * sev, sigmaY=2.0 * sev)
        rows[1].append(ours)
        labels[1].append(f"sigma {2 * sev} @256")

    band = 20
    cols = len(rows[0])
    sheet = np.full((2 * (256 + band), cols * 256, 3), 255, np.uint8)
    for r in range(2):
        for c in range(cols):
            y, x = r * (256 + band), c * 256
            sheet[y:y + 256, x:x + 256] = rows[r][c]
            cv2.putText(sheet, labels[r][c], (x + 4, y + 256 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    path = args.out / "CALIBRATION_gaussian_128_vs_256.png"
    cv2.imwrite(str(path), sheet[..., ::-1])
    print(f"[calibration] {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
