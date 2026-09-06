#!/usr/bin/env python3
"""学習データの npz から、中身をそのまま見るための動画を作る。

**シムを回し直さない。**書き出した npz に front / wrist の画素がそのまま入って
いるので、そこから作れば「学習データに何が入っているか」を正確に見られる。
シムを回し直すと乱数や fixture の引き直しでずれる余地が残る。

npz の画像は学習データの規約どおり 180 度回っている（`dataset_image`）ので、
**見るときは戻す**。左が front、右が wrist。

    python3 make_share_videos.py --npz-dir work/probe_017/reverse_npz \\
        --out work/shares --tag book --episodes 4061,4062,4063 --inits 0,25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

FPS = 20


def label(img, text, sub=""):
    """画の上に帯を敷いて字を置く。"""
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34 if sub else 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (170, 220, 255), 1, cv2.LINE_AA)
    return img


def write(path: Path, npz_path: Path, tag: str) -> str:
    z = np.load(npz_path, allow_pickle=True)
    front, wrist = z["front"], z["wrist"]
    ok = bool(z["success"]) if "success" in z else True
    instr = str(z["instruction"]) if "instruction" in z else ""
    n, h, w = front.shape[:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                         (w * 2 + 8, h))
    head = f"{tag}  {n} frames  {'success' if ok else 'FAILED'}"
    for t in range(n):
        # 学習データは 180 度回して入っているので、見るときは戻す
        f = cv2.cvtColor(front[t][::-1, ::-1], cv2.COLOR_RGB2BGR)
        r = cv2.cvtColor(wrist[t][::-1, ::-1], cv2.COLOR_RGB2BGR)
        pane = np.hstack([label(f, f"front  {head}", instr[:58]),
                          np.full((h, 8, 3), 255, np.uint8),
                          label(r, f"wrist  t={t}")])
        vw.write(pane)
    vw.release()
    return f"{path.name}  {n} コマ  {'成功' if ok else '失敗'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="出力名の頭に付ける（book など）")
    ap.add_argument("--episodes", required=True, help="順方向デモの番号をカンマ区切り")
    ap.add_argument("--inits", default="0,25", help="各デモから取る init の行")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes.split(","):
        for i in args.inits.split(","):
            src = args.npz_dir / f"rev_ep{ep}_i{i}_k+00.npz"
            if not src.exists():
                print(f"  無い: {src.name}")
                continue
            dst = args.out / f"{args.tag}_ep{ep}_init{int(i):02d}.mp4"
            print("  " + write(dst, src, f"{args.tag} ep{ep} init{i}"))


if __name__ == "__main__":
    main()
