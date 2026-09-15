#!/usr/bin/env python3
"""シムが書いた mp4 に、その回の結果を焼き込んで共有用に並べ直す。

**動画だけ渡されても、成功したのか失敗したのかが分からない。**掃引の結果
（`gripper_shift_sweep.json`）と突き合わせて、1 本ごとに「何が起きたか」を
帯にして載せる。ファイル名にも結果を入れる。

    python3 label_share_videos.py --json work/probe_019/gripper_shift_sweep.json \\
        --videos work/probe_019/videos --out work/shares --tag twomugs
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np

FPS = 20


def label(img, lines, color=(255, 255, 255)):
    """画の上に帯を敷いて字を置く。"""
    img = np.ascontiguousarray(img)
    h = 13 * len(lines) + 6
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (5, 13 * i + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    color if i == 0 else (170, 220, 255), 1, cv2.LINE_AA)
    return img


def write(src: Path, dst: Path, lines, color) -> str:
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return f"  読めない: {src.name}"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for t, f in enumerate(frames):
        vw.write(label(f.copy(), lines + [f"t={t}/{len(frames)}"], color))
    vw.release()
    return f"  {dst.name}  {len(frames)} コマ"


def _objs_in(text):
    """`skipped` の文字列から物の名前だけ取り出す（古い json 向けの保険）。"""
    _, _, tail = text.partition(": ")
    try:
        return list(ast.literal_eval(tail))
    except (ValueError, SyntaxError):
        return [tail]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for rec in json.loads(args.json.read_text(encoding="utf-8")):
        ep = rec["episode"]
        if rec.get("skipped"):
            # 逆向きを回していないので、**順再生のほうを出す**（理由が映る）
            src = args.videos / f"phased_ep{ep}_forward.mp4"
            # **帯は英字だけにする。**OpenCV の Hershey フォントに日本語は
            # 無く、混ぜると全部 "?" になる。
            objs = rec.get("skipped_objs") or _objs_in(rec["skipped"])
            print(write(src, args.out / f"{args.tag}_SKIP_ep{ep}_forward.mp4",
                        [f"{args.tag} ep{ep}  SKIPPED (forward replay shown)",
                         "did not move: " + ", ".join(
                             o.replace("_mug_1", "").replace("_1", "")
                             for o in objs)], (120, 170, 255)))
            continue
        row = rec["shifts"][0]
        ok = row["reverse_goal"]
        src = args.videos / f"phased_ep{ep}_i{row['t3_init']}_rev_k+00.mp4"
        sub = (f"{row['n_steps']} steps  drop {row.get('drop_mm', 0):.1f}mm  "
               f"others {row.get('disturb_max_mm', 0):.1f}mm")
        print(write(src,
                    args.out / f"{args.tag}_{'OK' if ok else 'NG'}_ep{ep}"
                               f"_init{row['t3_init']:02d}.mp4",
                    [f"{args.tag} ep{ep} init{row['t3_init']}  "
                     f"{'GOAL' if ok else 'FAILED'}", sub],
                    (140, 255, 140) if ok else (140, 140, 255)))


if __name__ == "__main__":
    main()
