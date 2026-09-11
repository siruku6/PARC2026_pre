#!/usr/bin/env python3
"""学習データの npz から、中身をそのまま見るための動画を作る。

**シムを回し直さない。**書き出した npz に front / wrist の画素がそのまま入って
いるので、そこから作れば「学習データに何が入っているか」を正確に見られる。
シムを回し直すと乱数や fixture の引き直しでずれる余地が残る。

npz の画像は学習データの規約どおり 180 度回っている（`dataset_image`）ので、
**見るときは戻す**。左が front、右が wrist。

    python3 make_share_videos.py --npz-dir work/probe_017/reverse_npz \\
        --out work/shares --tag book --episodes 4061,4062,4063 --inits 0,25

**`--json` に掃引の結果を渡すと、いま流れている区間の名前を焼き込む。**
逆向きエピソードは「接近・握り・上昇・復路・設置・開き・後退・tail」の 8 区間の
つなぎ合わせで、**無駄はいつも区間の継ぎ目に出る**
（[reference/31 の 1 節](../../../docs/reference/31_reverse_place_without_contact.md)）。
どの区間を見ているか分からないまま眺めても、「上がったあと止まっている」のが
上昇の終わりなのか復路の頭なのかが切り分けられない。

    python3 make_share_videos.py --npz-dir work/probe_028/L/npz \\
        --json work/probe_028/L/gripper_shift_sweep.json \\
        --out work/videos --tag final --episodes 4061,4062,4063 --inits 0
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


# 区間の名前と、掃引の結果でその長さが入っている鍵。**並びは `rollout_phased` の
# 段の順そのもの**（`grip_hold` の 10 コマは掃引が返さないので定数で持つ）。
PHASES = [("approach", "n_approach"), ("grip-close", 10), ("lift", "n_lift"),
          ("path", "n_path"), ("settle", "n_settle"), ("grip-open", 10),
          ("back-off", "n_backoff"), ("tail", "n_tail")]


def phase_spans(row: dict) -> list[tuple[str, int, int]]:
    """掃引の 1 行から `(区間名, 開始コマ, 終了コマ)` の列を作る。

    **足し合わせが `n_steps` に一致することを確かめてから返す。**一致しなければ
    区間の並びか `grip_hold` が変わっているので、**ずれた帯を焼き込むより
    帯を出さないほうがよい**（見る側は帯を信じてしまう）。
    """
    spans, at = [], 0
    for name, key in PHASES:
        length = key if isinstance(key, int) else int(row.get(key, 0))
        if length > 0:
            spans.append((name, at, at + length))
        at += length
    return spans if at == int(row.get("n_steps", -1)) else []


def load_rows(json_path: Path) -> dict:
    """掃引の結果を `(episode, t3_init)` で引ける形にする。"""
    import json

    return {(int(r["episode"]), int(sh["t3_init"])): sh
            for r in json.loads(json_path.read_text()) for sh in r["shifts"]}


def write(path: Path, npz_path: Path, tag: str, row: dict | None = None) -> str:
    z = np.load(npz_path, allow_pickle=True)
    front, wrist = z["front"], z["wrist"]
    ok = bool(z["success"]) if "success" in z else True
    instr = str(z["instruction"]) if "instruction" in z else ""
    n, h, w = front.shape[:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                         (w * 2 + 8, h))
    head = f"{tag}  {n} frames  {'success' if ok else 'FAILED'}"
    spans = phase_spans(row) if row else []
    # **release までのコマ数が本番の 300 打ち切りに掛かる値である**（tail は
    # `done` で切られるので実行されない）。見る側がいちばん知りたい数なので出す。
    pre_tail = int(row.get("n_pre_tail", 0)) if row else 0
    # **帯に載せる字は ASCII だけにする。**`cv2.putText` の HERSHEY フォントは
    # 日本語の字形を持たないので、混ぜると全部 `?` になる（実際にそうなった）。
    sub = instr[:58]
    if spans:
        sub = f"to-release {pre_tail}/300   " + sub[:32]
    for t in range(n):
        # 学習データは 180 度回して入っているので、見るときは戻す
        f = cv2.cvtColor(front[t][::-1, ::-1], cv2.COLOR_RGB2BGR)
        r = cv2.cvtColor(wrist[t][::-1, ::-1], cv2.COLOR_RGB2BGR)
        now = next((f"{nm} {t - a + 1}/{b - a}"
                    for nm, a, b in spans if a <= t < b), "")
        pane = np.hstack([label(f, f"front  {head}", sub),
                          np.full((h, 8, 3), 255, np.uint8),
                          label(r, f"wrist  t={t}", now)])
        if spans:
            # 下に区間の帯を敷く。**いまいる区間だけ明るくする**ので、
            # 「どこで止まっているか」が 1 目で分かる。
            y0 = pane.shape[0] - 10
            for i, (nm, a, b) in enumerate(spans):
                x0 = int(a / n * pane.shape[1])
                x1 = int(b / n * pane.shape[1])
                lit = a <= t < b
                cv2.rectangle(pane, (x0, y0), (x1 - 1, pane.shape[0] - 1),
                              (60, 220, 255) if lit else (70, 70, 70), -1)
            x = int(t / n * pane.shape[1])
            cv2.line(pane, (x, y0 - 4), (x, pane.shape[0] - 1), (0, 0, 255), 1)
        vw.write(pane)
    vw.release()
    extra = f"  release まで {pre_tail} コマ" if spans else ""
    return f"{path.name}  {n} コマ  {'成功' if ok else '失敗'}{extra}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="出力名の頭に付ける（book など）")
    ap.add_argument("--episodes", required=True, help="順方向デモの番号をカンマ区切り")
    ap.add_argument("--inits", default="0,25", help="各デモから取る init の行")
    ap.add_argument("--json", type=Path, default=None,
                    help="掃引の gripper_shift_sweep.json。渡すと区間の帯を焼き込む")
    args = ap.parse_args()

    rows = load_rows(args.json) if args.json else {}
    args.out.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes.split(","):
        for i in args.inits.split(","):
            src = args.npz_dir / f"rev_ep{ep}_i{i}_k+00.npz"
            if not src.exists():
                print(f"  無い: {src.name}")
                continue
            dst = args.out / f"{args.tag}_ep{ep}_init{int(i):02d}.mp4"
            print("  " + write(dst, src, f"{args.tag} ep{ep} init{i}",
                               rows.get((int(ep), int(i)))))


if __name__ == "__main__":
    main()
