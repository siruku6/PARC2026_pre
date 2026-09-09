#!/usr/bin/env python3
"""接触の判定が付いたデモを、何がどれだけ動いたかを焼き込んだ動画にする。

**数字だけでは「何が起きているのか」が読めない。**`push the plate` が 241 本中
201 本で落ちる、と言われても、皿を押すときに何を巻き込んでいるのかは分からない。
**記録された `states` をそのまま描画し、対象外の物体のずれを画面に焼く。**

**物理は 1 歩も進めない。**`states` を 1 コマずつ流し込んで描くだけなので、
映るのは記録そのものである。**描画は osmesa（CPU）で足りる**——JAIST サーバには
EGL が無いが、1 コマ 12 ms 程度なので 1 本 3 秒で終わる。

**選ぶのは 1 課題につき 3 本。**外れた組のうち**ずれが中央のもの**（典型）と
**いちばん大きいもの**（最悪）、それに**外れなかった組**（対照）である。
**対照が要るのは、「掃除して残る側がどう見えるか」が判断の材料だからである。**

用法（`parc2026_verify` の中で打つ）:
    python render_contact_videos.py --states /tmp/contact_clean/libero_states \\
        --scores /tmp/contact_clean/contact_scores.json \\
        --bddl-root /workspace/LIBERO-plus/libero/libero/bddl_files \\
        --out /tmp/contact_clean/videos \\
        --instruction "push the plate to the front of the stove"

Reads:
  - <states>/<suite>/<課題>_demo.npz
  - <scores> … score_contact.py が書いた判定表
  - <bddl-root>/<suite>/<課題>.bddl
Generates:
  - <out>/<課題の短い名前>_<種別>_<demo>.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

BASELINE_FRAME = 10
COLLISION_THRESHOLD_M = 0.001
CAMERA = "agentview"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--states", type=Path, required=True, metavar="DIR")
    parser.add_argument("--scores", type=Path, required=True, metavar="JSON")
    parser.add_argument("--bddl-root", type=Path, required=True, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--instruction", action="append", required=True,
                        metavar="TEXT", help="動画にする指示文（複数回渡せる）")
    parser.add_argument("--size", type=int, default=384, metavar="PX")
    return parser.parse_args()


def _worst(row: dict) -> tuple[str, float]:
    """その組でいちばん動いた対象外の物体と、その量（m）。"""
    table = row["score"]["displacement_l1_m"].get(str(BASELINE_FRAME), {})
    if not table:
        return ("", 0.0)
    name = max(table, key=table.get)
    return (name, table[name])


def _pick_demos(rows: list[dict]) -> list[tuple[str, dict]]:
    """典型・最悪・対照の 3 本を選ぶ。**外れた組が無ければ対照だけ返す。**"""
    dirty = sorted((row for row in rows if _worst(row)[1] > COLLISION_THRESHOLD_M),
                   key=lambda row: _worst(row)[1])
    clean = sorted((row for row in rows if _worst(row)[1] <= COLLISION_THRESHOLD_M),
                   key=lambda row: _worst(row)[1])
    picked: list[tuple[str, dict]] = []
    if dirty:
        picked.append(("typical", dirty[len(dirty) // 2]))
        picked.append(("worst", dirty[-1]))
    if clean:
        picked.append(("kept", clean[0]))
    return picked


def _make_render_env(bddl_file: Path, size: int):
    """描画付きの env。**物理は `_make_env` と同じで、画が乗るだけである。**"""
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_names=[CAMERA],
                              camera_heights=size, camera_widths=size,
                              control_freq=20, horizon=100000, ignore_done=True)


def _draw_caption(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]
                  ) -> np.ndarray:
    """画面の下に 1 行ずつ焼く。**日本語は `?` になるので英数字だけ**にする。

    **色は RGB で渡す。**書き出す直前に `RGB2BGR` を掛けるので、ここで BGR を
    渡すと赤と青が入れ替わる（9/9 に 1 度そうなった）。
    """
    import cv2

    canvas = np.ascontiguousarray(frame)
    height = canvas.shape[0]
    for number, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (6, height - 8 - 16 * (len(lines) - 1 - number)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return canvas


def _render_one(env, states: np.ndarray, body_id: int, object_name: str,
                path: Path, size: int) -> float:
    """1 デモぶんを描いて mp4 に書き、いちばん動いた量（mm）を返す。

    Generates:
      - <path> … agentview の mp4（20 fps・`states` と同じコマ数）
    """
    import cv2

    sim = env.env.sim
    frames: list[np.ndarray] = []
    baseline: np.ndarray | None = None
    peak = 0.0

    for number, state in enumerate(states):
        sim.set_state_from_flattened(state)
        sim.forward()
        position = np.array(sim.data.body_xpos[body_id], dtype=np.float64)
        if number == BASELINE_FRAME:
            baseline = position.copy()
        shift_mm = 0.0 if baseline is None else float(np.abs(position - baseline).sum()) * 1e3
        peak = max(peak, shift_mm)

        image = sim.render(width=size, height=size, camera_name=CAMERA)[::-1]
        over = shift_mm > COLLISION_THRESHOLD_M * 1e3
        frames.append(_draw_caption(image, [
            (f"frame {number:3d}/{len(states) - 1}", (255, 255, 255)),
            (f"{object_name}  {shift_mm:6.2f} mm" + ("   OVER 1mm" if over else ""),
             (255, 0, 0) if over else (0, 160, 0)),
        ]))

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                             (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return peak


def _short_name(instruction: str) -> str:
    """ファイル名に使える短い名前にする。"""
    return re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")[:48]


def _render_instruction(instruction: str, rows: list[dict], args: argparse.Namespace) -> None:
    """1 指示文ぶんの env を 1 度だけ作り、選んだ 3 本を描く。"""
    picked = _pick_demos(rows)
    suite = rows[0]["source_file"].split("/")[0]
    bddl_file = args.bddl_root / suite / (Path(rows[0]["source_file"]).name
                                          .replace("_demo.hdf5", ".bddl"))
    print(f"\n=== {instruction} ===")
    print(f"  bddl {bddl_file.name}  組 {len(rows)}  選んだ {len(picked)} 本")

    env = _make_render_env(bddl_file, args.size)
    try:
        sim = env.env.sim
        bundle_cache: dict[str, np.lib.npyio.NpzFile] = {}
        for kind, row in picked:
            object_name, value = _worst(row)
            if not object_name:
                object_name = row["score"]["objects"][0]
            body_id = sim.model.body_name2id(f"{object_name}_main")
            source = row["source_file"]
            if source not in bundle_cache:
                bundle_cache[source] = np.load(args.states / source.replace(".hdf5", ".npz"))
            states = bundle_cache[source][f"{row['source_demo']}/states"]
            path = args.out / f"{_short_name(instruction)}__{kind}__{row['source_demo']}.mp4"
            peak = _render_one(env, states, body_id, object_name, path, args.size)
            print(f"  {kind:8s} {row['source_demo']:>8s}  {object_name}  "
                  f"判定 {value * 1e3:.2f} mm / 描画 {peak:.2f} mm  -> {path.name}",
                  flush=True)
        for bundle in bundle_cache.values():
            bundle.close()
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")

    scores: list[dict] = json.loads(args.scores.read_text())
    by_instruction: dict[str, list[dict]] = {}
    for row in scores:
        by_instruction.setdefault(row["instruction"], []).append(row)

    for instruction in args.instruction:
        rows = by_instruction.get(instruction)
        if rows is None:
            raise SystemExit(f"その指示文は判定表に無い: {instruction}")
        _render_instruction(instruction, rows, args)

    print(f"\n=== 書いた: {args.out} ===")


if __name__ == "__main__":
    main()
