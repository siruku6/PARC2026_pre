#!/usr/bin/env python3
"""`lineage_probe.py` が貯めた json を、系譜の順に 1 枚の表へまとめる。

    python3 solution/tools/probe/summarize_lineage.py work/probe_color

**見たいのは「各学習で何を得て何を失ったか」である。**だから
重みごとの値ではなく、**系譜の順に並べた変化**として出す。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 系譜の順。**この順に並べないと「どの学習で変わったか」が読めない。**
ORDER = [
    ("base", "私たちは 1 歩も学習していない"),
    ("stage1_24000", "+24,000 歩・expert だけ・b16"),
    ("rp3000", "+3,000 歩・全部・b64"),
    ("trial1500_001500", "+1,500 歩・全部・カメラ拡張"),
    ("proto20_002000", "rp3000 +2,000 歩・全部・逆向き20%"),
    ("t3rev_008700", "rp3000 +8,700 歩・expert・逆向き49%"),
]
# 課題ごとの「正しい対象物」。**到達先が当たっているかの判定に使う。**
TARGET = {
    "take_the_red_mug_off_the_left_plate_and_put_it_on_the_table": "red_coffee_mug_1",
    "take_the_yellow_book_off_the_shelf_and_put_it_on_the_table": "yellow_book_2",
    "take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table": "alphabet_soup_1",
    # **黒椀は 2 つある。**皿とラメキンを結ぶ線分からの距離で決めた（9/8 に実測）——
    # `akita_black_bowl_1` は線分から 4.4 mm・位置比 0.46 で「間」、
    # `akita_black_bowl_2` は 130.4 mm 外れている。
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":
        "akita_black_bowl_1",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": "alphabet_soup_1",
}
# **対象を支えている物は競合ではない。**対象がその上（中）に載っているので、
# 手先から見た方向がほぼ同じになる。**これを競合に数えると、正しく狙えているのに
# 「正解は 2 位」と出る**（9/9 に踏んだ。`base` は缶と赤マグで正しく狙えていたのに
# トレイと皿に 1 位を取られていた。距離の差は缶 37.7 mm・赤マグ 7.2 mm しかない）。
SUPPORT = {
    "take_the_red_mug_off_the_left_plate_and_put_it_on_the_table": ("plate_1",),
    "take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table": ("wooden_tray_1",),
}
SHORT = {
    "take_the_red_mug_off_the_left_plate_and_put_it_on_the_table": "T3 赤マグ(色)",
    "take_the_yellow_book_off_the_shelf_and_put_it_on_the_table": "T3 黄本(色)",
    "take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table": "T3 缶(名前)",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": "対照 黒椀(位置)",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": "対照 缶(名前)",
}


def load(root: Path) -> dict[str, dict]:
    out = {}
    for name, _ in ORDER:
        path = root / f"lin_{name}.json"
        if path.is_file():
            out[name] = json.loads(path.read_text())
    return out


def rank_of(reach: list[dict], target: str,
            support: tuple[str, ...] = ()) -> tuple[str, float, str]:
    """(いちばん強い競合, 対象の cos, 判定) を返す。

    **判定は「対象より強い競合が居るか」で決める。**順位で見ないのは、
    **対象を支えている物（皿・トレイ）が同じ方向に立つ**ためである（`SUPPORT`）。

    **1 列目は「いちばん強い競合」であって「1 位の物」ではない。**
    1 位を出すと、支えが 1 位のときに「1 位=plate_1／判定=正」と同じ行で
    矛盾する（9/9 に踏んだ）。

    **物の位置を記録していない npz では `reach` が空になる。**その場合は
    「測っていない」と分かる形で返す（推測で埋めない）。
    """
    if not reach:
        return "（物の位置を記録していない）", float("nan"), "-"
    hit = next((r for r in reach if r["object"] == target), None)
    if hit is None:
        return reach[0]["object"], float("nan"), "対象が無し"
    rivals = [r for r in reach
              if r["object"] != target and r["object"] not in support]
    best = max(rivals, key=lambda r: r["cos"], default=None)
    if best is None:
        return "（競合なし）", hit["cos"], "正"
    margin = hit["cos"] - best["cos"]
    verdict = f"正（差 {margin:+.3f}）" if margin >= 0 else f"誤（差 {margin:+.3f}）"
    return f"{best['object']} {best['cos']:+.3f}", hit["cos"], verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    data = load(args.root)
    if not data:
        raise SystemExit(f"{args.root} に lin_*.json が無い")

    tasks = [t for t in SHORT if any(t in d["tasks"] for d in data.values())]

    print("=" * 100)
    print("【A】手先が向かう先 —— 1 位の物と、正しい物の順位")
    print("=" * 100)
    for task in tasks:
        target = TARGET[task]
        print(f"\n{SHORT[task]}   正解 = {target}")
        print(f"  {'重み':20}{'いちばん強い競合':30}{'対象 cos':>9}{'判定':>18}  その学習で変えたこと")
        for name, note in ORDER:
            d = data.get(name)
            if not d or task not in d["tasks"]:
                continue
            rival, cos, verdict = rank_of(d["tasks"][task]["reach"], target,
                                          SUPPORT.get(task, ()))
            m = d["tasks"][task].get("motion") or {}
            extra = (f"  並進{m['reach_mm']:5.0f}mm 閉じる{m['grip_close_steps']:2d}/50"
                     if m else "")
            print(f"  {name:20}{rival:30}{cos:>+9.3f}{verdict:>18}{extra}  {note}")

    print()
    print("=" * 100)
    print("【B】指示文の効き —— 種別ごとの平均（prefix / 行動列）")
    print("=" * 100)
    kinds = ("same", "color", "place", "noun")
    for task in tasks:
        print(f"\n{SHORT[task]}")
        head = "".join(f"{k:>18}" for k in kinds)
        print(f"  {'重み':20}{head}")
        for name, _ in ORDER:
            d = data.get(name)
            if not d or task not in d["tasks"]:
                continue
            rows = d["tasks"][task]["sensitivity"]
            cells = ""
            for k in kinds:
                vals = [(r["d_prefix_img"], r["d_chunk"]) for r in rows if r["kind"] == k]
                if not vals:
                    cells += f"{'—':>18}"
                    continue
                pre = sum(v[0] for v in vals) / len(vals)
                chu = sum(v[1] for v in vals) / len(vals)
                cells += f"{pre:>8.4f}/{chu:<9.4f}"
            print(f"  {name:20}{cells}")


if __name__ == "__main__":
    main()
