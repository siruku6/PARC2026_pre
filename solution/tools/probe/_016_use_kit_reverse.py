#!/usr/bin/env python3
"""016: 配布キットが持つ track3 タスクを、そのまま「逆向き側」として据える。

配布 4 件のうち **`take_the_two_mugs_off_the_plates_and_place_them_on_the_table` は、
学習データに入っている `LIVING_ROOM_SCENE5_...` の厳密な逆向き**である:

  - 物体の顔ぶれが同一（porcelain_mug_1・red_coffee_mug_1・white_yellow_mug_1・plate_1・plate_2）
  - 配布側の `:init` = SCENE5 の `:goal`（2 つのマグが皿の上に在る）
  - 配布側の `:goal` = SCENE5 の `:init`（2 つのマグが元のテーブル領域に戻る）

つまり**逆向きタスクを自分で合成する必要がない**。主催者が書いた定義と初期状態を
そのまま使えるので、「逆向きの作り方が悪かったのでは」という反論の余地が消える。
このプロブで一番強い 1 組である。

本スクリプトは、そのキットの assets を逆向きディレクトリへ `<順方向stem>_rev.*`
という名前で置き、`_016_assemble_t3assets.py` が合成ぶんと同じ扱いで拾えるようにする。

用法:
    python3 _016_use_kit_reverse.py --kit-assets <kit>/compe/t3/assets \
        --reverse-dir work/probe_016/reverse \
        --pair <順方向stem>=<キットのタスク名>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-assets", required=True,
                    help="キットの compe/t3/assets ディレクトリ")
    ap.add_argument("--reverse-dir", required=True)
    ap.add_argument("--pair", action="append", required=True,
                    metavar="FWD_STEM=KIT_TASK",
                    help="順方向 stem と、その逆向きに当たるキットのタスク名")
    args = ap.parse_args()

    kit = Path(args.kit_assets)
    rev_dir = Path(args.reverse_dir)
    rev_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for pair in args.pair:
        if "=" not in pair:
            print(f"  !! --pair は FWD_STEM=KIT_TASK の形: {pair}", file=sys.stderr)
            failed = True
            continue
        fwd_stem, kit_task = pair.split("=", 1)
        src_bddl = kit / "bddl_files" / "libero_t3" / f"{kit_task}.bddl"
        src_init = kit / "init_files" / "libero_t3" / f"{kit_task}.pruned_init"
        missing = [str(p) for p in (src_bddl, src_init) if not p.exists()]
        if missing:
            print(f"  !! キットに無い: {missing}", file=sys.stderr)
            failed = True
            continue
        shutil.copyfile(src_bddl, rev_dir / f"{fwd_stem}_rev.bddl")
        shutil.copyfile(src_init, rev_dir / f"{fwd_stem}_rev.pruned_init")
        print(f"  {fwd_stem}: 逆向きにキットの '{kit_task}' を据えた（合成なし）")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
