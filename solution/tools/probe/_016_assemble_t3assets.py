#!/usr/bin/env python3
"""016: 選んだタスクの順／逆向き assets をキットの t3 形式に組み立てる。

ホスト側で動く（torch 不要・ファイルのコピーのみ）。
extract + make_reverse が作ったファイルを、評価コンテナが読む
`compe/t3/assets/{bddl_files,init_files}/libero_t3/` と同じ相対構造で並べる。

  t3assets/
    bddl_files/libero_t3/<stem>.bddl            (順方向・そのまま)
    bddl_files/libero_t3/<stem>_rev.bddl        (逆向き)
    init_files/libero_t3/<stem>.pruned_init     (順方向の初期状態)
    init_files/libero_t3/<stem>_rev.pruned_init (逆向きの初期状態)

register_t3() はこのディレクトリの *.bddl を全部読んで 1 タスクとして登録し、
init は `<task>.pruned_init` を読むので、順／逆で別々の初期状態が正しく引かれる。

用法:
    python3 _016_assemble_t3assets.py \
        --forward-bddl <extract_dir> --forward-init <extract_dir> \
        --reverse <reverse_dir> --out <t3assets_dir> \
        --stem <stem1> --stem <stem2> ...
（--forward-bddl と --forward-init は同じ extract ディレクトリでよい）
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-bddl", required=True)
    ap.add_argument("--forward-init", required=True)
    ap.add_argument("--reverse", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", action="append", required=True)
    args = ap.parse_args()

    fwd_bddl = Path(args.forward_bddl)
    fwd_init = Path(args.forward_init)
    rev = Path(args.reverse)
    out = Path(args.out)

    bddl_dir = out / "bddl_files" / "libero_t3"
    init_dir = out / "init_files" / "libero_t3"
    # `register_t3()` はこのディレクトリの *.bddl を**全部**タスクとして登録するので、
    # 前回の残骸が居ると評価対象に紛れ込む。毎回まっさらにしてから並べる。
    for d in (bddl_dir, init_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    skipped = []
    for stem in args.stem:
        need = {
            "順方向 bddl": fwd_bddl / f"{stem}.bddl",
            "順方向 init": fwd_init / f"{stem}.pruned_init",
            "逆向き bddl": rev / f"{stem}_rev.bddl",
            "逆向き init": rev / f"{stem}_rev.pruned_init",
        }
        missing = [k for k, v in need.items() if not v.exists()]
        if missing:
            # 順／逆は対で比べるものなので、片側が欠けた stem は丸ごと外す
            print(f"  !! {stem}: {'・'.join(missing)} が無いので外す")
            skipped.append(stem)
            continue
        shutil.copyfile(need["順方向 bddl"], bddl_dir / f"{stem}.bddl")
        shutil.copyfile(need["順方向 init"], init_dir / f"{stem}.pruned_init")
        shutil.copyfile(need["逆向き bddl"], bddl_dir / f"{stem}_rev.bddl")
        shutil.copyfile(need["逆向き init"], init_dir / f"{stem}_rev.pruned_init")
        print(f"  {stem}: 順＋逆向き を組み立て")

    print(f"\nt3 assets 書き出し: {out}")
    print(f"  bddl {len(list(bddl_dir.glob('*.bddl')))} 件 / "
          f"init {len(list(init_dir.glob('*.pruned_init')))} 件")
    if skipped:
        print(f"  外した stem: {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()