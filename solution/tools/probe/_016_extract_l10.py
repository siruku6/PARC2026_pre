#!/usr/bin/env python3
"""016 下調べ: フォーク上の libero_10 全 bddl と初期状態をホストへ書き出す。

評価コンテナ（parc2026-final）の中で走る。ホスト側の出力ディレクトリを
`-v` で /out にマウントし、そこへ bddl と init_files をコピーする。

用法（ホスト側から、docker run はスクリプト経由で）:
    docker run --rm -v <host_out>:/out parc2026-final \
        python /app/_016_extract_l10.py
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/workspace/LIBERO-plus")
BDDL_DIR = Path(LIBERO_ROOT) / "libero" / "libero" / "bddl_files" / "libero_10"
INIT_DIR = Path(LIBERO_ROOT) / "libero" / "libero" / "init_files" / "libero_10"
OUT = Path(os.environ.get("OUT_DIR", "/out"))


def base_task_stems() -> list[str]:
    """variant を除いた元タスク（SCENE 名）の stem 一覧。"""
    stems = []
    for p in sorted(BDDL_DIR.glob("*.bddl")):
        f = p.stem
        if re.search(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+", f) is None:
            continue
        if any(t in f for t in ("_add_", "_view_", "_language_", "_level",
                                "copy", "_sample", "_tb_", "_light_", "_table_")):
            continue
        after = re.sub(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", f)
        if re.search(r"_\d+$", after):
            continue
        stems.append(f)
    return stems


def main() -> None:
    out_bddl = OUT / "bddl_files"
    out_init = OUT / "init_files"
    out_bddl.mkdir(parents=True, exist_ok=True)
    out_init.mkdir(parents=True, exist_ok=True)

    stems = base_task_stems()
    print(f"libero_10 の元タスク（SCENE 名・variant なし）は {len(stems)} 個")

    for stem in stems:
        src_bddl = BDDL_DIR / f"{stem}.bddl"
        if not src_bddl.exists():
            print(f"  [欠け] bddl 無い: {stem}")
            continue
        shutil.copyfile(src_bddl, out_bddl / f"{stem}.bddl")

        src_init = INIT_DIR / f"{stem}.pruned_init"
        if src_init.exists():
            shutil.copyfile(src_init, out_init / f"{stem}.pruned_init")
            print(f"  {stem}: bddl + init 書き出し")
        else:
            print(f"  {stem}: bddl だけ（init 無い: {INIT_DIR / (stem + '.pruned_init')}）")

    print(f"\n書き出し先: {OUT}")
    print(f"bddl {len(list(out_bddl.glob('*.bddl')))} 件 / "
          f"init {len(list(out_init.glob('*.pruned_init')))} 件")


if __name__ == "__main__":
    main()