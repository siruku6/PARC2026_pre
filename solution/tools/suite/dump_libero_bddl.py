#!/usr/bin/env python3
"""採点器が読む bddl の一式を、評価イメージの中から取り出す。

**`:goal` と `:init` は、指示文の書き換えが正しいかどうかの根拠になる。**
どの物を掴めば採点されるのか・同じ見た目の物が他にも置かれていないかは、
bddl を読まないと分からない
（[docs/reference/36](../../../docs/reference/36_instruction_rewriting_table.md) 5 節）。

bddl は LIBERO-plus に同梱されていて、**手元の木には無い。**評価イメージ
`parc2026-final` の中にしかないので、中で走らせて repo 側へ写す。

    IMAGE=parc2026-final bash solution/tools/regen/run_in_container.sh \\
        solution/tools/suite/dump_libero_bddl.py

Reads:
  - <LIBERO-plus>/libero/libero/bddl_files/**/*.bddl
Generates:
  - work/instr_audit/bddl_files/**/*.bddl
"""

from __future__ import annotations

import os
import pathlib
import shutil

import libero.libero

# `run_in_container.sh` は repo をここへ結びつける。
_REPOSITORY_IN_CONTAINER = pathlib.Path("/workspace/repo")
_DESTINATION = _REPOSITORY_IN_CONTAINER / "work/instr_audit/bddl_files"


def main() -> None:
    source = pathlib.Path(os.path.dirname(libero.libero.__file__)) / "bddl_files"
    if not source.is_dir():
        raise SystemExit(f"bddl_files が見つかりません: {source}")

    if _DESTINATION.exists():
        shutil.rmtree(_DESTINATION)
    _DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, _DESTINATION)

    count = sum(1 for _ in _DESTINATION.rglob("*.bddl"))
    print(f"{source} -> {_DESTINATION}")
    print(f"{count} 件")


if __name__ == "__main__":
    main()
