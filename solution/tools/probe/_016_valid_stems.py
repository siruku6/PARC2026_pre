#!/usr/bin/env python3
"""016: 検証（`_016_verify_reverse.py`）を通った組の stem だけを 1 行ずつ出す。

評価へ回すのは検証を通った組だけにする。逆向きの初期状態を作り損ねた組を混ぜると、
そこでの失敗が「重みが逆向きの型を持たない」のか「初期配置が変なだけ」なのか
分離できなくなり、このプロブの答え（重みの切り分け）が読めなくなる。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("用法: _016_valid_stems.py <verify_reverse.json>")
    path = Path(sys.argv[1])
    if not path.exists():
        return
    for record in json.loads(path.read_text(encoding="utf-8")):
        if record.get("valid"):
            print(record["stem"])


if __name__ == "__main__":
    main()
