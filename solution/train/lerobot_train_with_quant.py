#!/usr/bin/env python3
"""`lerobot-train`（lerobot.scripts.lerobot_train:main の console script）の代替。

量子化実験の各候補（quant_optim.py / quant_vision_4bit.py / quant_float8_expert.py。
work/quant_sweep/CANDIDATES.md 参照）を、対応する環境変数が立っているときだけ import して
から main() を呼ぶ。import の副作用（レジストリ登録・モンキーパッチ）だけが目的で、
CLI の引数は lerobot-train と完全に同じものをそのまま渡す —— ここでは一切解釈しない。

train_pi05.py の _build_command()/_build_resume_command() が、量子化候補のどれかを
使うときに "lerobot-train" の代わりにこれを呼ぶ（`["python", str(_QUANT_WRAPPER)]`）。
どの候補を使うかは train_pi05.py 側が以下の環境変数で伝える:

    QUANT_OPTIMIZER_MODE=bnb8bit   → quant_optim.py（C1）
    QUANT_VISION_MODE=bnb4bit      → quant_vision_4bit.py（C2）
    QUANT_EXPERT_MODE=float8       → quant_float8_expert.py（C3）

複数を同時に立てて組み合わせを試すこともできる（例: C1+C3 を両方 import）。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if os.environ.get("QUANT_OPTIMIZER_MODE") == "bnb8bit":
    import quant_optim  # noqa: F401  副作用: "adamw_bnb8bit" を登録

if os.environ.get("QUANT_VISION_MODE") == "bnb4bit":
    import quant_vision_4bit  # noqa: F401  副作用: vision encoder を 4bit 化するようパッチ

if os.environ.get("QUANT_EXPERT_MODE") == "float8":
    import quant_float8_expert  # noqa: F401  副作用: action expert を float8 化するようパッチ

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
