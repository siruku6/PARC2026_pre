#!/usr/bin/env python3
"""試し走りの学習ログから、1 歩の秒数と「何時間で何歩回せるか」を出す。

`runpod_initpose_ft.sh` の `TRIAL=1` が呼ぶ。本番の歩数と batch を決めるための表である。

    python3 summarize_speed.py <学習ログ> --batch 64 --peak-mib 49000

Reads:
  - LeRobot の学習ログ（`step:N ... updt_s:X data_s:Y ... mem_gb:Z` の行）
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

STEP_LINE = re.compile(r"step:(\d+) .*?updt_s:([\d.]+) data_s:([\d.]+).*?mem_gb:([\d.]+)")
STEADY_FROM = 40          # 最初の 20 歩は読み込みで遅く出るので外す
DATA_WAIT_WARN = 0.3      # データ待ちが 1 歩の 3 割を超えたら CPU 不足
HOURS = (4, 6, 8, 10)
ROUND_STEPS = 500


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--peak-mib", type=int, default=0, help="nvidia-smi で見た GPU メモリの山（MiB）")
    args = parser.parse_args()

    rows = [(int(s), float(u), float(d), float(m))
            for s, u, d, m in STEP_LINE.findall(args.log.read_text(errors="replace"))]
    if not rows:
        print("  歩の行が 1 つも無い")
        return 1
    for step, updt, data, mem in rows:
        print(f"  step {step:4d}  updt_s {updt:6.3f}  data_s {data:6.3f}  mem_gb {mem:5.2f}")

    steady = [r for r in rows if r[0] >= STEADY_FROM] or rows
    sec = statistics.median(updt + data for _, updt, data, _ in steady)
    data = statistics.median(data for _, _, data, _ in steady)
    print(f"\n  1 歩 {sec:.3f} 秒（{STEADY_FROM} 歩目以降の中央値）・{args.batch / sec:.1f} サンプル/秒"
          f"・GPU メモリの山 {args.peak_mib / 1024:.1f} GB")
    if data > DATA_WAIT_WARN * sec:
        print(f"  **データ待ちが 1 歩の {100 * data / sec:.0f}% ある。CPU が足りていない（batch を上げても速くならない）**")

    print(f"\n  時間    回せる歩数（{ROUND_STEPS} 刻みで切り捨て）  見るサンプル")
    for hours in HOURS:
        steps = int(hours * 3600 / sec) // ROUND_STEPS * ROUND_STEPS
        print(f"  {hours:2d} 時間  {steps:6d} 歩  {steps * args.batch:12,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
