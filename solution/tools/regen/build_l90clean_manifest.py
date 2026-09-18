#!/usr/bin/env python3
"""`libero_combined_bowl5_initpose_l90clean` の置き換え台帳を書き出す。

**なぜ要るか。** このデータセットは 2026-09-16 に手で組んだので、
`build_initpose_datasets.sh` に段が入っていない。**台帳が無いと、混合データを
消したときに二度と同じものが作れない**（親の `libero_combined_bowl5_initpose` は
既に削除済みである）。

**何を書き出すか。** 外した 250 本の番号・埋め戻しに使った 250 本の中身・
外した判定の基準を 1 つの JSON にまとめる。入力は 2 つとも手元にある現物である。

    python3 solution/tools/regen/build_l90clean_manifest.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

MERGED = Path("work/artifacts/libero_combined_bowl5_initpose_l90clean")
STATES = Path("work/artifacts/libero_90_states")
OUT = Path("solution/tools/regen/manifests/l90clean_replace.json")


def _first_task(value: object) -> str:
    """`tasks` 列は 1 要素の配列で入っている。"""
    if isinstance(value, str):
        return value
    return str(value[0])


def main() -> int:
    summary = json.loads((STATES / "dirty_summary.json").read_text())
    dropped = sorted(summary["dirty"])

    episodes = pd.concat(
        [pd.read_parquet(path) for path in sorted(
            glob.glob(str(MERGED / "meta/episodes/**/*.parquet"), recursive=True))],
        ignore_index=True,
    )
    info = json.loads((MERGED / "meta/info.json").read_text())

    # **置き換えた本はデータセットの末尾に付いている**（本数は外した数と同じ）。
    appended = episodes.tail(len(dropped))[["episode_index", "tasks", "length"]]

    manifest = {
        "dataset": "siruku6/libero_combined_bowl5_initpose_l90clean",
        "parent": "siruku6/libero_combined_bowl5_initpose（削除済み。本数 21,642・コマ 3,436,940）",
        "built": "2026-09-16（手で実施。この台帳は 2026-09-18 に書き出した）",
        "what_changed": (
            "親の libero_90 部分 3,911 本（番号 17731〜21641）のうち、"
            "対象外の物体に接触している 250 本を外し、同じ数の綺麗な手本を末尾に足した。"
            "本数と課題ごとの本数は変わらず、コマ数だけ 5,803 減る。"
        ),
        "criterion": {
            "rule": "対象外の物体への接触があれば外す",
            "source": "work/artifacts/libero_90_states/contact_scores_full.json（4,500 デモ）",
            "dropped_for_contact": 250,
            "dropped_for_length": 0,
            "note": "長さ超過（300 コマ超）で外した本はこの段では 1 本も無い",
        },
        "replacement_pool": {
            "path": "work/artifacts/libero_90_clean_appearance",
            "episodes": 3705,
            "frames": 521185,
            "tasks": 73,
            "note": "外観（照明・机と背景・カメラ）を重ねた libero_90 の綺麗な手本。"
                    "**HF には上がっていない**——この台帳と一緒に上げること",
        },
        "dropped_episode_indices": dropped,
        "appended_episodes": [
            {
                "episode_index": int(row.episode_index),
                "task": _first_task(row.tasks),
                "length": int(row.length),
            }
            for row in appended.itertuples()
        ],
        "totals": {
            "merged_episodes": int(info["total_episodes"]),
            "merged_frames": int(info["total_frames"]),
            "merged_tasks": int(info["total_tasks"]),
            "appended_frames": int(appended["length"].sum()),
            "dropped_frames": int(appended["length"].sum()) + 5803,
        },
    }

    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"書き出した: {OUT}")
    for key, value in manifest["totals"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
