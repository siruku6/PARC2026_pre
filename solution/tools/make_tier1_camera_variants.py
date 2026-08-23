#!/usr/bin/env python3
"""Tier 1（Camera Viewpoints 対策）の撮り直しに使う `--variant` の一覧を作る。

**台本がある 2 タスク専用。** Tier 1 の「撮り直し」（`docs/plan/09_demo_perturbation_pipeline.md`
2.2 節）は、`scripted_demo.py --variant` へ LIBERO-plus の公式 Camera Viewpoints 変種名
（`<task>_view_<horizon>_<vertical>_<scale*100>_<rot>_<vert>_initstate_0` の形）を渡すだけで
動く。**`scripted_demo.py` 自体の変更は要らない**——`ControlEnv.__init__`
（`env_wrapper.py`）がこの文字列を自分でパースしてカメラ設定を組み立てるためで、この
スクリプトはその変種名を選ぶだけの下ごしらえである。

台本は実測の位置から目標への差分を毎ステップ計算する閉ループ制御で、カメラ設定は
描画にしか関わらない（`libero_tabletop_manipulation.py` の `_setup_camera` が使うだけ）。
**つまり撮り直しは物理・行動列を変えない**——同じ初期状態番号なら Base 撮影と全く同じ
成功／失敗になり、画素だけが変わる。σ スイープのような成功率の確認は要らない
（[docs/plan/11_tier1_camera_augmentation.md](../../docs/plan/11_tier1_camera_augmentation.md) 3 節）。

**学習と評価で使う変種を分けること。** `compe/val/TRACK2_PROBE.csv` に載っている
Camera Viewpoints の変種を学習データにも使うと、検証が「見た条件を再現できたか」を
測るだけになる（08_training_data_creation_cautions.md 2.8 節）。このスクリプトは
`--exclude` に渡した CSV の `task_id` 列をそのまま除外する。

選び方: 難易度・`id` の昇順に並べ、等間隔に `--count` 件を取る
（`make_track2_probe.py` の `_pick_tasks` と同じ考え方）。**乱数は使わない**——
同じ引数なら同じ変種になる。

使い方:

    python3 solution/tools/make_tier1_camera_variants.py \\
        --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \\
        --exclude compe/val/TRACK2_PROBE.csv \\
        --count 6 --output work/tier1_camera/CAMERA_VARIANTS.csv

Reads:
  - <libero-plus>/libero/libero/benchmark/task_classification.json
  - --exclude に渡した CSV（`task_id` 列だけを読む）
Generates:
  - --output の CSV（`task,variant,suite,difficulty_level,id` の 4 列）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CATEGORY = "Camera Viewpoints"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", action="append", required=True, dest="tasks",
                        metavar="NAME",
                        help="台本がある元タスク名。複数回渡せる")
    parser.add_argument("--suite", type=str, default="libero_10",
                        help="task_classification.json の外側の鍵。台本がある"
                             "2 タスクはどちらも libero_10")
    parser.add_argument("--count", type=int, default=6,
                        help="タスクごとに選ぶ変種の数")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        metavar="CSV",
                        help="この CSV の task_id 列にある変種名を除外する"
                             "（評価スイートとの重複を避ける。複数回渡せる）")
    parser.add_argument("--libero-plus", type=Path,
                        default=_REPO_ROOT / "work/LIBERO-plus",
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--output", type=Path, required=True, help="出力する CSV")
    parser.add_argument("--dry-run", action="store_true", help="書かずに内容だけ出す")
    return parser.parse_args()


def _load_camera_variants(libero_plus: Path, suite: str, task: str) -> list[dict]:
    """分類表から、そのタスクの Camera Viewpoints 変種だけを取る。"""
    path = libero_plus / "libero/libero/benchmark/task_classification.json"
    if not path.is_file():
        raise SystemExit(f"[tier1-camera] 分類表が見つかりません: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if suite not in raw:
        raise SystemExit(f"[tier1-camera] スイートが分類表に無い: {suite}")
    return [
        record for record in raw[suite]
        if record["category"] == _CATEGORY and record["name"].startswith(task)
    ]


def _load_excluded_names(paths: list[Path]) -> set[str]:
    """`--exclude` に渡された CSV から `task_id` 列を集める。"""
    excluded: set[str] = set()
    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows and "task_id" not in rows[0]:
            raise SystemExit(f"[tier1-camera] task_id 列が無い: {path}")
        excluded.update(row["task_id"] for row in rows)
    return excluded


def _pick_evenly(variants: list[dict], count: int) -> list[dict]:
    """難易度・id の昇順に並べ、等間隔に `count` 件取る。

    **先頭から `count` 件取ってはいけない。** 分類表は変種を強さの順に
    並べていることがあり、先頭だけ取ると弱い摂動しか撮り直さない
    （`make_track2_probe.py` の `_pick_variant` と同じ注意点）。
    """
    ordered = sorted(variants, key=lambda entry: (entry["difficulty_level"], entry["id"]))
    take = min(count, len(ordered))
    if take <= 1:
        return ordered[:take]
    step = (len(ordered) - 1) / (take - 1)
    return [ordered[round(index * step)] for index in range(take)]


def main() -> None:
    args = _parse_args()
    excluded = _load_excluded_names(args.exclude)
    if excluded:
        print(f"[tier1-camera] 評価スイートと重ねない除外 {len(excluded)} 件"
              f"（{', '.join(str(path) for path in args.exclude)}）")

    rows: list[dict] = []
    for task in args.tasks:
        available = _load_camera_variants(args.libero_plus, args.suite, task)
        remaining = [entry for entry in available if entry["name"] not in excluded]
        picked = _pick_evenly(remaining, args.count)
        print(f"[tier1-camera] {task}: 分類表 {len(available)} 件 → "
              f"除外後 {len(remaining)} 件 → {len(picked)} 件選出")
        for entry in picked:
            print(f"[tier1-camera]   {entry['name']}"
                  f"（difficulty {entry['difficulty_level']}）")
        rows.extend({
            "task": task,
            "variant": entry["name"],
            "suite": entry.get("suite", args.suite),
            "difficulty_level": entry["difficulty_level"],
            "id": entry["id"],
        } for entry in picked)

    if not rows:
        raise SystemExit("[tier1-camera] 1 件も選べませんでした")

    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tier1-camera] 書き出し: {args.output}（{len(rows)} 行）")

    print("\n[tier1-camera] 撮る例（1 行につき、既存の record と同じ手順）:")
    for row in rows:
        print(f"[tier1-camera]   MUJOCO_GL=osmesa python3 solution/tools/scripted_demo.py "
              f"record --task {row['task']} --variant {row['variant']} "
              f"--output outputs/scripted/tier1_camera")


if __name__ == "__main__":
    main()
