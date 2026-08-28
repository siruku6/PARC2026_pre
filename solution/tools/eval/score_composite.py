#!/usr/bin/env python3
"""手元の評価結果に、採点側に近い形の合計点を付け直す。

**なぜ要るか。** 配布された `pipeline/scorer.py:124` は
`overall_score = float(overall_success)` で、**総合スコアが成功率そのもの**である。
一方、本選の採点ログ 7 本はどれも 8 エピソードなのに、返ってきた点は
0.370・0.305・0.136・0.154・0.207・0.035 で **1 つも 0.125 の倍数でない**。
**採点側は成功率のほかに連続値の項を持っている。**
`pipeline/total_score.py` の内蔵既定にも `weights` と
`use_total_score_as_overall` という口があり、採点環境は
`total_score_config.json` と `normalization_config.json` を秘匿ファイルとして持ち込む
（`evaluate.py` の `_EVAL_SECRET_PATHS`）。**重みは分からないが、器はこちらにもある。**

**重みは合わせにいかない。順位を変える力があるかどうかを見るための道具である。**
成功率を主にしたまま、歩数・ジャークの少なさ・滑らかさ・経路の短さを少しだけ足す。

**評価を回し直さなくてよい。** 結果 JSON はタスクごとに指標を持っているので、
過去の実行にもそのまま当てられる。

    python3 score_composite.py <実行ディレクトリかJSON> [...]
    python3 score_composite.py --by-category <実行ディレクトリ>   # 条件ごとに出す
    python3 score_composite.py --weights success=0.6,steps=0.2 <...>

## 正規化のしかた

補助の 4 指標は `ref / (ref + x)` で 0〜1 に写す。**基準値 `ref` のとき 0.5 になり、
0 に近づくほど 1、大きくなるほど 0 に近づく。**頭打ちの処理が要らず、
どちらの向きにも動く余地が残る。

**基準値は 2026-08-28 の `0828_002049_track2_val72base` の Base 12 行の実測値である**
（摂動なしの素の状態）。基準を実行ごとに取り直すと実行間で点を比べられなくなるので、
**固定値として置く。**
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Base 12 行の実測値（0828_002049_track2_val72base）。この値のとき補助項は 0.5 になる。
REFERENCE = {
    "steps": 120.3,      # avg_steps_to_success
    "jerk": 3.75,        # avg_cartesian_jerk
    "sparc": 2.40,       # |sparc|
    "path": 1.006,       # cartesian_path_length
}

# 成功率を主にしたまま、残りを少しずつ足す。合計 1.0。
DEFAULT_WEIGHTS = {
    "success": 0.70,
    "steps": 0.12,
    "jerk": 0.08,
    "sparc": 0.06,
    "path": 0.04,
}

# 成功したエピソードが 1 本も無いタスクでは、歩数が JSON に入らない。
# **打ち切りの 300 歩を当てる**——「速かった」ことにしてしまうと、
# 全部落ちたタスクほど点が高くなる。
MAX_STEPS = 300.0


def _smaller_is_better(value: float, ref: float) -> float:
    """小さいほど良い量を 0〜1 に写す。ref のとき 0.5。"""
    if value <= 0:
        return 1.0
    return ref / (ref + value)


def task_components(metrics: dict, success_rate: float) -> dict[str, float]:
    """1 タスクぶんの、正規化した 5 項目を返す。"""
    steps = metrics.get("avg_steps_to_success", MAX_STEPS)
    return {
        "success": success_rate,
        "steps": _smaller_is_better(steps, REFERENCE["steps"]),
        "jerk": _smaller_is_better(metrics["avg_cartesian_jerk"], REFERENCE["jerk"]),
        "sparc": _smaller_is_better(abs(metrics["sparc"]), REFERENCE["sparc"]),
        "path": _smaller_is_better(metrics["cartesian_path_length"], REFERENCE["path"]),
    }


def composite(components: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(weights.values())
    return sum(components[k] * w for k, w in weights.items()) / total


def _find_json(target: Path) -> Path:
    if target.is_file():
        return target
    for candidate in (target / "results" / "server_8000.json", target / "server_8000.json"):
        if candidate.is_file():
            return candidate
    hits = sorted(target.glob("results/*.json"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"結果 JSON が見つからない: {target}")


def _labels_for(run_dir: Path) -> dict[str, str]:
    """実行ディレクトリに写した CSV から、タスク名 → 条件名の対応を作る。"""
    csv_path = run_dir / "TASKS_CSV.csv"
    if not csv_path.is_file():
        return {}
    labels = {}
    for row in csv.DictReader(csv_path.open()):
        category = row.get("category", "")
        note = row.get("difficulty_level", "")
        labels[row["task_id"]] = (
            "Base" if category == "Base" else f"{category.split()[0]} {note}".strip()
        )
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="+", type=Path,
                        help="実行ディレクトリか、結果 JSON")
    parser.add_argument("--weights", type=str, default=None,
                        help="例 success=0.6,steps=0.2（書かないものは既定のまま）")
    parser.add_argument("--by-category", action="store_true",
                        help="TASKS_CSV.csv の条件ごとに分けて出す")
    parser.add_argument("--per-task", action="store_true", help="タスクごとに出す")
    args = parser.parse_args()

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        for pair in args.weights.split(","):
            key, _, value = pair.partition("=")
            key = key.strip()
            if key not in weights:
                print(f"知らない項目: {key}（使えるのは {', '.join(DEFAULT_WEIGHTS)}）")
                return 2
            weights[key] = float(value)

    print("重み  " + "  ".join(f"{k}={v:.2f}" for k, v in weights.items()))
    print("基準値 " + "  ".join(f"{k}={v}" for k, v in REFERENCE.items()))
    print()

    for target in args.targets:
        path = _find_json(target)
        run_dir = path.parent.parent if path.parent.name == "results" else path.parent
        data = json.loads(path.read_text())
        labels = _labels_for(run_dir) if args.by_category else {}

        for track in data["tracks"]:
            tasks = track["tasks"]
            rows = []
            for task in tasks:
                comp = task_components(task["metrics"], task["success_rate"])
                rows.append((task["task_name"], comp, composite(comp, weights),
                             task["metrics"].get("collision_rate", float("nan"))))

            print(f"== {run_dir.name}  [{track['track']}]  {len(tasks)} タスク")
            print(f"   配布キットの総合スコア（成功率そのもの） {track['overall_score']:.4f}")
            mean = {k: sum(r[1][k] for r in rows) / len(rows) for k in weights}
            overall = composite(mean, weights)
            print(f"   **合計点 {overall:.4f}**  内訳 "
                  + "  ".join(f"{k} {mean[k]:.3f}" for k in weights))
            collisions = [r[3] for r in rows if r[3] == r[3]]
            if collisions:
                print(f"   衝突率 {sum(collisions) / len(collisions):.3f}")

            if args.by_category and labels:
                groups: dict[str, list] = {}
                for name, comp, score, coll in rows:
                    groups.setdefault(labels.get(name, "?"), []).append((comp, score, coll))
                print(f"   {'条件':16s}{'合計点':>8s}{'成功':>8s}{'歩数':>8s}{'滑らかさ':>8s}{'衝突率':>8s}")
                for label in sorted(groups):
                    g = groups[label]
                    avg = lambda k: sum(c[k] for c, _, _ in g) / len(g)  # noqa: E731
                    print(f"   {label:16s}{sum(s for _, s, _ in g) / len(g):8.3f}"
                          f"{avg('success'):8.3f}{avg('steps'):8.3f}{avg('sparc'):8.3f}"
                          f"{sum(c for _, _, c in g) / len(g):8.3f}")

            if args.per_task:
                for name, _, score, coll in sorted(rows, key=lambda r: r[2]):
                    print(f"   {score:6.3f}  衝突 {coll:4.2f}  {name[:64]}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
