#!/usr/bin/env python3
"""手元の評価結果に、採点側と同じ形の合計点を付け直す。

**式は運営が公開している**（本選の資料。`docs/reference/17_scoring_formula.md`）。

    smooth metrics = w1*time + w2*jerk + w3*SPARC + w4*trajectory + w5*rotation
    Total Score    = (1/N) * Σ success_i * (1 - collision penalty_i) * smooth metrics_i

**読み方が 3 つある。**

1. **成功は重み付きの項ではなく、掛け算の門である。**
   失敗した回、または衝突した回は、**その回の点がまるごと 0 になる。**
   なめらかさがどれだけ良くても拾えない。
2. **重み付きの和に入るのは 5 つだけ**——time・jerk・SPARC・trajectory・rotation。
   重みは非公開だが `w1+...+w5 = 1` で、「特定の指標に偏らせないための重み付け」と
   説明されている。**だから既定は 5 等分（各 0.2）にした。**
3. **正規化の式は非公開**だが、各指標を 0〜1 の連続値に写すこと、
   collision penalty が [0,1] の 2 値であることは公開されている。

**目盛りの合わせどころ。** 予選では**全 rollout が成功しても点は 0.48** だった。
つまり `smooth metrics` の実力値がその辺りに来る正規化になっている。
下の基準値はそこに合わせてある（全問成功で 0.49 付近）。

**評価を回し直さなくてよい。** 結果 JSON はタスクごとに 5 指標をすべて持っているので、
過去の実行にもそのまま当てられる。

    python3 score_composite.py <実行ディレクトリかJSON> [...]
    python3 score_composite.py --by-category <実行ディレクトリ>
    python3 score_composite.py --weights time=0.3,jerk=0.1 <...>

## 正規化のしかた

5 指標はすべて「小さいほど良い」ので `ref / (ref + x)` で 0〜1 に写す。
**基準値 `ref` のとき 0.5** になり、頭打ちの処理が要らず、どちらの向きにも余地が残る。

**基準値は 2026-08-28 の `0828_002049_track2_val72base` の Base 12 行の実測値である**
（摂動なしの素の状態）。**実行ごとに取り直すと実行間で比べられなくなるので固定値にする。**

## 手元の JSON で式どおりに計算できない点

採点側は **1 rollout ごと**に `success * (1-collision) * smooth` を計算して平均する。
手元の JSON は**タスクごとの平均**しか持たないので、
`タスクの点 = success_rate * smooth(タスク)` で近似する。
**成功率が 0 でも 1 でもないタスクでは、成功した回と落ちた回のなめらかさを
区別できない**——ここだけは近似である。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Base 12 行の実測値（0828_002049_track2_val72base）。この値のとき補助項は 0.5 になる。
REFERENCE = {
    "time": 120.3,        # avg_steps_to_success（歩数を time の代わりに使う）
    "jerk": 3.75,         # avg_cartesian_jerk
    "sparc": 2.40,        # |sparc|
    "trajectory": 1.006,  # cartesian_path_length
    "rotation": 1.333,    # orientation_path_length
}

# 5 等分。運営の説明が「特定の指標に偏らせないための重み付け」なので、
# **偏らせない側に倒す。**合計は 1 になる。
DEFAULT_WEIGHTS = {
    "time": 0.20,
    "jerk": 0.20,
    "sparc": 0.20,
    "trajectory": 0.20,
    "rotation": 0.20,
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


def smooth_components(metrics: dict) -> dict[str, float]:
    """1 タスクぶんの、正規化した 5 指標を返す。成功は入らない（門なので）。"""
    return {
        "time": _smaller_is_better(metrics.get("avg_steps_to_success", MAX_STEPS),
                                   REFERENCE["time"]),
        "jerk": _smaller_is_better(metrics["avg_cartesian_jerk"], REFERENCE["jerk"]),
        "sparc": _smaller_is_better(abs(metrics["sparc"]), REFERENCE["sparc"]),
        "trajectory": _smaller_is_better(metrics["cartesian_path_length"],
                                          REFERENCE["trajectory"]),
        "rotation": _smaller_is_better(metrics["orientation_path_length"],
                                        REFERENCE["rotation"]),
    }


def smooth_metrics(components: dict[str, float], weights: dict[str, float]) -> float:
    """重み付きの和。運営の式の `smooth metrics` にあたる。"""
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
                comp = smooth_components(task["metrics"])
                smooth = smooth_metrics(comp, weights)
                success = task["success_rate"]
                collision = task["metrics"].get("collision_rate", float("nan"))
                # 運営の式: success * (1 - collision penalty) * smooth。
                # **手元の success は既に「ゴール到達 かつ 無衝突」なので、
                # (1 - collision) を重ねて掛けると二重に引くことになる。**
                # （`pipeline/rollout.py:234` の `success = bool(done) and not collided`）
                rows.append((task["task_name"], comp, smooth, success,
                             success * smooth, collision))

            n = len(rows)
            total = sum(r[4] for r in rows) / n
            mean_smooth = sum(r[2] for r in rows) / n
            mean_success = sum(r[3] for r in rows) / n
            mean_collision = sum(r[5] for r in rows if r[5] == r[5]) / n if n else 0.0

            print(f"== {run_dir.name}  [{track['track']}]  {n} タスク")
            print(f"   配布キットの総合スコア（成功率そのもの） {track['overall_score']:.4f}")
            print(f"   **合計点 {total:.4f}**"
                  f"   ＝ 成功率 {mean_success:.3f} × なめらかさ {mean_smooth:.3f} 相当")
            print(f"   全部成功したらいくつか（目盛りの確認・予選の実測は 0.48）: {mean_smooth:.4f}")
            print(f"   衝突率 {mean_collision:.3f}")
            comp_mean = {k: sum(r[1][k] for r in rows) / n for k in weights}
            print("   なめらかさの内訳  "
                  + "  ".join(f"{k} {comp_mean[k]:.3f}" for k in weights))

            if args.by_category and labels:
                groups: dict[str, list] = {}
                for name, comp, smooth, success, score, coll in rows:
                    groups.setdefault(labels.get(name, "?"), []).append(
                        (comp, smooth, success, score, coll))
                print(f"   {'条件':16s}{'合計点':>8s}{'成功':>8s}{'なめらかさ':>10s}{'衝突率':>8s}")
                for label in sorted(groups):
                    g = groups[label]
                    m = lambda i: sum(x[i] for x in g) / len(g)  # noqa: E731
                    print(f"   {label:16s}{m(3):8.3f}{m(2):8.3f}{m(1):10.3f}{m(4):8.3f}")

            if args.per_task:
                for name, _, smooth, success, score, coll in sorted(rows, key=lambda r: r[4]):
                    print(f"   {score:6.3f}  成功 {success:4.2f}  なめらかさ {smooth:5.3f}"
                          f"  衝突 {coll:4.2f}  {name[:56]}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
