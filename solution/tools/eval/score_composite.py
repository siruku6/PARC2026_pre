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
    python3 score_composite.py --weights n_steps=0.3,jerk=0.1 <...>

**実行ディレクトリを渡したときは `<実行ディレクトリ>/score_composite.md` にも書く。**
標準出力だけだと、あとから「この点はどの式で出したのか」を引けない。
**書き出すのは点だけでなく式・重み・基準値・近似した 3 点も一緒である**
（`--no-write` で止められる）。`run_final_eval.sh` が評価のあと自動で呼ぶので、
ふつうは自分で打たなくてもファイルはできている。

## 運営の `time` を、こちらは `n_steps` と呼ぶ

**運営の式の第 1 項は `time` という名前だが、中身は歩数である**
（運営 Q&A No.23。`docs/knowledge/06_step_count_drives_score.md`）。
**`time` のままだと かかった時間 と読み違えるので、この道具では `n_steps` と呼ぶ。**
入れている値は `avg_steps_to_success` である。結果 JSON には実時間
`avg_episode_time_sec` もあるが、**描画のやり方（egl / osmesa）とマシンで何倍も
変わるので実行間で比べられない。**

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

# **基準値。**`norm(x, ref) = ref / (ref + x)` の `ref` で、**実測がこの値のとき項が
# ちょうど 0.5 になる**。0〜1 に写すための目盛りの原点であって、目標値ではない。
# 値は 2026-08-28 の `0828_002049_track2_val72base` の **Base 12 行（摂動なし）の実測値**
# である。**実行ごとに取り直すと実行間で比べられなくなるので固定値にする。**
REFERENCE = {
    "n_steps": 120.3,     # 歩
    "jerk": 3.75,         # m/s^3
    "sparc": 2.40,        # 無次元（符号を落とした値）
    "trajectory": 1.006,  # m
    "rotation": 1.333,    # rad
}

# **運営の式の第 1 項 `time` を、この道具では `n_steps` と呼ぶ。**
# 運営の `time` は かかった時間 ではなく歩数だからである（運営 Q&A No.23）。結果 JSON には
# 実時間 `avg_episode_time_sec` もあるが、**描画のやり方（egl / osmesa）とマシンで
# 何倍も変わるので実行間で比べられない。**歩数は打ち切りの 300 で頭が揃う。
SOURCE = {
    "n_steps": ("avg_steps_to_success", "歩"),
    "jerk": ("avg_cartesian_jerk", "m/s^3"),
    "sparc": ("abs(sparc)", "無次元"),
    "trajectory": ("cartesian_path_length", "m"),
    "rotation": ("orientation_path_length", "rad"),
}

# 5 等分。運営の説明が「特定の指標に偏らせないための重み付け」なので、
# **偏らせない側に倒す。**合計は 1 になる。
DEFAULT_WEIGHTS = {
    "n_steps": 0.20,
    "jerk": 0.20,
    "sparc": 0.20,
    "trajectory": 0.20,
    "rotation": 0.20,
}

# 成功したエピソードが 1 本も無いタスクでは、歩数が JSON に入らない。
# **打ち切りの 300 歩を当てる**——「速かった」ことにしてしまうと、
# 全部落ちたタスクほど点が高くなる。
MAX_STEPS = 300.0


def formula_text(weights: dict[str, float]) -> str:
    """点と一緒に書き出す計算式。**点だけを見せない**ための本文である。"""
    terms = " + ".join(f"{w:.2f}·{k}" for k, w in weights.items())
    table = "\n".join(
        f"| `{k}` | {SOURCE[k][0]} | {SOURCE[k][1]} | {REFERENCE[k]} | {weights[k]:.2f} |"
        for k in weights)
    return f"""## 計算式

**運営が公開している式**（本選の資料。`docs/reference/17_scoring_formula.md`）

    smooth metrics = w1·time + w2·jerk + w3·SPARC + w4·trajectory + w5·rotation
    Total Score    = (1/N) · Σ success_i · (1 - collision penalty_i) · smooth metrics_i

**この道具が実際に計算している式**（第 1 項の呼び名だけ `time` → `n_steps` に変えてある。
運営の `time` は かかった時間 ではなく歩数だからである——運営 Q&A No.23）

    正規化      norm(x, ref) = ref / (ref + x)        ← ref のとき 0.5
    なめらかさ  smooth(task) = {terms}
                （各項は norm(その指標, 基準値)）
    タスクの点  score(task)  = success_rate(task) · smooth(task)
    合計点      Total        = (1/N) · Σ score(task)

**5 つの項の中身・単位・基準値・重み**

| 項 | 結果 JSON のどれか | 単位 | 基準値 | 重み |
|---|---|---|---|---|
{table}

**基準値（`ref`）は目標値ではない。**`norm(x, ref) = ref / (ref + x)` の分母に入る値で、
**実測がちょうど基準値のときその項が 0.5 になる**という意味しか持たない。
0 に近いほど 1 へ、大きいほど 0 へ寄る。頭打ちが無いのでどちらの向きにも余地が残る。
値は 2026-08-28 の `0828_002049_track2_val72base` の **Base 12 行（摂動なし）の実測値**
である。**実行ごとに取り直すと実行間で点を比べられなくなるので、固定値にしてある。**

**`n_steps` が運営の式の `time` にあたる。**運営の `time` は かかった時間 ではなく
「エピソード内のシミュレーションステップ数」である（運営 Q&A No.23）。結果 JSON には
実時間 `avg_episode_time_sec` も入っているが、**描画のやり方（egl / osmesa）とマシンで
何倍も変わるので実行間で比べられない。**歩数なら打ち切りの {MAX_STEPS:.0f} で頭が揃う。

**採点側と違えている点が 3 つある。どれも手元の JSON の粒度が足りないためである。**

| # | 何を | なぜ |
|---|---|---|
| 1 | **1 rollout ごとではなくタスクごとに平均してから掛ける** | 手元の JSON はタスクごとの平均しか持たない。成功率が 0 でも 1 でもないタスクで、成功した回と落ちた回のなめらかさを区別できない |
| 2 | **`(1 - collision penalty)` を掛けていない** | 手元の `success` は既に「ゴール到達 かつ 無衝突」である（`pipeline/rollout.py` の `success = bool(done) and not collided`）。重ねて掛けると二重に引く |
| 3 | **1 回も成功しなかったタスクの `n_steps` に打ち切りの {MAX_STEPS:.0f} 歩を当てる** | 歩数は成功した回しか JSON に入らない。空欄を「速かった」ことにすると、全部落ちたタスクほど点が高くなる |

**重みは非公開である。**「特定の指標に偏らせないための重み付け」と説明されているので
**偏らせない側に倒して 5 等分**にしてある。`--weights n_steps=0.3,jerk=0.1` で振れる。
"""


def _smaller_is_better(value: float, ref: float) -> float:
    """小さいほど良い量を 0〜1 に写す。ref のとき 0.5。"""
    if value <= 0:
        return 1.0
    return ref / (ref + value)


def smooth_components(metrics: dict) -> dict[str, float]:
    """1 タスクぶんの、正規化した 5 指標を返す。成功は入らない（門なので）。"""
    return {
        "n_steps": _smaller_is_better(metrics.get("avg_steps_to_success", MAX_STEPS),
                                      REFERENCE["n_steps"]),
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
    parser.add_argument("--no-write", action="store_true",
                        help="実行ディレクトリへ score_composite.md を書かない")
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
        report: list[str] = []

        def emit(line: str = "") -> None:
            """標準出力と、書き出すファイルの両方へ 1 行送る。"""
            print(line)
            report.append(line)

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

            emit(f"== {run_dir.name}  [{track['track']}]  {n} タスク")
            emit(f"   配布キットの総合スコア（成功率そのもの） {track['overall_score']:.4f}")
            emit(f"   **合計点 {total:.4f}**"
                  f"   ＝ 成功率 {mean_success:.3f} × なめらかさ {mean_smooth:.3f} 相当")
            emit(f"   全部成功したらいくつか（目盛りの確認・予選の実測は 0.48）: {mean_smooth:.4f}")
            emit(f"   衝突率 {mean_collision:.3f}")
            comp_mean = {k: sum(r[1][k] for r in rows) / n for k in weights}
            emit("   なめらかさの内訳（0〜1 に写したあとの値）  "
                 + "  ".join(f"{k} {comp_mean[k]:.3f}" for k in weights))

            if args.by_category and labels:
                groups: dict[str, list] = {}
                for name, comp, smooth, success, score, coll in rows:
                    groups.setdefault(labels.get(name, "?"), []).append(
                        (comp, smooth, success, score, coll))
                emit(f"   {'条件':16s}{'合計点':>8s}{'成功':>8s}{'なめらかさ':>10s}{'衝突率':>8s}")
                for label in sorted(groups):
                    g = groups[label]
                    m = lambda i: sum(x[i] for x in g) / len(g)  # noqa: E731
                    emit(f"   {label:16s}{m(3):8.3f}{m(2):8.3f}{m(1):10.3f}{m(4):8.3f}")

            if args.per_task:
                for name, _, smooth, success, score, coll in sorted(rows, key=lambda r: r[4]):
                    emit(f"   {score:6.3f}  成功 {success:4.2f}  なめらかさ {smooth:5.3f}"
                          f"  衝突 {coll:4.2f}  {name[:56]}")
            emit()

        if args.no_write or not target.is_dir():
            continue
        report_path = target / "score_composite.md"
        report_path.write_text(
            f"# 合計点 —— {run_dir.name}\n\n"
            "**この点は配布キットが出すものではない。**`score_composite.py` が\n"
            "結果 JSON から付け直したものである。式は下にそのまま書いてある。\n\n"
            "```\n" + "\n".join(report).rstrip() + "\n```\n\n"
            + formula_text(weights)
        )
        print(f"-- 書き出した: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
