#!/usr/bin/env python3
"""GPU マシンで流した検証用スイートの結果を、判断に使える形へ集計する。

**Colab の F セルがやっていたことを、マシン上で完結させる。**
Drive へ退避する必要がないので、退避漏れで結果を失う経路そのものが無い。

出すのは 5 つ。**どれも「次に何をするか」が変わる情報である。**

- **84 件そろったか** — 途中で落ちると終了コード 0 のまま件数だけ減る
- 結末の内訳（成功 / 衝突 / 未完了）と、**ゴール到達後に衝突で落ちた本数**
- **摂動カテゴリ別の成功率** — 学習の的を決めるのはこの表
- ぶつけた物体の偏り
- **同じ変位が繰り返し出ていないか** — 環境側で物が動いている印

    solution/tools/summarize_gpu_eval.py --output-dir work/gpu_eval/0811_...

Reads:
  - <output-dir>/eval.log
  - <output-dir>/server_8000.json
  - <output-dir>/../../compe/val/VAL_TASKS.csv（あればカテゴリを引く）
Generates:
  - <output-dir>/summary.txt（画面と同じ内容）
  - 標準出力。**結果が 1 件も無ければ終了コード 1**
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
from pathlib import Path

# `[結果] <タスク> ep<N>: <結末> <歩数> steps  ぶつけた物体: a=1.0mm, b=2.0mm
_RESULT_RE = re.compile(
    r"\[結果\] (?P<task>\S+) ep(?P<episode>\d+): "
    r"(?P<outcome>\S+) (?P<steps>\d+) steps(?P<detail>.*)"
)
_OFFENDER_RE = re.compile(r"(\S+?)=([\d.]+)mm")
_EXCEPTION_RE = re.compile(r"\[結果\] (\S+): 例外で測れず \((.+?)\)")
_TASK_TOTAL_RE = re.compile(r"=== タスク \d+/(\d+):")

# 摂動カテゴリを変種名から当てる。CSV が読めればそちらを使う。
_CATEGORY_MARKERS: tuple[tuple[str, str], ...] = (
    ("_language", "Language Instructions"),
    ("_add_", "Objects Layout"),
    ("level", "Objects Layout"),
    ("_noise_", "Sensor Noise"),
    ("_view_", "Camera Viewpoints"),
    ("_light_", "Light Conditions"),
    ("_table_", "Background Textures"),
    ("_tb_", "Background Textures"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="run_gpu_eval.sh の出力先")
    parser.add_argument("--csv", type=Path, default=None,
                        help="検証用スイートの CSV。カテゴリを引くのに使う")
    return parser.parse_args()


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """成功率の 95% 信頼区間（Wilson）を返す。

    **単純な ±1.96√(p(1-p)/n) は端で壊れる。** 成功率 0 や 1 のときに
    幅 0 の区間を返すため、「1 本も失敗していないから確実」と読めてしまう。

    Parameters
    ----------
    successes : int
        成功した本数。
    total : int
        全体の本数。

    Returns
    -------
    low, high : float
        下限と上限。`total` が 0 なら (0.0, 0.0)。
    """
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _load_categories(csv_path: Path | None) -> dict[str, str]:
    """CSV からタスク名 → 摂動カテゴリの対応を読む。無ければ空。"""
    if csv_path is None or not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return {
            row["task_id"]: row.get("category", "")
            for row in csv.DictReader(stream)
            if row.get("category")
        }


def _category_of(task_name: str, known: dict[str, str]) -> str:
    """摂動カテゴリを決める。CSV を優先し、無ければ変種名から当てる。"""
    if task_name in known:
        return known[task_name]
    for marker, category in _CATEGORY_MARKERS:
        if marker in task_name:
            return category
    return "その他"


def _parse_log(log_path: Path) -> tuple[list[dict], list[tuple[str, str]], int | None]:
    """評価ログから 1 エピソードずつの結果を取り出す。

    Returns
    -------
    episodes : list[dict]
        `task` / `episode` / `outcome` / `steps` / `offenders` を持つ辞書の列。
    exceptions : list[tuple[str, str]]
        (タスク名, 例外の内容)。パッチが例外を握った回。
    declared_total : int | None
        ログが宣言していたタスク総数（`=== タスク 1/84:` の 84）。
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")

    episodes: list[dict] = []
    for found in _RESULT_RE.finditer(text):
        offenders = {
            name: float(millimetres)
            for name, millimetres in _OFFENDER_RE.findall(found.group("detail"))
        }
        episodes.append({
            "task": found.group("task"),
            "episode": int(found.group("episode")),
            "outcome": found.group("outcome"),
            "steps": int(found.group("steps")),
            "offenders": offenders,
        })

    exceptions = _EXCEPTION_RE.findall(text)
    totals = _TASK_TOTAL_RE.findall(text)
    return episodes, exceptions, int(totals[0]) if totals else None


def _report_coverage(lines: list[str], episodes: list[dict],
                     exceptions: list[tuple[str, str]],
                     declared_total: int | None) -> None:
    """**まず件数を見る。** 落ちていても終了コードは 0 になる。"""
    tasks_seen = len({episode["task"] for episode in episodes})
    lines.append("=== 件数 ===")
    if declared_total is None:
        lines.append(f"  結果 {len(episodes)} 件（タスク {tasks_seen} 種）")
        lines.append("  **ログにタスク総数が無い。** 途中で落ちた可能性がある")
        return

    lines.append(f"  結果 {len(episodes)} 件 / タスク {tasks_seen} / 宣言 {declared_total}")
    if tasks_seen < declared_total:
        missing = declared_total - tasks_seen
        lines.append(f"  **{missing} タスクが測れていない。** 内訳を下で見ること")
    else:
        lines.append("  **全部そろった。**")

    if exceptions:
        lines.append(f"  例外で測れなかった回 {len(exceptions)} 件:")
        for task_name, reason in exceptions[:5]:
            lines.append(f"    {task_name[:60]}  ({reason})")


def _report_outcomes(lines: list[str], episodes: list[dict]) -> None:
    """結末の内訳。**ゴール到達後の衝突を分けて出す。**"""
    counts = collections.Counter(episode["outcome"] for episode in episodes)
    total = len(episodes)
    successes = counts.get("成功", 0)
    low, high = _wilson_interval(successes, total)

    lines.append("")
    lines.append("=== 結末 ===")
    lines.append(f"  成功率 {successes / total:.3f}  95%CI [{low:.3f}, {high:.3f}]  n={total}")
    for outcome in ("成功", "衝突", "未完了"):
        count = counts.get(outcome, 0)
        lines.append(f"  {outcome:<4} {count:>4}  ({count / total:>5.1%})")

    # **打ち切りに張り付いた回と、ゴールに着いてから落ちた回を分ける。**
    # 前者は学習の話、後者は衝突対策の話になる。
    capped = [e for e in episodes if e["outcome"] != "成功" and not e["offenders"]]
    collided_after_goal = [
        e for e in episodes if e["outcome"] == "衝突" and e["steps"] < 300
    ]
    lines.append(f"  うちゴール到達後に衝突で失格  {len(collided_after_goal)} 件")
    lines.append(f"  うち打ち切りまで走って未完了  {len(capped)} 件")

    successful_steps = sorted(e["steps"] for e in episodes if e["outcome"] == "成功")
    if successful_steps:
        middle = successful_steps[len(successful_steps) // 2]
        lines.append(
            f"  成功した回の歩数  最小 {successful_steps[0]} / "
            f"中央 {middle} / 最大 {successful_steps[-1]}"
        )


def _report_categories(lines: list[str], episodes: list[dict],
                       known: dict[str, str]) -> None:
    """摂動カテゴリ別。**学習の的を決めるのはこの表である。**"""
    by_category: dict[str, list[dict]] = collections.defaultdict(list)
    for episode in episodes:
        by_category[_category_of(episode["task"], known)].append(episode)

    lines.append("")
    lines.append("=== 摂動カテゴリ別（弱い順）===")
    rows = []
    for category, group in by_category.items():
        successes = sum(1 for e in group if e["outcome"] == "成功")
        rows.append((successes / len(group), category, successes, len(group), group))
    for rate, category, successes, total, group in sorted(rows):
        breakdown = collections.Counter(e["outcome"] for e in group)
        detail = " ".join(f"{k} {v}" for k, v in breakdown.most_common())
        lines.append(f"  {category:<24} {rate:.2f}  ({total:>3} 本)  {detail}")


def _report_offenders(lines: list[str], episodes: list[dict]) -> None:
    """ぶつけた物体の偏りと、**環境側で動いている疑い**を出す。"""
    counts: collections.Counter[str] = collections.Counter()
    displacements: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for episode in episodes:
        for name, millimetres in episode["offenders"].items():
            counts[name] += 1
            displacements[name][f"{millimetres:.1f}mm"] += 1

    if not counts:
        lines.append("")
        lines.append("=== 衝突 ===")
        lines.append("  1 件も無い")
        return

    lines.append("")
    lines.append("=== ぶつけた物体 ===")
    for name, count in counts.most_common(10):
        lines.append(f"  {count:>4}  {name}")

    # **同じ変位が何度も出るのは方策の振る舞いではない。** 摂動が違う回で
    # 変位が 1 mm も違わないなら、環境側で最初から動いていると考えるべきである
    # （8/10 に `ketchup_1=544.6mm` が 6 回そろって出た）。
    suspicious = [
        (name, value, times)
        for name, values in displacements.items()
        for value, times in values.items()
        if times >= 3
    ]
    if suspicious:
        lines.append("")
        lines.append("  **同じ変位が繰り返し出ている。環境側で動いている疑いがある：**")
        for name, value, times in sorted(suspicious, key=lambda row: -row[2]):
            lines.append(f"    {name}={value} が {times} 回")
        lines.append("  → この物体を含むタスクは、方策の性能を表していない可能性がある")


def _report_json(lines: list[str], output_dir: Path) -> None:
    """パイプラインの JSON から指標を読む。**例外の跡もここに出る。**"""
    candidates = sorted(output_dir.glob("*.json"))
    if not candidates:
        lines.append("")
        lines.append("=== JSON ===")
        lines.append("  無し")
        return

    payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    lines.append("")
    lines.append(f"=== 指標（{candidates[-1].name}）===")
    tracks = payload.get("tracks") or []
    if not tracks:
        lines.append("  トラックが 1 つも無い")
        return
    metrics = tracks[0].get("overall_metrics", {})
    if metrics.get("error"):
        lines.append("  **例外で終わっている。** JSON はエラー記録だけを持つ")
        lines.append(f"  {metrics}")
        return
    for key in (
        "mean_success_rate", "mean_collision_rate", "mean_avg_cartesian_jerk",
        "mean_rms_cartesian_jerk", "mean_sparc", "mean_cartesian_path_length",
        "n_tasks", "n_total_episodes",
    ):
        if key in metrics:
            lines.append(f"  {key:<32} {metrics[key]}")


def _report_videos(lines: list[str], output_dir: Path) -> None:
    """撮れた mp4 の本数を出す。**中身の確認は別の道具でやる。**"""
    videos = sorted((output_dir / "videos").glob("*.mp4"))
    lines.append("")
    lines.append("=== 録画 ===")
    if not videos:
        lines.append("  無し（RECORD= で無効にしたか、失敗した回が無かった）")
        return
    lines.append(f"  {len(videos)} 本")
    lines.append("  見るには:")
    lines.append(f"    python3 solution/tools/summarize_rollout_videos.py {output_dir / 'videos'}")


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output_dir.resolve()
    log_path = output_dir / "eval.log"

    if not log_path.is_file():
        raise SystemExit(f"評価ログが無い: {log_path}")

    episodes, exceptions, declared_total = _parse_log(log_path)
    if not episodes:
        print(f"**結果が 1 件も無い。** ログの末尾を見ること: {log_path}")
        raise SystemExit(1)

    csv_path = args.csv
    if csv_path is None:
        # run_gpu_eval.sh が置く場所を既定で探す。
        guess = output_dir.parents[2] / "eval-kit" / "compe" / "val" / "VAL_TASKS.csv"
        csv_path = guess if guess.is_file() else None

    lines: list[str] = [f"評価ログ  {log_path}", ""]
    _report_coverage(lines, episodes, exceptions, declared_total)
    _report_outcomes(lines, episodes)
    _report_categories(lines, episodes, _load_categories(csv_path))
    _report_offenders(lines, episodes)
    _report_json(lines, output_dir)
    _report_videos(lines, output_dir)

    report = "\n".join(lines)
    print(report)
    (output_dir / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\n（同じ内容を {output_dir / 'summary.txt'} に書いた）")


if __name__ == "__main__":
    main()
