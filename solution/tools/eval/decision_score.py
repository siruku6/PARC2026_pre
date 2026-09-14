#!/usr/bin/env python3
"""最終フォームに入れる重みを、手元評価の合計点から 1 本に決める。

**Track2 の判定セットと LIBERO90 の半分を、決めておいた比率で足して並べる。**
比率は「最終評価に libero_90 系が混ざる割合」で、配布 `libero_combined_20hz` の中の
libero_90 の本数の割合（3,921 / 19,533 ＝ 20%）を使う（ユーザ 9/15）。

    重み付きの点 = 0.8 × 判定セットの合計点 + 0.2 × LIBERO90 半分の合計点

**差が `--tie`（既定 0.02）以内の候補は差なしとみなし、先に並べた候補を選ぶ。**
だから `--candidate` は「変更の少ない順」（いま入っているもの → 新しいもの）に並べる。

**合計点の出し方は `score_composite.py` と同じ**（タスクごとに 成功率 × なめらかさ を平均する）。
**違いは、CSV に載っている行だけで数えること。**比べる相手の古い実行が
もっと多くの行を回していても、同じ行だけで比べられる。

    python3 solution/tools/eval/decision_score.py \\
        --candidate bowl5_full \\
            work/final_eval/<判定セットの実行>,work/final_eval/<別の実行> \\
            work/final_eval/<LIBERO90 の実行> \\
        --candidate C005500 work/final_eval/<...> work/final_eval/<...>

**1 つの候補の結果が複数の実行に分かれているときは、カンマでつなぐ。**
同じ行が 2 つの実行にあれば、前に書いた実行の値を使う。

Reads:
  - <実行ディレクトリ>/results/server_8000.json
  - compe/val/TRACK2_NOISE84.csv（既定の判定セット）
  - compe/val/LIBERO90_HALF45.csv（既定の LIBERO90 半分）
Generates:
  - 標準出力のみ
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_composite import DEFAULT_WEIGHTS, smooth_components, smooth_metrics  # noqa: E402

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _SetScore:
    """1 つの評価セットについて、1 つの候補が取った点。"""

    total: float
    success: float
    collision: float
    num_tasks: int


@dataclass
class _CandidateScore:
    """1 つの候補の、2 つの評価セットの点と重み付きの点。"""

    name: str
    judge: _SetScore
    libero: _SetScore
    weighted: float


def _parse_args() -> argparse.Namespace:
    """候補の並びと、判定に使う CSV・比率を受け取る。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--candidate", nargs=3, action="append", required=True,
        metavar=("NAME", "JUDGE_RUNS", "LIBERO_RUNS"),
        help="候補の名前・判定セットの実行ディレクトリ・LIBERO90 の実行ディレクトリ"
             "（複数の実行はカンマでつなぐ）。変更の少ない順に並べる")
    parser.add_argument("--judge-csv", type=Path,
                        default=_REPOSITORY_ROOT / "compe/val/TRACK2_NOISE84.csv")
    parser.add_argument("--libero-csv", type=Path,
                        default=_REPOSITORY_ROOT / "compe/val/LIBERO90_HALF45.csv")
    parser.add_argument("--libero-weight", type=float, default=0.2,
                        help="LIBERO90 側の比率（既定 0.2）。判定セット側は 1 から引いた値")
    parser.add_argument("--tie", type=float, default=0.02,
                        help="この差以内は差なしとみなす（既定 0.02）")
    return parser.parse_args()


def _read_task_ids(csv_path: Path) -> list[str]:
    """評価セットの CSV から、数える行の task_id を順に読む。"""
    with csv_path.open(newline="") as handle:
        return [row["task_id"] for row in csv.DictReader(handle)]


def _load_task_results(run_arg: str) -> dict[str, dict]:
    """カンマでつないだ実行ディレクトリから、タスク名 → 結果 を集める（前の実行が優先）。"""
    task_results: dict[str, dict] = {}
    for run_text in run_arg.split(","):
        run_dir = Path(run_text)
        if not run_dir.is_absolute():
            run_dir = _REPOSITORY_ROOT / run_dir
        json_path = run_dir / "results" / "server_8000.json"
        if not json_path.is_file():
            raise SystemExit(f"結果が無い: {json_path}（評価が終わっていないか、ディレクトリ名の誤り）")
        data = json.loads(json_path.read_text())
        for track in data["tracks"]:
            for task in track["tasks"]:
                task_results.setdefault(task["task_name"], task)
    return task_results


def _score_set(label: str, task_ids: list[str], task_results: dict[str, dict]) -> _SetScore:
    """CSV に載っている行だけで、`score_composite.py` と同じ合計点を出す。

    Parameters
    ----------
    label : str
        足りない行があったときの表示に使う名前（候補名と評価セット名）。
    task_ids : list[str]
        数える行の task_id。
    task_results : dict[str, dict]
        タスク名 → 結果 JSON のタスク 1 件。

    Returns
    -------
    _SetScore
        合計点・成功率・衝突率・数えた行数。
    """
    missing: list[str] = [task_id for task_id in task_ids if task_id not in task_results]
    if missing:
        raise SystemExit(f"{label}: {len(missing)} 行の結果が無い（例 {missing[0]}）。"
                         "評価が途中で止まったか、別の CSV で回した実行を渡している")
    totals: list[float] = []
    successes: list[float] = []
    collisions: list[float] = []
    for task_id in task_ids:
        task = task_results[task_id]
        smooth = smooth_metrics(smooth_components(task["metrics"]), DEFAULT_WEIGHTS)
        totals.append(task["success_rate"] * smooth)
        successes.append(task["success_rate"])
        collisions.append(task["metrics"].get("collision_rate", 0.0))
    count = len(task_ids)
    return _SetScore(sum(totals) / count, sum(successes) / count,
                     sum(collisions) / count, count)


def _score_candidates(args: argparse.Namespace) -> list[_CandidateScore]:
    """全候補について、2 つの評価セットの点と重み付きの点を出す（並べた順のまま）。"""
    judge_ids: list[str] = _read_task_ids(args.judge_csv)
    libero_ids: list[str] = _read_task_ids(args.libero_csv)
    scores: list[_CandidateScore] = []
    for name, judge_runs, libero_runs in args.candidate:
        judge = _score_set(f"{name} の判定セット", judge_ids, _load_task_results(judge_runs))
        libero = _score_set(f"{name} の LIBERO90", libero_ids, _load_task_results(libero_runs))
        weighted = (1.0 - args.libero_weight) * judge.total + args.libero_weight * libero.total
        scores.append(_CandidateScore(name, judge, libero, weighted))
    return scores


def _choose(scores: list[_CandidateScore], tie: float) -> _CandidateScore:
    """最高点から `tie` 以内の候補のうち、先に並べたものを選ぶ。"""
    best: float = max(score.weighted for score in scores)
    return next(score for score in scores if best - score.weighted <= tie)


def _print_table(scores: list[_CandidateScore], chosen: _CandidateScore, args: argparse.Namespace) -> None:
    """候補ごとの点を表にして、選んだ候補と理由を印字する。"""
    best: float = max(score.weighted for score in scores)
    print(f"判定セット {args.judge_csv.name}（{scores[0].judge.num_tasks} 行）× {1 - args.libero_weight:.1f}"
          f" ＋ {args.libero_csv.name}（{scores[0].libero.num_tasks} 行）× {args.libero_weight:.1f}"
          f"  ／ 差なしの幅 {args.tie}")
    print()
    print(f"  {'候補':<24}{'判定セット':>10}{'成功':>7}{'衝突':>7}"
          f"{'LIBERO90':>10}{'成功':>7}{'衝突':>7}{'重み付き':>10}{'最高との差':>11}")
    for score in scores:
        mark = "← 選ぶ" if score is chosen else ("差なし" if best - score.weighted <= args.tie else "")
        print(f"  {score.name:<24}{score.judge.total:10.4f}{score.judge.success:7.3f}"
              f"{score.judge.collision:7.3f}{score.libero.total:10.4f}{score.libero.success:7.3f}"
              f"{score.libero.collision:7.3f}{score.weighted:10.4f}{score.weighted - best:+11.4f}  {mark}")
    print()
    print(f"選ぶ: {chosen.name}（重み付き {chosen.weighted:.4f}）")


def main() -> int:
    args = _parse_args()
    scores = _score_candidates(args)
    chosen = _choose(scores, args.tie)
    _print_table(scores, chosen, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
