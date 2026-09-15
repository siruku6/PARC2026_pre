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

**比べられない実行を渡すと止まる**（9/15 に足した）。候補どうしで次のどれかが違えば、表を出さずに終わる。

| 揃えるもの | なぜ |
|---|---|
| **本数**（`n_eval_episodes`） | 1 行の中で episode 番号ごとに初期状態が決まっていて（`init_states[ep_id]`）、番号で難しさが違う。9/15 に C の判定セットは 0・1・2 番で 65・55・46/84 だった |
| **行の一覧**（実行ディレクトリの `TASKS_CSV.csv` が、渡した CSV と同じ並び） | 乱数はサーバー内部のリセット回数で決まる（`docs/plan/23` の 7 節）。行が違えば同じ行でも別の乱数になる |
| **描き方**（`eval_conditions.json` の `mujoco_gl`） | EGL と osmesa で画素が変わる。記録の無い実行（9/15 17 時より前）は描き方が分からないので比べない |
| **1 つの候補は 1 つの実行** | カンマでつないだ実行どうしは乱数の並びが違う |

探りのために揃っていない実行を並べたいときだけ `--allow-mixed` を付ける（警告を出して表を出す）。
**`--allow-mixed` を付けた表で重みを選ばない。**

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
    parser.add_argument("--allow-mixed", action="store_true",
                        help="本数・行の一覧・描き方が揃っていなくても表を出す（探り用。これで選ばない）")
    return parser.parse_args()


def _read_task_ids(csv_path: Path) -> list[str]:
    """評価セットの CSV から、数える行の task_id を順に読む。"""
    with csv_path.open(newline="") as handle:
        return [row["task_id"] for row in csv.DictReader(handle)]


def _load_task_results(run_arg: str) -> tuple[dict[str, dict], list[dict]]:
    """カンマでつないだ実行ディレクトリから、タスク名 → 結果 と、実行ごとの条件を集める（前の実行が優先）。"""
    task_results: dict[str, dict] = {}
    conditions: list[dict] = []
    for run_text in run_arg.split(","):
        run_dir = Path(run_text)
        if not run_dir.is_absolute():
            run_dir = _REPOSITORY_ROOT / run_dir
        json_path = run_dir / "results" / "server_8000.json"
        if not json_path.is_file():
            raise SystemExit(f"結果が無い: {json_path}（評価が終わっていないか、ディレクトリ名の誤り）")
        data = json.loads(json_path.read_text())
        conditions.append(_read_conditions(run_dir, data))
        for track in data["tracks"]:
            for task in track["tasks"]:
                task_results.setdefault(task["task_name"], task)
    return task_results, conditions


def _read_conditions(run_dir: Path, data: dict) -> dict:
    """1 つの実行の、比べるために揃えるべき条件（本数・描き方・行の一覧）を読む。"""
    recorded = run_dir / "eval_conditions.json"
    renderer = json.loads(recorded.read_text()).get("mujoco_gl") if recorded.is_file() else None
    tasks_csv = run_dir / "TASKS_CSV.csv"
    task_ids = _read_task_ids(tasks_csv) if tasks_csv.is_file() else None
    return {"run": run_dir.name, "episodes": data["metadata"]["eval_config"]["n_eval_episodes"],
            "renderer": renderer, "task_ids": task_ids}


def _check_comparable(set_label: str, csv_ids: list[str],
                      per_candidate: list[tuple[str, list[dict]]], allow_mixed: bool) -> str:
    """候補どうしで本数・行の一覧・描き方が揃っているかを見る。揃っていなければ止める。

    Returns
    -------
    str
        揃っていたときの条件の短い説明（表の見出しに使う）。
    """
    problems: list[str] = []
    for name, conditions in per_candidate:
        if len(conditions) != 1:
            problems.append(f"{name}: 実行が {len(conditions)} 個に分かれている（実行ごとに乱数の並びが違う）")
        for cond in conditions:
            if cond["renderer"] is None:
                problems.append(f"{name}: {cond['run']} に描き方の記録（eval_conditions.json）が無い")
            if cond["task_ids"] != csv_ids:
                count = "無し" if cond["task_ids"] is None else f"{len(cond['task_ids'])} 行"
                problems.append(f"{name}: {cond['run']} の行の一覧（{count}）が {len(csv_ids)} 行の CSV と同じ並びでない")
    episodes = {f"{name}={cond['episodes']}" for name, conds in per_candidate for cond in conds}
    if len({cond["episodes"] for _, conds in per_candidate for cond in conds}) > 1:
        problems.append("本数が揃っていない: " + "・".join(sorted(episodes)))
    renderers = {cond["renderer"] for _, conds in per_candidate for cond in conds}
    if len(renderers - {None}) > 1:
        problems.append("描き方が揃っていない: " + "・".join(
            sorted(f"{name}={cond['renderer']}" for name, conds in per_candidate for cond in conds)))
    if problems:
        text = f"{set_label} は比べられない:\n" + "\n".join(f"  - {p}" for p in problems)
        if not allow_mixed:
            raise SystemExit(text + "\n同じ CSV・同じ本数・同じ描き方で回した実行を 1 つずつ渡す"
                                    "（docs/howto/03_gpu_eval.md の「比べる評価の揃え方」）")
        print("**警告（--allow-mixed）: この表で重みを選ばないこと**\n" + text + "\n")
        return "揃っていない"
    first = per_candidate[0][1][0]
    return f"{first['episodes']} 本・{first['renderer']}"


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
    loaded = [(name, _load_task_results(judge_runs), _load_task_results(libero_runs))
              for name, judge_runs, libero_runs in args.candidate]
    args.judge_condition = _check_comparable(
        "判定セット", judge_ids, [(name, judge[1]) for name, judge, _ in loaded], args.allow_mixed)
    args.libero_condition = _check_comparable(
        "LIBERO90", libero_ids, [(name, libero[1]) for name, _, libero in loaded], args.allow_mixed)
    scores: list[_CandidateScore] = []
    for name, (judge_results, _), (libero_results, _) in loaded:
        judge = _score_set(f"{name} の判定セット", judge_ids, judge_results)
        libero = _score_set(f"{name} の LIBERO90", libero_ids, libero_results)
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
    print(f"判定セット {args.judge_csv.name}（{scores[0].judge.num_tasks} 行・{args.judge_condition}）"
          f"× {1 - args.libero_weight:.1f}"
          f" ＋ {args.libero_csv.name}（{scores[0].libero.num_tasks} 行・{args.libero_condition}）"
          f"× {args.libero_weight:.1f}  ／ 差なしの幅 {args.tie}")
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
