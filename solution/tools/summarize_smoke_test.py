#!/usr/bin/env python3
"""疎通確認の結果が「最後まで走った」と言えるかを判定する。

**成功率は見ない。** ランダム方策で回すので成功率は 0 が正常である。
見るのは「タスクを全部読み込めて、結果が出そろい、JSON が集計値を持つか」だけ。

終了コードが 0 でも失敗していることがある。トラックが例外で落ちると
`pipeline.py` の `_errored_track_score` が `{"error": 1.0}` を書いて
**正常終了する**ためである（2026-08-10 に 3 タスクで止まったのに気づけなかった）。

    solution/tools/summarize_smoke_test.py --output-dir work/smoke --expected-tasks 5

Reads:
  - <output-dir>/eval.log
  - <output-dir>/server_8000.json
Generates:
  - 標準出力のみ。判定が偽なら終了コード 1
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

_RESULT_RE = re.compile(r"\[結果\] (\S+) ep(\d+): (\S+) (\d+) steps")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="smoke_test_suite.sh の出力先")
    parser.add_argument("--expected-tasks", type=int, required=True,
                        help="回したタスク数。これだけ結果が出ていれば通り")
    return parser.parse_args()


def _check_log(log_path: Path, expected: int) -> tuple[bool, list[str]]:
    """評価ログを見て、結果が出そろっているかを返す。"""
    notes: list[str] = []
    if not log_path.is_file():
        return False, [f"評価ログが無い: {log_path}"]

    text = log_path.read_text(encoding="utf-8", errors="replace")
    results = _RESULT_RE.findall(text)
    notes.append(f"[結果] 行: {len(results)} / {expected}")

    if results:
        outcomes = collections.Counter(outcome for _, _, outcome, _ in results)
        notes.append("  内訳: " + " ".join(f"{k} {v}" for k, v in outcomes.most_common()))

    # 例外はトラックごとに握られるため、ログの中にだけ残る。
    errors = [
        line for line in text.splitlines()
        if "Traceback" in line or "FileNotFoundError" in line
        or "が失敗" in line or "失敗しました" in line
    ]
    if errors:
        notes.append(f"  **ログ中の例外らしい行: {len(errors)} 件**")
        notes.extend(f"    {line[:150]}" for line in errors[:5])

    return len(results) >= expected and not errors, notes


def _find_result_json(output_dir: Path) -> Path | None:
    """結果 JSON を探す。

    名前は `--server-url` のポートから決まる（`pipeline/cli.py` の
    `submission_id = f"server_{port}"`）ため、ポートを変えると名前も変わる。
    **決め打ちにすると「JSON が無い」と誤判定する。**
    """
    candidates = sorted(
        (path for path in output_dir.glob("*.json")),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _check_json(json_path: Path | None, expected: int) -> tuple[bool, list[str]]:
    """結果 JSON が集計値を持っているかを返す。"""
    if json_path is None or not json_path.is_file():
        return False, ["結果 JSON が無い（評価が JSON を書く前に落ちた）"]

    track = json.loads(json_path.read_text())["tracks"][0]
    overall = track["overall_metrics"]

    if "n_total_episodes" not in overall:
        return False, [
            "**JSON がエラー記録になっている（例外で落ちた）**",
            f"  overall_metrics = {overall}",
        ]

    notes = [
        f"JSON = {json_path.name}",
        f"n_tasks = {overall['n_tasks']} / {expected}",
        f"n_total_episodes = {overall['n_total_episodes']}",
        f"mean_success_rate = {overall['mean_success_rate']:.3f}"
        "  （ランダム方策なので 0 が正常）",
    ]
    return int(overall["n_tasks"]) >= expected, notes


def main() -> None:
    args = _parse_args()
    log_ok, log_notes = _check_log(args.output_dir / "eval.log", args.expected_tasks)
    json_ok, json_notes = _check_json(
        _find_result_json(args.output_dir), args.expected_tasks
    )

    for note in log_notes + json_notes:
        print(note)

    print()
    if log_ok and json_ok:
        print("**通った。** タスクを全部読み込み、結果が出そろい、JSON も集計値を持つ。")
        print("→ この構成なら 84 タスクを最後まで回せる。")
        raise SystemExit(0)

    print("**通っていない。** 上の行が理由である。")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
