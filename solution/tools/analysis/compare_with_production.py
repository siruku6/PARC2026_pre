#!/usr/bin/env python3
"""手元の検証が採点環境の代わりに使えるかを確かめる。

**採点環境は 8 エピソードしか実行しない。** 手元の 84 件の検証はそれとは別の
母集団で、実際に順序が逆になったことがある（手元では追加学習した重みが
素より上に見えたが、採点環境では 8.8 倍悪かった）。

**そこで、答えが分かっている提出を手元でも実行し、順序が一致するかを見る。**
一致すれば、その検証集合を提出前の判断に使える。一致しなければ使えない。

    # 1 組だけ見る（エピソードごとのステップ数を並べる）
    python3 solution/tools/analysis/compare_with_production.py \\
        --pair outputs/0812_base_t1=outputs/prod_eval_results/260810_評価結果_ema=0.3あり.md

    # 2 組以上で順序が一致するかを見る（**こちらが本題**）
    python3 solution/tools/analysis/compare_with_production.py \\
        --pair outputs/0812_base_t1=outputs/prod_eval_results/260810_評価結果_ema=0.3あり.md \\
        --pair outputs/0812_lora_t1=outputs/prod_eval_results/260811_評価結果_ema=0.3あり_初回学習後.md

採点ログから読むのは `POST /reset` と `POST /act` の回数である。
**評価パイプラインは 1 エピソードの開始ごとに `/reset` を 1 回投げ、
1 ステップごとに `/act` を 1 回投げる。** そのため

- `/reset` の回数 = エピソード本数
- ある `/reset` から次の `/reset` までの `/act` の回数 = そのエピソードのステップ数

Reads:
  - <手元の出力>/eval.log（`[結果]` 行。解析は summarize_gpu_eval と共通）
  - <手元の出力>/server_*.json（`overall_score` と評価条件）
  - <採点ログ>（Omnicampus から持ち帰った本文。貼り付けたままでよい）
Generates:
  - 標準出力
  - **順序が食い違ったら終了コード 1**
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

# `[結果]` 行の解析を共通にする。**同じ正規表現を 2 か所に置くと必ず片方が
# 古くなる。** summarize_gpu_eval は import しても main() を実行しない。
#
# **`summarize_gpu_eval` は `solution/tools/eval/` へ移った**（2026-08-23）。
# このファイルは `solution/tools/analysis/` にあるので、1 つ上（`tools/`）へ戻って
# `eval/` へ降りる。**`parents[1]` だけでは `tools/` を指すので import できない。**
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from summarize_gpu_eval import _pad, _parse_log  # noqa: E402

# 採点ログから読む値。
_RESET_RE = re.compile(r'"POST /reset')
_ACT_RE = re.compile(r'"POST /act')
_SCORE_RE = re.compile(r"総合スコア ([\d.]+)")
_ELAPSED_RE = re.compile(r"合計時間: ([\d.]+)秒")
_VERSION_RE = re.compile(r"\[parc\] policy_server (\S+)")

# 判定に使う許容差。**ステップ数がこれを超えて食い違ったら並べて示す。**
_STEP_TOLERANCE = 10

# **本番スコアがこの差の中なら同順として扱う。**
# 採点環境は 8 エピソードで、うち 1〜4 本目は 3 種類の重みすべてが 300 ステップに
# 張り付いた。**差が付くのは残り 4 本だけである。**
# 実際、素（0.1409）とλ=0.25（0.095）の違いは 1 本の結末が変わった分で、
# 差は 0.046 しかない。**この 2 つの順序を再現できたことを根拠に代理集合を
# 合格にすると、偶然を実力と読み違える。**
# 0.05 なら上の 2 つは同順になり、崩壊（0.016）とは別の段に分かれる
# （0.095 − 0.016 = 0.079）。
_SCORE_TIE = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pair", action="append", required=True, metavar="LOCAL_DIR=PROD_LOG",
        help="手元の出力ディレクトリと採点ログの組。**複数指定できる**",
    )
    parser.add_argument(
        "--tolerance", type=int, default=_STEP_TOLERANCE,
        help=f"ステップ数の差をどこから食い違いとみなすか（既定 {_STEP_TOLERANCE}）",
    )
    parser.add_argument(
        "--score-tie", type=float, default=_SCORE_TIE,
        help=f"本番スコアがこの差の中なら同順として扱う（既定 {_SCORE_TIE}）。"
             "8 本では 1 本の結末で 0.046 動くため、それ以下の差は判定に使わない",
    )
    return parser.parse_args()


def _read_production(log_path: Path) -> dict:
    """採点ログから、エピソードごとのステップ数とスコアを取り出す。

    `/reset` より前の `/act` は数えない。**評価が始まる前の疎通確認を
    1 エピソード目に混ぜないため。**

    Parameters
    ----------
    log_path : Path
        採点環境から持ち帰ったログ。

    Returns
    -------
    result : dict
        `steps`（エピソードごとのステップ数）、`score`、`elapsed_sec`、`version`。
        読めなかった値は入れない。
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")

    steps: list[int] = []
    current: int | None = None
    for line in text.splitlines():
        if _RESET_RE.search(line):
            if current is not None:
                steps.append(current)
            current = 0
        elif _ACT_RE.search(line) and current is not None:
            current += 1
    if current is not None:
        steps.append(current)

    result: dict = {"name": log_path.name, "steps": steps}
    for key, pattern, cast in (
        ("score", _SCORE_RE, float),
        ("elapsed_sec", _ELAPSED_RE, float),
        ("version", _VERSION_RE, str),
    ):
        found = pattern.search(text)
        if found:
            result[key] = cast(found.group(1))
    return result


def _read_local(output_dir: Path) -> dict:
    """手元の評価結果から、エピソードごとのステップ数とスコアを取り出す。

    ステップ数は `eval.log` の `[結果]` 行から読む。**JSON にはエピソードごとの
    ステップ数が入っていない**（`metrics` は集計値だけである）。

    Parameters
    ----------
    output_dir : Path
        `eval.log` と `server_*.json` があるディレクトリ。

    Returns
    -------
    result : dict
        `steps`・`outcomes`・`tasks`・`exceptions`・`score`・`max_steps`。
    """
    log_path = output_dir / "eval.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"eval.log が無い: {log_path}")

    episodes, exceptions, _ = _parse_log(log_path)
    result: dict = {
        "name": output_dir.name,
        "steps": [record["steps"] for record in episodes],
        "outcomes": [record["outcome"] for record in episodes],
        "tasks": [record["task"] for record in episodes],
        "exceptions": len(exceptions),
    }

    # **新しい方から見る。** 同じディレクトリに古い JSON が残っていることがある。
    # 壊れた JSON や中身が空の回は飛ばす（`summarize_gpu_eval` と同じ扱い）。
    payload: dict = {}
    for candidate in sorted(output_dir.glob("server_*.json"), reverse=True):
        try:
            candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if candidate_payload.get("tracks"):
            payload = candidate_payload
            break
    if payload:
        tracks = payload["tracks"]
        if not tracks[0].get("overall_metrics", {}).get("error"):
            result["score"] = tracks[0].get("overall_score")
        config = payload.get("metadata", {}).get("eval_config", {})
        if "max_steps_per_episode" in config:
            result["max_steps"] = config["max_steps_per_episode"]
    return result


def _at_cap(steps: list[int], cap: int | None) -> int:
    """打ち切りに張り付いた本数を返す。

    **合成スコアより素直な信号である。** 採点環境の 3 提出では 4 / 5 / 6 本と
    単調に増えており、スコアの並び（0.1409 / 0.095 / 0.016）と向きが揃っている。

    Parameters
    ----------
    steps : list[int]
        エピソードごとのステップ数。
    cap : int | None
        打ち切りのステップ数。`None` なら**観測された最大値**を上限とみなす
        （採点ログには上限が書かれていないため）。

    Returns
    -------
    count : int
        上限に達した本数。
    """
    if not steps:
        return 0
    limit = cap if cap is not None else max(steps)
    return sum(1 for value in steps if value >= limit)


def _format_score(score: float | None) -> str:
    """スコアを桁を落とさずに短く書く。`0.141` は `0.1410` にしない。"""
    return "—" if score is None else f"{score:.4g}"


def _describe(lines: list[str], label: str, data: dict) -> None:
    """1 回分の内容を 3 行で書く。**ステップ数が多い回は前後だけ出す。**"""
    steps = data["steps"]
    cap = max(steps) if steps else 0
    version = data.get("version")
    elapsed = data.get("elapsed_sec")
    stamp = "".join([
        f" / policy_server {version}" if version else "",
        f" / {elapsed:.0f} 秒" if elapsed else "",
    ])
    lines.append(f"  {label}  {data['name']}{stamp}")
    lines.append(
        f"    エピソード {len(steps)} 本 / 最大 {cap} 歩 / "
        f"打ち切りに張り付き {_at_cap(steps, data.get('max_steps'))} 本 / "
        f"スコア {_format_score(data.get('score'))}"
    )
    if len(steps) <= 16:
        lines.append(f"    ステップ数 {steps}")
    else:
        head = ", ".join(str(value) for value in steps[:8])
        tail = ", ".join(str(value) for value in steps[-4:])
        lines.append(f"    ステップ数 [{head}, … , {tail}]（{len(steps)} 本のうち先頭 8 と末尾 4）")


def _compare_pair(lines: list[str], local: dict, production: dict,
                  tolerance: int) -> list[str]:
    """1 組を突き合わせ、気になった点を文字列で返す。"""
    notes: list[str] = []

    _describe(lines, "採点環境", production)
    _describe(lines, "手元    ", local)

    if local.get("exceptions"):
        notes.append(f"手元で例外により測れなかった回が {local['exceptions']} 件ある")

    prod_steps, local_steps = production["steps"], local["steps"]
    if len(prod_steps) != len(local_steps):
        notes.append(
            f"**本数が違う**（採点 {len(prod_steps)} 本 / 手元 {len(local_steps)} 本）。"
            "`--n-episodes 2` で 4 タスク × 2 = 8 本にすると採点環境と同じ形になる"
        )
    if local.get("max_steps") is not None and prod_steps:
        if local["max_steps"] != max(prod_steps):
            notes.append(
                f"**打ち切りが違う**（採点 {max(prod_steps)} 歩 / "
                f"手元 {local['max_steps']} 歩）。`--max-steps` を合わせる"
            )

    if len(prod_steps) == len(local_steps) and prod_steps:
        lines.append("")
        lines.append("    エピソードごとのステップ数")
        lines.append(f"      {'#':>2}  {'採点':>5}  {'手元':>5}  {'差':>5}  結末 / タスク")
        # `outcomes` と `tasks` は `steps` と同じ `episodes` から作るので必ず同じ長さ。
        gaps = 0
        rows = zip(prod_steps, local_steps, local["outcomes"], local["tasks"])
        for index, (theirs, mine, outcome, task) in enumerate(rows):
            gap = mine - theirs
            if abs(gap) > tolerance:
                gaps += 1
            mark = " " if abs(gap) <= tolerance else "*"
            lines.append(
                f"    {mark} {index:>2}  {theirs:>5}  {mine:>5}  {gap:>+5}  "
                f"{outcome} / {task[:52]}"
            )
        lines.append(f"      `*` は差が {tolerance} 歩を超えた回（{gaps} 本）")
        if gaps > len(prod_steps) // 2:
            notes.append(
                f"**半分以上の回でステップ数が {tolerance} 歩を超えて違う**（{gaps}/{len(prod_steps)}）。"
                "同じタスクを実行していない可能性がある"
            )
    return notes


def _tier_by_production(usable: list[tuple[dict, dict]],
                        tie: float) -> list[list[tuple[dict, dict]]]:
    """本番スコアの近いものを同じ段にまとめる。

    **8 本では 1 本の結末で 0.046 動くため、それ以下の差を順序として扱わない。**
    並べたうえで隣との差が `tie` 以内なら同じ段に入れる。

    Parameters
    ----------
    usable : list[tuple[dict, dict]]
        （手元, 採点環境）の組。どちらもスコアが読めているもの。
    tie : float
        同順とみなす差。

    Returns
    -------
    tiers : list[list[tuple[dict, dict]]]
        本番スコアの高い段から並べたもの。
    """
    ordered = sorted(usable, key=lambda item: item[1]["score"], reverse=True)
    tiers: list[list[tuple[dict, dict]]] = []
    for item in ordered:
        if tiers and abs(tiers[-1][-1][1]["score"] - item[1]["score"]) <= tie:
            tiers[-1].append(item)
        else:
            tiers.append([item])
    return tiers


def _report_ordering(lines: list[str], pairs: list[tuple[dict, dict]],
                     tie: float) -> bool:
    """本番と手元で順序が食い違わないかを書き、代理として使えるかを返す。

    **これが本題である。** ただし**本番スコアの小さな差は順序として使わない。**
    採点環境は 8 エピソードで、うち 1〜4 本目は 3 種類の重みすべてが 300 ステップに
    張り付いた。差が付くのは残り 4 本だけで、**素（0.1409）とλ=0.25（0.095）の
    違いは 1 本の結末が変わった分にすぎない。**

    判定は 2 つの信号で行う。**段をまたぐ組み合わせだけ**を見る。

    | 信号 | 良い向き |
    |---|---|
    | 手元のスコア | 本番スコアが高い側が高い |
    | 手元の打ち切り本数 | 本番スコアが高い側が少ない |

    Parameters
    ----------
    lines : list[str]
        出力先。
    pairs : list[tuple[dict, dict]]
        （手元, 採点環境）の組。
    tie : float
        本番スコアを同順とみなす差。

    Returns
    -------
    usable_as_proxy : bool
        段をまたぐ組み合わせが 1 つ以上あり、**スコアの向きがすべて一致したか**。
    """
    lines.append("")
    lines.append("=== 代理として使えるか ===")

    usable = [
        (local, prod) for local, prod in pairs
        if local.get("score") is not None and prod.get("score") is not None
    ]
    if len(usable) < 2:
        lines.append(f"  **組が {len(usable)} 個しかないので比べられない。**")
        lines.append("  答えの分かっている提出をもう 1 つ手元で実行する。")
        lines.append("  素 0.1409 / λ=0.25 は 0.095 / LoRA 20k は 0.016")
        return False

    tiers = _tier_by_production(usable, tie)
    lines.append(f"  本番スコアの差が {tie:g} 以内は同順として扱う")
    lines.append("  （8 本では 1 本の結末で 0.046 動くため、それ以下の差は判定に使わない）")
    lines.append("")
    for number, tier in enumerate(tiers, start=1):
        members = "、".join(
            f"{prod['name'][:30]} {_format_score(prod['score'])}" for _, prod in tier
        )
        note = "  ← 同順（判定に使わない）" if len(tier) > 1 else ""
        lines.append(f"    {number} 段目  {members}{note}")

    if len(tiers) < 2:
        lines.append("")
        lines.append("  **すべて同順なので判定できない。**")
        lines.append(f"  本番スコアが {tie:g} を超えて違う提出どうしで比べる。")
        return False

    lines.append("")
    lines.append("  段をまたぐ組み合わせ")
    header = (f"    {_pad('本番で上 / 下', 40)} "
              f"{_pad('手元のスコア', 22)} 手元の打ち切り本数")
    lines.append(header)

    # 段をまたぐ組み合わせを先に平らにする。**上の段の 1 件と下の段の 1 件**を
    # すべて組み合わせるので、同じ段の中の順序は判定に出てこない。
    crossings = [
        (high, low)
        for upper, lower in itertools.combinations(tiers, 2)
        for high in upper
        for low in lower
    ]

    score_ok = cap_ok = 0
    for (high_local, high_prod), (low_local, low_prod) in crossings:
        high_score, low_score = high_local["score"], low_local["score"]
        high_cap = _at_cap(high_local["steps"], high_local.get("max_steps"))
        low_cap = _at_cap(low_local["steps"], low_local.get("max_steps"))
        # **判定は 1 度だけ行い、印字と集計で同じ値を使う。**
        score_hit = high_score > low_score
        cap_hit = high_cap <= low_cap
        score_ok += score_hit
        cap_ok += cap_hit
        label = f"{high_prod['name'][:16]} / {low_prod['name'][:16]}"
        score_text = (f"{high_score:.4g} > {low_score:.4g} "
                      f"{'一致' if score_hit else '**逆**'}")
        lines.append(
            f"    {_pad(label, 40)} {_pad(score_text, 22)} "
            f"{high_cap} <= {low_cap} {'一致' if cap_hit else '**逆**'}"
        )

    total = len(crossings)
    lines.append("")
    lines.append(f"  スコアの向き      {score_ok}/{total} 一致")
    lines.append(f"  打ち切り本数の向き  {cap_ok}/{total} 一致")
    lines.append("")

    agrees = score_ok == total
    if agrees:
        lines.append("  **向きが食い違わなかった。この検証集合は提出前の判断に使える。**")
        lines.append("  ただし読めるのは**段の違い**までである"
                     "（素の系統か、崩壊したかの区別）。")
        lines.append("  **同順にした差を再現できたかどうかは根拠にしない。**")
        if cap_ok != total:
            lines.append("  **打ち切り本数だけは向きが逆である。** "
                         "スコアより素直な信号なので、原因を確かめること。")
    else:
        lines.append("  **向きが食い違った。この検証集合は提出前の判断に使えない。**")
        lines.append("  84 件の検証でも同じことが起きている"
                     "（手元では追加学習が素より上に見えたが、採点環境では 8.8 倍悪かった）。")
        lines.append("  **原因は件数ではなく、学習に使ったタスクを検証にも使っていることである。**")
        lines.append("  学習に使っていないタスクだけで検証集合を作る"
                     "（docs/howto/04_proxy_eval.md）。")
    return agrees


def main() -> None:
    args = _parse_args()

    pairs: list[tuple[dict, dict]] = []
    for spec in args.pair:
        if "=" not in spec:
            raise SystemExit(
                f"--pair は LOCAL_DIR=PROD_LOG の形で渡す: {spec}"
            )
        local_part, production_part = spec.split("=", 1)
        pairs.append((
            _read_local(Path(local_part)),
            _read_production(Path(production_part)),
        ))

    lines: list[str] = []
    all_notes: list[str] = []
    for index, (local, production) in enumerate(pairs, start=1):
        lines.append("")
        lines.append(f"=== 組 {index} ===")
        all_notes.extend(_compare_pair(lines, local, production, args.tolerance))

    agrees = _report_ordering(lines, pairs, args.score_tie)

    if all_notes:
        lines.append("")
        lines.append("=== 気になった点 ===")
        # **同じ指摘を組の数だけ並べない。** 順序は保ったまま重複を落とす。
        for note in dict.fromkeys(all_notes):
            lines.append(f"  - {note}")

    print("\n".join(lines))
    sys.exit(0 if agrees else 1)


if __name__ == "__main__":
    main()
