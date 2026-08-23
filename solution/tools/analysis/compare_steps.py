#!/usr/bin/env python3
"""2 つの評価の回を、**ステップ数**で 1 件ずつ突き合わせる。

本番で効いたと確認できている量はステップ数だけである（版 3.10.0 はステップ数が
+29% になって総合スコアが 0.264 → 0.169 に落ちた）。秒数は 1 時間の上限に
対して 15 倍以上の余裕があるので、判断には使わない。

`summarize_gpu_eval.py` は成功率と滑らかさを出すが、**ステップ数の合計を
1 件ずつ対にして出さない**ため、この道具を足した。

Reads:
  - <各回>/eval.log の `[結果]` 行
Generates:
  - 標準出力のみ（ファイルは書かない）
"""
import argparse
import pathlib
import re

# 採点環境の 1 エピソードの上限。配布キットの既定は 600 だが、採点ログでは
# 複数の提出で「ちょうど 300」が現れるため、本番の上限は 300 である。
_MAX_STEPS = 300


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="基準にする回の出力先")
    parser.add_argument("--other", required=True, help="比べる回の出力先")
    parser.add_argument(
        "--base-label", default="基準", help="表の見出しに使う名前"
    )
    parser.add_argument("--other-label", default="比較", help="同上")
    return parser.parse_args()


def _read_results(run_dir: pathlib.Path) -> dict[str, tuple[str, int]]:
    """`eval.log` の `[結果]` 行から、タスク名 → (見出し, ステップ数) を作る。

    Parameters
    ----------
    run_dir : Path
        `run_docker_eval.sh` が作った出力先。

    Returns
    -------
    results : dict[str, tuple[str, int]]
        タスク名をキーに、`成功`／`衝突`／`未完了` とステップ数を持つ。
    """
    text = (run_dir / "eval.log").read_text(errors="replace")
    results: dict[str, tuple[str, int]] = {}
    for line in text.splitlines():
        if "[結果]" not in line:
            continue
        body = line.split("[結果]", 1)[1].strip()
        match = re.match(r"(\S+) ep0: (\S+) (\d+) steps", body)
        if match:
            results[match.group(1)] = (match.group(2), int(match.group(3)))
    return results


def _print_table(
    base: dict[str, tuple[str, int]],
    other: dict[str, tuple[str, int]],
    base_label: str,
    other_label: str,
) -> None:
    """共通するタスクだけを、ステップ数の差が大きい順に並べて出す。"""
    shared = sorted(set(base) & set(other))
    rows = []
    for name in shared:
        base_result, base_steps = base[name]
        other_result, other_steps = other[name]
        rows.append((other_steps - base_steps, name, base[name], other[name]))
    rows.sort(reverse=True)

    print(f"共通するタスク {len(shared)} 件"
          f"（{base_label} {len(base)} 件 / {other_label} {len(other)} 件）")
    print()
    print(f"{'タスク':46} {base_label:>16} {other_label:>16} {'ステップ数差':>8}")
    print("-" * 92)
    for delta, name, left, right in rows:
        mark = "" if left[0] == right[0] else "  ←変わった"
        print(f"{name[:46]:46} {left[0] + ' ' + str(left[1]):>16} "
              f"{right[0] + ' ' + str(right[1]):>16} {delta:>+8}{mark}")
    print("-" * 92)

    _print_totals(rows, base_label, other_label)


def _print_totals(
    rows: list[tuple[int, str, tuple[str, int], tuple[str, int]]],
    base_label: str,
    other_label: str,
) -> None:
    """合計ステップ数と成功件数、そして「成功した回だけのステップ数」を出す。

    打ち切り（300 ステップ）が混ざると合計が天井に張り付くため、
    **基準の回で完走していた件に絞ったステップ数**も併せて出す。**これがいちばん
    素直に「遅くなっていないか」を映す。**打ち切りに達していた件は、救われると
    大きく減るため、全体の合計では減速と相殺されて見えなくなる。
    """
    base_total = sum(left[1] for _, _, left, _ in rows)
    other_total = sum(right[1] for _, _, _, right in rows)
    base_wins = sum(left[0] == "成功" for _, _, left, _ in rows)
    other_wins = sum(right[0] == "成功" for _, _, _, right in rows)

    both = [(left, right) for _, _, left, right in rows
            if left[0] == "成功" and right[0] == "成功"]
    base_both = sum(left[1] for left, _ in both)
    other_both = sum(right[1] for _, right in both)

    # **基準の回で打ち切りに達していなかった件だけを見る。**
    # 「両方が成功した件」だけでは、比べる回で失敗に転じた件が除外され、
    # 減速を見落とす。版 3.10.0 では全体 +0.3% / 両方成功 +1.6% に対し、
    # この指標が +13.1% で、唯一悪化を捉えていた。
    finished = [(left, right) for _, _, left, right in rows if left[1] < _MAX_STEPS]
    base_finished = sum(left[1] for left, _ in finished)
    other_finished = sum(right[1] for _, right in finished)

    def _ratio(before: int, after: int) -> str:
        if before == 0:
            return "—"
        return f"{(after / before - 1) * 100:+.1f}%"

    print(f"成功件数          {base_label} {base_wins:3}  /  {other_label} {other_wins:3}")
    print(f"合計ステップ数          {base_label} {base_total:5}  /  {other_label} {other_total:5}"
          f"  {_ratio(base_total, other_total)}")
    print(f"両方が成功した {len(both):2} 件だけのステップ数        "
          f"{base_label} {base_both:5}  /  {other_label} {other_both:5}"
          f"  {_ratio(base_both, other_both)}")
    print(f"**基準で完走していた {len(finished):2} 件のステップ数**  "
          f"{base_label} {base_finished:5}  /  {other_label} {other_finished:5}"
          f"  {_ratio(base_finished, other_finished)}")
    print()
    print("**ステップ数が減っていれば良い。** 本番で効くと確認できている量はこれだけである")
    print("（版 3.10.0 はステップ数 +29% で総合スコアが 0.264 → 0.169 に落ちた）。")


def main() -> None:
    args = _parse_args()
    base = _read_results(pathlib.Path(args.base))
    other = _read_results(pathlib.Path(args.other))
    _print_table(base, other, args.base_label, args.other_label)


if __name__ == "__main__":
    main()
