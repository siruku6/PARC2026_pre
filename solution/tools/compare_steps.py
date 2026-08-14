#!/usr/bin/env python3
"""2 つの評価の回を、**歩数**で 1 件ずつ突き合わせる。

本番で効いたと確認できている量は歩数だけである（版 3.10.0 は歩数が
+29% になって総合スコアが 0.264 → 0.169 に落ちた）。秒数は 1 時間の上限に
対して 15 倍以上の余裕があるので、判断には使わない。

`summarize_gpu_eval.py` は成功率と滑らかさを出すが、**歩数の合計を
1 件ずつ対にして出さない**ため、この道具を足した。

Reads:
  - <各回>/eval.log の `[結果]` 行
Generates:
  - 標準出力のみ（ファイルは書かない）
"""
import argparse
import pathlib
import re


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
    """`eval.log` の `[結果]` 行から、タスク名 → (見出し, 歩数) を作る。

    Parameters
    ----------
    run_dir : Path
        `run_docker_eval.sh` が作った出力先。

    Returns
    -------
    results : dict[str, tuple[str, int]]
        タスク名をキーに、`成功`／`衝突`／`未完了` と歩数を持つ。
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
    """共通するタスクだけを、歩数の差が大きい順に並べて出す。"""
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
    print(f"{'タスク':46} {base_label:>16} {other_label:>16} {'歩数差':>8}")
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
    """合計歩数と成功件数、そして「成功した回だけの歩数」を出す。

    打ち切り（300 歩）が混ざると合計が天井に張り付くため、
    **両方が成功した回に絞った歩数**も併せて出す。ここがいちばん素直に
    「速くなったか」を映す。
    """
    base_total = sum(left[1] for _, _, left, _ in rows)
    other_total = sum(right[1] for _, _, _, right in rows)
    base_wins = sum(left[0] == "成功" for _, _, left, _ in rows)
    other_wins = sum(right[0] == "成功" for _, _, _, right in rows)

    both = [(left, right) for _, _, left, right in rows
            if left[0] == "成功" and right[0] == "成功"]
    base_both = sum(left[1] for left, _ in both)
    other_both = sum(right[1] for _, right in both)

    def _ratio(before: int, after: int) -> str:
        if before == 0:
            return "—"
        return f"{(after / before - 1) * 100:+.1f}%"

    print(f"成功件数          {base_label} {base_wins:3}  /  {other_label} {other_wins:3}")
    print(f"合計歩数          {base_label} {base_total:5}  /  {other_label} {other_total:5}"
          f"  {_ratio(base_total, other_total)}")
    print(f"**両方が成功した {len(both):2} 件だけの歩数**  "
          f"{base_label} {base_both:5}  /  {other_label} {other_both:5}"
          f"  {_ratio(base_both, other_both)}")
    print()
    print("**歩数が減っていれば良い。** 本番で効くと確認できている量はこれだけである")
    print("（版 3.10.0 は歩数 +29% で総合スコアが 0.264 → 0.169 に落ちた）。")


def main() -> None:
    args = _parse_args()
    base = _read_results(pathlib.Path(args.base))
    other = _read_results(pathlib.Path(args.other))
    _print_table(base, other, args.base_label, args.other_label)


if __name__ == "__main__":
    main()
