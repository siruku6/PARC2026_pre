#!/usr/bin/env python3
"""衝突と判定された回で、操作対象でない物体がどれだけ動いたかを表にする。

**何のためにあるか。** 衝突は 2 値の門で、掛かった回の点はまるごと 0 になる。
**だが「かすった」のか「倒した」のかは、真偽だけ見ていても分からない。**
1〜3 mm に固まっているなら推論側の後処理（接触の手前で減速する等）に意味があり、
数 cm に散らばっているなら後処理では届かず学習でしか直らない。**そこを分けるのが
この道具である。**

読むのは `kit_patches/rollout.py` が書いた `object_disp.jsonl` である
（1 行 1 ロールアウト）。**評価を回し直さなくてよいが、その JSONL が無い実行には
使えない**——採点器の JSON には変位の値が残っていないので、後から復元はできない。

**`done` で分けないと表が読めない。** 衝突した回には 2 種類ある——**ゴールに着いた
のに衝突で落ちた回**と、**そもそも 300 歩で着いていない回**である。後者は衝突が無くても
失敗なので、**衝突対策で取り戻せる点は前者だけである。**しかも後者は 300 歩ぶん暴れた
末の変位なので、**混ぜると分布の裾が実態より重く見える。**表 1 は必ず両方を並べて出す。

**距離が 2 通りあるのはキットの都合である。** キットが 1 mm のしきい値と比べて
`collided` を決めているのは、3 軸の絶対値の和（L1）である。実際に動いた距離
（L2）ではなく、最大 √3 倍だけ大きく出る。**衝突したかどうかは L1 で決まるので
そちらで数え、何 mm 動いたかは L2 で読む。**既定では両方を並べて出す。

    python3 summarize_object_disp.py <実行ディレクトリ>
    python3 summarize_object_disp.py <実行ディレクトリ>/results/object_disp.jsonl
    python3 summarize_object_disp.py --metric l2 <実行ディレクトリ>

Reads:
  - <実行ディレクトリ>/results/object_disp.jsonl
  - <実行ディレクトリ>/TASKS_CSV.csv（条件の名前。無ければ条件別の表を省く）
  - <実行ディレクトリ>/results/server_8000.json（衝突率の突き合わせ。無ければ省く）
Generates:
  - 標準出力に 3 つの表
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# 区切りは依頼どおりの 5 段。下限は 1 mm（しきい値そのもの）。
BINS_MM: list[tuple[float, float, str]] = [
    (1.0, 2.0, "1〜2 mm"),
    (2.0, 5.0, "2〜5 mm"),
    (5.0, 10.0, "5〜10 mm"),
    (10.0, 50.0, "10〜50 mm"),
    (50.0, float("inf"), "50 mm 以上"),
]

# しきい値を下回る値がここへ落ちたときだけ現れる行。
# **L2 の表では当たり前に出る**（L1 が 1 mm を超えても、実距離は下回りうる）。
# **L1 の表で出たら、しきい値の比較と食い違っているので要調査である。**
UNDER_LABEL = "1 mm 未満"

METRIC_KEYS: dict[str, str] = {"l1": "max_disp_m", "l2": "max_disp_l2_m"}


def _resolve_jsonl(target: Path) -> Path:
    """指定が実行ディレクトリでもファイルでも、JSONL の実体へたどり着く。"""
    if target.is_file():
        return target
    for candidate in (target / "results" / "object_disp.jsonl",
                      target / "object_disp.jsonl"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"object_disp.jsonl が見つからない: {target}")


def _load_records(jsonl_path: Path) -> list[dict]:
    """JSONL を 1 行 1 ロールアウトの辞書として読む。"""
    records: list[dict] = []
    for line_number, line in enumerate(jsonl_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{jsonl_path}:{line_number} が読めない: {error}")
    return records


def _load_condition_labels(run_dir: Path) -> dict[str, str]:
    """タスク名から条件名（Base / Camera az+45 など）への対応を作る。

    `score_composite.py --by-category` と同じ付け方にしてある。**表どうしを
    並べたときに条件名が食い違わないようにするため**で、CSV が無ければ空を返す。
    """
    csv_path = run_dir / "TASKS_CSV.csv"
    if not csv_path.is_file():
        return {}
    labels: dict[str, str] = {}
    for row in csv.DictReader(csv_path.open()):
        category = row.get("category", "")
        note = row.get("difficulty_level", "")
        labels[row["task_id"]] = (
            "Base" if category == "Base" else f"{category.split()[0]} {note}".strip()
        )
    return labels


def _coarse_condition(label: str) -> str:
    """Base / Camera / Robot の 3 つへまとめる。"""
    if label.startswith("Camera"):
        return "Camera"
    if label.startswith("Base") or not label:
        return "Base"
    return "Robot"


def _reported_collision_count(run_dir: Path) -> tuple[float, int] | None:
    """採点器の JSON が言う衝突率と総エピソード数を取り出す。"""
    json_path = run_dir / "results" / "server_8000.json"
    if not json_path.is_file():
        return None
    tracks = json.loads(json_path.read_text()).get("tracks") or []
    if not tracks:
        return None
    overall = tracks[0].get("overall_metrics", {})
    rate = overall.get("mean_collision_rate")
    episodes = overall.get("n_total_episodes")
    if rate is None or not episodes:
        return None
    return float(rate), int(episodes)


def _max_displacement_mm(record: dict, metric: str) -> float:
    """そのロールアウトで最も動いた物体の変位を mm で返す。"""
    values = record.get(METRIC_KEYS[metric], {}).values()
    return max(values, default=0.0) * 1000.0


def _worst_object(record: dict, metric: str) -> str | None:
    """そのロールアウトで最も動いた物体の名前。動いた物体が無ければ None。"""
    displacements = record.get(METRIC_KEYS[metric], {})
    if not displacements:
        return None
    return max(displacements, key=lambda name: displacements[name])


def _bin_label(value_mm: float) -> str:
    """変位を 5 段のどれかに割り当てる。"""
    for low, high, label in BINS_MM:
        if low <= value_mm < high:
            return label
    return UNDER_LABEL


def _format_distribution(counts: Counter, total: int) -> list[str]:
    """区切りごとの件数と割合を、表の本文の行として組む。"""
    rows: list[str] = []
    for _, _, label in BINS_MM:
        count = counts.get(label, 0)
        share = count / total if total else 0.0
        rows.append(f"| {label} | {count} | {share:.1%} |")
    if counts.get(UNDER_LABEL):
        count = counts[UNDER_LABEL]
        rows.append(f"| {UNDER_LABEL} | {count} | {count / total:.1%} |")
    return rows


def _under_threshold_note(counts: Counter, metric: str) -> str | None:
    """「1 mm 未満」の行が出た理由を、L1 と L2 で書き分ける。"""
    if not counts.get(UNDER_LABEL):
        return None
    if metric == "l2":
        return ("「1 mm 未満」は当たり前に出る。**しきい値と比べられているのは L1 で、"
                "実距離の L2 はそれを下回りうる**（3 軸に散れば最大 √3 倍の差）。")
    return ("**「1 mm 未満」が L1 の表に出るのはおかしい。**しきい値の比較と食い違うので、"
            "その行を JSONL で直接見ること。")


def _median_line(records: list[dict], metric: str) -> str:
    """中央値・最小・最大を 1 行にする。"""
    values = sorted(_max_displacement_mm(r, metric) for r in records)
    if not values:
        return "—"
    return (f"中央値 {values[len(values) // 2]:.2f} ／ 最小 {values[0]:.2f} "
            f"／ 最大 {values[-1]:.2f}")


def _print_distribution_table(collided: list[dict], metric: str) -> None:
    """表 1。衝突した回の最大変位が、どの区切りに何件入るか。

    **`done` で 2 つに分けて並べる。**ゴールに着いたのに衝突で落ちた回だけが、
    衝突対策で取り戻せる点である。着いていない回は衝突が無くても失敗で、しかも
    300 歩ぶん暴れた末の変位なので、混ぜると裾が実態より重く見える。
    """
    reached = [r for r in collided if r.get("done")]
    never = [r for r in collided if not r.get("done")]
    groups = [("全体", collided), ("着いたのに衝突で落ちた", reached),
              ("着いていない（打ち切り）", never)]
    counts_all = Counter(_bin_label(_max_displacement_mm(r, metric)) for r in collided)

    print(f"\n## 1. 衝突した回の最大変位（{metric.upper()}・{len(collided)} 本）\n")
    print("| 区切り | " + " | ".join(f"{name}（{len(rs)} 本）" for name, rs in groups)
          + " |")
    print("|---|" + "---:|" * len(groups))
    labels = [label for _, _, label in BINS_MM]
    if counts_all.get(UNDER_LABEL):
        labels.append(UNDER_LABEL)
    for bin_label in labels:
        cells = []
        for _, records in groups:
            counts = Counter(_bin_label(_max_displacement_mm(r, metric))
                             for r in records)
            count, total = counts.get(bin_label, 0), len(records)
            cells.append(f"{count}（{count / total:.0%}）" if total else "—")
        print(f"| {bin_label} | " + " | ".join(cells) + " |")
    print("| **mm** | " + " | ".join(_median_line(rs, metric) for _, rs in groups) + " |")

    note = _under_threshold_note(counts_all, metric)
    if note:
        print(f"\n{note}")


def _print_culprit_table(collided: list[dict], metric: str, top: int) -> None:
    """表 2。衝突した回で最も動いた物体は何か、上位いくつか。"""
    counts: Counter = Counter()
    worst_by_object: dict[str, float] = {}
    for record in collided:
        name = _worst_object(record, metric)
        if name is None:
            continue
        counts[name] += 1
        value_mm = record[METRIC_KEYS[metric]][name] * 1000.0
        worst_by_object[name] = max(worst_by_object.get(name, 0.0), value_mm)
    reached_counts: Counter = Counter()
    for record in collided:
        if not record.get("done"):
            continue
        name = _worst_object(record, metric)
        if name is not None:
            reached_counts[name] += 1

    print(f"\n## 2. どの物体が犯人か（上位 {top}）\n")
    print("| 物体 | 件数 | 割合 | うち着いた | その物体の最大変位 |")
    print("|---|---:|---:|---:|---:|")
    total = len(collided)
    for name, count in counts.most_common(top):
        print(f"| {name} | {count} | {count / total:.1%} "
              f"| {reached_counts.get(name, 0)} | {worst_by_object[name]:.2f} mm |")
    print(f"\n物体の種類は {len(counts)} 個。**「うち着いた」がその物体で失った点である。**")


def _print_by_condition_table(
    collided: list[dict], metric: str, labels: dict[str, str], all_records: list[dict]
) -> None:
    """表 3。条件（Base / Camera / Robot）ごとに表 1 がどう変わるか。"""
    if not labels:
        print("\n## 3. 条件ごと\n\nTASKS_CSV.csv が無いので出せない。")
        return
    conditions = ["Base", "Camera", "Robot"]
    counts_by_condition: dict[str, Counter] = {c: Counter() for c in conditions}
    totals: dict[str, int] = {c: 0 for c in conditions}
    episodes_by_condition: dict[str, int] = {c: 0 for c in conditions}

    for record in all_records:
        condition = _coarse_condition(labels.get(record["task"], ""))
        episodes_by_condition[condition] += 1
    for record in collided:
        condition = _coarse_condition(labels.get(record["task"], ""))
        counts_by_condition[condition][
            _bin_label(_max_displacement_mm(record, metric))
        ] += 1
        totals[condition] += 1

    print(f"\n## 3. 条件ごとの最大変位（{metric.upper()}）\n")
    header = "| 区切り | " + " | ".join(
        f"{c}（{totals[c]}/{episodes_by_condition[c]} 本）" for c in conditions
    ) + " |"
    print(header)
    print("|---|" + "---:|" * len(conditions))
    labels_to_show = [label for _, _, label in BINS_MM]
    if any(counts_by_condition[c].get(UNDER_LABEL) for c in conditions):
        labels_to_show.append(UNDER_LABEL)  # 理由は表 1 の脚注と同じ
    for bin_label in labels_to_show:
        cells = []
        for condition in conditions:
            count = counts_by_condition[condition].get(bin_label, 0)
            total = totals[condition]
            cells.append(f"{count}（{count / total:.0%}）" if total else "—")
        print(f"| {bin_label} | " + " | ".join(cells) + " |")


def _print_consistency_check(records: list[dict], run_dir: Path) -> None:
    """JSONL の行数と衝突の件数が、採点器の JSON と合うかを確かめる。"""
    collided_count = sum(1 for r in records if r["collided"])
    print("\n## 0. 突き合わせ\n")
    print(f"- JSONL の行数: {len(records)}")
    print(f"- うち衝突した回: {collided_count}")
    reported = _reported_collision_count(run_dir)
    if reported is None:
        print("- results/server_8000.json が無いので突き合わせを省いた")
        return
    rate, episodes = reported
    expected = rate * episodes
    print(f"- 採点器の JSON: 衝突率 {rate:.4f} × {episodes} 本 = {expected:.1f}")
    verdict = "一致" if abs(expected - collided_count) < 0.5 else "**食い違う**"
    print(f"- 判定: {verdict}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path,
                        help="実行ディレクトリか、object_disp.jsonl そのもの")
    parser.add_argument("--metric", choices=["l1", "l2", "both"], default="both",
                        help="L1 はキットが衝突の判定に使う値、L2 は実距離（既定 both）")
    parser.add_argument("--top", type=int, default=10, help="表 2 に出す物体の数")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    jsonl_path: Path = _resolve_jsonl(args.target)
    run_dir: Path = (
        jsonl_path.parent.parent if jsonl_path.parent.name == "results"
        else jsonl_path.parent
    )
    records: list[dict] = _load_records(jsonl_path)
    if not records:
        raise SystemExit(f"{jsonl_path} が空である")
    collided: list[dict] = [r for r in records if r["collided"]]
    labels: dict[str, str] = _load_condition_labels(run_dir)

    print(f"# 衝突した回の変位 — {run_dir.name}")
    _print_consistency_check(records, run_dir)
    if not collided:
        print("\n衝突した回が 1 本も無い。")
        return 0
    for metric in (["l1", "l2"] if args.metric == "both" else [args.metric]):
        _print_distribution_table(collided, metric)
        _print_culprit_table(collided, metric, args.top)
        _print_by_condition_table(collided, metric, labels, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
