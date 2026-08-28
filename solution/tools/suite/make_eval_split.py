#!/usr/bin/env python3
"""学習に使わなかった実演がある指示文から、評価用のタスク集合を作る。

**手元の 84 件が本番の代わりにならないのは、学習に使ったものを検証にも
使っているからである。** 追加学習した重みは手元 84 件で素より上に見えたが
（0.500 対 0.476）、採点環境では 8.8 倍悪かった（0.016 対 0.1409）。

**そこで、学習が引かなかった実演がある指示文だけを評価に使う。**
判定は `episodes.json`（学習が使ったエピソード番号）とデータセットのメタデータから
機械的に決まるので、選ぶ人の裁量が入らない。

    # 既定（指示文ごとに 1 件・本番と同じ 2 カテゴリ）
    python3 solution/tools/suite/make_eval_split.py --record outputs/train2/episodes.json

    # 選び直す（**後から変えられる**）
    ... --per-task 2                      # 指示文ごとの件数を増やす
    ... --offset 1                        # 同じ指示文から別の variant を選ぶ
    ... --categories 'Light Conditions'   # カテゴリを絞る
    ... --dry-run                          # 書かずに内容だけ見る

**出力の CSV が正である。** 手で書き換えてもよいし、引数を変えて作り直しても
よい。隣に置く `.meta.json` に、そのとき使った引数と根拠の数を残す。

## この集合が言えること・言えないこと

| | |
|---|---|
| **言える** | その指示文には、**学習が 1 度も引かなかった実演があり、しかもその見た目は学習に使った実演のどれとも違う** |
| **言えない** | 選んだ variant そのものが学習に入っていないこと |

**variant とエピソードは対応付けられない。** variant は 10,030 件、データセットの実演は
14,347 本で、指示文ごとの比は 0.58〜2.95（中央 1.45）と整数にならない。
データセットは variant の一部を不揃いな本数で収録している。**だから
「この variant は学習に使っていない」とは言えない。** 強めるには、variant の初期画面を
描いて明るさを測り、`.meta.json` の `seen_brightness` の外側にあるものだけを
残す（描画には GPU マシンが要る）。

Reads:
  - <record>（`train.py` が書く `episodes.json`。`episodes` に使った番号が入る）
  - <episodes-parquet>（データセットの `meta/episodes/*.parquet` を 1 つにしたもの）
  - <libero-plus>/libero/libero/benchmark/task_classification.json
Generates:
  - compe/val/EVAL_TASKS.csv（`compe/t1/T1_TASKS.csv` と同じ列。**評価にそのまま渡せる**）
  - compe/val/EVAL_TASKS.meta.json（引数と根拠の数）
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# 採点環境の摂動カテゴリ。運営 Q&A No.21 の回答は「7 カテゴリすべてが
# 含まれているわけではありません」「同一タスク・同一ドメインにおける、
# テクスチャなどの摂動」であり、配布された 4 タスクもこの 2 つだけである。
_PRODUCTION_CATEGORIES = ("Background Textures", "Light Conditions")

# 明るさを場面の識別子として使うときの刻み。**0.001 で 1 指示文あたり
# 約 94 通りに分かれる**（中央 385 本）。粗くすると別の場面を同じと見なす。
_BRIGHTNESS_RESOLUTION = 3

# **CSV の列は `make_validation_suite` に作らせる。**
# `T1_TASKS.csv` の列だけでは足りない。`init_states_file` が無いと、
# 評価側が接尾辞を正規表現で剥がして初期状態を探すことになり、
# **`_tb_N` のように剥がせない摂動で「ファイルが無い」と落ちる**
# （`compe/t1/register.py` の `_SUFFIX_RE` は `_light_*` と `_table_\d+` だけ）。
# 物体が増える摂動は variant 名そのものの初期状態を持つため、相対パスも要る。
# **置き場の解決は 1 か所に置く。** ここで作り直すと必ず食い違う。


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True,
                        help="train.py が書いた episodes.json")
    parser.add_argument("--episodes-parquet", type=Path,
                        default=_REPO_ROOT / "outputs/tmp/libero_plus_episodes.parquet",
                        help="データセットのエピソードのメタデータ")
    parser.add_argument("--libero-plus", type=Path,
                        default=_REPO_ROOT / "work/LIBERO-plus",
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--per-task", type=int, default=1,
                        help="指示文ごとに選ぶ variant の数（既定 1）")
    parser.add_argument("--offset", type=int, default=0,
                        help="同じ指示文から別の variant を選び直すときにずらす数")
    parser.add_argument("--categories", type=str,
                        default=",".join(_PRODUCTION_CATEGORIES),
                        help="使う摂動カテゴリ。**読点で区切る**")
    parser.add_argument("--max-demo-steps", type=int, default=300,
                        help="未使用の実演がこのステップ数を超える指示文は候補から外す。"
                             "**採点は 300 ステップで打ち切るため**")
    parser.add_argument("--output", type=Path,
                        default=_REPO_ROOT / "compe/val/EVAL_TASKS.csv")
    # **置き場は libero の設定（`~/.libero/config.yaml`）から決まる。**
    # 評価パイプラインが読むのと同じ場所を見ないと、CSV は作れたのに
    # 評価の開始直後に落ちる。`make_validation_suite.py` と同じ引数にしてある。
    parser.add_argument("--bddl-root", type=Path, default=None,
                        help="bddl の置き場。省略すると libero の設定に従う")
    parser.add_argument("--init-states-root", type=Path, default=None,
                        help="初期状態の置き場。省略すると libero の設定に従う")
    parser.add_argument("--dry-run", action="store_true",
                        help="書かずに内容だけ出す")
    return parser.parse_args()


def _flatten(value) -> list[float]:
    """入れ子になった統計値を平坦な float のリストにする。

    parquet の統計値は `[[[0.35]]]` のように三重に入れ子になっている。
    `train.py` の `_flatten_numbers` と同じ扱いにする。
    """
    if isinstance(value, (int, float)):
        return [float(value)]
    return [number for item in value for number in _flatten(item)]


def _load_used(path: Path) -> set[int]:
    """記録から、学習に使ったエピソード番号を読む。"""
    record = json.loads(path.read_text(encoding="utf-8"))
    episodes = record.get("episodes")
    if episodes in ("all", None):
        raise SystemExit(
            f"[eval] {path} は「全エピソードを使った」と言っています。\n"
            "  **未使用の実演が無いので、この方法では作れません。**"
        )
    return {int(value) for value in episodes}


def _load_episodes(path: Path) -> list[dict]:
    """エピソードのメタデータを読み、指示文・ステップ数・明るさだけ返す。"""
    if not path.is_file():
        raise SystemExit(f"[eval] メタデータがありません: {path}")

    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[
        "episode_index", "tasks", "length", "stats/observation.images.front/mean",
    ])
    instructions = [
        value if isinstance(value, str) else str(value[0])
        for value in table.column("tasks").to_pylist()
    ]
    brightness = [
        statistics.fmean(_flatten(value))
        for value in table.column("stats/observation.images.front/mean").to_pylist()
    ]
    return [
        {"episode_index": int(index), "instruction": name,
         "length": int(length), "brightness": float(value)}
        for index, name, length, value in zip(
            table.column("episode_index").to_pylist(), instructions,
            table.column("length").to_pylist(), brightness,
        )
    ]


def _novel_by_instruction(
    episodes: list[dict], used: set[int], max_demo_steps: int
) -> dict[str, dict]:
    """指示文ごとに、学習が引かなかった実演のうち見た目も未知のものを数える。

    **見た目は明るさを丸めた値で識別する。** 使った実演に同じ値があれば、
    同じ場面が学習に入っていたと見なす。

    Parameters
    ----------
    episodes : list[dict]
        全エピソードのメタデータ。
    used : set[int]
        学習に使ったエピソード番号。
    max_demo_steps : int
        未使用の実演のステップ数の上限。**採点の打ち切りを超える実演は数えない。**

    Returns
    -------
    summary : dict[str, dict]
        指示文ごとの `novel`（本数）・`used`・`unused`・`seen_brightness`。
    """
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"used": [], "unused": []}
    )
    for episode in episodes:
        key = "used" if episode["episode_index"] in used else "unused"
        grouped[episode["instruction"]][key].append(episode)

    summary: dict[str, dict] = {}
    for instruction, split in grouped.items():
        seen = {
            round(item["brightness"], _BRIGHTNESS_RESOLUTION)
            for item in split["used"]
        }
        novel = [
            item for item in split["unused"]
            if round(item["brightness"], _BRIGHTNESS_RESOLUTION) not in seen
            and item["length"] <= max_demo_steps
        ]
        summary[instruction] = {
            "novel": len(novel),
            "used": len(split["used"]),
            "unused": len(split["unused"]),
            "seen_brightness": (
                [round(min(seen), 3), round(max(seen), 3)] if seen else []
            ),
            "novel_brightness": (
                [round(min(item["brightness"] for item in novel), 3),
                 round(max(item["brightness"] for item in novel), 3)] if novel else []
            ),
        }
    return summary


def _load_variants(libero_plus: Path, categories: list[str]) -> dict[str, list[dict]]:
    """指示文ごとに、使えるカテゴリの variant を集める。

    素のタスク名の取り出しは `make_holdout_split` と共通にする。
    **同じ正規表現を 2 か所に置くと必ず片方が古くなる。**
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_holdout_split import base_task, instruction as to_instruction

    path = (libero_plus / "libero" / "libero" / "benchmark"
            / "task_classification.json")
    if not path.is_file():
        raise SystemExit(f"[eval] variant の一覧がありません: {path}")

    wanted = set(categories)
    variants: dict[str, list[dict]] = defaultdict(list)
    for suite, items in json.loads(path.read_text(encoding="utf-8")).items():
        for item in items:
            if item["category"] not in wanted:
                continue
            key = to_instruction(base_task(item["name"]))
            variants[key].append({**item, "suite": suite})
    # **並びを固定する。** 難易度、次に名前で並べれば、同じ引数なら同じものが出る。
    # **難易度が入っていない variant がある**ので、その分は後ろへ回す（`None` は
    # `int` と比べられず、そのまま並べ替えると例外になる）。
    for key in variants:
        variants[key].sort(
            key=lambda item: (
                item["difficulty_level"] is None,
                item["difficulty_level"] or 0,
                item["name"],
            )
        )
    return variants


def _pick(items: list[dict], count: int, offset: int) -> list[dict]:
    """難易度で並べた中から、両端を含めて等間隔に選ぶ。

    `offset` は選ぶ位置をずらす。**同じ指示文から別の variant へ替えたいときに使う。**
    """
    if not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[(len(items) // 2 + offset) % len(items)]]
    positions = [
        (round(rank * (len(items) - 1) / (count - 1)) + offset) % len(items)
        for rank in range(count)
    ]
    return [items[position] for position in sorted(set(positions))]


def main() -> None:
    args = _parse_args()
    categories = [name.strip() for name in args.categories.split(",") if name.strip()]

    used = _load_used(args.record)
    episodes = _load_episodes(args.episodes_parquet)
    summary = _novel_by_instruction(episodes, used, args.max_demo_steps)
    variants = _load_variants(args.libero_plus, categories)

    print(f"[eval] 記録 {args.record}（使った実演 {len(used):,} 本）")
    print(f"[eval] 指示文 {len(summary)} 種 / カテゴリ {categories}")

    eligible = sorted(
        name for name, info in summary.items() if info["novel"] > 0
    )
    without_variants = [name for name in eligible if not variants.get(name)]
    eligible = [name for name in eligible if variants.get(name)]

    print(f"[eval] 未使用かつ見た目も未知の実演がある指示文 "
          f"{len(eligible)} 種（variant がある分）")
    if without_variants:
        print(f"[eval]   **このカテゴリに variant が無い指示文 {len(without_variants)} 種は外した**")
    dropped = sorted(
        name for name, info in summary.items() if info["novel"] == 0
    )
    if dropped:
        print(f"[eval]   未使用に未知の見た目が無い指示文 {len(dropped)} 種は外した")

    selected = [
        item for name in eligible
        for item in _pick(variants[name], args.per_task, args.offset)
    ]
    print(f"[eval] 選んだ variant {len(selected)} 件")

    # **行の組み立ては `make_validation_suite` に任せる。**
    # bddl と初期状態が実際に在る置き場を探し、`init_states_file` を決めきる。
    # **実体が無い variant はここで落ちる。** 黙って減らさず、下で名指しする。
    from make_validation_suite import (
        _build_rows, _index_init_state_stems, _report, _resolve_libero_path,
    )

    init_states_root = _resolve_libero_path(args.init_states_root, "init_states")
    bddl_root = _resolve_libero_path(args.bddl_root, "bddl_files")
    for root in (init_states_root, bddl_root):
        if not root.is_dir():
            raise SystemExit(f"[eval] 置き場が見つかりません: {root}")
    print(f"[eval] bddl {bddl_root} / 初期状態 {init_states_root}")

    rows, skipped = _build_rows(
        selected, bddl_root, init_states_root,
        _index_init_state_stems(init_states_root),
    )
    _report(rows, skipped)
    if len(rows) < len(selected):
        print(f"[eval] **実体が無くて落ちた variant {len(selected) - len(rows)} 件。**")
        print("[eval]   `--offset` をずらして別の variant にすると埋まることがある")
    if not rows:
        raise SystemExit("[eval] 1 件も載せられませんでした。上の理由を読むこと")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    for key, value in sorted(counts.items()):
        print(f"[eval]   {key}: {value} 件")
    print(f"[eval] 1 件あたり約 1.7 分 → 全部で約 {len(rows) * 1.7 / 60:.1f} 時間")

    if args.dry_run:
        print()
        for row in rows:
            print(f"  {row['task_num']:>3} {row['difficulty_level']} "
                  f"{row['category'][:20]:20s} {row['task_id'][:56]}")
        print("\n[eval] --dry-run のため書いていません")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        # **列は `_build_rows` が決めたものをそのまま使う。**
        # `register.py` は `DictReader` で読むので列が増えても困らない。
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] 書いた: {args.output}（{len(rows)} 件）")

    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "record": str(args.record),
        "used_episodes": len(used),
        "per_task": args.per_task,
        "offset": args.offset,
        "categories": categories,
        "max_demo_steps": args.max_demo_steps,
        "brightness_resolution": _BRIGHTNESS_RESOLUTION,
        "selected": len(rows),
        "instructions_eligible": len(eligible),
        "instructions_dropped": dropped,
        "instructions_without_variants": without_variants,
        "by_instruction": {name: summary[name] for name in eligible},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[eval] 根拠を書いた: {meta_path}")
    print("[eval] 次に: この CSV を compe/t1/T1_TASKS.csv の位置へ渡して評価する"
          "（docs/howto/04_proxy_eval.md）")


if __name__ == "__main__":
    main()
