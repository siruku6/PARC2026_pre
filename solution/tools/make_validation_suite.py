#!/usr/bin/env python3
"""本番に近い「未知タスク」の検証用スイートを組み立てる。

配布された `compe/t1/T1_TASKS.csv` の 4 件は **example** であり、
本番は非公開タスクで採点される（`compe/t1/README.md`）。実際、この 4 件で
成功率 0.713 を出しても本番は 0.141 だった。**手元の評価が本番を予測できない。**

そこで LIBERO-plus の 10,030 タスク変種から、example と重ならないものを
選んで CSV にする。`compe/t1/register.py` は CSV を読んでスイートを組み立てる
だけなので、この CSV に差し替えれば評価パイプラインはそのまま動く。

**7 つの摂動カテゴリを全部入れる。** example 4 件は Background Textures と
Light Conditions しか含まないが、それは 4 サンプルにすぎない。運営 Q&A No.21 の
「7 カテゴリすべてではない」も、どれが出るかは言っていない。

ここで 2 カテゴリに絞ると、**当たっていれば無駄がなく、外れていれば
何も見えない**。全部入れてカテゴリ別に集計すれば、当たっていてもその行を
読めばよく、外れていれば気づける。**どこが弱いかが分かれば学習の的も決まる。**

スイート（spatial / object / goal / 10）も 4 つとも混ぜる。

**Colab のキット環境で実行すること。** LIBERO-plus の実体が要る。

    PYTHONPATH=$KIT_DIR/LIBERO-plus:$KIT_DIR:$KIT_DIR/compe \\
    $KIT_DIR/venv/bin/python make_validation_suite.py \\
        --libero-plus $KIT_DIR/LIBERO-plus --per-cell 3

**初期状態ファイルは、正規表現で接尾辞を剥がすのではなく、実在するファイル名との
最長前方一致で決める。** 摂動の付け方はカテゴリごとに違い（`_table_7`・`_tb_3`・
`_light_15`・`_noise_2`・`_language_25`・`_add_13`・`view_0_0_100_0_0_initstate_1`）、
regex で列挙すると必ず取りこぼす。実在ファイルに当てれば取りこぼしようがない。

Reads:
  - <libero-plus>/libero/libero/benchmark/task_classification.json
  - <bddl-root>/<suite>/<task_id>.bddl（指示文の取得と存在確認）
  - <init-states-root>/<suite>/*.pruned_init（初期状態ファイルの実在確認）
  - compe/t1/T1_TASKS.csv（除外する example）
Generates:
  - compe/val/VAL_TASKS.csv（検証用スイートの定義）
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

# LIBERO-plus の摂動カテゴリ 7 つ。**既定では全部使う。**
# どれが本番に出るかは非公開であり、絞ると外れたときに何も見えなくなる。
_DEFAULT_CATEGORIES = (
    "Background Textures",      # 机や壁のテクスチャ
    "Camera Viewpoints",        # カメラの位置と向き
    "Language Instructions",    # 指示文の言い換え
    "Light Conditions",         # 照明
    "Objects Layout",           # 物体の配置と種類
    "Robot Initial States",     # ロボットの初期姿勢
    "Sensor Noise",             # 観測ノイズ
)

_LANGUAGE_RE = re.compile(r"\(:language\s+(.+?)\)", re.S)

# 初期状態ファイルの拡張子。`.init` もあるが、評価側は pruned を読む。
_INIT_SUFFIX = ".pruned_init"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--libero-plus", type=Path, required=True,
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--exclude", type=Path,
                        default=Path("compe/t1/T1_TASKS.csv"),
                        help="除外する example の CSV")
    parser.add_argument("--output", type=Path,
                        default=Path("compe/val/VAL_TASKS.csv"),
                        help="出力する CSV")
    parser.add_argument("--per-cell", type=int, default=3,
                        help="スイート × カテゴリ 1 組あたりの本数")
    parser.add_argument("--categories", type=str, nargs="*",
                        default=list(_DEFAULT_CATEGORIES),
                        help="対象の摂動カテゴリ")
    parser.add_argument("--init-states-root", type=Path, default=None,
                        help="初期状態ファイルの置き場。省略時は libero から取得する")
    parser.add_argument("--bddl-root", type=Path, default=None,
                        help="bddl の置き場。省略時は libero から取得する")
    return parser.parse_args()


def _load_classification(libero_plus: Path) -> list[dict]:
    """タスク変種の分類表を読み、スイート名を各件に持たせて返す。

    実物（revision `main`）は **スイート名を鍵にした辞書**で、値が変種の列である。

        {"libero_spatial": [{"id": 1, "name": "...", "category": "...",
                             "difficulty_level": 2}, ...], ...}

    **変種そのものには `suite` 鍵が無い。** スイートは外側の鍵にしか現れないため、
    ここで各件へ写しておく。これを取り違えると `category` にスイート名が入り、
    摂動カテゴリでの絞り込みが 1 件も通らなくなる（実際に起きた）。

    Returns
    -------
    entries : list[dict]
        `id` / `name` / `category` / `difficulty_level` / `suite` を持つ辞書の列。
    """
    path = libero_plus / "libero/libero/benchmark/task_classification.json"
    if not path.is_file():
        raise FileNotFoundError(f"分類表が見つかりません: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict] = []

    if isinstance(raw, dict):
        for suite, records in raw.items():
            if not isinstance(records, list):
                raise TypeError(
                    f"分類表の '{suite}' の値が列ではありません: {type(records).__name__}"
                )
            for record in records:
                if not isinstance(record, dict):
                    raise TypeError(
                        f"分類表の '{suite}' の要素が辞書ではありません: {record!r}"
                    )
                entries.append({**record, "suite": record.get("suite", suite)})
    elif isinstance(raw, list):
        # 版が変わって平坦な列になった場合。スイートは各件が持っているはず。
        for record in raw:
            suite = record.get("suite") or record.get("problem_folder")
            if not suite:
                raise KeyError(
                    f"平坦な分類表だがスイートが分かりません。鍵: {sorted(record)}"
                )
            entries.append({**record, "suite": suite})
    else:
        raise TypeError(f"分類表の形が想定外です: {type(raw).__name__}")

    if not entries:
        raise RuntimeError(f"分類表が空です: {path}")

    missing = [key for key in ("name", "category") if key not in entries[0]]
    if missing:
        raise KeyError(
            f"分類表に {missing} がありません。実際の鍵: {sorted(entries[0])}"
        )
    return entries


def _resolve_libero_path(explicit: Path | None, key: str) -> Path:
    """置き場を決める。省略されたら libero の設定（`~/.libero/config.yaml`）に従う。

    **評価パイプラインが読むのと同じ場所を見る**ことが目的である。
    ここが食い違うと、CSV は作れたのに評価の開始直後に落ちる。
    """
    if explicit is not None:
        return explicit
    from libero.libero import get_libero_path
    return Path(get_libero_path(key))


def _index_init_state_stems(init_states_root: Path) -> dict[str, list[str]]:
    """スイートごとに、実在する初期状態ファイルの名前（拡張子なし）を集める。

    長い順に並べて返す。**最長前方一致で引くため**、この順序に意味がある。

    Returns
    -------
    stems_by_suite : dict[str, list[str]]
        スイート名 → 名前の列（長い順）。
    """
    stems_by_suite: dict[str, list[str]] = {}
    for suite_dir in sorted(p for p in init_states_root.iterdir() if p.is_dir()):
        stems = [
            path.name[: -len(_INIT_SUFFIX)]
            for path in suite_dir.iterdir()
            if path.name.endswith(_INIT_SUFFIX)
        ]
        stems_by_suite[suite_dir.name] = sorted(stems, key=len, reverse=True)
    return stems_by_suite


def _resolve_init_state_stem(
    task_id: str, suite: str, stems_by_suite: dict[str, list[str]]
) -> str | None:
    """変種名に対応する初期状態ファイルの名前を、最長前方一致で決める。

    LIBERO-plus は変種の bddl を大量に持つが、**初期状態ファイルは元タスク分しか
    無い**（`libero_spatial` なら 10 件）。`register.py` は正規表現で接尾辞を
    剥がしているが、摂動の付け方はカテゴリごとに違うため列挙しきれない。
    実在する名前に当てるほうが確実である。

    Parameters
    ----------
    task_id : str
        変種のタスク名（例 `pick_up_..._on_the_plate_view_0_0_100_0_0_initstate_1`）。
    suite : str
        元のスイート名。
    stems_by_suite : dict[str, list[str]]
        `_index_init_state_stems` の戻り値。

    Returns
    -------
    stem : str | None
        対応する初期状態ファイルの名前。見つからなければ None。
    """
    for stem in stems_by_suite.get(suite, ()):
        if task_id == stem:
            return stem
        # 途中で切れた別タスクを拾わないよう、続きが区切りであることを要求する。
        if task_id.startswith(stem) and task_id[len(stem)] in "_ ":
            return stem
    return None


def _load_excluded(path: Path) -> set[str]:
    """example のタスク ID を読む。検証用スイートから外すため。"""
    if not path.is_file():
        return set()
    with path.open(newline="") as stream:
        return {row["task_id"] for row in csv.DictReader(stream)}


def _spread(items: list[dict], count: int) -> list[dict]:
    """並びの両端を含めて等間隔に count 件選ぶ。

    乱数を使わないので、**何度実行しても同じスイートになる。**
    条件を変えて比べるときに母集団が動くと比較にならない。
    """
    if count >= len(items):
        return list(items)
    if count <= 1:
        return [items[len(items) // 2]]
    positions = [round(rank * (len(items) - 1) / (count - 1)) for rank in range(count)]
    return [items[position] for position in positions]


def _check_categories(entries: list[dict], categories: list[str]) -> None:
    """求めたカテゴリが分類表に実在するかを先に確かめる。

    **合致が 0 件なら、その場で名前の一覧を見せて落とす。** 後段まで進んで
    「1 件も選べませんでした」とだけ言われると、原因が分類表の読み方なのか
    ファイルの不在なのか区別できない。
    """
    present = collections.Counter(entry["category"] for entry in entries)
    unknown = [name for name in categories if name not in present]
    if len(unknown) == len(categories):
        raise RuntimeError(
            "指定したカテゴリが分類表に 1 つもありません。\n"
            f"  指定: {categories}\n"
            f"  分類表にあるもの: {dict(present)}"
        )
    if unknown:
        print(f"[val] 分類表に無いカテゴリ（無視する）: {unknown}", file=sys.stderr)


def _select(
    entries: list[dict], categories: list[str], excluded: set[str], per_cell: int
) -> list[dict]:
    """スイート × カテゴリの組ごとに、等間隔で選ぶ。"""
    cells: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        if entry["category"] not in categories or entry["name"] in excluded:
            continue
        cells.setdefault((entry["suite"], entry["category"]), []).append(entry)

    selected: list[dict] = []
    for key in sorted(cells):
        group = sorted(cells[key], key=lambda item: item["name"])
        selected.extend(_spread(group, per_cell))
    return selected


def _instruction_of(entry: dict, bddl: Path) -> str | None:
    """指示文を得る。分類表に無ければ bddl の `:language` を読む。"""
    for key in ("instruction", "language", "goal"):
        if entry.get(key):
            return str(entry[key]).strip().strip('"')

    found = _LANGUAGE_RE.search(bddl.read_text(encoding="utf-8"))
    if not found:
        return None
    return " ".join(found.group(1).split()).strip('"')


def _build_rows(
    selected: list[dict],
    bddl_root: Path,
    stems_by_suite: dict[str, list[str]],
) -> tuple[list[dict], list[str]]:
    """CSV の行を作り、実体が無いものは落とす。

    **評価パイプラインが実際に開くパスをそのまま確かめる。** bddl は
    `<bddl_root>/<suite>/<task_id>.bddl` に無ければならない（`register.py` が
    `problem_folder` にスイート名を入れるため）。別のスイートの下から
    同名ファイルを拾ってしまうと、CSV は通るのに評価の開始直後に落ちる。

    Returns
    -------
    rows : list[dict]
        `T1_TASKS.csv` と同じ列 ＋ `init_states_file`。
    skipped : list[str]
        落とした理由つきのタスク名。**黙って減らさない。**
    """
    rows: list[dict] = []
    skipped: list[str] = []

    for entry in selected:
        task_id = entry["name"]
        suite = entry["suite"]

        bddl = bddl_root / suite / f"{task_id}.bddl"
        if not bddl.is_file():
            skipped.append(f"{task_id}: bddl が無い（{bddl}）")
            continue

        instruction = _instruction_of(entry, bddl)
        if not instruction:
            skipped.append(f"{task_id}: 指示文を取れない")
            continue

        stem = _resolve_init_state_stem(task_id, suite, stems_by_suite)
        if stem is None:
            skipped.append(f"{task_id}: 初期状態ファイルが無い（{suite}/）")
            continue

        rows.append({
            "task_num": len(rows) + 1,
            "task_id": task_id,
            "instruction": instruction,
            "libero_plus_id": entry.get("id", ""),
            "suite": suite,
            "category": entry["category"],
            "difficulty_level": entry.get("difficulty_level", ""),
            # **接尾辞を剥がした結果をここに書き出す。** 評価側で regex を
            # 当て直させると、カテゴリごとに違う摂動の付け方を取りこぼす。
            "init_states_file": f"{stem}{_INIT_SUFFIX}",
        })
    return rows, skipped


def _print_population(entries: list[dict], excluded: set[str]) -> None:
    """母集団の内訳を先に見せる。選べなかったときの手がかりになる。"""
    counts = collections.Counter(
        (entry["suite"], entry["category"])
        for entry in entries if entry["name"] not in excluded
    )
    print(f"[val] 母集団: {sum(counts.values())} 件（スイート × カテゴリ {len(counts)} 組）")
    for (suite, category), count in sorted(counts.items()):
        print(f"       {suite:16s} {category:22s} {count}")


def _report(rows: list[dict], skipped: list[str]) -> None:
    """選んだ結果を表で見せる。"""
    print(f"\n選んだタスク変種: {len(rows)} 件")
    counts = collections.Counter((row["suite"], row["category"]) for row in rows)
    for (suite, category), count in sorted(counts.items()):
        print(f"  {suite:16s} {category:22s} {count}")

    if skipped:
        print(f"\n**落としたもの: {len(skipped)} 件**")
        for reason in skipped[:20]:
            print(f"  {reason}")
        if len(skipped) > 20:
            print(f"  ... 他 {len(skipped) - 20} 件")


def main() -> None:
    args = _parse_args()

    init_states_root = _resolve_libero_path(args.init_states_root, "init_states")
    bddl_root = _resolve_libero_path(args.bddl_root, "bddl_files")
    print(f"[val] 初期状態: {init_states_root}")
    print(f"[val] bddl:     {bddl_root}")
    for root in (init_states_root, bddl_root):
        if not root.is_dir():
            raise FileNotFoundError(f"置き場が見つかりません: {root}")

    stems_by_suite = _index_init_state_stems(init_states_root)
    print("[val] 初期状態ファイル: "
          + ", ".join(f"{suite} {len(stems)} 件"
                      for suite, stems in sorted(stems_by_suite.items())))

    entries = _load_classification(args.libero_plus)
    print(f"[val] 分類表: {len(entries)} 件")

    excluded = _load_excluded(args.exclude)
    print(f"[val] 除外する example: {len(excluded)} 件")

    _check_categories(entries, args.categories)
    _print_population(entries, excluded)

    selected = _select(entries, args.categories, excluded, args.per_cell)
    print(f"[val] 候補: {len(selected)} 件")

    rows, skipped = _build_rows(selected, bddl_root, stems_by_suite)
    if not rows:
        _report(rows, skipped)
        raise RuntimeError(
            "1 件も CSV に載せられませんでした。**上の「落としたもの」が理由である。**"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _report(rows, skipped)
    print(f"\n書き出した: {args.output}")
    print(f"1 タスクあたり 1 エピソードなら計 {len(rows)} エピソード")


if __name__ == "__main__":
    main()
