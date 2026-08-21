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
import itertools
import json
import re
import sys
from pathlib import Path

# 素のタスク名の取り出しは `make_holdout_split` と共通にする。
# **同じ正規表現を 2 か所に置くと必ず片方が古くなる。**
# 兄弟の台本を読む書き方は `check_validation_suite.py` と同じである。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_holdout_split import base_task, instruction  # noqa: E402

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

# bddl の物体一覧と、操作対象の宣言。**衝突しうる相手を数えるのに使う。**
_OBJECTS_RE = re.compile(r"\(:objects\s+(.*?)\n\s*\)", re.S)
_OBJ_OF_INTEREST_RE = re.compile(r"\(:obj_of_interest\s+(.*?)\)", re.S)

# 初期状態ファイルの拡張子。`.init` もあるが、評価側は pruned を読む。
_INIT_SUFFIX = ".pruned_init"

# 物体を増やす摂動の初期状態だけが置かれているディレクトリ。中がもう 1 段
# （`libero_newobj/libero_goal/...`）で、他のスイートとは構造が違う。
_NEWOBJ_DIR = "libero_newobj"

# 物体が増える摂動の目印。**元タスクの初期状態では自由度が合わない。**
_NEWOBJ_MARKERS = ("_add_", "_level")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--libero-plus", type=Path, required=True,
                        help="LIBERO-plus の clone 先")
    parser.add_argument("--exclude", type=Path,
                        default=Path("compe/t1/T1_TASKS.csv"),
                        help="除外する example の CSV")
    # **検証集合を「学習から外したタスク」だけに絞る。**
    # 摂動版は元のタスクの実演を共有するため、学習に使ったタスクの摂動版で
    # 測ると「覚えた度合い」になる。**学習側と同じ CSV を読むことで、
    # 片方だけ古くなる事故を防ぐ**（docs/howto/04_proxy_eval.md）。
    parser.add_argument("--holdout", type=Path, default=None, metavar="CSV",
                        help="make_holdout_split.py が作った CSV。"
                             "**この instruction 列にあるタスクだけを検証に使う**")
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

    `libero_newobj/` は中がもう 1 段（`libero_newobj/libero_goal/...`）なので
    見ない。**物体を増やす摂動の初期状態はそちらにあり、扱いが違う**ので
    `_resolve_newobj_init` が別に受け持つ。

    Returns
    -------
    stems_by_suite : dict[str, list[str]]
        スイート名 → 名前の列（長い順）。
    """
    stems_by_suite: dict[str, list[str]] = {}
    for suite_dir in sorted(p for p in init_states_root.iterdir() if p.is_dir()):
        if suite_dir.name == _NEWOBJ_DIR:
            continue
        stems = [
            path.name[: -len(_INIT_SUFFIX)]
            for path in suite_dir.iterdir()
            if path.name.endswith(_INIT_SUFFIX)
        ]
        stems_by_suite[suite_dir.name] = sorted(stems, key=len, reverse=True)
    return stems_by_suite


def _resolve_newobj_init(
    task_id: str, suite: str, init_states_root: Path
) -> str | None:
    """物体を増やす摂動（`_add_` / `_level`）の初期状態ファイル名を返す。

    **これらは元タスクの初期状態を使えない。** bddl に物体が増えているぶん
    自由度も増えるので、元タスクの状態を流し込むと大きさが合わずに落ちる。

        ValueError: could not broadcast input array from shape (23,) into shape (49,)

    LIBERO-plus は専用の初期状態を **`init_files/libero_newobj/<suite>/`** に
    2,400 件持っており、本家の `benchmark/__init__.py` も `_add_` と `_level`
    のときだけそこを見ている（`get_task_init_states`）。
    **公式カタログの変種名（`_add_N`単独）には前方一致は要らない**——
    ファイル名がそのまま変種名と一致する。

    **Camera・Robot Initial States・Sensor Noiseと組み合わせた複合摂動名
    （`<add系のベース>_view_..._initstate_..._noise_...`）は、この関数が
    素の`task_id`と完全一致するファイルを探すため、そのままでは見つからない。**
    物体配置の初期状態はカメラ・ロボット姿勢・ノイズの値に依存しないので、
    `_view_`より前（＝Objects Layout部分だけ）で探せば同じ初期状態ファイルを
    再利用できる（`_resolve_bddl`が bddl を`_view_`より前で探すのと同じ理由）。
    そのため`task_id`自身と`_view_`より前の2通りを、長い方から順に試す。

    Returns
    -------
    relative_path : str | None
        初期状態の置き場からの相対パス（`libero_newobj/<suite>/<変種名>.pruned_init`）。
        **`/` を含めることがパッチ側の分岐の目印になる。** 無ければ None。
    """
    candidates = dict.fromkeys((task_id, task_id.split("_view_")[0]))
    for candidate_id in candidates:
        name = f"{candidate_id}{_INIT_SUFFIX}"
        if (init_states_root / _NEWOBJ_DIR / suite / name).is_file():
            return f"{_NEWOBJ_DIR}/{suite}/{name}"
    return None


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


def _resolve_bddl(task_id: str, suite: str, bddl_root: Path) -> Path | None:
    """変種の bddl を返す。**変種専用のファイルが無いのが普通である。**

    LIBERO-plus の 10,030 変種のうち **6,287 件は自分の名前の bddl を持たない。**
    カメラ視点・ロボット初期姿勢・観測ノイズの摂動は、bddl ではなく
    **変種名そのものに数値として埋め込まれている**（`libero/libero/envs/
    env_wrapper.py` の `_view_` / `_initstate_` / `_noise_` の解釈）。

        ..._place_it_on_the_plate_view_0_0_100_2_352_initstate_0
                                  └ 水平 0・垂直 0・倍率 1.00・端点回転 2・端点垂直 352
                                                        └ ロボット初期姿勢 0

    したがって bddl は `_view_` より前で探す。**ここを「ファイルが無い」として
    落とすと、7 カテゴリのうち 4 つが丸ごと消える**（実際に起きた）。

    Returns
    -------
    bddl : Path | None
        実体のある bddl。見つからなければ None。
    """
    # `_view_` が無ければ候補は 1 つ。同じ名前を 2 度 stat しないよう畳む。
    stems = dict.fromkeys((task_id, task_id.split("_view_")[0]))
    for stem in stems:
        candidate = bddl_root / suite / f"{stem}.bddl"
        if candidate.is_file():
            return candidate
    return None


def _place_variant(
    task_id: str,
    suite: str,
    bddl_root: Path,
    init_states_root: Path,
    stems_by_suite: dict[str, list[str]],
) -> tuple[str, Path, str] | None:
    """変種の bddl と初期状態がそろって存在する置き場を決める。

    `register.py` は `problem_folder` **1 つ**で bddl と初期状態の両方を探す
    （`bddl_files/<folder>/<...>.bddl` と `init_files/<folder>/<init_states_file>`）。
    したがって**両方が同じ置き場にそろっていなければ、評価の開始直後に落ちる。**

    物体が増える摂動（`_add_` / `_level`）だけは初期状態が
    `init_files/libero_newobj/<folder>/` にある。**元タスクの初期状態を
    当ててはいけない**（自由度が合わず broadcast で落ちる）。

    Returns
    -------
    placed : tuple[str, Path, str] | None
        (置き場の名前, bddl のパス, 初期状態ファイルの名前（拡張子つき）)。
        そろう置き場が無ければ None。
    """
    needs_newobj = any(marker in task_id for marker in _NEWOBJ_MARKERS)
    candidates = [suite] + [name for name in stems_by_suite if name != suite]
    for folder in candidates:
        bddl = _resolve_bddl(task_id, folder, bddl_root)
        if bddl is None:
            continue
        if needs_newobj:
            name = _resolve_newobj_init(task_id, folder, init_states_root)
            if name is not None:
                return folder, bddl, name
            continue
        stem = _resolve_init_state_stem(task_id, folder, stems_by_suite)
        if stem is not None:
            return folder, bddl, f"{stem}{_INIT_SUFFIX}"
    return None


def _keep_holdout_only(entries: list[dict], path: Path) -> list[dict]:
    """学習から外したタスクの摂動版だけを残す。

    Parameters
    ----------
    entries : list[dict]
        分類表の全件。
    path : Path
        `instruction` 列を持つ CSV。

    Returns
    -------
    kept : list[dict]
        指示文が一致した件だけ。

    Raises
    ------
    SystemExit
        CSV が読めないとき、または 1 件も残らなかったとき。
        **黙って 0 件を返すと、空の検証集合で学習の可否を決めてしまう。**
    """
    if not path.is_file():
        raise SystemExit(f"[val] 除外リストが見つかりません: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "instruction" not in reader.fieldnames:
            raise SystemExit(
                f"[val] 除外リストに instruction 列がありません: {path}"
                f"（見出し: {reader.fieldnames}）"
            )
        wanted = {row["instruction"].strip() for row in reader if row["instruction"].strip()}
    if not wanted:
        raise SystemExit(f"[val] 除外リストが空です: {path}")

    kept = [
        entry for entry in entries
        if instruction(base_task(entry["name"])) in wanted
    ]
    print(f"[val] 検証に使うのは外した {len(wanted)} タスクの摂動版だけ: "
          f"{len(entries)} 件 -> {len(kept)} 件")
    if not kept:
        raise SystemExit(
            f"[val] 外したタスクの摂動版が 1 件も見つかりません: {path}\n"
            "  **make_holdout_split.py の導出と分類表の版が合っていない。**"
        )
    return kept


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

    by_category: dict[str, list[dict]] = {}
    for suite, category in sorted(cells):
        group = sorted(cells[(suite, category)], key=lambda item: item["name"])
        by_category.setdefault(category, []).extend(_spread(group, per_cell))

    # **カテゴリを 1 件ずつ順番に取り出して並べる。**
    # 評価パイプラインの `--max-tasks N` は先頭 N 件を取るだけなので
    # （`pipeline.py` の `task_infos[:max_tasks]`）、セル単位に固めて並べると
    # 「5 件だけ試す」が 1 カテゴリしか踏まない。**疎通確認の意味が薄れる。**
    # カテゴリを巡回させておけば、先頭 7 件で 7 カテゴリ全部に当たる。
    return [
        entry
        for column in itertools.zip_longest(
            *(by_category[name] for name in sorted(by_category))
        )
        for entry in column
        if entry is not None
    ]


def _count_nontarget_objects(bddl: Path) -> int | None:
    """操作対象**以外**の物体が bddl にいくつ書かれているかを返す。

    **衝突で失格になりうる相手の数である。** 採点は「操作対象以外の物体が
    初期位置から 1mm を超えて動いたら失格」なので（`pipeline/rollout.py`）、
    この数が 0 のタスクは**衝突では落ちない。** 逆に 5 個あるタスクは、
    同じ方策でも失格の機会がそれだけ多い。**衝突率をタスク構成の差と
    切り分けるために、CSV へ書いておく。**

    LIBERO-plus の Track 1 候補 2,218 件で数えると平均 3.10 個・最多 5 個で、
    **55 件は 0 個**である（`docs/reference/04_track1_population.md`）。

    > **採点が見ているものと厳密には一致しない。** ハーネスは観測の
    > `*_pos` という鍵を数えるので、bddl の物体名とずれることがある
    > （棚などの什器も含まれる）。**目安として使う。**

    Returns
    -------
    count : int | None
        物体数。bddl を読めない・節が無いときは None。
    """
    text = bddl.read_text(encoding="utf-8")
    objects = _OBJECTS_RE.search(text)
    targets = _OBJ_OF_INTEREST_RE.search(text)
    if not objects or not targets:
        return None

    # `名前1 名前2 - 型` の形で並ぶ。型は数えない。
    names = [
        name
        for line in objects.group(1).splitlines()
        for name in line.strip().split(" - ")[0].split()
    ]
    of_interest = set(targets.group(1).split())
    return sum(1 for name in names if name not in of_interest)


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
    init_states_root: Path,
    stems_by_suite: dict[str, list[str]],
) -> tuple[list[dict], list[dict]]:
    """CSV の行を作り、実体が無いものは落とす。

    **`task_id` は一切書き換えずに CSV へ入れる。** 摂動の値は変種名そのものに
    埋め込まれており、`env_wrapper.py` が `_view_` で切って解釈する。
    ここで実体のある名前（`_view_` より前）に置き換えてしまうと、
    **ファイルは開けるのに摂動がすべて無効になり、素のタスクを測ってしまう。**
    `bddl` は存在確認と指示文の読み取りにだけ使う。

    Returns
    -------
    rows : list[dict]
        `T1_TASKS.csv` と同じ列 ＋ `init_states_file`。
    skipped : list[dict]
        `category` / `reason` / `detail` を持つ辞書の列。**黙って減らさない。**
    """
    rows: list[dict] = []
    skipped: list[dict] = []

    def drop(entry: dict, reason: str, detail: str) -> None:
        skipped.append(
            {"category": entry["category"], "reason": reason, "detail": detail}
        )

    for entry in selected:
        task_id = entry["name"]
        placed = _place_variant(
            task_id, entry["suite"], bddl_root, init_states_root, stems_by_suite
        )
        if placed is None:
            drop(
                entry, "bddl と初期状態がそろう置き場が無い",
                f"{task_id} → 分類表のスイートは {entry['suite']}",
            )
            continue
        folder, bddl, init_states_file = placed

        instruction = _instruction_of(entry, bddl)
        if not instruction:
            drop(entry, "指示文を取れない", f"{task_id} → {bddl}")
            continue

        rows.append({
            "task_num": len(rows) + 1,
            "task_id": task_id,
            "instruction": instruction,
            "libero_plus_id": entry.get("id", ""),
            # **bddl が実際に在った置き場を書く。** register.py はこの値を
            # problem_folder にして bddl と初期状態の**両方**を探すため、
            # 分類表のスイート名とは違うことがある。
            "suite": folder,
            "category": entry["category"],
            "difficulty_level": entry.get("difficulty_level", ""),
            # **どのファイルを開くかをここで決めきる。** 評価側で regex を
            # 当て直させると、カテゴリごとに違う摂動の付け方を取りこぼす。
            #
            # 物体が増える摂動だけ `libero_newobj/<suite>/<変種名>.pruned_init`
            # という**相対パス**を書く。`/` の有無がパッチ側の分岐になっており、
            # `problem_folder` には畳めない（あちらは bddl の置き場でもある）。
            # それ以外は接尾辞を剥がした元タスク名のファイル名だけを書く。
            "init_states_file": init_states_file,
            # **衝突で失格になりうる相手の数。** 集計側が衝突率をタスク構成と
            # 切り分けるのに使う。`register.py` は DictReader で読むので、
            # 列が増えても評価には影響しない。
            "n_nontarget_objects": _count_nontarget_objects(bddl),
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


def _report(rows: list[dict], skipped: list[dict]) -> None:
    """選んだ結果と落とした理由を表で見せる。

    落としたものは**カテゴリ × 理由で束ねて数える。** 1 件ずつ並べると数十行になり、
    ノートブックが末尾しか表示しない設定だと**全部流れて消える**（実際に起きた）。
    """
    print(f"\n選んだタスク変種: {len(rows)} 件")
    counts = collections.Counter((row["suite"], row["category"]) for row in rows)
    for (suite, category), count in sorted(counts.items()):
        print(f"  {suite:16s} {category:22s} {count}")

    # **衝突で落ちうるかどうかは、タスクによって違う。** 0 個のタスクは
    # 衝突では失格しないので、衝突率を読むときに分母から外して考える。
    exposures = [
        row["n_nontarget_objects"] for row in rows
        if row["n_nontarget_objects"] is not None
    ]
    if exposures:
        print(f"\n操作対象以外の物体（衝突しうる相手）: "
              f"平均 {sum(exposures) / len(exposures):.2f} / 最多 {max(exposures)}")
        zeros = exposures.count(0)
        print(f"  0 個（衝突では落ちない）: {zeros} 件 / {len(exposures)}")

    if skipped:
        print(f"\n落としたもの: {len(skipped)} 件（カテゴリ × 理由）")
        grouped = collections.Counter(
            (item["category"], item["reason"]) for item in skipped
        )
        for (category, reason), count in sorted(
            grouped.items(), key=lambda pair: -pair[1]
        ):
            print(f"  {count:4d}  {category:22s} {reason}")
        print("\n  例:")
        for item in skipped[:3]:
            print(f"    {item['detail']}")


def _report_coverage(
    rows: list[dict], categories: list[str], per_cell: int, suites: int
) -> list[str]:
    """カテゴリごとに何本取れたかを出し、全滅したカテゴリの名前を返す。

    **この表を最後に出す。** ノートブックの `run()` は成功時に末尾の数行しか
    表示しないため、いちばん大事な情報を最後に置く必要がある。

    Returns
    -------
    empty : list[str]
        1 本も取れなかったカテゴリ。**測定の目的が崩れているので呼び出し側で警告する。**
    """
    counts = collections.Counter(row["category"] for row in rows)
    print(f"\n=== カテゴリ別の本数（目標 各 {per_cell * suites} 本）===")
    empty: list[str] = []
    for category in categories:
        count = counts.get(category, 0)
        if count == 0:
            empty.append(category)
            print(f"  {category:22s} {count:4d}  ← **全滅**")
        else:
            print(f"  {category:22s} {count:4d}")
    return empty


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

    if args.holdout:
        entries = _keep_holdout_only(entries, args.holdout)

    excluded = _load_excluded(args.exclude)
    print(f"[val] 除外する example: {len(excluded)} 件")

    _check_categories(entries, args.categories)
    _print_population(entries, excluded)

    selected = _select(entries, args.categories, excluded, args.per_cell)
    print(f"[val] 候補: {len(selected)} 件")

    rows, skipped = _build_rows(
        selected, bddl_root, init_states_root, stems_by_suite
    )
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

    # **カテゴリ別の本数を最後に置く。** ノートブックは成功時に末尾しか
    # 表示しないため、いちばん大事な行が流れて消えることを防ぐ。
    suites = len({row["suite"] for row in rows})
    empty = _report_coverage(rows, args.categories, args.per_cell, suites)
    if empty:
        print(
            f"\n**警告: {len(empty)} カテゴリが全滅している。**\n"
            f"  {empty}\n"
            "  この状態で測ると、そのカテゴリの弱さは**見えない**。\n"
            "  上の「落としたもの」の理由を見て、置き場の対応を直すこと。"
        )


if __name__ == "__main__":
    main()
