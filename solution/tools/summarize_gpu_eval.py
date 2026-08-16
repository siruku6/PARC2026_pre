#!/usr/bin/env python3
"""GPU マシンで流した検証用スイートの結果を、判断に使える形へ集計する。

**Colab の F セルがやっていたことを、マシン上で完結させる。**
Drive へ退避する必要がないので、退避漏れで結果を失う経路そのものが無い。

出すのは 8 つ。**どれも「次に何をするか」が変わる情報である。**

- **84 件そろったか** — 途中で落ちると終了コード 0 のまま件数だけ減る
- 結末の内訳（成功 / 衝突 / 未完了）と、**ゴールに着いてから衝突で落ちた本数**
- **ゴール到達率と成功率の差** — ここが開いていれば衝突対策、閉じていれば到達の話
- **動きの内訳（静止 / 往復）** — 全件について。**1 件も動いていなければ最初に警告する**
- **摂動カテゴリ別の成功率**（背景テクスチャは `table` / `tb` に分ける） — 学習の的を決める表
- **衝突を「方策由来」と「環境由来」に割る** — 衝突対策に手を入れるかを決める
- ぶつけた物体の偏りと、**初めて 1mm を超えた歩**
- **2 回の評価を並べた差** — `--compare` を渡したとき

    solution/tools/summarize_gpu_eval.py --output-dir work/gpu_eval/0811_...

    # 素の重みと学習後を並べる
    solution/tools/summarize_gpu_eval.py \
        --output-dir work/gpu_eval/0811_..._lora --compare work/gpu_eval/0811_..._base

**古いログでも動く。** ゴール到達・動き・初衝突は 2026-08-11 のパッチから
出るようになったもので、無ければその節を飛ばす（黙って 0 と読まない）。

**本番のスコアは手元で計算できない。** 配点は `total_score_config.json` と
`normalization_config.json` にあるが、**どちらも配布されていない**
（`pipeline/total_score.py` の内蔵既定は `weights` も `references` も空）。
キットの `overall_score` は成功率そのものである（`pipeline/scorer.py`）。
**だから比べられるのは成功率と滑らかさの指標までで、順位の予測はできない。**

Reads:
  - <output-dir>/eval.log
  - <output-dir>/<submission_id>.json（タスク単位の指標。jerk / SPARC）
  - <output-dir>/../../compe/val/VAL_TASKS.csv（あればカテゴリを引く）
Generates:
  - <output-dir>/summary.txt（画面と同じ内容）
  - 標準出力。**結果が 1 件も無ければ終了コード 1**
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import unicodedata
from pathlib import Path

# `[結果] <タスク> ep<N>: <結末> <ステップ数> steps  ゴール 到達  経路 1.12m …
_RESULT_RE = re.compile(
    r"\[結果\] (?P<task>\S+) ep(?P<episode>\d+): "
    r"(?P<outcome>\S+) (?P<steps>\d+) steps(?P<detail>.*)"
)
_OFFENDER_RE = re.compile(r"(\S+?)=([\d.]+)mm")
_EXCEPTION_RE = re.compile(r"\[結果\] (\S+): 例外で測れず \((.+?)\)")
_TASK_TOTAL_RE = re.compile(r"=== タスク \d+/(\d+):")

# `[結果]` 行の後半にある項目。**無い版のログもあるので、すべて任意とする。**
_GOAL_RE = re.compile(r"ゴール (到達|未達)")
_PATH_RE = re.compile(r"経路 ([\d.]+)m")
_DISPLACEMENT_RE = re.compile(r"変位 ([\d.]+)m")
# **`固まった` を必ず入れる。** 8/11 に入れ忘れていて、`eval.log` に 8 件
# あったラベルを「ラベル無し」に数えていた。**打ち切りの中身を読む唯一の
# ラベルなので、落とすと未完了 22 件の正体が分からなくなる。**
_MOTION_RE = re.compile(r"\s(静止|往復|固まった)(?:\s|$)")
_FIRST_COLLISION_RE = re.compile(r"初衝突 (\d+) 歩")

# 終盤 50 ステップの経路長。**全体の変位では「動いてから固まった」を捉えられない。**
_TAIL_PATH_RE = re.compile(r"終盤\d+歩 ([\d.]+)m")

# グリッパを何回開閉したか。**本番の悪化を検知できた唯一の指標である。**
# 素の重み 11/84 (13%) 対 lora20k 23/84 (27%) で、20 回以上だった 34 件の
# うち成功は 2 件（0.06）だった。成功した回の中央は 2 回である。
_GRIP_RE = re.compile(r"握り x(\d+)")

# これ以上開閉していたら「掴み直しで空回りしている」と数える。
_GRIP_CHATTER_THRESHOLD = 20

# 変種名の**末尾**の接尾辞 → 摂動カテゴリ。CSV が読めればそちらを使う。
#
# **末尾で見る。** `_table_` を含むかどうかで見ると、
# `pick_up_the_black_bowl_from_table_center_..._tb_1` のような名前を
# `table` 側に数えてしまう（LIBERO-plus の 2,218 件のうち 23 件が該当し、
# `table` 613 / `tb` 463 が 636 / 440 にずれる）。
_TAIL_SUFFIX_RE = re.compile(r"_(light|table|tb|noise|add|initstate)_\d+$")
_SUFFIX_CATEGORIES: dict[str, str] = {
    "light": "Light Conditions",
    "table": "Background Textures",
    "tb": "Background Textures",
    "noise": "Sensor Noise",
    "add": "Objects Layout",
}

# **名前だけでは分けられない 3 カテゴリ。** 視点・指示文・初期姿勢の摂動は
# 名前がどれも `..._view_<数値>_initstate_<数値>` の形で、**同じ形のまま
# 別のカテゴリに属する**（分類表を数えた実測：Camera Viewpoints 1,599 件・
# Language Instructions 1,537 件・Robot Initial States 1,550 件がすべてこの形）。
# 指示文の摂動は名前ではなく CSV の instruction 側に出るため、
# **名前から当てることは原理的にできない。**
#
# ここを「とりあえず Camera Viewpoints」と埋めると、**学習の的を決める表が
# 静かに間違う。** 分からないと言わせる。
_AMBIGUOUS_CATEGORY = "不明（視点・指示文・初期姿勢のどれか）"

# 末尾に数字が付かない摂動の目印。
_CATEGORY_MARKERS: tuple[tuple[str, str], ...] = (
    ("_level", "Objects Layout"),
)

# 環境側で最初から動いている物体を見分ける回数。
#
# **同じ物体が同じ変位で 3 回以上出たら環境由来と見る。** 摂動が違う回で
# 変位が 0.1mm も違わないのは方策の振る舞いではない（2026-08-10 に
# `ketchup_1=544.6mm` が 6 回そろって出た）。
#
# 原因は採点ハーネス側にある。`pipeline/rollout.py` は初期位置を捉える前に
# **10 ステップしか落ち着かせない**ため、**まだ落下中の物体は空中の位置が
# 「初期位置」として記録される。** 残りの落下がそのまま変位に化ける。
# 映像でも、ケチャップの瓶が 3 本すべて **4 ステップ目**に画面の上から入ってきた。
#
# **3 回にした理由。** 2 回では偶然の一致がありうる（変位を 0.1mm に丸めて
# 比べているため）。4 回以上にすると、6 回出た実例のような明らかな組しか
# 拾えなくなる。**手元の 84 件では 1 タスクあたり最大 3 本しか回らない**
# （`--per-cell 3`）ので、3 回は「同じタスクの全エピソードで揃った」に当たる。
_ENVIRONMENT_REPEAT_MINIMUM = 3

# 回数に頼らずに環境由来と決める条件（下の 2 つを同時に満たすとき）。
#
# **回数だけの判定は 1 タスク 1 本の一覧では発火しない。** 36 件の一覧
# （`compe/val/EVAL_TASKS.csv`）は 1 タスク 1 本しか回さないため、
# 同じ変位が 3 回そろうには**別のタスク 3 件**が同じ落下物を持つ必要がある。
# 実際には LIVING_ROOM_SCENE2 の 2 件しか無く、`ketchup_1=544.6mm` が
# 2 回どまりで見逃された（2026-08-14。π0.5 n=10 / n=16 / weak3x の
# **3 つの別モデルで 0.1mm も違わず同じ値**が出ていた）。
#
# **1 ステップで 10cm 動かすのは方策には物理的に無理である。** 制御周期は 0.05 秒
# （`pipeline/total_score.py` の `control_dt`）で、手先が 1 ステップに進む距離は
# 成功した回で平均 5mm 前後（`rollout_recorder.py` の `_TAIL_STALL_M` の根拠）。
# 1〜2 ステップ目で 100mm 超の変位が付いた物体は、**落下の続きが変位に化けたもの**
# と断定できる。**回数の一致を待たずに落とせる。**
_IMPOSSIBLE_COLLISION_STEPS = 2
_IMPOSSIBLE_DISPLACEMENT_MM = 100.0

# 早すぎる衝突を疑うステップ数。**判定には使わず、報告だけする。**
# 方策が本当に早く衝突することもあるので、これだけでは決められない。
_EARLY_COLLISION_STEPS = 20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="run_gpu_eval.sh の出力先")
    parser.add_argument("--csv", type=Path, default=None,
                        help="検証用スイートの CSV。カテゴリを引くのに使う")
    parser.add_argument("--compare", type=Path, default=None,
                        help="**もう 1 つの出力先。並べて差を出す**"
                             "（素の重みと学習後を比べるときに使う）")
    parser.add_argument("--only-categories", type=str, nargs="*", default=None,
                        metavar="CATEGORY",
                        help="**この摂動カテゴリだけで集計し直す。**"
                             '例: --only-categories "Light Conditions" "Background Textures"'
                             "（出題の見込みに近い 24 件だけで比べるときに使う）")
    return parser.parse_args()


def _pad(text: str, width: int) -> str:
    """表示桁をそろえて左詰めする。**全角の文字を 2 桁で数える。**

    `f"{text:<28}"` は文字数で詰めるため、日本語の見出しを混ぜると表がずれる。
    「項目」は 2 文字だが端末では 4 桁を占める。

    Parameters
    ----------
    text : str
        詰める文字列。
    width : int
        そろえたい表示桁数。足りていればそのまま返す。
    """
    used = sum(
        2 if unicodedata.east_asian_width(character) in "WF" else 1
        for character in text
    )
    return text + " " * max(0, width - used)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """成功率の 95% 信頼区間（Wilson）を返す。

    **単純な ±1.96√(p(1-p)/n) は端で壊れる。** 成功率 0 や 1 のときに
    幅 0 の区間を返すため、「1 本も失敗していないから確実」と読めてしまう。

    Parameters
    ----------
    successes : int
        成功した本数。
    total : int
        全体の本数。

    Returns
    -------
    low, high : float
        下限と上限。`total` が 0 なら (0.0, 0.0)。
    """
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _load_categories(csv_path: Path | None) -> dict[str, str]:
    """CSV からタスク名 → 摂動カテゴリの対応を読む。無ければ空。"""
    if csv_path is None or not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return {
            row["task_id"]: row.get("category", "")
            for row in csv.DictReader(stream)
            if row.get("category")
        }


def _load_nontarget_counts(csv_path: Path | None) -> dict[str, int]:
    """CSV からタスク名 → 操作対象以外の物体数を読む。無ければ空。

    **衝突率をタスク構成と切り分けるために使う。** 0 個のタスクは衝突では
    失格しないので、分母に入れると衝突率が薄まる。
    列は `make_validation_suite.py` が書く（2026-08-11 以降）。
    """
    if csv_path is None or not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return {
            row["task_id"]: int(row["n_nontarget_objects"])
            for row in csv.DictReader(stream)
            if (row.get("n_nontarget_objects") or "").isdigit()
        }


def _category_of(task_name: str, known: dict[str, str]) -> str:
    """摂動カテゴリを決める。CSV を優先し、無ければ変種名から当てる。

    **名前からは 7 カテゴリのうち 4 つしか分からない**（上の
    `_AMBIGUOUS_CATEGORY` の説明）。CSV が読めているかどうかで
    表の意味が変わるので、呼び出し側はそれを画面に出すこと。
    """
    if task_name in known:
        return known[task_name]

    found = _TAIL_SUFFIX_RE.search(task_name)
    if found:
        if found.group(1) == "initstate":
            return _AMBIGUOUS_CATEGORY
        return _SUFFIX_CATEGORIES[found.group(1)]

    for marker, category in _CATEGORY_MARKERS:
        if marker in task_name:
            return category
    return "その他"


def _appearance_suffix(task_name: str) -> str | None:
    """見た目の摂動の接尾辞（`light` / `table` / `tb`）を返す。無ければ None。

    **`table` と `tb` を分けて数えるために要る。** どちらも分類表では
    Background Textures だが、`tb` のほうが明るさの振れ幅が大きい
    （外部の EDA 報告。`docs/reference/04_track1_population.md`）。
    12 本の内訳が偏っていれば、カテゴリ別の成功率の読み方が変わる。
    """
    found = _TAIL_SUFFIX_RE.search(task_name)
    if found and found.group(1) in ("light", "table", "tb"):
        return found.group(1)
    return None


def _filter_by_categories(
    episodes: list[dict], known: dict[str, str], wanted: list[str] | None
) -> list[dict]:
    """指定した摂動カテゴリの回だけを残す。指定が無ければそのまま返す。

    **84 件の平均は本番より低く出る。** 出題の見込みに近いのは
    `Light Conditions` と `Background Textures` の 24 件で、そこだけなら 0.708
    だった（`docs/strategy/05_実測結果.md`）。**判定はその 24 件で行う**ため、
    集計全体を絞れる口を用意する。

    Parameters
    ----------
    episodes : list[dict]
        `_parse_log` の結果。
    known : dict[str, str]
        タスク名 → 摂動カテゴリ。CSV から読んだもの。
    wanted : list[str] | None
        残すカテゴリ名。`None` または空なら絞らない。

    Returns
    -------
    kept : list[dict]
        指定したカテゴリに属する回だけ。
    """
    if not wanted:
        return episodes
    targets = set(wanted)
    return [e for e in episodes if _category_of(e["task"], known) in targets]


def _sign_test_probability(wins: int, losses: int) -> float:
    """勝ち負けの偏りが偶然で説明できるかを、符号検定の両側 p 値で返す。

    **ハーネスは決定的なので、2 回の評価は 1 件ずつ対になっている。**
    対の内訳（片方だけ成功した件数）だけを使う両側の二項検定である。
    引き分け（両方成功・両方失敗）は情報を持たないので数えない。

    Parameters
    ----------
    wins : int
        こちらだけが成功した件数。
    losses : int
        あちらだけが成功した件数。

    Returns
    -------
    probability : float
        両側 p 値。対が 0 件なら 1.0。
    """
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _report_pairing(lines: list[str], mine: list[dict], theirs: list[dict],
                    known: dict[str, str]) -> None:
    """同じ 84 件を 1 件ずつ突き合わせ、入れ替わった数を出す。

    **平均を見比べるだけでは、ほぼ何も判定できない。** 84 件での成功率の
    95%CI は ±0.10 あり、24 件なら ±0.19 になる。
    **一方でハーネスは完全に決定的である**（同じ初期状態・同じ乱数）。
    **対にして数えれば、平均の差より桁違いに鋭い。**

    平均が同じでも中身が入れ替わっていれば別物であり、
    **「拾えた回」と「失った回」のカテゴリを見れば学習の的が決まる。**

    Parameters
    ----------
    lines : list[str]
        報告文の行。ここへ追記する。
    mine, theirs : list[dict]
        `_parse_log` の結果。`mine` が `--output-dir` の側。
    known : dict[str, str]
        タスク名 → 摂動カテゴリ。
    """
    mine_by_key = {(e["task"], e["episode"]): e for e in mine}
    theirs_by_key = {(e["task"], e["episode"]): e for e in theirs}
    shared = sorted(set(mine_by_key) & set(theirs_by_key))

    lines.append("")
    lines.append("=== 1 件ずつ突き合わせる ===")
    if not shared:
        lines.append("  **対になる回が 1 つも無い。** 別のスイートを測っている")
        return

    only_mine: list[tuple[str, str]] = []
    only_theirs: list[tuple[str, str]] = []
    both, neither = 0, 0
    for key in shared:
        left = mine_by_key[key]["outcome"] == "成功"
        right = theirs_by_key[key]["outcome"] == "成功"
        if left and right:
            both += 1
        elif left:
            only_mine.append((key[0], _category_of(key[0], known)))
        elif right:
            only_theirs.append((key[0], _category_of(key[0], known)))
        else:
            neither += 1

    lines.append(f"  対になった回 {len(shared)} 件")
    if len(shared) != len(mine) or len(shared) != len(theirs):
        lines.append(
            f"  **片方にしか無い回がある**（こちら {len(mine)} / あちら {len(theirs)}）"
        )
    lines.append(f"  **こちらだけ成功  {len(only_mine):3d}**  ← 学習で拾えた分")
    lines.append(f"  **あちらだけ成功  {len(only_theirs):3d}**  ← 学習で失った分")
    lines.append(f"  両方成功        {both:3d}")
    lines.append(f"  両方失敗        {neither:3d}")

    probability = _sign_test_probability(len(only_mine), len(only_theirs))
    lines.append("")
    lines.append(
        f"  差 {len(only_mine) - len(only_theirs):+d} 件"
        f"（入れ替わった {len(only_mine) + len(only_theirs)} 件のうち）"
        f"  符号検定 p = {probability:.3f}"
    )
    if probability >= 0.05:
        lines.append("  **偶然の範囲である。** この 2 つに差があるとは言えない")
    else:
        lines.append("  **偶然では説明しにくい。** 差があると読んでよい")

    for label, flipped in (("拾えた", only_mine), ("失った", only_theirs)):
        if not flipped:
            continue
        counted = collections.Counter(category for _, category in flipped)
        breakdown = " / ".join(
            f"{category} {count}" for category, count in counted.most_common()
        )
        lines.append(f"  {label} {len(flipped)} 件のカテゴリ: {breakdown}")

    # **名前を出す。** どのタスクが入れ替わったかが分からないと、
    # 録画を見に行けない（失敗した回だけが残っているため）。
    for label, flipped in (("拾えた", only_mine), ("失った", only_theirs)):
        for task, category in flipped[:10]:
            lines.append(f"    {label}  {category:22s} {task}")
        if len(flipped) > 10:
            lines.append(f"    （{label} は残り {len(flipped) - 10} 件を省略）")


def _parse_log(log_path: Path) -> tuple[list[dict], list[tuple[str, str]], int | None]:
    """評価ログから 1 エピソードずつの結果を取り出す。

    Returns
    -------
    episodes : list[dict]
        `task` / `episode` / `outcome` / `steps` / `offenders` を持つ辞書の列。
        2026-08-11 以降のログはさらに `goal`（bool）・`path_m`・
        `displacement_m`・`motion`・`first_collision_step`・
        **`tail_path_m`（終盤 50 ステップの経路長）・`grip_toggles`（グリッパの開閉数）**
        を持つ。**古いログでは鍵そのものが無い**（None を入れない）。
    exceptions : list[tuple[str, str]]
        (タスク名, 例外の内容)。パッチが例外を握った回。
    declared_total : int | None
        ログが宣言していたタスク総数（`=== タスク 1/84:` の 84）。
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")

    episodes: list[dict] = []
    for found in _RESULT_RE.finditer(text):
        detail = found.group("detail")
        record = {
            "task": found.group("task"),
            "episode": int(found.group("episode")),
            "outcome": found.group("outcome"),
            "steps": int(found.group("steps")),
            "offenders": {
                name: float(millimetres)
                for name, millimetres in _OFFENDER_RE.findall(detail)
            },
        }

        # **無い項目は鍵ごと入れない。** 0 を入れると「測ったが 0 だった」と
        # 見分けが付かず、古いログを混ぜたときに平均が引き下がる。
        goal = _GOAL_RE.search(detail)
        if goal:
            record["goal"] = goal.group(1) == "到達"
        for key, pattern in (
            ("path_m", _PATH_RE), ("displacement_m", _DISPLACEMENT_RE),
        ):
            number = pattern.search(detail)
            if number:
                record[key] = float(number.group(1))
        motion = _MOTION_RE.search(detail)
        if motion:
            record["motion"] = motion.group(1)
        first_collision = _FIRST_COLLISION_RE.search(detail)
        if first_collision:
            record["first_collision_step"] = int(first_collision.group(1))
        tail_path = _TAIL_PATH_RE.search(detail)
        if tail_path:
            record["tail_path_m"] = float(tail_path.group(1))
        grip = _GRIP_RE.search(detail)
        if grip:
            record["grip_toggles"] = int(grip.group(1))

        episodes.append(record)

    exceptions = _EXCEPTION_RE.findall(text)
    totals = _TASK_TOTAL_RE.findall(text)
    return episodes, exceptions, int(totals[0]) if totals else None


def _report_coverage(lines: list[str], episodes: list[dict],
                     exceptions: list[tuple[str, str]],
                     declared_total: int | None) -> None:
    """**まず件数を見る。** 落ちていても終了コードは 0 になる。"""
    tasks_seen = len({episode["task"] for episode in episodes})
    lines.append("=== 件数 ===")
    if declared_total is None:
        lines.append(f"  結果 {len(episodes)} 件（タスク {tasks_seen} 種）")
        lines.append("  **ログにタスク総数が無い。** 途中で落ちた可能性がある")
        return

    lines.append(f"  結果 {len(episodes)} 件 / タスク {tasks_seen} / 宣言 {declared_total}")
    if tasks_seen < declared_total:
        missing = declared_total - tasks_seen
        lines.append(f"  **{missing} タスクが測れていない。** 内訳を下で見ること")
    else:
        lines.append("  **全部そろった。**")

    if exceptions:
        lines.append(f"  例外で測れなかった回 {len(exceptions)} 件:")
        for task_name, reason in exceptions[:5]:
            lines.append(f"    {task_name[:60]}  ({reason})")


def _goal_reached(episode: dict, cap: int) -> bool:
    """ゴール条件を満たしたか。**新しいログは記録があり、古いログは推定する。**

    パッチが `ゴール 到達` を出す前のログでは、打ち切り前に終わったこと
    （`steps < 上限`）でしか判定できない。ループの抜け道は `done` だけなので
    （`pipeline/rollout.py` の `if done: break`）、**上限が分かっていれば
    推定は正確**である。上限は「観測された最大ステップ数」で置く。決め打ちにすると
    `MAX_STEPS` を変えた回で静かに壊れる。
    """
    if "goal" in episode:
        return bool(episode["goal"])
    return episode["steps"] < cap


def _report_outcomes(lines: list[str], episodes: list[dict]) -> None:
    """結末の内訳。**ゴール到達と成功を分けて出す。**"""
    counts = collections.Counter(episode["outcome"] for episode in episodes)
    total = len(episodes)
    successes = counts.get("成功", 0)
    low, high = _wilson_interval(successes, total)
    cap = max(episode["steps"] for episode in episodes)

    lines.append("")
    lines.append("=== 結末 ===")
    lines.append(f"  成功率 {successes / total:.3f}  95%CI [{low:.3f}, {high:.3f}]  n={total}")
    for outcome in ("成功", "衝突", "未完了"):
        count = counts.get(outcome, 0)
        lines.append(f"  {outcome:<4} {count:>4}  ({count / total:>5.1%})")

    # **ゴール到達率と成功率の差が、衝突で落としている分である。**
    # ここが開いていれば軌道の話（近づき方・置き方）、閉じていれば
    # 到達そのものの話（視覚・掴み）で、**打つ手が正反対になる。**
    goals = [episode for episode in episodes if _goal_reached(episode, cap)]
    collided_after_goal = [e for e in goals if e["offenders"]]
    estimated = "" if all("goal" in e for e in episodes) else \
        f"（{sum(1 for e in episodes if 'goal' not in e)} 件はステップ数から推定）"
    lines.append("")
    lines.append(f"  ゴール到達率 {len(goals) / total:.3f}  ({len(goals)}/{total}){estimated}")
    lines.append(
        f"  ├ 到達して成功            {len(goals) - len(collided_after_goal):>4}"
    )
    lines.append(
        f"  └ **到達したが衝突で失格  {len(collided_after_goal):>4}**"
        "  ← 衝突対策で拾える分"
    )
    lines.append(
        f"  未到達                    {total - len(goals):>4}"
        f"  （うち衝突も {sum(1 for e in episodes if not _goal_reached(e, cap) and e['offenders'])}）"
    )
    if goals:
        lines.append(
            f"  → ゴールに着いた回のうち {len(collided_after_goal) / len(goals):.1%} を"
            "衝突で失っている"
        )

    firsts = [e["first_collision_step"] for e in episodes if "first_collision_step" in e]
    if firsts:
        lines.append(
            f"  初めて 1mm を超えた歩  平均 {sum(firsts) / len(firsts):.1f} / "
            f"最小 {min(firsts)} / 最大 {max(firsts)}  (n={len(firsts)})"
        )

    successful_steps = sorted(e["steps"] for e in episodes if e["outcome"] == "成功")
    if successful_steps:
        middle = successful_steps[len(successful_steps) // 2]
        lines.append(
            f"  成功した回のステップ数  最小 {successful_steps[0]} / "
            f"中央 {middle} / 最大 {successful_steps[-1]}  （上限は {cap}）"
        )


def _report_motion(lines: list[str], episodes: list[dict]) -> None:
    """手先の動きの内訳。**「そもそも動いていない」を最初に捕まえる。**

    他の参加者から「全タスク 0%・卓上物体の最大移動量が常に 0.0mm」という
    症状の相談が出ている。行動の非正規化・グリッパの符号・画像の向きの
    いずれかが外れると、**エラーも警告も出ないまま腕が動かない。**
    こちらは 0.1409 を出しているので今のところ無縁だが、**重みを差し替えた
    直後に同じ穴へ落ちうる。** 84 件の平均経路長を見れば一目で分かる。
    """
    measured = [episode for episode in episodes if "path_m" in episode]
    if not measured:
        lines.append("")
        lines.append("=== 動き ===")
        lines.append("  記録が無い（2026-08-11 より前のログ）")
        return

    paths = [episode["path_m"] for episode in measured]
    displacements = [
        episode["displacement_m"] for episode in measured if "displacement_m" in episode
    ]
    notes = collections.Counter(
        episode.get("motion", "ラベル無し") for episode in measured
    )

    lines.append("")
    lines.append("=== 動き ===")
    lines.append(
        f"  経路長 平均 {sum(paths) / len(paths):.2f}m / "
        f"最小 {min(paths):.2f}m / 最大 {max(paths):.2f}m  (n={len(measured)})"
    )
    if displacements:
        lines.append(
            f"  変位   平均 {sum(displacements) / len(displacements):.2f}m / "
            f"最大 {max(displacements):.2f}m"
        )
    for note, count in notes.most_common():
        lines.append(f"  {note:<8} {count:>4}  ({count / len(measured):>5.1%})")

    _report_grip(lines, episodes)
    _report_timeout_kinds(lines, episodes)

    # **腕が動いていないなら、ここから先の表を読む意味が無い。**
    stalled = notes.get("静止", 0)
    if stalled >= len(measured) * 0.8:
        lines.append("")
        lines.append(
            f"  **{stalled}/{len(measured)} 件が「静止」である。方策が動いていない疑いが強い。**"
        )
        lines.append("  重み・行動の非正規化・グリッパの符号・画像の向きを先に確かめること")
        lines.append("  （docs/plan/01_failure_accounting.md の「動いていないときに見る順」）")


def _report_grip(lines: list[str], episodes: list[dict]) -> None:
    """グリッパの開閉回数を結末ごとに出す。**本番の悪化を検知できた指標である。**

    84 件の検証用データでは、素の重みと lora20k の成功率に差が無かった
    （0.476 対 0.500、1 件ずつ対にして 9 拾い 7 失い、符号検定 p=0.80）。
    **差が出たのはここだけだった。**

    | | 素の重み | lora20k |
    |---|---|---|
    | 開閉 20 回以上の割合 | 11/84 (13%) | **23/84 (27%)** |

    **開閉が 20 回以上だった 34 件（両方の回の合計）のうち成功は 2 件（0.06）**
    で、ほぼ完全な失敗の目印である。成功した回の中央は 2 回しかない。
    掴み直しを繰り返すとジャークも悪化するため、本番の配点に直接効く。
    """
    measured = [e for e in episodes if "grip_toggles" in e]
    if not measured:
        return

    chatter = [e for e in measured if e["grip_toggles"] >= _GRIP_CHATTER_THRESHOLD]
    lines.append("")
    lines.append("=== グリッパの開閉 ===")
    for outcome in ("成功", "衝突", "未完了"):
        group = sorted(e["grip_toggles"] for e in measured if e["outcome"] == outcome)
        if group:
            lines.append(
                f"  {outcome:<4} 中央 {group[len(group) // 2]:>3} 回 "
                f"(最小 {group[0]} / 最大 {group[-1]}, n={len(group)})"
            )
    ratio = len(chatter) / len(measured)
    lines.append(
        f"  **{_GRIP_CHATTER_THRESHOLD} 回以上（掴み直しで空回り） "
        f"{len(chatter)}/{len(measured)} = {ratio:.1%}**"
    )
    if chatter:
        succeeded = sum(1 for e in chatter if e["outcome"] == "成功")
        lines.append(
            f"    そのうち成功 {succeeded} 件（{succeeded / len(chatter):.2f}）"
            "  ← **失敗の目印。素の重みは 0.13 だった**"
        )


def _report_timeout_kinds(lines: list[str], episodes: list[dict]) -> None:
    """未完了を「固まった」と「空回り」に割る。**打つ手が正反対である。**

    終盤 50 ステップの経路長で分ける。**全体の変位では「動いてから固まった」回を
    取りこぼす**（前半で 0.31m 動いてから 200 ステップ固まった回を実際に見逃した）。
    """
    timeouts = [e for e in episodes if e["outcome"] == "未完了" and "tail_path_m" in e]
    if not timeouts:
        return

    lines.append("")
    lines.append("=== 未完了の内訳（終盤 50 ステップの移動量で割る）===")
    bands = (
        (0.0, 0.03, "**固まっている**（<0.03m）"),
        (0.03, 0.10, "わずかに動く（0.03〜0.10m）"),
        (0.10, float("inf"), "**動き続けている**（>0.10m）"),
    )
    for low, high, label in bands:
        group = [e for e in timeouts if low <= e["tail_path_m"] < high]
        if not group:
            continue
        grips = sorted(e["grip_toggles"] for e in group if "grip_toggles" in e)
        note = f"  グリッパ開閉 中央 {grips[len(grips) // 2]} 回" if grips else ""
        lines.append(f"  {len(group):>3} 件  {label}{note}")
    lines.append(
        "  **固まっている回は引き直し（項目 9）、空回りは掴み方の学習が的である**"
    )


def _report_categories(lines: list[str], episodes: list[dict],
                       known: dict[str, str]) -> None:
    """摂動カテゴリ別。**学習の的を決めるのはこの表である。**"""
    by_category: dict[str, list[dict]] = collections.defaultdict(list)
    for episode in episodes:
        by_category[_category_of(episode["task"], known)].append(episode)

    lines.append("")
    lines.append("=== 摂動カテゴリ別（弱い順）===")
    if not known:
        # **CSV が読めていないと 3 カテゴリが束になる。** この表を学習の的の
        # 根拠にすると外すので、必ず言う。
        lines.append("  **CSV が読めていない。** 視点・指示文・初期姿勢は名前から")
        lines.append("  分けられないため 1 行に束ねてある（--csv で CSV を渡すこと）")
    rows = []
    for category, group in by_category.items():
        successes = sum(1 for e in group if e["outcome"] == "成功")
        rows.append((successes / len(group), category, successes, len(group), group))
    for rate, category, successes, total, group in sorted(rows):
        breakdown = collections.Counter(e["outcome"] for e in group)
        detail = " ".join(f"{k} {v}" for k, v in breakdown.most_common())
        lines.append(f"  {_pad(category, 26)} {rate:.2f}  ({total:>3} 本)  {detail}")

    # **見た目の摂動は接尾辞まで割る。** `table` と `tb` はどちらも
    # Background Textures だが、`tb` のほうが明るさの振れ幅が大きい。
    # 12 本が `table` に偏っていれば、背景の弱さを過小に見ることになる。
    by_suffix: dict[str, list[dict]] = collections.defaultdict(list)
    for episode in episodes:
        suffix = _appearance_suffix(episode["task"])
        if suffix:
            by_suffix[suffix].append(episode)
    if by_suffix:
        lines.append("")
        lines.append("=== 見た目の摂動を接尾辞で割る ===")
        lines.append("  母集団の比 light 1,142 / table 613 / tb 463（LIBERO-plus 全体）")
        for suffix in ("light", "table", "tb"):
            group = by_suffix.get(suffix, [])
            if not group:
                lines.append(f"  {suffix:<6} **0 本**  ← この見た目は測れていない")
                continue
            successes = sum(1 for e in group if e["outcome"] == "成功")
            lines.append(
                f"  {suffix:<6} {successes / len(group):.2f}  ({len(group):>3} 本)"
            )


def _environment_offenders(episodes: list[dict]) -> set[tuple[str, str]]:
    """環境側で最初から動いている疑いのある (物体名, 変位) を集める。

    **物体の名前では決めない。** 同じ物体でも、方策がぶつける回と環境が
    落とす回があるからである。代わりに次の 2 つの道で拾う。

    | 道 | 決め手 | 効く場面 |
    |---|---|---|
    | 回数 | 同じ変位が `_ENVIRONMENT_REPEAT_MINIMUM` 回そろった | 1 タスクを何本も回す 84 件の一覧 |
    | **物理** | **1〜2 ステップ目で 100mm 超動いた** | **1 タスク 1 本の 36 件の一覧** |

    **物理の道が要る理由。** 36 件は 1 タスク 1 本なので、回数の道は
    「別のタスク 3 件が同じ落下物を持つ」ときしか発火しない。2 件どまりで
    `ketchup_1=544.6mm` を見逃した実例がある（定数のところに記録した）。

    Parameters
    ----------
    episodes : list[dict]
        `_parse_log` が返したエピソードの列。

    Returns
    -------
    suspicious : set[tuple[str, str]]
        (物体名, 0.1mm へ丸めた変位) の組。`_policy_offenders` が使う。
    """
    counts = collections.Counter(
        (name, f"{millimetres:.1f}")
        for episode in episodes
        for name, millimetres in episode["offenders"].items()
    )
    suspicious = {
        key for key, times in counts.items()
        if times >= _ENVIRONMENT_REPEAT_MINIMUM
    }

    # **1 ステップで 10cm は方策には出せない。** 回数の一致を待たずに落とす。
    for episode in episodes:
        first_step = episode.get("first_collision_step")
        if first_step is None or first_step > _IMPOSSIBLE_COLLISION_STEPS:
            continue
        for name, millimetres in episode["offenders"].items():
            if millimetres > _IMPOSSIBLE_DISPLACEMENT_MM:
                suspicious.add((name, f"{millimetres:.1f}"))

    return suspicious


def _policy_offenders(
    episode: dict, environment: set[tuple[str, str]]
) -> dict[str, float]:
    """その回にぶつけた物体から、環境由来を除いたものを返す。

    Parameters
    ----------
    episode : dict
        1 エピソードの記録。
    environment : set[tuple[str, str]]
        `_environment_offenders` の結果。

    Returns
    -------
    offenders : dict[str, float]
        物体名 → 変位 (mm)。**空なら「方策はぶつけていない」**。
    """
    return {
        name: millimetres
        for name, millimetres in episode["offenders"].items()
        if (name, f"{millimetres:.1f}") not in environment
    }


def _report_collision_source(lines: list[str], episodes: list[dict]) -> None:
    """衝突を「方策由来」と「環境由来」に割る。**衝突対策に手を入れるかを決める。**

    **本番のスコアには環境由来の分も乗る。** 落ち着かせる 10 ステップは採点側の
    コードなので、こちらでは直せない。**だからここで出す「方策の衝突率」は
    スコアの予測ではなく、対策の効き目を測るための数字である。**

    この区別が無いと、**衝突対策を入れても数字が動かず「効かなかった」と
    読んでしまう**（環境由来の分が下限として残るため）。
    """
    collided = [episode for episode in episodes if episode["offenders"]]
    lines.append("")
    lines.append("=== 衝突の切り分け ===")
    if not collided:
        lines.append("  衝突が 1 件も無い")
        return

    environment = _environment_offenders(episodes)
    by_policy = [
        episode for episode in collided
        if _policy_offenders(episode, environment)
    ]
    by_environment_only = len(collided) - len(by_policy)
    total = len(episodes)

    lines.append(f"  衝突した回 {len(collided)} 件 / 全 {total} 件")
    lines.append(f"  ├ 環境由来だけ            {by_environment_only:>4}"
                 "  （落下中の物体しか動いていない）")
    lines.append(f"  └ **方策由来を含む        {len(by_policy):>4}**"
                 "  ← 衝突対策で拾えるのはこの分")
    lines.append("")
    lines.append(f"  **方策の衝突率      {len(by_policy) / total:.3f}**"
                 f"  ({len(by_policy)}/{total})  ← 対策の効き目はこれで測る")
    lines.append(f"  環境を含む衝突率    {len(collided) / total:.3f}"
                 f"  ({len(collided)}/{total})  ← **本番のスコアに乗るのはこちら**")

    if environment:
        lines.append("")
        lines.append("  環境由来と見た物体（同じ変位が "
                     f"{_ENVIRONMENT_REPEAT_MINIMUM} 回以上、または "
                     f"{_IMPOSSIBLE_COLLISION_STEPS} 歩以内で "
                     f"{_IMPOSSIBLE_DISPLACEMENT_MM:.0f}mm 超）:")
        for name, value in sorted(environment):
            lines.append(f"    {name}={value}mm")
        lines.append("  → 原因は `pipeline/rollout.py` が初期位置を捉える前に")
        lines.append("    **10 ステップしか落ち着かせない**ことである（採点側のコード）")

    # **早すぎる衝突は補強材料である。判定には使っていない。**
    # 環境由来なら方策が届く前にぶつかるので、初衝突の歩が小さく出る。
    firsts = [
        (episode["first_collision_step"], episode["task"])
        for episode in collided if "first_collision_step" in episode
    ]
    early = [step for step, _ in firsts if step <= _EARLY_COLLISION_STEPS]
    if firsts:
        lines.append("")
        lines.append(
            f"  初衝突が {_EARLY_COLLISION_STEPS} 歩以内  {len(early)}/{len(firsts)} 件"
        )
        if early and by_environment_only == 0:
            lines.append("    **早い衝突があるのに環境由来と判定された回が無い。**")
            lines.append("    変位が毎回わずかに違うと上の判定は当たらない。映像で確かめること")


def _report_exposure(lines: list[str], episodes: list[dict],
                     exposures: dict[str, int]) -> None:
    """衝突しうる相手がいる回だけで衝突率を出し直す。

    **衝突率はタスク構成に左右される。** 操作対象以外の物体が 0 個のタスクは
    衝突では落ちないので、分母に入れると衝突率が薄まる。逆に 5 個あるタスクは
    同じ方策でも失格の機会が多い。**方策を比べるときは、相手がいる回だけを見る。**

    **数えるのは方策由来の衝突だけである**（環境由来は `_report_collision_source`
    が除く）。この節の目的は方策を比べることなので、**どの方策でも同じだけ
    出る分を混ぜると差が薄まる。**
    """
    known = [e for e in episodes if e["task"] in exposures]
    if not known:
        return

    environment = _environment_offenders(episodes)
    exposed = [e for e in known if exposures[e["task"]] > 0]
    lines.append("")
    lines.append("=== 衝突しうる相手の数で割る（方策由来だけ）===")
    lines.append(
        f"  相手がいない回 {len(known) - len(exposed)} 件"
        "（**衝突では落ちない**）は下の分母から外す"
    )
    if exposed:
        collided = sum(1 for e in exposed if _policy_offenders(e, environment))
        lines.append(
            f"  相手がいる回の衝突率 {collided / len(exposed):.3f}  "
            f"({collided}/{len(exposed)})"
        )
    for threshold, label in ((1, "1〜2 個"), (3, "3〜4 個"), (5, "5 個以上")):
        group = [
            e for e in exposed
            if threshold <= exposures[e["task"]] < threshold + 2
            or (threshold == 5 and exposures[e["task"]] >= 5)
        ]
        if not group:
            continue
        collided = sum(1 for e in group if _policy_offenders(e, environment))
        successes = sum(1 for e in group if e["outcome"] == "成功")
        lines.append(
            f"  {label:<8} 衝突 {collided / len(group):.2f}  "
            f"成功 {successes / len(group):.2f}  ({len(group):>3} 本)"
        )


def _report_offenders(lines: list[str], episodes: list[dict]) -> None:
    """ぶつけた物体の偏りと、**環境側で動いている疑い**を出す。"""
    counts: collections.Counter[str] = collections.Counter()
    displacements: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for episode in episodes:
        for name, millimetres in episode["offenders"].items():
            counts[name] += 1
            displacements[name][f"{millimetres:.1f}mm"] += 1

    if not counts:
        lines.append("")
        lines.append("=== 衝突 ===")
        lines.append("  1 件も無い")
        return

    lines.append("")
    lines.append("=== ぶつけた物体 ===")
    for name, count in counts.most_common(10):
        lines.append(f"  {count:>4}  {name}")

    # **同じ変位が何度も出るのは方策の振る舞いではない。** 摂動が違う回で
    # 変位が 0.1mm も違わないなら、環境側で最初から動いていると考えるべきである
    # （8/10 に `ketchup_1=544.6mm` が 6 回そろって出た）。
    #
    # **切り分けの結果は `_report_collision_source` に出す。** ここでは
    # 何回ずつ出たかだけを示す（判定の材料が見えるようにするため）。
    suspicious = [
        (name, value, times)
        for name, values in displacements.items()
        for value, times in values.items()
        if times >= _ENVIRONMENT_REPEAT_MINIMUM
    ]
    if suspicious:
        lines.append("")
        lines.append("  **同じ変位が繰り返し出ている。環境側で動いている疑いがある：**")
        for name, value, times in sorted(suspicious, key=lambda row: -row[2]):
            lines.append(f"    {name}={value} が {times} 回")
        lines.append("  → 切り分けた本数は「衝突の切り分け」の節に出る")


def _report_json(lines: list[str], output_dir: Path) -> None:
    """パイプラインの JSON から指標を読む。**例外の跡もここに出る。**"""
    candidates = sorted(output_dir.glob("*.json"))
    if not candidates:
        lines.append("")
        lines.append("=== JSON ===")
        lines.append("  無し")
        return

    payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    lines.append("")
    lines.append(f"=== 指標（{candidates[-1].name}）===")
    tracks = payload.get("tracks") or []
    if not tracks:
        lines.append("  トラックが 1 つも無い")
        return
    metrics = tracks[0].get("overall_metrics", {})
    if metrics.get("error"):
        lines.append("  **例外で終わっている。** JSON はエラー記録だけを持つ")
        lines.append(f"  {metrics}")
        return
    for key in (
        "mean_success_rate", "mean_collision_rate", "mean_avg_cartesian_jerk",
        "mean_rms_cartesian_jerk", "mean_sparc", "mean_cartesian_path_length",
        "n_tasks", "n_total_episodes",
    ):
        if key in metrics:
            lines.append(f"  {key:<32} {metrics[key]}")


def _overall_metrics(output_dir: Path) -> dict[str, float]:
    """パイプラインの JSON から全体の指標を読む。読めなければ空。"""
    for candidate in sorted(output_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        tracks = payload.get("tracks") or []
        if not tracks:
            continue
        metrics = tracks[0].get("overall_metrics", {})
        if metrics and not metrics.get("error"):
            return metrics
    return {}


def _key_figures(output_dir: Path, known: dict[str, str],
                 only: list[str] | None = None) -> dict[str, object]:
    """1 回の評価から、比べるのに使う数字だけを取り出す。

    **報告文をそのまま並べても差は読めない。** 84 件の内訳が数十行あるので、
    2 つ見比べると見落とす。**判断が変わる数字だけを抜く。**

    Parameters
    ----------
    output_dir : Path
        `run_docker_eval.sh` の出力先。
    known : dict[str, str]
        タスク名 → 摂動カテゴリ。CSV から読んだもの。
    only : list[str] | None
        この摂動カテゴリだけで数え直す。`None` なら全件。
        **`jerk` と `sparc` には効かない**（JSON が全体の平均しか持たない）。

    Returns
    -------
    figures : dict[str, object]
        `n` / `success` / `goal` / `policy_collision` / `any_collision` /
        `timeout` / `stalled` / `jerk` / `sparc` / `by_category`。
        **読めなかった項目は入れない**（None を入れて 0 と混同させない）。
    """
    episodes, _, _ = _parse_log(output_dir / "eval.log")
    episodes = _filter_by_categories(episodes, known, only)
    if not episodes:
        return {"n": 0}

    total = len(episodes)
    cap = max(episode["steps"] for episode in episodes)
    environment = _environment_offenders(episodes)
    metrics = _overall_metrics(output_dir)

    figures: dict[str, object] = {
        "n": total,
        "success": sum(1 for e in episodes if e["outcome"] == "成功") / total,
        "goal": sum(1 for e in episodes if _goal_reached(e, cap)) / total,
        "policy_collision": sum(
            1 for e in episodes if _policy_offenders(e, environment)
        ) / total,
        "any_collision": sum(1 for e in episodes if e["offenders"]) / total,
        "timeout": sum(1 for e in episodes if e["steps"] >= cap) / total,
    }
    motions = [e["motion"] for e in episodes if "motion" in e]
    if motions:
        figures["stalled"] = motions.count("静止") / total
        figures["wandering"] = motions.count("往復") / total

    # **`固まった` は録画側が新しい版でないと 1 件も出ない。**
    # 古いログで 0 を入れると「固まった回が無かった」と読めてしまう。
    # 終盤の記録があるかどうかで、版を見分ける。
    if any("tail_path_m" in e for e in episodes):
        figures["froze"] = motions.count("固まった") / total

    # **これが本番の悪化を検知できた唯一の指標である**（`_report_grip`）。
    gripped = [e["grip_toggles"] for e in episodes if "grip_toggles" in e]
    if gripped:
        figures["grip_chatter"] = (
            sum(1 for value in gripped if value >= _GRIP_CHATTER_THRESHOLD) / len(gripped)
        )
    for key, source in (("jerk", "mean_avg_cartesian_jerk"), ("sparc", "mean_sparc")):
        if source in metrics:
            figures[key] = float(metrics[source])

    by_category: dict[str, float] = {}
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for episode in episodes:
        grouped[_category_of(episode["task"], known)].append(episode)
    for category, group in grouped.items():
        by_category[category] = (
            sum(1 for e in group if e["outcome"] == "成功") / len(group)
        )
    figures["by_category"] = by_category
    return figures


def _format_delta(mine: float, theirs: float, higher_is_better: bool) -> str:
    """差を、どちらが良いか分かる形で書く。"""
    delta = mine - theirs
    if abs(delta) < 5e-4:
        return f"{delta:+.3f}  同じ"
    improved = delta > 0 if higher_is_better else delta < 0
    return f"{delta:+.3f}  {'良くなった' if improved else '**悪くなった**'}"


# 比べる項目。(鍵, 表示名, 大きいほうが良いか)
_COMPARISON_ROWS: tuple[tuple[str, str, bool], ...] = (
    ("success", "成功率", True),
    ("goal", "ゴール到達率", True),
    ("policy_collision", "方策の衝突率", False),
    ("any_collision", "環境を含む衝突率", False),
    ("timeout", "打ち切りに張り付いた率", False),
    # **グリッパは成功率より鋭い。** 本番で 8.8 倍悪化したとき、手元で
    # 動いたのはこの行とジャークだけだった。
    ("grip_chatter", "グリッパ 20 回以上の率", False),
    ("stalled", "静止した率", False),
    ("froze", "固まった率", False),
    ("wandering", "往復した率", False),
    ("jerk", "平均ジャーク", False),
    ("sparc", "SPARC（0 に近いほど滑らか）", True),
)


def _report_comparison(lines: list[str], mine: dict[str, object],
                       theirs: dict[str, object], label: str) -> None:
    """2 回の評価を並べる。**どちらが良いかを毎行に書く。**

    **滑らかさの指標を必ず並べる。** 本番では 3 回目から 4 回目で
    成功率が 2 分の 1 になったのに、**スコアは 8.8 分の 1 まで落ちた。**
    残る 4.4 倍は滑らかさと時間の配点で削られた分である。
    **成功率だけを見ると「同じくらい」と読み違える。**

    Parameters
    ----------
    mine, theirs : dict[str, object]
        `_key_figures` の結果。`mine` が `--output-dir` の側。
    label : str
        比較相手の名前。画面に出す。
    """
    lines.append("")
    lines.append("=== 並べて比べる ===")
    lines.append(f"  こちら: --output-dir  (n={mine.get('n')})")
    lines.append(f"  あちら: {label}  (n={theirs.get('n')})")
    if mine.get("n") != theirs.get("n"):
        lines.append("  **件数が違う。** どちらかが途中で落ちている。差を読む前に確かめること")

    lines.append("")
    # **見出しは `>9` では合わない。** 書式指定は文字数で詰めるため、
    # 全角 3 文字は 9 桁ではなく 6 桁に見える。数値の欄（9 桁）に合わせて
    # 空白を自分で置く。
    lines.append(f"  {_pad('項目', 28)}    こちら    あちら   差")
    missing: list[str] = []
    for key, name, higher_is_better in _COMPARISON_ROWS:
        if key not in mine or key not in theirs:
            missing.append(name)
            continue
        left, right = float(mine[key]), float(theirs[key])  # type: ignore[arg-type]
        lines.append(
            f"  {_pad(name, 28)} {left:>9.3f} {right:>9.3f}   "
            f"{_format_delta(left, right, higher_is_better)}"
        )

    # **出せなかった行を黙って落とさない。** 行が無いのを「差が無い」と
    # 読み違えるのが、いちばん危ない誤読である。とくに滑らかさの指標は
    # 本番の悪化 8.8 倍のうち 4.4 倍を占めていた。
    if missing:
        lines.append("")
        lines.append(f"  **出せなかった項目: {' / '.join(missing)}**")
        lines.append("    どちらかの回に記録が無い（古いログ、または JSON が残っていない）")
        lines.append("    **「差が無い」ではない。**")

    left_categories = mine.get("by_category") or {}
    right_categories = theirs.get("by_category") or {}
    shared = sorted(set(left_categories) & set(right_categories))  # type: ignore[arg-type]
    if shared:
        lines.append("")
        lines.append("  摂動カテゴリ別の成功率（差の大きい順）")
        rows = [
            (left_categories[c] - right_categories[c], c,  # type: ignore[index]
             left_categories[c], right_categories[c])      # type: ignore[index]
            for c in shared
        ]
        for delta, category, left, right in sorted(rows, key=lambda row: row[0]):
            mark = " **悪くなった**" if delta < -0.05 else (
                " 良くなった" if delta > 0.05 else "")
            lines.append(
                f"    {_pad(category, 26)} {left:>5.2f} {right:>5.2f}  {delta:+.2f}{mark}"
            )

    lines.append("")
    lines.append("  **本番のスコアはここから計算できない。** 配点の設定ファイルが")
    lines.append("  配布されていないため、比べられるのは上の指標までである")


def _guess_csv(output_dir: Path) -> Path | None:
    """検証用スイートの CSV を探す。見つからなければ None。

    **`run_gpu_eval.sh` の既定は `<リポジトリ>/work/eval-kit` である。**
    出力先が `<リポジトリ>/work/gpu_eval/<日時>_<設定>` なので、キットは
    **1 つ上**（`parents[1]`）にある。ここを 1 段間違えると CSV が見つからず、
    **摂動カテゴリ別の表が静かに変種名からの推定に落ちる**（視点・指示文・
    初期姿勢が 1 行に束ねられる）。**気づけないまま学習の的を決めうる。**

    上の階層から順に見て、`KIT_DIR` を変えた場合でも当たるようにする。
    """
    for parent in output_dir.parents:
        for kit in ("eval-kit", "work/eval-kit"):
            candidate = parent / kit / "compe" / "val" / "VAL_TASKS.csv"
            if candidate.is_file():
                return candidate
    return None


def _report_videos(lines: list[str], output_dir: Path) -> None:
    """撮れた mp4 の本数を出す。**中身の確認は別の道具でやる。**"""
    videos = sorted((output_dir / "videos").glob("*.mp4"))
    lines.append("")
    lines.append("=== 録画 ===")
    if not videos:
        lines.append("  無し（RECORD= で無効にしたか、失敗した回が無かった）")
        return
    lines.append(f"  {len(videos)} 本")
    lines.append("  見るには:")
    lines.append(f"    python3 solution/tools/summarize_rollout_videos.py {output_dir / 'videos'}")


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output_dir.resolve()
    log_path = output_dir / "eval.log"

    if not log_path.is_file():
        raise SystemExit(f"評価ログが無い: {log_path}")

    episodes, exceptions, declared_total = _parse_log(log_path)
    if not episodes:
        print(f"**結果が 1 件も無い。** ログの末尾を見ること: {log_path}")
        raise SystemExit(1)

    csv_path = args.csv
    if csv_path is None:
        csv_path = _guess_csv(output_dir)

    categories = _load_categories(csv_path)
    lines: list[str] = [f"評価ログ  {log_path}"]
    lines.append(
        f"CSV       {csv_path}" if categories
        else "CSV       **見つからない**（摂動カテゴリは変種名から当てる）"
    )

    # **件数の確認は絞り込む前の全件で行う。** 「最後まで走ったか」は
    # 走り全体の性質であり、絞った件数と突き合わせると偽の警告が出る。
    all_episodes = episodes
    only: list[str] | None = args.only_categories
    if only:
        if not categories:
            raise SystemExit(
                "**--only-categories は CSV が無いと使えない。**"
                "変種名からは 7 カテゴリのうち 4 つしか分からないため、"
                "絞った結果が静かに間違う。--csv で渡すこと"
            )
        episodes = _filter_by_categories(all_episodes, categories, only)
        if not episodes:
            raise SystemExit(
                f"**指定したカテゴリの回が 1 つも無い: {', '.join(only)}**"
                "（名前は CSV の category 列と同じ綴りで渡すこと）"
            )
        lines.append(
            f"絞り込み  {' / '.join(only)}  → {len(episodes)} 件 / 全 {len(all_episodes)} 件"
        )
        lines.append("  **ジャークと SPARC は絞り込みが効かない**"
                     "（パイプラインの JSON が全体の平均しか持たないため）")

    lines.append("")
    _report_coverage(lines, all_episodes, exceptions, declared_total)
    _report_outcomes(lines, episodes)
    _report_motion(lines, episodes)
    _report_categories(lines, episodes, categories)
    _report_collision_source(lines, episodes)
    _report_exposure(lines, episodes, _load_nontarget_counts(csv_path))
    _report_offenders(lines, episodes)
    _report_json(lines, output_dir)
    _report_videos(lines, output_dir)

    # **比較は最後に出す。** 上の内訳を読んだあとで差を見るほうが、
    # どこが動いたのかを追いやすい。
    if args.compare is not None:
        other = args.compare.resolve()
        if not (other / "eval.log").is_file():
            lines.append("")
            lines.append(f"=== 並べて比べる ===\n  **比較相手に eval.log が無い: {other}**")
        else:
            other_episodes, _, _ = _parse_log(other / "eval.log")
            _report_comparison(
                lines,
                _key_figures(output_dir, categories, only),
                _key_figures(other, categories, only),
                str(other),
            )
            _report_pairing(
                lines, episodes,
                _filter_by_categories(other_episodes, categories, only),
                categories,
            )

    report = "\n".join(lines)
    print(report)

    # **絞り込みや比較のときは summary.txt を上書きしない。**
    # あれは「その回そのもの」の記録で、あとから読み返す唯一の材料である。
    name = "summary.txt"
    if only or args.compare is not None:
        name = "summary_" + ("compare" if args.compare is not None else "subset") + ".txt"
    (output_dir / name).write_text(report + "\n", encoding="utf-8")
    print(f"\n（同じ内容を {output_dir / name} に書いた）")


if __name__ == "__main__":
    main()
