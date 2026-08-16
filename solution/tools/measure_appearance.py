#!/usr/bin/env python3
"""録画済みロールアウトの 1 コマ目から、変種ごとの見た目の散らばりを測る。

**なぜ要るか。** 画像を明るくしたり暗くしたりして学習に混ぜる「画像拡張」の
振れ幅（既定は明るさ ×0.8〜1.2）が、評価で出てくる変種の見た目の幅を覆えて
いるかを判断したい。いま拠りどころにしているのは他人が 15 コマで作った集計で、
**こちらの手元では 1 度も数えていない。** 手元の 84 件の評価には録画があり、
各タスクの 1 コマ目はその変種の初期画面そのものなので、そこから測る。

**測り方。** [rollout_recorder.py](rollout_recorder.py) が書いた mp4 の 1 コマ目を
読み、灰色にしてから 2 つの数を出す。

| 数 | 意味 | 出し方 |
|---|---|---|
| 明るさ | 画面全体の明るさの平均 | 灰色画素の平均 ÷ 255（0〜1） |
| コントラスト | 明暗の開き | 灰色画素の標準偏差 ÷ 255（0〜1） |

1 コマは左が広い視野（front）、右が手先（wrist）の 2 枚を横に並べたものなので、
左右に分けて測る。**明るさ・背景の摂動は広い視野に出る**ので front を主に見る。

**焼かれた文字は外す。** 録画にはステップ数やカメラ名が白い文字で焼かれている
（`rollout_recorder._annotate`。上端 y=18、下端 y=高さ-30・高さ-10）。この文字が
コントラストを最大 0.03 ほど水増しするので、既定では上下の文字帯を切り落として
測る（`--include-text-band` で切らずに測れる）。

**必ず頭に入れる偏り。録画は失敗した回だけ残る。**
[rollout_recorder.py](rollout_recorder.py) の既定は `PARC_RECORD_WHICH=failures`
で、成功した回（84 件中 40 件）は 1 コマも書き出されない。**さらにファイル名は
70 文字で切られる**ので、名前の末尾にある変種名が消えた回が多い。よってここで
出る数は**母集団の振れ幅ではなく、失敗した回・名前が残った回だけの振れ幅**である。
母集団を出すには `PARC_RECORD_WHICH=all` で撮り直すか、変種の初期状態から直接
画像を作る（`docs/reference/08_appearance_spread.md` の末尾）。

    # 一覧を標準出力に
    PYTHONPATH=<cv2 の置き場> python3 solution/tools/measure_appearance.py \\
        outputs/0811_101824_base_n5_ep1_max300/videos

    # ドキュメントへ貼る表を出す
    ... measure_appearance.py <dir> --markdown

Reads:
  - <dir>/*.mp4（rollout_recorder.py が書いたもの）
Generates:
  - 標準出力に、変種の接尾辞ごとの明るさ・コントラストと、1 本ごとの明細
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

# ファイル名の形： <結末>_<タスク名[:70]>_ep<N>_<ステップ数>steps.mp4
# （rollout_recorder.py の `_write` が付ける）。
_NAME_RE = re.compile(
    r"^(?P<outcome>成功|衝突|未完了)_(?P<task>.+)_ep(?P<ep>\d+)_(?P<steps>\d+)steps$"
)

# rollout_recorder.py はタスク名を 70 文字で切る（`self._task_name[:70]`）。
# ちょうど 70 文字なら切られたとみなす。切られた回は末尾の変種名が消えている
# ので、接尾辞での集計から外す。
_TASK_NAME_CAP = 70

# 変種名の末尾に付く目印。LIBERO-plus のカテゴリ 7 つのうち、名前で見分けが
# 付くもの（docs/reference/04_track1_population.md・make_validation_suite.py）。
# `view`/`initstate` は Camera Viewpoints・Robot Initial States・Language の
# 3 カテゴリで同じ形になり、名前だけでは区別できない（分類表の category 列が要る）。
_TAIL_RE = re.compile(
    r"_(?P<token>light|table|tb|noise|add|language|initstate)_\d+$"
    r"|_(?P<level>level)\d+(?:_sample\d+)?$"
)

_TRUNCATED = "（不明: 名前が 70 字で切れている）"
_NO_SUFFIX = "（不明: 接尾辞が見当たらない）"

# 焼かれた文字を外すために切り落とす帯（画素）。rollout_recorder._annotate の
# 文字位置（上端 y=18、下端 y=高さ-30 と高さ-10）から決めた絶対値。倍率に
# よらず文字位置は固定なので絶対値で切る。
_TEXT_BAND_TOP = 24
_TEXT_BAND_BOTTOM = 44


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="mp4 の置き場")
    parser.add_argument(
        "--include-text-band", action="store_true",
        help="焼かれた文字を外さずに測る（既定は上下の文字帯を切り落とす）",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="ドキュメントへ貼る Markdown の表を出す",
    )
    return parser.parse_args()


def _variant_suffix(task_name: str) -> str:
    """タスク名から、変種の接尾辞（`light` など）を取り出す。

    **70 字で切れている名前からは取らない。** 変種名は末尾に付くので、
    切られた回は接尾辞が消えているか、数字が途中で切れて別物に見える。
    そういう回は集計から外し、後で件数だけ数える。

    Returns
    -------
    suffix : str
        接尾辞の目印（`light`/`table`/`tb`/`noise`/`add`/`language`/
        `initstate`/`level`）。切れている・見当たらないときは `_TRUNCATED`
        か `_NO_SUFFIX` を返す。
    """
    if len(task_name) >= _TASK_NAME_CAP:
        return _TRUNCATED
    match = _TAIL_RE.search(task_name)
    if not match:
        return _NO_SUFFIX
    return match.group("token") or match.group("level")


def _measure_gray(gray: Any, exclude_text_band: bool) -> tuple[float, float]:
    """灰色の 1 枚から、明るさ（平均）とコントラスト（標準偏差）を 0〜1 で返す。

    Parameters
    ----------
    gray : numpy.ndarray
        灰色画像（0〜255 の 2 次元）。
    exclude_text_band : bool
        True なら上下の文字帯を切り落としてから測る。

    Returns
    -------
    brightness : float
        画素の平均 ÷ 255。
    contrast : float
        画素の標準偏差 ÷ 255。
    """
    height = gray.shape[0]
    if exclude_text_band and height > (_TEXT_BAND_TOP + _TEXT_BAND_BOTTOM) * 2:
        gray = gray[_TEXT_BAND_TOP: height - _TEXT_BAND_BOTTOM, :]
    return float(gray.mean()) / 255.0, float(gray.std()) / 255.0


def _read_panels(path: Path) -> tuple[Any, Any] | None:
    """mp4 の 1 コマ目を読み、灰色にして front/wrist の 2 枚に割る。

    録画は front（広い視野）と wrist（手先）を横に並べた 1 枚である。
    左半分を front、右半分を wrist として返す。

    Returns
    -------
    panels : tuple | None
        (front の灰色画像, wrist の灰色画像)。読めなければ None。
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    read, frame = capture.read()
    capture.release()
    if not read or frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    half = gray.shape[1] // 2
    return gray[:, :half], gray[:, half:]


def _describe(path: Path, exclude_text_band: bool) -> dict[str, Any]:
    """mp4 1 本から、集計に使う情報を取り出す。

    Returns
    -------
    entry : dict
        `outcome`/`task`/`suffix`/`readable` と、front・wrist の
        `brightness`/`contrast`。読めなければ数値は None。
    """
    match = _NAME_RE.match(path.stem)
    task = match.group("task") if match else path.stem
    entry: dict[str, Any] = {
        "path": path,
        "outcome": match.group("outcome") if match else "（不明）",
        "task": task,
        "suffix": _variant_suffix(task),
    }

    panels = _read_panels(path)
    if panels is None:
        entry.update(readable=False, front_brightness=None, front_contrast=None,
                     wrist_brightness=None, wrist_contrast=None)
        return entry

    front, wrist = panels
    front_brightness, front_contrast = _measure_gray(front, exclude_text_band)
    wrist_brightness, wrist_contrast = _measure_gray(wrist, exclude_text_band)
    entry.update(
        readable=True,
        front_brightness=front_brightness, front_contrast=front_contrast,
        wrist_brightness=wrist_brightness, wrist_contrast=wrist_contrast,
    )
    return entry


def _summarize_suffix(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """接尾辞ごとに front の明るさ・コントラストの平均と幅をまとめる。

    **接尾辞が不明な回（名前が切れている・見当たらない）は数えるだけで、
    平均には入れない。** 変種が特定できない見た目を混ぜると意味が薄れる。

    Returns
    -------
    rows : list[dict]
        接尾辞ごとの `suffix`/`count`/明るさとコントラストの平均・最小・最大。
        件数の多い順、同数なら接尾辞名の順。
    """
    by_suffix: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if not entry["readable"]:
            continue
        by_suffix.setdefault(entry["suffix"], []).append(entry)

    rows: list[dict[str, Any]] = []
    for suffix, group in by_suffix.items():
        brightness = [item["front_brightness"] for item in group]
        contrast = [item["front_contrast"] for item in group]
        rows.append({
            "suffix": suffix,
            "count": len(group),
            "brightness_mean": statistics.fmean(brightness),
            "brightness_min": min(brightness),
            "brightness_max": max(brightness),
            "contrast_mean": statistics.fmean(contrast),
            "contrast_min": min(contrast),
            "contrast_max": max(contrast),
        })
    rows.sort(key=lambda row: (-row["count"], row["suffix"]))
    return rows


def _is_measurable(suffix: str) -> bool:
    """接尾辞が特定できて、平均に使える変種か。"""
    return suffix not in (_TRUNCATED, _NO_SUFFIX)


def _print_report(entries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """人が読む形で、偏りの断り書き・接尾辞ごとの表・明細を出す。"""
    readable = [entry for entry in entries if entry["readable"]]
    outcomes = Counter(entry["outcome"] for entry in entries)
    measurable = sum(1 for entry in readable if _is_measurable(entry["suffix"]))

    print(f"mp4 {len(entries)} 本（読めた {len(readable)} 本）")
    print("  結末: " + "  ".join(f"{key} {value}" for key, value in outcomes.most_common()))
    print(f"  接尾辞が特定できた: {measurable} 本 / 読めた {len(readable)} 本")
    print("  **これは失敗した回だけの数字である。**"
          " 成功した 40 件（84 件中）は録画されていない。")

    print("\n=== 接尾辞ごと（front・広い視野。明るさ・コントラストは 0〜1）===")
    print(f"  {'接尾辞':10s} {'本数':>4s}  {'明るさ':>7s} {'（最小〜最大）':>15s}"
          f"  {'コントラスト':>9s} {'（最小〜最大）':>15s}")
    for row in rows:
        note = "" if _is_measurable(row["suffix"]) else "  ← 平均には使わない"
        print(
            f"  {row['suffix']:10s} {row['count']:4d}  "
            f"{row['brightness_mean']:7.3f} "
            f"({row['brightness_min']:.3f}〜{row['brightness_max']:.3f})  "
            f"{row['contrast_mean']:7.3f} "
            f"({row['contrast_min']:.3f}〜{row['contrast_max']:.3f}){note}"
        )

    print("\n=== 1 本ごと（front）===")
    for entry in sorted(readable, key=lambda item: (item["suffix"], item["task"])):
        print(
            f"  {entry['suffix']:10s} 明るさ {entry['front_brightness']:.3f}  "
            f"コントラスト {entry['front_contrast']:.3f}  "
            f"[{entry['outcome']}] {entry['task']}"
        )

    broken = [entry for entry in entries if not entry["readable"]]
    if broken:
        print(f"\n**読めなかった {len(broken)} 本**")
        for entry in broken:
            print(f"  {entry['path'].name}")


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    """ドキュメントへ貼る Markdown の表を出す。平均に使える接尾辞だけ。"""
    print("| 接尾辞 | 本数 | 明るさ 平均 | 明るさ 最小〜最大 | "
          "コントラスト 平均 | コントラスト 最小〜最大 |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        if not _is_measurable(row["suffix"]):
            continue
        print(
            f"| `{row['suffix']}` | {row['count']} | "
            f"{row['brightness_mean']:.3f} | "
            f"{row['brightness_min']:.3f}〜{row['brightness_max']:.3f} | "
            f"{row['contrast_mean']:.3f} | "
            f"{row['contrast_min']:.3f}〜{row['contrast_max']:.3f} |"
        )


def main() -> None:
    args = _parse_args()
    directory: Path = args.directory
    if not directory.is_dir():
        raise SystemExit(f"エラー: {directory} がありません")

    videos = sorted(directory.glob("*.mp4"))
    if not videos:
        raise SystemExit(
            f"{directory} に mp4 がありません。\n"
            "  PARC_RECORD_DIR を設定して評価を回しましたか？"
        )

    exclude_text_band = not args.include_text_band
    entries = [_describe(path, exclude_text_band) for path in videos]
    rows = _summarize_suffix(entries)

    if args.markdown:
        _print_markdown(rows)
    else:
        _print_report(entries, rows)


if __name__ == "__main__":
    main()
