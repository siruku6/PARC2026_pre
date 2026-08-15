#!/usr/bin/env python3
"""予選レポートの Markdown を、A4 2 ページに組める HTML へ変換する。

**pandoc を入れずに PDF 化するための道具である。**出力した HTML をブラウザで開き、
「PDF に書き出す」を選ぶと、そのまま提出できる PDF になる。**ページ数がその場で
見えるので、2 ページに収まったかを目で確認できる。**

対応する記法は、レポートで実際に使っているものだけに絞ってある
（見出し・表・箇条書き・強調・インラインコード・コードブロック・水平線）。
汎用の Markdown 変換器ではない。

Reads:
  - docs/report/qualifier_report.md
Generates:
  - docs/report/qualifier_report.html
"""

import argparse
import html
import pathlib
import re

_PAGE_STYLE = """
@page { size: A4; margin: 14mm 13mm; }
* { box-sizing: border-box; }
body {
    font-family: "Hiragino Sans", "Noto Sans JP", "Yu Gothic", sans-serif;
    font-size: {BODY_FONT_SIZE}; line-height: {LINE_HEIGHT}; color: #111; margin: 0;
}
h1 { font-size: 13pt; margin: 0 0 2mm; border-bottom: 1.4pt solid #111;
     padding-bottom: 1mm; }
h2 { font-size: 10pt; margin: 3.5mm 0 1.5mm; border-left: 3pt solid #111;
     padding-left: 2mm; }
h3 { font-size: 9pt; margin: 3mm 0 1.2mm; display: table;
     border-bottom: 0.5pt solid #666; padding-bottom: 0.6mm; }
p, li { margin: 0 0 1.2mm; }
ul { margin: 0 0 1.5mm; padding-left: 5mm; }
table { border-collapse: collapse; width: 100%; margin: 1mm 0 2mm;
        font-size: {TABLE_FONT_SIZE}; }
th, td { border: 0.4pt solid #999; padding: 0.7mm 1.4mm; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; font-weight: 600; }
/* 表の 1 列目（項目名）の幅。広げると項目名の折り返しが減って行数が縮むが、
   広げすぎると 2 列目以降が折り返して逆に増える。--first-column-width で調整する。 */
th:first-child, td:first-child { width: {FIRST_COLUMN_WIDTH}; }
code { font-family: "SF Mono", Consolas, monospace; font-size: 7.8pt;
       background: #f4f4f4; padding: 0 0.6mm; }
pre { background: #f4f4f4; border: 0.4pt solid #ccc; padding: 1.2mm;
      margin: 1mm 0 2mm; font-size: 7.4pt; line-height: 1.35; overflow: hidden; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 0.4pt solid #ccc; margin: 2.5mm 0; }
strong { font-weight: 600; }
"""


def _parse_args() -> argparse.Namespace:
    """コマンドラインの引数を読む。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="docs/report/qualifier_report.md",
        help="変換する Markdown のパス",
    )
    parser.add_argument(
        "--output", default="docs/report/qualifier_report.html",
        help="書き出す HTML のパス",
    )
    parser.add_argument(
        "--first-column-width", default="26%",
        help="表の 1 列目の幅。項目名の折り返しを減らしたいときに広げる（既定 26%%）",
    )
    parser.add_argument(
        "--body-font-size", default="8.6pt", help="本文の文字サイズ（既定 8.6pt）",
    )
    parser.add_argument(
        "--table-font-size", default="8.1pt", help="表の文字サイズ（既定 8.1pt）",
    )
    parser.add_argument(
        "--line-height", default="1.5", help="行間（既定 1.5）",
    )
    return parser.parse_args()


def _render_inline(text: str) -> str:
    """強調とインラインコードだけを HTML に変換する。

    Parameters
    ----------
    text : str
        1 行ぶんの Markdown。

    Returns
    -------
    rendered : str
        HTML に変換した 1 行。
    """
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    # 表のセル内で改行したいときだけ <br> を許す。
    # それ以外のタグは escape したままにする（意図しない HTML を混ぜないため）。
    escaped = escaped.replace("&lt;br&gt;", "<br>")
    return escaped


def _render_table(rows: list[str]) -> str:
    """Markdown の表を HTML の table にする。

    2 行目の区切り行（`|---|---|`）は読み飛ばす。
    """
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], cells[2:]
    parts = ["<table><thead><tr>"]
    parts += [f"<th>{_render_inline(c)}</th>" for c in header]
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>" + "".join(f"<td>{_render_inline(c)}</td>" for c in row) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _render_document(source: str) -> str:
    """Markdown 全体を HTML の本文に変換する。"""
    lines = source.splitlines()
    parts: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]

        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            parts.append(_render_table(table_lines))
            continue

        if line.startswith("```"):
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(html.escape(lines[index]))
                index += 1
            index += 1
            parts.append("<pre><code>" + "\n".join(code_lines) + "</code></pre>")
            continue

        if line.startswith("- "):
            item_lines = []
            while index < len(lines) and (lines[index].startswith("- ")
                                          or lines[index].startswith("  ")):
                item_lines.append(lines[index])
                index += 1
            items: list[str] = []
            for item in item_lines:
                if item.startswith("- "):
                    items.append(item[2:])
                else:
                    items[-1] += " " + item.strip()
            parts.append("<ul>" + "".join(f"<li>{_render_inline(i)}</li>" for i in items) + "</ul>")
            continue

        heading = re.match(r"^(#{1,3}) (.+)$", line)
        if heading:
            level = len(heading.group(1))
            parts.append(f"<h{level}>{_render_inline(heading.group(2))}</h{level}>")
        elif line.strip() == "---":
            parts.append("<hr>")
        elif line.strip():
            paragraph = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and not re.match(
                r"^(\||#|-|```|---)", lines[index]
            ):
                paragraph.append(lines[index])
                index += 1
            parts.append("<p>" + _render_inline(" ".join(paragraph)) + "</p>")
            continue
        index += 1

    return "\n".join(parts)


def main() -> None:
    args = _parse_args()
    source_path = pathlib.Path(args.source)
    output_path = pathlib.Path(args.output)

    style = (_PAGE_STYLE
             .replace("{FIRST_COLUMN_WIDTH}", args.first_column_width)
             .replace("{BODY_FONT_SIZE}", args.body_font_size)
             .replace("{TABLE_FONT_SIZE}", args.table_font_size)
             .replace("{LINE_HEIGHT}", args.line_height))

    body = _render_document(source_path.read_text())
    output_path.write_text(
        "<!doctype html>\n<html lang=\"ja\"><head><meta charset=\"utf-8\">\n"
        "<title>PARC2026 予選レポート</title>\n"
        f"<style>{style}</style></head>\n<body>\n{body}\n</body></html>\n"
    )
    print(f"書き出した: {output_path}")
    print(f"  1 列目の幅 {args.first_column_width} / 本文 {args.body_font_size}"
          f" / 表 {args.table_font_size} / 行間 {args.line_height}")
    print("ブラウザで開いて「PDF に書き出す」を選ぶ。**ページ数をその場で確認できる。**")


if __name__ == "__main__":
    main()
