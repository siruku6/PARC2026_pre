#!/usr/bin/env python3
"""1 往復の待ち時間が何で決まっているかを、会話ログから測る。

**8/29 に「質問するたびに待たされる」原因を切り分けるために書いた。**
そのとき分かったのは次の 3 つで、**同じことをもう一度手で書き下ろさないために道具にした。**

- 1 往復の長さを決めているのは**私が生成した字数**である（相関 0.93・1 万字あたり 3.2 分）
- **context の大きさはほぼ効かない**（10 万を超えると横ばい。だから小まめな圧縮は速さに効かない）
- **探して空振りした回は少ない**（14%。しかも多くは短い出力が正しく返った回）

    python3 measure_turn_cost.py                 # 今日
    python3 measure_turn_cost.py --date 08-28    # 日を指定（JST の月-日）
    python3 measure_turn_cost.py --ctx           # context の大きさ別の速さ（全セッション）

ログの置き場は `~/.claude/projects/<スラッグ>/*.jsonl`。**既定は lab のもの。**
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path

JST = timedelta(hours=9)
_DEFAULT_LOGS = Path.home() / ".claude/projects/-home-ubuntu-lab"


def _load(path: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue          # 書き込み途中の行が混じることがある
    return rows


def _ts(d: dict):
    t = d.get("timestamp")
    return datetime.fromisoformat(t.replace("Z", "+00:00")) + JST if t else None


def _is_human(d: dict) -> bool:
    """ユーザが打った行だけを真にする（道具の結果を除く）。"""
    if d.get("type") != "user" or d.get("toolUseResult") is not None:
        return False
    c = d.get("message", {}).get("content")
    if isinstance(c, str):
        return True
    if isinstance(c, list):
        return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)
    return False


def _kind(name: str, inp: dict) -> str:
    """Bash の中身を、書く／その場の python／読む・探す／git に分ける。"""
    if name != "Bash":
        return f"{name} ツール"
    cmd = inp.get("command", "")
    # **書くか調べるかは、python の中身まで見ないと分かれない。**
    # `python3 - <<PY ... write_text` は「その場の python」ではなく「ファイルを書く」である
    # （8/29 に最初の版がこれを取り違え、内訳を 8% と 32% で言い違えた）。
    writes = re.search(r"cat\s*>\s|tee\s|>\s*\S+\.(md|py|sh|json|csv)"
                       r"|write_text|open\([^)]*['\"][wa]", cmd)
    if re.search(r"python3?\s+-\s*<<|python3?\s+-c", cmd):
        return "Bash:python でファイルを書く" if writes else "Bash:その場の python（調べる）"
    if writes:
        return "Bash:ファイルを書く"
    if cmd.strip().startswith("git") or " git " in cmd:
        return "Bash:git"
    return "Bash:読む・探す"


def per_day(rows: list[dict], day: str) -> None:
    idx = [i for i, d in enumerate(rows) if _is_human(d)]
    gen_by_kind: collections.Counter = collections.Counter()
    read_total = 0
    pts = []                                   # (分, 生成字数, 呼び出し回数)
    for k, i in enumerate(idx):
        st = _ts(rows[i])
        if not st or st.strftime("%m-%d") != day:
            continue
        seg = rows[i + 1:(idx[k + 1] if k + 1 < len(idx) else len(rows))]
        asst = [d for d in seg if d.get("type") == "assistant" and _ts(d)]
        if not asst:
            continue
        gen = calls = 0
        for d in asst:
            for b in d.get("message", {}).get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") in ("text", "thinking"):
                    n = len(b.get(b["type"], ""))
                    gen += n
                    gen_by_kind["返信の本文" if b["type"] == "text" else "考えたこと"] += n
                elif b.get("type") == "tool_use":
                    n = len(json.dumps(b.get("input", {}), ensure_ascii=False))
                    gen += n
                    calls += 1
                    gen_by_kind[_kind(b.get("name", "?"), b.get("input", {}))] += n
        for d in seg:
            if d.get("toolUseResult") is not None:
                read_total += len(json.dumps(d["toolUseResult"], ensure_ascii=False))
        dur = (max(_ts(d) for d in asst) - st).total_seconds() / 60
        if dur > 0:
            pts.append((dur, gen, calls))

    if not pts:
        print(f"{day} の往復が無い")
        return
    dur = [p[0] for p in pts]
    gen = [p[1] for p in pts]
    calls = [p[2] for p in pts]

    def corr(x, y):
        mx, my = statistics.mean(x), statistics.mean(y)
        sx = sum((a - mx) ** 2 for a in x) ** .5
        sy = sum((b - my) ** 2 for b in y) ** .5
        return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0.0

    print(f"== {day}（JST）  往復 {len(pts)} 回")
    print(f"   私が答えていた時間の合計 {sum(dur) / 60:.1f} 時間"
          f"（中位 {statistics.median(dur):.1f} 分・最長 {max(dur):.1f} 分）")
    print(f"   私が生成した字数 {sum(gen):,}   道具が返してきた字数 {read_total:,}")
    print(f"   1 往復あたり返信の本文 {gen_by_kind['返信の本文'] // len(pts):,} 字")
    print()
    print("   生成量の内訳")
    for k, v in gen_by_kind.most_common():
        print(f"     {k:<22} {v:8,} 字  {100 * v / sum(gen):3.0f}%")
    print()
    print(f"   相関（1.0 に近いほど待ち時間を決めている）"
          f"  生成字数 {corr(dur, gen):.2f}   呼び出し回数 {corr(dur, calls):.2f}")
    slope = sum(a * b for a, b in zip(dur, gen)) / sum(b * b for b in gen) * 10000
    print(f"   傾き  生成 1 万字あたり {slope:.1f} 分")


def by_context(paths: list[str]) -> None:
    """context の大きさ別に、出力 1,000 トークンあたりの秒数を出す。"""
    buck = collections.defaultdict(list)
    for p in paths:
        rows = _load(p)
        for a, b in zip(rows, rows[1:]):
            if b.get("type") != "assistant":
                continue
            ta, tb = _ts(a), _ts(b)
            if not ta or not tb:
                continue
            gap = (tb - ta).total_seconds()
            if not 0 < gap <= 300:
                continue
            u = b.get("message", {}).get("usage") or {}
            ctx = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                   + u.get("cache_creation_input_tokens", 0))
            out = u.get("output_tokens", 0)
            if ctx < 1000 or out < 50:
                continue
            buck[min(ctx // 100000, 6)].append(gap / out * 1000)
    print("== context の大きさ別（全セッション）")
    print("   %-12s %7s %14s" % ("context", "件数", "出力1kあたり秒"))
    for k in sorted(buck):
        v = buck[k]
        print("   %-12s %7d %14.1f" % (f"{k * 100}-{(k + 1) * 100}k", len(v), statistics.median(v)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", default=str(_DEFAULT_LOGS), help="会話ログのディレクトリ")
    ap.add_argument("--date", help="JST の月-日（既定は今日）")
    ap.add_argument("--ctx", action="store_true", help="context の大きさ別の速さも出す")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.logs) / "*.jsonl")))
    if not paths:
        raise SystemExit(f"会話ログが無い: {args.logs}")
    day = args.date or (datetime.utcnow() + JST).strftime("%m-%d")

    newest = max(paths, key=lambda p: Path(p).stat().st_mtime)
    per_day(_load(newest), day)
    if args.ctx:
        print()
        by_context(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
