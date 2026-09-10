#!/usr/bin/env python3
"""提出用 zip の中の `policy_server.py` を、長さを変えずに書き換える。

**9.4 GB を組み直さずに 1 行だけ直すための道具である。**

`build_submission.py` が書く zip では **`policy_server.py` が無圧縮（method=0）で
入っている**（2026-08-28 に `submission_5.0.0.zip` の実物で確認）。
だから**同じ長さの置き換えなら、その場でバイトを差し替えて CRC を直すだけで済む。**
オフセットも項目の大きさも動かないので、zip の構造に手を入れる必要が無い。

**長さが変わる置き換えは受け付けない。** 変わると後続の全項目のオフセットがずれ、
中央ディレクトリを作り直すことになる——それは `build_submission.py` の仕事である。
`0.65` → `0.50` や `"4.6.0"` → `"7.0.0"` のように**桁数が同じ置き換えだけ**を通す。

    python3 patch_zip_policy_server.py <zip> \
        --replace '_MIN_LEXICAL = 0.65' '_MIN_LEXICAL = 0.50' \
        --replace '__version__ = "4.6.0"' '__version__ = "7.0.0"'

    python3 patch_zip_policy_server.py <zip> --show      # 中の設定を出すだけ

**書き換える前に必ず `--show` で中を見ること。** どの版のどの設定が入っているかは、
ファイル名からは分からない（5.0.0 と 6.0.0 で `policy_server.py` は同一だった）。

**`--in-place` を付けない限り書き換えない。** 付けたときは元のファイルを
`<zip>.bak` として残さず、**直接書き換える**（9.4 GB の複製を作らないため）。
**やり直せるように、書き換える前の sha256 を必ず印字する。**
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
import zlib
from pathlib import Path

# 中を見るときに出す設定。`--show` と、書き換えの前後で使う。
_INTERESTING = (
    r'__version__ = "[^"]+"',
    r"^_MIN_LEXICAL = [0-9.]+",
    r"^_MIN_LEXICAL_MARGIN = [0-9.]+",
    r"^_USE_EMBEDDING = \w+",
    r'"PARC_NUM_INFERENCE_STEPS", "[^"]*"',
    r'"PARC_N_ACTION_STEPS", "[^"]*"',
    r'"PARC_BLEND_STEPS", "[^"]*"',
    r'"PARC_EMA_ALPHA", "[^"]*"',
    r'"PARC_SAMPLES", "[^"]*"',
    r'"PARC_GAIN_TRANSLATION", "[^"]*"',
    r'"PARC_GAIN_ROTATION", "[^"]*"',
    r'"PARC_SEED", "[^"]*"',
    r'"PARC_NORMALIZE_INSTRUCTION", "[^"]*"',
    r'"PARC_RTC", "[^"]*"',
    r'"PARC_ADAPTIVE_CADENCE", "[^"]*"',
    r'"PARC_ADAPTIVE_CADENCE_HIGH", "[^"]*"',
)

_TARGET = "policy_server.py"


class Entry:
    """`policy_server.py` の 1 項目ぶんの位置。"""

    def __init__(self, local_offset: int, central_offset: int, method: int,
                 csize: int, usize: int, crc: int, data_offset: int) -> None:
        self.local_offset = local_offset
        self.central_offset = central_offset
        self.method = method
        self.csize = csize
        self.usize = usize
        self.crc = crc
        self.data_offset = data_offset


def _find_entry(fh) -> Entry:
    """中央ディレクトリから `policy_server.py` を探す。ZIP64 でも動く。"""
    fh.seek(0, 2)
    size = fh.tell()
    tail_len = min(size, 1 << 20)
    fh.seek(size - tail_len)
    tail = fh.read(tail_len)

    found = None
    i = 0
    while True:
        i = tail.find(b"PK\x01\x02", i)
        if i < 0:
            break
        v = struct.unpack_from("<IHHHHHHIIIHHHHHII", tail, i)
        method, crc, csize, usize = v[4], v[7], v[8], v[9]
        nlen, elen, clen, local = v[10], v[11], v[12], v[16]
        name = tail[i + 46: i + 46 + nlen].decode("utf-8", "replace")
        if name == _TARGET:
            found = (size - tail_len + i, method, crc, csize, usize, local)
        i += 46 + nlen + elen + clen

    if found is None:
        raise SystemExit(f"{_TARGET} が zip の中に無い")
    central_offset, method, crc, csize, usize, local = found

    fh.seek(local)
    head = fh.read(30)
    if head[:4] != b"PK\x03\x04":
        raise SystemExit(f"ローカルヘッダが壊れている（位置 {local}）")
    nlen, elen = struct.unpack_from("<HH", head, 26)
    return Entry(local, central_offset, method, csize, usize, crc, local + 30 + nlen + elen)


def _show(text: str, label: str) -> None:
    print(f"== {label}")
    for pat in _INTERESTING:
        for m in re.finditer(pat, text, re.M):
            print("   ", m.group(0).strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip", type=Path)
    ap.add_argument("--replace", nargs=2, action="append", metavar=("FROM", "TO"),
                    help="長さが同じ置き換え。何度でも指定できる")
    ap.add_argument("--show", action="store_true", help="中の設定を出すだけ")
    ap.add_argument("--in-place", action="store_true",
                    help="実際に書き換える。付けないと何も書かない")
    args = ap.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"zip が無い: {args.zip}")

    with args.zip.open("rb") as fh:
        entry = _find_entry(fh)
        if entry.method != 0:
            raise SystemExit(
                f"{_TARGET} が無圧縮ではない（method={entry.method}）。"
                "この道具は使えないので build_submission.py で組み直すこと")
        fh.seek(entry.data_offset)
        original = fh.read(entry.csize)

    if zlib.crc32(original) & 0xFFFFFFFF != entry.crc:
        raise SystemExit("読み出した中身の CRC が中央ディレクトリと合わない")

    text = original.decode("utf-8")
    _show(text, f"{args.zip.name} のいま")

    if args.show or not args.replace:
        if not args.replace:
            print("\n--replace が無いので何もしない")
        return 0

    for src, dst in args.replace:
        if len(src) != len(dst):
            raise SystemExit(
                f"長さが違う置き換えは通せない: {len(src)} → {len(dst)}\n"
                f"  {src!r}\n  {dst!r}\n"
                "  桁数を合わせるか、build_submission.py で組み直すこと")
        count = text.count(src)
        if count != 1:
            raise SystemExit(f"{src!r} が {count} 箇所ある（1 箇所でないと危ない）")
        text = text.replace(src, dst, 1)

    new = text.encode("utf-8")
    if len(new) != len(original):
        raise SystemExit(f"長さが変わった: {len(original)} → {len(new)}")
    if new == original:
        print("\n中身が変わらなかった。何もしない")
        return 0

    new_crc = zlib.crc32(new) & 0xFFFFFFFF
    print(f"\n置き換え {len(args.replace)} 件。CRC {entry.crc:08x} → {new_crc:08x}")
    _show(text, f"{args.zip.name} の書き換え後")

    if not args.in_place:
        print("\n**--in-place が無いので書いていない。**中身を確かめてから付け直すこと")
        return 0

    before = hashlib.sha256(args.zip.read_bytes()).hexdigest()
    print(f"\n書き換える前の sha256: {before}")

    with args.zip.open("r+b") as fh:
        fh.seek(entry.data_offset)
        fh.write(new)
        # ローカルヘッダの CRC（先頭から 14 バイト目）
        fh.seek(entry.local_offset + 14)
        fh.write(struct.pack("<I", new_crc))
        # 中央ディレクトリの CRC（先頭から 16 バイト目）
        fh.seek(entry.central_offset + 16)
        fh.write(struct.pack("<I", new_crc))
        fh.flush()

    after = hashlib.sha256(args.zip.read_bytes()).hexdigest()
    print(f"書き換えた後の sha256: {after}")

    # 標準の zip 実装で読み直せることを確かめる。**ここを通らないものは出さない。**
    import zipfile
    with zipfile.ZipFile(args.zip) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"壊れた項目がある: {bad}")
        got = zf.read(_TARGET)
    if got != new:
        raise SystemExit("読み直した中身が書いたものと違う")
    print(f"zipfile で読み直して一致。項目数 {len(zipfile.ZipFile(args.zip).namelist())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
