#!/usr/bin/env python3
"""複数のチェックポイントの重みを平均して 1 本の safetensors にする。

**なぜ効くと考えたか。** 8/27 の 24,000 歩の学習で、loss は 6,000 歩から
-1.0〜-3.3% しか動かなかったのに、行動側の重みは 0.4〜3.4% 動いた
（`docs/reference/19_weights_catalog.md`）。**谷が平らな場所を横に動いている**
状態で、平均を取ると谷の中心に寄る。

**ホストに torch が無いので numpy だけで書いた。** safetensors の形式
（先頭 8 バイトがヘッダ長・続いて JSON ヘッダ・その後が本体）を直接読む。

**凍結側はビット単位で同じなので、バイト列をそのまま写す。**
`train_expert_only=true` の学習では PaliGemma 側が 1 ビットも動かないため、
変換を通さずに写せる。動いたテンソルだけを fp32 に上げて平均し、元の dtype へ戻す。

    python3 average_checkpoints.py --out <出力ディレクトリ> \
        <入力1>/model.safetensors <入力2>/model.safetensors [...]
    python3 average_checkpoints.py --check <入力1>/... <入力2>/...   # 書かずに調べる

`--weights 0.5,0.5` で重みを指定できる（既定は等分）。出力ディレクトリには
`model.safetensors` だけを書くので、**`config.json` などの小さなファイルは
呼び出し側が最初の入力からコピーする**（`--copy-siblings` で自動化）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

_DT = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),   # numpy に bf16 が無いので生ビットで扱う
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "U8": np.dtype("|u1"),
    "BOOL": np.dtype("|u1"),
}
# 平均を取らずに 1 本目をそのまま写す dtype（整数・真偽値）
_NON_FLOAT = {"I64", "I32", "U8", "BOOL"}


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """最近接偶数への丸め。切り捨てると 24,000 歩ぶんの差より大きい誤差が乗る。"""
    u = np.ascontiguousarray(x, dtype="<f4").view(np.uint32)
    u = u + (((u >> 16) & 1) + np.uint32(0x7FFF))
    return (u >> 16).astype("<u2")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path, help="model.safetensors を 2 本以上")
    ap.add_argument("--out", type=Path, default=None, help="出力ディレクトリ")
    ap.add_argument("--weights", type=str, default=None, help="例 0.5,0.5（既定は等分）")
    ap.add_argument("--check", action="store_true", help="書かずに、調べた結果だけ出す")
    ap.add_argument("--copy-siblings", action="store_true",
                    help="1 本目と同じディレクトリの他のファイルも出力先へ写す")
    a = ap.parse_args()

    if len(a.inputs) < 2:
        print("入力は 2 本以上要る", file=sys.stderr)
        return 2
    for p in a.inputs:
        if not p.is_file():
            print(f"見つからない: {p}", file=sys.stderr)
            return 2

    hdrs, starts = [], []
    for p in a.inputs:
        h, s = read_header(p)
        hdrs.append(h)
        starts.append(s)

    keysets = [set(h) - {"__metadata__"} for h in hdrs]
    if len(set(map(frozenset, keysets))) != 1:
        only = keysets[0].symmetric_difference(keysets[1])
        print(f"キーが一致しない（例: {sorted(only)[:3]}）", file=sys.stderr)
        return 2
    keys = sorted(keysets[0])

    for k in keys:
        shapes = {tuple(h[k]["shape"]) for h in hdrs}
        dtypes = {h[k]["dtype"] for h in hdrs}
        if len(shapes) != 1 or len(dtypes) != 1:
            print(f"{k} の形か dtype が食い違う: {shapes} {dtypes}", file=sys.stderr)
            return 2

    if a.weights:
        w = np.array([float(x) for x in a.weights.split(",")], dtype=np.float64)
        if len(w) != len(a.inputs):
            print("--weights の個数が入力の本数と合わない", file=sys.stderr)
            return 2
    else:
        w = np.full(len(a.inputs), 1.0 / len(a.inputs), dtype=np.float64)
    w = w / w.sum()

    files = [p.open("rb") for p in a.inputs]
    try:
        # --- 1 周目: どのテンソルが動いているかを数える ---
        moved, same, nbytes_total = [], [], 0
        for k in keys:
            info = hdrs[0][k]
            o0, o1 = info["data_offsets"]
            n = o1 - o0
            nbytes_total += n
            blobs = []
            for f, h, st in zip(files, hdrs, starts):
                b0, b1 = h[k]["data_offsets"]
                f.seek(st + b0)
                blobs.append(f.read(b1 - b0))
            (same if all(b == blobs[0] for b in blobs[1:]) else moved).append(k)

        print(f"テンソル {len(keys)} 個 / 本体 {nbytes_total:,} バイト")
        print(f"  同じ（写すだけ）  {len(same):>4} 個")
        print(f"  動いた（平均する）{len(moved):>4} 個")
        mb = sum(hdrs[0][k]['data_offsets'][1] - hdrs[0][k]['data_offsets'][0] for k in moved)
        print(f"  動いた分のバイト  {mb:,}（全体の {100*mb/nbytes_total:.1f}%）")
        print(f"  重み {', '.join(f'{x:.4f}' for x in w)}")
        if a.check:
            return 0
        if a.out is None:
            print("--out が要る", file=sys.stderr)
            return 2

        # --- 2 周目: 書く ---
        a.out.mkdir(parents=True, exist_ok=True)
        dst = a.out / "model.safetensors"
        tmp = dst.with_suffix(".safetensors.new")
        header = {k: dict(hdrs[0][k]) for k in keys}
        if "__metadata__" in hdrs[0]:
            header["__metadata__"] = hdrs[0]["__metadata__"]
        # オフセットを詰め直す（入力と同じ順序・同じ大きさなので実質そのまま）
        pos = 0
        for k in keys:
            n = hdrs[0][k]["data_offsets"][1] - hdrs[0][k]["data_offsets"][0]
            header[k]["data_offsets"] = [pos, pos + n]
            pos += n
        blob = json.dumps(header, separators=(",", ":")).encode()
        pad = (-len(blob)) % 8
        blob += b" " * pad

        with tmp.open("wb") as g:
            g.write(struct.pack("<Q", len(blob)))
            g.write(blob)
            for k in keys:
                dt = hdrs[0][k]["dtype"]
                if k in set(same) or dt in _NON_FLOAT:
                    f, h, st = files[0], hdrs[0], starts[0]
                    b0, b1 = h[k]["data_offsets"]
                    f.seek(st + b0)
                    g.write(f.read(b1 - b0))
                    continue
                acc = None
                for f, h, st, wi in zip(files, hdrs, starts, w):
                    b0, b1 = h[k]["data_offsets"]
                    f.seek(st + b0)
                    raw = np.frombuffer(f.read(b1 - b0), dtype=_DT[dt])
                    x = bf16_to_f32(raw) if dt == "BF16" else raw.astype(np.float32)
                    acc = x * wi if acc is None else acc + x * wi
                out = f32_to_bf16(acc) if dt == "BF16" else acc.astype(_DT[dt])
                g.write(out.tobytes())
        tmp.replace(dst)
        print(f"書いた: {dst}  {dst.stat().st_size:,} バイト")

        if a.copy_siblings:
            src_dir = a.inputs[0].parent
            for p in sorted(src_dir.iterdir()):
                if p.name == "model.safetensors" or not p.is_file():
                    continue
                shutil.copy2(p, a.out / p.name)
                print(f"  写した: {p.name}")
    finally:
        for f in files:
            f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
