#!/usr/bin/env python3
"""混ぜたデータセットの正規化統計が正しいかを検算する。

**なぜ要るか。**pi05 は `observation.state` と `action` を **mean/std で正規化する**
（`norm_map` が `MEAN_STD`。**rp3000 と all111 の `config.json` で確認済み**）。
そして `lerobot_train.py` は **`normalizer_processor` の `stats` を
`dataset.meta.stats` で上書きする**（315〜332 行）。
**つまり混ぜた瞬間に、配布データの正規化も変わる。**どれだけ変わるかを数で押さえる。

使い方は 3 つ。

1. **式の検算**——2 つ以上の元データから合成した統計が、実際に合成された
   データセットの `meta/stats.json` と一致するかを見る。

       python check_merge_stats.py verify --parts A B C --merged D

2. **混ぜる前の予測**——配布データに手元のデータを N 本コピーで足したときの
   統計と、正規化の歪みを表にする。`--provided` には
   `meta/stats.json` か、重みに同梱された
   `policy_preprocessor_step_*_normalizer_processor.safetensors` を渡せる。

       python check_merge_stats.py predict --provided <stats か safetensors> \
           --ours <stats.json> --copies 1 5 11 22 45

3. **実測との突き合わせ**——`predict` に `--merged` を足すと、予測と
   実際の `meta/stats.json` を並べる。

`aggregate_feature_stats`（lerobot の `datasets/compute_stats.py`）と同じ式を使う。
**平均はコマ数の重み付き平均、分散は群間の散らばりを足した式**である。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("action", "observation.state")


def load_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """`meta/stats.json` でも、重みの normalizer の safetensors でも読む。"""
    if path.is_dir():
        path = path / "meta" / "stats.json"
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        flat = load_file(str(path))
        out: dict[str, dict[str, np.ndarray]] = {}
        for name, value in flat.items():
            feature, _, stat = name.rpartition(".")
            out.setdefault(feature, {})[stat] = np.asarray(value, dtype=np.float64)
        return out
    raw = json.loads(path.read_text())
    return {
        feature: {stat: np.asarray(value, dtype=np.float64)
                  for stat, value in stats.items()}
        for feature, stats in raw.items()
    }


def pool(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """lerobot の `aggregate_feature_stats` と同じ式で 1 列ぶんを合成する。"""
    means = np.stack([p["mean"] for p in parts])
    variances = np.stack([p["std"] ** 2 for p in parts])
    counts = np.stack([np.asarray(p["count"]).reshape(-1) for p in parts])
    total = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    mean = (means * counts).sum(axis=0) / total
    var = ((variances + (means - mean) ** 2) * counts).sum(axis=0) / total
    return {"mean": mean, "std": np.sqrt(var), "count": total}


def report_diff(name: str, got: np.ndarray, want: np.ndarray, tol: float) -> bool:
    """食い違いを 1 行で印字する。**合っていれば True。**"""
    err = np.abs(got - want)
    rel = err / np.maximum(np.abs(want), 1e-12)
    ok = bool(np.all(err <= tol) or np.all(rel <= tol))
    mark = "OK  " if ok else "ちがう"
    print(f"  {mark} {name}: 最大差 {err.max():.3e}（相対 {rel.max():.3e}）")
    if not ok:
        print(f"        予測 {np.array2string(got, precision=6)}")
        print(f"        実測 {np.array2string(want, precision=6)}")
    return ok


def command_verify(args: argparse.Namespace) -> None:
    """元データから合成した統計が、合成済みデータセットと一致するか見る。"""
    parts = [load_stats(Path(p)) for p in args.parts]
    merged = load_stats(Path(args.merged))
    print(f"=== {len(parts)} 個を合成した予測 vs {args.merged} の実測 ===")
    good = True
    for key in KEYS:
        got = pool([p[key] for p in parts])
        want = merged[key]
        print(f"{key}（コマ数 予測 {int(got['count'][0])} / 実測 {int(np.asarray(want['count']).reshape(-1)[0])}）")
        good &= report_diff("mean", got["mean"], want["mean"], args.tol)
        good &= report_diff("std ", got["std"], want["std"], args.tol)
    print("=== 一致した ===" if good else "=== 食い違いがある ===")
    raise SystemExit(0 if good else 1)


def command_predict(args: argparse.Namespace) -> None:
    """コピー本数ごとの統計と、正規化の歪みを表にする。"""
    provided = load_stats(Path(args.provided))
    ours = load_stats(Path(args.ours))
    merged = load_stats(Path(args.merged)) if args.merged else None

    for key in KEYS:
        p, o = provided[key], ours[key]
        n_p = float(np.asarray(p["count"]).reshape(-1)[0])
        n_o = float(np.asarray(o["count"]).reshape(-1)[0])
        print(f"\n=== {key} ===")
        print(f"  配布 {int(n_p):,} コマ / こちら {int(n_o):,} コマ")
        for copies in args.copies:
            share = copies * n_o / (n_p + copies * n_o)
            got = pool([p, {**o, "count": np.array([n_o * copies])}])
            # 正規化の歪み。**配布データ側の見え方がどれだけずれるか。**
            #   もとは (x - mean_p) / std_p が平均 0・分散 1 だった。
            #   混ぜた後は (x - mean) / std なので、平均が (mean_p - mean)/std へ、
            #   分散が (std_p/std)^2 へ動く。
            bias = (p["mean"] - got["mean"]) / got["std"]
            scale = p["std"] / got["std"]
            bias_o = (o["mean"] - got["mean"]) / got["std"]
            scale_o = o["std"] / got["std"]
            print(f"  N={copies:<3d} 割合 {share:6.2%}  "
                  f"配布のずれ |bias| 最大 {np.abs(bias).max():.4f}・"
                  f"倍率 {scale.min():.4f}〜{scale.max():.4f}  ｜  "
                  f"こちらのずれ |bias| 最大 {np.abs(bias_o).max():.4f}・"
                  f"倍率 {scale_o.min():.4f}〜{scale_o.max():.4f}")
            if args.verbose:
                print(f"        mean {np.array2string(got['mean'], precision=5)}")
                print(f"        std  {np.array2string(got['std'], precision=5)}")
            if merged is not None and copies == args.copies[-1]:
                print("     ↓ 実測との突き合わせ")
                report_diff("mean", got["mean"], merged[key]["mean"], args.tol)
                report_diff("std ", got["std"], merged[key]["std"], args.tol)


def command_freeze(args: argparse.Namespace) -> None:
    """混ぜたデータセットの正規化統計を、配布データのものに据え置く。

    **なぜ据え置くか。**pi05 は state と action を mean/std で正規化し、
    `lerobot_train.py` は **その値を `dataset.meta.stats` から取る**。
    だから混ぜたままだと、**配布データ側の正規化まで変わってしまう**
    （こちらの回転指令が配布の 4〜5 倍広いので、割合 20% で
    配布の回転チャンネルが 0.43 倍に縮む）。**据え置けば、111 課題の
    見え方は混ぜる前と 1 ビットも変わらない。**

    **書き換えるのは `observation.state` と `action` だけである。**画像は
    `VISUAL: IDENTITY` で正規化に使われず、`index` などは方策の入力ではない。

    Generates:
      - <merged>/meta/stats.json（書き換える）
      - <merged>/meta/stats.json.bak（書き換える前のもの）
    """
    target = Path(args.merged) / "meta" / "stats.json"
    original = target.read_text()
    raw = json.loads(original)
    source = load_stats(Path(args.source))

    for key in KEYS:
        if key not in raw:
            raise SystemExit(f"{target} に {key} が無い")
        if key not in source:
            raise SystemExit(f"{args.source} に {key} が無い")
        before = np.asarray(raw[key]["std"], dtype=np.float64)
        after = source[key]["std"]
        if before.shape != after.shape:
            raise SystemExit(f"{key} の形が違う: {before.shape} と {after.shape}")
        ratio = before / after
        print(f"  {key}: std の倍率 {ratio.min():.4f}〜{ratio.max():.4f} を 1.0 に戻す")
        raw[key] = {stat: value.tolist() for stat, value in source[key].items()}

    target.with_suffix(".json.bak").write_text(original)
    target.write_text(json.dumps(raw, indent=4))
    print(f"=== 据え置いた。元は {target.with_suffix('.json.bak').name} にある ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="合成した統計が実測と一致するか")
    v.add_argument("--parts", nargs="+", required=True, metavar="DIR")
    v.add_argument("--merged", required=True, metavar="DIR")
    v.add_argument("--tol", type=float, default=1e-5)
    v.set_defaults(func=command_verify)

    p = sub.add_parser("predict", help="コピー本数ごとの統計と歪み")
    p.add_argument("--provided", required=True, metavar="PATH")
    p.add_argument("--ours", required=True, metavar="PATH")
    p.add_argument("--copies", type=int, nargs="+", default=[1, 5, 11, 22, 45])
    p.add_argument("--merged", default=None, metavar="DIR")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=command_predict)

    f = sub.add_parser("freeze", help="正規化統計を配布データのものに据え置く")
    f.add_argument("--merged", required=True, metavar="DIR")
    f.add_argument("--source", required=True, metavar="PATH",
                   help="据え置く値の出どころ（配布データの meta/stats.json か "
                        "重みの normalizer の safetensors）")
    f.set_defaults(func=command_freeze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
