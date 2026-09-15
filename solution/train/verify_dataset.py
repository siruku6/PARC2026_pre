#!/usr/bin/env python3
"""混ぜたデータセットを HF へ上げる前に 7 つ確かめる。

**「確かめてから流す」と書くだけでは何も確かめられない。**2026-09-08 に
`jaist_dataset_build.sh` がそう書いていて、何を見るのかが書かれていなかった。
上げてしまうと取り消しに手間がかかるので、**機械で見て 1 つでも落ちたら 1 で終わる。**

    python3 verify_dataset.py <データセット> --norm <重みの normalizer> \
        --expect "5723 975828 43" --rev-tasks 3

見るもの:

===  ===============================================================
 1   本数・コマ数・課題数が期待どおりか
 2   fps・codebase_version・video_path が学習の前提どおりか
 3   **正規化が据え置かれ、出発点の重みと 1 ビットも違わないか**
 4   逆向きの指示文が期待の数だけ入っているか
 5   parquet が読めて、`action` の形が (N,7) か
 6   **動画が本当に復号できるか**（`build` は壊れた動画でも通る）
 7   読めないファイルが無いか（docker が root で書いた分が残っていないか）
===  ===============================================================

**6 は docker が要る**（`av` はこのマシンの素の python に無い）。
`--skip-video` で飛ばせるが、**飛ばしたら学習で初めて落ちる。**
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OK, NG = "  ✓", "  ✗"
_failed = 0


def check(passed: bool, label: str, detail: str = "") -> None:
    """1 行出して、落ちていたら数える。**落ちても続ける**——全部見せてから終わる。"""
    global _failed
    if not passed:
        _failed += 1
    print(f"{OK if passed else NG} {label}" + (f"   {detail}" if detail else ""))


def load_norm(path: Path) -> dict[str, np.ndarray]:
    """重みに同梱された normalizer の safetensors を読む。"""
    from safetensors.numpy import load_file

    return load_file(path)


def check_counts(root: Path, expect: str | None) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    got = (info["total_episodes"], info["total_frames"], info["total_tasks"])
    if expect:
        want = tuple(int(x) for x in expect.split())
        check(got == want, "本数・コマ数・課題数", f"{got} / 期待 {want}")
    else:
        print(f"{OK} 本数・コマ数・課題数   {got}（期待の指定なし）")
    check(
        info["fps"] == 20 and info["codebase_version"] == "v3.0",
        "fps と版",
        f"fps {info['fps']} / {info['codebase_version']}",
    )
    # **配布 tar は 6 桁で、lerobot は 3 桁でしか読まない**（reference/27 の 5 節）。
    check(
        "file_index:03d" in info["video_path"],
        "動画パスの桁",
        info["video_path"],
    )
    return info


def check_norm(root: Path, norm_path: Path) -> None:
    """**据え置きが効いているか。**出発点の重みと差 0 でなければ意味が無い。"""
    if not (root / "meta/stats.json.bak").exists():
        check(False, "据え置き", "meta/stats.json.bak が無い＝freeze を打っていない")
        return
    weights = load_norm(norm_path)
    stats = json.loads((root / "meta/stats.json").read_text())
    worst = 0.0
    for key in ("observation.state", "action"):
        for stat in ("mean", "std"):
            hit = [k for k in weights if key in k and stat in k]
            if not hit:
                check(False, f"据え置き {key}.{stat}", "重み側に見つからない")
                return
            got = np.asarray(stats[key][stat], dtype=np.float64).ravel()
            want = np.asarray(weights[hit[0]]).ravel()
            worst = max(worst, float(np.abs(got - want).max()))
    check(worst == 0.0, "据え置きが重みと一致", f"最大差 {worst:.3e}")


def check_tasks(root: Path, rev_tasks: int | None) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    col = table.get("task") or next(iter(table.values()))
    rev = [x for x in col if str(x).startswith("take the")]
    if rev_tasks is None:
        print(f"{OK} 逆向きの指示文   {len(rev)} 件（期待の指定なし）")
    else:
        check(len(rev) == rev_tasks, "逆向きの指示文", f"{len(rev)} 件 / 期待 {rev_tasks}")
    for x in rev:
        print(f"      {x}")


def check_parquet(root: Path, info: dict) -> None:
    import pyarrow.parquet as pq

    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        check(False, "parquet", "1 つも無い")
        return
    table = pq.read_table(files[0], columns=["action", "observation.state"])
    act = np.stack(table["action"].to_numpy())
    st = np.stack(table["observation.state"].to_numpy())
    check(
        act.shape[1] == 7 and st.shape[1] == 8,
        "列の形",
        f"action {act.shape} / state {st.shape}（{len(files)} ファイル）",
    )
    check(
        bool(np.isfinite(act).all() and np.isfinite(st).all()),
        "NaN・inf が無い",
    )


def check_videos(root: Path, image: str) -> None:
    """**`build` は壊れた動画でも通る。**読み出しはここで初めて落ちる。"""
    code = (
        "import av, glob\n"
        "for k in ('front','wrist'):\n"
        "    f = sorted(glob.glob(f'/d/videos/observation.images.{k}/**/*.mp4', recursive=True))\n"
        "    if not f: print(k, 'NONE'); continue\n"
        "    n0 = sum(1 for _ in av.open(f[0]).decode(video=0))\n"
        "    n1 = sum(1 for _ in av.open(f[-1]).decode(video=0))\n"
        "    print(k, len(f), n0, n1)\n"
    )
    proc = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{root.resolve()}:/d:ro",
         "--entrypoint", "python", image, "-c", code],
        capture_output=True, text=True,
    )
    lines = [x for x in proc.stdout.splitlines() if x.split()[:1] in (["front"], ["wrist"])]
    if proc.returncode != 0 or len(lines) != 2:
        check(False, "動画の復号", (proc.stderr or proc.stdout).strip()[-200:])
        return
    detail = " / ".join(lines)
    check(all("NONE" not in x for x in lines), "動画の復号", detail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="確かめるデータセット")
    ap.add_argument("--norm", type=Path, default=None,
                    help="据え置き先の normalizer（重みに同梱の safetensors）")
    ap.add_argument("--expect", default=None, metavar='"本 コマ 課題"',
                    help="期待する 3 つの数。空白区切り")
    ap.add_argument("--rev-tasks", type=int, default=None,
                    help="逆向きの指示文の数")
    ap.add_argument("--image", default="parc2026-train-pi05")
    ap.add_argument("--skip-video", action="store_true",
                    help="**飛ばしたら学習で初めて落ちる。**docker が使えないときだけ")
    args = ap.parse_args()

    if not (args.root / "meta/info.json").exists():
        sys.exit(f"**{args.root} にデータセットが無い**（meta/info.json が読めない）")

    print(f"=== {args.root} を確かめる ===")
    info = check_counts(args.root, args.expect)
    if args.norm:
        check_norm(args.root, args.norm)
    else:
        print("  - 据え置きの確認は飛ばした（--norm を渡すこと）")
    check_tasks(args.root, args.rev_tasks)
    check_parquet(args.root, info)
    if args.skip_video:
        print("  - 動画の復号は飛ばした（**学習で初めて落ちる**）")
    else:
        check_videos(args.root, args.image)
    bad = [p for p in args.root.rglob("*") if p.is_file() and not p.stat().st_mode & 0o444]
    check(not bad, "読めないファイルが無い", f"{len(bad)} 件" if bad else "")

    print()
    if _failed:
        sys.exit(f"**{_failed} 件落ちた。上げないこと。**")
    print("**7 つとも通った。上げてよい。**")


if __name__ == "__main__":
    main()
