#!/usr/bin/env python3
"""HF に上がった歩数が、手元のチェックポイントと同じ中身かを確かめる。

**なぜ要るか。**`hf upload` の終了コード 0 は「上がった」と同じではない。
2026-09-03 に**上げるループが生きたまま 3 時間 1 個も上げず、跡も残らなかった**
（`howto/14` の分岐 5）。**上げた印を付ける前に HF を読んで数えれば、
その事象は「確かめられない」として次の巡へ回る。**

**手元にあるファイルが 1 つでも HF に無い／大きさが違うなら失敗にする。**
HF 側に余分なファイルが在るのは失敗にしない（別の回の残りかもしれない）。

使い方:

    python3 verify_upload.py --repo siruku6/pi05_proto20 --step 000500 \
        --local /opt/dlami/nvme/out/pi05_proto20/checkpoints/000500/pretrained_model

終了コード 0 なら一致、1 なら不一致（印を付けてはいけない）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, metavar="REPO_ID")
    parser.add_argument("--step", required=True, metavar="NAME",
                        help="HF 上の置き場（`hf upload` の第 3 引数と同じ）")
    parser.add_argument("--local", required=True, type=Path, metavar="DIR")
    parser.add_argument("--repo-type", default="model")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    local = {
        str(p.relative_to(args.local)): p.stat().st_size
        for p in sorted(args.local.rglob("*")) if p.is_file()
    }
    if not local:
        print(f"NG 手元が空: {args.local}")
        return 1

    api = HfApi()
    info = (api.model_info if args.repo_type == "model" else api.dataset_info)(
        args.repo, files_metadata=True)
    # **`.` は `hf upload` の第 3 引数でリポジトリの根を表す。**そのまま繋ぐと
    # `./` になって 1 つも一致しない（9/9 に踏んだ。上がっているのに NG と出た）。
    prefix = "" if args.step in ("", ".") else f"{args.step}/"
    remote = {
        s.rfilename[len(prefix):]: (s.size or 0)
        for s in info.siblings if s.rfilename.startswith(prefix)
    }

    missing = [name for name in local if name not in remote]
    # **大きさは HF が返さないことがある**（`size` が None）。そのときは数えない
    wrong = [(name, local[name], remote[name]) for name in local
             if name in remote and remote[name] and remote[name] != local[name]]

    if missing or wrong:
        print(f"NG {args.repo}/{args.step}: 手元 {len(local)} 個 / HF {len(remote)} 個"
              f"・足りない {len(missing)} 個・大きさ違い {len(wrong)} 個")
        for name in missing[:5]:
            print(f"   足りない {name}")
        for name, mine, theirs in wrong[:5]:
            print(f"   大きさ違い {name}: 手元 {mine} / HF {theirs}")
        return 1

    total = sum(local.values())
    print(f"OK {args.repo}/{args.step}: {len(local)} 個・{total / 1e9:.2f} GB が一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
