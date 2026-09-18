#!/usr/bin/env bash
# `libero_combined_bowl5_initpose_l90clean` を、親と部品から組み直す。
#
# **この段だけ 2026-09-16 に手で実施したため、スクリプトが無かった。**
# 判断は `manifests/l90clean_replace.json` に残してあるので、ここはそれを実行するだけである。
#
# **やること。**親の libero_90 部分から汚れた 250 本を外し、同じ数の綺麗な手本を末尾に足す。
#   1. 台帳から keep-file を 2 つ作る（残す 21,392 本／足す 250 本）
#   2. `rebuild_dataset.py` で 250 本を外す（**動画は作り直さない。ハードリンクで済む**）
#   3. `rebuild_dataset.py` で部品から 250 本を台帳の順に抜く
#   4. `merge_many.py` で末尾に繋ぐ
#
# **要るもの**（この台に無ければ先に落とす）
#   PARENT  親 libero_combined_bowl5_initpose（`build_initpose_datasets.sh STAGE=c` で作る）
#   POOL    siruku6/libero_90_clean（`hf download --repo-type dataset`）
#
# **`merge_many.py` は lerobot を import する**ので、素の python では動かない。
# 既定では `build_initpose_datasets.sh` と同じくコンテナ（`parc2026-train-pi05`）で回す。
# **lerobot が入った環境が手元にあるなら `NATIVE=1` を渡せば直に回す。**
#
# **一度も通していない**（この機械のディスクが 26 GB しか空いておらず、
# 親 27.6 GB ＋ 出来上がり 27.8 GB が入らなかった。2026-09-18）。
#
#   PARENT=work/artifacts/libero_combined_bowl5_initpose \
#   POOL=work/artifacts/libero_90_clean \
#   bash solution/tools/regen/rebuild_l90clean.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

# **置き場をまるごと移せる**（`build_initpose_datasets.sh` と同じ約束）。
WORK_ROOT="${WORK_ROOT:-work}"
case "$WORK_ROOT" in
    /*) export MOUNTS="${MOUNTS:-}${MOUNTS:+ }$WORK_ROOT" ;;
esac

PARENT="${PARENT:-$WORK_ROOT/artifacts/libero_combined_bowl5_initpose}"
POOL="${POOL:-$WORK_ROOT/artifacts/libero_90_clean}"
OUT="${OUT:-$WORK_ROOT/artifacts/libero_combined_bowl5_initpose_l90clean_rebuilt}"
WORK="${WORK:-$WORK_ROOT/regen/l90clean_rebuild}"
MANIFEST=solution/tools/regen/manifests/l90clean_replace.json

die() { echo "** $1" >&2; [ $# -gt 1 ] && echo "   $2" >&2; exit 1; }
# **lerobot が要るものはコンテナで回す。**`rebuild_dataset.py` は numpy と pyarrow
# だけなので素の python でよい（だから下で使い分けている）。
NATIVE="${NATIVE:-0}"
IMAGE="${IMAGE:-parc2026-train-pi05}"
if [ "$NATIVE" = 1 ]; then
    R()     { python3 "$@"; }
    pyrun() { python3 "$@"; }
else
    # **`rebuild_dataset.py` は numpy・pyarrow・pandas を要る。**素の python3 に
    # それが入っている保証は無いので、既定ではこれもコンテナへ回す。
    R()     { IMAGE="$IMAGE" bash solution/tools/regen/run_in_container.sh "$@"; }
    pyrun() { IMAGE="$IMAGE" bash solution/tools/regen/run_in_container.sh "$@"; }
fi
[ -d "$PARENT/meta" ] || die "親が無い: $PARENT" "build_initpose_datasets.sh STAGE=c で作る"
[ -d "$POOL/meta" ]   || die "部品が無い: $POOL" "hf download siruku6/libero_90_clean --repo-type dataset --local-dir $POOL"

mkdir -p "$WORK"
echo "-- 1. 台帳から keep-file を作る"
python3 - "$MANIFEST" "$PARENT" "$WORK" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
parent_info = json.loads((Path(sys.argv[2]) / "meta/info.json").read_text())
work = Path(sys.argv[3])

total = int(parent_info["total_episodes"])
dropped = set(manifest["dropped_episode_indices"])
if total != 21642:
    raise SystemExit(f"親の本数が 21,642 でない（{total}）。台帳と合わない")

kept = [i for i in range(total) if i not in dropped]
sources = [row["source_episode_index"] for row in manifest["appended_episodes"]]
if len(kept) != total - len(dropped) or len(sources) != len(dropped):
    raise SystemExit("台帳の数が合わない")

(work / "keep_parent.json").write_text(json.dumps({"episodes": kept}))
(work / "keep_pool.json").write_text(json.dumps({"episodes": sources}))
print(f"   残す {len(kept)} 本 / 足す {len(sources)} 本")
PY

echo "-- 2. 親から 250 本を外す"
pyrun solution/tools/contact_clean/rebuild_dataset.py \
    --root "$PARENT" --output "$WORK/parent_kept" --keep-file "$WORK/keep_parent.json"

echo "-- 3. 部品から 250 本を台帳の順に抜く"
pyrun solution/tools/contact_clean/rebuild_dataset.py \
    --root "$POOL" --output "$WORK/replacements" --keep-file "$WORK/keep_pool.json"

echo "-- 4. 末尾に繋ぐ（lerobot が要るのでコンテナで回す）"
R solution/tools/regen/merge_many.py \
    --base "$WORK/parent_kept" --base-repo-id local/l90clean_base \
    --add "$WORK/replacements" \
    --output "$OUT" --output-repo-id local/libero_combined_bowl5_initpose_l90clean

# **コンテナで書いたものは root のものになる。**走り終えたら持ち主を戻す。
if [ "$NATIVE" != 1 ] && [ -d "$OUT" ]; then
    for d in "$OUT" "$WORK"; do
        case "$d" in /*) ;; *) d="$PWD/$d" ;; esac
        docker run --rm -v "$d:/out" --entrypoint chown "$IMAGE" \
            -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
    done
fi

echo "-- 5. 数を突き合わせる"
python3 - "$OUT" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads((Path(sys.argv[1]) / "meta/info.json").read_text())
want = json.loads(Path(sys.argv[2]).read_text())["totals"]
got = {"merged_episodes": int(info["total_episodes"]),
       "merged_frames": int(info["total_frames"]),
       "merged_tasks": int(info["total_tasks"])}
for key, value in got.items():
    mark = "OK" if value == want[key] else "**違う**"
    print(f"   {key}: {value}（台帳 {want[key]}）{mark}")
if any(got[k] != want[k] for k in got):
    raise SystemExit("一致しなかった")
print("   一致した")
PY
echo "-- 出来た: $OUT"
