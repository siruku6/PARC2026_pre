#!/usr/bin/env bash
# C1 のデータを Blackwell で組む：`libero_plus_bowl5` ＋ 配布 `libero_combined_20hz` の残り（`libero_plus` 以外）。
# **T1′ が学習している間に、同じ機械で打ってよい**（重い段は nice で後ろに回す）。
#
#   . ~/env_pi05.sh
#   bash ~/pi05_code/solution/train/blackwell_c1_data.sh             # 組んで門を通すまで
#   UPLOAD=1 bash ~/pi05_code/solution/train/blackwell_c1_data.sh    # 通ったら HF へ上げる
#
# 設計は docs/talk/tmp_reply/260913_SESSION_next_training.md の 6-0-4 節。
#
# | 段 | 中身 |
# |---|---|
# | 1 | 配布 tar を展開する（19,533 本・3,028,708 コマ・111 課題） |
# | 2 | **先頭 14,347 本が私たちの `libero_plus` と同じ並びか**を、長さと指示文で突き合わせる |
# | 3 | 残す一覧：先頭以外から、丼 5 課題の元の手本を外す（`KEEP_MODE=new_tasks` なら「libero_plus に無い課題」だけ） |
# | 4 | 動画を作り直さずに組み直す → 3 桁の symlink → `align` → `merge` |
# | 5 | 門：本数・コマ数・課題数／正規化の倍率 0.5〜2 倍／足した分の動画が読める |
set -uo pipefail

NVME="${NVME:-/opt/dlami/nvme}"
CODE_ROOT="${CODE_ROOT:-$HOME/pi05_code}"
T="$CODE_ROOT/solution/tools"
REF="$CODE_ROOT/solution/train/data_refs/libero_plus_episodes.json"
DATA="$NVME/data/libero_combined_20hz"
BOWL5="$NVME/data/libero_plus_bowl5"
REST="$NVME/data/combined_rest"
C1="$NVME/data/c1_bowl5_combined"
C1_ID="local/c1_bowl5_combined"
C1_REPO="${C1_REPO:-siruku6/c1_bowl5_combined}"
KEEP="$NVME/data/c1_keep.json"
KEEP_MODE="${KEEP_MODE:-prefix}"
UPLOAD="${UPLOAD:-0}"
WANT_DATA="19533 3028708 111"
WANT_BOWL5="14085 2264704 40"
NICE="nice -n 10"

step() { echo; echo "=== $* ==="; }
die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }
counts() {
    python3 -c "
import json,sys
try: i = json.load(open('$1/meta/info.json'))
except Exception: sys.exit(1)
print(i['total_episodes'], i['total_frames'], i['total_tasks'])" 2>/dev/null
}
have() { [ "$(counts "$1")" = "$2" ]; }

step "前提"
command -v hf >/dev/null || die "hf が無い" ". ~/env_pi05.sh を先に読む"
python3 -c "import lerobot" 2>/dev/null || die "lerobot が import できない" ". ~/env_pi05.sh を先に読む"
for f in "$REF" "$T/scripted_demo/npz_to_lerobot.py" "$T/contact_clean/rebuild_dataset.py" \
         "$T/track3_reversal/link_video_index_width.py" "$T/regen/make_replace_keep.py"; do
    [ -f "$f" ] || die "$f が無い" "hf download siruku6/pi05_code --local-dir $CODE_ROOT を打ち直す"
done
have "$BOWL5" "$WANT_BOWL5" || die "libero_plus_bowl5 が無い（$BOWL5）" \
    "先に blackwell_initpose.sh を打つか、hf download siruku6/libero_plus_bowl5 --repo-type dataset --local-dir $BOWL5"
echo "  空き: $(df -h "$NVME" | awk 'NR==2{print $4}')（要る量は約 80 GB）"

step "1. 配布 tar を展開する"
if have "$DATA" "$WANT_DATA"; then
    echo "  在る"
else
    TAR=$(ls "$HOME"/dataset/lerobot/*combined*.tar 2>/dev/null | head -1)
    [ -n "$TAR" ] || die "配布 tar が見つからない" "ls ~/dataset/lerobot/ を見る"
    mkdir -p "$DATA"
    $NICE tar -xf "$TAR" -C "$DATA" --strip-components=1 || die "展開に失敗" "空きを確かめる"
    have "$DATA" "$WANT_DATA" || die "展開した数が違う: $(counts "$DATA")（期待 $WANT_DATA）"
fi

step "2・3. 並びを突き合わせ、残す一覧を作る（KEEP_MODE=$KEEP_MODE）"
python3 - "$DATA" "$REF" "$KEEP_MODE" "$KEEP" "$T/regen" "$BOWL5" <<'PY'
import glob, json, sys
import pyarrow.parquet as pq
data, ref_path, mode, keep_path, regen, bowl5 = sys.argv[1:7]
sys.path.insert(0, regen)
from make_replace_keep import REPLACED
ref = json.load(open(ref_path))
rows = []
for f in sorted(glob.glob(f"{data}/meta/episodes/**/*.parquet", recursive=True)):
    t = pq.read_table(f, columns=["episode_index", "length", "tasks"])
    for i, n, task in zip(t.column("episode_index").to_pylist(), t.column("length").to_pylist(),
                          t.column("tasks").to_pylist()):
        rows.append((int(i), int(n), task[0] if isinstance(task, list) else task))
rows.sort()
N = ref["total_episodes"]
same = sum(1 for (i, n, task), rn, rt in zip(rows[:N], ref["lengths"], ref["tasks"])
           if n == rn and task == rt)
print(f"  先頭 {N} 本のうち、長さと指示文が私たちの libero_plus と一致: {same} 本")
ref_tasks = set(ref["tasks"])
if mode == "prefix":
    if same != N:
        print("**先頭が一致しない。KEEP_MODE=new_tasks で打ち直すこと**")
        sys.exit(2)
    cand = rows[N:]
elif mode == "new_tasks":
    cand = [r for r in rows if r[2] not in ref_tasks]
else:
    print(f"KEEP_MODE が読めない: {mode}")
    sys.exit(3)
bowl = set(REPLACED)
keep = [r for r in cand if r[2] not in bowl]
tasks = {r[2] for r in keep}
frames = sum(r[1] for r in keep)
info = json.load(open(f"{bowl5}/meta/info.json"))
print(f"  候補 {len(cand)} 本 → 丼 5 課題の元の手本 {len(cand) - len(keep)} 本を外し、残す {len(keep)} 本・{frames:,} コマ")
print(f"  残す分の課題 {len(tasks)} 種（うち libero_plus に無い課題 {len(tasks - ref_tasks)} 種）")
json.dump({
    "episodes": [r[0] for r in keep], "mode": mode,
    "rest": [len(keep), frames, len(tasks)],
    "merged": [info["total_episodes"] + len(keep), info["total_frames"] + frames,
               len(ref_tasks | tasks)],
}, open(keep_path, "w"))
PY
rc=$?
[ $rc = 0 ] || die "残す一覧を作れない（終了コード $rc）" "2 なら KEEP_MODE=new_tasks を付けて打ち直す"
read -r WANT_REST WANT_C1 <<< "$(python3 -c "
import json; k = json.load(open('$KEEP'))
print(' '.join(map(str, k['rest'])).replace(' ', '_'), ' '.join(map(str, k['merged'])).replace(' ', '_'))")"
WANT_REST=${WANT_REST//_/ }; WANT_C1=${WANT_C1//_/ }
echo "  組み直した後の期待 $WANT_REST ／ 混ぜた後の期待 $WANT_C1"

step "4. 組み直す → symlink → 混ぜる"
if ! have "$REST" "$WANT_REST"; then
    rm -rf "$REST"
    $NICE python3 "$T/contact_clean/rebuild_dataset.py" --root "$DATA" --output "$REST" --keep-file "$KEEP" \
        || die "組み直しが落ちた"
fi
have "$REST" "$WANT_REST" || die "組み直した数が違う: $(counts "$REST")（期待 $WANT_REST）"
python3 "$T/track3_reversal/link_video_index_width.py" --root "$REST" || die "symlink を張れない"
if ! have "$C1" "$WANT_C1"; then
    rm -rf "$C1"
    python3 "$T/scripted_demo/npz_to_lerobot.py" align --base "$BOWL5" --add "$REST" || die "align が落ちた"
    $NICE python3 "$T/scripted_demo/npz_to_lerobot.py" merge \
        --base "$BOWL5" --base-repo-id local/libero_plus_bowl5 \
        --add "$REST" --add-repo-id local/combined_rest \
        --output "$C1" --output-repo-id "$C1_ID" \
        || die "merge が落ちた" "動画で落ちるなら symlink が張れていない"
fi

step "5. 門"
have "$C1" "$WANT_C1" || die "混ぜた数が違う: $(counts "$C1")（期待 $WANT_C1）"
echo "  本数・コマ数・課題数 $WANT_C1"
python3 - "$BOWL5" "$C1" <<'PY' || die "正規化の倍率が 0.5〜2 倍を外れた" "混ぜ方を疑う（howto/16 の 6.3 節）"
import json, sys
import numpy as np
a = json.load(open(f"{sys.argv[1]}/meta/stats.json"))
b = json.load(open(f"{sys.argv[2]}/meta/stats.json"))
bad = False
for key in ("action", "observation.state"):
    r = np.array(b[key]["std"]) / np.array(a[key]["std"])
    print(f"  {key} の std 倍率（混ぜた後 / 元）: {np.round(r, 3).tolist()}")
    bad |= bool((r < 0.5).any() or (r > 2.0).any())
sys.exit(1 if bad else 0)
PY
python3 - "$C1" "$C1_ID" "$(counts "$BOWL5" | cut -d' ' -f2)" <<'PY' || die "足した分の動画を読めない"
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root, rid, first = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = LeRobotDataset(rid, root=root)
for i in (0, first, len(ds) - 1):
    item = ds[i]
    f, w = item["observation.images.front"], item["observation.images.wrist"]
    print(f"  コマ {i}: front {tuple(f.shape)} 平均 {float(f.mean()):.3f} / wrist 平均 {float(w.mean()):.3f} / {item['task'][:48]!r}")
PY
read -r E F K <<< "$WANT_C1"
python3 -c "
import json
json.dump({'total_episodes': $E, 'total_frames': $F, 'total_tasks': $K},
          open('$C1/c1_expected.json', 'w'), indent=2)"
echo "  大きさ: $(du -sh "$C1" | cut -f1)"

if [ "$UPLOAD" = 1 ]; then
    step "6. HF へ上げる（$C1_REPO・非公開）"
    hf upload "$C1_REPO" "$C1" . --repo-type dataset --private || die "上げられない" "HF の非公開枠を確かめる"
    echo "  上げた。RunPod で runpod_c1.sh を打てる"
else
    echo
    echo "=== 門は通った。上げるときは: UPLOAD=1 bash $0 ==="
    echo "  （組んだ段は数えて飛ばすので、打ち直しても数分で上げる段に着く）"
fi
