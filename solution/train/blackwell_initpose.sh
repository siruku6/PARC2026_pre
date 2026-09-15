#!/usr/bin/env bash
# T1′：15.0.0（bowl5_full）と同じ作り方で、データだけ初期姿勢データ 1,250 本を足して回す（Blackwell）。
#
#   . ~/env_pi05.sh
#   export HF_TOKEN="$(cat ~/.hf_token)"
#   hf download siruku6/pi05_code --local-dir ~/pi05_code
#   bash ~/pi05_code/solution/train/blackwell_initpose.sh
#
# **起動（T0）から 1 時間以内に打ち終えること。**5,500 歩は 10.5 時間かかり、12 時間の枠に
# ぎりぎり収まる。設計は docs/talk/tmp_reply/260913_SESSION_next_training.md の 6-0-3 節。
#
# | 段 | 中身 |
# |---|---|
# | 1 | `libero_plus_bowl5`・`lerobot_init_pose_all`・出発点を並列に降ろす |
# | 2 | `align` → `merge`（動画は再符号化しない） |
# | 3 | 門：本数・コマ数・課題数／正規化の倍率が 0.5〜2 倍／足した分の動画が読める |
# | 4 | `blackwell_bowl5.sh` に渡して起こす（15.0.0 と同じ学習率・warmup・batch・種） |
set -uo pipefail

NVME="${NVME:-/opt/dlami/nvme}"
CODE_ROOT="${CODE_ROOT:-$HOME/pi05_code}"
CODE_DIR="$CODE_ROOT/solution/train"
T="$CODE_ROOT/solution/tools"
RUN="${RUN:-pi05_bowl5_initpose}"
REPO="${REPO:-siruku6/$RUN}"
BOWL5="$NVME/data/libero_plus_bowl5"
INIT="$NVME/data/lerobot_init_pose_all"
MERGED="$NVME/data/bowl5_initpose"
MERGED_ID="local/bowl5_initpose"
WANT_BOWL5="14085 2264704 40"
WANT_INIT="1250 215691 38"
WANT_MERGED="15335 2480395 40"
BASE_REPO="lerobot/pi05_libero_base"

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
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN が空" 'export HF_TOKEN="$(cat ~/.hf_token)"'
python3 -c "import lerobot" 2>/dev/null || die "lerobot が import できない" ". ~/env_pi05.sh を先に読む"
for f in "$CODE_DIR/train_pi05.py" "$CODE_DIR/blackwell_bowl5.sh" \
         "$T/scripted_demo/npz_to_lerobot.py" "$T/scripted_demo/scripted_demo.py"; do
    [ -f "$f" ] || die "$f が無い" "hf download siruku6/pi05_code --local-dir $CODE_ROOT を打ち直す"
done
BASE_REV=$(sed -n 's/^_BASE_MODEL_REVISION = "\(.*\)"$/\1/p' "$CODE_DIR/train_pi05.py")
[ ${#BASE_REV} = 40 ] || die "出発点の版を train_pi05.py から読めない"
mkdir -p "$NVME/data"
echo "  起動からの経過: $(( ($(date +%s) - $(date -d "$(uptime -s)" +%s)) / 60 )) 分（60 分を超えていたら STEPS を減らすか RunPod へ回す）"

step "1. 降ろす（3 つを並列に）"
dl() {  # 名前 コマンド…
    local name=$1; shift
    ( "$@" > "$HOME/dl_$name.log" 2>&1; echo $? > "$HOME/dl_$name.rc" ) &
}
have "$BOWL5" "$WANT_BOWL5" && { echo "  在る libero_plus_bowl5"; echo 0 > "$HOME/dl_bowl5.rc"; } \
    || dl bowl5 hf download siruku6/libero_plus_bowl5 --repo-type dataset --local-dir "$BOWL5"
have "$INIT" "$WANT_INIT" && { echo "  在る lerobot_init_pose_all"; echo 0 > "$HOME/dl_init.rc"; } \
    || dl init hf download siruku6/lerobot_init_pose_all --repo-type dataset --local-dir "$INIT"
dl base hf download "$BASE_REPO" --revision "$BASE_REV"
wait
for n in bowl5 init base; do
    [ "$(cat "$HOME/dl_$n.rc")" = 0 ] || { tail -5 "$HOME/dl_$n.log"; die "$n を降ろせない"; }
done
have "$BOWL5" "$WANT_BOWL5" || die "libero_plus_bowl5 の数が違う: $(counts "$BOWL5")（期待 $WANT_BOWL5）"
have "$INIT" "$WANT_INIT"   || die "lerobot_init_pose_all の数が違う: $(counts "$INIT")（期待 $WANT_INIT）"
echo "  ok"

step "2. 混ぜる"
if have "$MERGED" "$WANT_MERGED"; then
    echo "  在る（$MERGED）"
else
    rm -rf "$MERGED"
    python3 "$T/scripted_demo/npz_to_lerobot.py" align --base "$BOWL5" --add "$INIT" \
        || die "align が落ちた"
    python3 "$T/scripted_demo/npz_to_lerobot.py" merge \
        --base "$BOWL5" --base-repo-id local/libero_plus_bowl5 \
        --add "$INIT" --add-repo-id local/lerobot_init_pose_all \
        --output "$MERGED" --output-repo-id "$MERGED_ID" \
        || die "merge が落ちた"
fi

step "3. 門"
have "$MERGED" "$WANT_MERGED" || die "混ぜた数が違う: $(counts "$MERGED")（期待 $WANT_MERGED）"
echo "  本数・コマ数・課題数 $WANT_MERGED"
python3 - "$BOWL5" "$MERGED" <<'PY' || die "正規化の倍率が 0.5〜2 倍を外れた" "混ぜ方を疑う（howto/16 の 6.3 節）"
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
python3 - "$MERGED" "$MERGED_ID" 2264704 <<'PY' || die "足した分の動画を読めない"
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root, rid, first = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = LeRobotDataset(rid, root=root)
for i in (0, first, len(ds) - 1):
    item = ds[i]
    f, w = item["observation.images.front"], item["observation.images.wrist"]
    print(f"  コマ {i}: front {tuple(f.shape)} 平均 {float(f.mean()):.3f} / wrist 平均 {float(w.mean()):.3f} / {item['task'][:48]!r}")
PY
python3 -c "
import json
json.dump({'total_episodes': 15335, 'total_frames': 2480395, 'total_tasks': 40},
          open('$MERGED/bowl5_expected.json', 'w'), indent=2)"
python3 -c "
from huggingface_hub import HfApi
HfApi().create_repo('$REPO', private=True, exist_ok=True); print('  上げ先 $REPO')" \
    || die "上げ先を作れない" "HF の非公開枠を確かめる"

step "4. 起こす（blackwell_bowl5.sh へ）"
export RUN REPO CODE_DIR NVME
export DS="$MERGED" DATASET_ID="$MERGED_ID" SKIP_DL_DATASET=1
export STEPS="${STEPS:-5500}" SAVE_FREQ="${SAVE_FREQ:-2750}" RESERVE_MIN="${RESERVE_MIN:-30}"
exec bash "$CODE_DIR/blackwell_bowl5.sh"
