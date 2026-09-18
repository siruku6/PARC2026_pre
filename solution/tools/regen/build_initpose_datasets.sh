#!/usr/bin/env bash
# 初期姿勢データセットを 2 つ組み、門を通してから HF へ上げる（9/14）。
#
# | 名前 | 土台 | 中身 | HF |
# |---|---|---|---|
# | P `libero_plus_bowl5_initpose` | `libero_plus` | `libero_plus_bowl5` から 1,532 本を外し、撃ち直し・台本・道 A を足す | **公開**（比較用・ユーザ 9/14） |
# | C `libero_combined_bowl5_initpose` | `libero_combined_20hz` | **P ＋ combined の先頭**（標準スイートの元データと `libero_90`） | 非公開 |
#
# **C を「combined を土台に全部組み直す」のではなく「P に先頭を足す」形にしている。**
# combined の後ろ 14,347 本は `libero_plus` と 1 ビットも違わない（番号が 5,186 ずれるだけ）ので、
# P を組めば後ろは済んでいる。**`libero_plus` 部分の番号を読み替える場面が 1 つも無いので、取り違えが起きない。**
#
# 先頭 5,186 本の扱い（`work/regen/combined/head_keep.json`）
#   - 標準スイートの元データ（1,265 本）… 今までと同じ——丼 5 課題の元の手本・300 コマ超・接触（基準コマ 10 で 1 mm 超）を外す
#   - `libero_90`（3,921 本）… 300 コマ超だけ外し、接触は判定しない（ユーザ 9/14）
#   - **指の値を {0,1} から ±1 に直す**（`fix_gripper_encoding.py`。直さないと「開く」に逆の正解が付く）
#
#   STAGE=p|c|all（既定 all）  UPLOAD=1 で門を通ったあと上げる
#   nohup bash solution/tools/regen/build_initpose_datasets.sh > work/regen/logs/build_initpose.log 2>&1 &
#
# Reads:
#   - work/artifacts/libero_plus_bowl5・work/regen/keep_replace_v2.json
#   - work/regen/lerobot_init_pose_v2/c*・lerobot_scripted_v2・lerobot_merged_v2
#   - $COMBINED（展開した配布 tar）・work/regen/combined/head_keep.json
# Generates:
#   - work/artifacts/libero_plus_bowl5_initpose（＋ bowl5_expected.json）
#   - work/artifacts/libero_combined_bowl5_initpose（＋ bowl5_expected.json）
#   - work/regen/assembly/ … 中継物（bowl5_keep・combined_head）
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# **素の python3 には頼らない。** numpy・pyarrow・pandas が要る呼び出しは
# すべてコンテナへ回す（`pyrun`）。lerobot が要るものは `R`。
# **`NATIVE=1` を渡すと両方とも手元の python3 で走る**——lerobot と
# numpy/pyarrow/pandas の入った環境を自分で用意している場合だけ。
IMAGE="${IMAGE:-parc2026-train-pi05}"
NATIVE="${NATIVE:-0}"
# **落ちたときに終了コードを出す。**コンテナの中で殺されると、
# 途中まで出力して黙って止まったように見える（137 がその典型）。
_reported() {
    "$@"; local s=$?
    [ "$s" -eq 0 ] && return 0
    echo "  （終了コード $s）" >&2
    case "$s" in
        137) echo "  **137 は OOM Kill である。**Docker に割り当てたメモリが足りない" >&2 ;;
        139) echo "  139 はセグメンテーション違反" >&2 ;;
    esac
    return "$s"
}
if [ "$NATIVE" = 1 ]; then
    R()     { _reported python3 "$@"; }
    pyrun() { _reported python3 "$@"; }
else
    R()     { _reported env IMAGE="$IMAGE" bash solution/tools/regen/run_in_container.sh "$@"; }
    pyrun() { _reported env IMAGE="$IMAGE" bash solution/tools/regen/run_in_container.sh "$@"; }
fi
own() { docker run --rm -v "$PWD/$1:/out" --entrypoint chown "$IMAGE" \
            -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true; }
die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }
# **`date -Is` は GNU にしか無い**（macOS の BSD date は -I を取らない）。
step() { echo; echo "=== $* === $(date '+%Y-%m-%dT%H:%M:%S%z')"; }

# **ハードリンクで木を写す。** `cp -al` は GNU の書き方で、macOS の cp には無い。
# 写し（実体のコピー）にならないよう、ハードリンクを張れる手を順に試す。
hardlink_tree() {   # $1 = 元  $2 = 先
    cp -al "$1" "$2" 2>/dev/null && return 0
    rm -rf "$2"
    mkdir -p "$2" && (cd "$1" && pax -rw -l . "$OLDPWD/$2") 2>/dev/null && return 0
    rm -rf "$2"
    python3 - "$1" "$2" <<'PY'
import os, sys
src, dst = sys.argv[1:3]
for root, _, files in os.walk(src):
    out = os.path.join(dst, os.path.relpath(root, src))
    os.makedirs(out, exist_ok=True)
    for name in files:
        target = os.path.join(out, name)
        if not os.path.exists(target):
            os.link(os.path.join(root, name), target)
PY
}
field() { python3 -c "import json;print(json.load(open('$1/meta/info.json'))['$2'])" 2>/dev/null || echo 0; }

STAGE="${STAGE:-all}"
UPLOAD="${UPLOAD:-0}"
BOWL5=work/artifacts/libero_plus_bowl5
KEEP=work/regen/keep_replace_v2.json
WORK=work/regen/assembly
BASE=$WORK/bowl5_keep
P=work/artifacts/libero_plus_bowl5_initpose
P_REPO=siruku6/libero_plus_bowl5_initpose
COMBINED="${COMBINED:-/home/ubuntu/src/train_data/libero_combined_20hz_extracted/libero_combined_20hz}"
HEAD_KEEP=work/regen/combined/head_keep.json
HEAD=$WORK/combined_head
C=work/artifacts/libero_combined_bowl5_initpose
C_REPO=siruku6/libero_combined_bowl5_initpose
GRADER=work/regen/grader_texts.txt
mkdir -p "$WORK"

gripper_gate() {   # $1 = データセット。指の値が ±1 だけか（全コマ）
    pyrun - "$1" <<'PY'
import glob, sys
import numpy as np, pyarrow.parquet as pq
values = set()
for f in glob.glob(f"{sys.argv[1]}/data/**/*.parquet", recursive=True):
    a = np.stack(pq.read_table(f, columns=["action"]).column("action").to_numpy(zero_copy_only=False))
    values |= set(np.unique(a[:, 6]).tolist())
print(f"  指の値の種類: {sorted(values)}")
sys.exit(0 if values <= {-1.0, 1.0} else 1)
PY
}
stats_gate() {     # $1 = 組んだもの  $2 = 比べる相手。std が 2 倍以上動いたら止める
    pyrun - "$1" "$2" <<'PY'
import json, sys
import numpy as np
out, ref = (json.load(open(f"{p}/meta/stats.json")) for p in sys.argv[1:3])
worst = 1.0
for key in ("action", "observation.state"):
    r = np.array(out[key]["std"]) / np.array(ref[key]["std"])
    worst = max(worst, r.max(), 1 / r.min())
    print(f"  {key:18} std 倍率 {np.round(r, 3).tolist()}")
sys.exit(0 if worst < 2.0 else 1)
PY
}
expected() {       # $1 = データセット。学習スクリプトが突き合わせる期待値を同梱する
    python3 - "$1" <<'PY'
import json, sys
info = json.load(open(f"{sys.argv[1]}/meta/info.json"))
want = {k: info[k] for k in ("total_episodes", "total_frames", "total_tasks")}
json.dump(want, open(f"{sys.argv[1]}/bowl5_expected.json", "w"), indent=2)
print("  bowl5_expected.json:", want)
PY
}
upload() {         # $1 = データセット  $2 = repo  $3 = private（True/False）
    python3 -c "
from huggingface_hub import HfApi
HfApi().create_repo('$2', repo_type='dataset', private=$3, exist_ok=True)
print('  上げ先 $2（private=$3）')" || die "上げ先を作れない"
    hf upload-large-folder "$2" "$1" --repo-type dataset || die "上げられない（$2）"
}

# ================================================================= P
if [ "$STAGE" = p ] || [ "$STAGE" = all ]; then
    step "P0. 撃ち直しの全塊の変換が終わるのを待つ"
    until grep -q "=== 出来たもの ===" work/regen/logs/init_pose_v2.log; do sleep 120; done
    # **部品の一覧は PARTS_FILE（1 行 1 ディレクトリ）で差し替えられる。**9/14 夜に「止めて待つ」39 本を
    # 撃ち直したときは、元の塊から 39 本を抜いた部品と撃ち直した部品を並べた一覧を渡した（work/regen/release/parts.txt）。
    if [ -n "${PARTS_FILE:-}" ]; then
        mapfile -t PARTS < <(grep -v '^\s*$' "$PARTS_FILE")
    else
        mapfile -t PARTS < <(ls -d work/regen/lerobot_init_pose_v2/c*/ | sed 's:/$::'; \
                             echo work/regen/lerobot_scripted_v2; echo work/regen/lerobot_merged_v2)
        [ "${#PARTS[@]}" = 19 ] || die "部品が 19 個でない（${#PARTS[@]} 個）" "塊が欠けていないか ls する"
    fi
    ADD_EPS=0; ADD_FRAMES=0
    for part in "${PARTS[@]}"; do
        n=$(field "$part" total_episodes); [ "$n" -gt 0 ] || die "$part に info.json が無い"
        ADD_EPS=$((ADD_EPS + n)); ADD_FRAMES=$((ADD_FRAMES + $(field "$part" total_frames)))
        printf '  %5s 本  %s\n' "$n" "$part"
    done
    echo "  足す分 $ADD_EPS 本 / $ADD_FRAMES コマ"

    step "P1. 土台から 1,532 本を外す（動画はハードリンク）"
    if [ "$(field "$BASE" total_episodes)" != 12553 ]; then
        rm -rf "$BASE"
        pyrun solution/tools/contact_clean/rebuild_dataset.py --root "$BOWL5" --output "$BASE" \
            --keep-file "$KEEP" || die "外せない"
    fi
    [ "$(field "$BASE" total_episodes)" = 12553 ] || die "残りが 12,553 本でない"

    step "P2. 列の書き方を揃えて、1 回で繋ぐ"
    if [ ! -f "$P/meta/info.json" ]; then
        for part in "${PARTS[@]}"; do
            R solution/tools/scripted_demo/npz_to_lerobot.py align --base "$BASE" --add "$part" \
                > /dev/null || die "揃えられない: $part"
        done
        ADDS=(); for part in "${PARTS[@]}"; do ADDS+=(--add "$part"); done
        rm -rf "$P"
        R solution/tools/regen/merge_many.py --base "$BASE" "${ADDS[@]}" \
            --output "$P" --output-repo-id local/libero_plus_bowl5_initpose || die "繋げない"
        own "$P"
    fi

    step "P 門 1. 本数・コマ数・課題数が足し算に一致するか"
    WANT_EPS=$((12553 + ADD_EPS)); WANT_FRAMES=$(( $(field "$BASE" total_frames) + ADD_FRAMES ))
    echo "  本数 $(field "$P" total_episodes)（期待 $WANT_EPS）/ コマ $(field "$P" total_frames)（期待 $WANT_FRAMES）/ 課題 $(field "$P" total_tasks)（期待 40）"
    [ "$(field "$P" total_episodes)" = "$WANT_EPS" ] && [ "$(field "$P" total_frames)" = "$WANT_FRAMES" ] \
        && [ "$(field "$P" total_tasks)" = 40 ] || die "P 門 1 で落ちた"
    step "P 門 2. 指示文が評価でモデルが見る文と合っているか"
    R solution/tools/scripted_demo/check_dataset_instructions.py "$P" --grader-texts "$GRADER" \
        || die "P 門 2 で落ちた"
    step "P 門 3. 指の値が ±1 だけか（生成の「止めて待つ」コマの 0 を直前の値で埋めてから見る）"
    # 9/14 までの生成は、撃ち切ってもゴールが立たないと行動 7 次元すべて 0 で待っていた（39 本・238 コマ）。
    # 何度打っても同じ結果になる（埋める 0 が無ければ何もしない）。
    pyrun solution/tools/regen/fix_gripper_encoding.py --root "$P" --mode hold || die "指の 0 を埋められない"
    gripper_gate "$P" || die "P 門 3 で落ちた"
    step "P 門 4. 正規化の尺度（libero_plus_bowl5 と比べる）"
    stats_gate "$P" "$BOWL5" || die "P 門 4 で落ちた"
    expected "$P"
    echo "  大きさ $(du -sh "$P" | cut -f1)"
    [ "$UPLOAD" = 1 ] && { step "P を上げる（公開）"; upload "$P" "$P_REPO" False; }
fi

# ================================================================= C
if [ "$STAGE" = c ] || [ "$STAGE" = all ]; then
    [ -f "$P/bowl5_expected.json" ] || die "P が門を通っていない" "STAGE=p を先に通す"
    [ -f "$HEAD_KEEP" ] || die "$HEAD_KEEP が無い"
    HEAD_EPS=$(python3 -c "import json;print(len(json.load(open('$HEAD_KEEP'))['episodes']))")

    step "C1. combined の先頭から残す $HEAD_EPS 本を抜く（動画はハードリンク）"
    if [ "$(field "$HEAD" total_episodes)" != "$HEAD_EPS" ]; then
        rm -rf "$HEAD"
        pyrun solution/tools/contact_clean/rebuild_dataset.py --root "$COMBINED" --output "$HEAD" \
            --keep-file "$HEAD_KEEP" || die "抜けない"
        [ "$(field "$HEAD" total_episodes)" = "$HEAD_EPS" ] || die "抜いた本数が違う"
        step "C2. 指の値を ±1 に直す"
        pyrun solution/tools/regen/fix_gripper_encoding.py --root "$HEAD" || die "直せない"
        pyrun solution/tools/track3_reversal/link_video_index_width.py --root "$HEAD" \
            || die "動画の名前の桁を揃えられない"
    fi
    gripper_gate "$HEAD" || die "先頭に ±1 以外が残っている" "rm -rf $HEAD して打ち直す"

    step "C3. P に先頭を繋ぐ"
    if [ ! -f "$C/meta/info.json" ]; then
        R solution/tools/scripted_demo/npz_to_lerobot.py align --base "$P" --add "$HEAD" \
            > /dev/null || die "揃えられない"
        rm -rf "$C"
        R solution/tools/regen/merge_many.py --base "$P" --add "$HEAD" \
            --output "$C" --output-repo-id local/libero_combined_bowl5_initpose || die "繋げない"
        own "$C"
    fi

    step "C 門 1. 本数・コマ数・課題数"
    WANT_TASKS=$(pyrun - "$P" "$HEAD" <<'PY'
import sys, pyarrow.parquet as pq
names = set()
for root in sys.argv[1:3]:
    table = pq.read_table(f"{root}/meta/tasks.parquet").to_pandas()
    names |= set(table.index if "task" not in table.columns else table["task"])
print(len(names))
PY
)
    WANT_EPS=$(( $(field "$P" total_episodes) + HEAD_EPS ))
    WANT_FRAMES=$(( $(field "$P" total_frames) + $(field "$HEAD" total_frames) ))
    echo "  本数 $(field "$C" total_episodes)（期待 $WANT_EPS）/ コマ $(field "$C" total_frames)（期待 $WANT_FRAMES）/ 課題 $(field "$C" total_tasks)（期待 $WANT_TASKS）"
    [ "$(field "$C" total_episodes)" = "$WANT_EPS" ] && [ "$(field "$C" total_frames)" = "$WANT_FRAMES" ] \
        && [ "$(field "$C" total_tasks)" = "$WANT_TASKS" ] || die "C 門 1 で落ちた"
    step "C 門 2. 指の値が ±1 だけか"
    gripper_gate "$C" || die "C 門 2 で落ちた"
    step "C 門 3. 正規化の尺度（P と比べる）"
    stats_gate "$C" "$P" || die "C 門 3 で落ちた"
    step "C 門 4. 繋いだ先頭が、元の combined と同じコマを指しているか（画素・state・指の直し）"
    # コンテナから見えるのはリポジトリだけなので、元データをハードリンクで置く（容量は増えない）。
    LINK=$WORK/combined_link
    [ -f "$LINK/meta/info.json" ] || hardlink_tree "$COMBINED" "$LINK" || die "ハードリンクを張れない"
    R solution/tools/regen/verify_appended_frames.py --merged "$C" --offset "$(field "$P" total_episodes)" \
        --source "$LINK" --keep-file "$HEAD_KEEP" --expect-gripper-fix || die "C 門 4 で落ちた"
    expected "$C"
    echo "  大きさ $(du -sh "$C" | cut -f1)"
    [ "$UPLOAD" = 1 ] && { step "C を上げる（非公開）"; upload "$C" "$C_REPO" True; }
fi
echo; echo "END_OF_BUILD $(date '+%Y-%m-%dT%H:%M:%S%z')"
