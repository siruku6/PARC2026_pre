#!/usr/bin/env bash
# Blackwell で「配布データ ＋ 逆向きデータ」の学習用データセットを作るまでを 1 本で通す。
#
#   bash blackwell_setup.sh                       # 既定（逆向き 11 倍 ＝ 割合 20%）
#   COPIES=8 bash blackwell_setup.sh              # 倍率を変える
#
# **何度打っても良い。**各段は出来上がりを数えて、正しければ飛ばす。
# **ターミナルが落ちても、もう一度打てば続きから進む。**
#
# 2026-09-07 に手で打ったときは作業だけで 2 時間かかった。踏んだ穴を全部ここで潰す。
#
# | 踏んだ穴 | ここでどう潰すか |
# |---|---|
# | `hf download` が `$HOME` ではなく HF キャッシュに置く | `find` で実体を探す |
# | `npz_to_lerobot.py` が隣の `scripted_demo.py` を要る | 道具を 6 つまとめて落とす |
# | `align` が軸名の綴りで止まる | 直した版を落とす |
# | 配布 tar の動画が 6 桁で `merge` が落ちる | `link_video_index_width.py` を必ず通す |
# | `M() { ... || exit 1; }` が対話シェルを終わらせる | スクリプトなので `exit` で構わない |
# | ターミナルが落ちて変数が消える | 最後に `~/run_env.sh` を書く。`. ~/run_env.sh` で戻る |
# | `/opt/dlami/nvme` が消えている | 全部作り直す。この 1 本で済む |
#
# Generates:
#   - $DATA    配布データの展開（19,533 本）
#   - $P/x{N}  逆向きデータの N 倍
#   - $MERGED  混ぜて正規化を据え置いたもの
#   - ~/run_env.sh  変数の張り直し用
#   - /opt/dlami/nvme/weights/rp3000  出発点の重み
set -uo pipefail

COPIES="${COPIES:-11}"
NVME="${NVME:-/opt/dlami/nvme}"
DATA="${DATA:-$NVME/data/libero_combined_20hz}"
P="${P:-$NVME/data/proto}"
MERGED="${MERGED:-$NVME/data/proto${COPIES}}"
TOOLS="${TOOLS:-$HOME/tools}"
WEIGHTS="${WEIGHTS:-$NVME/weights/rp3000}"
CODE_DIR="${CODE_DIR:-$HOME/pi05_code/solution/train}"
DATASET_REPO="${DATASET_REPO:-siruku6/reverse_lerobot_389}"
REV_TAR="${REV_TAR:-proto389.tar.gz}"
REV_DIR="$P/reverse_lerobot_all"

# 期待値。**ここが違ったら止める。**
WANT_DATA="19533 3028708 111"
WANT_REV="389 67775 3"

step() { echo; echo "=== $* ==="; }
die() {   # 原因と、次に何をすればよいかを必ず一緒に出す
    echo
    echo "**止まった: $1**"
    shift
    for line in "$@"; do echo "  $line"; done
    exit 1
}
counts() {  # $1=データセット → "本数 コマ数 タスク数"
    python3 -c "
import json,sys
try:
    i = json.load(open('$1/meta/info.json'))
except Exception:
    sys.exit(1)
print(i['total_episodes'], i['total_frames'], i['total_tasks'])" 2>/dev/null
}
have() { [ "$(counts "$1")" = "$2" ]; }

step "0. 前提"
[ -d "$NVME" ] || die "$NVME が無い" "Blackwell 以外で打っている。NVME= で場所を渡す"
command -v hf >/dev/null || die "hf コマンドが無い" ". ~/env_pi05.sh を先に打つ"
python3 -c "import lerobot" 2>/dev/null || die "lerobot が import できない" ". ~/env_pi05.sh を先に打つ"
echo "空き: $(df -h "$NVME" | awk 'NR==2{print $4}')  （要る量は約 70 GB）"

step "1. 道具（6 つ）"
mkdir -p "$TOOLS"
for f in npz_to_lerobot.py scripted_demo.py check_merge_stats.py \
         link_video_index_width.py upload_ckpt.sh verify_upload.py; do
    if [ ! -f "$TOOLS/$f" ]; then
        hf download "$DATASET_REPO" "$f" --repo-type dataset --local-dir "$TOOLS" >/dev/null \
            || die "$f を落とせない" "$DATASET_REPO に在るか確認する"
        echo "  落とした $f"
    else
        echo "  在る     $f"
    fi
done
chmod +x "$TOOLS/upload_ckpt.sh"

step "2. 配布データ"
if have "$DATA" "$WANT_DATA"; then
    echo "  在る（$WANT_DATA）"
else
    TAR=$(ls "$HOME"/dataset/lerobot/*combined*.tar 2>/dev/null | head -1)
    [ -n "$TAR" ] || die "配布 tar が見つからない" "ls ~/dataset/lerobot/ を見る" \
        "~/dataset は /opt/parc/dataset への symlink である"
    mkdir -p "$DATA"
    echo "  展開する: $TAR"
    tar -xf "$TAR" -C "$DATA" --strip-components=1 \
        || die "展開に失敗" "空きを確認する（tar は 19.6 GiB）"
    have "$DATA" "$WANT_DATA" \
        || die "配布データの数が違う: $(counts "$DATA")" "期待は $WANT_DATA" \
               "--strip-components=1 が要る。別の tar かもしれない"
    echo "  出来た（$WANT_DATA）"
fi

step "3. 動画の通し番号を 3 桁に見せる"
# **配布 tar は file-{:06d}.mp4 だが、aggregate_datasets は 3 桁でしか読まない。**
python3 "$TOOLS/link_video_index_width.py" --root "$DATA" \
    || die "symlink を張れない" "$DATA/videos の中身を見る"

step "4. 逆向きデータ"
if have "$REV_DIR" "$WANT_REV"; then
    echo "  在る（$WANT_REV）"
else
    hf download "$DATASET_REPO" "$REV_TAR" --repo-type dataset >/dev/null \
        || die "$REV_TAR を落とせない" "hf auth whoami で認証を確認する"
    # **`hf download` は $HOME ではなく HF キャッシュに置く。**実体を探す
    TAR=$(find "${HF_HOME:-$HOME/.cache/huggingface}/hub" -name "$REV_TAR" 2>/dev/null | head -1)
    [ -n "$TAR" ] || die "$REV_TAR の実体が見つからない" \
        "find \${HF_HOME}/hub -name $REV_TAR で探す"
    mkdir -p "$P"
    tar -xzf "$TAR" -C "$P" || die "$REV_TAR を展開できない" "$TAR を見る"
    have "$REV_DIR" "$WANT_REV" \
        || die "逆向きデータの数が違う: $(counts "$REV_DIR")" "期待は $WANT_REV"
    echo "  出来た（$WANT_REV）"
fi

step "5. 逆向きデータを $COPIES 倍にする"
merge_one() {  # $1 + $2 → $3
    [ -d "$3/meta" ] && { echo "  在る $(basename "$3")"; return 0; }
    python "$TOOLS/npz_to_lerobot.py" merge \
        --base "$1" --base-repo-id local/a \
        --add  "$2" --add-repo-id  local/b \
        --output "$3" --output-repo-id local/proto >/dev/null \
        || die "merge に失敗: $(basename "$3")" "$3 に中身が残っていれば消す"
    echo "  作った $(basename "$3") → $(counts "$3")"
}
# **倍々で組む。**1 本ずつ足すと書き込みが 3 倍になる
XN="$P/x$COPIES"
[ "$COPIES" = 1 ] && XN="$REV_DIR"     # **1 倍はそのまま使う**
if [ -d "$XN/meta" ]; then
    echo "  在る x$COPIES → $(counts "$XN")"
else
    python3 - "$COPIES" <<'PY' > /tmp/_plan.txt
import sys
n = int(sys.argv[1])
# n を「1 と 2 の冪」に分ける。x2, x4, x8… を作ってから足す
pow2, rest, plan = 1, n, []
while pow2 * 2 <= n:
    plan.append((f"x{pow2}", f"x{pow2}", f"x{pow2*2}")); pow2 *= 2
cur, acc = pow2, pow2
while acc < n:
    take = 1
    while take * 2 <= n - acc: take *= 2
    plan.append((f"x{acc}", f"x{take}", f"x{acc+take}")); acc += take
for a, b, c in plan: print(a, b, c)
PY
    while read -r a b c; do
        [ -n "$a" ] || continue
        pa="$P/$a"; pb="$P/$b"; pc="$P/$c"
        [ "$a" = "x1" ] && pa="$REV_DIR"
        [ "$b" = "x1" ] && pb="$REV_DIR"
        merge_one "$pa" "$pb" "$pc"
    done < /tmp/_plan.txt
    [ -d "$XN/meta" ] || die "x$COPIES を作れなかった" "/tmp/_plan.txt の並びを見る"
fi
REV_FRAMES=$(counts "$XN" | awk '{print $2}')

step "6. 配布データに混ぜる"
if [ -d "$MERGED/meta" ]; then
    echo "  在る → $(counts "$MERGED")"
else
    python "$TOOLS/npz_to_lerobot.py" align --base "$DATA" --add "$XN" \
        || die "align に失敗" "形（dtype と shape）が本当に違う。データを作り直す"
    python "$TOOLS/npz_to_lerobot.py" merge \
        --base "$DATA" --base-repo-id lerobot/libero_combined_20hz \
        --add  "$XN"   --add-repo-id  local/t3_reverse \
        --output "$MERGED" --output-repo-id local/t3_proto >/dev/null \
        || die "merge に失敗" "$MERGED に中身が残っていれば消す" \
               "動画で落ちるなら 3 節の symlink が張れていない"
    echo "  出来た → $(counts "$MERGED")"
fi
WANT_MERGED="$(( 19533 + $(counts "$XN" | awk '{print $1}') )) $(( 3028708 + REV_FRAMES )) 114"
have "$MERGED" "$WANT_MERGED" \
    || die "混ぜた数が合わない: $(counts "$MERGED")" "期待は $WANT_MERGED"

step "7. 正規化統計を配布データの値に据え置く"
if [ -f "$MERGED/meta/stats.json.bak" ]; then
    echo "  据え置き済み（stats.json.bak が在る）"
else
    python3 "$TOOLS/check_merge_stats.py" verify --parts "$DATA" "$XN" --merged "$MERGED" \
        || die "混ぜた統計が式と合わない" "merge がおかしい。$MERGED を消してやり直す"
    python3 "$TOOLS/check_merge_stats.py" freeze --merged "$MERGED" --source "$DATA" \
        || die "据え置きに失敗" "$MERGED/meta/stats.json を見る"
fi
python3 - "$MERGED" "$DATA" <<'PY' || die "据え置きが効いていない" "stats.json.bak を戻して freeze を打ち直す"
import json, sys
m = json.load(open(sys.argv[1] + "/meta/stats.json"))
d = json.load(open(sys.argv[2] + "/meta/stats.json"))
ok = m["action"]["std"] == d["action"]["std"] and m["observation.state"]["std"] == d["observation.state"]["std"]
print("  据え置き確認:", "配布と一致" if ok else "**一致しない**")
sys.exit(0 if ok else 1)
PY

step "8. 出発点の重み"
if [ -d "$WEIGHTS/vlm" ] && [ -f "$WEIGHTS/model.safetensors" ]; then
    echo "  在る（$(du -sh "$WEIGHTS" | cut -f1)）"
else
    hf download siruku6/pi05_full_runpod --local-dir "$WEIGHTS" >/dev/null \
        || die "rp3000 を落とせない" "hf auth whoami で認証を確認する"
    [ "$(ls "$WEIGHTS/vlm" | wc -l)" = 4 ] || die "vlm/ が 4 ファイルでない" "落とし直す"
    echo "  落とした（$(du -sh "$WEIGHTS" | cut -f1)）"
fi

step "9. 変数を書き出す"
cat > "$HOME/run_env.sh" <<EOF
# blackwell_setup.sh が書いた。**ターミナルが落ちたら . ~/run_env.sh で戻る**
. ~/env_pi05.sh
export NVME=$NVME
export DATA=$DATA
export P=$P
export XN=$XN
export MERGED=$MERGED
export TOOLS=$TOOLS
export WEIGHTS=$WEIGHTS
export CODE_DIR=$CODE_DIR
export PATH="\$HOME/.local/bin:\$PATH"
EOF
echo "  ~/run_env.sh"

echo
echo "=== 出来上がり ==="
printf '  %-10s %s\n' 配布 "$(counts "$DATA")" 逆向き "$(counts "$XN")" 混ぜた "$(counts "$MERGED")"
echo "  逆向きの割合: $(python3 -c "print(f'{$REV_FRAMES/(3028708+$REV_FRAMES):.2%}')")"
echo
echo "次に打つもの:"
echo "  RUN=<走りの名前> EXPERT_ONLY=1 bash $CODE_DIR/blackwell_launch.sh"
echo
echo "**ターミナルが落ちたら . ~/run_env.sh で変数が戻る。このスクリプトも打ち直してよい**"
