#!/usr/bin/env bash
# JAIST で回す「配布データを間引いたもの ＋ 逆向きデータ」を、このマシンで組む。
#
#   bash solution/train/jaist_dataset_build.sh              # 既定（逆向き 7 倍 ＝ 割合 48.6%）
#   COPIES=5 PER_TASK=50 bash solution/train/jaist_dataset_build.sh
#   REV_DIR=work/probe_rot/reverse_lerobot_all bash ...      # 作り直した逆向きを使う
#
# **何度打っても良い。**各段は出来上がりの本数・コマ数・タスク数を数えて、
# 正しければ飛ばす。**数が 1 つでも違えば原因を出して止まる。**
#
# **Blackwell ではなくこのマシンで組む理由。**JAIST は VRAM 20 GB で
# action expert 限定しか載らず、ディスクの空きも 22 GB しかない。
# **配布データを丸ごと持ち込めないので、先にここで間引いて混ぜてから上げる。**
#
# **なぜ `libero_plus` を間引くのか**（配布 tar の `libero_combined_20hz` ではなく）:
#   - `rp3000` は `libero_plus` で学習済みなので、配布側は「もう知っている」側に
#     なり、行動側をほとんど押さない（`stage4_b16_24k` の実測）
#   - **据え置く統計が `rp3000` 同梱の normalizer そのもの**になる
#   - 配布 tar の 71 個の未学習の指示文は `all111` の退化の筋である
#
# Generates:
#   work/artifacts/$SUB          間引いた配布データ
#   work/artifacts/${REV_TAG}    逆向きの N 倍
#   work/artifacts/$MIX          混ぜて正規化を据え置いたもの
set -uo pipefail

REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
cd "$REPO" || exit 1

COPIES="${COPIES:-7}"
PER_TASK="${PER_TASK:-75}"
SRC="${SRC:-work/artifacts/libero_plus}"
REV_DIR="${REV_DIR:-work/probe_017/reverse_lerobot_all}"
NORM="${NORM:-work/artifacts/pi05_rp3000/policy_preprocessor_step_2_normalizer_processor.safetensors}"
SUB="${SUB:-libero_plus_sub$PER_TASK}"
REV_TAG="${REV_TAG:-rev_x$COPIES}"
MIX="${MIX:-jaist_mix_${PER_TASK}_x$COPIES}"
HF_REPO="${HF_REPO:-siruku6/jaist_mix}"
ART="$REPO/work/artifacts"

# 期待値。**ここが違ったら止める。**
WANT_SRC="14347 2238036 40"

step() { echo; echo "=== $* ==="; }
die() {
    echo; echo "**止まった: $1**"; shift
    for line in "$@"; do echo "  $line"; done
    exit 1
}
counts() {  # $1=データセット → "本数 コマ数 タスク数"。無ければ空
    python3 -c "
import json,sys
try:
    i = json.load(open('$1/meta/info.json'))
except Exception:
    sys.exit(1)
print(i['total_episodes'], i['total_frames'], i['total_tasks'])" 2>/dev/null
}
want() {  # $1=データセット $2=期待値 → 一致すれば 0
    [ "$(counts "$1")" = "$2" ]
}

# ---------------------------------------------------------------- 1 元データ
step "1/6 元データを確かめる"
got=$(counts "$SRC")
[ -n "$got" ] || die "$SRC が無い" \
    "hf download lerobot/libero_plus --repo-type dataset --local-dir $SRC" \
    "15 GB ある。落ちきるまで 20 分ほど"
[ "$got" = "$WANT_SRC" ] || die "$SRC の数が違う（出た: $got ／ 期待: $WANT_SRC）" \
    "落とし切れていない可能性がある。.incomplete が残っていないか見ること:" \
    "  find $SRC -name '*.incomplete' | head"
echo "  $SRC = $got  ✓"

got=$(counts "$REV_DIR")
[ -n "$got" ] || die "$REV_DIR が無い" \
    "逆向きデータを先に作るか、置き場を REV_DIR で渡すこと"
REV_EP=$(echo "$got" | cut -d' ' -f1)
REV_FR=$(echo "$got" | cut -d' ' -f2)
REV_TK=$(echo "$got" | cut -d' ' -f3)
echo "  $REV_DIR = $got  ✓"

# ---------------------------------------------------------------- 2 間引く
WANT_SUB=""
step "2/6 配布データを課題ごとに $PER_TASK 本へ間引く"
if [ -n "$(counts "$ART/$SUB")" ]; then
    echo "  もうある: $(counts "$ART/$SUB")  — 飛ばす"
else
    ROOT="$SRC" OUT_DIR="$ART" \
        bash solution/tools/track3_reversal/run_lerobot_subset.sh "$SUB" "$PER_TASK" \
        || die "間引きが落ちた" "ログを見ること。docker のイメージ名は IMAGE で変えられる"
fi
SUB_N=$(counts "$ART/$SUB")
[ -n "$SUB_N" ] || die "間引いた $ART/$SUB が出来ていない" "上のログを見ること"
SUB_EP=$(echo "$SUB_N" | cut -d' ' -f1)
SUB_FR=$(echo "$SUB_N" | cut -d' ' -f2)
echo "  $ART/$SUB = $SUB_N"

# ---------------------------------------------------------------- 3 N 倍
step "3/6 逆向きデータを $COPIES 倍にする（倍々で組む）"
WANT_REV="$((REV_EP * COPIES)) $((REV_FR * COPIES)) $REV_TK"
if want "$ART/$REV_TAG" "$WANT_REV"; then
    echo "  もうある: $WANT_REV  — 飛ばす"
else
    # 倍々の作り方を組む。**1 本ずつ足すと merge が COPIES-1 回になる。**
    plan=$(python3 -c "
n = $COPIES
have, steps = [1], []
while have[-1] < n:
    d = have[-1] * 2
    if d <= n: steps.append((have[-1], have[-1], d)); have.append(d)
    else: break
cur = have[-1]
for h in reversed(have):
    while cur + h <= n:
        steps.append((cur, h, cur + h)); cur += h
    if cur == n: break
for a, b, c in steps: print(a, b, c)")
    [ -n "$plan" ] || echo "  1 倍なのでそのまま使う"
    src_of() { [ "$1" = 1 ] && echo "probe_017/$(basename "$REV_DIR")" || echo "artifacts/rev_x$1"; }
    while read -r a b c; do
        [ -n "$a" ] || continue
        if want "$ART/rev_x$c" "$((REV_EP * c)) $((REV_FR * c)) $REV_TK"; then
            echo "  rev_x$c はもうある — 飛ばす"; continue
        fi
        echo "  x$a + x$b -> x$c"
        OUT="$REPO/work" bash solution/tools/track3_reversal/run_lerobot_merge.sh \
            "artifacts/rev_x$c" "$(src_of "$a")" "$(src_of "$b")" \
            || die "逆向きの x$c で落ちた" "書きかけの $ART/rev_x$c を消してから打ち直すこと"
    done <<< "$plan"
fi
[ "$COPIES" = 1 ] && REV_MIX="probe_017/$(basename "$REV_DIR")" || REV_MIX="artifacts/$REV_TAG"
want "$ART/$REV_TAG" "$WANT_REV" || [ "$COPIES" = 1 ] \
    || die "$ART/$REV_TAG の数が違う（出た: $(counts "$ART/$REV_TAG") ／ 期待: $WANT_REV）" \
           "途中の rev_x* を消してから打ち直すこと"

# ---------------------------------------------------------------- 4 混ぜる
step "4/6 間引いた配布データと混ぜる"
WANT_MIX="$((SUB_EP + REV_EP * COPIES)) $((SUB_FR + REV_FR * COPIES)) $((40 + REV_TK))"
if want "$ART/$MIX" "$WANT_MIX"; then
    echo "  もうある: $WANT_MIX  — 飛ばす"
else
    OUT="$REPO/work" bash solution/tools/track3_reversal/run_lerobot_merge.sh \
        "artifacts/$MIX" "artifacts/$SUB" "$REV_MIX" \
        || die "混ぜるところで落ちた" \
               "`align` の食い違いなら軸名ではなく形が違う。書きかけの $ART/$MIX を消すこと"
fi
want "$ART/$MIX" "$WANT_MIX" \
    || die "$ART/$MIX の数が違う（出た: $(counts "$ART/$MIX") ／ 期待: $WANT_MIX）" \
           "$ART/$MIX を消してから打ち直すこと"
share=$(python3 -c "print(f'{$REV_FR * $COPIES / ($SUB_FR + $REV_FR * $COPIES) * 100:.1f}')")
echo "  $ART/$MIX = $WANT_MIX  ／ **逆向きの割合 ${share}%**"

# ------------------------------------------------------- 5 正規化を据え置く
step "5/6 正規化統計を rp3000 の normalizer に据え置く"
if [ -f "$ART/$MIX/meta/stats.json.bak" ]; then
    echo "  もう据え置いてある（stats.json.bak がある）— 飛ばす"
else
    [ -f "$NORM" ] || die "$NORM が無い" \
        "hf download siruku6/pi05_full_runpod --local-dir work/artifacts/pi05_rp3000"
    python3 solution/tools/track3_reversal/check_merge_stats.py \
        freeze --merged "$ART/$MIX" --source "$NORM" \
        || die "据え置きが落ちた" "$NORM が normalizer の safetensors かを見ること"
fi
echo
echo "  **据え置いた値は rp3000 が学習した尺度そのものである。**"
echo "  元の値は $ART/$MIX/meta/stats.json.bak にある。"

# ---------------------------------------------------------------- 6 上げる
step "6/6 HF へ上げる"
sz=$(du -sh "$ART/$MIX" | cut -f1)
cat <<EOS
  出来上がり: $ART/$MIX（$sz）

  **上げるのは手で打つ。**非公開リポジトリを作るので、確かめてから流したい。

    hf upload $HF_REPO $ART/$MIX . --repo-type dataset --private

  JAIST 側:

    hf download $HF_REPO --repo-type dataset --local-dir ~/data/$MIX
    RUN=pi05_t3rev DATASET=~/data/$MIX STEPS=$(python3 -c "
print(round($REV_FR / ($REV_FR * $COPIES / ($SUB_FR + $REV_FR * $COPIES)) / 16 / 100) * 100)") \\
      bash solution/train/jaist_launch.sh

  歩数の根拠: 逆向き $REV_FR コマを **1 コマ 1 回**見る量
  （必要なサンプル数 ＝ こちらのコマ数 ÷ 割合 ${share}%、batch 16）。
EOS
