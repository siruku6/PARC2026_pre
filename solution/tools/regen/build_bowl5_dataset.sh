#!/usr/bin/env bash
# 丼 5 課題を入れ替えた学習データを組み、**4 つの門を通してから**渡せる形にする。
#
#   bash solution/tools/regen/build_bowl5_dataset.sh
#
# 先に `run_scripted_bowls.sh` を通して `work/regen/lerobot_bowl/<段>/` を作っておくこと。
#
# **[howto/16 の 5.2 節](../../../docs/howto/16_bowl_regen_train_eval.md) と 1 か所だけ違う。**
# あちらは 12,467 本の base に段を 1 つずつ足していくので、**15 GB の写しが 8 回**できる。
# ここは**先に段どうしを 1 つにまとめてから、base へ 1 回だけ足す。**
# 出来上がりは同じで、写しが 8 回から 1 回になる。
#
# **上げるのは手で打つ。**外へ出す操作なので、門が 4 つ通ったのを見てから流す
# （`jaist_dataset_build.sh` と同じ扱い）。
#
# Reads:
#   - work/artifacts/libero_plus（配布 14,347 本）
#   - work/regen/lerobot_bowl/<段>/（作り直した手本）
#   - work/regen/grader_texts.txt（採点器が送る 16 文）
# Generates:
#   - work/artifacts/libero_plus_nobowl5   丼 5 課題を外した 12,467 本
#   - work/regen/lerobot_bowl_all          段を 1 つにまとめたもの
#   - work/artifacts/libero_plus_bowl5     学習に渡すもの（＋ bowl5_expected.json）
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="${SRC:-work/artifacts/libero_plus}"
NOBOWL="${NOBOWL:-work/artifacts/libero_plus_nobowl5}"
STAGES="${STAGES:-work/regen/lerobot_bowl}"
ALL="${ALL:-work/regen/lerobot_bowl_all}"
OUT="${OUT:-work/artifacts/libero_plus_bowl5}"
KEEP="${KEEP:-work/artifacts/libero_states/keep_without_bowl5.json}"
GRADER="${GRADER:-work/regen/grader_texts.txt}"
HF_REPO="${HF_REPO:-siruku6/libero_plus_bowl5}"
R() { IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh "$@"; }

die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }
step() { echo; echo "=== $* ==="; }
eps() { python3 -c "import json;print(json.load(open('$1/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0; }

# ---------------------------------------------------------------- 1 外す
step "1. 配布データから丼 5 課題を外す"
if [ "$(eps "$NOBOWL")" = 12467 ]; then
    echo "  済み（12,467 本）"
else
    [ -f "$KEEP" ] || R solution/tools/regen/make_replace_keep.py \
        --root "$SRC" --out "$KEEP" || die "keep-file を作れない"
    R solution/tools/contact_clean/rebuild_dataset.py \
        --root "$SRC" --output "$NOBOWL" --keep-file "$KEEP" || die "外せない"
    [ "$(eps "$NOBOWL")" = 12467 ] || die "残った本数が 12,467 でない（$(eps "$NOBOWL")）" \
        "配布データが期待と違う。make_replace_keep.py の出力を読む"
    echo "  12,467 本（動画はハードリンクなのでディスクは増えない）"
fi

# ---------------------------------------------------------------- 2 まとめる
# **段どうしを先に 1 つにする。**base へ 1 段ずつ足すと 15 GB の写しが 8 回できる。
step "2. 作り直した段を 1 つにまとめる"
mapfile -t DIRS < <(ls -d "$STAGES"/*/ 2>/dev/null)
[ "${#DIRS[@]}" -gt 0 ] || die "$STAGES に段が無い" "run_scripted_bowls.sh を先に通す"
echo "  段 ${#DIRS[@]} 個: $(for d in "${DIRS[@]}"; do printf '%s(%s) ' "$(basename "$d")" "$(eps "$d")"; done)"
if [ "$(eps "$ALL")" -gt 0 ]; then
    echo "  済み（$(eps "$ALL") 本）"
else
    BASE="${DIRS[0]%/}"
    for d in "${DIRS[@]:1}"; do
        NEXT="${ALL}_tmp_$(basename "${d%/}" | tr ':' '_')"
        R solution/tools/scripted_demo/npz_to_lerobot.py merge \
            --base "$BASE" --base-repo-id local/bowl_all \
            --add "${d%/}" --add-repo-id local/bowl_stage \
            --output "$NEXT" --output-repo-id local/bowl_all || die "段をまとめられない: $d"
        [ "$BASE" != "${DIRS[0]%/}" ] && rm -rf "$BASE"
        BASE="$NEXT"
    done
    mv "$BASE" "$ALL"
    echo "  $(eps "$ALL") 本"
fi

# ---------------------------------------------------------------- 3 足す
step "3. 配布データへ 1 回だけ足す"
if [ "$(eps "$OUT")" -gt 0 ]; then
    echo "  済み（$(eps "$OUT") 本）"
else
    R solution/tools/scripted_demo/npz_to_lerobot.py merge \
        --base "$NOBOWL" --base-repo-id lerobot/libero_plus \
        --add "$ALL" --add-repo-id local/bowl_all \
        --output "$OUT" --output-repo-id local/libero_plus_bowl5 || die "足せない"
fi

# ---------------------------------------------------------------- 門
step "門 1. 本数・コマ数・課題数"
python3 - "$OUT" "$SRC" <<'PY'
import json, sys
a = json.load(open(f"{sys.argv[1]}/meta/info.json"))
b = json.load(open(f"{sys.argv[2]}/meta/info.json"))
print(f"  {'':10} {'入れ替え後':>12} {'配布':>12}")
for k in ("total_episodes", "total_frames", "total_tasks"):
    print(f"  {k:10} {a[k]:>12,} {b[k]:>12,}")
assert a["total_tasks"] == b["total_tasks"], "**課題数が違う。未学習のタスクが混ざっている**"
assert 14200 <= a["total_episodes"] <= 14400, "**本数が想定の外**"
PY
[ $? = 0 ] || die "門 1 で落ちた"

step "門 2. 指示文が、評価でモデルが見る文と合っているか"
R solution/tools/scripted_demo/check_dataset_instructions.py "$OUT" \
    --grader-texts "$GRADER" || die "門 2 で落ちた" \
    "寄せられていない文がある。**評価で一度も引かれない**（knowledge/23）"

step "門 3. 指令の形が配布データと揃っているか"
R solution/tools/motion/report_smoothness.py --dataset "$SRC" --episodes 200 \
    --npz "作り直し=work/regen/npz_bowl/*/*.npz" 2>/dev/null \
    || echo "  （npz は変換後に消えている。段の生成時のログを見ること）"

step "門 4. 正規化統計のずれ"
# **pi05 は state と action を mean/std で正規化し、`lerobot_train.py` が
# `dataset.meta.stats` で上書きする。**混ぜた瞬間に配布データ側の正規化も変わる。
# **対照の `pi05_full_local` は配布データの統計で学習されている**ので、
# ずれが大きければ `freeze` で据え置いて 1 変数を保つ。
R solution/tools/track3_reversal/check_merge_stats.py predict \
    --provided "$SRC/meta/stats.json" --ours "$ALL/meta/stats.json" \
    --copies 1 --merged "$OUT" || echo "  （predict が通らない。verify で見る）"
echo
echo "  **歪みが大きければ据え置く（打つのは手で）:**"
echo "    IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \\"
echo "        solution/tools/track3_reversal/check_merge_stats.py freeze \\"
echo "        --merged $OUT --source $SRC/meta/stats.json"

# ---------------------------------------------------------------- 期待値
# **Blackwell 側で突き合わせる値を、ここで書き出す。**手で書き写すと、
# 写し間違いを検算で捕まえられない。
step "期待値を同梱する"
python3 - "$OUT" <<'PY'
import json, sys
root = sys.argv[1]
info = json.load(open(f"{root}/meta/info.json"))
want = {k: info[k] for k in ("total_episodes", "total_frames", "total_tasks")}
json.dump(want, open(f"{root}/bowl5_expected.json", "w"), indent=2)
print("  bowl5_expected.json:", want)
PY

step "できあがり"
du -sh "$OUT"
cat <<EOS

  **上げるのは手で打つ。**上の門が 4 つとも通ったのを見てから流すこと。

    hf upload $HF_REPO $OUT . --repo-type dataset --private

  降ろす側（Blackwell）:

    bash ~/pi05_code/solution/train/blackwell_bowl5.sh

  JAIST 側（action expert だけ・対照と 1 変数）:

    python3 solution/train/train_pi05.py --run-name pi05_bowl5 \\
        --dataset-root $OUT --steps 24000 --batch-size 16
EOS
