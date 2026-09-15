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
# **コンテナは root で書く。**戻さないと、こちらから中継物を消せない
# （`run_scripted_bowls.sh` と同じ手）。
own() { docker run --rm -v "$PWD/$1:/out" --entrypoint chown parc2026-final \
            -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true; }

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
        # **中継物は捨てる。**段を 8 つ重ねるので、残すと 8 個ぶん場所を食う。
        # 消してよいのはこの loop が作ったものだけ（1 段目は元データである）。
        if [ "$BASE" != "${DIRS[0]%/}" ] && [ -d "$BASE" ]; then
            rm -rf -- "$BASE"
        fi
        BASE="$NEXT"
    done
    mv "$BASE" "$ALL"
    own "$ALL"
    echo "  $(eps "$ALL") 本"
fi

# ---------------------------------------------------------------- 3 足す
step "3. 配布データへ 1 回だけ足す"
if [ "$(eps "$OUT")" -gt 0 ]; then
    echo "  済み（$(eps "$OUT") 本）"
else
    # **先に列の書き方を揃える。**中身は同じでも、lerobot の版が上がると
    # `features` の飾りが変わり、`merge_datasets` が一致を求めて落ちる
    # （非動画の列の `fps`・動画の `is_depth_map` の綴り・軸名の 3 つ）。
    # **段どうしの merge では起きない**——どちらも同じ版で作ったからである。
    R solution/tools/scripted_demo/npz_to_lerobot.py align \
        --base "$NOBOWL" --add "$ALL" || die "列の書き方を揃えられない"
    R solution/tools/scripted_demo/npz_to_lerobot.py merge \
        --base "$NOBOWL" --base-repo-id lerobot/libero_plus \
        --add "$ALL" --add-repo-id local/bowl_all \
        --output "$OUT" --output-repo-id local/libero_plus_bowl5 || die "足せない"
    own "$OUT"
fi

# ---------------------------------------------------------------- 門
step "門 1. 本数・コマ数・課題数"
# **足し算が合うかを見る。**「だいたいこの範囲」で見ると、段が 1 つ落ちていても
# 通ってしまう。**外した残りと作り直しの足し算に、ちょうど一致すること。**
python3 - "$OUT" "$SRC" "$NOBOWL" "$ALL" <<'GATE1'
import json, sys
def info(path):
    return json.load(open(f"{path}/meta/info.json"))
out, src, nobowl, add = (info(x) for x in sys.argv[1:5])
print(f"  {'':16} {'入れ替え後':>12} {'配布':>12}")
for k in ("total_episodes", "total_frames", "total_tasks"):
    print(f"  {k:16} {out[k]:>12,} {src[k]:>12,}")
print(f"  {'内訳':16} {nobowl['total_episodes']:,}（外した残り）"
      f" + {add['total_episodes']:,}（作り直し）")
assert out["total_tasks"] == src["total_tasks"], (
    "**課題数が違う。未学習のタスクが混ざっている**")
assert out["total_episodes"] == nobowl["total_episodes"] + add["total_episodes"], (
    "**本数の足し算が合わない。段が落ちているか、二重に足している**")
assert out["total_frames"] == nobowl["total_frames"] + add["total_frames"], (
    "**コマ数の足し算が合わない**")
GATE1
[ $? = 0 ] || die "門 1 で落ちた"

step "門 2. 指示文が、評価でモデルが見る文と合っているか"
R solution/tools/scripted_demo/check_dataset_instructions.py "$OUT" \
    --grader-texts "$GRADER" || die "門 2 で落ちた" \
    "寄せられていない文がある。**評価で一度も引かれない**（knowledge/23）"

step "門 3. 指令の形が配布データと揃っているか"
# **npz ではなくデータセットの `action` 列から測る。**npz は段ごとに捨てる
# 設計なので、ここへ来る頃には残っていない。**捨てたものを測ろうとする門は、
# 黙って飛ばされる門である。**
R -c "
import sys; sys.path.insert(0, 'solution/tools')
from motion.smoothness import format_table, load_dataset_episodes
print(format_table({
    '配布': load_dataset_episodes('$SRC', 200),
    '作り直し': load_dataset_episodes('$ALL', 200),
    '入れ替え後': load_dataset_episodes('$OUT', 200),
}))" || die "門 3 で落ちた"

step "門 4. 正規化統計のずれ"
# **pi05 は state と action を mean/std で正規化し、`lerobot_train.py` が
# `dataset.meta.stats` で上書きする。**混ぜた瞬間に配布データ側の正規化も変わる。
#
# **この走りは据え置かない。**据え置くのは「統計を持っている重みから続ける」
# ときだけである（[reference/29 の 6.3](../../../docs/reference/29_training_data_requirements.md)）。
# ここは統計を持たない `pi05_libero_base` から通すので、**最初の学習が尺度を決める。**
#
# **揃えるべきは数値ではなく手順である。**対照の `pi05_full_local` は
# 「そのデータ自身の統計」で学習している。据え置くと、こちらだけが
# 「自分のデータに合わない統計を使う」ことになり、**差が 1 つ増える。**
# 出るずれはデータを入れ替えたことの帰結であって、別の変数ではない。
#
# **それでも数は見る。**桁が違えば話が変わる——逆向きを 2 割混ぜた回は
# 配布側の回転チャンネルが 0.43 倍に縮み、Track2 が 0.9167 から 0.5833 へ落ちた
# （`plan/23`）。**倍率が 0.5 を割るか 2 を超えたら、混ぜ方そのものを疑うこと。**
python3 - "$OUT" "$SRC" <<'GATE4'
import json, sys
import numpy as np
def stats(path):
    return json.load(open(f"{path}/meta/stats.json"))
out, src = stats(sys.argv[1]), stats(sys.argv[2])
worst = 1.0
for key in ("action", "observation.state"):
    m0, s0 = (np.array(src[key][f]) for f in ("mean", "std"))
    m1, s1 = (np.array(out[key][f]) for f in ("mean", "std"))
    ratio, shift = s1 / s0, np.abs((m1 - m0) / s0)
    worst = max(worst, ratio.max(), 1.0 / ratio.min())
    print(f"  {key:18} std 倍率 {ratio.min():.4f}〜{ratio.max():.4f}"
          f"  平均のずれ 最大 {shift.max():.4f} σ")
assert worst < 2.0, "**尺度が 2 倍以上動いた。混ぜ方そのものを疑うこと**"
GATE4
[ $? = 0 ] || die "門 4 で落ちた"

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
