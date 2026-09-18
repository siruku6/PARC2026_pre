#!/usr/bin/env bash
# 016 推奨方式・一括パイプライン（1 回の Bash で済むよう docker run を内部にまとめる）。
#
#   1. 逆向き bddl ＋ 逆向き初期状態を生成（docker run）※合成する組だけ
#   2. 配布キットが既に持っている逆向きタスクを、そのまま逆向き側に据える（合成なし）
#   3. 逆向き定義の妥当性検証（docker run）
#      - bddl が読める / t=0 で逆向き goal 未達（偽 Yes 防止）/ A が順方向着地先にある
#   4. 順／逆の初期状態を 1 枚ずつ描画（docker run）。目視で破綻を弾く関門
#   5. 順／逆 assets をキットの t3 形式に組み立て（ホスト側、docker 不要）
#
# 引き出し（extract）は別途済んでいること（work/probe_016/l10 に bddl と init が在る）。
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
L10="$REPO/work/probe_016/l10"
REV="$REPO/work/probe_016/reverse"
IMG="$REPO/work/probe_016/init_frames"
T3A="$REPO/work/probe_016/t3assets"
KIT_ROOT="${KIT_ROOT:-$(cd "$REPO/../b_PARC2026_final" 2>/dev/null && pwd)}"
EPISODES="${EPISODES:-2}"
mkdir -p "$REV" "$IMG" "$T3A"

# 逆向きを**こちらで合成する**組（順方向は libero_10 ＝ 学習データ内）
GEN_STEMS=(
  "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
  "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
)
# 逆向きが**配布キットに既に在る**組（合成なし＝「逆向きの作り方が悪い」の反論が効かない）
#   順方向 stem = キットのタスク名
KIT_PAIRS=(
  "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate=take_the_two_mugs_off_the_plates_and_place_them_on_the_table"
)

PROBE="$REPO/solution/tools/probe"
# コンテナの雑音（CUDA バナー・robosuite の警告・gym の非推奨告知）を落とす
QUIET='^\[libero\]|WARNING|warn|is not|^\[Mj|NVIDIA|license|License|Copyright|^=+$|CUDA|^\s*$|docs\.nvidia|Gym has been|Please upgrade|migration_guide|unmaintained'

ALL_STEMS=("${GEN_STEMS[@]}")
for pair in "${KIT_PAIRS[@]}"; do ALL_STEMS+=("${pair%%=*}"); done

stage_for() { # $1... = stem 群 → 一時 /in を作り、そのパスを返す
    local d; d=$(mktemp -d)
    local s
    for s in "$@"; do
        cp "$L10/bddl_files/$s.bddl" "$d/" 2>/dev/null
        cp "$L10/init_files/$s.pruned_init" "$d/" 2>/dev/null
        cp "$REV/${s}_rev.bddl" "$REV/${s}_rev.pruned_init" "$d/" 2>/dev/null
    done
    echo "$d"
}

run_in_container() { # $1=/in $2=/out $3=スクリプト名 ... 以降は python 引数
    # probe ディレクトリを丸ごと /app へ置く。スクリプトを 1 本ずつ mount すると
    # 共通モジュール（_016_libero_env.py）が import できない。
    local indir="$1" outdir="$2" script="$3"; shift 3
    docker run --rm \
        -v "$indir:/in:ro" -v "$outdir:/out" \
        -e LIBERO_ROOT=/workspace/LIBERO-plus -e MUJOCO_GL=egl \
        -v "$PROBE:/app:ro" \
        parc2026-final python -u "/app/$script" --in /in --out /out "$@" \
        2>&1 | grep -vE "$QUIET"
}

echo "==== [1/5] 逆向き生成（${#GEN_STEMS[@]} タスク）===="
for s in "${GEN_STEMS[@]}"; do
    echo "-- $s"
    # 前回の生成物を消してから作る（失敗時に古い逆向きが残って評価へ流れるのを防ぐ）
    rm -f "$REV/${s}_rev.bddl" "$REV/${s}_rev.pruned_init" "$REV/${s}_rev.placement.json"
    stage=$(stage_for "$s")
    run_in_container "$stage" "$REV" _016_make_reverse.py --stem "$s" --episodes "$EPISODES"
    rm -rf "$stage"
    if [ -f "$REV/${s}_rev.bddl" ] && [ -f "$REV/${s}_rev.pruned_init" ]; then
        echo "   OK: ${s}_rev.{bddl,pruned_init}"
    else
        echo "   !! ${s} の逆向き生成が不完了（この組は以降から外れる）"
    fi
done

echo
echo "==== [2/5] 配布キットの逆向きタスクを据える（合成なし）===="
if [ -n "$KIT_ROOT" ] && [ -d "$KIT_ROOT/compe/t3/assets" ]; then
    pair_args=()
    for pair in "${KIT_PAIRS[@]}"; do pair_args+=(--pair "$pair"); done
    python3 "$PROBE/_016_use_kit_reverse.py" \
        --kit-assets "$KIT_ROOT/compe/t3/assets" --reverse-dir "$REV" "${pair_args[@]}"
else
    echo "   !! キットが見つからない（KIT_ROOT=${KIT_ROOT}）。この組は外れる"
fi

echo
echo "==== [3/5] 逆向き定義の妥当性検証 ===="
vstage=$(stage_for "${ALL_STEMS[@]}")
run_in_container "$vstage" "$REV" _016_verify_reverse.py --episodes "$EPISODES"
rm -rf "$vstage"

echo
echo "==== [4/5] 初期状態の描画（目視用）===="
for s in "${ALL_STEMS[@]}"; do
    [ -f "$REV/${s}_rev.pruned_init" ] || continue
    stage=$(stage_for "$s")
    run_in_container "$stage" "$IMG" _016_render_init.py --stem "$s" --episodes "$EPISODES"
    rm -rf "$stage"
done

echo
echo "==== [5/5] t3 assets 組み立て（ホスト側）===="
# **検証を通った組だけ**を評価へ回す（理由は _016_valid_stems.py の docstring）。
mapfile -t VALID_STEMS < <(python3 "$PROBE/_016_valid_stems.py" "$REV/verify_reverse.json")
for s in "${ALL_STEMS[@]}"; do
    printf '%s\n' "${VALID_STEMS[@]}" | grep -qxF "$s" \
        || echo "   !! $s は検証不合格のため評価から外す"
done
if [ "${#VALID_STEMS[@]}" -eq 0 ]; then
    echo "   !! 検証を通った組が無い。評価へ進めない" >&2
    exit 1
fi
ARGS=()
for s in "${VALID_STEMS[@]}"; do ARGS+=(--stem "$s"); done
python3 "$PROBE/_016_assemble_t3assets.py" \
    --forward-bddl "$L10/bddl_files" \
    --forward-init "$L10/init_files" \
    --reverse "$REV" \
    --out "$T3A" \
    "${ARGS[@]}"

echo
echo "==== 成果物 ===="
echo "逆向き bddl / init: $REV"
echo "初期状態の画:       $IMG"
echo "t3 assets:          $T3A"
ls -1 "$T3A/bddl_files/libero_t3/" 2>/dev/null
