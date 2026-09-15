#!/usr/bin/env bash
# 丼 5 課題の手本を `RimPickPlace` で作り直し、LeRobot 形式まで持っていく。
#
# **npz を全部溜めてから変換すると入らない。** 1 本 54 MB・2,000 本で 108 GB
# になるのに対し、空きは 120 GB ほどしかない。**小分けに作って変換し、
# 変換できたぶんの npz をその場で捨てる。**
#
# 段は「摂動 × 初期状態の出どころ」で切る。
#
#   摂動   base（なし）・initstate_282（2.17 cm）・242（3.58 cm）・372（9.67 cm）
#   初期   pi（`.pruned_init` の 50 通り＝採点器と同じ）・di（人の手本の 50 通り）
#
# **変換では指示文を学習データの書き方へ寄せる**（`--normalize-instructions`）。
# `record` が残すのは bddl の書き方で、配布データとは綴りが違う。寄せずに足すと
# **41 番目の課題**になり、評価で一度も引かれない（`policy_server.py` が入り口で
# 学習データの 40 種へ寄せてからモデルへ渡すため）。
#
# **`on_the_ramekin` だけ di を作らない。**手本側の置き場所では 0/50 しか
# 通らない（`.pruned_init` では 50/50）。作っても門で全部落ちるので回さない。
#
# 使い方
#   bash solution/tools/regen/run_scripted_bowls.sh            # 全部
#   STAGES="base:pi base:di" bash solution/tools/regen/run_scripted_bowls.sh
#
# 環境変数（省略時の既定）
#   OUT_NPZ        npz の置き場所（段ごとに掃除する）
#   OUT_LEROBOT    変換したデータセットの置き場所
#   KEEP_NPZ=0     1 にすると npz を消さない（デバッグ用。ディスクに注意）
#   ROUND=4        同時に変換する段の数。**1 段 13.5 GB の npz を抱える**
#
# Reads:
#   - LIBERO-plus の bddl（コンテナの中）・work/artifacts/libero_states
# Generates:
#   - $OUT_LEROBOT/<段>/ … 段ごとの LeRobot データセット
#   - work/regen/logs/bowls_*.log
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"
OUT_NPZ="${OUT_NPZ:-work/regen/npz_bowl}"
OUT_LEROBOT="${OUT_LEROBOT:-work/regen/lerobot_bowl}"
KEEP_NPZ="${KEEP_NPZ:-0}"
ROUND="${ROUND:-4}"
STATES="work/artifacts/libero_states"
mkdir -p "$OUT_NPZ" "$OUT_LEROBOT" work/regen/logs

TASKS=(
  pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
)
DEFAULT_STAGES="base:pi base:di 282:pi 282:di 242:pi 242:di 372:pi 372:di"
read -r -a STAGE_LIST <<< "${STAGES:-$DEFAULT_STAGES}"

generate_stage() {          # $1 = "<摂動>:<初期>"
    local stage="$1" perturb="${1%%:*}" source="${1##*:}"
    local dir="$OUT_NPZ/$stage"
    mkdir -p "$dir"
    for task in "${TASKS[@]}"; do
        # 手本側の置き場所では通らない組を飛ばす
        if [ "$source" = "di" ] && [ "$task" = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate" ]; then
            continue
        fi
        local args=(--task "$task" --output "$dir" --max-steps 300)
        [ "$perturb" != "base" ] && args+=(--variant "${task}_view_0_0_100_0_0_initstate_${perturb}")
        [ "$source" = "di" ] && args+=(--init-from-demos "$STATES")
        bash solution/tools/regen/run_in_container.sh \
            solution/tools/scripted_demo/scripted_demo.py record "${args[@]}" \
            > "work/regen/logs/bowls_gen_${stage}_${task:26:24}.log" 2>&1 &
    done
    wait
    docker run --rm -v "$REPOSITORY_ROOT/$OUT_NPZ:/out" --entrypoint chown \
        parc2026-final -R "$(id -u):$(id -g)" /out
    echo "  生成 $stage: $(ls "$dir" | wc -l) 本  $(du -sh "$dir" | cut -f1)"
}

build_stage() {             # $1 = 段の名前
    local stage="$1"
    IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
        solution/tools/scripted_demo/npz_to_lerobot.py build \
        --npz-dir "$OUT_NPZ/$stage" --output "$OUT_LEROBOT/$stage" \
        --repo-id "local/bowl_${stage/:/_}" --normalize-instructions \
        > "work/regen/logs/bowls_build_${stage/:/_}.log" 2>&1
    echo "  変換 $stage: exit=$?"
}

pending=()
for stage in "${STAGE_LIST[@]}"; do
    echo "=== 生成 $stage ==="
    generate_stage "$stage"
    pending+=("$stage")
    if [ "${#pending[@]}" -ge "$ROUND" ]; then
        echo "=== 変換 ${pending[*]} ==="
        for stage2 in "${pending[@]}"; do build_stage "$stage2" & done
        wait
        docker run --rm -v "$REPOSITORY_ROOT/$OUT_LEROBOT:/out" --entrypoint chown \
            parc2026-final -R "$(id -u):$(id -g)" /out
        [ "$KEEP_NPZ" = "0" ] && for stage2 in "${pending[@]}"; do rm -rf "${OUT_NPZ:?}/$stage2"; done
        pending=()
    fi
done
if [ "${#pending[@]}" -gt 0 ]; then
    echo "=== 変換 ${pending[*]} ==="
    for stage2 in "${pending[@]}"; do build_stage "$stage2" & done
    wait
    docker run --rm -v "$REPOSITORY_ROOT/$OUT_LEROBOT:/out" --entrypoint chown \
        parc2026-final -R "$(id -u):$(id -g)" /out
    [ "$KEEP_NPZ" = "0" ] && for stage2 in "${pending[@]}"; do rm -rf "${OUT_NPZ:?}/$stage2"; done
fi
echo "=== $OUT_LEROBOT ==="
ls "$OUT_LEROBOT"
