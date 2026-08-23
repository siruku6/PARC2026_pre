#!/usr/bin/env bash
# ロボット初期姿勢（Robot Initial States）の摂動下で、学習データを撮る。
#
# **GPU は要らない。** 物理と描画だけなので手元の Mac の Docker でも回る
# （描画はどちらのマシンでも CPU＝osmesa。GPU マシンの vGPU は OpenGL 非対応）。
#
#     cd <リポジトリ>
#     PARALLEL=6 bash solution/tools/scripted_demo/record_tier2_robot_init.sh
#
# **`PARALLEL` はコア数を超えないこと。** 1 本が 1.25 GB 使うので、メモリからも
# 上限が決まる（2 コア 9.69 GB で 8 並列にして OOM で 1 本消えた前例がある）。
#     docker run --rm parc2026 nproc                  # 使えるコア数
#     docker info --format 'CPUs={{.NCPU}}'
#
# 撮り終わったら、**同じ Mac で LeRobot 形式へ変換してから運ぶ**。
# npz のままだと 26 GB、変換後は約 390 MB で 67 分の 1 になる。
#     docker run --rm -v "$PWD/solution/tools:/host/tools:ro" \
#         -v "$PWD/work/scripted:/data" \
#         parc2026-train python /host/tools/scripted_demo/npz_to_lerobot.py build \
#         --npz-dir /data/tier2_robot_init --output /data/tier2_lerobot \
#         --repo-id local/tier2_robot_init
set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"

PARALLEL="${PARALLEL:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/out/tier2_robot_init}"

# **妨害の設定は測って決めた**（2026-08-23、`verify --disturb-report`）。
#   大きさ 1.0 × 10 歩、開始は 0〜74 歩から本ごとに引く
#   → ずれ 20.1cm・戻り 1.2cm・10 回のうち 7 回が残る
# 40〜140 歩にすると、精度が 2.31cm 要る「コンロの火力つまみを掴む位置へ」に
# 当たって 10 回のうち 4 回しか残らない。**この範囲を広げないこと。**
DISTURB_SCALE="${DISTURB_SCALE:-1.0}"
DISTURB_STEPS="${DISTURB_STEPS:-10}"
DISTURB_AT="${DISTURB_AT:-0,74}"

# **初期配置 0〜8 は妨害なし、9〜27 は妨害あり。** 妨害なしが 3 分の 1 になる
# 割り当てで、残る見込みは 9×0.9 + 19×0.7 ≒ 21 本／変種。
SMOOTH_EPISODES="${SMOOTH_EPISODES:-0,1,2,3,4,5,6,7,8}"
DISTURB_EPISODES="${DISTURB_EPISODES:-9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}"

# **学習用の変種。検証用とは排他にしてある**（08 の 2.8 節：分けないと
# 「未知ドメインへの汎化」ではなく「見た条件を再現できたか」を測ることになる）。
# **検証用に SCENE3 は 38 変種・SCENE8 は 28 変種残る。**
#
# **選び方を 2026-08-23 に差し替えた。** 最初は難易度ラベル
# （`benchmark/task_classification.json` の `difficulty_level`）がばらけるように
# 選んだが、**そのラベルは台本が通るかどうかを予測しない**と実測で分かった
# （難易度 1 の initstate_113 が 9 回中 1 回しか成功せず、難易度 5 の
# initstate_139 が 9 回中 6 回成功した）。
# いまの 12 変種は、**用意されている 78 変種すべてを妨害なしで 3 回ずつ試し、
# 3 回とも成功した 30 変種**（タスクごとに 15）から、難易度がばらけるように
# 選んだものである（12_tier2_recording_spec.md の 4 節に実測値）。
SCENE3="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
SCENE8="KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
SCENE3_VARIANTS="13 83 3 133 123 433"
SCENE8_VARIANTS="109 169 159 289 419 139"

jobs_file="$(mktemp)"
for pair in "$SCENE3:$SCENE3_VARIANTS" "$SCENE8:$SCENE8_VARIANTS"; do
    task="${pair%%:*}"
    for number in ${pair#*:}; do
        variant="${task}_view_0_0_100_0_0_initstate_${number}"
        printf '%s\t%s\tsmooth\n'  "$task" "$variant" >> "$jobs_file"
        printf '%s\t%s\tdisturb\n' "$task" "$variant" >> "$jobs_file"
    done
done
echo "撮る組み合わせ $(wc -l < "$jobs_file") 件（変種 12 × 妨害あり／なし）"

run_one() {
    local task="$1" variant="$2" mode="$3"
    local flags=(--episodes "$SMOOTH_EPISODES")
    [ "$mode" = disturb ] && flags=(
        --episodes "$DISTURB_EPISODES"
        --disturb-scale "$DISTURB_SCALE"
        --disturb-steps "$DISTURB_STEPS"
        --disturb-at "$DISTURB_AT")
    echo "===== $mode  $variant ====="
    docker run --rm \
        -e MUJOCO_GL=osmesa -e LP_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
        -v "$PWD/solution/tools:/host/tools:ro" \
        -v "$PWD/work/scripted:/out" \
        parc2026 python /host/tools/scripted_demo/scripted_demo.py record \
        --task "$task" --variant "$variant" \
        --output "$OUTPUT_DIR" "${flags[@]}"
}
export -f run_one
export DISTURB_SCALE DISTURB_STEPS DISTURB_AT SMOOTH_EPISODES DISTURB_EPISODES OUTPUT_DIR

mkdir -p work/scripted/tier2_robot_init
# **`LP_NUM_THREADS=1` と `OMP_NUM_THREADS=1` が要る。** osmesa は勝手に複数
# スレッドを使うので、指定しないと並列の意味が薄れる。
tr '\t' '\n' < "$jobs_file" | xargs -n 3 -P "$PARALLEL" \
    bash -c 'run_one "$0" "$1" "$2"'
rm -f "$jobs_file"

echo ""
echo "撮れた本数: $(ls work/scripted/tier2_robot_init/*.npz 2>/dev/null | wc -l)"
echo "  うち妨害あり: $(ls work/scripted/tier2_robot_init/*_d1*.npz 2>/dev/null | wc -l)"
echo "**Linux では root で書かれる。持ち主を戻す:**"
echo "  docker run --rm -v \"\$PWD/work/scripted:/out\" --entrypoint chown parc2026 -R \"\$(id -u):\$(id -g)\" /out"
