#!/usr/bin/env bash
# 摂動下再生を、**初期姿勢の番号を増やして**作り、LeRobot 形式まで持っていく。
#
# **なぜ番号を増やすのか。** 9/11 に作った 1,009 本は **12 番号だけ**で出来ている
# （1 番号あたり 41〜107 本）。ところが採点で崩れるかどうかを決めるのは
# **ずれの大きさではなく向き**である（`docs/reference/21_weakness_map.md` の 2 節——
# 同じ課題で 3.24 cm が 3/3 成功し、2.97 cm が 0/3）。**大きさは 12 個で覆えているが、
# 向きは 12 通りしか入っていない。**採点は 450 番号から引かれる。
# **だから同じ総数のまま、番号を増やして 1 番号あたりを薄くする。**
#
# **npz を全部溜めてから変換すると入らない。** 1 本 60〜97 MB で、1,000 本で
# 60〜97 GB になる。**小分けに作って変換し、変換できたぶんの npz をその場で捨てる**
# （`run_scripted_bowls.sh` と同じ手）。
#
# **変換は次の塊の生成と重ねる。**変換は 1 プロセスで 1 本 9〜14 秒かかり、
# 生成（14 並列）の後ろに直列で置くと全体の半分を占めていた（9/12 の実測で 6.5 時間中 3.3 時間）。
# **重ねると npz が 2 塊ぶん同時にディスクに載る**ので、`PER_CHUNK` はそれを見て決める。
#
# **変換では指示文を配布データの書き方へ寄せる**（`--normalize-instructions`）。
# 寄せずに足すと評価で一度も引かれない（`docs/knowledge/23_…` ——
# 9/11 に作った 1,009 本は 62% がこれで死んでいた）。
#
# 使い方
#   nohup bash solution/tools/regen/run_init_pose_batches.sh \
#       > work/regen/logs/init_pose_all.log 2>&1 &
#
# 環境変数（省略時の既定）
#   INITSTATES    使う番号の JSON（`{"initstates": [...]}`）
#   OLD_MANIFEST  既に撃った回の manifest。**そこに出てくる番号と組は撃たない**
#   SKIP_OLD=1    0 にすると、`OLD_MANIFEST` に出てくる番号も撃つ（組は飛ばす）
#   PER_NUMBER    1 番号あたりの目標本数
#   PER_CHUNK     1 回に回す番号の個数。**npz は PER_NUMBER × PER_CHUNK × 60〜97 MB が 2 塊ぶん**
#   WORKERS       生成の並列数（**変換に 1 コア残す**）
#   KEEP_NPZ=0    1 にすると npz を消さない（デバッグ用）
#   APPEARANCE    外観を重ねる配り方（`appearance.py`）。既定 `off`＝素の見た目だけ。
#                 033 の 4,000 本は `plain=25,light=25,background=25,camera=25`
#   QUOTA         課題ごとの本数の JSON（`plan_replacement.py --quota-out`）。渡すと
#                 PER_NUMBER の代わりに、**塊ごとに「番号を消化した割合」ぶんまで**課題を埋める
#   PER_DEMO_CAP  同じ (課題, デモ) を何本まで採るか（既定 0＝上限なし）
#   FALLBACK_PREALIGN=1  open_loop で落ちたデモを prealign（PREALIGN_PLANNER・既定 glide）で撃ち直す
#   CONVERT_PARALLEL  同時に走らせる変換の数（既定 1）。**npz が (この数 + 1) 塊ぶんディスクに載る。**
#                 変換は 1 本 13.5 秒で、3,460 本を 1 本ずつだと 13 時間かかり生成より遅い
#
# Reads:
#   - $INITSTATES・work/artifacts/libero_states・$OLD_MANIFEST
# Generates:
#   - $OUT_LEROBOT/c<NN>/ … 塊ごとの LeRobot データセット
#   - $OUT_NPZ/manifest_all.jsonl … 撃った全回の記録（塊を越えて残る）
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPOSITORY_ROOT"
INITSTATES="${INITSTATES:-work/regen/initstates_48.json}"
OUT_NPZ="${OUT_NPZ:-work/regen/npz_init_pose}"
OUT_LEROBOT="${OUT_LEROBOT:-work/regen/lerobot_init_pose}"
STATES="work/artifacts/libero_states"
OLD_MANIFEST="${OLD_MANIFEST:-work/regen/npz_batch1/manifest.jsonl}"
SKIP_OLD="${SKIP_OLD:-1}"
PER_NUMBER="${PER_NUMBER:-25}"
PER_CHUNK="${PER_CHUNK:-10}"
WORKERS="${WORKERS:-13}"
KEEP_NPZ="${KEEP_NPZ:-0}"
APPEARANCE="${APPEARANCE:-off}"
QUOTA="${QUOTA:-}"
PER_DEMO_CAP="${PER_DEMO_CAP:-0}"
FALLBACK_PREALIGN="${FALLBACK_PREALIGN:-0}"
PREALIGN_PLANNER="${PREALIGN_PLANNER:-glide}"
CONVERT_PARALLEL="${CONVERT_PARALLEL:-1}"
mkdir -p "$OUT_NPZ" "$OUT_LEROBOT" work/regen/logs

# 撃つ番号を並べる。**既に撃った番号は `OLD_MANIFEST` から読む**——
# 生成器の既定値を書き写すと、既定が変わったときに黙って食い違う。
mapfile -t NUMBERS < <(python3 - "$INITSTATES" "$SKIP_OLD" "$OLD_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
picked = json.load(open(sys.argv[1]))["initstates"]
old = set()
if sys.argv[2] == "1" and Path(sys.argv[3]).exists():
    old = {json.loads(line)["initstate"] for line in open(sys.argv[3]) if line.strip()}
for n in picked:
    if n not in old:
        print(n)
PY
)
TOTAL=$(( ${#NUMBERS[@]} * PER_NUMBER ))
echo "=== 撃つ番号 ${#NUMBERS[@]} 個 × $PER_NUMBER 本 = $TOTAL 本 ==="
echo "    ${NUMBERS[*]}"

# 飛ばす組は「古い manifest ＋ これまでの塊」の足し算である。
SKIP_FILE="$OUT_NPZ/skip.jsonl"
: > "$SKIP_FILE"
[ -f "$OLD_MANIFEST" ] && cat "$OLD_MANIFEST" >> "$SKIP_FILE"
[ -f "$OUT_NPZ/manifest_all.jsonl" ] && cat "$OUT_NPZ/manifest_all.jsonl" >> "$SKIP_FILE"

convert_chunk() {           # $1 = 塊の名前。**背面で走る**
    local name="$1"
    IMAGE=parc2026-train-pi05 CHOWN_PATH="$REPOSITORY_ROOT/$OUT_LEROBOT" \
        bash solution/tools/regen/run_in_container.sh \
        solution/tools/scripted_demo/npz_to_lerobot.py build \
        --npz-dir "$OUT_NPZ/$name" --output "$OUT_LEROBOT/$name" \
        --repo-id "local/init_pose_${name}" --normalize-instructions \
        > "work/regen/logs/init_pose_build_${name}.log" 2>&1
    local status=$?
    local made
    made=$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(f\"{d['total_episodes']} 本 {d['total_frames']} コマ {d['total_tasks']} 課題\")
" "$OUT_LEROBOT/$name/meta/info.json" 2>/dev/null || echo "info.json が無い")
    [ "$KEEP_NPZ" = "0" ] && rm -rf "${OUT_NPZ:?}/$name"
    echo "=== [$name] 変換 exit=$status / $made / $(date -Is) ==="
}

chunk=0
for start in $(seq 0 "$PER_CHUNK" $(( ${#NUMBERS[@]} - 1 )) ); do
    chunk=$(( chunk + 1 ))
    name=$(printf "c%02d" "$chunk")
    slice=("${NUMBERS[@]:start:PER_CHUNK}")
    target=$(( ${#slice[@]} * PER_NUMBER ))
    EXTRA=(--per-demo-cap "$PER_DEMO_CAP" --prealign-planner "$PREALIGN_PLANNER")
    [ "$FALLBACK_PREALIGN" = "1" ] && EXTRA+=(--fallback-prealign)
    if [ -n "$QUOTA" ]; then
        # **累計の割合で渡す。**前の塊で足りなかった課題の不足は次の塊へ持ち越される。
        fraction=$(python3 -c "print(($start + ${#slice[@]}) / ${#NUMBERS[@]})")
        EXTRA+=(--quota "$QUOTA" --quota-fraction "$fraction")
        target=1000000
    fi
    dir="$OUT_NPZ/$name"
    echo ""
    echo "=== [$name] 生成 番号 ${slice[*]} / 目標 $target 本 / $(date -Is) ==="

    CHOWN_PATH="$REPOSITORY_ROOT/$OUT_NPZ" bash solution/tools/regen/run_in_container.sh \
        solution/tools/regen/generate_perturbed.py \
        --states "$STATES" --out-npz "$dir" --target "$target" \
        --initstates "${slice[@]}" --workers "$WORKERS" --skip "$SKIP_FILE" \
        --appearance "$APPEARANCE" "${EXTRA[@]}" \
        > "work/regen/logs/init_pose_gen_${name}.log" 2>&1
    echo "  生成 exit=$? / $(ls "$dir"/*.npz 2>/dev/null | wc -l) 本 / $(du -sh "$dir" 2>/dev/null | cut -f1)"

    # **manifest は塊を越えて残す。**npz を消すと一緒に消えてしまうので先に写す。
    if [ -f "$dir/manifest.jsonl" ]; then
        tee -a "$OUT_NPZ/manifest_all.jsonl" < "$dir/manifest.jsonl" >> "$SKIP_FILE"
    fi

    # 変換は同時に CONVERT_PARALLEL 本まで。空きが出るのを待ってから背面で始める。
    while [ "$(jobs -rp | wc -l)" -ge "$CONVERT_PARALLEL" ]; do wait -n; done
    convert_chunk "$name" &
done
wait

echo ""
echo "=== 出来たもの ==="
ls "$OUT_LEROBOT"
date -Is
