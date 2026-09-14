#!/usr/bin/env bash
# (a)：15.0.0（`pi05_bowl5_full` の 005500）を出発点に、`libero_plus_bowl5_initpose` で学習する（RunPod）。
#
#   # 前提: runpod_setup.sh 済み・hf auth login 済み・データセットが HF に上がっている
#   export HF_HOME=/workspace/hf_home
#   hf download siruku6/pi05_code --local-dir /workspace/pi05_code
#   bash /workspace/pi05_code/solution/train/runpod_initpose_ft.sh
#
# **設定は対になる学習と揃える**（ユーザ・9/14）。対になるのは Blackwell の (b)——
# 素の `pi05_libero_base` から同じデータを `blackwell_bowl5.sh` で 5,500 歩回すもの。
# **違うのは出発点だけ**である。
#
# | 項目 | (a) この走り | (b) Blackwell |
# |---|---|---|
# | 出発点 | **HF `siruku6/pi05_bowl5_full` の `005500`** | `lerobot/pi05_libero_base` |
# | データ | `libero_plus_bowl5_initpose` | 同じ |
# | 範囲・学習率・warmup・batch・種・歩数 | 全部・2.5e-5・500・64・42・5,500 | 同じ |
#
# **`INIT_REPO=base` なら素の `lerobot/pi05_libero_base` から始める**（9/14 夜。Blackwell が空かないので、
# Blackwell で回す予定だった combined の全パラメータ学習をここで回すため）。版は `train_pi05.py` から読む。
#
#   RUN=pi05_combined_initpose_full INIT_REPO=base SAVE_FREQ=5500 \
#     DATA_REPO=siruku6/libero_combined_bowl5_initpose DATASET_ID=local/libero_combined_bowl5_initpose \
#     DS=/workspace/data/libero_combined_bowl5_initpose \
#     bash /workspace/pi05_code/solution/train/runpod_initpose_ft.sh
#
# **学習率は既定では頂点からやり直す。**15.0.0 は 2.5e-6 まで下げ切って終わっているので、
# その 10 倍から始まる（`train_pi05._resolve_init_from` の注意）。**揃えると決めたうえでそうしている。**
# **9/15 に `LR`・`WARMUP` を渡せるようにした。**続きの学習で頂点に戻したくないときに使う。
#
# **`EXPERT_ONLY=1` なら action expert だけを動かす**（9/15 に足した。既定 0 ＝全パラメータ）。
# 画像側と言語側を凍結し、20 歩の検算は 693,422,112 と突き合わせる。
#
#   RUN=pi05_combined_ae4k INIT_REPO=siruku6/pi05_combined_initpose_full INIT_STEP=005500 \
#     EXPERT_ONLY=1 LR=5e-6 STEPS=4000 SAVE_FREQ=4000 \
#     DATA_REPO=siruku6/libero_combined_bowl5_initpose DATASET_ID=local/libero_combined_bowl5_initpose \
#     DS=/workspace/data/libero_combined_bowl5_initpose \
#     bash /workspace/pi05_code/solution/train/runpod_initpose_ft.sh
#
# Reads:
#   - HF `$DATA_REPO`（手元で組んだデータセット）と、その中の `bowl5_expected.json`
#   - HF `$INIT_REPO` の `$INIT_STEP/`（出発点の重み）
# Generates:
#   - $OUTPUT_ROOT/$RUN.log / .nohup   学習のログ
#   - $W/upload_ckpt_$RUN.log          上げるループのログ（HF `$REPO` へ）
set -uo pipefail

W="${W:-/workspace}"
RUN="${RUN:-pi05_initpose_from_bowl5full}"
REPO="${REPO:-siruku6/$RUN}"
DATA_REPO="${DATA_REPO:-siruku6/libero_plus_bowl5_initpose}"
DATASET_ID="${DATASET_ID:-local/libero_plus_bowl5_initpose}"
DS="${DS:-$W/data/libero_plus_bowl5_initpose}"
INIT_REPO="${INIT_REPO:-siruku6/pi05_bowl5_full}"
INIT_STEP="${INIT_STEP:-005500}"
INIT_DIR="$W/init/$(basename "$INIT_REPO")"
BASE_REPO="lerobot/pi05_libero_base"
OUTPUT_ROOT="$W/out"
TOOLS="$W/tools"
CODE_DIR="${CODE_DIR:-$W/pi05_code/solution/train}"
STEPS="${STEPS:-5500}"
# **HF の非公開枠を食うのでチェックポイントは 2 個**（2,750・5,500）。保存の間隔は学習の中身を変えない。
SAVE_FREQ="${SAVE_FREQ:-2750}"
# **既定は対照と同じ 64。**96 GB の GPU なら `BATCH=128` も渡せる（対照とは別の作り方になる。howto/16 の 10 節）
BATCH="${BATCH:-64}"
LR="${LR:-2.5e-5}"
WARMUP="${WARMUP:-500}"
EXPERT_ONLY="${EXPERT_ONLY:-0}"
if [ "$EXPERT_ONLY" = 1 ]; then
    TRAIN_RANGE=(--freeze-vision-encoder true --train-expert-only true)
    WANT_PARAMS=693422112
    RANGE_NAME="action expert だけ"
else
    TRAIN_RANGE=(--freeze-vision-encoder false --train-expert-only false)
    WANT_PARAMS=4143404816
    RANGE_NAME="全パラメータ"
fi
export HF_HOME="${HF_HOME:-$W/hf_home}"

step() { echo; echo "=== $* ==="; }
die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }

step "前提"
command -v hf >/dev/null || die "hf が無い" "runpod_setup.sh を先に打つ"
hf auth whoami >/dev/null 2>&1 || die "HF にログインしていない" "hf auth login"
python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null || die "GPU が見えない"
python3 -c "import lerobot" 2>/dev/null || die "lerobot が import できない" "runpod_setup.sh を先に打つ"
[ -f "$CODE_DIR/train_pi05.py" ] || die "$CODE_DIR/train_pi05.py が無い" \
    "hf download siruku6/pi05_code --local-dir $W/pi05_code"
BASE_REV=$(sed -n 's/^_BASE_MODEL_REVISION = "\(.*\)"$/\1/p' "$CODE_DIR/train_pi05.py")
[ ${#BASE_REV} = 40 ] || die "出発点の版を train_pi05.py から読めない" "pi05_code を取り直す"
if [ "$INIT_REPO" = base ]; then START="$BASE_REPO@${BASE_REV:0:8}"; else START="$INIT_REPO/$INIT_STEP"; fi
echo "  出発点 $START"
[ -e "$OUTPUT_ROOT/$RUN" ] && die "出力先が既にある: $OUTPUT_ROOT/$RUN" "RUN を変えるか、消してから打つ"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  $W の空き: $(df -h "$W" | awk 'NR==2{print $4}')（要る量は約 80 GB）"
mkdir -p "$W/data" "$OUTPUT_ROOT" "$TOOLS" "$INIT_DIR"

step "降ろす（データと出発点を並列に）"
( hf download "$DATA_REPO" --repo-type dataset --local-dir "$DS" > "$W/dl_data.log" 2>&1; echo $? > "$W/dl_data.rc" ) &
if [ "$INIT_REPO" = base ]; then
    # **`--init-from` を渡さなければ `train_pi05.py` が同じ版を自分で引く**（ここで降ろしたキャッシュに当たる）
    ( hf download "$BASE_REPO" --revision "$BASE_REV" > "$W/dl_init.log" 2>&1; echo $? > "$W/dl_init.rc" ) &
else
    ( hf download "$INIT_REPO" --include "$INIT_STEP/*" --local-dir "$INIT_DIR" > "$W/dl_init.log" 2>&1; echo $? > "$W/dl_init.rc" ) &
fi
wait
[ "$(cat "$W/dl_data.rc")" = 0 ] || { tail -5 "$W/dl_data.log"; die "データを降ろせない" "$DATA_REPO が上がっているか確かめる"; }
[ "$(cat "$W/dl_init.rc")" = 0 ] || { tail -5 "$W/dl_init.log"; die "出発点を降ろせない" "$START があるか確かめる"; }
INIT_ARGS=()
if [ "$INIT_REPO" != base ]; then
    # **渡すのは重みの入った階層そのもの**（`config.json` と `model.safetensors` が直下にある所）。
    for f in config.json model.safetensors; do
        [ -f "$INIT_DIR/$INIT_STEP/$f" ] || die "$INIT_DIR/$INIT_STEP/$f が無い" "降ろした階層を ls で確かめる"
    done
    INIT_ARGS=(--init-from "$INIT_DIR/$INIT_STEP")
fi

step "降りたものを数える"
[ -f "$DS/bowl5_expected.json" ] || die "$DS/bowl5_expected.json が無い" "手元で組んだときの期待値が同梱された版を上げたか確かめる"
read -r GOT WANT <<< "$(python3 - "$DS" <<'PY'
import json, sys
i = json.load(open(f"{sys.argv[1]}/meta/info.json")); w = json.load(open(f"{sys.argv[1]}/bowl5_expected.json"))
g = [i["total_episodes"], i["total_frames"], i["total_tasks"]]
e = [w["total_episodes"], w["total_frames"], w["total_tasks"]]
print("_".join(map(str, g)), "_".join(map(str, e)))
PY
)"
echo "  実際 ${GOT//_/ } ／ 期待 ${WANT//_/ }"
[ "$GOT" = "$WANT" ] || die "降りたデータが期待と違う" "$DS を消して打ち直す"
WANT_EPS=${WANT%%_*}

step "上げる道具と上げ先"
for f in upload_ckpt.sh verify_upload.py; do cp "$CODE_DIR/$f" "$TOOLS/$f" || die "$f が無い"; done
chmod +x "$TOOLS/upload_ckpt.sh"
python3 -c "
from huggingface_hub import HfApi
HfApi().create_repo('$REPO', private=True, exist_ok=True); print('  上げ先 $REPO')" \
    || die "上げ先を作れない" "HF の非公開枠を確かめる"
NCKPT=$(( STEPS / SAVE_FREQ + (STEPS % SAVE_FREQ != 0) ))
echo "  チェックポイント $NCKPT 個（約 $((NCKPT * 9)) GB を HF へ）"

step "学習を起こす（出発点 $START・$RANGE_NAME・$STEPS 歩・学習率 $LR・warmup $WARMUP・batch $BATCH）"
NW=$(python3 -c "import os;print(min(os.cpu_count()//2, 16))")
cd "$CODE_DIR" || die "$CODE_DIR へ入れない"
setsid nohup python train_pi05.py --run-name "$RUN" --output-root "$OUTPUT_ROOT" \
    --dataset-repo "$DATASET_ID" --dataset-root "$DS" \
    ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} \
    --steps "$STEPS" --batch-size "$BATCH" --num-workers "$NW" \
    --learning-rate "$LR" --save-freq "$SAVE_FREQ" --seed 42 \
    "${TRAIN_RANGE[@]}" \
    -- "--policy.scheduler_warmup_steps=$WARMUP" \
       "--policy.scheduler_decay_steps=$STEPS" \
    > "$OUTPUT_ROOT/$RUN.nohup" 2>&1 &
sleep 5
pgrep -f "train_pi05.py --run-name $RUN" >/dev/null \
    || { tail -20 "$OUTPUT_ROOT/$RUN.nohup"; die "学習が立ち上がらない"; }

RUN="$RUN" REPO="$REPO" LOG="$W/upload_ckpt_$RUN.log" OUT="$OUTPUT_ROOT/$RUN" TRAIN_LOG_DIR="$OUTPUT_ROOT" VERIFY="$TOOLS/verify_upload.py" \
    setsid nohup "$TOOLS/upload_ckpt.sh" > "$W/upload_ckpt_$RUN.nohup" 2>&1 &

step "動き出しを確かめる（最初の 20 歩まで待つ）"
for _ in $(seq 1 240); do
    grep -q "step:20 " "$OUTPUT_ROOT/$RUN.log" 2>/dev/null && break
    pgrep -f "train_pi05.py --run-name $RUN" >/dev/null \
        || { tail -20 "$OUTPUT_ROOT/$RUN.log" 2>/dev/null; tail -20 "$OUTPUT_ROOT/$RUN.nohup"; die "学習が途中で落ちた"; }
    sleep 10
done
GOT_PARAMS=$(grep -m1 -o "num_learnable_params=[0-9]*" "$OUTPUT_ROOT/$RUN.log" | cut -d= -f2)
GOT_DSEPS=$(grep -m1 -o "dataset.num_episodes=[0-9]*" "$OUTPUT_ROOT/$RUN.log" | cut -d= -f2)
printf '  %-24s %-12s %s\n' 見たもの 実際 期待 num_learnable_params "$GOT_PARAMS" "$WANT_PARAMS" dataset.num_episodes "$GOT_DSEPS" "$WANT_EPS"
[ "$GOT_PARAMS" = "$WANT_PARAMS" ] || die "動かす範囲が「$RANGE_NAME」になっていない（$GOT_PARAMS）" "pkill -f 'train_pi05.py --run-name $RUN'; pkill -f 'lerobot[-_]train'"
[ "$GOT_DSEPS" = "$WANT_EPS" ] || die "読んでいるデータが違う（$GOT_DSEPS 本）" "pkill -f 'train_pi05.py --run-name $RUN'; pkill -f 'lerobot[-_]train'"
[ "$INIT_REPO" = base ] || grep -q "初期重みにローカルのチェックポイントを使う" "$OUTPUT_ROOT/$RUN.nohup" "$OUTPUT_ROOT/$RUN.log" 2>/dev/null \
    || die "出発点が $START になっていない（--init-from の行がログに無い）" "pkill -f 'train_pi05.py --run-name $RUN'; pkill -f 'lerobot[-_]train'"
grep -q "開始 RUN=$RUN" "$W/upload_ckpt_$RUN.log" 2>/dev/null || echo "  **上げるループの開始行がまだ無い。**tail $W/upload_ckpt_$RUN.nohup を見る"

UPDT=$(grep -m1 -o "updt_s:[0-9.]*" "$OUTPUT_ROOT/$RUN.log" | cut -d: -f2)
MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)
echo "  GPU メモリ（20 歩の時点）: $MEM（batch $BATCH）"
grep -m1 -oE "mem_gb:[0-9.]*" "$OUTPUT_ROOT/$RUN.log" | sed 's/^/  ログの /'
echo
echo "=== 走っている ==="
[ -n "$UPDT" ] && python3 -c "s = $UPDT * $STEPS; print(f'  終わる目安: {int(s//3600)} 時間 {int(s%3600//60)} 分（1 歩 $UPDT 秒 × $STEPS 歩）')"
echo "  ログ: tail -f $OUTPUT_ROOT/$RUN.log"
echo "  上げ: tail -f $W/upload_ckpt_$RUN.log"
