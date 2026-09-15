#!/usr/bin/env bash
# 学習と「上げるループ」を起こし、**動き出したことを確かめてから**離席できる形にする。
#
#   RUN=pi05_proto21 EXPERT_ONLY=1 bash blackwell_launch.sh
#
# 先に blackwell_setup.sh を通しておくこと（`~/run_env.sh` を読む）。
#
# **2026-09-07 に踏んだ穴を潰してある。**
#
# | 踏んだ穴 | ここでどう潰すか |
# |---|---|
# | ターミナルが落ちて学習ごと消える | `setsid nohup` で別セッションにする |
# | `setsid` の `$!` は本体の PID ではない | `pgrep` で本体を探して出す |
# | 上げるループの版が古く `pi05_camdrop2500` を見ていた | `RUN`/`REPO` を渡して起こし、ログの書式で新しい版かを判定する |
# | `verify_upload.py` が無くて印を付けられない | 起こす前に在ることを確かめる |
# | 出力先が既にあると LeRobot が落ちる | 先に見て、原因を出して止まる |
# | 20 歩の行が出るまで待たずに離席した | `dataset.num_episodes` と最初の loss まで待って表示する |
#
# Generates:
#   - $OUTPUT_ROOT/$RUN.log / .nohup   学習のログ
#   - ~/upload_ckpt.log                上げるループのログ
set -uo pipefail

[ -f "$HOME/run_env.sh" ] || { echo "**~/run_env.sh が無い。先に blackwell_setup.sh を打つ**"; exit 1; }
. "$HOME/run_env.sh"

RUN="${RUN:?RUN を渡すこと（例 RUN=pi05_proto21）}"
REPO="${REPO:-siruku6/$RUN}"
STEPS="${STEPS:-2000}"
BATCH="${BATCH:-64}"
LR="${LR:-5e-6}"
SAVE_FREQ="${SAVE_FREQ:-500}"
EXPERT_ONLY="${EXPERT_ONLY:-0}"       # 1 なら action expert だけ（視覚と VLM を固定）
DATASET_REPO_ID="${DATASET_REPO_ID:-local/t3_proto}"

die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }

echo "=== 打つ条件 ==="
printf '  %-16s %s\n' RUN "$RUN" REPO "$REPO" 歩数 "$STEPS" batch "$BATCH" 学習率 "$LR" \
    データ "$MERGED" 出発点 "$WEIGHTS"
if [ "$EXPERT_ONLY" = 1 ]; then
    FREEZE=(--freeze-vision-encoder true --train-expert-only true)
    printf '  %-16s %s\n' 動かす範囲 "action expert だけ（693M）"
    WANT_PARAMS=693422112
else
    FREEZE=(--freeze-vision-encoder false --train-expert-only false)
    printf '  %-16s %s\n' 動かす範囲 "全部（4.14B）"
    WANT_PARAMS=4143404816
fi

echo "=== 前提の確認 ==="
[ -d "$MERGED/meta" ]      || die "$MERGED が無い" "blackwell_setup.sh を打つ"
[ -f "$MERGED/meta/stats.json.bak" ] || die "正規化が据え置かれていない" \
    "check_merge_stats.py freeze --merged $MERGED --source $DATA を打つ"
[ -d "$WEIGHTS/vlm" ]      || die "$WEIGHTS に vlm/ が無い" "blackwell_setup.sh を打つ"
[ -f "$TOOLS/upload_ckpt.sh" ]   || die "$TOOLS/upload_ckpt.sh が無い" "blackwell_setup.sh を打つ"
[ -f "$TOOLS/verify_upload.py" ] || die "$TOOLS/verify_upload.py が無い" \
    "これが無いと上げた印を付けられず、毎巡 9 GB を上げ直す"
[ -e "$OUTPUT_ROOT/$RUN" ] && die "出力先が既にある: $OUTPUT_ROOT/$RUN" \
    "別の RUN 名にするか、消してから打つ"
grep -q "dataset-repo" "$CODE_DIR/train_pi05.py" || die "train_pi05.py が古い" \
    "hf download siruku6/pi05_code --local-dir ~/pi05_code を打ち直す"
# **チェックポイントを作りすぎない歯止め。**1 個 9.35 GB を上げるので、
# 20,000 歩 × save-freq 500 だと 40 個 ＝ 374 GB になる（9/8 に気付いた）
NCKPT=$(( STEPS / SAVE_FREQ ))
[ "$NCKPT" -gt 12 ] && die "チェックポイントが $NCKPT 個になる（1 個 9.35 GB ＝ 約 $((NCKPT * 9)) GB）" \
    "SAVE_FREQ を大きくする。目安は $STEPS 歩なら SAVE_FREQ=$(( (STEPS + 7) / 8 ))" \
    "どうしても要るなら SAVE_FREQ はそのままで、この行の 12 を上げる"
echo "  チェックポイント $NCKPT 個（約 $((NCKPT * 9)) GB を上げる）"
EPS=$(python3 -c "import json;print(json.load(open('$MERGED/meta/info.json'))['total_episodes'])")
echo "  データ $EPS 本・据え置き済み・重みと道具は揃っている"

echo "=== 学習を起こす ==="
NW=$(python -c "import os;print(min(os.cpu_count()//2, 16))")
cd "$CODE_DIR" || die "$CODE_DIR へ入れない" "~/pi05_code の階層は solution/train である"
setsid nohup python train_pi05.py --run-name "$RUN" \
    --init-from "$WEIGHTS" \
    --dataset-repo "$DATASET_REPO_ID" --dataset-root "$MERGED" \
    --steps "$STEPS" --batch-size "$BATCH" --num-workers "$NW" \
    --learning-rate "$LR" --save-freq "$SAVE_FREQ" \
    "${FREEZE[@]}" --camera-viewpoint-augment \
    > "$OUTPUT_ROOT/$RUN.nohup" 2>&1 &
sleep 5
PID=$(pgrep -f "train_pi05.py --run-name $RUN" | head -1)
[ -n "$PID" ] || { tail -20 "$OUTPUT_ROOT/$RUN.nohup"; die "学習が立ち上がらない" \
    "上の nohup の末尾を読む"; }
echo "  PID $PID（setsid なので端末が落ちても生きる）"

echo "=== 上げるループを起こす ==="
# **走りごとに別のログにする。**共通のログだと、前の走りの行を見て
# 「掴んだ」と誤判定する（9/7 に古い版のログを読んで誤判定した）
ULOG="$HOME/upload_ckpt_$RUN.log"
OLD=$(pgrep -af "upload_ckpt.sh" | head -1)
[ -n "$OLD" ] && die "上げるループが既に動いている: $OLD" \
    "別の走りのものなら pkill -f upload_ckpt.sh してから打つ"
# **出力とログの在り処はここで渡す。**学習もログもここが置いたので、上げる側に推測させない。
RUN="$RUN" REPO="$REPO" LOG="$ULOG" OUT="$OUTPUT_ROOT/$RUN" TRAIN_LOG_DIR="$OUTPUT_ROOT" \
    setsid nohup "$TOOLS/upload_ckpt.sh" > "$HOME/upload_ckpt_$RUN.nohup" 2>&1 &
sleep 25
grep -q "学習を見つけた" "$ULOG" 2>/dev/null \
    || { tail -5 "$ULOG" 2>/dev/null; tail -5 "$HOME/upload_ckpt_$RUN.nohup";
         die "上げるループが学習を掴めていない" \
             "ログに 見回り 未上げ N 個 の書式が無ければ古い版が動いている" \
             "pkill -f upload_ckpt.sh してから打ち直す"; }
echo "  掴んだ（ログ $ULOG）"

echo "=== 動き出しを確かめる（最初の 20 歩まで待つ・数分）==="
for _ in $(seq 1 120); do
    grep -q "step:20 \|step:2K \|step:1K " "$OUTPUT_ROOT/$RUN.log" 2>/dev/null && break
    pgrep -f "train_pi05.py --run-name $RUN" >/dev/null \
        || { tail -20 "$OUTPUT_ROOT/$RUN.log"; die "学習が途中で落ちた" "上のログを読む"; }
    sleep 10
done

GOT_PARAMS=$(grep -m1 -o "num_learnable_params=[0-9]*" "$OUTPUT_ROOT/$RUN.log" | cut -d= -f2)
GOT_EPS=$(grep -m1 -o "dataset.num_episodes=[0-9]*" "$OUTPUT_ROOT/$RUN.log" | cut -d= -f2)
FIRST=$(grep -m1 -o "loss:[0-9.]* grdn:[0-9.]*" "$OUTPUT_ROOT/$RUN.log")
echo
printf '  %-26s %-14s %s\n' 見たもの 実際 期待
printf '  %-26s %-14s %s\n' num_learnable_params "$GOT_PARAMS" "$WANT_PARAMS"
printf '  %-26s %-14s %s\n' dataset.num_episodes "$GOT_EPS" "$EPS"
printf '  %-26s %s\n' "最初の 20 歩" "$FIRST"
[ "$GOT_PARAMS" = "$WANT_PARAMS" ] || die "動かす範囲が違う（$GOT_PARAMS）" \
    "EXPERT_ONLY の指定と食い違っている。止めて打ち直す" \
    "pkill -f 'train_pi05.py --run-name $RUN'; pkill -f 'lerobot[-_]train'"
[ "$GOT_EPS" = "$EPS" ] || die "読んでいるデータが違う（$GOT_EPS 本）" \
    "--dataset-root が効いていない。止めて打ち直す"

echo
echo "=== 走っている。離席してよい ==="
# **見積もりはログの updt_s から出す。**推測値を書かない
UPDT=$(grep -m1 -o "updt_s:[0-9.]*" "$OUTPUT_ROOT/$RUN.log" | cut -d: -f2)
[ -n "$UPDT" ] && echo "  終わる目安: $(python3 -c "
s = $UPDT * $STEPS
print(f'{int(s//3600)} 時間 {int(s%3600//60)} 分（1 歩 $UPDT 秒 × $STEPS 歩・実測から）')")"
echo "  ログ:   tail -f $OUTPUT_ROOT/$RUN.log"
echo "  上げ:   tail -f $ULOG"
echo "  監視セル（B6）は howto/14 の版を貼る。RUN = \"$RUN\" の 1 行だけ直す"
