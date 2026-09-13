#!/usr/bin/env bash
# 丼 5 課題を入れ替えたデータで、Blackwell で**全パラメータ学習**を 1 本回す。
# 準備から「離席してよい」までを 1 本にする。**30 分で離席できることを狙う。**
#
#   . ~/env_pi05.sh
#   bash ~/pi05_code/solution/train/blackwell_bowl5.sh          # 歩数は残り時間から決まる
#   STEPS=5000 bash ~/pi05_code/solution/train/blackwell_bowl5.sh   # 歩数を指定する
#   DRY=1 bash .../blackwell_bowl5.sh                           # 降ろすところまでで止める
#
# **`blackwell_setup.sh` は打たない。**あちらは Blackwell の上でデータセットを
# 組む（配布 tar の展開 → 3 桁 symlink → N 倍 → align → merge → 検算）ので
# 2 時間かかる。**この走りのデータは手元で組んで HF に置いてあるので、
# 降ろすだけで済む。**それが 2 時間を 30 分にする唯一の要点である。
#
# **`blackwell_launch.sh` とは別にしてある。**あちらは逆向きデモの走り専用で、
# `--camera-viewpoint-augment` と `--init-from` が埋め込まれている。
# **この走りはどちらも使わない**（対照の `pi05_full_local` が使っていないため）ので、
# あちらを直すと別のセッションの走りが変わってしまう。
#
# | この走りが対照と違うところ | |
# |---|---|
# | データ | 丼 5 課題の 1,880 本を、作り直した手本と入れ替えた |
# | 動かす範囲 | **全部（4.14B）。**対照と JAIST 側は action expert だけ（693M） |
# | batch | **64。**JAIST 側は 16（20 GB に載らないため） |
#
# Reads:
#   - HF `$DATASET_REPO`（手元で組んだデータセット）と `bowl5_expected.json`
#   - HF `lerobot/pi05_libero_base`（出発点。`train_pi05.py` が自分で降ろす）
# Generates:
#   - $OUTPUT_ROOT/$RUN.log / .nohup   学習のログ
#   - ~/upload_ckpt_$RUN.log           上げるループのログ
#   - ~/run_env.sh                     ターミナルが落ちたとき用
set -uo pipefail

RUN="${RUN:-pi05_bowl5_full}"
REPO="${REPO:-siruku6/$RUN}"
DATASET_REPO="${DATASET_REPO:-siruku6/libero_plus_bowl5}"
DATASET_ID="${DATASET_ID:-local/libero_plus_bowl5}"   # --dataset-repo に渡す名前（中身は --dataset-root が決める）
SKIP_DL_DATASET="${SKIP_DL_DATASET:-0}"   # 1 なら $DS を降ろさずそのまま使う（blackwell_initpose.sh が組んだもの）
BASE_REPO="${BASE_REPO:-lerobot/pi05_libero_base}"
# **版は `train_pi05.py` から読む。**ここに書き写すと、あちらが上げたときに
# 黙って別の重みから始まる。
BASE_REV="${BASE_REV:-}"
NVME="${NVME:-/opt/dlami/nvme}"
DS="${DS:-$NVME/data/libero_plus_bowl5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$NVME/out}"
TOOLS="${TOOLS:-$HOME/tools}"
CODE_DIR="${CODE_DIR:-$HOME/pi05_code/solution/train}"
BATCH="${BATCH:-64}"
LR="${LR:-2.5e-5}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
WARMUP="${WARMUP:-500}"
RESERVE_MIN="${RESERVE_MIN:-60}"       # 後片付け（最後の送出・ループ停止・自分で止める）
SEC_PER_STEP="${SEC_PER_STEP:-6.85}"   # reference/27 の 2 節の実測（全パラ・batch 64）
DRY="${DRY:-0}"

die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }
step() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------- 前提
step "前提の確認"
command -v hf >/dev/null || die "hf が無い" ". ~/env_pi05.sh を先に読む"
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN が空" \
    'export HF_TOKEN="$(cat ~/.hf_token)" を打つ'
[ -f "$CODE_DIR/train_pi05.py" ] || die "$CODE_DIR/train_pi05.py が無い" \
    "hf download siruku6/pi05_code --local-dir ~/pi05_code を打つ"
[ -n "$BASE_REV" ] || BASE_REV=$(sed -n 's/^_BASE_MODEL_REVISION = "\(.*\)"$/\1/p' \
    "$CODE_DIR/train_pi05.py")
[ ${#BASE_REV} = 40 ] || die "出発点の版を train_pi05.py から読めない（'$BASE_REV'）" \
    "pi05_code を取り直すか、BASE_REV=<40 桁> を渡す"
echo "  出発点 $BASE_REPO@${BASE_REV:0:8}"
[ -e "$OUTPUT_ROOT/$RUN" ] && die "出力先が既にある: $OUTPUT_ROOT/$RUN" \
    "別の RUN 名にするか、消してから打つ"
# **DRY のときは見ない。**降ろすところまでの空打ちは GPU の無い機械でもやりたい
# （Blackwell の 12 時間を使わずに、取得と検算だけを別の機械で通すため）。
if [ "$DRY" != 1 ]; then
    python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null \
        || die "GPU が見えない" "nvidia-smi を打って確かめる"
fi
mkdir -p "$NVME/data" "$OUTPUT_ROOT" "$TOOLS"
echo "  ok"

# ---------------------------------------------------------------- 降ろす
# **2 つを並列に降ろす。**互いに独立なので、直列にすると待ち時間が足し算になる。
step "降ろす（データ約 15 GB ＋ 出発点 8.8 GB を並列に）"
T_DL=$(date +%s)
if [ "$SKIP_DL_DATASET" = 1 ]; then
    echo "  データは降ろさない（SKIP_DL_DATASET=1。$DS をそのまま使う）"
    ( echo 0 > "$HOME/dl_dataset.rc" ) &
else
    ( hf download "$DATASET_REPO" --repo-type dataset --local-dir "$DS" \
        > "$HOME/dl_dataset.log" 2>&1 ; echo $? > "$HOME/dl_dataset.rc" ) &
fi
PID_DS=$!
( hf download "$BASE_REPO" --revision "$BASE_REV" \
    > "$HOME/dl_base.log" 2>&1 ; echo $? > "$HOME/dl_base.rc" ) &
PID_BASE=$!
wait "$PID_DS" "$PID_BASE"
[ "$(cat "$HOME/dl_dataset.rc")" = 0 ] || { tail -5 "$HOME/dl_dataset.log"; die "データを降ろせない"; }
[ "$(cat "$HOME/dl_base.rc")"    = 0 ] || { tail -5 "$HOME/dl_base.log";    die "出発点を降ろせない"; }
echo "  $(( ($(date +%s) - T_DL) / 60 )) 分 $(( ($(date +%s) - T_DL) % 60 )) 秒"

# ---------------------------------------------------------------- 検算
# **本数・コマ数・課題数の 3 つを突き合わせる。**1 つでも違えば止める。
# 期待値は手元でデータを組んだときに書き出したもので、転記していない。
step "降りたものを数える"
EXPECT="$DS/bowl5_expected.json"
[ -f "$EXPECT" ] || die "$EXPECT が無い" \
    "手元で組んだときの期待値が同梱されていない。データセットの版が古い"
read -r GOT_EPS GOT_FRAMES GOT_TASKS WANT_EPS WANT_FRAMES WANT_TASKS <<< "$(python3 - "$DS" <<'PY'
import json, sys
root = sys.argv[1]
info = json.load(open(f"{root}/meta/info.json"))
want = json.load(open(f"{root}/bowl5_expected.json"))
print(info["total_episodes"], info["total_frames"], info["total_tasks"],
      want["total_episodes"], want["total_frames"], want["total_tasks"])
PY
)"
printf '  %-14s %-12s %s\n' 見たもの 実際 期待
printf '  %-14s %-12s %s\n' 本数 "$GOT_EPS" "$WANT_EPS" \
    コマ数 "$GOT_FRAMES" "$WANT_FRAMES" 課題数 "$GOT_TASKS" "$WANT_TASKS"
[ "$GOT_EPS" = "$WANT_EPS" ] && [ "$GOT_FRAMES" = "$WANT_FRAMES" ] \
    && [ "$GOT_TASKS" = "$WANT_TASKS" ] \
    || die "降りたデータが期待と違う" "降ろし直す（$DS を消してから打ち直す）"

# ---------------------------------------------------------------- 歩数
# **歩数は「いま何時か」から決める。**準備に何分かかったかは毎回違うので、
# 先に決めておくと、余らせるか終わらないかのどちらかになる。
step "歩数を決める"
BOOT=$(date -u -d "$(uptime -s)" +%s)
if [ -n "${T0:-}" ]; then
    BOOT=$(date -u -d "$(date -u +%F) ${T0}" +%s)
    echo "  T0 を渡された: $(date -u -d "@$BOOT" '+%F %H:%M UTC')"
else
    echo "  起動時刻（uptime -s）: $(date -u -d "@$BOOT" '+%F %H:%M UTC')"
    echo "  **画面に控えた T0 と違うなら T0=HH:MM を渡して打ち直すこと**"
fi
if [ -n "${STEPS:-}" ]; then
    echo "  STEPS=$STEPS を渡された（残り時間からは決めない）"
else
    STEPS=$(python3 - "$BOOT" "$RESERVE_MIN" "$SEC_PER_STEP" "$SAVE_FREQ" <<'PY'
import sys, time
boot, reserve_min, sec_per_step, save_freq = (
    int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
budget = (boot + 12 * 3600) - time.time() - 120 - reserve_min * 60   # 120 = 読み込み
print(f"  締切 {time.strftime('%H:%M UTC', time.gmtime(boot + 12*3600))}"
      f" / 学習に使えるのは {budget/3600:.2f} 時間", file=sys.stderr)
for cand in range(6000, 0, -500):
    if cand * sec_per_step + (cand // save_freq) * 21 <= budget:
        print(cand); break
else:
    print(0)
PY
)
    [ "$STEPS" -ge 500 ] || die "500 歩も入らない" \
        "学習は回さず、環境を残して止める（venv は \$HOME に残る）"
    echo "  --steps $STEPS（$(python3 -c "print(f'{$STEPS*$SEC_PER_STEP/3600:.2f}')") 時間・後片付けに $RESERVE_MIN 分を残す）"
fi
# **最後の 1 個を数え落とさない。**`STEPS` が `SAVE_FREQ` の倍数でなければ、
# LeRobot は終端でもう 1 個保存する（5,500 歩 / 2,000 なら 2000・4000・5500 の 3 個）。
NCKPT=$(( STEPS / SAVE_FREQ + (STEPS % SAVE_FREQ != 0) ))
echo "  チェックポイント $NCKPT 個（1 個 8.8 GB ＝ 約 $((NCKPT * 9)) GB を HF へ）"
[ "$NCKPT" -gt 5 ] && die "チェックポイントが $NCKPT 個は多い（非公開の空きは約 57 GB）" \
    "SAVE_FREQ を大きくする。目安は SAVE_FREQ=$(( (STEPS + 2) / 3 ))"

# ---------------------------------------------------------------- 道具
step "上げる道具を置く"
for f in upload_ckpt.sh verify_upload.py; do
    [ -f "$TOOLS/$f" ] || cp "$CODE_DIR/$f" "$TOOLS/$f" \
        || die "$CODE_DIR/$f が無い" "pi05_code を取り直す"
done
chmod +x "$TOOLS/upload_ckpt.sh"
cat > "$HOME/run_env.sh" <<EOF
# blackwell_bowl5.sh が書いた。**ターミナルが落ちたら . ~/run_env.sh で戻る**
. ~/env_pi05.sh
export NVME=$NVME DS=$DS OUTPUT_ROOT=$OUTPUT_ROOT TOOLS=$TOOLS CODE_DIR=$CODE_DIR
export RUN=$RUN REPO=$REPO STEPS=$STEPS
export PATH="\$HOME/.local/bin:\$PATH"
EOF
echo "  ~/run_env.sh・$TOOLS"
[ "$DRY" = 1 ] && { echo; echo "=== DRY=1 なので、ここで止める ==="; exit 0; }

# ---------------------------------------------------------------- 起動
# **`--init-from` を渡さない。**渡さなければ `train_pi05.py` が
# `lerobot/pi05_libero_base` を自分で引く（上で降ろしたキャッシュに当たる）。
# 対照の `pi05_full_local` と同じ出発点にするために、ここは素の base である。
# **`--camera-viewpoint-augment` も付けない。**対照が付けていない。
step "学習を起こす"
NW=$(python3 -c "import os;print(min(os.cpu_count()//2, 16))")
cd "$CODE_DIR" || die "$CODE_DIR へ入れない"
setsid nohup python train_pi05.py --run-name "$RUN" \
    --dataset-repo "$DATASET_ID" --dataset-root "$DS" \
    --steps "$STEPS" --batch-size "$BATCH" --num-workers "$NW" \
    --learning-rate "$LR" --save-freq "$SAVE_FREQ" --seed 42 \
    --freeze-vision-encoder false --train-expert-only false \
    -- "--policy.scheduler_warmup_steps=$WARMUP" \
       "--policy.scheduler_decay_steps=$STEPS" \
    > "$OUTPUT_ROOT/$RUN.nohup" 2>&1 &
sleep 5
PID=$(pgrep -f "train_pi05.py --run-name $RUN" | head -1)
[ -n "$PID" ] || { tail -20 "$OUTPUT_ROOT/$RUN.nohup"; die "学習が立ち上がらない" \
    "上の nohup の末尾を読む"; }
echo "  PID $PID（setsid なので端末が落ちても生きる）"

step "上げるループを起こす"
ULOG="$HOME/upload_ckpt_$RUN.log"
OLD=$(pgrep -af "upload_ckpt.sh" | head -1)
[ -n "$OLD" ] && die "上げるループが既に動いている: $OLD" \
    "別の走りのものなら pkill -f upload_ckpt.sh してから打つ"
RUN="$RUN" REPO="$REPO" LOG="$ULOG" OUT="$OUTPUT_ROOT/$RUN" TRAIN_LOG_DIR="$OUTPUT_ROOT" \
    setsid nohup "$TOOLS/upload_ckpt.sh" > "$HOME/upload_ckpt_$RUN.nohup" 2>&1 &
sleep 25
grep -q "開始 RUN=$RUN" "$ULOG" 2>/dev/null || { tail -5 "$ULOG" 2>/dev/null
    tail -5 "$HOME/upload_ckpt_$RUN.nohup"; die "上げるループが立ち上がらない"; }
echo "  掴んだ（ログ $ULOG）"

# ---------------------------------------------------------------- 検算
# **20 歩まで待って 3 つ突き合わせる。**ここまで見ないと離席できない——
# 動かす範囲・読んでいるデータ・学習率のどれかが違っても、学習は黙って進む。
step "動き出しを確かめる（最初の 20 歩まで待つ・数分）"
for _ in $(seq 1 180); do
    grep -q "step:20 " "$OUTPUT_ROOT/$RUN.log" 2>/dev/null && break
    pgrep -f "train_pi05.py --run-name $RUN" >/dev/null \
        || { tail -20 "$OUTPUT_ROOT/$RUN.log" 2>/dev/null
             tail -20 "$OUTPUT_ROOT/$RUN.nohup"; die "学習が途中で落ちた"; }
    sleep 10
done
GOT_PARAMS=$(grep -m1 -o "num_learnable_params=[0-9]*" "$OUTPUT_ROOT/$RUN.log" | cut -d= -f2)
GOT_DSEPS=$(grep -m1 -o "dataset.num_episodes=[0-9]*" "$OUTPUT_ROOT/$RUN.log" | cut -d= -f2)
FIRST=$(grep -m1 -o "loss:[0-9.]* grdn:[0-9.]*" "$OUTPUT_ROOT/$RUN.log")
echo
printf '  %-26s %-14s %s\n' 見たもの 実際 期待
printf '  %-26s %-14s %s\n' num_learnable_params "$GOT_PARAMS" 4143404816
printf '  %-26s %-14s %s\n' dataset.num_episodes "$GOT_DSEPS" "$WANT_EPS"
printf '  %-26s %s\n' "最初の 20 歩" "$FIRST"
[ "$GOT_PARAMS" = 4143404816 ] || die "全パラメータになっていない（$GOT_PARAMS）" \
    "693422112 なら action expert だけである。止めて打ち直す" \
    "pkill -f 'train_pi05.py --run-name $RUN'; pkill -f 'lerobot[-_]train'"
[ "$GOT_DSEPS" = "$WANT_EPS" ] || die "読んでいるデータが違う（$GOT_DSEPS 本）" \
    "--dataset-root が効いていない。止めて打ち直す"

echo
echo "=== 走っている。離席してよい ==="
UPDT=$(grep -m1 -o "updt_s:[0-9.]*" "$OUTPUT_ROOT/$RUN.log" | cut -d: -f2)
[ -n "$UPDT" ] && python3 -c "
s = $UPDT * $STEPS
print(f'  終わる目安: {int(s//3600)} 時間 {int(s%3600//60)} 分（1 歩 $UPDT 秒 × $STEPS 歩・実測から）')
print(f'  **1 歩が 6.85 秒より目立って遅ければ、締切に間に合わない。**いま止めて歩数を減らすこと')"
echo "  ログ:   tail -f $OUTPUT_ROOT/$RUN.log"
echo "  上げ:   tail -f $ULOG"
echo "  監視セル（B6）は howto/14 の版を貼る。RUN = \"$RUN\" の 1 行だけ直す"
