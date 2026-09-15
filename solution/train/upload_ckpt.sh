#!/usr/bin/env bash
# チェックポイントの pretrained_model と学習ログを HF へ上げ、**上がったことを確かめてから**印を付ける。
#
#   RUN=pi05_proto20 REPO=siruku6/pi05_proto20 nohup ~/upload_ckpt.sh > ~/upload_ckpt.nohup 2>&1 &
#
# 摘み・ログの読み方・ログも上げる理由は `docs/howto/17_checkpoint_upload.md` にある。
#
# | 踏んだこと | ここでどう潰すか |
# |---|---|
# | **`000500` の次から 1 個も上がらず、跡も残らなかった**（9/3・`howto/14` の分岐 5） | **上げた後に HF を読んで数える**（`verify_upload.py`）。**確かめられなければ印を付けない**ので次の巡でやり直す。**止まりうる場所の手前に必ず 1 行書く**ので、跡が残る |
# | **上げている 10 分の間に学習が終わると `break` して最後の 1 個を落とす**（9/3） | **`break` の条件を「学習が終わった かつ 未上げが 0 個」にした。**学習が終わってからも、残りを上げ切るまで抜けない |
# | **チェックポイントは上がったのに、学習ログが 1 行も上がっていなかった**（9/13 に `pi05_bowl5_full` で踏んだ） | **チェックポイントを上げるたびに、ログも `logs/` へ上げる。**最後の 1 回は学習が終わってから上げ、**それが確かめられるまで `_all_uploaded` を置かない** |
#
# **抜けるときに `$MARK/_all_uploaded` を置く。**監視セルはこれが在るときだけ
# サーバを止める。**在らなければ止めない**（GPU 時間は減るが、上がっていない
# チェックポイントは消えない。`/opt/dlami/nvme` はインスタンス終了で消える）。
set -u
RUN="${RUN:?RUN を渡すこと（--run-name と同じ文字列）}"
REPO="${REPO:?REPO を渡すこと（例 siruku6/pi05_proto20）}"
OUT="${OUT:-/opt/dlami/nvme/out/$RUN}"
MARK="${MARK:-$HOME/uploaded/$RUN}"
LOG="${LOG:-$HOME/upload_ckpt.log}"
VERIFY="${VERIFY:-$HOME/tools/verify_upload.py}"
DONE="$MARK/_all_uploaded"
# **学習ログの在り処は、起こすスクリプトが渡す**（ログを置いた本人が知っている）。
# 既定は `OUT` の親——`train_pi05.py` が `<出力先の親>/<RUN>.log` に書く既定と同じで、
# 手で起こすとき用である。
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-$(dirname "$OUT")}"
# 伸び続けるログをそのまま上げると大きさが変わって突き合わせられないので、写しを上げる。
LOG_COPY="$HOME/train_logs/$RUN"

mkdir -p "$MARK"
say() { echo "$(date -Iseconds) $*" >> "$LOG"; }

# **`last` は除く。**LeRobot は保存のたびに `checkpoints/last` の symlink を貼り直すので、
# 拾うと HF に「last」という歩数のふりをした複製ができる（9/3 に踏んだ）。
pending_list() {
  local dir step
  for dir in "$OUT"/checkpoints/*/pretrained_model; do
    [ -d "$dir" ] || continue
    [ -L "${dir%/pretrained_model}" ] && continue
    step=$(basename "$(dirname "$dir")")
    [ -f "$MARK/$step" ] || echo "$step"
  done
}

training_alive() { pgrep -f "train_pi05.py --run-name $RUN" >/dev/null; }

# `hf upload` し、HF を読んで確かめる。確かめられたら 0 を返す。
#   upload_verified <ログの札> <手元のディレクトリ> <HF 上の置き場> <上げる時間の上限 [秒]>
upload_verified() {
  local label="$1" local_dir="$2" dest="$3" limit="$4" rc
  timeout "$limit" hf upload "$REPO" "$local_dir" "$dest" --repo-type model
  rc=$?
  if [ "$rc" -ne 0 ]; then
    say "$label **hf upload が失敗**（終了コード $rc）"
    return 1
  fi
  say "$label hf upload は 0 で返った。HF を読んで確かめる"
  if timeout $(( limit < 600 ? limit : 600 )) python3 "$VERIFY" \
      --repo "$REPO" --step "$dest" --local "$local_dir" >> "$LOG" 2>&1; then
    return 0
  fi
  say "$label **確かめられない**（次の巡でやり直す）"
  return 1
}

# 学習ログを写して `logs/` へ上げる。確かめられたら 0 を返す。
push_logs() {
  local f copied=0
  mkdir -p "$LOG_COPY"
  for f in "$RUN.log" "$RUN.nohup"; do
    [ -f "$TRAIN_LOG_DIR/$f" ] && cp "$TRAIN_LOG_DIR/$f" "$LOG_COPY/$f" && copied=$((copied + 1))
  done
  if [ "$copied" -eq 0 ]; then
    say "ログ **$TRAIN_LOG_DIR に $RUN.log も $RUN.nohup も無い**（TRAIN_LOG_DIR を渡すこと）"
    return 1
  fi
  say "ログ 上げ始める（$copied 個・写しは $LOG_COPY）"
  # 数十 KB なので、2 分で上がらなければ失敗として次の巡に回す
  upload_verified ログ "$LOG_COPY" logs 120 && say "ログ **確かめた**"
}

rm -f "$DONE"
say "=== 開始 RUN=$RUN REPO=$REPO OUT=$OUT TRAIN_LOG_DIR=$TRAIN_LOG_DIR"
say "    既にある印: $(ls "$MARK" 2>/dev/null | tr '\n' ' ')"

# **学習が立ち上がるのを待つ。**待たずに入ると、未上げ 0 個・学習なしの状態で
# **「全部上がった」の印を誤って置く**（監視セルがそれを見てサーバを止める）。
WAIT_START="${WAIT_START:-600}"
waited=0
while ! training_alive && [ "$waited" -lt "$WAIT_START" ]; do
  sleep 10; waited=$((waited + 10))
done
if ! training_alive; then
  say "**$WAIT_START 秒待っても $RUN が見つからない。印は置かず終わる**"
  exit 1
fi
say "    学習を見つけた（$waited 秒待った）"

while true; do
  for step in $(pending_list); do
    dir="$OUT/checkpoints/$step/pretrained_model"

    # **書き終える前に上げない。**30 秒あけて大きさが変わらないことを見る。
    # **`du` にも timeout を掛ける**——9/3 に止まった場所の候補である
    say "$step 大きさを測る"
    a=$(timeout 120 du -sb "$dir" | cut -f1) || { say "$step du が返らない"; continue; }
    sleep 30
    b=$(timeout 120 du -sb "$dir" | cut -f1) || { say "$step du が返らない"; continue; }
    [ "$a" = "$b" ] || { say "$step まだ書いている（$a → $b）"; continue; }

    say "$step 上げ始める（$((a / 1000000)) MB）"
    if upload_verified "$step" "$dir" "$step" 3600; then
      date -Iseconds > "$MARK/$step"
      say "$step **確かめた。印を付けた**"
      rm -rf "$OUT/checkpoints/$step/training_state"   # 16 GB。上げないので消す
      push_logs   # ここまでの loss の推移を一緒に上げる
    fi
  done

  left=$(pending_list | wc -l)
  if training_alive; then
    say "見回り 未上げ $left 個・学習は走っている"
  else
    say "見回り 未上げ $left 個・学習は終わった"
    if [ "$left" -gt 0 ]; then
      say "**学習は終わったが未上げが $left 個ある。抜けない**"
    # **最後のログは学習が終わってから上げる。**最後の保存のあとにも、
    # 終了コードや落ちたときの traceback が書かれる。
    elif push_logs; then
      date -Iseconds > "$DONE"
      say "=== 全部上がったことを確かめた（チェックポイントとログ）。ループを終える ==="
      break
    else
      say "**チェックポイントは全部上がったが、最後のログが上がっていない。抜けない**"
    fi
  fi
  sleep 60
done
