#!/usr/bin/env bash
# チェックポイントの pretrained_model を HF へ上げ、**上がったことを確かめてから**印を付ける。
#
#   RUN=pi05_proto20 REPO=siruku6/pi05_proto20 nohup ~/upload_ckpt.sh > ~/upload_ckpt.nohup 2>&1 &
#
# **2026-09-03 に踏んだ 2 つを潰すために作り直した**（`howto/14` の分岐 5）。
#
# | 踏んだこと | ここでどう潰すか |
# |---|---|
# | **`000500` の次から 1 個も上がらず、跡も残らなかった** | **上げた後に HF を読んで数える**（`verify_upload.py`）。**確かめられなければ印を付けない**ので次の巡でやり直す。**止まりうる場所の手前に必ず 1 行書く**ので、跡が残る |
# | **上げている 10 分の間に学習が終わると `break` して最後の 1 個を落とす** | **`break` の条件を「学習が終わった かつ 未上げが 0 個」にした。**学習が終わってからも、残りを上げ切るまで抜けない |
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

rm -f "$DONE"
say "=== 開始 RUN=$RUN REPO=$REPO OUT=$OUT"
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
    if timeout 3600 hf upload "$REPO" "$dir" "$step" --repo-type model; then
      say "$step hf upload は 0 で返った。HF を読んで確かめる"
    else
      say "$step **hf upload が失敗**（終了コード $?）"
      continue
    fi

    if timeout 600 python3 "$VERIFY" --repo "$REPO" --step "$step" --local "$dir" >> "$LOG" 2>&1; then
      date -Iseconds > "$MARK/$step"
      say "$step **確かめた。印を付けた**"
      rm -rf "$OUT/checkpoints/$step/training_state"   # 16 GB。上げないので消す
    else
      say "$step **確かめられない。印を付けない**（次の巡でやり直す）"
    fi
  done

  left=$(pending_list | wc -l)
  if training_alive; then
    say "見回り 未上げ $left 個・学習は走っている"
  else
    say "見回り 未上げ $left 個・学習は終わった"
    if [ "$left" -eq 0 ]; then
      date -Iseconds > "$DONE"
      say "=== 全部上がったことを確かめた。ループを終える ==="
      break
    fi
    say "**学習は終わったが未上げが $left 個ある。抜けない**"
  fi
  sleep 60
done
