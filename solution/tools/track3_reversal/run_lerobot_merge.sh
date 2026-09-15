#!/usr/bin/env bash
# タスクごとに作った逆向きデータセットを、1 つにまとめる。
#
#   bash solution/tools/track3_reversal/run_lerobot_merge.sh 出力 データセット...
#
# 例（$OUT からの相対で指定する）:
#   bash .../run_lerobot_merge.sh probe_017/reverse_lerobot_all \
#     probe_017/reverse_lerobot_book probe_017/reverse_lerobot_soup \
#     probe_017_mug/reverse_lerobot_mug
#
# **$OUT の下ならディレクトリをまたいで指定できる**（タスクごとに別の出力先で
# 回していても、集めるために動かさずに済む）。既定の $OUT は work/。
#
# **`merge` は 2 つずつしか混ぜられない**ので、3 つ以上は順に足す。途中結果は
# `<出力>_tmpN` に置いて最後に消す。
#
# **入力は書き換えない。**タスクを 1 つ足したくなったら、同じコマンドを流し直せば
# よい（数百 MB のコピーで済む）。配布データセット（20 GB）へ混ぜるのは最後に
# 1 度だけにする。
set -uo pipefail
REPO="${REPO:-/home/ubuntu/src/PARC2026_pre}"
OUT="${OUT:-$REPO/work}"
IMAGE="${IMAGE:-parc2026-train-pi05}"
DEST="$1"; shift

run() {
  docker run --rm \
    -v "$REPO/solution/tools:/host/tools:ro" -v "$OUT:/data" \
    --entrypoint python "$IMAGE" \
    /host/tools/scripted_demo/npz_to_lerobot.py "$@"
}

base="$1"; shift
i=0
for add in "$@"; do
  i=$((i + 1))
  out=$([ $# -eq $i ] && echo "$DEST" || echo "${DEST}_tmp$i")
  echo "=== $base + $add -> $out ==="
  run merge --base "/data/$base" --base-repo-id "local/t3_reverse_$(basename "$base")" \
            --add "/data/$add" --add-repo-id "local/t3_reverse_$(basename "$add")" \
            --output "/data/$out" --output-repo-id "local/t3_reverse_all" || exit 1
  base="$out"
done
run inspect --root "/data/$DEST"
# **所有を戻すのが先である。**コンテナが root で書いたので、その前に消そうと
# すると中間物が消えずに残る。
docker run --rm -v "$OUT:/data" --entrypoint chown "$IMAGE" \
  -R "$(id -u):$(id -g)" /data >/dev/null 2>&1 || true
rm -rf "$OUT/${DEST}_tmp"*
echo "=== 出来上がり: $OUT/$DEST ==="
