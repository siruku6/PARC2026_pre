#!/usr/bin/env bash
# JAIST サーバ（H100-20C・20 GB）で action expert だけの学習を起こす。
#
#   RUN=pi05_proto22 STEPS=8500 bash solution/train/jaist_launch.sh
#
# **Blackwell とは起こし方が違う。**あちらは素の python、こちらは docker である
# （[howto/07](../../docs/howto/07_train_pi05_local.md)）。
#
# **20 GB には action expert 限定しか載らない。**全パラメータは RunPod か Blackwell
# でしかできない（`resources/infra_and_docker.md` の 1 節）。だから
# `--freeze-vision-encoder true --train-expert-only true` を外せないようにしてある。
#
# **イメージに焼かれた train_pi05.py は古い可能性がある**ので、
# `solution/train` を `/opt/pi05` に被せて、いま手元にある版を使う。
#
# Reads:
#   - $DATASET   混ぜたデータセット（このリポジトリの外でよい）
#   - $WEIGHTS   出発点の重み
# Generates:
#   - $OUT/<RUN>/  チェックポイントとログ
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN="${RUN:?RUN を渡すこと（例 RUN=pi05_proto22）}"
DATASET="${DATASET:-$REPO/work/artifacts/t3_mix}"
WEIGHTS="${WEIGHTS:-$REPO/work/artifacts/pi05_rp3000}"
OUT="${OUT:-$REPO/work/train}"
IMAGE="${IMAGE:-parc2026-train-pi05}"
STEPS="${STEPS:-8500}"
BATCH="${BATCH:-16}"
LR="${LR:-5e-6}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/t3_mix}"

die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }

echo "=== 打つ条件 ==="
printf '  %-14s %s\n' RUN "$RUN" データ "$DATASET" 出発点 "$WEIGHTS" \
    歩数 "$STEPS" batch "$BATCH" 学習率 "$LR" 動かす範囲 "action expert だけ（693M）"

echo "=== 前提の確認 ==="
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "イメージ $IMAGE が無い" \
    "cd $REPO/solution/train && docker build -f Dockerfile.pi05 -t $IMAGE ."
[ -d "$DATASET/meta" ] || die "$DATASET が無い" "混ぜたデータセットを落としてくる"
[ -f "$DATASET/meta/stats.json.bak" ] || die "正規化が据え置かれていない" \
    "作った側で freeze を打ってから運ぶこと"
[ -d "$WEIGHTS/vlm" ] || die "$WEIGHTS に vlm/ が無い" \
    "hf download siruku6/pi05_full_runpod --local-dir $WEIGHTS"
[ -e "$OUT/$RUN" ] && die "出力先が既にある: $OUT/$RUN" "別の RUN 名にするか消す"
NCKPT=$(( STEPS / SAVE_FREQ ))
[ "$NCKPT" -gt 8 ] && die "チェックポイントが $NCKPT 個（約 $((NCKPT * 9)) GB）になる" \
    "JAIST は空きが狭い。SAVE_FREQ=$(( (STEPS + 3) / 4 )) くらいにする"
EPS=$(python3 -c "import json;print(json.load(open('$DATASET/meta/info.json'))['total_episodes'])") \
    || die "$DATASET/meta/info.json が読めない" "落とし方が途中で切れている"
FREE=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
echo "  データ $EPS 本・据え置き済み・空き ${FREE} GB（要る量は約 $((NCKPT * 9 + 20)) GB）"
[ "$FREE" -lt $((NCKPT * 9 + 10)) ] && die "空きが足りない（${FREE} GB）" \
    "work/submission_geo12k.zip（12 GB・howto/02 で作り直せる）" \
    "work/artifacts/pi05_rp3000（HF siruku6/pi05_full_runpod）" \
    "work/artifacts/trial1500（HF siruku6/pi05_trial1500）"

echo "=== 起こす ==="
mkdir -p "$OUT"
docker run -d --gpus all --name "parc-$RUN" \
    --user "$(id -u):$(id -g)" \
    -v "$REPO/solution/train:/opt/pi05:ro" \
    -v "$DATASET:/data:ro" -v "$WEIGHTS:/weights:ro" \
    -v "$OUT:/out" -v "$HOME/.cache/huggingface:/hf_home" \
    "$IMAGE" \
    python /opt/pi05/train_pi05.py --run-name "$RUN" \
        --output-root /out \
        --init-from /weights \
        --dataset-repo "$DATASET_REPO_ID" --dataset-root /data \
        --steps "$STEPS" --batch-size "$BATCH" --num-workers 8 \
        --learning-rate "$LR" --save-freq "$SAVE_FREQ" \
        --freeze-vision-encoder true --train-expert-only true \
        --camera-viewpoint-augment \
    || die "docker run が失敗" "docker logs parc-$RUN を読む"

echo "  コンテナ parc-$RUN"
echo "=== 動き出しを確かめる（最初の 20 歩まで待つ）==="
for _ in $(seq 1 180); do
    docker logs "parc-$RUN" 2>&1 | tr '\r' '\n' | grep -q "loss:" && break
    docker inspect -f '{{.State.Running}}' "parc-$RUN" 2>/dev/null | grep -q true \
        || { docker logs --tail 30 "parc-$RUN" 2>&1 | tr '\r' '\n'; die "落ちた" "上のログを読む"; }
    sleep 10
done
LOG=$(docker logs "parc-$RUN" 2>&1 | tr '\r' '\n')
GOT_PARAMS=$(echo "$LOG" | grep -m1 -o "num_learnable_params=[0-9]*" | cut -d= -f2)
GOT_EPS=$(echo "$LOG"    | grep -m1 -o "dataset.num_episodes=[0-9]*" | cut -d= -f2)
FIRST=$(echo "$LOG"      | grep -m1 -o "loss:[0-9.]* grdn:[0-9.]*")
UPDT=$(echo "$LOG"       | grep -m1 -o "updt_s:[0-9.]*" | cut -d: -f2)
printf '  %-24s %-14s %s\n' 見たもの 実際 期待
printf '  %-24s %-14s %s\n' num_learnable_params "$GOT_PARAMS" 693422112
printf '  %-24s %-14s %s\n' dataset.num_episodes "$GOT_EPS" "$EPS"
printf '  %-24s %s\n' "最初の 20 歩" "$FIRST"
[ "$GOT_PARAMS" = 693422112 ] || die "動かす範囲が違う（$GOT_PARAMS）" "docker stop parc-$RUN"
[ "$GOT_EPS" = "$EPS" ]       || die "読んでいるデータが違う（$GOT_EPS 本）" "docker stop parc-$RUN"
[ -n "$UPDT" ] && echo "  終わる目安: $(python3 -c "
s = $UPDT * $STEPS
print(f'{int(s//3600)} 時間 {int(s%3600//60)} 分（1 歩 $UPDT 秒・実測から）')")"

echo
echo "=== 走っている ==="
echo "  ログ: docker logs -f parc-$RUN"
echo "  歩数: docker logs --tail 2 parc-$RUN 2>&1 | tr '\\r' '\\n' | grep step:"
echo "  止める: docker stop -t 60 parc-$RUN"
