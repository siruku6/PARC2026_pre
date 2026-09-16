#!/usr/bin/env bash
# JAIST サーバ（H100-20C・20 GB・1 枚）で MolmoAct2 の LoRA 学習を起こす。
#
#   全パラメータ学習は 20GB には載らない（`docs/howto/19` の 4.2 節。batch 8 でも
#   48.3GB 要る）ので、選べるのは `SCOPE=lora`（既定。VLM に LoRA・action expert は
#   フル学習）か `SCOPE=expert`（action expert だけ）の 2 つである。
#
#   **Docker から呼ぶ口も持っている**（`VENV=none`・`FOREGROUND=1`。
#   このマシン以外の 20 GB 機から使うため。`local_molmoact2_lora.sh` が呼ぶ）。
#
#   RUN=molmoact2_lora_v1 STEPS=4000 BATCH=2 bash solution/train/jaist_molmoact2_lora.sh
#
# **ssh を切るなら背面に置く**（jaist_launch.sh と同じ理由）。
#
#   RUN=molmoact2_lora_v1 STEPS=4000 BATCH=2 \
#       nohup bash solution/train/jaist_molmoact2_lora.sh > ~/launch_molmoact2_lora_v1.log 2>&1 &
#
# ## 根拠（9/16・RTX 6000 24GB での実測。JAIST の 20GB はこれより 4GB 狭い）
#
# | batch | GPU メモリの山 | 1 歩 |
# |---|---|---|
# | 2 | 18.7 GB | 7.2 秒 |
# | 4 | 19 GiB | 13.3 秒 |
# | 8 | 21 GiB | 25.6 秒 |
# | 16 | **OOM** | — |
#
# **batch 2 が最有力。**JAIST は 4GB 狭いので、batch 4 は載らない可能性がある
# （試すなら batch 2 の実測を先に取ってから）。
#
# ## この検証で見つけた 3 つのバグへの対処（すべてこのスクリプトが当てる）
#
# 1. **`enable_lora_vlm=true` は `--policy.checkpoint_path` に「アーキテクチャの
#    ひな型」を要求する。** ここに `--policy.path`（学習済みポリシー全体）と
#    同じ場所を渡すと、LeRobot の外側フォーマットを内側の HF モデル設定として
#    誤読し、隠れ次元が合わずに落ちる。`CKPT_MODE=small`（`SMALL_DIR`、重み無し・
#    形だけ合う版）を使うのが正しい——`runpod_molmoact2_ft.sh` と同じパッチ
#    （`lerobot_patches/modeling_molmoact2.py`）で対応する。
# 2. **`lerobot_train.py` が前処理ステップ名を `normalizer_processor` に決め打ち
#    している**が、MolmoAct2 は `molmoact2_masked_normalizer` という自前の名前で
#    登録しており、`--policy.path` から学習しようとすると 100% KeyError で落ちる。
#    `lerobot_patches/lerobot_train.py` で対応する（LoRA 固有ではなく全パラメータ
#    学習でも起きる）。
# 3. **前処理側でも素の重みを二重に読んでしまう。** `MolmoAct2PackInputsProcessorStep`
#    は保存済みパイプラインの `checkpoint_path`（チェックポイントの publisher が使った
#    値。たいてい素の `allenai/MolmoAct2-LIBERO` のまま）から自分を再構築し、
#    `--policy.checkpoint_path` の上書きが効かない。`SMALL_DIR` を使っていても
#    **このステップだけは毎回 21.8 GB を裏でネットワークから取りに行く**
#    （9/16 に実測。オフライン + 空の HF キャッシュで再現し、`lerobot_train.py`
#    パッチに `molmoact2_pack_inputs` の上書きを足して解消したことを検証済み）。
#    同じ `lerobot_patches/lerobot_train.py` で対応する。
#
# ## データの前提
#
# **QUANTILES 統計（q01/q99）が要る。**π0.5 向けに組んだデータセットには無いので、
# 先に `solution/tools/train/add_quantile_stats.py --root $DS` を打っておくこと
# （parquet の数値列だけを読むので数分で済む。**公式の
# `augment_dataset_quantile_stats.py` は全コマの動画を復号するので、21,642 本では
# 日をまたぐ**）。このスクリプトは無ければ止まる。
#
# Reads:
#   - $VENV/bin/activate（下の 0 節を先に一度だけ実行）
#   - $DS（混ぜたデータセット。QUANTILES 統計込み）
#   - HF `$INIT_REPO`@`$INIT_REV` と `$SMALL_REPO`@`$SMALL_REV`（出発点）
# Generates:
#   - $OUT/$RUN/checkpoints/**
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN="${RUN:?RUN を渡すこと（例 RUN=molmoact2_lora_v1）}"
VENV="${VENV:-$HOME/venvs/molmoact2_lora}"
DS="${DS:?DS を渡すこと（QUANTILES 統計込みのデータセットの場所）}"
DATASET_ID="${DATASET_ID:-local/$(basename "$DS")}"
INIT_REPO="${INIT_REPO:-allenai/MolmoAct2-LIBERO-LeRobot}"
INIT_REV="${INIT_REV:-f0c5a9567c2b72faadec901e16e055c8b098c2f5}"
SMALL_REPO="${SMALL_REPO:-allenai/MolmoAct2-LIBERO}"
SMALL_REV="${SMALL_REV:-0d24a92bd1faf321ef497c3bbd5681af97c65aa2}"
INIT_DIR="${INIT_DIR:-$HOME/molmoact2_init/molmoact2_libero_lerobot}"
SMALL_DIR="${SMALL_DIR:-$HOME/molmoact2_init/molmoact2_base_small}"
OUT="${OUT:-$REPO/work/train}"
STEPS="${STEPS:-4000}"
BATCH="${BATCH:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
SEED="${SEED:-1000}"
WARMUP="${WARMUP:-200}"
LR="${LR:-1e-5}"
VIT_LR="${VIT_LR:-5e-6}"
CONNECTOR_LR="${CONNECTOR_LR:-5e-6}"
EXPERT_LR="${EXPERT_LR:-5e-5}"
DECAY_LR="${DECAY_LR:-1e-6}"
# **切り分け用の 2 つ。**`ACTION_MODE=continuous` は離散の行動の損失を外す（loss が nan に
# なるときの切り分け）。`LOG_FREQ=1` は 1 歩ごとにログを出す（何歩目から nan かを見る）。
ACTION_MODE="${ACTION_MODE:-both}"
LOG_FREQ="${LOG_FREQ:-20}"
# **動かす範囲は `SCOPE` の 1 つで決める。**`lora`（既定）は VLM に LoRA・action expert は
# フル学習。`expert` は action expert だけ（20 GB で載るもう 1 つの形）。
# **設定側が「両立しない」と弾く組み合わせを作れないよう、口を 1 つにしてある**
# ——`train_action_expert_only` は `enable_lora_vlm` と併用できず、`action_mode=continuous`
# を要求するので、`expert` のときは `ACTION_MODE` もここで合わせる。
SCOPE="${SCOPE:-lora}"
case "$SCOPE" in
    lora)   SCOPE_LABEL="LoRA（VLM）＋ action expert フル学習"
            SCOPE_ARGS=(--policy.enable_lora_vlm=true) ;;
    expert) SCOPE_LABEL="action expert だけ"
            ACTION_MODE=continuous
            SCOPE_ARGS=(--policy.enable_lora_vlm=false --policy.train_action_expert_only=true) ;;
    *)      echo "**SCOPE は lora か expert（渡された値: $SCOPE）**" >&2; exit 2 ;;
esac
PATCH="${PATCH:-$REPO/solution/tools/eval/lerobot_patches/modeling_molmoact2.py}"
TRAIN_PATCH="${TRAIN_PATCH:-$REPO/solution/tools/eval/lerobot_patches/lerobot_train.py}"

RENAME_MAP='{"observation.images.front":"observation.images.image","observation.images.wrist":"observation.images.wrist_image"}'

die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }
step() { echo; echo "=== $* ==="; }

step "前提"
# **`VENV=none` なら venv を使わない。**Docker のイメージの中から呼ぶときに使う
# （local_molmoact2_lora.sh。lerobot はイメージに最初から入っている）。
if [ "$VENV" != none ]; then
    [ -f "$VENV/bin/activate" ] || die "venv が無い: $VENV" \
        "python3 -m venv $VENV && $VENV/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && ..." \
        "solution/train/Dockerfile と同じ版を cu128 で入れ、torchcodec は入れず av だけにする（pyav backend）"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
python3 -c "import torch; assert torch.cuda.is_available()" || die "GPU が見えない"
python3 -c "import peft" 2>/dev/null || die "peft が無い" "pip install 'peft>=0.18.0,<1.0.0'"
[ -d "$DS/meta" ] || die "$DS が無い"
python3 -c "
import json
d = json.load(open('$DS/meta/stats.json'))
assert 'q01' in d.get('action', {}), 'action に q01/q99 が無い'
assert 'q01' in d.get('observation.state', {}), 'observation.state に q01/q99 が無い'
" || die "QUANTILES 統計（q01/q99）が無い" \
    "python3 $REPO/solution/tools/train/add_quantile_stats.py --root $DS を先に打つ"

step "出発点の重みを確かめる（無ければ降ろす）"
mkdir -p "$INIT_DIR" "$SMALL_DIR" "$OUT"
if [ ! -f "$INIT_DIR/model.safetensors" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$INIT_REPO', revision='$INIT_REV', local_dir='$INIT_DIR')
" || die "$INIT_REPO の取得に失敗"
fi
if [ ! -f "$SMALL_DIR/config.json" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$SMALL_REPO', revision='$SMALL_REV', local_dir='$SMALL_DIR',
    allow_patterns=['config.json', 'processor_config.json', 'tokenizer.json',
                    'tokenizer_config.json', 'chat_template.jinja', 'norm_stats.json', '*.py'])
" || die "$SMALL_REPO の取得に失敗"
fi

# **目印が無いファイルにだけ直しを当てる。**Docker から呼ぶときは呼ぶ側が同じ 2 ファイルを
# 読み取り専用で bind-mount しているので（local_molmoact2_lora.sh）、既に目印があり当てない
# ——`--user` で非 root にすると `/opt/lerobot` は書けないので、当てようとすると落ちる。
step "2 つの直しを確かめる（無ければ当てる）"
LEROBOT_DIR=$(python3 -c "import lerobot,os;print(os.path.dirname(lerobot.__file__))")
for spec in "$PATCH:policies/molmoact2/modeling_molmoact2.py:has_base_weights" \
            "$TRAIN_PATCH:scripts/lerobot_train.py:is_molmoact2"; do
    IFS=: read -r src rel mark <<< "$spec"
    target="$LEROBOT_DIR/$rel"
    if grep -q "$mark" "$target"; then
        echo "  当たっている: $target"
        continue
    fi
    [ -f "$src" ] || die "直しが無い: $src"
    [ -f "$target.orig" ] || cp "$target" "$target.orig" || die "元ファイルを控えられない: $target"
    cp "$src" "$target" || die "直しを当てられない: $target" "書けない場所なら呼ぶ側で mount する"
    echo "  当てた: $target（元は $target.orig）"
done
python3 -c "import lerobot.policies.molmoact2.modeling_molmoact2; import lerobot.scripts.lerobot_train" \
    || die "直しを当てたら import できない" "$LEROBOT_DIR 配下の .orig から戻す"

step "出力先の確認"
[ -e "$OUT/$RUN" ] && die "出力先が既にある: $OUT/$RUN" "別の RUN 名にするか消す"
NCKPT=$(( STEPS / SAVE_FREQ ))
FREE=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
echo "  空き ${FREE} GB（チェックポイント約 $NCKPT 個 × 11GB）"
[ "$FREE" -lt $(( NCKPT * 11 + 10 )) ] && die "空きが足りない（${FREE} GB）" "SAVE_FREQ を上げるか古いチェックポイントを消す"

step "打つ条件"
printf '  %-14s %s\n' RUN "$RUN" データ "$DS" \
    歩数 "$STEPS" batch "$BATCH" 学習率 "本体 $LR ／ ViT $VIT_LR ／ つなぎ $CONNECTOR_LR ／ action expert $EXPERT_LR" \
    動かす範囲 "$SCOPE_LABEL"

TRAIN_ARGS=(
    --policy.path="$INIT_DIR"
    --policy.checkpoint_path="$SMALL_DIR"
    --policy.device=cuda
    --policy.action_mode="$ACTION_MODE"
    --policy.model_dtype=bfloat16
    --policy.chunk_size=10
    --policy.n_action_steps=10
    --policy.num_flow_timesteps=8
    --policy.gradient_checkpointing=true
    "${SCOPE_ARGS[@]}"
    --policy.optimizer_lr="$LR"
    --policy.optimizer_vit_lr="$VIT_LR"
    --policy.optimizer_connector_lr="$CONNECTOR_LR"
    --policy.optimizer_action_expert_lr="$EXPERT_LR"
    --policy.scheduler_warmup_steps="$WARMUP"
    --policy.scheduler_decay_steps="$STEPS"
    --policy.scheduler_decay_lr="$DECAY_LR"
    --policy.push_to_hub=false
    --dataset.repo_id="$DATASET_ID"
    --dataset.root="$DS"
    --dataset.video_backend=pyav
    --dataset.image_transforms.enable=false
    --rename_map="$RENAME_MAP"
    --output_dir="$OUT/$RUN"
    --job_name="$RUN"
    --steps="$STEPS"
    --batch_size="$BATCH"
    --num_workers="$NUM_WORKERS"
    --seed="$SEED"
    --save_checkpoint=true
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --env_eval_freq=-1
    --wandb.enable=false
)

# **`FOREGROUND=1` なら切り離さずに前面で走らせる。**Docker のコンテナから呼ぶときは、
# 切り離すとこのスクリプトが終わった時点でコンテナごと落ちる。
LAUNCH=(accelerate launch --num_processes=1 --mixed_precision=bf16
        -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}")
if [ "${FOREGROUND:-0}" = 1 ]; then
    step "起こす（前面。ログは $OUT/$RUN.nohup にも書く）"
    "${LAUNCH[@]}" 2>&1 | tee "$OUT/$RUN.nohup"
    exit "${PIPESTATUS[0]}"
fi

step "起こす"
setsid nohup "${LAUNCH[@]}" > "$OUT/$RUN.nohup" 2>&1 < /dev/null &
echo "  PID $!"
echo "=== 動き出しを確かめる（最初の 20 歩まで待つ） ==="
for _ in $(seq 1 180); do
    grep -q "step:20 \|step:$STEPS " "$OUT/$RUN.nohup" 2>/dev/null && break
    kill -0 "$!" 2>/dev/null || { tail -n 30 "$OUT/$RUN.nohup"; die "落ちた" "上のログを読む"; }
    sleep 10
done
echo "  ログ: tail -f $OUT/$RUN.nohup | tr '\\r' '\\n'"
echo "  止める: kill -0 $! で生存確認 → kill -TERM $!"
