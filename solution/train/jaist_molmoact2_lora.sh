#!/usr/bin/env bash
# JAIST サーバ（H100-20C・20 GB・1 枚）で MolmoAct2 の LoRA 学習を起こす。
#
#   ENABLE_LORA=1 が常時のこのスクリプト固有の版。全パラメータ学習は
#   20GB には載らない（`docs/howto/19` の 4.2 節。batch 8 でも 48.3GB 要る）ので、
#   JAIST では LoRA（VLM に LoRA・action expert はフル学習）しか選べない。
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
# 先に `augment_dataset_quantile_stats.py --repo-id=... --root=$DS` を打っておく
# こと（21,642 本規模だと動画を読み直すため長時間かかる。9/16 の検証では
# 近似値で代用した——本番では本物を使うこと）。このスクリプトは無ければ止まる。
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
PATCH="${PATCH:-$REPO/solution/tools/eval/lerobot_patches/modeling_molmoact2.py}"
TRAIN_PATCH="${TRAIN_PATCH:-$REPO/solution/tools/eval/lerobot_patches/lerobot_train.py}"

RENAME_MAP='{"observation.images.front":"observation.images.image","observation.images.wrist":"observation.images.wrist_image"}'

die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }
step() { echo; echo "=== $* ==="; }

step "前提"
[ -f "$VENV/bin/activate" ] || die "venv が無い: $VENV" \
    "python3 -m venv $VENV && $VENV/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && ..." \
    "solution/train/Dockerfile と同じ版を cu128 で入れ、torchcodec は入れず av だけにする（pyav backend）"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python3 -c "import torch; assert torch.cuda.is_available()" || die "GPU が見えない"
python3 -c "import peft" 2>/dev/null || die "peft が無い" "$VENV/bin/pip install 'peft>=0.18.0,<1.0.0'"
[ -d "$DS/meta" ] || die "$DS が無い"
python3 -c "
import json
d = json.load(open('$DS/meta/stats.json'))
assert 'q01' in d.get('action', {}), 'action に q01/q99 が無い'
assert 'q01' in d.get('observation.state', {}), 'observation.state に q01/q99 が無い'
" || die "QUANTILES 統計（q01/q99）が無い" \
    "augment_dataset_quantile_stats.py --repo-id=... --root=$DS を先に打つ"

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

step "2 つのパッチを当てる（CKPT_MODE=small・前処理ステップ名）"
[ -f "$PATCH" ] || die "直しが無い: $PATCH"
TARGET=$(python3 -c "import lerobot,os;print(os.path.join(os.path.dirname(lerobot.__file__),'policies/molmoact2/modeling_molmoact2.py'))")
[ -f "$TARGET.orig" ] || cp "$TARGET" "$TARGET.orig" || die "元ファイルを控えられない: $TARGET"
cp "$PATCH" "$TARGET" || die "直しを当てられない: $TARGET"
[ -f "$TRAIN_PATCH" ] || die "直しが無い: $TRAIN_PATCH"
TRAIN_TARGET=$(python3 -c "import lerobot,os;print(os.path.join(os.path.dirname(lerobot.__file__),'scripts/lerobot_train.py'))")
[ -f "$TRAIN_TARGET.orig" ] || cp "$TRAIN_TARGET" "$TRAIN_TARGET.orig" || die "元ファイルを控えられない: $TRAIN_TARGET"
cp "$TRAIN_PATCH" "$TRAIN_TARGET" || die "直しを当てられない: $TRAIN_TARGET"
python3 -c "import lerobot.policies.molmoact2.modeling_molmoact2; import lerobot.scripts.lerobot_train" \
    || die "パッチ後に import できない" "$TARGET.orig / $TRAIN_TARGET.orig から戻す"

step "出力先の確認"
[ -e "$OUT/$RUN" ] && die "出力先が既にある: $OUT/$RUN" "別の RUN 名にするか消す"
NCKPT=$(( STEPS / SAVE_FREQ ))
FREE=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
echo "  空き ${FREE} GB（チェックポイント約 $NCKPT 個 × 11GB）"
[ "$FREE" -lt $(( NCKPT * 11 + 10 )) ] && die "空きが足りない（${FREE} GB）" "SAVE_FREQ を上げるか古いチェックポイントを消す"

step "打つ条件"
printf '  %-14s %s\n' RUN "$RUN" データ "$DS" \
    歩数 "$STEPS" batch "$BATCH" 学習率 "本体 $LR ／ ViT $VIT_LR ／ つなぎ $CONNECTOR_LR ／ action expert $EXPERT_LR" \
    動かす範囲 "LoRA（VLM）＋ action expert フル学習"

TRAIN_ARGS=(
    --policy.path="$INIT_DIR"
    --policy.checkpoint_path="$SMALL_DIR"
    --policy.device=cuda
    --policy.action_mode=both
    --policy.model_dtype=bfloat16
    --policy.chunk_size=10
    --policy.n_action_steps=10
    --policy.num_flow_timesteps=8
    --policy.gradient_checkpointing=true
    --policy.enable_lora_vlm=true
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
    --log_freq=20
    --env_eval_freq=-1
    --wandb.enable=false
)

step "起こす"
setsid nohup accelerate launch --num_processes=1 --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}" \
    > "$OUT/$RUN.nohup" 2>&1 < /dev/null &
echo "  PID $!"
echo "=== 動き出しを確かめる（最初の 20 歩まで待つ） ==="
for _ in $(seq 1 180); do
    grep -q "step:20 \|step:$STEPS " "$OUT/$RUN.nohup" 2>/dev/null && break
    kill -0 "$!" 2>/dev/null || { tail -n 30 "$OUT/$RUN.nohup"; die "落ちた" "上のログを読む"; }
    sleep 10
done
echo "  ログ: tail -f $OUT/$RUN.nohup | tr '\\r' '\\n'"
echo "  止める: kill -0 $! で生存確認 → kill -TERM $!"
