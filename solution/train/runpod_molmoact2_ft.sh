#!/usr/bin/env bash
# MolmoAct2-LIBERO-LeRobot を出発点に、手元で組んだ LeRobot データで追加学習する（RunPod）。
#
#   # 前提: runpod_setup.sh 済み・hf auth login 済み・データセットが HF に上がっている
#   export HF_HOME=/workspace/hf_home
#   hf download siruku6/pi05_code --local-dir /workspace/pi05_code
#   bash /workspace/pi05_code/solution/train/runpod_molmoact2_ft.sh
#
# **`train_pi05.py` は使わない。**あちらは `--policy.type=pi05` と π0.5 専用の引数を
# 固定で組み立てるので、molmoact2 は通らない。**このスクリプトは `lerobot-train` を直に叩く。**
#
# ## 既定値は「出発点の重みが作られた条件」に揃えてある
#
# 一次情報は `allenai/MolmoAct2-LIBERO-LeRobot` に同梱の `train_config.json`（系譜は
# [reference/38](../../docs/reference/38_lr_lineage.md)）。
#
# | 設定 | 出発点の重み | このスクリプトの既定 |
# |---|---|---|
# | 学習率（本体 / ViT / つなぎ / action expert） | 1e-5 / 5e-6 / 5e-6 / 5e-5 | 同じ |
# | optimizer | adamw・betas (0.9,0.95)・eps 1e-6・weight decay 0・grad clip 1.0 | 同じ（設定の既定値） |
# | scheduler | cosine・warmup 200・最終 1e-6 | 同じ |
# | 範囲 | 全パラメータ（LoRA 無し・`freeze_embedding=true`） | 同じ |
# | 形 | `action_mode=both`・bf16・勾配チェックポイント有効・chunk 10・flow 8 歩 | 同じ |
# | 画像を崩す細工 | ColorJitter 4 種＋SharpnessJitter＋RandomAffine（1 コマ最大 3 種） | **同じものを明示的に渡す**（`ORIG_TFS`） |
# | 歩数 × batch | 10,000 歩 × 32 × 8 枚（実効 256） | `STEPS`・`BATCH`・`GPUS` で指定 |
#
# **lerobot v0.6.0 に勾配累積は無い。**実効 batch は `BATCH × GPUS` だけである。
#
# ## このデータのために足しているもの（2 つとも外せない）
#
# 1. **`--rename_map`。** 出発点は `observation.images.image` / `observation.images.wrist_image` を
#    期待するが、こちらのデータの列は `front` / `wrist` である。`--policy.path` から始めるときだけ
#    使える引数で、保存済み前処理の rename 段に差し込まれる。
# 2. **`--policy.scheduler_decay_steps`。** チェックポイントの設定は `null` で、draccus が int に
#    読めずに落ちる（9/15 に評価側で踏んだのと同じ）。**歩数と同じ値を明示する。**
#
# ## 出発点の重みは 2 つ要る（既定は小さいほうだけ降ろす）
#
# `MolmoAct2Policy.__init__` は `checkpoint_path`（既定 `allenai/MolmoAct2-LIBERO`・fp32 21.8 GB）を
# 読んでから、`--policy.path` の重みで全部上書きする。**1 回目は捨てられる。**
# `CKPT_MODE=small`（既定）なら、前処理に要る小さいファイル（約 12 MB）だけを降ろし、
# `solution/tools/eval/lerobot_patches/modeling_molmoact2.py`（評価で使っている直し）を
# 入っている lerobot へ上書きして、1 回目の読み込みを飛ばす。`CKPT_MODE=full` なら素直に 21.8 GB 降ろす
# （そのとき直しは元に戻す）。
#
# **評価側は同じ直しをコンテナへ bind-mount している**（`run_molmoact2_stage0.sh`）。RunPod の学習は
# コンテナを挟まずに `accelerate launch` を直に叩くので、**ここだけは入っている lerobot を上書きする**。
# 使い捨ての pod なので元に戻せるようにしてある（`.orig`）。
#
# ## `--policy.path` を使うと必ず踏む 2 つのバグ（`lerobot_patches/lerobot_train.py` で両方直す）
#
# 1. **前処理ステップ名の不一致。** `lerobot_train.py` は前処理の上書きキーを
#    `normalizer_processor` / `unnormalizer_processor` に決め打ちしているが、MolmoAct2 は
#    自前の名前（`molmoact2_masked_normalizer` / `molmoact2_masked_unnormalizer`）で
#    登録している。直さないと `--policy.path` から学習しようとした瞬間に
#    `KeyError`（`_validate_overrides_used`）で **100% 落ちる**（CKPT_MODE に関係ない）。
# 2. **前処理側でも素の重みを二重に読んでしまう。** `MolmoAct2PackInputsProcessorStep`
#    は保存済みパイプラインの `checkpoint_path`（チェックポイントの publisher が使った値。
#    たいてい素の `allenai/MolmoAct2-LIBERO` のまま）から自分を再構築する。
#    `--policy.checkpoint_path`（＝上の 1 の話）の上書きはこのステップには効かないので、
#    **`CKPT_MODE=small` で 1 回目の重み読み込みを避けても、このステップだけは毎回
#    21.8 GB を裏でネットワークから取りに行ってしまう**（9/16 に実測・検証済み）。
#
# **どちらも lerobot_train.py 側の話であり、上の `modeling_molmoact2.py` の直しとは別物。**
# 詳細と検証手順は `lerobot_patches/lerobot_train.py` の先頭コメントを参照。
#
# ## 使い方
#
#   # 1. まず速さと VRAM を測る（上げ先を作らず、短く回して片付ける）
#   for b in 8 16 32; do
#     TRIAL=1 BATCH=$b bash /workspace/pi05_code/solution/train/runpod_molmoact2_ft.sh
#   done
#
#   # 2. 本番（歩数と batch を決めてから）
#   RUN=molmoact2_l90clean STEPS=6000 BATCH=16 SAVE_FREQ=2000 \
#     bash /workspace/pi05_code/solution/train/runpod_molmoact2_ft.sh
#
# Reads:
#   - HF `$DATA_REPO`（手元で組んだデータセット）と、その中の `bowl5_expected.json`
#   - HF `$INIT_REPO`@`$INIT_REV`（LeRobot 版の出発点）と `$BASE_REPO`（前処理用のファイル）
# Generates:
#   - $OUTPUT_ROOT/$RUN.nohup   学習のログ
#   - $W/upload_ckpt_$RUN.log   上げるループのログ（HF `$REPO` へ）
set -uo pipefail

W="${W:-/workspace}"
RUN="${RUN:-molmoact2_l90clean}"
REPO="${REPO:-siruku6/$RUN}"
DATA_REPO="${DATA_REPO:-siruku6/libero_combined_bowl5_initpose_l90clean}"
DATASET_ID="${DATASET_ID:-local/libero_combined_bowl5_initpose_l90clean}"
DS="${DS:-$W/data/libero_combined_bowl5_initpose_l90clean}"
INIT_REPO="${INIT_REPO:-allenai/MolmoAct2-LIBERO-LeRobot}"
INIT_REV="${INIT_REV:-f0c5a9567c2b72faadec901e16e055c8b098c2f5}"
# 前処理（tokenizer・norm_stats）の出どころ。`CKPT_MODE=full` のときは重みごと降ろす
BASE_REPO="${BASE_REPO:-allenai/MolmoAct2-LIBERO}"
BASE_REV="${BASE_REV:-0d24a92bd1faf321ef497c3bbd5681af97c65aa2}"
CKPT_MODE="${CKPT_MODE:-small}"
INIT_DIR="$W/init/molmoact2_libero_lerobot"
BASE_DIR="$W/init/molmoact2_base"
OUTPUT_ROOT="$W/out"
TOOLS="$W/tools"
CODE_DIR="${CODE_DIR:-$W/pi05_code/solution/train}"
PATCH="${PATCH:-$W/pi05_code/solution/tools/eval/lerobot_patches/modeling_molmoact2.py}"
TRAIN_PATCH="${TRAIN_PATCH:-$W/pi05_code/solution/tools/eval/lerobot_patches/lerobot_train.py}"
# 直しは lerobot v0.6.0 の同ファイルの写しなので、**別の版に当てると他の直しごと巻き戻る**
WANT_LEROBOT="${WANT_LEROBOT:-0.6.0}"
STEPS="${STEPS:-6000}"
BATCH="${BATCH:-16}"
GPUS="${GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
SEED="${SEED:-1000}"
WARMUP="${WARMUP:-200}"
LR="${LR:-1e-5}"
VIT_LR="${VIT_LR:-5e-6}"
CONNECTOR_LR="${CONNECTOR_LR:-5e-6}"
EXPERT_LR="${EXPERT_LR:-5e-5}"
DECAY_LR="${DECAY_LR:-1e-6}"
TRIAL="${TRIAL:-0}"
TRIAL_STEPS="${TRIAL_STEPS:-100}"
export HF_HOME="${HF_HOME:-$W/hf_home}"

# 出発点の重みが使っていた画像の崩し方（`train_config.json` の `image_transforms.tfs` そのもの）
ORIG_TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},'\
'"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},'\
'"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},'\
'"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},'\
'"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},'\
'"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'
RENAME_MAP='{"observation.images.front":"observation.images.image","observation.images.wrist":"observation.images.wrist_image"}'

if [ "$TRIAL" = 1 ]; then
    # **本番と同じ名前の出力先・上げ先に触らない**
    RUN="${RUN}_trial_b${BATCH}"
    STEPS=$(( TRIAL_STEPS + 1000 ))   # 止める前に終わらない長さ（終われば 11 GB の保存が起きる）
    SAVE_FREQ=$STEPS
fi
# 学習の生死を見る目印。`upload_ckpt.sh` にも同じものを渡す
ALIVE="lerobot_train.*--job_name=$RUN"

step() { echo; echo "=== $* ==="; }
die() { echo; echo "**止まった: $1**"; shift; for l in "$@"; do echo "  $l"; done; exit 1; }
# 学習を起こした後で止めるときは、切り離してあるので明示的に殺す
stop_training() { pkill -f "$ALIVE" 2>/dev/null; }

step "前提"
command -v hf >/dev/null || die "hf が無い" "runpod_setup.sh を先に打つ"
hf auth whoami >/dev/null 2>&1 || die "HF にログインしていない" "hf auth login"
command -v accelerate >/dev/null || die "accelerate が無い" "pip install 'accelerate>=1.14,<2'"
python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null || die "GPU が見えない"
LEROBOT_VERSION=$(python3 -c "
import lerobot, lerobot.policies.molmoact2.modeling_molmoact2  # noqa: F401
print(getattr(lerobot, '__version__', '?'))" 2>/dev/null) \
    || die "lerobot の molmoact2 を import できない" "runpod_setup.sh を先に打つ（lerobot v0.6.0）"
[ "${GPUS:-0}" -ge 1 ] || die "GPU の枚数を数えられない" "GPUS=1 のように渡す"
NUM_WORKERS="${NUM_WORKERS:-$(python3 -c "import os;print(max(2, min(os.cpu_count() // (2 * $GPUS), 8)))")}"
# **共有メモリが小さいと 1 歩も進まずに落ちる**（9/10 に π0.5 で踏んだ。`docs/reference/19`）。
# 読み手は枚数ぶん並ぶので、多 GPU ほど効く。RunPod の pod なら作るときに Shared Memory を増やす。
SHM_GB=$(df -BG /dev/shm 2>/dev/null | awk 'NR==2{gsub("G","",$2);print $2}')
[ "${SHM_GB:-0}" -ge 8 ] \
    || echo "  **/dev/shm が ${SHM_GB:-?} GB しかない。**読み手が落ちるおそれがある（8 GB 以上を推奨）"
[ -e "$OUTPUT_ROOT/$RUN" ] && die "出力先が既にある: $OUTPUT_ROOT/$RUN" "RUN を変えるか、消してから打つ"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  lerobot $LEROBOT_VERSION ／ $W の空き: $(df -h "$W" | awk 'NR==2{print $4}')（要る量は約 $([ "$CKPT_MODE" = full ] && echo 120 || echo 100) GB）"
echo "  実効 batch: $BATCH × $GPUS 枚 = $(( BATCH * GPUS ))（出発点の重みは 256）／ 読み手 $NUM_WORKERS"
mkdir -p "$W/data" "$OUTPUT_ROOT" "$TOOLS" "$INIT_DIR" "$BASE_DIR"

# **重いものを降ろす前に、落ちうる軽い検査を先に済ませる**（39 GB 降ろしてから落ちると pod 代が丸損）
step "道具をそろえる"
for f in upload_ckpt.sh verify_upload.py summarize_speed.py; do
    cp "$CODE_DIR/$f" "$TOOLS/$f" || die "$CODE_DIR/$f が無い" "hf download siruku6/pi05_code --local-dir $W/pi05_code"
done
# **`upload_ckpt.sh` は生存判定が `train_pi05.py` 決め打ちである。**リポジトリの共有物には手を入れず、
# ここに写した版だけを `$ALIVE_PATTERN` を見るように書き換える（元の既定の振る舞いは変えない）。
python3 - "$TOOLS/upload_ckpt.sh" <<'PY' || die "上げるループを molmoact2 用に直せない" "upload_ckpt.sh の版を確かめる"
import sys
path = sys.argv[1]
text = open(path).read()
old = 'training_alive() { pgrep -f "train_pi05.py --run-name $RUN" >/dev/null; }'
new = 'training_alive() { pgrep -f "${ALIVE_PATTERN:-train_pi05.py --run-name $RUN}" >/dev/null; }'
if old not in text:
    raise SystemExit(f"**{path} に生存判定の行が無い**（upload_ckpt.sh の版が変わった）")
open(path, "w").write(text.replace(old, new))
print("  生存判定を ALIVE_PATTERN で差し替えられるようにした（写しだけ）")
PY
chmod +x "$TOOLS/upload_ckpt.sh"
if [ "$CKPT_MODE" = small ]; then
    [ -f "$PATCH" ] || die "直しが無い: $PATCH" "CKPT_MODE=full で打つか、pi05_code を取り直す"
    grep -q "has_base_weights" "$PATCH" || die "直しの中身が違う（has_base_weights が無い）"
    [ "$LEROBOT_VERSION" = "$WANT_LEROBOT" ] || die \
        "直しは lerobot $WANT_LEROBOT の写しだが、入っているのは $LEROBOT_VERSION" \
        "当てると他の直しごと巻き戻る。CKPT_MODE=full で打つか、WANT_LEROBOT を確かめる"
fi
if [ "$TRIAL" != 1 ]; then
    python3 -c "
from huggingface_hub import HfApi
HfApi().create_repo('$REPO', private=True, exist_ok=True); print('  上げ先 $REPO')" \
        || die "上げ先を作れない" "HF の非公開枠を確かめる"
    NCKPT=$(( STEPS / SAVE_FREQ + (STEPS % SAVE_FREQ != 0) ))
    echo "  チェックポイント $NCKPT 個（約 $((NCKPT * 11)) GB を HF へ）"
fi

step "降ろす（出発点は裏で、データは目録から先に）"
( hf download "$INIT_REPO" --revision "$INIT_REV" --local-dir "$INIT_DIR" > "$W/dl_init.log" 2>&1; echo $? > "$W/dl_init.rc" ) &
BASE_INCLUDE=()
[ "$CKPT_MODE" = small ] && BASE_INCLUDE=(--include "config.json" "processor_config.json" "tokenizer.json" \
    "tokenizer_config.json" "chat_template.jinja" "norm_stats.json" "*.py")
( hf download "$BASE_REPO" --revision "$BASE_REV" --local-dir "$BASE_DIR" "${BASE_INCLUDE[@]}" \
    > "$W/dl_base.log" 2>&1; echo $? > "$W/dl_base.rc" ) &

# **目録（数十 KB）だけ先に降ろして突き合わせる。**取り違えていれば 27.8 GB を運ぶ前に分かる。
hf download "$DATA_REPO" --repo-type dataset --local-dir "$DS" \
    --include "meta/*" "bowl5_expected.json" > "$W/dl_meta.log" 2>&1 \
    || { tail -5 "$W/dl_meta.log"; die "データの目録を降ろせない" "$DATA_REPO が上がっているか確かめる"; }
[ -f "$DS/bowl5_expected.json" ] || die "$DS/bowl5_expected.json が無い" \
    "手元で組んだときの期待値を上げる（howto/19 の 2.1 節）"
read -r GOT WANT <<< "$(python3 - "$DS" <<'PY'
import json, sys
root = sys.argv[1]
info = json.load(open(f"{root}/meta/info.json"))
want = json.load(open(f"{root}/bowl5_expected.json"))
need = ["observation.images.front", "observation.images.wrist", "observation.state", "action"]
missing = [key for key in need if key not in info["features"]]
if missing:
    raise SystemExit(f"**列が足りない: {missing}**（--rename_map の前提が崩れている）")
keys = ("total_episodes", "total_frames", "total_tasks")
print("_".join(str(info[k]) for k in keys), "_".join(str(want[k]) for k in keys))
PY
)" || die "降りた目録を読めない" "$DS を消して打ち直す"
echo "  実際 ${GOT//_/ } ／ 期待 ${WANT//_/ }（列名も確認した）"
[ "$GOT" = "$WANT" ] || die "降りたデータが期待と違う" "$DS を消して打ち直す"
WANT_EPS=${GOT%%_*}

hf download "$DATA_REPO" --repo-type dataset --local-dir "$DS" > "$W/dl_data.log" 2>&1 \
    || { tail -5 "$W/dl_data.log"; die "データを降ろせない" "$DATA_REPO が上がっているか確かめる"; }
wait
[ "$(cat "$W/dl_init.rc")" = 0 ] || { tail -5 "$W/dl_init.log"; die "出発点を降ろせない" "$INIT_REPO@$INIT_REV"; }
[ "$(cat "$W/dl_base.rc")" = 0 ] || { tail -5 "$W/dl_base.log"; die "前処理用のファイルを降ろせない" "$BASE_REPO"; }
for f in config.json model.safetensors policy_preprocessor.json; do
    [ -f "$INIT_DIR/$f" ] || die "$INIT_DIR/$f が無い" "降ろした階層を ls で確かめる"
done

step "素の重みを読ませない直し（CKPT_MODE=$CKPT_MODE）"
TARGET=$(python3 -c "import lerobot,os;print(os.path.join(os.path.dirname(lerobot.__file__),'policies/molmoact2/modeling_molmoact2.py'))")
if [ "$CKPT_MODE" = small ]; then
    [ -f "$TARGET.orig" ] || cp "$TARGET" "$TARGET.orig" || die "元ファイルを控えられない"
    cp "$PATCH" "$TARGET" || die "直しを当てられない: $TARGET"
    python3 -c "import lerobot.policies.molmoact2.modeling_molmoact2" \
        || die "直しを当てたら import できなくなった" "cp $TARGET.orig $TARGET で戻す"
    echo "  当てた: $TARGET（元は $TARGET.orig）"
elif [ -f "$TARGET.orig" ]; then
    # **同じ pod を使い回すとき、前の回の直しが残っていると full の意味が無くなる**
    cp "$TARGET.orig" "$TARGET" && echo "  前の回の直しを戻した: $TARGET"
else
    echo "  直しは当てない（素の重みを読む）"
fi

step "--policy.path から読むときの直し（前処理ステップ名・二重ダウンロード。CKPT_MODE によらず要る）"
[ -f "$TRAIN_PATCH" ] || die "直しが無い: $TRAIN_PATCH" "pi05_code を取り直す"
grep -q "is_molmoact2" "$TRAIN_PATCH" || die "直しの中身が違う（is_molmoact2 が無い）"
[ "$LEROBOT_VERSION" = "$WANT_LEROBOT" ] || die \
    "直しは lerobot $WANT_LEROBOT の写しだが、入っているのは $LEROBOT_VERSION" \
    "当てると他の直しごと巻き戻る。WANT_LEROBOT を確かめる"
TRAIN_TARGET=$(python3 -c "import lerobot,os;print(os.path.join(os.path.dirname(lerobot.__file__),'scripts/lerobot_train.py'))")
[ -f "$TRAIN_TARGET.orig" ] || cp "$TRAIN_TARGET" "$TRAIN_TARGET.orig" || die "元ファイルを控えられない"
cp "$TRAIN_PATCH" "$TRAIN_TARGET" || die "直しを当てられない: $TRAIN_TARGET"
python3 -c "import lerobot.scripts.lerobot_train" \
    || die "直しを当てたら import できなくなった" "cp $TRAIN_TARGET.orig $TRAIN_TARGET で戻す"
echo "  当てた: $TRAIN_TARGET（元は $TRAIN_TARGET.orig）"

TRAIN_ARGS=(
    --policy.path="$INIT_DIR"
    --policy.checkpoint_path="$BASE_DIR"
    --policy.device=cuda
    --policy.action_mode=both
    --policy.model_dtype=bfloat16
    --policy.chunk_size=10
    --policy.n_action_steps=10
    --policy.num_flow_timesteps=8
    --policy.gradient_checkpointing=true
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
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms=3
    --dataset.image_transforms.tfs="$ORIG_TFS"
    --rename_map="$RENAME_MAP"
    --output_dir="$OUTPUT_ROOT/$RUN"
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
LAUNCH=(accelerate launch --num_processes="$GPUS" --mixed_precision=bf16 -m lerobot.scripts.lerobot_train)

if [ "$TRIAL" = 1 ]; then
    step "速さを測る（batch $BATCH × $GPUS 枚・$TRIAL_STEPS 歩で止める。上げ先は作らない）"
    setsid nohup "${LAUNCH[@]}" "${TRAIN_ARGS[@]}" > "$OUTPUT_ROOT/$RUN.nohup" 2>&1 &
    sleep 5
    PEAK=0
    # **待つ長さは測る対象から決める。**1 歩 120 秒でも $TRIAL_STEPS 歩ぶん待てるようにする
    for _ in $(seq 1 $(( TRIAL_STEPS * 12 ))); do
        grep -q "step:$TRIAL_STEPS " "$OUTPUT_ROOT/$RUN.nohup" 2>/dev/null && break
        pgrep -f "$ALIVE" >/dev/null || break
        M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
        [ "${M:-0}" -gt "$PEAK" ] && PEAK=$M
        sleep 10
    done
    stop_training
    for _ in $(seq 1 30); do pgrep -f "$ALIVE" >/dev/null || break; sleep 2; done
    if grep -q "step:$TRIAL_STEPS " "$OUTPUT_ROOT/$RUN.nohup" 2>/dev/null; then
        # **表は π0.5 と同じ道具で出す**（1 歩の秒数の定義とデータ待ちの警告を揃える）
        python3 "$TOOLS/summarize_speed.py" "$OUTPUT_ROOT/$RUN.nohup" \
            --batch $(( BATCH * GPUS )) --peak-mib "$PEAK"
    else
        echo "  **$TRIAL_STEPS 歩に届く前に止まった（batch $BATCH）。**"
        grep -m3 -iE "out of memory|OutOfMemoryError|Error" "$OUTPUT_ROOT/$RUN.nohup" | sed 's/^/    /'
        echo "  out of memory ならこの batch は載らない。小さい batch で打つ"
    fi
    rm -rf "${OUTPUT_ROOT:?}/$RUN" "$OUTPUT_ROOT/$RUN.nohup"
    echo
    echo "=== 試し走りを片付けた。本番は TRIAL を外し、STEPS・BATCH・SAVE_FREQ を決めて打つ ==="
    exit 0
fi

step "学習を起こす（出発点 $INIT_REPO・全パラメータ・$STEPS 歩・batch $BATCH × $GPUS 枚）"
echo "  学習率 本体 $LR ／ ViT $VIT_LR ／ つなぎ $CONNECTOR_LR ／ action expert $EXPERT_LR（warmup $WARMUP・最終 $DECAY_LR）"
setsid nohup "${LAUNCH[@]}" "${TRAIN_ARGS[@]}" > "$OUTPUT_ROOT/$RUN.nohup" 2>&1 &
sleep 5
pgrep -f "$ALIVE" >/dev/null \
    || { tail -20 "$OUTPUT_ROOT/$RUN.nohup"; die "学習が立ち上がらない"; }

# ログは lerobot が標準出力へ書くので `$RUN.nohup` に溜まる。`upload_ckpt.sh` は
# `<TRAIN_LOG_DIR>/<RUN>.log` と `.nohup` の両方を探し、片方あれば上げる（写しは作らない）。
ALIVE_PATTERN="$ALIVE" RUN="$RUN" REPO="$REPO" LOG="$W/upload_ckpt_$RUN.log" \
    OUT="$OUTPUT_ROOT/$RUN" TRAIN_LOG_DIR="$OUTPUT_ROOT" VERIFY="$TOOLS/verify_upload.py" \
    setsid nohup "$TOOLS/upload_ckpt.sh" > "$W/upload_ckpt_$RUN.nohup" 2>&1 &

step "動き出しを確かめる（最初の 20 歩まで待つ）"
for _ in $(seq 1 240); do
    grep -q "step:20 " "$OUTPUT_ROOT/$RUN.nohup" 2>/dev/null && break
    pgrep -f "$ALIVE" >/dev/null \
        || { tail -30 "$OUTPUT_ROOT/$RUN.nohup"; die "学習が途中で落ちた"; }
    sleep 10
done
GOT_DSEPS=$(grep -m1 -o "dataset.num_episodes=[0-9]*" "$OUTPUT_ROOT/$RUN.nohup" | cut -d= -f2)
grep -m1 "Effective batch size" "$OUTPUT_ROOT/$RUN.nohup" | sed 's/^/  /'
printf '  %-24s %-12s %s\n' 見たもの 実際 期待 dataset.num_episodes "${GOT_DSEPS:-?}" "$WANT_EPS"
# **die しても学習は切り離してあるので止まらない。**間違ったデータで GPU 時間と HF の枠を食う前に殺す
[ "${GOT_DSEPS:-0}" = "$WANT_EPS" ] || { stop_training; die "読んでいるデータが違う（${GOT_DSEPS:-?} 本）"; }
grep -q "molmoact2" "$OUTPUT_ROOT/$RUN.nohup" || echo "  **ログに molmoact2 の行が無い。**方策の種類を確かめること"
grep -q "開始 RUN=$RUN" "$W/upload_ckpt_$RUN.log" 2>/dev/null || echo "  **上げるループの開始行がまだ無い。**tail $W/upload_ckpt_$RUN.nohup を見る"

UPDT=$(grep -m1 -o "updt_s:[0-9.]*" "$OUTPUT_ROOT/$RUN.nohup" | cut -d: -f2)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU メモリ（20 歩の時点）: /'
echo
echo "=== 走っている ==="
[ -n "$UPDT" ] && python3 -c "
s = $UPDT * $STEPS
print(f'  終わる目安: {int(s//3600)} 時間 {int(s%3600//60)} 分（1 歩 $UPDT 秒 × $STEPS 歩）')"
echo "  ログ: tail -f $OUTPUT_ROOT/$RUN.nohup"
echo "  上げ: tail -f $W/upload_ckpt_$RUN.log"
