#!/usr/bin/env python3
"""SmolVLA の追加学習を、Colab を使わずに回す。

**既定値は「出発点の重みが作られた条件」に揃えてある**（2026-08-11 に変更）。
出発点は `model_weights.bak/train_config.json` を読めば分かり、
**optimizer と scheduler は LeRobot の既定値そのもの**だった。

| 設定 | 出発点の重み | いまの既定 | 8/11 までの既定 |
|---|---|---|---|
| データ | **全 14,347 本**（`episodes=None`） | **全部** | 200 本（40 タスク × 5） |
| 歩数 × バッチ | 20,000 × 32 | 20,000 × 32 | 20,000 × **8** |
| `optimizer_lr` | 1e-4 | 1e-4 | **3e-4** |
| `scheduler_warmup_steps` | 1,000 | 1,000 | **100** |
| `scheduler_decay_lr` | 2.5e-6 | 2.5e-6 | **3e-5** |
| 画像拡張 | **無効** | **無効** | **有効** |

**8/11 までの既定で 20,000 歩まわしたら、本番スコアが 0.1409 から 0.016 へ
落ちた**（8.8 倍の悪化）。

**上の表のうち、一次情報で裏が取れるのは学習率・warmup・最後の学習率・歩数だけである。**
それらは重みの `config.json` に残る（`optimizer_lr=3e-4`、
`scheduler_warmup_steps=100`、`scheduler_decay_steps=20000`）。
**データの本数とバッチ数は残らない。** あの回は学習マシンから `merged_model/` だけを
持ち帰ったため、`train_config.json` が失われており、
**「1 コマを 5.1 回見た」という数字は確かめられない。**
一次情報だけで言えることは
[docs/reference/11_training_schedule.md](../../docs/reference/11_training_schedule.md)
に分けて置いた。**この docstring より、そちらを信用すること。**
（`merge.py` は 8/11 以降、学習の条件を `merged_model/` へ同梱する。）

**LoRA は既定で有効のままにしてある。** LeRobot が LoRA を当てるのは
action expert の q/v 射影と 5 つの射影層だけで、
**`freeze_vision_encoder` と `train_expert_only` を変えても学習される重みは
変わらない**（`modeling_smolvla.py` の `_get_default_peft_targets`）。
視覚側まで学ばせたいときは `--peft.target_modules` を自分で渡すか、
LoRA を外して全パラメータ学習にする。

Colab と違い、出力はホストのディレクトリに残るため Drive 同期は要らない。
中断しても `--resume` で続きから始められる。

    python train.py                          # 出発点と同じ条件で 20,000 歩
    python train.py --steps 5000             # まず短く回して速さを測る
    python train.py --batch-size 16          # VRAM が足りないとき
    python train.py --episodes-per-task 5    # 少ないデータで試す（**悪化の再現**）
    python train.py --resume                 # 直近のチェックポイントから

    # LoRA を VLM の q/v まで広げ、歩数を 3 倍にする（8/12 の本命）
    python train.py --run-name vlm_lora32_60k --peft-scope vlm \
        --lora-r 32 --lora-alpha 32 --steps 60000

    # 苦手タスクの割合を 3 倍にし、300 歩を超える手本を落とす（8/11 の本命）
    python train.py --run-name weak3x_max300 --max-demo-steps 300 \
        --emphasize-tasks \
          "put the bowl on top of the cabinet" \
          "put both the alphabet soup and the cream cheese box in the basket" \
          "pick up the black bowl next to the ramekin and place it on the plate"

Reads:
  - lerobot/smolvla_libero_plus（HF、初期重み）
  - lerobot/libero_plus（HF、学習データ。**全部使うと 16 GB 前後**）
Generates:
  - $OUTPUT_ROOT/<run-name>/checkpoints/**（チェックポイント）
  - $OUTPUT_ROOT/<run-name>_episodes.json（使ったエピソードの記録）
  - $OUTPUT_ROOT/<run-name>/episodes.json（学習が始まったあとに上の複製を置く）

**記録を出力先の外へ先に書くのは、LeRobot が既存のディレクトリを拒否するためである**
（`configs/train.py` の `validate` が `not resume and output_dir.is_dir()` で
`FileExistsError` を投げる。**空でも拒否する。**）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

_BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
_BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"
_VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
_DATASET_REPO = "lerobot/libero_plus"
_DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"

_LEROBOT_DIR = Path(os.environ.get("LEROBOT_DIR", "/opt/lerobot"))
_BRIGHTNESS_COLUMN = "stats/observation.images.front/mean"

# **`--peft-scope vlm` のときに LoRA を当てる先。**
#
# LeRobot の既定（`_get_default_peft_targets`）は `lm_expert` の q/v と下の 5 射影
# だけで、**VLM 350M（全体の 77.8%）に 1 つも当たらない。** 当て先を `lm_expert`
# から `(lm_expert|vlm)` へ広げると、次の 3 か所すべてに当たる。
#
#   model.vlm_with_expert.lm_expert.layers.N.self_attn.(q|v)_proj              16 層 ×2
#   model.vlm_with_expert.vlm.model.text_model.layers.N.self_attn.(q|v)_proj   16 層 ×2  ← 言語
#   model.vlm_with_expert.vlm.model.vision_model.encoder.layers.N.…            12 層 ×2  ← 視覚
#
# 学習される重みは 37 層 742,656 個（0.17%）から **93 層 4,303,360 個（r=32 で
# 0.96%）** になる。**この数は起動直後の `num_learnable_params=` で確かめられる。**
#
# 根拠は 2 つ。π0 で当て先を切り分けた研究（arXiv:2607.10172）は VLM を凍結すると
# 成功率 0.76 → 0.15、視覚エンコーダの凍結で 0.14 まで落ちるとし、「r=16〜32 を
# VLM と action expert に均等に当てよ」を推奨している。LIBERO-plus で測った
# VLA-GSE（arXiv:2605.06175）は逆に全パラメータ学習が忘却で PEFT に負ける
# （74.9%）とし、**2.51% を学習する帯**を良しとしている。0.17% はどちらの
# 推奨帯よりも小さい。
#
# **視覚エンコーダを LoRA でなく全学習させたいときは `--peft.full_training_modules`**
# を `--` の後ろから渡す（研究では 0.43 → 0.74 と、ここがいちばん大きく動く）。
_PEFT_COMMON_PROJECTIONS = (
    "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
)
_PEFT_TARGETS_VLM = (
    rf"(model\.vlm_with_expert\.(lm_expert|vlm)\..*\.(q|v)_proj"
    rf"|model\.({_PEFT_COMMON_PROJECTIONS}))"
)

# **視覚エンコーダを外した版。** 20 GB では上の `vlm` が入らなかった（8/12、
# 1 歩も進まず OutOfMemory）。**入らない理由は当て先の数ではなく、順番である。**
#
#   視覚エンコーダ ──→ 言語（VLM の 16 層）──→ action expert ──→ 損失
#
# **学習する層から損失までの経路にある中間結果は、全部持ち続けることになる。**
# 視覚を当て先にすると、そこへ勾配を返すために言語の 16 層も通るので、
# **言語を外しても減らない。** 逆に視覚を外すと、逆伝播は言語の入り口で止まり、
# 視覚エンコーダの中間結果を丸ごと捨てられる。
#
# 量の差も大きい。バッチ 32 のとき、視覚は 512×512 の画像 2 台ぶんを 1,024
# トークンずつ 12 層通す（65,536 トークン位置 × 12）のに対し、言語は連結後
# 178 トークンを 16 層通すだけ（5,696 × 16）で、**視覚は言語の 8.6 倍**にあたる。
_PEFT_TARGETS_TEXT = (
    rf"(model\.vlm_with_expert\.(lm_expert|vlm\.model\.text_model)\..*\.(q|v)_proj"
    rf"|model\.({_PEFT_COMMON_PROJECTIONS}))"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("OUTPUT_ROOT", "/out")),
                        help="出力の置き場（既定: $OUTPUT_ROOT または /out）")
    parser.add_argument("--run-name", type=str, default="smolvla_lora_all_suites",
                        help="実行の名前。再開時は同じ名前を指定する")
    parser.add_argument("--steps", type=int, default=20000,
                        help="学習ステップ数。学習率はこの値をかけて減衰する（出発点も 20,000）")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="出発点と同じ 32。VRAM が足りなければ 16 へ下げる")
    parser.add_argument("--episodes-per-task", type=int, default=0,
                        help="1 タスクあたりのエピソード数。**0 なら全 14,347 本を使う**"
                             "（出発点がそうである）。転送量は本数に比例しない —— "
                             "動画が共有ファイルに詰まっているため、200 本でも 6.4 GB 落ちる")
    # **苦手タスクの割合を上げる。** 84 件の検証用データで切り分けたら、
    # 落としているのは摂動ではなく特定のタスクだった（照明 +0.21・背景 +0.16 で
    # 見た目の摂動は「易しい側」）。**苦手タスクにも 370〜445 本あり、
    # 1 コマ 0.29 回しか引いていないので 7 割は一度も見られていない。**
    parser.add_argument("--emphasize-tasks", type=str, nargs="*", default=None,
                        metavar="INSTRUCTION",
                        help="割合を上げるタスクの指示文（**完全一致**）。"
                             "例: --emphasize-tasks \"put the bowl on top of the cabinet\"")
    parser.add_argument("--emphasis-multiplier", type=float, default=3.0,
                        help="--emphasize-tasks の割合を何倍にするか（既定 3 倍）。"
                             "**得意タスクを間引いて実現する** —— LeRobot は"
                             "エピソード番号を集合として扱うため、重複させても効かない")
    parser.add_argument("--max-demo-steps", type=int, default=0,
                        help="手本がこの歩数を超えるエピソードを使わない。**0 なら落とさない。**"
                             "300 を渡すと 401 本（フレームの 6.3%%）が落ちる —— "
                             "**上限 300 歩の採点では再現できないことが証明済み**である"
                             "（成功した回は手本の 0.95 倍の歩数で終わっている）")
    # **検証専用にするタスクを学習から外す。** 摂動版は元のタスクの実演を
    # 共有するため、変種の id を抜いても同じ指示文の別の変種が学習に残る。
    # **指示文で丸ごと外さないと、手元の検証は「覚えた度合い」を測ってしまう。**
    # 一覧は solution/tools/make_holdout_split.py が作り、
    # **検証集合も同じ CSV を読む**（docs/howto/04_proxy_eval.md）。
    parser.add_argument("--holdout-tasks", type=Path, default=None,
                        metavar="CSV",
                        help="このCSVの instruction 列にあるタスクを学習から外す。"
                             "**データセットに無い指示文があればその場で止める**")
    # **台本の手本を混ぜたデータセットで学習するための口。**
    # 既定では番号選びが `lerobot/libero_plus` のメタデータを Hub から落とすので、
    # **混ぜたデータセットとエピソード番号が対応しない**（足した分が丸ごと
    # 学習に入らないまま、黙って通ってしまう）。ここを渡すと
    # **番号選びと学習の両方が同じディレクトリを読む。**
    parser.add_argument("--dataset-root", type=Path, default=None,
                        metavar="DIR",
                        help="混ぜたデータセットの場所。**番号選び（--emphasize-tasks / "
                             "--max-demo-steps / --episodes-per-task）と学習の両方が"
                             "ここを読む。** 渡さなければ Hub の lerobot/libero_plus。"
                             "**渡したら `meta/episodes/*.parquet` の本数が"
                             "画面に出るので、足した分が入っているか必ず数える**")
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2,
                        help="データ供給のプロセス数。**CPU のコア数に合わせる。**"
                             "多すぎると取り合いになる（評価マシンは 2 コア）")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="出発点と同じ 1e-4。LeRobot の既定でもある")
    parser.add_argument("--final-learning-rate", type=float, default=2.5e-6,
                        help="減衰後の学習率。出発点と同じ 2.5e-6")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="出発点と同じ 1,000")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    # **LoRA をどこに当てるか。rank より当て先のほうが効く。**
    # LeRobot の既定（`expert`）は action expert の q/v と 5 つの射影だけで、
    # **450,046,176 個のうち 742,656 個＝ 0.17% しか学ばない。**
    # 出発点の重みは逆に 450M を全部学習して作られている
    # （`train_config.json` が `use_peft: false` `freeze_vision_encoder: false`）。
    parser.add_argument("--peft-scope", choices=("expert", "text", "vlm"),
                        default="expert",
                        help="LoRA の当て先。**expert は action expert だけ**"
                             "（LeRobot の既定、0.17%）。**text は言語も当てる**"
                             "（r=32 で 0.69%）。**vlm は視覚も当てる**（0.96%。"
                             "ただし 20 GB では入らなかった）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-transforms", action="store_true",
                        help="画像拡張を有効にする。**既定は無効で、出発点もそうだった。**"
                             "評価側は見た目の摂動なので効きそうに見えるが、"
                             "学習データ自体が明るさ 0.086〜0.723 を張っているため"
                             "**別の回として分けて測る**")
    parser.add_argument("--video-backend", type=str, default="torchcodec",
                        choices=("torchcodec", "pyav"),
                        help="動画のデコーダ。torchcodec が torch と対で壊れたときの逃げ道")
    parser.add_argument("--resume", action="store_true",
                        help="直近のチェックポイントから再開する")
    # **`--` の後ろは `lerobot-train` へそのまま渡す。**
    # ここを用意しないと、この台本が旗を持たない設定（拡張の強さなど）に
    # 手が届かない。旗を 1 つずつ足していくよりも、通し口を 1 つ開けるほうがよい。
    #
    #   python train.py --image-transforms -- \
    #       --dataset.image_transforms.max_num_transforms=6
    parser.add_argument("extra", nargs="*", metavar="-- ...",
                        help="`--` の後ろに書いたものを lerobot-train へそのまま渡す")
    return parser.parse_args()


def _patch_dtype_for_gpu() -> None:
    """bf16 非対応の GPU では VLM の読み込みを fp16 に書き換える。

    配布ノートブックが同じ置換をしている。揃えないと、学習を回した GPU に
    よってモデルの挙動が変わってしまう。Ampere 以降なら何もしない。

    `torch.cuda.is_bf16_supported()` が GPU の無い環境で何を返すかは torch の
    版によって違う（内部で `current_device()` を呼ぶ実装では例外になる）。
    版に依存しないよう、先に `is_available()` で切る。
    """
    if not torch.cuda.is_available():
        print("[train] GPU が無いため dtype の書き換えはしない")
        return

    if torch.cuda.is_bf16_supported():
        print("[train] bf16 対応の GPU。書き換えは不要")
        return

    target = _LEROBOT_DIR / "src/lerobot/policies/smolvla/smolvlm_with_expert.py"
    source = target.read_text(encoding="utf-8")
    if 'torch_dtype="bfloat16",' not in source:
        print("[train] 既に fp16 へ書き換え済み")
        return

    target.write_text(
        source.replace('torch_dtype="bfloat16",', 'torch_dtype="float16",', 1),
        encoding="utf-8",
    )
    print("[train] bf16 非対応の GPU のため fp16 へ書き換えた")


def _flatten_numbers(value: Any) -> list[float]:
    """入れ子になった配列を、平坦な float のリストにする。

    parquet の統計値は array([array([array([0.35])])]) のように object dtype で
    三重に入れ子になっており、np.mean がスカラーを返さない。再帰的にほどく。
    """
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    return [number for item in value for number in _flatten_numbers(item)]


def _load_episode_metadata(dataset_root: Path) -> pd.DataFrame:
    """メタデータの parquet を読み、タスク名と明るさの列を足して返す。

    LeRobot の `load_episodes()` は `stats/` で始まる列を落とすため、
    `LeRobotDatasetMetadata` 経由では明るさが取れない。parquet を直接読む。
    """
    files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"メタデータが見つかりません: {dataset_root}")

    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["task"] = frame["tasks"].apply(
        lambda cell: cell if isinstance(cell, str) else str(cell[0])
    )
    frame["brightness"] = [
        float(np.mean(_flatten_numbers(cell))) for cell in frame[_BRIGHTNESS_COLUMN]
    ]
    return frame


def _choose_spread_by_value(
    episode_indices: list[int], count: int, value_by_episode: dict[int, float]
) -> list[int]:
    """値で並べ替えてから、両端を含めて等間隔に count 本選ぶ。"""
    ordered = sorted(episode_indices, key=lambda index: value_by_episode[index])

    if count >= len(ordered):
        return list(ordered)
    if count <= 1:
        return [ordered[len(ordered) // 2]]

    positions = [round(rank * (len(ordered) - 1) / (count - 1)) for rank in range(count)]
    return [ordered[position] for position in positions]


def _select_episodes(dataset_root: Path, episodes_per_task: int,
                     holdout: list[str]) -> list[int]:
    """全 40 タスクから、明るさで散らしてエピソードを選ぶ。

    Parameters
    ----------
    dataset_root : Path
        メタデータを取得済みのデータセットのルート。
    episodes_per_task : int
        1 タスクあたりの本数。

    holdout : list[str]
        検証専用にするタスクの指示文。空なら何も外さない。
    Returns
    -------
    episode_indices : list[int]
        選んだエピソード番号の昇順リスト。
    """
    frame = _load_episode_metadata(dataset_root)
    frame = _drop_holdout(frame, holdout)
    brightness = dict(zip(frame["episode_index"].astype(int), frame["brightness"]))

    selected: list[int] = []
    for task_name, group in frame.groupby("task"):
        indices = group["episode_index"].astype(int).tolist()
        selected.extend(_choose_spread_by_value(indices, episodes_per_task, brightness))

    selected = sorted(selected)
    chosen = [brightness[index] for index in selected]
    whole = list(brightness.values())

    print(f"[train] {frame['task'].nunique()} タスク × {episodes_per_task} 本 "
          f"= {len(selected)} エピソード")
    print(f"[train]   選んだ分の明るさ {min(chosen):.3f} 〜 {max(chosen):.3f}")
    print(f"[train]   データセット全体 {min(whole):.3f} 〜 {max(whole):.3f}")
    return selected


def _check_task_names(frame: pd.DataFrame, names: list[str]) -> None:
    """渡したタスク名がデータセットに実在することを確かめる。

    **綴りが 1 文字違うと、黙って 1 本も選ばれない。** 指示文は 40 種類しか
    ないので、外れていれば候補を出して止める。
    """
    known = set(frame["task"])
    missing = [name for name in names if name not in known]
    if not missing:
        return

    lines = [f"データセットに無いタスク名がある: {missing}"]
    for name in missing:
        head = name.split()[0] if name.split() else name
        similar = sorted(task for task in known if head and head in task)
        if similar:
            lines.append(f"  「{name}」に近いもの: {similar[:3]}")
    lines.append(f"  指示文は全部で {len(known)} 種類しかない（完全一致で渡すこと）")
    raise SystemExit("\n".join(lines))


def _select_emphasized_episodes(
    dataset_root: Path, emphasize: list[str], multiplier: float,
    max_demo_steps: int, draws: int, holdout: list[str],
) -> list[int]:
    """苦手タスクの割合を上げ、長すぎる手本を落としてエピソードを選ぶ。

    **引ける回数が足りないのであって、データは足りていない。**
    20,000 歩 × バッチ 32 = 640,000 回しか引けず、全体は約 219 万フレームなので
    **1 コマあたり 0.29 回**しか見られない。**苦手タスクにも 370〜445 本あり、
    その 7 割は一度も見られていない。** だから「苦手タスクの、まだ見ていない
    コマ」を増やすのは筋が通る。

    **エピソード番号を重複させて重みを付ける道は使えない。** LeRobot は
    `episode_index.isin(episodes)` で**集合として**扱う（`datasets/io_utils.py`）。
    **だから得意タスクを間引いて割合を上げる。**

    `max_demo_steps` を超える手本は落とす。**上限 300 歩の採点では再現できない
    ことが証明済み**で（成功した回は手本の 0.95 倍の歩数で終わっている）、
    学習枠を使う意味がないうえ、遅い動きを教えてしまう。

    Parameters
    ----------
    dataset_root : Path
        メタデータを取得済みのデータセットのルート。
    emphasize : list[str]
        割合を上げるタスクの指示文。**完全一致で渡す。**
    multiplier : float
        そのタスクの割合を何倍にするか。
    max_demo_steps : int
        手本の歩数の上限。**0 なら落とさない。**
    draws : int
        学習で引く回数（歩数 × バッチ）。1 コマを何回見るかの表示に使う。

    holdout : list[str]
        検証専用にするタスクの指示文。空なら何も外さない。
    Returns
    -------
    episode_indices : list[int]
        選んだエピソード番号の昇順リスト。
    """
    frame = _load_episode_metadata(dataset_root)
    frame = _drop_holdout(frame, holdout)
    _check_task_names(frame, emphasize)

    total_before = len(frame)
    if max_demo_steps > 0:
        dropped = frame[frame["length"] > max_demo_steps]
        frame = frame[frame["length"] <= max_demo_steps]
        gone = sorted(set(dropped["task"]) - set(frame["task"]))
        print(f"[train] 手本が {max_demo_steps} 歩を超える {len(dropped)} 本を落とした"
              f"（全 {total_before} 本の {len(dropped) / total_before:.1%}）")
        if gone:
            print(f"[train]   丸ごと落ちたタスク {len(gone)} 種: {gone}")

    brightness = dict(zip(frame["episode_index"].astype(int), frame["brightness"]))
    length = dict(zip(frame["episode_index"].astype(int), frame["length"].astype(int)))

    wanted = set(emphasize)
    strong = frame[~frame["task"].isin(wanted)]
    weak_indices = sorted(frame[frame["task"].isin(wanted)]["episode_index"].astype(int))

    weak_frames = sum(length[index] for index in weak_indices)
    strong_frames = int(strong["length"].sum())
    share_now = weak_frames / (weak_frames + strong_frames)
    share_target = min(multiplier * share_now, 0.9)
    if share_target != multiplier * share_now:
        print(f"[train] **倍率が高すぎるので {share_target / share_now:.1f} 倍で打ち切った**")

    # 目標の割合になるよう、得意タスクを残す比率を解く。
    # **強調するタスクが無いときは間引かない。** そのまま計算すると
    # `weak_frames` も `share_target` も 0 になり `0.0 / 0.0` で
    # ZeroDivisionError になる。**`--max-demo-steps` だけを渡す使い方で踏む**
    # （手本の長さで落とすだけにしたいときは、強調は要らない）。
    if wanted:
        keep_ratio = weak_frames * (1.0 - share_target) / (share_target * strong_frames)
        keep_ratio = min(keep_ratio, 1.0)
    else:
        keep_ratio = 1.0

    selected = list(weak_indices)
    for _, group in strong.groupby("task"):
        indices = group["episode_index"].astype(int).tolist()
        count = max(1, round(len(indices) * keep_ratio))
        selected.extend(_choose_spread_by_value(indices, count, brightness))

    selected = sorted(selected)
    chosen_frames = sum(length[index] for index in selected)
    chosen_brightness = [brightness[index] for index in selected]
    whole_brightness = list(brightness.values())

    if wanted:
        print(f"[train] 苦手 {len(wanted)} タスクを {multiplier:g} 倍にする"
              f"（割合 {share_now:.1%} → {weak_frames / chosen_frames:.1%}）")
        print(f"[train]   得意タスクは {keep_ratio:.0%} に間引いた")
    else:
        print("[train] 強調するタスクの指定が無いので、割合はいじらない"
              "（手本の長さで落とすだけ）")
    print(f"[train]   選んだのは {len(selected):,} 本 / {chosen_frames:,} フレーム")
    # **引く回数も出す。** `--steps 1` の下見だと 0.00 回になり、
    # 「選び方が壊れている」と読み違える。分子を見せれば理由が分かる。
    print(f"[train]   引く回数 {draws:,} 回 / {chosen_frames:,} フレーム"
          f" = **1 コマを見る回数 {draws / chosen_frames:.2f} 回**"
          "（4 回目の提出は 5.1 回で本番 8.8 倍悪化）")
    print(f"[train]   選んだ分の明るさ {min(chosen_brightness):.3f} 〜 "
          f"{max(chosen_brightness):.3f}")
    print(f"[train]   落とす前の全体   {min(whole_brightness):.3f} 〜 "
          f"{max(whole_brightness):.3f}")
    return selected


def _find_resume_checkpoint(output_dir: Path) -> Path | None:
    """再開に使う最新のチェックポイントを返す。無ければ None。

    ディレクトリ名はゼロ埋めされるが桁数は steps に依存するため、
    文字列ではなく数値として並べる。
    """
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return None

    checkpoints = sorted(
        (path for path in root.glob("*") if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    return checkpoints[-1] if checkpoints else None


def _download_base_model() -> Path:
    """初期重みをこの環境のキャッシュへ取り、その場所を返す。

    revision を commit で固定しているので、同じ場所に既にあれば取り直さない。
    評価用の重み（`eval/`）は学習に要らないので除く。
    """
    return Path(snapshot_download(
        _BASE_MODEL_REPO, revision=_BASE_MODEL_REVISION,
        ignore_patterns=["eval/*", "eval/**"],
    ))


def _repair_base_model_path(resume_from: Path) -> None:
    """チェックポイントに焼き付いた初期重みの場所を、この環境の場所へ直す。

    LoRA のアダプタは「どの重みへの差分か」を**絶対パスで**覚えている。Colab で
    作ったチェックポイントには Colab の `HF_HOME`（`/content/hf_cache/...`）が
    入っており、コンテナ側の `HF_HOME`（`/root/.cache/huggingface`）では
    存在しない場所になる。

    LeRobot はこの値を初期重みの指定としてそのまま使い（policies/factory.py の
    `make_policy()`）、`from_pretrained()` は `os.path.isdir()` が偽なら
    **HF のリポジトリ名として解釈する。** 結果こう落ちる。

        HFValidationError: Repo id must be in the form 'repo_name' or
        'namespace/repo_name': '/content/hf_cache/hub/models--...'

    revision を指定して重みを取り直し、その場所へ書き換える。**モデルは同じもので、
    変わるのは場所の文字列だけである。**

    `config.json` と `train_config.json` にも同じパスが入っているが、再開時は
    `_resolve_resume_checkpoint()`（configs/train.py）が `policy.pretrained_path`
    をチェックポイント自身のパスへ代入するため読まれない。**実際に読まれる
    `adapter_config.json` だけを直す。**

    Parameters
    ----------
    resume_from : Path
        再開に使うチェックポイント（`<step>` のディレクトリ）。
    """
    config_path = resume_from / "pretrained_model" / "adapter_config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    recorded = config.get("base_model_name_or_path") or ""
    if Path(recorded).is_dir():
        return

    # 別のモデルを指していたら書き換えない。ここは「同じ重みの置き場所が
    # 違うだけ」を直す処理であって、モデルを差し替える処理ではない。
    # HF のキャッシュのパスには必ず models--<repo> が入るので、それで見分ける。
    marker = _BASE_MODEL_REPO.replace("/", "--")
    if marker not in recorded:
        raise RuntimeError(
            f"初期重みの場所が {_BASE_MODEL_REPO} 以外を指しています: {recorded!r}\n"
            "自動では直せないので、adapter_config.json を手で直すこと。"
        )

    base_model = str(_download_base_model())
    config["base_model_name_or_path"] = base_model
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[train] 初期重みの場所を書き換えた（{config_path.name}）")
    print(f"[train]   前: {recorded}")
    print(f"[train]   後: {base_model}")


def _build_resume_command(
    args: argparse.Namespace, output_dir: Path, resume_from: Path
) -> list[str]:
    """再開用の引数を組み立てる。**最小限しか渡さない。**

    再開時は学習条件がチェックポイントの `train_config.json` から復元される。
    ここで `--policy.*` や `--dataset.*` を重ねて渡すと、元の条件を上書きして
    しまう。特に `--policy.path` は `_resolve_pretrained_from_cli()` が
    優先するため、渡すと再開そのものが黙って無視される。

    例外は `--output_dir` である。チェックポイントの config には**元の環境の
    パス**（Colab なら `/content/outputs/...`）が入っているため、明示的に
    上書きしないと存在しない場所へ書こうとする。

    バッチサイズとワーカー数は学習条件を壊さないので渡してよい。
    バッチサイズを変えるとデータ順の再現性は落ちるが、LeRobot は警告を出すだけで
    エポックとオフセットは正しく復元する。

    `--steps` は渡さない。学習率は `scheduler_decay_steps` で決まっており、
    ここを変えるとスケジュールが途中で別物になる。
    """
    return [
        "lerobot-train",
        "--resume=true",
        f"--config_path={resume_from / 'pretrained_model' / 'train_config.json'}",
        f"--output_dir={output_dir}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--persistent_workers={'true' if args.num_workers > 0 else 'false'}",
        f"--save_freq={args.save_freq}",
    ] + list(args.extra)


def _build_command(
    args: argparse.Namespace,
    base_model: Path,
    episode_indices: list[int] | None,
    output_dir: Path,
) -> list[str]:
    """最初から学習するときの引数を組み立てる。

    Parameters
    ----------
    episode_indices : list[int] | None
        使うエピソード番号。**None なら `--dataset.episodes` を渡さない。**
        LeRobot はそのとき全エピソードを使う（出発点の `train_config.json` も
        `episodes: None` である）。

        **全部使うときに番号を並べてはいけない。** 14,347 個の番号を並べると
        コマンド 1 行が 100 KB を超える。旗を省くのが正しい。
    """
    command = [
        "lerobot-train",
        f"--policy.path={base_model}",
        f"--policy.vlm_model_name={_VLM_REPO}",
        "--policy.push_to_hub=false",
        "--policy.repo_id=null",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        f"--policy.optimizer_lr={args.learning_rate}",
        f"--policy.scheduler_decay_lr={args.final_learning_rate}",
        f"--policy.scheduler_warmup_steps={args.warmup_steps}",
        f"--policy.scheduler_decay_steps={args.steps}",
        f"--dataset.repo_id={_DATASET_REPO}",
        f"--dataset.revision={_DATASET_REVISION}",
        # **番号選びと学習は同じデータセットを見なければならない。**
        # `--dataset-root` を渡したときだけ足す（`revision` は Hub から取る
        # ときしか見られないので、固定値が残っていてよい）。
        *([f"--dataset.root={args.dataset_root.expanduser().resolve()}"]
          if args.dataset_root is not None else []),
        "--dataset.use_imagenet_stats=false",
        f"--dataset.video_backend={args.video_backend}",
        f"--output_dir={output_dir}",
        f"--job_name={args.run_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--persistent_workers={'true' if args.num_workers > 0 else 'false'}",
        "--env_eval_freq=0",
        "--eval_steps=0",
        f"--seed={args.seed}",
        "--save_checkpoint=true",
        f"--save_freq={args.save_freq}",
        "--save_checkpoint_to_hub=false",
        "--log_freq=100",
        "--wandb.enable=false",
        "--peft.method_type=LORA",
        f"--peft.r={args.lora_r}",
        f"--peft.lora_alpha={args.lora_alpha}",
    ]

    # **`expert` のときは旗を渡さず、LeRobot の既定に任せる。**
    # `--peft.target_modules=null` のような打ち消しを渡すより、そもそも書かない
    # ほうが取り違えが起きない。
    targets = {"text": _PEFT_TARGETS_TEXT, "vlm": _PEFT_TARGETS_VLM}.get(args.peft_scope)
    if targets is not None:
        command.append(f"--peft.target_modules={targets}")

    if episode_indices is not None:
        command.append(f"--dataset.episodes=[{','.join(map(str, episode_indices))}]")

    if args.image_transforms:
        command.append("--dataset.image_transforms.enable=true")

    # **最後に置く。** 同じ設定を 2 度渡したときは後ろが勝つので、
    # `--` の後ろに書いたもので上書きできる。
    return command + list(args.extra)


def _load_holdout(path: Path) -> list[str]:
    """検証専用にするタスクの指示文を読む。

    **摂動版の id ではなく指示文で外す。** 摂動版は元のタスクの実演データを
    共有するため、変種の id を抜いても同じ指示文の別の変種が学習に残り、
    中身は見たままになる。

    一覧は `solution/tools/make_holdout_split.py` が作る。
    **検証集合も同じ CSV を読む**ので、片方だけ古くなることがない。

    **読み取りを `make_validation_suite.py` と共有していないのは、
    学習イメージが `train.py` と `merge.py` だけを COPY するためである**
    （`solution/train/Dockerfile`）。`solution/tools/` は学習側に存在しない。

    Parameters
    ----------
    path : Path
        `instruction` 列を持つ CSV。

    Returns
    -------
    instructions : list[str]
        外す指示文。重複は落とす。
    """
    if not path.is_file():
        raise SystemExit(f"[train] 除外リストが見つかりません: {path}")

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        # **見出しと中身を別々に確かめる。** どちらが欠けたのかで直し方が違う。
        if reader.fieldnames is None or "instruction" not in reader.fieldnames:
            raise SystemExit(
                f"[train] 除外リストに instruction 列がありません: {path}"
                f"（見出し: {reader.fieldnames}）"
            )
        rows = list(reader)

    seen: dict[str, None] = {}
    for row in rows:
        instruction = row["instruction"].strip()
        if instruction:
            seen.setdefault(instruction, None)
    if not seen:
        raise SystemExit(f"[train] 除外リストが空です: {path}")
    return list(seen)


def _resolve_dataset_root(dataset_root: Path | None) -> Path:
    """番号選びが読むメタデータの場所を決める。

    **混ぜたデータセットを渡されたら Hub へ取りに行かない。** 取りに行くと
    `lerobot/libero_plus` の 14,347 本ぶんの番号が返り、**足した分が 1 本も
    選ばれないまま学習が始まる**（エラーにならないので気付けない）。

    Parameters
    ----------
    dataset_root : Path | None
        `--dataset-root`。None なら Hub から取る。

    Returns
    -------
    root : Path
        `meta/episodes/*.parquet` があるディレクトリ。

    Raises
    ------
    FileNotFoundError
        渡された場所にメタデータが無いとき。
    """
    if dataset_root is None:
        return _download_metadata()

    root = dataset_root.expanduser().resolve()
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"[train] 渡された場所にメタデータがありません: {root}/meta/episodes/*.parquet"
        )

    # **本数とタスクの内訳を出す。** 混ぜ忘れ・混ぜ損ねはここでしか気付けない。
    frame = _load_episode_metadata(root)
    counts = frame["task"].value_counts()
    print(f"[train] 渡されたデータセットを使う: {root}")
    print(f"[train]   {len(frame):,} 本 / {frame['task'].nunique()} タスク")
    print(f"[train]   1 本あたりの歩数 {int(frame['length'].min())}〜"
          f"{int(frame['length'].max())}（中央 {int(frame['length'].median())}）")
    print(f"[train]   本数の少ないタスク 3 つ: "
          f"{[(name, int(value)) for name, value in counts.tail(3).items()]}")
    return root


def _download_metadata() -> Path:
    """学習データのメタデータだけを取り寄せる。

    **動画は落とさない。** エピソードの選び方を決めるのに要るのは
    `meta/**` だけで、そこに 1 本ごとのタスク名・長さ・統計量が入っている。

    Returns
    -------
    dataset_root : Path
        `meta/episodes/*.parquet` があるディレクトリ。
    """
    print("[train] データセットのメタデータを取得")
    return Path(snapshot_download(
        _DATASET_REPO, revision=_DATASET_REVISION, repo_type="dataset",
        allow_patterns=["meta/**"],
    ))


def _drop_holdout(frame: pd.DataFrame, holdout: list[str]) -> pd.DataFrame:
    """検証専用のタスクをメタデータから落とす。

    **外したい指示文がデータセットに 1 つも無ければ、その場で止める。**
    黙って 0 件だけ外した学習を走らせると、**検証集合が汚れていることに
    気付かないまま結果を信じてしまう。** LIBERO-plus の名前とデータセットの
    `task` 列は書き方が違う場合があるため、ここで必ず突き合わせる。

    Parameters
    ----------
    frame : pd.DataFrame
        `_load_episode_metadata` が返したもの。
    holdout : list[str]
        外す指示文。空なら何もしない。

    Returns
    -------
    kept : pd.DataFrame
        外した後のメタデータ。
    """
    if not holdout:
        return frame

    # `task` 列は `_load_episode_metadata` が文字列で作るので変換は要らない。
    available = set(frame["task"])
    missing = [name for name in holdout if name not in available]
    if missing:
        sample = sorted(available)[:8]
        raise SystemExit(
            "[train] 除外リストの指示文がデータセットに見つかりません:\n"
            + "".join(f"    {name}\n" for name in missing)
            + f"  データセット側の書き方の例（全 {len(available)} 種）:\n"
            + "".join(f"    {name}\n" for name in sample)
            + "  **make_holdout_split.py の導出が LIBERO-plus の版と"
              "合っていない可能性がある。**"
        )

    kept = frame[~frame["task"].isin(holdout)]
    kept_tasks = kept["task"].nunique()
    print(f"[train] 除外: {len(available) - kept_tasks} タスク / "
          f"{len(frame) - len(kept):,} エピソードを学習から外した")
    print(f"[train]   残り {len(kept):,} エピソード / {kept_tasks} タスク")
    return kept


def _select_without_holdout(dataset_root: Path, holdout: list[str]) -> list[int]:
    """外すタスクを除いた全エピソードの番号を返す。

    Parameters
    ----------
    dataset_root : Path
        `meta/episodes/*.parquet` があるディレクトリ。
    holdout : list[str]
        外す指示文。

    Returns
    -------
    episode_indices : list[int]
        学習に使うエピソード番号。
    """
    kept = _drop_holdout(_load_episode_metadata(dataset_root), holdout)
    indices = sorted(int(value) for value in kept["episode_index"])
    # **番号がコマンド 1 行に並ぶ。** 全部使う場合は旗を省けるが、
    # 外すときは並べるしかない。長さを出しておく。
    # `_build_command` はカンマ区切りで繋ぐので、その形で数える
    # （`str(list)` は空白も入るため 1 割ほど多く見える）。
    width = len(indices) - 1 + sum(len(str(value)) for value in indices)
    print(f"[train]   `--dataset.episodes` の長さ 約 {width / 1024:.0f} KB")
    return indices


def _warn_if_schedule_mixed(args: argparse.Namespace) -> None:
    """学習率と歩数が、信用できる 2 つの条件のどちらにも当てはまらないなら言う。

    一次情報で分かっている条件は 2 つだけである。

    | 出典 | 学習率 / warmup / 最後 | 歩数 |
    |---|---|---|
    | 出発点の重みの `train_config.json` | 1e-4 / 1,000 / 2.5e-6 | 20,000 |
    | 配布ノートブック（`examples/` の LoRA ノート） | 3e-4 / 100 / 3e-5 | 3,000 |

    **8/11 に本番 0.1409 → 0.016 と 8.8 倍悪化した回は、後者の学習率で
    前者の歩数を回していた**（重みの `config.json` に `optimizer_lr=3e-4`、
    `scheduler_warmup_steps=100`、`scheduler_decay_steps=20000` が残っている）。
    **この組み合わせを認めている出典は無い。**
    経緯は `docs/reference/11_training_schedule.md`。

    止めはしない。**歩数を減らして試す使い方は正しい**ので、
    判断は渡す側に残す。

    Parameters
    ----------
    args : argparse.Namespace
        解析済みの引数。
    """
    notebook_lr = 3e-4
    notebook_steps = 3000

    if args.learning_rate >= notebook_lr and args.steps > notebook_steps:
        print(
            f"[train] **注意: 学習率 {args.learning_rate:g} は配布ノートブックの値で、"
            f"{notebook_steps:,} 歩を想定したものです。**"
            f" いま {args.steps:,} 歩を指定しています。"
        )
        print(
            "[train]   これは本番スコアを 0.1409 から 0.016 へ落とした回と"
            "同じ組み合わせです（docs/reference/11_training_schedule.md）。"
        )
        print(
            "[train]   長く回すなら出発点と同じ --learning-rate 1e-4"
            " --warmup-steps 1000 --final-learning-rate 2.5e-6 に揃えてください。"
        )


def main() -> None:
    args = _parse_args()
    output_dir: Path = (args.output_root / args.run_name).resolve()

    # **出発点から作るときは、出力先を先に作ってはいけない。**
    # LeRobot は `not resume and output_dir.is_dir()` で FileExistsError を投げる
    # （`configs/train.py` の `validate`）。**空のディレクトリでも拒否する。**
    # 記録（`episodes.json`）は隣へ書き、学習が始まってから中へ写す。
    if args.resume:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif output_dir.is_dir():
        raise SystemExit(
            f"[train] 出力先が既にある: {output_dir}\n"
            "  LeRobot は既存のディレクトリへ上書きしない。次のどれかを選ぶこと:\n"
            f"    --run-name を変える / rm -rf {output_dir} / --resume で続ける"
        )
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[train] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'なし'}")
    _patch_dtype_for_gpu()

    resume_from = _find_resume_checkpoint(output_dir) if args.resume else None
    if args.resume and resume_from is None:
        print("[train] 再開できるチェックポイントが無いため、最初から学習する")

    if resume_from is not None:
        # 再開時は学習条件をチェックポイントから復元する。エピソードの選び直しも
        # 初期重みの取得も要らない。ステップ数は training_state から読まれる。
        print(f"[train] 再開: {resume_from.name}（step は training_state から復元される）")
        _repair_base_model_path(resume_from)
        command = _build_resume_command(args, output_dir, resume_from)
        print(f"[train] 学習を再開（batch {args.batch_size}）")
    else:
        print("[train] 初期重みを取得")
        base_model = _download_base_model()

        # **0 なら選ばない。** 出発点の重みも `episodes: None`（全部）で作られている。
        # 選ぶ処理を通すと 14,347 個の番号がコマンド 1 行に並ぶので、旗ごと省く。
        record: dict[str, Any]
        selects_by_task = bool(args.emphasize_tasks) or args.max_demo_steps > 0
        holdout = _load_holdout(args.holdout_tasks) if args.holdout_tasks else []
        if args.episodes_per_task <= 0 and not selects_by_task and not holdout:
            episode_indices = None
            record = {"episodes": "all", "per_task": None}
            print("[train] **全エピソードを使う**（出発点の重みと同じ条件）")
        elif args.episodes_per_task <= 0 and not selects_by_task:
            # **除外だけを指定した場合。** 外した分以外は全部使うが、
            # 「全部」を旗の省略で表せないので番号を並べる。
            print(f"[train] 除外リストを読む: {args.holdout_tasks}")
            dataset_root = _download_metadata()
            episode_indices = _select_without_holdout(dataset_root, holdout)
            record = {
                "episodes": episode_indices,
                "per_task": None,
                "holdout_tasks": holdout,
                "holdout_source": str(args.holdout_tasks),
            }
        else:
            dataset_root = _resolve_dataset_root(args.dataset_root)
            if selects_by_task:
                episode_indices = _select_emphasized_episodes(
                    dataset_root, list(args.emphasize_tasks or []),
                    args.emphasis_multiplier, args.max_demo_steps,
                    args.steps * args.batch_size, holdout=holdout,
                )
                record = {
                    "episodes": episode_indices,
                    "per_task": None,
                    "emphasize_tasks": list(args.emphasize_tasks or []),
                    "emphasis_multiplier": args.emphasis_multiplier,
                    "max_demo_steps": args.max_demo_steps,
                }
            else:
                episode_indices = _select_episodes(
                    dataset_root, args.episodes_per_task, holdout=holdout
                )
                record = {
                    "episodes": episode_indices, "per_task": args.episodes_per_task
                }

        # **レポートの申告項目になるので必ず残す**（タスク数・エピソード数）。
        # マージの出力に `train_config.json` は入らないため、ここが唯一の記録になる。
        #
        # **出力先の中へは書けない。** 存在すると LeRobot が学習を始めない。
        # だから隣へ書き、学習が始まって出力先ができてから写す（下）。
        record_path = output_dir.with_name(output_dir.name + "_episodes.json")
        record_path.write_text(json.dumps(record), encoding="utf-8")
        print(f"[train] 使うエピソードの記録: {record_path}")

        command = _build_command(args, base_model, episode_indices, output_dir)
        print(f"[train] 学習を開始（{args.steps} ステップ、batch {args.batch_size}、"
              f"lr {args.learning_rate}、warmup {args.warmup_steps}、"
              f"画像拡張 {'有効' if args.image_transforms else '無効'}）")
        _warn_if_schedule_mixed(args)

    process = subprocess.run(command)

    # **記録を出力先の中へ写す。** 隣に置いたままだと、チェックポイントを
    # 別のマシンへ運んだときに一緒に付いてこない。
    if not args.resume:
        record_path = output_dir.with_name(output_dir.name + "_episodes.json")
        if record_path.is_file() and output_dir.is_dir():
            shutil.copy2(record_path, output_dir / "episodes.json")

    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
