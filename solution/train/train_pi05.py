#!/usr/bin/env python3
"""π0.5 (pi05) の追加学習を、SmolVLA と同じ Docker/GPU 環境で回す。

**これは実験的な追加パスである。** 予選のモデルは SmolVLA（`train.py`）で
既に提出できる状態にあり、**そちらを保険として残したまま**、並行して
π0.5 が間に合うか試すために書いた。時間が無ければ `train.py` 側の運用を優先する。

π0.5 は SmolVLA の約 9 倍のパラメータ数（後述）で、**この環境（GPU 1 枚 20GB、
`nvidia-smi` 実測）で本当に載るかは未検証**である。まず `--smoke-test` で
数十歩だけ回し、OOM しないこと・1 ステップあたりの時間を確かめてから、
本番のステップ数へ進むこと。

## 出発点

    BASE_MODEL_REPO     = lerobot/pi05_libero_base
    BASE_MODEL_REVISION = a217bfd3b14673cf2ce597e69997ab21866438dd
        （2026-08-13 に Hugging Face の /api/models で確認したコミット。
        再確認: curl -s https://huggingface.co/api/models/lerobot/pi05_libero_base | python3 -m json.tool）
    DATASET_REPO / REVISION は train.py と同じ lerobot/libero_plus（変更なし）

`lerobot/pi05_libero_base` の `config.json`（2026-08-13 確認）:

| 項目 | 値 |
|---|---|
| 画像入力 | `observation.images.image` `observation.images.image2`（各 [3,256,256]） |
| state | `observation.state` [8] |
| action | [7] |
| `empty_cameras` | 1 |
| `chunk_size` / `n_action_steps` | 50 / 10 |
| `paligemma_variant` / `action_expert_variant` | gemma_2b / gemma_300m |

state=8・action=7 は、このコンペの観測変換（`processor/env_processor.py` の
`LiberoProcessorStep`）が作る形とそのまま一致する。カメラは `--policy.pretrained_path`
使用時に学習データの列名（`lerobot/libero_plus` の `front`/`wrist`）へ
自動的に差し替わるため、SmolVLA と同様 rename_map は要らない
（[docs/reference/10_observation_interface.md](../../docs/reference/10_observation_interface.md)
と同じ仕組みが pi05 でも成立する前提。**pi05 側での実測はしていない**）。

## SmolVLA との違いで、時間の見積もりに効くもの

1. **パラメータ数がおよそ 9 倍**（gemma_2b + gemma_300m ≈ 2.3B、
   SmolVLA は VLM 350M + 350M ≈ 450M）。公式のフル学習コマンドは
   「80GB GPU 1 枚」を前提にしている
   （https://huggingface.co/docs/lerobot/en/pi05 の Training セクション）。
   この環境（20GB）では `--policy.freeze_vision_encoder=true
   --policy.train_expert_only=true --policy.gradient_checkpointing=true` で
   VLM を固定し action expert だけを学習する構成が前提になる。
   **それでも 20GB に収まるかは未検証** — 公式の縮小レシピも batch_size=64 を
   使っており、この環境ではもっと下げる必要がある可能性が高い。
2. **libero_plus 向けの公開チェックポイントが無い。** SmolVLA は
   `lerobot/smolvla_libero_plus`（このコンペのデータで一度学習済み）を出発点に
   できたが、pi05 は素の `lerobot/pi05_libero_base`（通常の LIBERO）からしか
   始められない。分布への適応をゼロからやり直す分だけ、
   同じステップ数でも到達点は SmolVLA より手前になりやすい
   （[docs/reference/01_base_model.md](../../docs/reference/01_base_model.md) 参照）。
3. **LoRA の対応は未確認。** `train.py` は SmolVLA 向けに `--peft.method_type=LORA`
   を使っているが、pi05 の公式ドキュメントは PEFT に触れていない。
   **この台本は LoRA を使わず、公式レシピどおり `train_expert_only` の
   フルパラメータ学習にしている。** マージ手順（`merge.py`）も不要になる —
   出力はそのまま提出用チェックポイントとして読める。

## ライセンス（手作業が要る・自動化できない）

π0.5 は Hugging Face の **gemma ライセンス**で配布されている
（`lerobot/pi05_libero_base` のモデルカード）。学習前に、Hugging Face の
アカウントで https://huggingface.co/google/paligemma-3b-pt-224 の利用規約に
同意し、`hf auth login`（または `huggingface-cli login`）でログインしておくこと。
**これは Claude では代行できない。** 同意していない状態で学習を始めると
gated repo のエラーで止まる。

## 使い方

    # まず数十歩だけ回して、20GB に収まるか・1 ステップ何秒かを確かめる
    python train_pi05.py --smoke-test

    # 収まったら本番。ステップ数はまだ検証していない見積もり値（下記参照）
    python train_pi05.py --run-name pi05_libero_plus_v1 --steps 8000 --batch-size 8

    # 再開
    python train_pi05.py --run-name pi05_libero_plus_v1 --resume

    # 段を分けた学習の 2 段目（全データで馴染ませた重みを、苦手タスクを厚くした
    # データで仕上げる）。**--learning-rate を必ず明示すること** —— 既定のままだと
    # 前の段が 2.5e-6 で終わっている 10 倍から焚き直す。
    # 詳しくは docs/talk/handoff_pi05_stage2.md
    python train_pi05.py --run-name pi05_weak3x_stage2 \
        --init-from /out/pi05_full_local/checkpoints/024000/pretrained_model \
        --steps 6000 --batch-size 16 --learning-rate 5e-6 --max-demo-steps 300 \
        --emphasize-tasks "put the bowl on top of the cabinet" ...

**ステップ数のデフォルト（8,000）は検証済みの数字ではない。** SmolVLA の出発点は
20,000 ステップ × batch 32（640,000 サンプル）だが、pi05 は残り時間で
1 ステップあたりの実測時間を見てから調整すること。ログに出る
`steps/sec` を見て、残り GPU 時間で何ステップ回せるか逆算するのが安全。

Reads:
  - lerobot/pi05_libero_base（HF、初期重み。gemma ライセンス）
  - lerobot/libero_plus（HF、学習データ。train.py と共有）
Generates:
  - $OUTPUT_ROOT/<run-name>/checkpoints/**（そのまま提出用として読めるフル重み。
    LoRA アダプタではないので merge.py は不要）
  - $OUTPUT_ROOT/<run-name>.log（学習ログ。--log-file で変えられる。
    **run ディレクトリの中ではなく隣に置く** —— 理由は _log_path のコメント）
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

# train.py のデータセット選択ロジック（テスト済み）を再利用する。
# 同じディレクトリに置くこと（Dockerfile が両方を /workspace へ COPY する）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (  # noqa: E402
    _download_metadata,
    _find_resume_checkpoint,
    _load_holdout,
    _resolve_dataset_root,
    _select_emphasized_episodes,
    _select_episodes,
    _select_without_holdout,
)

_BASE_MODEL_REPO = "lerobot/pi05_libero_base"
_BASE_MODEL_REVISION = "a217bfd3b14673cf2ce597e69997ab21866438dd"
_DATASET_REPO = "lerobot/libero_plus"
_DATASET_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"

# pi05 の既定正規化（quantile）は lerobot/libero_plus に q01/q99 統計が
# 無ければ最初のバッチで落ちる。公式ドキュメントが「lerobot/libero でも
# 同じ理由で mean_std に落とした」と明記しているレシピに揃える。
_NORMALIZATION_MAPPING = '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("OUTPUT_ROOT", "/out")))
    parser.add_argument("--run-name", type=str, default="pi05_libero_plus")
    parser.add_argument("--steps", type=int, default=8000,
                        help="学習ステップ数。**未検証の見積もり。** 実測の"
                             "steps/sec を見てから決め直すこと")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="出発点は SmolVLA の 32 よりずっと小さい 8 から始める"
                             "（未検証。OOM したらさらに下げる）")
    parser.add_argument("--smoke-test", action="store_true",
                        help="20 ステップ・batch 2 で動かし、OOM しないこと・"
                             "1 ステップの時間を確かめる。--steps/--batch-size を上書きする")
    parser.add_argument("--episodes-per-task", type=int, default=0,
                        help="train.py と同じ意味。0 なら全エピソード")
    parser.add_argument("--emphasize-tasks", type=str, nargs="*", default=None,
                        metavar="INSTRUCTION")
    parser.add_argument("--emphasis-multiplier", type=float, default=3.0)
    parser.add_argument("--max-demo-steps", type=int, default=0)
    parser.add_argument("--holdout-tasks", type=Path, default=None, metavar="CSV")
    parser.add_argument("--dataset-root", type=Path, default=None, metavar="DIR")
    parser.add_argument("--init-from", type=Path, default=None, metavar="DIR",
                        help="初期重みに HF の pi05_libero_base ではなく、"
                             "前の学習の checkpoints/<ステップ数>/pretrained_model を使う"
                             "（段を分けた学習の 2 段目。"
                             "docs/talk/handoff_pi05_stage2.md）")
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5,
                        help="pi05_libero_base の config.json に残る値をそのまま既定にした")
    parser.add_argument("--gradient-checkpointing", type=str, default="true",
                        choices=("true", "false"),
                        help="20GB で載せるための必須級の節約。外すのは"
                             "80GB 級 GPU が使えるときだけ")
    parser.add_argument("--freeze-vision-encoder", type=str, default="true",
                        choices=("true", "false"))
    parser.add_argument("--train-expert-only", type=str, default="true",
                        choices=("true", "false"),
                        help="true: action expert とプロジェクションだけ学習"
                             "（メモリ節約、公式ドキュメント推奨の縮小レシピ）。"
                             "false: VLM も含めて全学習（80GB 級を想定、まず載らない）")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=("bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None, metavar="FILE",
                        help="学習ログの書き出し先（既定は "
                             "$OUTPUT_ROOT/<run-name>.log）。--no-log で切れる")
    parser.add_argument("--no-log", action="store_true",
                        help="ログファイルを作らない（端末に出すだけ）")
    parser.add_argument("extra", nargs="*", metavar="-- ...",
                        help="`--` の後ろは lerobot-train へそのまま渡す")
    args = parser.parse_args()

    if args.smoke_test:
        args.steps = 20
        args.batch_size = 2
        args.save_freq = 20
        if args.run_name == "pi05_libero_plus":
            args.run_name = "pi05_smoke"
    return args


def _download_base_model() -> Path:
    """初期重みを取得する。gated license 未同意のときは対処法を出して止める。"""
    try:
        return Path(snapshot_download(_BASE_MODEL_REPO, revision=_BASE_MODEL_REVISION))
    except GatedRepoError as exc:
        raise SystemExit(
            "[train_pi05] gemma ライセンスに未同意、または未ログインです。\n"
            "  1. https://huggingface.co/google/paligemma-3b-pt-224 を開き、\n"
            "     ライセンスに同意する（Hugging Face アカウントが要る）\n"
            "  2. コンテナ内で `hf auth login`（または `huggingface-cli login`）\n"
            "     を実行してから、もう一度このコマンドを実行すること\n"
            f"  元のエラー: {exc}"
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"[train_pi05] 初期重みの取得に失敗: {exc}") from exc


def _resolve_init_from(path: Path) -> Path:
    """`--init-from` に渡されたディレクトリを検査して返す。

    **段を分けた学習の 2 段目で使う。** `--resume` は同じデータ・同じステップ数の
    続きしか出来ない（`_build_resume_command` が前回の `train_config.json` を
    そのまま読み直すため）ので、**データの選び方を変えて続けたいときは
    「前の段の出力を初期重みとして読み直す」形になる。**

    渡すのは `checkpoints/<ステップ数>/pretrained_model` である。そこには
    `config.json`・`model.safetensors`・正規化統計を含む前後処理が揃っており、
    HF から取る初期重みと同じように `--policy.pretrained_path` へ渡せる。

    **`checkpoints/<ステップ数>` を渡す間違いが起きやすい**（1 階層浅い）ので、
    その場で気付けるよう言い直す。**10 時間の学習を始めた直後に落ちるのは高い。**

    **学習率は呼び出し側で明示すること。** 既定のままだと頂点 2.5e-5 に戻り、
    前の段が 2.5e-6 で終わっているぶんの 10 倍から再スタートする
    （docs/reference/15_pi05_lr_schedule.md）。
    """
    resolved = path.expanduser()
    if not resolved.is_dir():
        raise SystemExit(f"[train_pi05] --init-from のディレクトリが無い: {resolved}")

    missing = [name for name in ("config.json", "model.safetensors")
               if not (resolved / name).is_file()]
    if missing:
        inner = resolved / "pretrained_model"
        hint = (f"\n  もしかして: --init-from {inner}"
                if (inner / "config.json").is_file() else "")
        raise SystemExit(
            f"[train_pi05] --init-from に {' と '.join(missing)} が無い: {resolved}\n"
            "  渡すのは checkpoints/<ステップ数>/pretrained_model である。" + hint)

    print(f"[train_pi05] 初期重みにローカルのチェックポイントを使う: {resolved}")
    return resolved


def _build_command(
    args: argparse.Namespace, base_model: Path,
    episode_indices: list[int] | None, output_dir: Path,
) -> list[str]:
    command = [
        "lerobot-train",
        "--policy.type=pi05",
        f"--policy.pretrained_path={base_model}",
        f"--policy.normalization_mapping={_NORMALIZATION_MAPPING}",
        "--policy.n_action_steps=10",
        "--policy.empty_cameras=1",
        f"--policy.freeze_vision_encoder={args.freeze_vision_encoder}",
        f"--policy.train_expert_only={args.train_expert_only}",
        f"--policy.gradient_checkpointing={args.gradient_checkpointing}",
        f"--policy.dtype={args.dtype}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.optimizer_lr={args.learning_rate}",
        f"--dataset.repo_id={_DATASET_REPO}",
        f"--dataset.revision={_DATASET_REVISION}",
        *([f"--dataset.root={args.dataset_root.expanduser().resolve()}"]
          if args.dataset_root is not None else []),
        f"--dataset.video_backend=torchcodec",
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
        "--log_freq=20",
        "--wandb.enable=false",
    ]
    if episode_indices is not None:
        command.append(f"--dataset.episodes=[{','.join(map(str, episode_indices))}]")
    return command + list(args.extra)


def _build_resume_command(args: argparse.Namespace, output_dir: Path,
                          resume_from: Path) -> list[str]:
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


def _log_path(args: argparse.Namespace, output_dir: Path) -> Path:
    """ログの置き場所を決める。**run ディレクトリの中には置けない。**

    LeRobot v0.6.0 は、出力先ディレクトリが既にあると学習を始めずに落ちる
    （`lerobot/configs/train.py` の `TrainPipelineConfig.validate`:
    `if not self.resume and ... self.output_dir.is_dir(): raise FileExistsError`）。
    ログを置くために run ディレクトリを先に作ると、それだけで学習が
    始まらなくなる。そのため既定では**隣**（`<run-name>.log`）に置く。
    """
    if args.log_file is not None:
        return args.log_file.expanduser().resolve()
    return output_dir.parent / f"{output_dir.name}.log"


def _write_all(fd: int, data: bytes) -> None:
    """os.write は途中までしか書かないことがあるので、書き切るまで回す。"""
    while data:
        data = data[os.write(fd, data):]


class _Tee:
    """このスクリプト自身の print を、端末とログの両方へ流す。

    子プロセス（lerobot-train）の出力は fd を継承しないので、そちらは
    _run_and_log が別に受け持つ。
    """

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._stream.flush()
        self._log_file.write(text.encode("utf-8", "replace"))
        self._log_file.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()


def _run_and_log(command: list[str], log_file) -> int:
    """lerobot-train を動かし、出力を端末とログファイルへ同時に流す。

    **ログ側には進捗バーの書き換えを残さない。** tqdm は `\\r` で同じ行を
    上書きしていくので、そのまま保存すると 24,000 ステップぶんの途中経過
    （1回あたり百数十バイト）が全部残って数十MBになる。`\\r` で区切った
    最後の一片だけを書けば、端末に最後に見えていた状態＝`log_freq` ごとの
    `step:` 行だけが残る（`step:` 行は改行で終わるため）。

    端末側には受け取ったものをそのまま流す（進捗バーも動く）。sys.stdout は
    _Tee に差し替わっている場合があるので、fd 1 へ直接書く。
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    assert process.stdout is not None
    buffer = b""
    for chunk in iter(lambda: process.stdout.read1(65536), b""):
        _write_all(1, chunk)
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            log_file.write(line.rsplit(b"\r", 1)[-1] + b"\n")
        log_file.flush()
    if buffer:
        log_file.write(buffer.rsplit(b"\r", 1)[-1] + b"\n")
        log_file.flush()
    return process.wait()


def main() -> None:
    args = _parse_args()
    output_dir: Path = (args.output_root / args.run_name).resolve()

    if args.resume:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif output_dir.is_dir():
        raise SystemExit(
            f"[train_pi05] 出力先が既にある: {output_dir}\n"
            f"    --run-name を変える / rm -rf {output_dir} / --resume で続ける"
        )
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    # ログは追記で開く（--resume で何度も再開しても、前の分が消えない）。
    # 以降の print はここから端末とログの両方へ出る。
    log_file = None
    if not args.no_log:
        log_path = _log_path(args, output_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("ab")
        sys.stdout = _Tee(sys.stdout, log_file)
        print(f"[train_pi05] === {datetime.now():%F %T} 開始 ===")
        print(f"[train_pi05] ログ: {log_path}（`tail -f` で追える）")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[train_pi05] GPU: {name}（{total_gb:.1f} GB）")
        if total_gb < 24:
            print(
                f"[train_pi05] **注意: {total_gb:.1f} GB しかない。"
                "公式の縮小レシピ（train_expert_only）ですら 20GB で動く保証はない。"
                "まず --smoke-test で確かめること。**"
            )
    else:
        print("[train_pi05] GPU が無い。CPU では現実的な時間で終わらない")

    resume_from = _find_resume_checkpoint(output_dir) if args.resume else None
    if args.resume and resume_from is None:
        print("[train_pi05] 再開できるチェックポイントが無いため、最初から学習する")

    if resume_from is not None:
        print(f"[train_pi05] 再開: {resume_from.name}")
        command = _build_resume_command(args, output_dir, resume_from)
    else:
        if args.init_from is not None:
            base_model = _resolve_init_from(args.init_from)
        else:
            print("[train_pi05] 初期重みを取得: "
                  f"{_BASE_MODEL_REPO}@{_BASE_MODEL_REVISION[:8]}")
            base_model = _download_base_model()

        holdout = _load_holdout(args.holdout_tasks) if args.holdout_tasks else []
        selects_by_task = bool(args.emphasize_tasks) or args.max_demo_steps > 0

        if args.episodes_per_task <= 0 and not selects_by_task and not holdout:
            episode_indices = None
            print("[train_pi05] 全エピソードを使う")
        elif args.episodes_per_task <= 0 and not selects_by_task:
            dataset_root = _download_metadata()
            episode_indices = _select_without_holdout(dataset_root, holdout)
        else:
            dataset_root = _resolve_dataset_root(args.dataset_root)
            if selects_by_task:
                episode_indices = _select_emphasized_episodes(
                    dataset_root, list(args.emphasize_tasks or []),
                    args.emphasis_multiplier, args.max_demo_steps,
                    args.steps * args.batch_size, holdout=holdout,
                )
            else:
                episode_indices = _select_episodes(
                    dataset_root, args.episodes_per_task, holdout=holdout
                )

        command = _build_command(args, base_model, episode_indices, output_dir)
        print(
            f"[train_pi05] 学習開始（{args.steps} 歩、batch {args.batch_size}、"
            f"freeze_vision_encoder={args.freeze_vision_encoder}、"
            f"train_expert_only={args.train_expert_only}、"
            f"dtype={args.dtype}、grad_ckpt={args.gradient_checkpointing}）"
        )
        if args.smoke_test:
            print("[train_pi05] **smoke test** — 20 ステップで OOM しないか・"
                  "1 ステップの時間だけを見る。スコアは見ない")

    print("[train_pi05] コマンド: " + shlex.join(command))
    if log_file is None:
        returncode = subprocess.run(command).returncode
    else:
        returncode = _run_and_log(command, log_file)
        print(f"[train_pi05] === {datetime.now():%F %T} 終了（終了コード "
              f"{returncode}）===")
        log_file.close()
    sys.exit(returncode)


if __name__ == "__main__":
    main()
