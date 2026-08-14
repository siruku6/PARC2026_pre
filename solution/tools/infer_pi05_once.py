#!/usr/bin/env python3
"""学習途中の π0.5 チェックポイントで、本当に 1 手ぶんの行動が出るかを確かめる。

**スコアを測る道具ではない。** 見たいのは次の 4 点だけで、どれも
[docs/howto/07_train_pi05_local.md](../../docs/howto/07_train_pi05_local.md) の
「まだ確認していないこと」に並んでいたものである。

1. 学習が吐いたチェックポイントをそのまま読めるか（LoRA ではないので
   `merge.py` に相当する手順が要らない、という前提の確認）
2. **カメラ名が学習データ側（`front` / `wrist`）へ差し替わっているか。**
   `lerobot/pi05_libero_base` の素の設定は `image` / `image2` である。SmolVLA では
   rename_map 無しで差し替わっていた（docs/reference/10_observation_interface.md）が、
   π0.5 では未実測だった
3. 評価パイプラインが渡す形の観測（128×128 の 2 枚と関節の値）から、
   `(7,)` の行動が有限な数として出るか
4. **1 回の推論に何秒かかるか。** ただしここで測るのはこのスクリプトを動かした
   機械の速さであって、採点環境（L4 24GB・10 秒/推論、規定 PDF p.12）の値ではない

観測は**ダミー（真っ黒な画像）**である。LIBERO を立てずに済ませるためで、
行動の中身が正しいかは見ない。見るのは「通るか」「何秒か」だけである。

## メモリの話（学習と同時に動かすなら、ここが本題）

チェックポイントの重みは **8.71 GB**（fp32 が 1.99 GB・bf16 が 6.72 GB の混在。
`model.safetensors` のヘッダ実測）。ところが LeRobot 本体の
`PI05Policy.from_pretrained` を素で使うと、**ピークで 18 GB 級**を必要とする。

| 段階 | 要る量 | なぜ |
|---|---|---|
| 組み立て | **16.6 GB** | PaliGemma を **fp32 で作ってから** bf16 へ落とす作りである（`to_bfloat16_for_selected_params`。vision_tower と layernorm は fp32 のまま残す） |
| 読み込み | **17.4 GB** | `load_file` が 8.71 GB を丸ごと辞書にしてから `load_state_dict` で写すため、模型と辞書が同時に生きる |

**このマシンは合計 28 GB で、学習中は 10 GB 前後が埋まっている**（学習本体の RSS
7.1 GB ＋ dataloader）。18 GB を要求すると足りず、**OOM killer が「いちばん
大きいプロセス」＝学習本体を狙いうる。** 10 時間の学習を落とすのは高すぎる。

そこで既定の `--load-mode lowmem` では次の 2 つを変え、**ピークを 10 GB 前後**に
下げてある。**出来上がりの dtype は本体の道と同じになる**（下の 2 点目）。

1. 組み立てを最初から bf16 の既定 dtype で行う（fp32 で作る 16.6 GB を通らない）
2. 重みを **1 テンソルずつ** `safe_open` で読み、`param.data` を**置き換える**。
   置き換えなので、**チェックポイントに入っている dtype がそのまま採用される**
   （fp32 のテンソルは fp32 に戻る）。同時に生きるのは模型＋最大 1 個
   （0.98 GB の embed_tokens）だけになる

おまけに、この道は**重みを読めたことが数で確かめられる**。本体の
`from_pretrained` は読み込みに失敗しても
`Returning model without loading pretrained weights` と出して**初期化したままの
模型を返す**ので、推論が通っただけでは「学習の成果を使った」と言えない。

## 使い方

学習に使ったイメージの中で動かす（ホスト側に torch も LeRobot も無い）。

    # 学習と同時に、CPU で（既定。1 推論あたり数分かかる）
    #   **--memory を必ず付ける。** はみ出したときに死ぬのをこちら側に限定する
    docker run --rm -it --memory=12g --user "$(id -u):$(id -g)" \
        -v "$PWD:/repo" -v "$HOME/.cache/huggingface:/hf_home" \
        parc2026-train-pi05 \
        python /repo/solution/tools/infer_pi05_once.py \
            /repo/work/train/pi05_full_local_old/checkpoints/000500/pretrained_model \
            --device cpu --steps 2

    # GPU が空いているとき（数秒で終わる。**学習中は VRAM が無いので不可**）
    docker run --rm -it --gpus all --user "$(id -u):$(id -g)" \
        -v "$PWD:/repo" -v "$HOME/.cache/huggingface:/hf_home" \
        parc2026-train-pi05 \
        python /repo/solution/tools/infer_pi05_once.py \
            /repo/work/train/pi05_full_local_old/checkpoints/000500/pretrained_model

`--steps` の既定は 12 で、**チェックポイントの `n_action_steps`（10）を必ず 1 度
またぐ**ようにしてある。またいだ歩でだけ推論が走り、それ以外は作った列を順に
出すだけなので、時間の出方が 2 通りあることが表に出る。CPU で試すときは
`--steps 2`（重い 1 歩 ＋ 軽い 1 歩）で十分である。

Reads:
  - <weights>/config.json, model.safetensors, policy_pre/postprocessor*
  - $HF_HOME/hub/models--google--paligemma-3b-pt-224（**トークナイザだけ**。
    VLM 本体の重みはチェックポイントに入っているので取りに行かない）
Generates:
  - 標準出力に、1 歩ごとの所要時間と行動の値（異常があれば終了コード 1）
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

# 評価パイプラインが渡してくる画像の大きさ（policy_server.py の _warm_up と同じ）。
_CAMERA_SIZE = 128

# 画像を [0, 1] に直すための割る数（policy_server.py の _CHW_SCALE と同じ）。
_IMAGE_SCALE = 255.0

# トークナイザの出どころ。**チェックポイントの policy_preprocessor.json には
# この repo 名が焼き込まれている。** ローカルに落ちていればそこを指し、無ければ
# HF から取りに行く（gemma ライセンス同意済みのアカウントでのログインが要る）。
_TOKENIZER_REPO = "google/paligemma-3b-pt-224"
_TOKENIZER_CACHE_GLOB = "hub/models--google--paligemma-3b-pt-224/snapshots/*"

# 学習データ（lerobot/libero_plus）の列名。素の pi05_libero_base は
# image / image2 なので、**ここへ差し替わっていることが確認したい点である。**
_EXPECTED_CAMERAS = ("observation.images.front", "observation.images.wrist")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", type=Path,
                        help="チェックポイントの pretrained_model ディレクトリ")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "cpu", "cuda"),
                        help="auto は GPU があれば cuda（既定）")
    parser.add_argument("--load-mode", type=str, default="lowmem",
                        choices=("lowmem", "builtin"),
                        help="lowmem（既定）はピーク 10GB 前後で読む。"
                             "builtin は LeRobot 本体の from_pretrained を使い、"
                             "**18GB 級を要求する**（学習と同時には使わないこと）")
    parser.add_argument("--steps", type=int, default=12,
                        help="get_action を何回呼ぶか（既定 12。n_action_steps=10 を"
                             "またぐので、推論が走る歩と走らない歩の両方が出る）")
    parser.add_argument("--n-action-steps", type=int, default=0,
                        help="1 回の推論で作った列を何歩ぶん使うか。"
                             "0 ならチェックポイントの値（10）のまま")
    parser.add_argument("--num-inference-steps", type=int, default=0,
                        help="flow matching の刻み数。0 なら設定のまま（既定 10）")
    parser.add_argument("--instruction", type=str,
                        default="pick up the black bowl next to the plate "
                                "and place it on the plate",
                        help="指示文。学習データにある 40 種の 1 つを既定にしてある")
    parser.add_argument("--tokenizer", type=str, default="auto",
                        help=f"auto なら $HF_HOME の中を探し、無ければ {_TOKENIZER_REPO}")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """使うデバイスを決める。auto は GPU があれば cuda。

    **`replay_pi05_on_dataset.py` からも呼ぶので `_` を付けない。**
    """
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_tokenizer(requested: str) -> str:
    """トークナイザの置き場所を返す。

    `$HF_HOME` に落ちていればそのパスを返す。**通信の要らない道を先に選ぶ**のは、
    採点環境が外部通信を切っているのと同じ条件で試せるようにするためである。

    Parameters
    ----------
    requested : str
        `--tokenizer` の値。`auto` 以外はそのまま返す。

    Returns
    -------
    tokenizer_name : str
        `make_pre_post_processors` に渡す、ローカルパスまたは HF の repo 名。
    """
    if requested != "auto":
        return requested

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    for snapshot in sorted(hf_home.glob(_TOKENIZER_CACHE_GLOB)):
        if (snapshot / "tokenizer.json").is_file():
            return str(snapshot)

    print(f"[infer] トークナイザが $HF_HOME に無いので HF から取る: {_TOKENIZER_REPO}")
    return _TOKENIZER_REPO


def _read_config(weights: Path, device: str, n_action_steps: int,
                 num_inference_steps: int):
    """チェックポイントの設定を読み、呼び出し側のつまみで上書きして返す。

    Parameters
    ----------
    n_action_steps, num_inference_steps : int
        0 なら設定の値をそのまま使う。
    """
    from lerobot.configs import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(weights)
    config.device = device
    if n_action_steps > 0:
        config.n_action_steps = n_action_steps
    if num_inference_steps > 0:
        config.num_inference_steps = num_inference_steps

    # **`pretrained_path` は学習コンテナの中のパスのまま残っている。** π0.5 は
    # PaliGemma を設定から組んでから safetensors を被せる作りなので
    # （lerobot v0.6.0 の PaliGemmaWithExpertModel.__init__）ここを見に行くことは
    # 無いが、別の機械へ持って行ったときに誤解を生むので消しておく。
    config.pretrained_path = None
    return config


def _build_empty_policy(policy_class, config, torch_module):
    """重みを入れる前の器を、**軽い dtype で**組み立てる。

    本体の道は fp32 で作ってから bf16 へ落とすため、一瞬 16.6 GB を要する
    （モジュール冒頭の表）。最初から bf16 の既定 dtype で作れば 8.3 GB で済み、
    **どうせ全テンソルをチェックポイントの値で置き換えるので出来上がりは同じ**に
    なる。

    Parameters
    ----------
    policy_class : type
        `PI05Policy`。
    config : PreTrainedConfig
        読み込んだ設定。`dtype` を既定 dtype に使う。

    Returns
    -------
    policy : PI05Policy
        重みが初期値のままの方策。
    """
    default_dtype = (torch_module.bfloat16 if config.dtype == "bfloat16"
                     else torch_module.float32)
    previous_dtype = torch_module.get_default_dtype()
    torch_module.set_default_dtype(default_dtype)
    try:
        return policy_class(config)
    finally:
        torch_module.set_default_dtype(previous_dtype)


def _stream_weights(policy, weights: Path) -> dict[str, int]:
    """`model.safetensors` を 1 テンソルずつ模型へ移し、突き合わせの結果を返す。

    `param.data` に**代入**する（`copy_` ではない）。代入ならチェックポイント側の
    dtype がそのまま採用されるので、fp32 で保存されている vision_tower や
    layernorm は fp32 に戻り、**本体の道と同じ dtype の並びになる。**

    Parameters
    ----------
    policy : PI05Policy
        `_build_empty_policy` で作った器。
    weights : Path
        `pretrained_model` ディレクトリ。

    Returns
    -------
    tally : dict[str, int]
        `copied`（写した数）、`unknown`（模型に無いキーの数）、
        `missing_parameters`（写されなかった重みの数。**0 でなければ異常**）、
        `missing_buffers`（写されなかった buffer の数。rotary の inv_freq など
        保存対象でないものが入るので、0 でなくても異常ではない）。

    Reads:
      - <weights>/model.safetensors
    """
    from safetensors import safe_open

    parameters = dict(policy.named_parameters())
    buffers = dict(policy.named_buffers())
    targets = {**parameters, **buffers}

    copied: set[str] = set()
    unknown: list[str] = []
    mismatched: list[str] = []

    with safe_open(weights / "model.safetensors", framework="pt", device="cpu") as reader:
        for key in reader.keys():
            target = targets.get(key)
            if target is None:
                unknown.append(key)
                continue
            tensor = reader.get_tensor(key)
            if tuple(tensor.shape) != tuple(target.shape):
                mismatched.append(f"{key}: {tuple(tensor.shape)} ≠ {tuple(target.shape)}")
                continue
            target.data = tensor
            copied.add(key)

    if mismatched:
        raise SystemExit("エラー: 形の合わないテンソルがあります\n  "
                         + "\n  ".join(mismatched[:5]))

    return {
        "copied": len(copied),
        "unknown": len(unknown),
        "missing_parameters": len(set(parameters) - copied),
        "missing_buffers": len(set(buffers) - copied),
    }


def load_policy(weights: Path, device: str, *, load_mode: str = "lowmem",
                n_action_steps: int = 0, num_inference_steps: int = 0,
                tokenizer: str = "auto"):
    """チェックポイントを読み、方策と前後処理を組み立てて返す。

    **`replay_pi05_on_dataset.py` からも呼ぶので `_` を付けない。**

    Parameters
    ----------
    weights : Path
        `pretrained_model` ディレクトリ。
    device : str
        `cpu` または `cuda`。
    load_mode : str
        `lowmem`（既定、ピーク 10GB 前後）または `builtin`（18GB 級）。
    n_action_steps, num_inference_steps : int
        0 ならチェックポイントの値のまま。
    tokenizer : str
        `auto` なら `$HF_HOME` を探す。

    Returns
    -------
    policy : PI05Policy
        `eval()` 済みの方策。
    preprocessor : PolicyProcessorPipeline
        rename / batch / normalize / tokenize / device をまとめたもの。
    postprocessor : PolicyProcessorPipeline
        行動を逆正規化して CPU に戻すもの。
    tally : dict[str, int] | None
        重みの突き合わせの結果（`load_mode="builtin"` では None）。
    """
    import torch

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    config = _read_config(weights, device, n_action_steps, num_inference_steps)

    tally: dict[str, int] | None = None
    if load_mode == "builtin":
        policy = PI05Policy.from_pretrained(weights, config=config)
    else:
        policy = _build_empty_policy(PI05Policy, config, torch)
        tally = _stream_weights(policy, weights)

    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        str(weights),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": _resolve_tokenizer(tokenizer)},
        },
    )
    return policy, preprocessor, postprocessor, tally


def _make_dummy_observation() -> dict[str, np.ndarray]:
    """評価パイプラインが渡す形の観測を、真っ黒な画像で 1 つ作る。

    キーと形は policy_server.py の BasePolicy.get_action の仕様に合わせてある。
    姿勢は原点・回転なし（クォータニオンの実部だけ 1）で、掴んでいない状態。
    """
    return {
        "agentview_image": np.zeros((_CAMERA_SIZE, _CAMERA_SIZE, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((_CAMERA_SIZE, _CAMERA_SIZE, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def _build_batch(observation: dict[str, np.ndarray], instruction: str,
                 env_preprocessor, preprocessor, torch_module) -> dict:
    """LIBERO の生の観測を、方策が食べられる batch に直す。

    policy_server.py の MyPolicy._build_batch と同じ経路を通す（画像の 180 度
    回転と 8 次元 state の組み立ては LiberoProcessorStep が行う）。
    """
    def to_image(image: np.ndarray):
        tensor = torch_module.from_numpy(np.ascontiguousarray(image))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(dtype=torch_module.float32) / _IMAGE_SCALE

    def to_batched(values: np.ndarray):
        tensor = torch_module.from_numpy(np.ascontiguousarray(values))
        return tensor.to(dtype=torch_module.float32).unsqueeze(0)

    raw = {
        "observation.images.front": to_image(observation["agentview_image"]),
        "observation.images.wrist": to_image(observation["robot0_eye_in_hand_image"]),
        "observation.robot_state": {
            "eef": {
                "pos": to_batched(observation["robot0_eef_pos"]),
                "quat": to_batched(observation["robot0_eef_quat"]),
            },
            "gripper": {"qpos": to_batched(observation["robot0_gripper_qpos"])},
        },
        "task": [instruction],
    }
    return preprocessor(env_preprocessor(raw))


def _print_run_header(weights: Path, device: str, config,
                      tally: dict[str, int] | None) -> None:
    """何をどの設定で動かすのかを、1 か所にまとめて出す。"""
    print(f"[infer] 重み       {weights}")
    print(f"[infer] device     {device}   dtype {config.dtype}")
    print(f"[infer] chunk_size {config.chunk_size}   "
          f"n_action_steps {config.n_action_steps}   "
          f"num_inference_steps {config.num_inference_steps}")
    if tally is not None:
        print(f"[infer] 重みの突き合わせ  写した {tally['copied']} 個 / "
              f"模型に無いキー {tally['unknown']} 個 / "
              f"写されなかった重み {tally['missing_parameters']} 個 "
              f"(buffer {tally['missing_buffers']} 個)")
    print("[infer] 観測の受け口（config.json の input_features）")
    for key, feature in config.input_features.items():
        print(f"           {key}  {tuple(feature.shape)}")


def _check_camera_names(config) -> list[str]:
    """カメラ名が学習データ側へ差し替わっているかを見て、問題を並べて返す。

    `lerobot/pi05_libero_base` の素の設定は `image` / `image2` である。学習データ
    （`lerobot/libero_plus`）の列名は `front` / `wrist` で、**差し替わっていなければ
    評価側の観測を渡す先が無い。**
    """
    keys = set(config.input_features)
    return [f"{expected} が受け口にありません（実際: {sorted(keys)}）"
            for expected in _EXPECTED_CAMERAS if expected not in keys]


def _peak_memory_gb(device: str) -> float:
    """ピークで使ったメモリ量（GB）。cuda なら GPU 側、cpu ならプロセスの RSS。"""
    if device == "cuda":
        import torch

        return torch.cuda.max_memory_allocated() / 2**30
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _run_steps(policy, preprocessor, postprocessor, env_preprocessor,
               instruction: str, steps: int, torch_module) -> list[np.ndarray]:
    """行動を `steps` 歩ぶん出し、1 歩ごとの時間を表にして出す。

    Parameters
    ----------
    steps : int
        `select_action` を呼ぶ回数。**`n_action_steps` の倍数の歩でだけ**
        重い推論が走り、残りは作り置きの列を出すだけである。

    Returns
    -------
    actions : list[np.ndarray]
        各歩の行動 `(7,)`。
    """
    observation = _make_dummy_observation()
    policy.reset()

    print(f"\n[infer] 指示文: {instruction!r}")
    print(f"  {'歩':>3}  {'秒':>8}  行動 (7,)")

    actions: list[np.ndarray] = []
    for step in range(1, steps + 1):
        batch = _build_batch(observation, instruction, env_preprocessor,
                             preprocessor, torch_module)
        started = time.perf_counter()
        with torch_module.no_grad():
            action = policy.select_action(batch)
        action = postprocessor(action)
        elapsed = time.perf_counter() - started

        values = np.asarray(action[0].detach().to("cpu").numpy(),
                            dtype=np.float32).reshape(7)
        actions.append(values)
        formatted = " ".join(f"{value:+.3f}" for value in values)
        print(f"  {step:>3}  {elapsed:>8.2f}  [{formatted}]", flush=True)

    return actions


def _collect_problems(policy, tally: dict[str, int] | None,
                      actions: list[np.ndarray]) -> list[str]:
    """結果を検分して、異常の一覧を返す。空なら合格である。"""
    problems = _check_camera_names(policy.config)

    if tally is not None:
        if tally["copied"] == 0:
            problems.append("重みを 1 つも写せていません。model.safetensors を"
                            "読めているか確認してください")
        if tally["missing_parameters"] > 0:
            problems.append(f"チェックポイントに無い重みが "
                            f"{tally['missing_parameters']} 個あります。"
                            "**初期値のまま推論しています**")
        if tally["unknown"] > 0:
            problems.append(f"模型側に受け皿の無いキーが {tally['unknown']} 個"
                            "あります。設定と重みが噛み合っていません")

    stacked = np.stack(actions)
    if stacked.shape[1:] != (7,):
        problems.append(f"行動の形が (7,) ではありません: {stacked.shape}")
    if not np.all(np.isfinite(stacked)):
        problems.append("行動に NaN または Inf が混ざっています")
    if np.all(stacked == 0.0):
        problems.append("行動が全部 0 です。重みか正規化統計を読めていません")
    return problems


def main() -> None:
    args = _parse_args()
    weights: Path = args.weights.resolve()
    if not (weights / "config.json").is_file():
        raise SystemExit(f"エラー: {weights} に config.json がありません")

    # 採点環境と同じ「外へ出ない」条件で試す。トークナイザが $HF_HOME に
    # 無い場合だけ、_resolve_tokenizer が断ってから取りに行く。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch

    device = resolve_device(args.device)
    if device == "cpu" and args.load_mode == "builtin":
        print("[infer] **注意: --load-mode builtin はピーク 18GB 級を要求する。**"
              "学習と同時に動かすなら lowmem にすること（モジュール冒頭の表）")

    loading_started = time.perf_counter()
    policy, preprocessor, postprocessor, tally = load_policy(
        weights, device, load_mode=args.load_mode,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        tokenizer=args.tokenizer,
    )
    loading_seconds = time.perf_counter() - loading_started

    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.env_processor import LiberoProcessorStep

    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    _print_run_header(weights, device, policy.config, tally)
    print(f"[infer] 読み込み   {loading_seconds:.1f} 秒"
          f"（{args.load_mode}）")

    actions = _run_steps(policy, preprocessor, postprocessor, env_preprocessor,
                         args.instruction, args.steps, torch)

    print(f"\n[infer] ピークメモリ {_peak_memory_gb(device):.2f} GB（{device}）")

    problems = _collect_problems(policy, tally, actions)
    if problems:
        print("\n**問題あり**")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("\n  通った。このチェックポイントはそのまま推論に使える。")
    print("  **秒数はこの機械の速さである。**"
          "採点環境（L4 24GB・10 秒/推論）の値ではない")
    print("  記録用: " + json.dumps({
        "weights": str(weights),
        "device": device,
        "dtype": policy.config.dtype,
        "load_mode": args.load_mode,
        "n_action_steps": policy.config.n_action_steps,
        "num_inference_steps": policy.config.num_inference_steps,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
