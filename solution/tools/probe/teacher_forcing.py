#!/usr/bin/env python3
"""学習データの 1 コマを入れ直して、**記録された行動を再現できるか**を測る。

## なぜこの測りが要るか

`t3rev` は逆向きデータ 397 本を **35 周**見ている。**それでも評価では再現しない。**
ここには 2 つしか可能性がない。

| 結果 | 読み | 次に打つ手 |
|---|---|---|
| 記録どおりの行動を返す | **学習は入っている。**壊れているのは**評価時の入力の通り道** | 観測と行動の突き合わせ |
| 返さない | **最適化が届いていない** | 学習率と warmup を変える |

**この 1 回の測定で探索空間が半分になる。**設計は
`docs/plan/22_grasp_and_color_next_moves.md` の 3.2 節。

## 何をするか —— 学習はしない。推論 1 回ずつである

**学習が使ったのと同じ `LeRobotDataset` から 1 行取る**ので、
ここで食い違いが出たら、それは**学習時の入力そのもの**で食い違っている。

1. `delta_timestamps` で `action` を 50 コマぶん引く（学習と同じ引き方）
2. 画・姿勢・指示文を方策へ入れ、noise を固定して 50 歩を出す
3. 逆正規化して、**記録されている 50 歩**と比べる

## 出す数と、その読み方

| 数 | 意味 | 再現できているとき |
|---|---|---|
| `rel_l2` | 記録との相対 L2（`‖予測 − 記録‖ / ‖記録‖`） | **小さい** |
| `rel_l2_chance` | **何も学ばない予測**（正規化空間の 0＝データの平均）の相対 L2 | 比較の基準。`rel_l2` がこれを下回らなければ**学習は入っていない** |
| `cos_translation` | 50 歩を積んだ並進の向きの一致 | **+1 に近い** |
| `grip_agree` | 指の符号が一致したコマの割合 | **1.0 に近い** |

**`rel_l2_chance` を必ず一緒に出す。**flow matching の出力は分布の平均へ寄るので、
**「平均を出しただけ」でも `rel_l2` はそれなりに小さく見える。**基準が無いと読めない。

## 使い方

    bash solution/tools/probe/run_teacher_forcing.sh <重み> <データセット> [出力json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# **`infer_pi05_once` は viz の下にある。**`lineage_probe.py` と同じ解決をする。
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "solution/tools/viz"))


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(pred - ref) / denom) if denom else float("nan")


def action_mean_std(weights: Path, fallback: Path | None):
    """行動の逆正規化に使う mean / std を、その重み自身から取る。

    **重みごとに尺度が違う**ので共通の尺度で戻してはいけない。
    統計を持たない重み（`pi05_libero_base`）だけ `fallback` から借りる。
    """
    from safetensors.numpy import load_file
    for root in [weights, fallback]:
        if root is None:
            continue
        for name in ("policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                     "policy_preprocessor_step_2_normalizer_processor.safetensors"):
            path = root / name
            if not path.is_file():
                continue
            blob = load_file(str(path))
            mean = next(v for k, v in blob.items() if k.endswith("action.mean"))
            std = next(v for k, v in blob.items() if k.endswith("action.std"))
            return mean.reshape(-1), std.reshape(-1), str(path)
    raise SystemExit("行動の統計が見つからない")


def translation_direction(act: np.ndarray) -> np.ndarray:
    """50 歩の並進を積んだ向き（単位ベクトル）。"""
    total = act[:, :3].sum(axis=0)
    norm = float(np.linalg.norm(total))
    return total / norm if norm else total


def compare(pred: np.ndarray, ref: np.ndarray, mean, std) -> dict:
    """予測と記録を、4 つの数で比べる。"""
    chance = np.broadcast_to(mean[: ref.shape[1]], ref.shape)
    d_pred, d_ref = translation_direction(pred), translation_direction(ref)
    grip_pred, grip_ref = np.sign(pred[:, 6]), np.sign(ref[:, 6])
    return {
        "rel_l2": relative_l2(pred, ref),
        "rel_l2_chance": relative_l2(chance, ref),
        "cos_translation": float(d_pred @ d_ref),
        "grip_agree": float((grip_pred == grip_ref).mean()),
        "reach_mm_pred": float(np.abs(pred[:, :3]).sum() * 1000),
        "reach_mm_ref": float(np.abs(ref[:, :3]).sum() * 1000),
    }


def pick_frames(dataset, per_task: int, offsets: tuple[int, ...]) -> list[int]:
    """課題ごとに `per_task` 本のエピソードを取り、各本から `offsets` のコマを引く。

    **序盤だけを見てはいけない。**逆向きデータの先頭は「開いたまま接近」なので、
    掴む瞬間（中盤）を外すと「掴む気があるか」が測れない。
    """
    meta = dataset.meta
    by_task: dict[str, list[int]] = {}
    for episode_index in range(meta.total_episodes):
        row = meta.episodes[episode_index]
        task = row["tasks"][0] if isinstance(row["tasks"], (list, np.ndarray)) else row["tasks"]
        by_task.setdefault(str(task), []).append(episode_index)

    picked: list[int] = []
    for task, episodes in by_task.items():
        for episode_index in episodes[:per_task]:
            start = int(dataset.episode_data_index["from"][episode_index])
            end = int(dataset.episode_data_index["to"][episode_index])
            length = end - start
            for fraction in offsets:
                index = start + min(int(length * fraction / 100), length - 1)
                picked.append(index)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-mode", default="lowmem", choices=("lowmem", "builtin"))
    ap.add_argument("--tokenizer", default="auto")
    ap.add_argument("--fallback-norm", type=Path, default=REPO / "work/artifacts/pi05_rp3000")
    ap.add_argument("--episodes-per-task", type=int, default=2)
    ap.add_argument("--offsets", default="0,25,50,75",
                    help="エピソードの何 %% の位置を引くか（コンマ区切り）")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from infer_pi05_once import load_policy, resolve_device

    device = resolve_device(args.device)

    tokenizer = args.tokenizer
    if tokenizer == "auto":
        for root in (args.weights, args.fallback_norm):
            if root and (root / "vlm/tokenizer.json").is_file():
                tokenizer = str(root / "vlm")
                break

    # **統計を持たない重みは借りる**（判定は `.safetensors` の有無で行う。
    # `pi05_libero_base` は json だけ持っていて、統計が空のまま組まれる）。
    proc_root = features_from = None
    load_mode = args.load_mode
    if not list(args.weights.glob("policy_preprocessor_step_*normalizer*.safetensors")):
        proc_root = features_from = args.fallback_norm
        print(f"  **前後処理と列の名前を借りる: {proc_root}**")
    from safetensors import safe_open
    with safe_open(args.weights / "model.safetensors", framework="np") as handle:
        has_prefix = any(key.startswith("model.") for key in handle.keys())
    if not has_prefix and load_mode == "lowmem":
        load_mode = "builtin"
        print("  **本体の loader に切り替える**（`model.` 接頭辞が無い重みである）")

    print(f"=== {args.weights} を {device} で読む ===", flush=True)
    policy, pre, _post, tally = load_policy(args.weights, device,
                                            load_mode=load_mode,
                                            tokenizer=tokenizer,
                                            processor_root=proc_root,
                                            features_from=features_from)
    if tally:
        print(f"  重みの突き合わせ: {tally}")
    mean, std, origin = action_mean_std(args.weights, args.fallback_norm)
    print(f"  行動の逆正規化: {origin}")

    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim

    # **学習と同じ引き方で action を 50 コマぶん引く。**
    dataset = LeRobotDataset("local", root=str(args.dataset),
                             delta_timestamps={"action": [i / 20.0 for i in range(chunk_size)]})
    print(f"  データセット: {dataset.meta.total_episodes} 本 / "
          f"{dataset.meta.total_frames} コマ / fps {dataset.meta.fps}")

    offsets = tuple(int(x) for x in args.offsets.split(","))
    indices = pick_frames(dataset, args.episodes_per_task, offsets)
    print(f"  測るコマ: {len(indices)} 個"
          f"（課題ごと {args.episodes_per_task} 本 × 位置 {offsets} %）")

    results = []
    for count, index in enumerate(indices, 1):
        item = dataset[index]
        task = item["task"]
        batch = {
            "observation.images.front": item["observation.images.front"].unsqueeze(0),
            "observation.images.wrist": item["observation.images.wrist"].unsqueeze(0),
            "observation.state": item["observation.state"].unsqueeze(0),
            "task": [task],
        }
        batch = pre(batch)
        generator = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn((1, chunk_size, max_action_dim), generator=generator)
        noise = noise.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            predicted = policy.predict_action_chunk(batch, noise=noise)
        predicted = predicted[0, :, :7].to(torch.float32).cpu().numpy()
        predicted = predicted * std[:7] + mean[:7]
        recorded = item["action"][:, :7].to(torch.float32).cpu().numpy()

        row = {"index": index, "task": task}
        row.update(compare(predicted, recorded, mean, std))
        results.append(row)
        print(f"  [{count:3d}/{len(indices)}] rel_l2 {row['rel_l2']:.3f} "
              f"(何も学ばない予測 {row['rel_l2_chance']:.3f})  "
              f"並進 cos {row['cos_translation']:+.3f}  "
              f"指の一致 {row['grip_agree']:.2f}  {task[:38]}", flush=True)

    print()
    print("=" * 92)
    print(f"【まとめ】{args.weights.name}")
    print("=" * 92)
    by_task: dict[str, list[dict]] = {}
    for row in results:
        by_task.setdefault(row["task"], []).append(row)
    header = f"  {'課題':46}{'rel_l2':>8}{'基準':>8}{'並進cos':>9}{'指一致':>8}{'判定':>10}"
    print(header)
    for task, rows in by_task.items():
        value = float(np.mean([r["rel_l2"] for r in rows]))
        chance = float(np.mean([r["rel_l2_chance"] for r in rows]))
        cos = float(np.mean([r["cos_translation"] for r in rows]))
        grip = float(np.mean([r["grip_agree"] for r in rows]))
        verdict = "再現" if value < chance * 0.7 else ("基準以下" if value < chance else "**基準より悪い**")
        print(f"  {task[:46]:46}{value:>8.3f}{chance:>8.3f}{cos:>+9.3f}{grip:>8.2f}{verdict:>10}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"weights": str(args.weights), "dataset": str(args.dataset),
             "unnormalizer": origin, "frames": results}, indent=1, ensure_ascii=False))
        print(f"\n  書いた: {args.json_out}")


if __name__ == "__main__":
    main()
