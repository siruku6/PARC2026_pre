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
# **序盤だけを見てはいけない。**逆向きデータの先頭は「開いたまま接近」なので、
# 掴む瞬間（中盤）を外すと「掴む気があるか」が測れない。
OFFSET_PERCENTS = (0, 25, 50, 75)
# **`infer_pi05_once` は viz の下にある。**`lineage_probe.py` と同じ解決をする。
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "solution/tools/viz"))

from lineage_probe import (FALLBACK_NORM, action_std_mean,  # noqa: E402
                           head_direction, open_probe_weights)


def relative_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """記録に対する相対誤差。**分母は記録側**である。

    `lineage_probe.relative_l2` は**分母が第 1 引数**で、指示文を変えたときの
    変化量を測るためのものである。**同じ名前にすると片方を消したときに
    もう片方の数が黙って変わる**ので、名前を分けてある。
    """
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(pred - ref) / denom) if denom else float("nan")


def compare(pred: np.ndarray, ref: np.ndarray, mean: np.ndarray) -> dict:
    """予測と記録を、4 つの数で比べる。"""
    chance = np.broadcast_to(mean[: ref.shape[1]], ref.shape)
    return {
        "rel_l2": relative_error(pred, ref),
        "rel_l2_chance": relative_error(chance, ref),
        "cos_translation": float(head_direction(pred) @ head_direction(ref)),
        "grip_agree": float((np.sign(pred[:, 6]) == np.sign(ref[:, 6])).mean()),
    }


def pick_frames(dataset, per_task: int, offsets: tuple[int, ...]) -> list[int]:
    """課題ごとに `per_task` 本のエピソードを取り、各本から `offsets` のコマを引く。

    **序盤だけを見てはいけない。**逆向きデータの先頭は「開いたまま接近」なので、
    掴む瞬間（中盤）を外すと「掴む気があるか」が測れない。
    """
    meta = dataset.meta
    by_task: dict[str, list[dict]] = {}
    for row in meta.episodes:
        task = row["tasks"][0] if isinstance(row["tasks"], (list, np.ndarray)) else row["tasks"]
        by_task.setdefault(str(task), []).append(row)

    picked: list[int] = []
    for _task, rows in by_task.items():
        for row in rows[:per_task]:
            # **境界は `meta.episodes` の列から取る**（lerobot 0.6.0 に
            # `episode_data_index` は無い）。
            start = int(row["dataset_from_index"])
            length = int(row["dataset_to_index"]) - start
            for fraction in offsets:
                picked.append(start + min(int(length * fraction / 100), length - 1))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-mode", default="lowmem", choices=("lowmem", "builtin"))
    ap.add_argument("--tokenizer", default="auto")
    ap.add_argument("--fallback-norm", type=Path, default=FALLBACK_NORM)
    ap.add_argument("--episodes-per-task", type=int, default=2)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    import json as json_module

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # **重みを読む前に、安く落ちるものを先に落とす。**読み込みは 1〜2 分かかるので、
    # パスの打ち間違いで 2 分待たされるのは避ける（`run_final_eval.sh` の 8/28 の教訓）。
    config = json_module.loads((args.weights / "config.json").read_text())
    chunk_size = int(config["chunk_size"])
    max_action_dim = int(config["max_action_dim"])

    # **学習と同じ引き方で action を 50 コマぶん引く。**
    dataset = LeRobotDataset("local", root=str(args.dataset))
    fps = dataset.meta.fps
    dataset = LeRobotDataset("local", root=str(args.dataset),
                             delta_timestamps={"action": [i / fps for i in range(chunk_size)]})
    print(f"  データセット: {dataset.meta.total_episodes} 本 / "
          f"{dataset.meta.total_frames} コマ / fps {fps}")
    indices = pick_frames(dataset, args.episodes_per_task, OFFSET_PERCENTS)
    print(f"  測るコマ: {len(indices)} 個"
          f"（課題ごと {args.episodes_per_task} 本 × 位置 {OFFSET_PERCENTS} %）")

    policy, pre, _post, device, _tally = open_probe_weights(
        args.weights, args.fallback_norm, args.device,
        load_mode=args.load_mode, tokenizer=args.tokenizer)
    mean, std, origin = action_std_mean(args.weights, args.fallback_norm)
    if mean is None:
        raise SystemExit("**行動の統計が見つからない。**`--fallback-norm` を確かめること")
    print(f"  行動の逆正規化: {origin}")

    # **noise は 1 度だけ作る。**固定した同じ値を全コマで使う（flow matching）。
    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn((1, chunk_size, max_action_dim), generator=generator)
    noise = noise.to(device=device, dtype=torch.float32)

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
        with torch.inference_mode():
            predicted = policy.predict_action_chunk(batch, noise=noise)
        predicted = predicted[0, :, :7].to(torch.float32).cpu().numpy()
        predicted = predicted * std[:7] + mean[:7]
        recorded = item["action"][:, :7].to(torch.float32).cpu().numpy()

        row = {"index": index, "task": task}
        row.update(compare(predicted, recorded, mean))
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
