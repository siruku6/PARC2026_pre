#!/usr/bin/env python3
"""場面カメラ（机を写す固定カメラ）にだけ、学習中に確率で blur を掛ける。

**なぜ設定では書けないのか。**同梱の LeRobot は画像の変換をカメラの列ごとに呼ぶが、
どの列を渡しているのかを変換関数に教えない（`lerobot/datasets/dataset_reader.py` の
`get_item`）。**採点は場面カメラだけを劣化させ、手首カメラは無傷である**
（LIBERO-plus の `env_wrapper.py` の `step`・`reset`）。だから `camera_dropout_patch.py`
と同じく、列の名前を見て分岐できる `get_item` の外側を包む。

**採点と同じ形にそろえてあるところ。**

| 採点（`env_wrapper.py`） | ここ |
|---|---|
| 1 本のエピソードに 1 種類だけ掛ける | 1 コマに 1 種類だけ選ぶ（重ねない） |
| 128×128 に描いた画像に掛ける | 256×256 を 128×128 に縮めて掛け、元の大きさへ戻す |
| 強度 1〜10 | 1〜10 から一様に引く |
| 場面カメラだけ | `SENSOR_NOISE_KEY` の列だけ。手首は触らない |

**黙って効かないことがいちばん悪いので、そうならないようにしてある。**列の名前が
合わなければ最初の 1 コマで止め、効いていることは `[sensor_noise]` の行で確かめられる
（種類ごとに実際に掛けた回数を出す）。

環境変数（`experiments/sensor_noise/__init__.py` の `set_environment` が立てる）::

    SENSOR_NOISE_PROB=0.3                              1 コマごとに掛ける確率
    SENSOR_NOISE_KEY=observation.images.front          掛ける列の名前
    SENSOR_NOISE_SIZE=128                              掛けるときの一辺（採点の描画の大きさ）
    SENSOR_NOISE_WEIGHTS=motion_blur=0.3,gaussian_blur=0.25,zoom_blur=0.25,fog=0.1,glass_blur=0.1
"""
from __future__ import annotations

import os

import numpy as np
import torch

# 何コマごとに「実際に掛けた割合」を出すか（camera_dropout_patch.py と同じ間隔）。
_REPORT_EVERY = 20_000

DEFAULT_WEIGHTS = "motion_blur=0.3,gaussian_blur=0.25,zoom_blur=0.25,fog=0.1,glass_blur=0.1"


def parse_weights(text: str) -> dict[str, float]:
    """`name=weight,...` を辞書にする。知らない種類と負の重みはその場で止める。"""
    from experiments.sensor_noise.corruptions import FAMILIES

    weights: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in FAMILIES:
            raise SystemExit(f"[sensor_noise] 知らない種類 {name!r}。使えるのは {FAMILIES}")
        weight = float(value)
        if weight < 0:
            raise SystemExit(f"[sensor_noise] 重みは 0 以上である: {name}={weight}")
        weights[name] = weight
    if not weights or sum(weights.values()) <= 0:
        raise SystemExit(f"[sensor_noise] 重みが 1 つも正でない: {text!r}")
    return weights


def corrupt_tensor(image: torch.Tensor, family: str, severity: int,
                   rng: np.random.Generator, size: int) -> torch.Tensor:
    """(C, H, W) の float [0, 1] 画像を、一辺 `size` に縮めて劣化させ、元の大きさへ戻す。

    Parameters
    ----------
    image : torch.Tensor
        LeRobot が読み込んだ 1 コマ。(C, H, W) か、時間の軸が付いた (T, C, H, W)。
    family, severity, rng
        `corruptions.corrupt` にそのまま渡す。
    size : int
        劣化を掛けるときの一辺。採点の描画の大きさ（128）にそろえる。

    Returns
    -------
    corrupted : torch.Tensor
        入力と同じ形・同じ dtype。
    """
    import cv2

    from experiments.sensor_noise.corruptions import corrupt

    if image.ndim == 4:
        return torch.stack([corrupt_tensor(frame, family, severity, rng, size)
                            for frame in image])
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    height, width = array.shape[:2]
    resized = (height, width) != (size, size)
    small = cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA) if resized else array
    corrupted = corrupt(np.ascontiguousarray(small), family, severity, rng)
    if resized:
        corrupted = cv2.resize(corrupted, (width, height), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(corrupted).permute(2, 0, 1).to(image.dtype) / 255.0


def apply() -> None:
    """`DatasetReader.get_item` を包んで、場面カメラに確率で blur を掛けるようにする。

    **`SENSOR_NOISE_PROB` が無いか 0 のときは何もしない**（包まない）。
    """
    probability = float(os.environ.get("SENSOR_NOISE_PROB", "0") or "0")
    if probability <= 0.0:
        return

    key = os.environ.get("SENSOR_NOISE_KEY", "observation.images.front")
    size = int(os.environ.get("SENSOR_NOISE_SIZE", "128"))
    weights = parse_weights(os.environ.get("SENSOR_NOISE_WEIGHTS", DEFAULT_WEIGHTS))
    families = list(weights)
    weight_tensor = torch.tensor([weights[name] for name in families], dtype=torch.float64)

    import cv2
    from lerobot.datasets.dataset_reader import DatasetReader

    # worker ごとに別プロセスで、1 プロセスが 1 コマずつ処理する。cv2 の内部スレッドは
    # 並列の worker と取り合うだけなので切る。
    cv2.setNumThreads(1)

    original = DatasetReader.get_item
    # worker ごとに別プロセスなので、この数は worker ごとに独立に数える。
    state = {"seen": 0, "corrupted": 0, "announced": False,
             "by_family": {name: 0 for name in families}}

    def get_item_with_sensor_noise(self, idx):
        item = original(self, idx)

        if not state["announced"]:
            state["announced"] = True
            if key not in item:
                # **黙って効かないのを防ぐ。**列の名前が違っていても学習自体は通って
                # しまい、何時間も後に「効かなかった」という誤った結論だけが残る。
                available = sorted(k for k in item if k.startswith("observation.images"))
                raise SystemExit(
                    f"[sensor_noise] 掛ける列 {key!r} がデータの中に無い。\n"
                    f"  このデータにある画像の列: {available}\n"
                    "  --scene-camera-key で正しい名前を渡すこと。"
                )
            info = torch.utils.data.get_worker_info()
            worker = "main" if info is None else f"worker{info.id}"
            print(f"[sensor_noise] {worker}: {key} に確率 {probability} で掛ける"
                  f"（一辺 {size}・重み {weights}・画像の列は "
                  f"{sorted(k for k in item if k.startswith('observation.images'))}）",
                  flush=True)

        state["seen"] += 1
        # **乱数はすべて torch から引く。**DataLoader が worker ごとに別の種を振るので、
        # `--seed` を固定すれば同じ並びが再現する。劣化の中で使う numpy の乱数も、
        # ここで引いた種から作る（numpy の大域乱数は worker 間で同じ状態になりうる）。
        if torch.rand(()).item() < probability:
            family = families[int(torch.multinomial(weight_tensor, 1).item())]
            severity = int(torch.randint(1, 11, ()).item())
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
            item[key] = corrupt_tensor(item[key], family, severity, rng, size)
            state["corrupted"] += 1
            state["by_family"][family] += 1

        if state["seen"] % _REPORT_EVERY == 0:
            rate = state["corrupted"] / state["seen"]
            print(f"[sensor_noise] {state['seen']} コマ中 {state['corrupted']} コマに掛けた"
                  f"（実測 {rate:.3f}・指定 {probability}・内訳 {state['by_family']}）",
                  flush=True)

        return item

    DatasetReader.get_item = get_item_with_sensor_noise
