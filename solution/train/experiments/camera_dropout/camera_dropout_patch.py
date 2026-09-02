#!/usr/bin/env python3
"""場面カメラ（机を写す固定カメラ）を、学習中だけ確率で真っ黒にする。

**なぜ設定では書けないのか。** 同梱の LeRobot は画像の変換をカメラの列ごとに
呼ぶが、**どの列を渡しているのかを変換関数に教えない**
（`lerobot/datasets/dataset_reader.py` の `get_item`）::

    for cam in self._meta.camera_keys:
        item[cam] = self._image_transforms(item[cam])

だから torchvision の変換をいくら並べても「`front` だけ」は表せない。
**列の名前を見て分岐できる場所は `get_item` の外側しか無いので、そこを包む。**

**なぜ真っ黒（0）なのか。** π0.5 自身が「カメラが無い」を表すのに使う値と
同じにするためである。`lerobot/policies/pi05/modeling_pi05.py` は画像を
`img * 2.0 - 1.0` で [0,1] から [-1,1] へ写し、**足りないカメラを
`torch.ones_like(img) * -1` で埋める**（1207〜1211 行）。
**0 を渡すと正規化後に -1 になり、その埋め草と一致する。**

**マスクは 1 のままにする（ここが要点である）。** 同じ箇所で π0.5 は、
埋め草のカメラには `mask = 0` を立てる。**こちらはそれを真似しない。**
理由は採点のときの形に合わせるためである——**採点では場面カメラは必ず在って、
ただし視点がずれている**（Track2 の `Camera Viewpoints`）。**つまりマスクは 1 で、
中身が当てにならない、という状態しか起こらない。**
学習で `mask = 0` を使うと、モデルは「マスクが 0 なら無視する」を覚えるが、
**採点ではそのマスクが立たないので、覚えたことが発火しない。**
**マスクを 1 のまま中身を潰せば、「場面カメラが当てにならないときは手首に頼る」
を、採点で実際に起こる形のまま学べる。**

**黙って効かないことがいちばん悪いので、そうならないようにしてある。**
列の名前が 1 つも合わなければ**その場で止める**（4 時間 46 分回してから
「効いていなかった」と気づくのを防ぐ）。効いていることは、
`[camera_dropout]` の行がログに出ることで確かめられる。

環境変数（`experiments/camera_dropout/__init__.py` の `set_environment` が立てる）::

    CAMERA_DROPOUT_PROB=0.5                        1 コマごとに潰す確率
    CAMERA_DROPOUT_KEY=observation.images.front    潰す列の名前
"""
from __future__ import annotations

import os

import torch

# 何コマごとに「実際に潰した割合」を出すか。**確率が効いていることを、
# 立ち上がりの 1 行だけでなく実測値でも確かめられるようにするためにある**
# （読み込みの worker ごとに数えるので、行数は worker 数ぶんに分かれる）。
_REPORT_EVERY = 20_000


def apply() -> None:
    """`DatasetReader.get_item` を包んで、場面カメラを確率で潰すようにする。

    **`CAMERA_DROPOUT_PROB` が無いか 0 のときは何もしない**（包まない）。
    """
    probability = float(os.environ.get("CAMERA_DROPOUT_PROB", "0") or "0")
    if probability <= 0.0:
        return

    key = os.environ.get("CAMERA_DROPOUT_KEY", "observation.images.front")

    from lerobot.datasets.dataset_reader import DatasetReader

    original = DatasetReader.get_item
    # worker ごとに別プロセスなので、この 2 つは worker ごとに独立に数える。
    state = {"seen": 0, "dropped": 0, "announced": False}

    def get_item_with_camera_dropout(self, idx):
        item = original(self, idx)

        if not state["announced"]:
            state["announced"] = True
            if key not in item:
                # **黙って効かないのを防ぐ。**列の名前が違っていても学習自体は
                # 通ってしまい、4 時間 46 分後に「効かなかった」という誤った
                # 結論だけが残る。だからここで止める。
                available = sorted(k for k in item if k.startswith("observation.images"))
                raise SystemExit(
                    f"[camera_dropout] 潰す列 {key!r} がデータの中に無い。\n"
                    f"  このデータにある画像の列: {available}\n"
                    "  --scene-camera-key で正しい名前を渡すこと。"
                )
            info = torch.utils.data.get_worker_info()
            worker = "main" if info is None else f"worker{info.id}"
            print(f"[camera_dropout] {worker}: {key} を確率 {probability} で潰す"
                  f"（画像の列は {sorted(k for k in item if k.startswith('observation.images'))}）",
                  flush=True)

        state["seen"] += 1
        # **torch の乱数を使う。**DataLoader が worker ごとに base_seed から
        # 別々に、しかも決まった手順で種を振るので、`--seed` を固定すれば
        # 同じ並びが再現する（Python の `random` は worker 間で同じ種になりうる）。
        if torch.rand(()).item() < probability:
            item[key] = torch.zeros_like(item[key])
            state["dropped"] += 1

        if state["seen"] % _REPORT_EVERY == 0:
            rate = state["dropped"] / state["seen"]
            print(f"[camera_dropout] {state['seen']} コマ中 {state['dropped']} コマを潰した"
                  f"（実測 {rate:.3f}・指定 {probability}）", flush=True)

        return item

    DatasetReader.get_item = get_item_with_camera_dropout
