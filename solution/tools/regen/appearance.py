#!/usr/bin/env python3
"""生成する手本に**外観（照明・机と背景・カメラ）を重ねる**ための、引き当ての家。

**なぜ要るか。**配布データの 14,347 本は、1,681 個の物理を平均 8.5 通りの見た目で
焼き直したものである（`contact_clean/map_demos.py`）。**そこへ素の見た目だけの手本を
4,000 本（約 24%）足すと、見た目の分布が素へ寄り、背景と照明だけで争う Track1 を払う**
（`docs/talk/inter_sessions/033_init_pose_dataset_4000.md` の「A の理由」）。

**追加費用はゼロである。**物理は `states[0]` で決まり、再生する行動列も同じなので、
変わるのは描画だけ——**照明・机・背景の bddl は素と `problem` の名前 1 行しか違わない**
（9/14 に 2 課題 × 3 変種を diff して確認）。

**変種名の組み立て**（`env_wrapper.py:207-237` が `_view_` で切って読む）:

    <課題><scene>_view_<方位>_<仰角>_<拡大率×100>_<レンズ回転>_<レンズ上下>_initstate_<番号>
          │                         └ 名前側で読む（一覧に無い組でもよい）
          └ bddl 側（`_light_N`・`_table_N`・`_tb_N` の bddl が実在するものだけ）

| 軸 | 引く母集団 | 備考 |
|---|---|---|
| `plain` | — | 素の見た目 |
| `light` | その課題の `_light_N` | Track1 の本体 |
| `background` | その課題の `_table_N` と `_tb_N` を合わせたもの | Track1 の本体 |
| `camera` | その課題の Camera Viewpoints の `view` | 配布データの 32 本中 7 本が別位置（`reference/06`） |

**雑音（`_noise_N`）は重ねない。**`ControlEnv.step`/`reset` の後処理でしか掛からず、
`_Rollout` は内側の `env.env.step` を呼ぶので**記録するコマに乗らない。**しかも学習側に
同じ 5 系統のぼかしを当てる拡張がすでにある（`solution/train/` の sensor_noise・f3f82d5）。

**一覧はコンテナの `task_classification.json` から毎回作る**（書き写さない——
LIBERO-plus を差し替えたときに黙って食い違うため）。
"""
from __future__ import annotations

import functools
import json
import random
import re
from pathlib import Path

KINDS = ("plain", "light", "background", "camera")

DEFAULT_WEIGHTS = {"plain": 25.0, "light": 25.0, "background": 25.0, "camera": 25.0}
"""既定の配り方。

**配布データの内訳は測れていない**（variant 名が残っておらず、明るさからは照明と背景しか
読めない・`reference/08`）。そこで**素を 4 分の 1 に抑え**（033 の門「素 40% 以下」の内側）、
Track1 の本体である照明と背景を厚くし、カメラは配布データで測った割合（22%）に近い 25% にした。
"""


@functools.cache
def catalog() -> dict[str, dict[str, list[str]]]:
    """課題 → 軸 → **重ねる文字列**（`scene` は bddl 側の接尾辞、`camera` は view 文字列）。"""
    from libero.libero import get_libero_path

    root = Path(get_libero_path("bddl_files"))
    classification = json.loads(
        (root.parent / "benchmark" / "task_classification.json").read_text())
    base_names = {path.stem for suite in classification
                  for path in (root / suite).glob("*.bddl")}
    table: dict[str, dict[str, list[str]]] = {}

    def entry(task: str) -> dict[str, list[str]]:
        return table.setdefault(task, {"light": [], "background": [], "camera": []})

    for suite, entries in classification.items():
        for item in entries:
            name = item["name"]
            if match := re.fullmatch(r"(.+)_(light|table|tb)_(\d+)", name):
                task, axis, number = match.groups()
                if task in base_names and (root / suite / f"{name}.bddl").exists():
                    kind = "light" if axis == "light" else "background"
                    entry(task)[kind].append(f"_{axis}_{number}")
            elif item["category"] == "Camera Viewpoints" and (
                    match := re.fullmatch(r"(.+)_view_([-\d]+(?:_[-\d]+){4})_initstate_\d+", name)):
                task, view = match.groups()
                if task in base_names:
                    entry(task)["camera"].append(view)
    return {task: {kind: sorted(set(values)) for kind, values in axes.items()}
            for task, axes in table.items()}


def parse_weights(text: str) -> dict[str, float]:
    """`plain=25,light=25,background=25,camera=25` を読む。`off` は素だけ。"""
    if text.strip() == "off":
        return {"plain": 1.0}
    weights = {}
    for part in text.split(","):
        kind, _, value = part.partition("=")
        if kind.strip() not in KINDS:
            raise ValueError(f"外観の軸は {KINDS} のどれか: {kind!r}")
        weights[kind.strip()] = float(value)
    return weights


def draw(task: str, key: str, weights: dict[str, float]) -> dict[str, str]:
    """1 つの単位に当てる外観を引く。**`key` が同じなら毎回同じものが出る。**

    その課題に無い軸が当たったら素にする（例: `_tb_N` が 0 個の課題）。

    Returns
    -------
    dict
        `kind`（`KINDS` のどれか）・`scene`（bddl 側の接尾辞。素なら ""）・
        `view`（カメラの 5 つ組。素なら "0_0_100_0_0"）。
    """
    rng = random.Random(key)
    kinds = [kind for kind in KINDS if weights.get(kind, 0.0) > 0.0]
    kind = rng.choices(kinds, weights=[weights[k] for k in kinds])[0]
    choices = catalog().get(task, {}).get(kind, []) if kind != "plain" else []
    if kind != "plain" and not choices:
        kind = "plain"
    picked = rng.choice(choices) if choices else ""
    return {"kind": kind,
            "scene": picked if kind in ("light", "background") else "",
            "view": picked if kind == "camera" else "0_0_100_0_0"}
