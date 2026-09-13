#!/usr/bin/env python3
"""初期姿勢の番号 1〜500 を、**実際の手先のずれ**で並べる。

**なぜ要るか。**番号は摂動量の順に並ばない。`new_init.py` は「乱数の単位ベクトル ×
倍率」で姿勢を作るので、**同じ倍率でも向きによって手先の動く量が 10 倍違う**
（`docs/reference/21_weakness_map.md`）。実際に `initstate_170`（倍率 0.2）は
**0.88 cm** しか動かず、`initstate_330`（倍率 0.4）は **3.93 cm** 動く。
**倍率で選ぶと、崩れていない範囲（2 cm 未満）ばかり作ってしまう。**

**大きさだけでなく、向きも測る**（9/13 に足した）。番号 1 つは手先を決まった向きへずらすので、
**向きを記録しておけば、崩れやすい向きの番号を多めに選んだり、偏りを打ち消す番号を選んだりできる**
（`docs/reference/21_weakness_map.md` の 2 節——崩れるかどうかは大きさより向きで決まる）。
手首の向きのずれ（回転）も同時に記録する。

**測るのは「方策が引き継ぐ時点」のずれである。**採点ハーネスは
`reset` → `set_state_from_flattened` → **ゼロ行動 10 歩**と進める。`.pruned_init` の
状態には腕の関節角も入っているので**腕はいったん摂動なしの姿勢に戻り**、
摂動が効くのはそのあとの 10 歩——**OSC が握ったままの目標へ腕が流れる分**である。
だから物理を 10 歩進めないと本当のずれは出ない。

用法（コンテナの中で打つ）:
    bash solution/tools/regen/run_in_container.sh \\
        solution/tools/regen/screen_initstates.py \\
        --out work/regen/initstate_offsets.json --workers 14

Reads:
  - LIBERO-plus の bddl と `.pruned_init`（コンテナの中）
Generates:
  - <out> … 番号ごとの `offsets_cm`（大きさ）・`vectors_cm`（向き 3 成分）・
    `rotation_deg`（手首の回転の角度）・`rotation_vectors_deg`（回転軸 × 角度）
  - 標準出力 … 帯ごとの本数・2 cm 以上の番号の一覧・向きの偏り
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripted_demo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASKS = (
    ("libero_10", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"),
    ("libero_spatial", "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"),
)
"""測る場面。**2 つ測るのは、ずれが場面に依らないことを確かめるためである。**"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, metavar="FILE")
    parser.add_argument("--first", type=int, default=1, metavar="N")
    parser.add_argument("--last", type=int, default=500, metavar="N")
    parser.add_argument("--workers", type=int, default=14, metavar="N")
    parser.add_argument("--layout", type=int, default=0, metavar="N",
                        help="使う初期配置の番号（`.pruned_init` の行）")
    parser.add_argument("--selection", type=Path, default=None, metavar="FILE",
                        help="選んだ番号の JSON（`{\"initstates\": [...]}`）。**渡すとその組の向きの偏りも出す**")
    return parser.parse_args()


def _measure(numbers: list[int], suite: str, task: str, layout: int
             ) -> dict[int, dict | None]:
    """番号ごとに、摂動なしの手先からのずれを測る。**位置は向きつき、手首は回転で。**

    回転は `motion.rotation.relative_axisangle`（world 合成 `R_now @ R_base^T`）で取る。
    **軸角ベクトルどうしの引き算は相対回転にならない**ので使わない。
    """
    import scripted_demo as sd
    from libero.libero import get_libero_path
    from motion.rotation import quaternion_to_axisangle, relative_axisangle

    libero_root = Path(get_libero_path("bddl_files")).parent
    init_states = sd._init_states(libero_root, suite, task)

    def pose_after_warmup(variant: str) -> tuple[np.ndarray, np.ndarray]:
        env = sd._make_env(sd._bddl_path(libero_root, suite, variant), False)
        try:
            env.seed(0)
            observation = sd.reset_to(env, init_states[layout])
            return (np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy(),
                    quaternion_to_axisangle(np.array(observation["robot0_eef_quat"],
                                                     dtype=np.float64)))
        finally:
            env.close()

    base_position, base_rotation = pose_after_warmup(task)
    out: dict[int, dict | None] = {}
    for number in numbers:
        variant = f"{task}_view_0_0_100_0_0_initstate_{number}"
        try:
            position, rotation = pose_after_warmup(variant)
            turn = relative_axisangle(base_rotation, rotation)
            out[number] = {
                "vector_cm": [round(100.0 * float(v), 3) for v in position - base_position],
                "rotation_vector_deg": [round(float(np.degrees(v)), 3) for v in turn]}
        except Exception as error:  # 姿勢が作れない番号は落として数える
            out[number] = None
            print(f"  initstate {number}: {type(error).__name__}", flush=True)
    return out


def _worker(chunk: list[int], suite: str, task: str, layout: int,
            results: "mp.Queue") -> None:
    results.put(_measure(chunk, suite, task, layout))


def _run_task(suite: str, task: str, numbers: list[int], layout: int,
              workers: int) -> dict[int, dict | None]:
    chunks = [list(part) for part in np.array_split(np.asarray(numbers), workers)
              if len(part)]
    results: "mp.Queue" = mp.Queue()
    procs = [mp.Process(target=_worker, args=(chunk, suite, task, layout, results))
             for chunk in chunks]
    for process in procs:
        process.start()
    merged: dict[int, dict | None] = {}
    for _ in chunks:
        merged.update(results.get())
    for process in procs:
        process.join()
    return merged


def direction_bias(vectors: np.ndarray) -> tuple[float, np.ndarray, float]:
    """ずれの**向き**の偏りを測る。**大きさを捨てて、単位ベクトルの平均の長さを見る。**

    0 なら全方向に均等、1 なら全部同じ向きである。**「無作為ならこのくらい」も返す**
    ——等方な 3 次元で N 本引いたときの平均の長さの期待値は約 `0.921 / √N` で、
    **これを下回る偏りは偏りと呼べない。**

    Returns
    -------
    bias : float
        単位ベクトルの平均の長さ。
    positive_pct : np.ndarray
        軸ごとの「正の側の割合」[%]（左右 x・前後 y・上下 z）。
    expected : float
        無作為に選んだときの `bias` の目安。
    """
    units = np.asarray(vectors, dtype=np.float64)
    units = units / np.maximum(np.linalg.norm(units, axis=1, keepdims=True), 1e-12)
    return (float(np.linalg.norm(units.mean(axis=0))),
            100.0 * (units > 0).mean(axis=0),
            0.921 / np.sqrt(len(units)))


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    numbers = list(range(args.first, args.last + 1))

    per_task: dict[str, dict[int, float]] = {}
    for suite, task in TASKS:
        print(f"=== {task} で {len(numbers)} 番号を測る（並列 {args.workers}）===",
              flush=True)
        per_task[task] = _run_task(suite, task, numbers, args.layout, args.workers)

    names = list(per_task)
    first, second = per_task[names[0]], per_task[names[1]]
    both = [n for n in numbers if first.get(n) and second.get(n)]

    def scenes(number: int, key: str) -> list[np.ndarray]:
        """その番号の値を、2 つの場面ぶん並べて返す。"""
        return [np.asarray(scene[number][key], dtype=np.float64)
                for scene in (first, second)]

    gap, agree, turn_gap = [], [], []
    for n in both:
        one, other = scenes(n, "vector_cm")
        lengths = np.linalg.norm([one, other], axis=1)
        gap.append(abs(lengths[0] - lengths[1]))
        agree.append(float(one @ other / max(lengths.prod(), 1e-12)))
        turn_gap.append(float(np.linalg.norm(np.subtract(*scenes(n, "rotation_vector_deg")))))
    print(f"\n2 つの場面での差（{len(both)} 番号）: 大きさ 中央 {np.median(gap):.3f}・最大 {np.max(gap):.3f} cm"
          f" ／ 向きの一致 cos 中央 {np.median(agree):.3f}・最小 {np.min(agree):.3f}"
          f" ／ 手首の回転の差 中央 {np.median(turn_gap):.2f}・最大 {np.max(turn_gap):.2f} 度")
    print("**差が小さければ、ずれは場面に依らず関節角だけで決まる。**")

    # 採用する値は 2 場面の平均にする。**大きさは今までどおり「大きさの平均」**
    # （向きの平均の長さではない）——9/11 に作った一覧と同じ物差しに揃えるため。
    offsets = {n: round(float(np.linalg.norm(scenes(n, "vector_cm"), axis=1).mean()), 3)
               for n in both}
    vectors = {n: [round(float(v), 3) for v in np.mean(scenes(n, "vector_cm"), axis=0)]
               for n in both}
    turns = {n: [round(float(v), 3)
                 for v in np.mean(scenes(n, "rotation_vector_deg"), axis=0)]
             for n in both}
    # `turns` の長さ。**JSON に両方載せる**——選び直すときに向きも角度も引きたい。
    angles = {n: round(float(np.linalg.norm(turns[n])), 3) for n in both}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"offsets_cm": offsets, "vectors_cm": vectors, "rotation_deg": angles,
         "rotation_vectors_deg": turns, "tasks": names, "layout": args.layout},
        indent=2, ensure_ascii=False))

    values = np.asarray([offsets[n] for n in both])
    print(f"\n{'帯':>12} {'番号の数':>9} {'割合':>7}")
    bands = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 100)]
    for low, high in bands:
        count = int(((values >= low) & (values < high)).sum())
        print(f"{f'{low}〜{high} cm':>12} {count:>9} {100 * count / len(values):>6.1f}%")
    usable = sorted(n for n in both if offsets[n] >= 2.0)
    print(f"\n**2 cm 以上の番号は {len(usable)} 個**（{100 * len(usable) / len(both):.1f}%）")
    if usable:
        print(f"  中央 {np.median([offsets[n] for n in usable]):.2f} cm・"
              f"最大 {max(offsets[n] for n in usable):.2f} cm")
        # **帯ごとに散らして選べるように、代表を印字する。**
        for low, high in ((2, 3), (3, 4), (4, 6), (6, 100)):
            band = [n for n in usable if low <= offsets[n] < high]
            print(f"  {low}〜{high} cm: {len(band)} 個  例 {band[:8]}")
    def bias_report(label: str, chosen: list[int]) -> None:
        """向きの偏り（`direction_bias`）と、手首の回転の中央を 1 行で出す。"""
        bias, positive, expected = direction_bias([vectors[n] for n in chosen])
        print(f"  {label}: {len(chosen)} 個  偏り {bias:.3f}"
              f"（無作為なら約 {expected:.3f}）"
              f"  正の割合 左右x {positive[0]:.0f}%"
              f"・前後y {positive[1]:.0f}%・上下z {positive[2]:.0f}%"
              f"  手首の回転 中央 {np.median([angles[n] for n in chosen]):.1f} 度")

    print("\n### ずれの向き")
    if usable:
        bias_report("2 cm 以上の全番号", usable)
    if args.selection is not None:
        chosen = [n for n in json.loads(args.selection.read_text())["initstates"] if n in vectors]
        bias_report(f"選んだ組（{args.selection.name}）", chosen)
    print(f"\n=== {args.out} ===")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
