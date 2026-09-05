#!/usr/bin/env python3
"""Track3 試作(006 やること1)・パス①: 時間反転プロトタイプ。

黄色い本タスク（配布 T3 「棚から黄色い本を下ろす」）について、
順方向デモ（tar `libero_combined_20hz` の task 92 =
`STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf`）を
このシミュレータで順再生し、最終状態を捕まえた上で、
逆方向（T3 の初期状態から反転 action を再生）を T3 の :goal 判定に通す。

実行先は parc2026-final 画像内（/workspace に LIBERO-plus が入っている）。
CPU 完結（描画不要）。GPU は不要。

使い方（スクリプト経由の docker run、例）:

  docker run --rm \
    -v /home/ubuntu/src/PARC2026_pre:/host:ro \
    -v /home/ubuntu/src/train_data/libero_combined_20hz_extracted:/data:ro \
    -v /tmp/t3c1_out:/out \
    --entrypoint python3 \
    parc2026-final:latest \
    /host/solution/tools/track3_reversal/reversal_probe.py \
      --task-92-episodes 0,1,2 \
      --forward-dddl /workspace/LIBERO-plus/libero/libero/bddl_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl \
      --forward-inits /workspace/LIBERO-plus/libero/libero/init_files/libero_90/STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.pruned_init \
      --t3-dddl /workspace/compe/t3/assets/bddl_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.bddl \
      --t3-inits /workspace/compe/t3/assets/init_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.pruned_init

タスク 92 の episode_index は meta/episodes/*.parquet の `task_index==92` で
絞り込む。`--task-92-episodes` で未指定なら、成功フラグ付きのエピソードを
先頭から 3 本選ぶ。
"""

import argparse
import io
import json
import os
import pickle
import sys
import zipfile

import numpy as np

import pyarrow.parquet as pq


# --------------------------------------------------------------------------
# データ読み込み（ホスト venv と同様の pyenv python に pyarrow があるので、
# 画像内でも同様に使っている）
# --------------------------------------------------------------------------

def load_episode_actions(dataset_root: str, episode_index: int, task_index: int):
    """tar 展開先の LeRobot v3.0 データセットから、1 エピソードの action 列を読む。

    v3.0 の data ファイルは parquet で、episode_index で区切られている。
    どの chunk/file に載っているかは meta/episodes/*.parquet の frame 番号から
    ではなく、全 data parquet を走査して episode_index を含むものを探す。
    試作（数本だけ読む）なので単純な全走査で十分。
    """
    data_dir = os.path.join(dataset_root, "data")
    # episodes meta からこの episode の長さを知る
    ep_meta = {}
    for meta_file in sorted(
        f
        for f in os.listdir(os.path.join(dataset_root, "meta", "episodes", "chunk-000"))
        if f.endswith(".parquet")
    ):
        t = pq.read_table(
            os.path.join(dataset_root, "meta", "episodes", "chunk-000", meta_file)
        )
        for r in t.to_pylist():
            ep_meta[r["episode_index"]] = r

    if episode_index not in ep_meta:
        raise SystemExit(f"episode {episode_index} not found in meta")
    row = ep_meta[episode_index]
    if row.get("task_index") != task_index:
        raise SystemExit(
            f"episode {episode_index} is task {row.get('task_index')}, "
            f"expected {task_index}"
        )
    length = int(row["length"])

    # data parquet を全部開いて episode を探す
    hits = []
    for chunk in sorted(os.listdir(data_dir)):
        cdir = os.path.join(data_dir, chunk)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".parquet"):
                continue
            path = os.path.join(cdir, fn)
            table = pq.read_table(path)
            for batch in table.to_batches():
                b = batch.to_pydict()
                eps = b.get("episode_index")
                if eps is None:
                    continue
                if episode_index in eps:
                    # 該当フレームのみをフレーム順に並べる
                    frame_idx = b["frame_index"]
                    keep = [
                        i for i, e in enumerate(eps) if e == episode_index
                    ]
                    keep.sort(key=lambda i: frame_idx[i])
                    actions = np.array(
                        [b["action"][i] for i in keep], dtype=np.float32
                    )
                    states = np.array(
                        [b["observation.state"][i] for i in keep], dtype=np.float32
                    )
                    hits.append((path, actions, states, length))
    if not hits:
        raise SystemExit(f"episode {episode_index} not found in data parquet")
    path, actions, states, length = hits[0]
    # 長さの突合（データファイル跨ぎで複数ヒットしたら連結する）
    if len(hits) > 1:
        actions = np.concatenate([a for _, a, _, _ in hits])
        states = np.concatenate([s for _, _, s, _ in hits])
    assert len(actions) == length, f"action length {len(actions)} != meta {length}"
    return {
        "source_file": path,
        "actions": actions,  # (T, 7)
        "states": states,    # (T, 8) robot state only — full sim state は無い
        "length": length,
    }


# --------------------------------------------------------------------------
# 初期状態の読み込み（.pruned_init = zip の中の data.pkl = 平坦化シム状態のリスト）
# --------------------------------------------------------------------------

def load_pruned_init(path: str):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            pick = [n for n in names if n.endswith("data.pkl")]
            if not pick:
                raise SystemExit(f"no data.pkl inside {path}: {names}")
            blob = z.read(pick[0])
    else:
        blob = open(path, "rb").read()
    states = pickle.loads(blob)
    states = [np.asarray(s, dtype=np.float32) for s in states]
    return states


# --------------------------------------------------------------------------
# 反転 action の生成
# --------------------------------------------------------------------------

def build_reversed_actions(
    actions: np.ndarray, gripper_mode: str = "keep", skip_gripper: bool = False
) -> np.ndarray:
    """順方向 action 列 (T, 7) を時間反転する。

    データ規約: use_relative_actions=False（絶対 EEF 姿勢 6 次元 + gripper 1 次元）。
    絶対姿勢の反転は意味が通る（同じ空間経路を戻る）。
    gripper は開閉の絶対値なので、そのまま反転すると「掴む」が「放す」に
    ならない。gripper_mode で扱いを選ぶ:

      keep   — action_6 をそのまま反転列に乗せる（デフォルト・最初に試す）
      invert — action_6 を 1 - x で反転する（掴み/放しの切替を補正する）
      skip_gripper=True — action_6 を 0（掴んだまま）にする
    """
    rev = np.ascontiguousarray(actions[::-1].copy())
    if skip_gripper:
        rev[:, 6] = 0.0
    elif gripper_mode == "invert":
        rev[:, 6] = 1.0 - rev[:, 6]
    return rev


# --------------------------------------------------------------------------
# 環境の生成と再生
# --------------------------------------------------------------------------

def make_env(bddl_file: str, control_freq: int = 20):
    """評価と同じ ControlEnv を作る（scripted_demo._make_env と同じ経路）。"""
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=bddl_file,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        has_renderer=False,
        control_freq=control_freq,
        horizon=2000,
        ignore_done=True,
    )


def replay(env, init_state, actions, max_steps=None):
    """初期状態から action 列を再生する。最終シム状態（平坦化）と step 数を返す。

    scripted_demo.reset_to と同じ手順:
    set_state_from_flattened → forward → qvel ゼロの空撃ち 10 ステップ。
    step は ControlEnv の内部 env に対して行う（ndarray action を受理する）。
    """
    inner = env.env
    sim = inner.sim
    sim.set_state_from_flattened(init_state)
    sim.forward()
    for _ in range(10):
        sim.data.qvel[:] = 0
        inner.step(np.zeros(7))
    n = len(actions) if max_steps is None else min(len(actions), max_steps)
    for i in range(n):
        inner.step(np.asarray(actions[i], dtype=np.float32))
    return sim.get_state().flatten(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        default="",
        help="tar 展開先（LeRobot v3.0）。--actions-npz を渡す場合、"
        "画像内では data/ を読まないので空でよい（/data をマウントしなくてもよい）。",
    )
    ap.add_argument("--task-index", type=int, default=92)
    ap.add_argument(
        "--task-92-episodes",
        default="",
        help="コンマ区切り。未指定なら task_index の成功エピソードを先頭3本",
    )
    ap.add_argument("--forward-dddl", required=True)
    ap.add_argument("--forward-inits", required=True)
    ap.add_argument("--t3-dddl", required=True)
    ap.add_argument("--t3-inits", required=True)
    ap.add_argument("--out", default="/out")
    ap.add_argument(
        "--actions-npz",
        default="",
        help="ホスト側で action 列を抽出した npz (host_extract_actions.py が作る)。"
        "指定すると画像内で pyarrow が不要になる。"
        "キー: actions_<ep> (T,7) と length_<ep>。",
    )
    ap.add_argument(
        "--gripper-mode",
        choices=["keep", "invert"],
        default="keep",
        help="反転時の gripper 軸(action_6)の扱い。keep=そのまま、invert=1-x で反転。"
        "データは use_relative_actions=False（絶対姿勢）で、gripper は開閉の絶対値なので"
        "単純反転では意味が狂う可能性がある。両方試すのが安全。",
    )
    ap.add_argument(
        "--skip-gripper",
        action="store_true",
        help="反転再生時に action_6 を 0（掴んだまま）にする。"
        "『放す瞬間の位置をそのまま辿る』解釈。",
    )
    args = ap.parse_args()

    # 1) 候補エピソードの決定
    if args.task_92_episodes.strip():
        ep_ids = [int(x) for x in args.task_92_episodes.split(",") if x.strip()]
    elif args.actions_npz:
        z = np.load(args.actions_npz)
        ep_ids = sorted(
            int(k.split("_")[1]) for k in z.files if k.startswith("actions_")
        )
    else:
        eps = []
        meta_dir = os.path.join(args.dataset_root, "meta", "episodes", "chunk-000")
        for fn in sorted(os.listdir(meta_dir)):
            if not fn.endswith(".parquet"):
                continue
            for r in pq.read_table(os.path.join(meta_dir, fn)).to_pylist():
                if r.get("task_index") == args.task_index and r.get("is_success"):
                    eps.append(int(r["episode_index"]))
                if len(eps) >= 3:
                    break
            if len(eps) >= 3:
                break
        ep_ids = eps
    print(f"candidate episodes: {ep_ids}", flush=True)

    # 2) 環境（順方向用 / T3 用）を各 1 回だけ作って使い回す
    print("creating forward env ...", flush=True)
    env_fwd = make_env(args.forward_dddl)
    print("forward env ready", flush=True)
    print("creating t3 env ...", flush=True)
    env_t3 = make_env(args.t3_dddl)
    print("t3 env ready", flush=True)

    fwd_inits = load_pruned_init(args.forward_inits)
    t3_inits = load_pruned_init(args.t3_inits)
    print(f"forward inits: {len(fwd_inits)}, t3 inits: {len(t3_inits)}", flush=True)

    results = []
    for ep in ep_ids:
        rec = {"episode_index": ep}
        print(f"=== episode {ep} ===", flush=True)

        if args.actions_npz:
            z = np.load(args.actions_npz)
            actions = z[f"actions_{ep}"].astype(np.float32)
            data = {"actions": actions, "length": int(z[f"length_{ep}"])}
        else:
            data = load_episode_actions(args.dataset_root, ep, args.task_index)
        actions = data["actions"]
        rec["n_actions"] = len(actions)
        print(f"  actions: {actions.shape}", flush=True)

        # 2a) 順再生: task92 の初期状態から → 棚上面まで行くはず
        fwd_state, n_steps = replay(env_fwd, fwd_inits[0], actions)
        env_fwd_env_ok = bool(env_fwd.check_success())
        rec["forward"] = {
            "n_steps": n_steps,
            "success_at_end": env_fwd_env_ok,
        }
        # 最終シム状態をバイトで残す（後で .pruned_init に書き出せる）
        with open(os.path.join(args.out, f"fwd_final_ep{ep}.npz"), "wb") as f:
            np.savez(f, state=fwd_state)
        print(
            f"  forward replay: {n_steps} steps, success={env_fwd_env_ok}",
            flush=True,
        )
        if not env_fwd_env_ok:
            print("  !! forward demo itself does not reach its goal in our sim", flush=True)

        # 2b) 逆再生: T3 の初期状態（＝順方向デモのゴールに近い配置）から、
        #     action 列を反転して再生 → T3 の :goal（机上 left region）を満たすか
        rev_actions = build_reversed_actions(
            actions, gripper_mode=args.gripper_mode, skip_gripper=args.skip_gripper
        )
        t3_state, n_steps_r = replay(env_t3, t3_inits[0], rev_actions)
        t3_ok = bool(env_t3.check_success())
        rec["reverse"] = {
            "gripper_mode": args.gripper_mode,
            "skip_gripper": args.skip_gripper,
            "n_steps": n_steps_r,
            "success_at_end": t3_ok,
        }
        with open(os.path.join(args.out, f"rev_final_ep{ep}.npz"), "wb") as f:
            np.savez(f, state=t3_state)
        print(
            f"  reverse replay: {n_steps_r} steps, success={t3_ok} "
            f"(gripper={args.gripper_mode}, skip_gripper={args.skip_gripper})",
            flush=True,
        )
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)

    out_path = os.path.join(args.out, "reversal_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"wrote {out_path}", flush=True)

    # go/no-go サマリー
    ok = [r for r in results if r["reverse"]["success_at_end"]]
    print(
        f"\n=== SUMMARY: {len(ok)}/{len(results)} reversed replays satisfy T3 :goal ===",
        flush=True,
    )


if __name__ == "__main__":
    main()