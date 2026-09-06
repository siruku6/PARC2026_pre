#!/usr/bin/env python3
"""016: 評価ハーネスと同じ形で LIBERO env を開くための共通部品。

**このプロブで一番load-bearing な取り決めを 1 箇所に置くためのモジュールである。**

逆向きの初期状態は「評価が実際に使う場面」の上で作らないと意味が無い。ところが
据え置き(fixture)——caddy・コンロ・棚——の位置は `qpos` に入っておらず
`model.body_pos` に書かれるため、**flattened state を復元しても復らない**。
fixture は `reset()` のたびに乱数で置き直され、その乱数は `env.seed()` に従う。

つまり生成・検証・描画のすべてが、キットの評価ハーネス
（`pipeline/environment.py` の `create_env`）と

  - 同じ `seed`
  - 同じ `reset()` の回数（エピソード k は k 回目の reset の配置）

を揃えないと、A が着地先から外れる。ここが生成・検証・描画に散らばっていると、
片方だけ直したときに**黙って無効なデータが出る**ので 1 箇所にまとめる。
（カメラの解像度や描画の有無は乱数を引かないので、用途ごとに変えてよい——
`make_eval_env` の docstring 参照。）
"""
from __future__ import annotations

import os

import numpy as np

# キットの `EvalConfig` の既定値に合わせる（`seed: int = 42`、カメラは 128）。
EVAL_SEED = int(os.environ.get("PARC_EVAL_SEED", "42"))
EVAL_CAMERA = int(os.environ.get("LIBERO_EVAL_CAMERA", "128"))

# 評価ハーネスが初期状態を復元した直後に空撃ちするステップ数。obs を得るためと、
# 復元直後の接触を落ち着かせるため。生成・検証・描画も同じ数だけ撃つ。
SETTLE_STEPS = 10


def make_eval_env(bddl_path, *, use_camera_obs=True, long_horizon=False,
                  camera_size=EVAL_CAMERA, camera_names=None):
    """キットの `create_env` と同じ `seed` で env を開く。

    **揃えないといけないのは `seed` と（呼び出し側が進める）`reset()` の回数**で、
    これが fixture の配置を決める。以下の引数はどれも**乱数を引かない**ので、
    用途に合わせて変えてよい:

    - `use_camera_obs=False`: 置き方候補の掃き出しのように**画を 1 枚も使わない**
      用途で描画を省く。描画は物理の後に走るだけ。
    - `long_horizon=True`: 1 エピソードで候補を何十通りも試して数千ステップ回る
      用途向け。既定の horizon だと途中で終了扱いになり
      `executing action in terminated episode` で落ちる。終了条件だけを変える。
    - `camera_size` / `camera_names`: 目視用に大きく撮る・別視点で撮るとき。
      既定は評価と同じ（`EVAL_CAMERA`・env の既定カメラ一式）。
    """
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    kwargs = {
        "bddl_file_name": str(bddl_path),
        "camera_heights": camera_size,
        "camera_widths": camera_size,
        "control_freq": int(os.environ.get("LIBERO_CONTROL_FREQ", "20")),
        "use_camera_obs": use_camera_obs,
    }
    if camera_names is not None:
        kwargs["camera_names"] = camera_names
    if long_horizon:
        kwargs["ignore_done"] = True
        kwargs["horizon"] = 1_000_000
    env = OffScreenRenderEnv(**kwargs)
    env.seed(EVAL_SEED)
    return env


def settle_zero(env, n=SETTLE_STEPS):
    """ゼロ行動で n ステップ進め、最後の obs を返す。"""
    obs = None
    for _ in range(n):
        obs, _, _, _ = env.env.step(np.zeros(7))
    return obs


def load_flat_state(env, state):
    """flattened sim state を復元し、評価と同じ空撃ちをしてから obs を返す。

    `reset()` は呼ばない——**呼び出し側が評価と同じ回数だけ進める責任を持つ**
    （reset の回数が fixture の配置を決めるため）。
    """
    sim = env.env.sim
    sim.set_state_from_flattened(np.asarray(state, dtype=np.float32))
    sim.forward()
    return settle_zero(env)


def region_site_xpos(env, region):
    """region 参照名の site の位置を返す。

    LIBERO は region の site 名を**参照名そのまま**（`desk_caddy_1_back_contain_region`）
    で登録しており、`env.object_sites_dict` がその名前で引ける。名前を推測して
    suffix 一致させる必要は無い。
    """
    inner = env.env
    if region not in inner.object_sites_dict:
        raise SystemExit(
            f"region '{region}' の site が無い。"
            f" ある site: {sorted(inner.object_sites_dict)}")
    return np.asarray(inner.sim.data.get_site_xpos(region)).copy()


def eval_predicate(env, pred, *args):
    """LIBERO 本体の goal 判定機構でそのまま判定する。

    採点器と同じ `_eval_predicate` を呼ぶ。距離のしきい値による近似を使うと、
    「縁に乗っただけ」を「着地先に在る」と通してしまう。

    `pred` は bddl パーサが小文字化した名前（`in` / `on`）で渡すこと。
    """
    return bool(env.env._eval_predicate([pred, *args]))
