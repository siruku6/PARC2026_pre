#!/usr/bin/env python3
"""手本より速い軌道を、シミュレータの中で台本から作る。

**学習データの手本より速く動くことはできない。** 方策は手本を真似るので、
手本が 400 ステップかけている動きを 300 ステップで終えるようには育たない。実際
`put both moka pots on the stove` は 146 本すべてが 341 ステップ以上で、
**300 ステップ以下の手本が 1 本も無い**（`libero_plus_episodes.parquet` の `length` 列）。
このタスクは、いまの学習データからは原理的に解けない。

そこで**手本を自分で作る。** 運営 Q&A No.33 が
「配布 kit をご自身の環境で実行し、ご自身で選んだ LIBERO-plus タスクの
ロールアウト（観測画像・状態・行動）を保存して学習データに使うことは可能です」
と回答している。

やり方は「中継点を順に追う」だけである。**物体の位置はシミュレータが正確に
持っているので、画像から推測する必要がない。** 掴む・運ぶ・置くの中継点を
物体の位置から決め、腕をそこへ寄せる行動を1 ステップごとに計算する。

    # 1. 場面の寸法を測る（新しいタスクの台本を書く前に必ずやる）
    python3 solution/tools/scripted_demo.py measure \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove

    # 2. 初期配置すべてで通るか数える（描画しないので 1 本 2.4 秒）
    python3 solution/tools/scripted_demo.py verify \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove

    # 3. 学習データとして書き出す（描画するので 1 本 100 秒。**並列で回す**）
    MUJOCO_GL=osmesa python3 solution/tools/scripted_demo.py record \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \\
        --variant KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_2 \\
        --episodes 0,1,2 --output /out/scripted/

    # 4. **ずれてから戻す手本にする**（本選 Track2 向け。DART のやり方）
    #    送る行動にだけノイズを混ぜ、正解としては混ぜる前を記録する。
    #    **σ は大きすぎると成功率が落ちて逆効果になる**ので、
    #    描画しない verify で振ってから決める（1 本 2.4 秒）。
    for sigma in 0 0.02 0.05 0.10; do
      python3 solution/tools/scripted_demo.py verify \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \\
        --action-noise $sigma
    done

**描画は CPU（`osmesa`）で行う。** 手元の GPU マシンの vGPU は
`NVIDIA Virtual Compute Server` で OpenGL に対応せず、`egl` を指定すると
**空の場面は 0.4 ms で描けるのに、実際の場面で止まる。**
速くする手は並列だけである（`--episodes` を分けて同時に流す。手順は
`docs/howto/05_scripted_demos.md` の 3 節）。

**台本を書くときに必ず先に測ること**（`measure` が出す）。

| 測るもの | なぜ | moka pots で実際に踏んだ穴 |
|---|---|---|
| 指の開き 対 物体の幅 | 開かなければ掴めない | 本体は 5cm 角、指は 4.8cm しか開かない。**掴めるのは蓋のつまみだけ** |
| 当たり判定のある什器の上端 | 運搬中にぶつける | コンロ上面 0.9305 に対し運搬中の鍋の底が 0.922。8mm 潜っていた |
| 1 ステップで進む距離 | ステップ数の見積もり | OSC の追従の限界で **1 ステップ 1.25cm**。行動の上限 5cm には届かない |

Reads:
  - <libero-root>/bddl_files/<suite>/<変種>.bddl
  - <libero-root>/init_files/<suite>/<元タスク>.pruned_init
Generates:
  - 標準出力
  - `record` のときのみ <output>/<変種>_ep<n>.npz（観測・行動・成否）
"""

from __future__ import annotations

import argparse
import functools
import os
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

# 行動の値。**robosuite の Panda はこの向き**である。
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# 目標まであとどれだけの所から減速を始めるか [m]。
# **1 ステップぶん（腕は 1 ステップ 1.25cm 進む）に取ると、そこまで全速で動ける。**
# 差にそのまま比例させると遠いうちから減速し、同じ移動に 171 ステップかかった
# （振り切れる形なら 19 ステップ）。
BRAKE_WITHIN_STOP = 0.012     # 止まる点。短く詰める
BRAKE_WITHIN_PASS = 0.045     # 通過してから止まる点。長くかけて速度を残さない

# 観測の作り。**学習データ（`lerobot/libero_plus`）の宣言に合わせる。**
CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SIZE = 256

# 方策へ渡す前の空撃ちのステップ数。**評価パイプラインと同じにする**
# （`pipeline/rollout.py`）。違うと、腕の出発姿勢が採点時と変わる。
WARMUP_STEPS = 10

_DEFAULT_LIBERO_ROOT = Path(
    os.environ.get("LIBERO_ROOT", "/workspace/LIBERO-plus")
) / "libero" / "libero"


@dataclass
class PickPlace:
    """物体を 1 つ掴んでどこかへ置く、という 1 まとまり。

    Attributes
    ----------
    target : str
        掴む物体の観測名（`moka_pot_1` なら観測の `moka_pot_1_pos` を見る）。
    place_site : str
        置く先の基準にする site。**ゴール判定に使われている枠そのものを指す。**
        什器は初期配置ごとに数 mm ずれるので、**絶対座標で書くと変種で外れる。**
    place_offset : tuple[float, float]
        `place_site` の中心から見た、置く先の水平方向のずれ。
    grasp_dz : float
        物体の原点から見て、どの高さを掴むか。**指が閉じられる細い所を狙う。**
    approach_dz : float
        物体の手前で一度上を通る高さ。**通過点であって、止まらない。**
    hover_dz : float
        掴む直前に降りてくる高さ。**ここから下は真下へまっすぐ降りる**ので、
        物体の当たり判定の上端よりわずかに高く取る。
    """

    target: str
    place_site: str
    place_offset: tuple[float, float]
    grasp_dz: float
    approach_dz: float = 0.16
    hover_dz: float = 0.11


@dataclass
class TurnKnob:
    """つまみを掴んで回す。**「点ける」は角度で判定される。**

    SCENE3 のコンロなら `flat_stove_1_button` が **0.5 rad 以上**で点火
    （`libero/libero/envs/objects/articulated_objects.py` の
    `default_turnon_ranges` の下限。どの関節でも 1 つ超えれば真になる）。
    **ばねが無いので、一度回せば戻らない**（`jnt_stiffness` が 0）。

    **押すだけでは足りない。** 実測で、まっすぐ押すと 0.12、円弧に沿って
    押しても 0.05 で止まる（接線から外れて滑る）。**掴んで、円弧に沿って
    動かしながら手首も回すと 0.83 まで回った。** 3 つ揃って初めて回る。

    Attributes
    ----------
    joint : str
        回る関節の名前。**回転の中心はこの関節から実行時に読む**（什器は
        初期配置ごとに数 mm ずれるので、絶対座標では書けない）。
    radius : float
        中心から、掴む位置までの距離。**羽の半径 4cm より内側を掴む。**
    height : float
        掴む高さ（絶対 z）。**実測でここが最も効く** —— 0.910 で 0.83、
        0.925 では 0.21 しか回らなかった（羽の上端をつまんで滑る）。
    sweep : float
        手先を動かす角（rad）。関節が回る角そのものではない（滑るぶん多く要る）。
    turn_steps : int
        回すのに使うステップ数。**1 ステップごとに中継点を 1 つ進める。**
    yaw : float
        手首を回す指令。**1.0 が上限**（`output_max` は 0.5 rad）。
    hover_dz : float
        掴む前に構える高さ。
    """

    joint: str
    radius: float
    height: float
    sweep: float = 1.2
    turn_steps: int = 50
    yaw: float = 1.0
    hover_dz: float = 0.06


@dataclass
class TaskPlan:
    """1 タスクぶんの台本。

    Attributes
    ----------
    suite : str
        `libero_10` など、bddl の置き場所。
    steps : list[PickPlace | TurnKnob]
        前から順に実行する。**動きの種類ごとに実行の中身が違う**
        （`_RUNNERS` で引く）。
    carry_z : float
        運ぶときの手先の高さ。**什器の上端より、物体の背丈のぶん高く取る。**
    place_z : float
        離すときの手先の高さ。**置く面のすぐ上。**
    retreat_z : float
        離したあとに逃げる高さ。
    close_steps : int
        指を閉じきるのに使うステップ数。**何も掴まない状態で 12 ステップかかる。**
    open_steps : int
        指を開くのに使うステップ数。
    settle_steps : int
        置いてから指を開くまで、動かずに待つステップ数。**角を丸めて運ぶと
        到着時に横向きの速度が残る。** そのまま離すと物体を引きずって
        置き場所から外れる（実測で狙いから 2.7cm ずれた）。ここで止める。
    final_settle_steps : int
        全部置き終わったあと、ゴール判定をする前に待つステップ数。**置いた直後は
        まだ物体が揺れていて、接触の判定が安定しない。**
    budget_move : int
        水平に動く 1 段に許すステップ数の上限。
    budget_descend : int
        下ろす 1 段に許すステップ数の上限。
    budget_lift : int
        上げる 1 段に許すステップ数の上限。
    blend_radius : float
        通過点をどれだけ手前で見切るか。**大きいほど滑らかだが、角が内側へ
        膨らむ。** 什器の際を通る経路では小さくする。
    smoothing : float
        指令を前回からなます強さ。**既定の 1.0 はなまさない。**
        向きを保つ指令と角の丸めだけでジャークは 18.3 → 12.3 まで下がり、
        **なましを足すと滑らかさはほとんど変わらないままステップ数だけ増えた**
        （0.6 で 234 ステップ・0.35 で 264 ステップ）。細かく動かしたいときだけ下げる。
    """

    suite: str
    steps: list[PickPlace | TurnKnob]
    carry_z: float
    place_z: float
    retreat_z: float
    close_steps: int = 14
    open_steps: int = 8
    settle_steps: int = 6
    final_settle_steps: int = 8
    budget_move: int = 45
    budget_descend: int = 25
    budget_lift: int = 22
    blend_radius: float = 0.050
    smoothing: float = 1.0


# **台本はここに足していく。** 鍵は元タスクの名前（bddl から摂動の接尾辞を
# 落としたもの）。摂動が見た目だけなら（照明・背景）、同じ台本がそのまま通る。
TASK_PLANS: dict[str, TaskPlan] = {
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": TaskPlan(
        suite="libero_10",
        steps=[
            # コンロの枠は半幅 0.075。**2 つを 7cm 離して置く**（鍋の半径 2.5cm）。
            # 掴めるのは蓋のつまみ（原点から 7.2cm 上）だけ。**本体は 5cm 角で、
            # 指は 4.8cm しか開かない。**
            PickPlace("moka_pot_1", "flat_stove_1_cook_region", (-0.035, 0.0),
                      grasp_dz=0.072),
            PickPlace("moka_pot_2", "flat_stove_1_cook_region", (0.035, 0.0),
                      grasp_dz=0.072),
        ],
        # コンロの当たる上端 0.9652、鍋の底は手先から 0.1398 下。
        carry_z=1.14,     # 鍋の底が 1.000 まで上がる
        place_z=1.072,    # 鍋の底が 0.932。コンロ上面 0.9305 のすぐ上
        retreat_z=1.16,
    ),
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": TaskPlan(
        suite="libero_10",
        steps=[
            # 鍋は 1 つ。**枠の中心へ置く**（半幅 0.075）。掴めるのは
            # SCENE8 と同じく蓋のつまみ（原点から 7.2cm 上）だけ。
            PickPlace("moka_pot_1", "flat_stove_1_cook_region", (0.0, 0.0),
                      grasp_dz=0.072),
            # **点けるのは最後にする。** 手首を 50 ステップ回すと姿勢が戻らず、
            # **そのあとの動きが届かなくなる**（実測で逃げる誤差 5〜24cm、
            # 続く「近づく」が 9〜34cm 外れ、3 本のうち 2 本がステップ数切れ）。
            # ゴールは「点いている」と「置いてある」の連言で1 ステップごとに見られるので、
            # **置いてから点ければ、点いた瞬間に成功が立つ。**
            # ばねが無いので回した角は戻らない（`jnt_stiffness` が 0）。
            TurnKnob("flat_stove_1_button", radius=0.036, height=0.910),
        ],
        # コンロの当たる上端は SCENE8 と同じ 0.9652。**フライパン（上端 1.0097）は
        # y = -0.27 にあり、鍋（y ≈ 0）からコンロ（y ≈ 0.20）へ運ぶ経路の外**なので、
        # 運ぶ高さを上げる必要は無い。
        carry_z=1.14,
        place_z=1.072,
        retreat_z=1.16,
    ),
}


@dataclass
class EpisodeRecord:
    """1 本ぶんの結果。

    画像はここに持たない。**ステップ数ぶんの置き場を先に確保して書き込む**ほうが
    複製もピークも半分で済むので、`_Rollout` が持っている（`images()`）。
    """

    steps: int = 0
    goal_reached: bool = False
    max_displacement: float = 0.0
    collision_threshold_m: float = 0.001
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    # **実際にシミュレータへ送った行動。** `--action-noise` を使ったときだけ
    # `actions` と違う（学習データに入れるのは `actions` のほうである）。
    # 送ったほうも残すのは、**ノイズが本当に効いていたかを後から確かめる**ため。
    executed_actions: list[np.ndarray] = field(default_factory=list)
    # 台本に混ぜたノイズの大きさ（標準偏差）。0.0 なら混ぜていない。
    action_noise: float = 0.0
    ee_positions: list[np.ndarray] = field(default_factory=list)
    # 画像はステップ数ぶんに切った「眺め」で、複製ではない（`attach_images`）。
    front: np.ndarray | None = None
    wrist: np.ndarray | None = None

    @property
    def collided(self) -> bool:
        """対象外の物体を動かしたか。**しきい値は採点設定から来る。**"""
        return self.max_displacement > self.collision_threshold_m

    @property
    def success(self) -> bool:
        """採点と同じ成功の定義。**ゴール到達 かつ 衝突なし。**"""
        return self.goal_reached and not self.collided


@functools.lru_cache(maxsize=1)
def scoring_rules() -> tuple[float, float, bool]:
    """採点側の判定基準を、採点設定から読む。**1 回だけ読む。**

    **数値を書き写さない。** `total_score_config.json` や環境変数
    `TOTAL_SCORE_CONFIG` で採点側が上書きできる値なので、書き写すと
    「手元の成功数だけが古い基準のまま」という静かなずれになる。

    Returns
    -------
    collision_threshold_m : float
        対象外の物体がこれを超えて動いたら失格。
    control_dt : float
        1 ステップの時間。滑らかさの計算に使う。
    collision_enabled : bool
        衝突で失格させるか。
    """
    from pipeline.total_score import load_scoring_config

    config = load_scoring_config()
    collision = config.get("collision", {})
    return (float(collision.get("threshold_m", 0.001)),
            float(config.get("control_dt", 0.05)),
            bool(collision.get("enabled", True)))


def smoothness(ee_positions: list[np.ndarray],
               control_dt: float | None = None) -> dict[str, float]:
    """手先の軌跡の滑らかさ。**採点と同じ式を使う。**

    式を書き写すと必ず片方が古くなるので、配布キットの
    `pipeline.scorer` と `pipeline.total_score` からそのまま呼ぶ。
    いずれも採点の総合スコアに入っている指標である。

    Returns
    -------
    metrics : dict[str, float]
        `avg_jerk` / `rms_jerk` … ジャーク（小さいほど滑らか）
        `sparc` … 0 に近いほど滑らか
        `path_length` … 手先が動いた距離 [m]
    """
    from pipeline.scorer import Scorer
    from pipeline.total_score import compute_sparc

    if control_dt is None:
        control_dt = scoring_rules()[1]
    track = np.asarray(ee_positions)
    if len(track) < 4:
        return {"avg_jerk": float("nan"), "rms_jerk": float("nan"),
                "sparc": float("nan"), "path_length": 0.0}
    average_jerk, rms_jerk = Scorer._compute_jerk(track, control_dt)
    return {
        "avg_jerk": average_jerk,
        "rms_jerk": rms_jerk,
        "sparc": float(compute_sparc(track, control_dt)),
        "path_length": float(np.sum(np.linalg.norm(np.diff(track, axis=0),
                                                   axis=1))),
    }


class SavedEpisode:
    """書き出した 1 本。**npz の取り決めはこの 1 か所だけに置く。**

    書き手（`command_record`）と同じファイルに置いてあるので、鍵を足しても
    読み手が古くならない。`visualize_demo.py` と `npz_to_lerobot.py` は
    これを import する。**このモジュールは lerobot を import しない**ので、
    シミュレータ側（Python 3.10）からも呼べる。

    **画像は触るまで伸長しない。** 1 本で 118 MB あり、形だけ見たい・
    成否で捨てたい呼び出し側が展開を払う必要は無い（伸長は 1 本 0.34 秒で、
    50 本なら 17 秒と 5.9 GB の一時確保になる）。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._payload = np.load(path, allow_pickle=True)
        self.state: np.ndarray = self._payload["state"]
        self.action: np.ndarray = self._payload["action"]
        self.steps = int(self._payload["steps"])
        self.success = bool(self._payload["success"])
        self.instruction = str(self._payload["instruction"])
        # **2026-08-21 より前に書いた npz にはこの 2 つが無い。**
        # 無い鍵を読むと `KeyError` になるので、既定を置いて読めるままにする
        # （`action` はノイズを混ぜる前の行動で、そこは昔から変わらない）。
        self.action_noise = (float(self._payload["action_noise"])
                             if "action_noise" in self._payload else 0.0)

    @property
    def action_executed(self) -> np.ndarray:
        """実際にシミュレータへ送った行動。**学習に使うのは `action` である。**

        `--action-noise` を付けずに作った本、および 2026-08-21 より前に
        書いた本では `action` と同じものを返す。
        """
        if "action_executed" not in self._payload:
            return self.action
        return self._payload["action_executed"]

    @property
    def front(self) -> np.ndarray:
        """正面カメラ。**ここで初めて伸長する。**"""
        return self._payload["front"]

    @property
    def wrist(self) -> np.ndarray:
        """手首カメラ。**ここで初めて伸長する。**"""
        return self._payload["wrist"]

    def image_spec(self, key: str) -> tuple[tuple[int, ...], np.dtype]:
        """画像の形と型だけを、画素を伸長せずに読む。

        npz は `.npy` を並べた zip なので、**ヘッダだけ読めば形が分かる。**
        """
        with self._payload.zip.open(f"{key}.npy") as stream:
            version = np.lib.format.read_magic(stream)
            # **公開されている読み口を使う。** `_read_array_header` は private で、
            # numpy 2 系で引数が増えている。
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, _, dtype = reader(stream)
        return shape, dtype


class _Rollout:
    """1 本ぶんの実行。手先を中継点へ寄せる行動を1 ステップごとに作る。"""

    def __init__(self, env: Any, init_state: np.ndarray, max_steps: int,
                 record_images: bool, action_noise: float = 0.0,
                 rng: np.random.Generator | None = None) -> None:
        self.env = env
        self.max_steps = max_steps
        self.record_images = record_images
        self.action_noise = float(action_noise)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        threshold, _, collision_enabled = scoring_rules()
        self.record = EpisodeRecord(
            # **採点が衝突を見ない設定なら、こちらも見ない。** 届かない
            # しきい値にしておけば、判定の分岐を増やさずに従える。
            collision_threshold_m=threshold if collision_enabled else np.inf)
        self.phases: list[tuple[str, int, float]] = []
        self._last_command = np.zeros(3)
        self.record.action_noise = self.action_noise

        self.obs = reset_to(env, init_state)

        # 採点と同じ衝突判定。**動かしてよい物体（obj_of_interest）は外す。**
        # 毎ステップぶん回すので、鍵と初期位置を (N, 3) に積んで 1 回の引き算で済ます。
        interest = set(env.obj_of_interest)
        watched = [key for key in self.obs
                   if key.endswith("_pos")
                   and not key.startswith("robot0")
                   and not key.endswith("_to_robot0_eef_pos")
                   and key[:-4] not in interest]
        self._watched_keys = tuple(watched)
        self._watched_origin = (np.stack([self.obs[key] for key in watched])
                                if watched else np.zeros((0, 3)))

        # 画像は先に置き場を確保して書き込む。**list に溜めて最後に np.stack
        # すると、複製が 2 回と、ピークが 2 倍（1 本 236 MB）になる。**
        self._front_buffer: np.ndarray | None = None
        self._wrist_buffer: np.ndarray | None = None
        if record_images:
            shape = (max_steps, *np.asarray(self.obs["agentview_image"]).shape)
            self._front_buffer = np.empty(shape, dtype=np.uint8)
            self._wrist_buffer = np.empty(shape, dtype=np.uint8)

    @property
    def eef_position(self) -> np.ndarray:
        """手先の位置。"""
        return np.asarray(self.obs["robot0_eef_pos"])

    def object_position(self, name: str) -> np.ndarray:
        """物体の位置。"""
        return np.asarray(self.obs[f"{name}_pos"])

    def site_position(self, name: str) -> np.ndarray:
        """site の位置。**ゴール判定の枠は初期配置ごとに数 mm ずれる。**"""
        return np.asarray(self.env.env.sim.data.get_site_xpos(name))

    def _observation_state(self) -> np.ndarray:
        """学習データと同じ 8 次元の状態。位置 3 ＋ 軸角 3 ＋ グリッパ 2。

        軸角の変換は **robosuite の `quat2axisangle` をそのまま呼ぶ。**
        `lerobot/libero_plus` を作った LIBERO-plus 側の
        `scripts/create_dataset.py` が同じ関数で同じ 8 次元を組んでおり、
        同梱 LeRobot の `LiberoProcessorStep` も同じ式（`2·acos(w)`・
        折り返し無し）である。**自分で書くと折り返しの扱いが必ずずれる**
        —— Panda の下向き把持は `w ≈ 0` に居るので符号が絶えず反転し、
        折り返す実装では state の 3〜5 番目が 1 ステップで 2π ぶん飛ぶ。
        """
        from robosuite.utils.transform_utils import quat2axisangle

        return np.concatenate([
            np.asarray(self.obs["robot0_eef_pos"], dtype=np.float32),
            # **複製を渡す。** robosuite の実装は |w| > 1 のとき引数を
            # その場で書き換えるので、観測の配列を直に渡すと壊れる。
            np.asarray(quat2axisangle(np.array(self.obs["robot0_eef_quat"],
                                               dtype=np.float64)),
                       dtype=np.float32),
            np.asarray(self.obs["robot0_gripper_qpos"], dtype=np.float32),
        ])

    def _velocity_command(self, delta: np.ndarray, full_speed_within: float
                          ) -> np.ndarray:
        """目標へ向かう指令を作る。**向きを崩さずに大きさだけ絞る。**

        軸ごとに ±1 で切ると、遠いうちは 3 軸とも振り切れて
        **「立方体の角の向き」に進む。** 近づくにつれて軸ごとに順番に飽和が
        解けるので、**そのたびに進行方向が折れてガタつく。**

        かといって長さで正規化する（`delta / |delta|`）と、今度は**斜めの
        移動が遅くなる。** 軸ごとに切っていたときは 3 軸とも 1 まで使えるので
        最大 √3 倍の速さが出ていた。実際それで 210 ステップが 292 ステップに伸びた。

        **いちばん大きい軸がちょうど振り切れる大きさに揃える。** 向きは
        目標をぴたりと指したまま、使える速度は全部使う。
        """
        distance = float(np.linalg.norm(delta))
        largest = float(np.max(np.abs(delta)))
        if distance < 1e-9 or largest < 1e-9:
            return np.zeros(3)
        return delta / largest * min(1.0, distance / full_speed_within)

    def _smoothed(self, command: np.ndarray, smoothing: float) -> np.ndarray:
        """指令を前回からなます。**加加速度（ジャーク）を直に下げる。**

        `smoothing` が 1.0 なら何もしない。小さいほど滑らかになるが、
        追従が遅れるぶんステップ数は増える。
        """
        if smoothing >= 1.0:
            return command
        blended = smoothing * command + (1.0 - smoothing) * self._last_command
        self._last_command = blended
        return blended

    def _noisy(self, action: np.ndarray) -> np.ndarray:
        """送る行動にだけノイズを混ぜる。**記録するのは混ぜる前である。**

        DART（Laskey+ 2017）のやり方をそのまま写している。手本を書いた側に
        ノイズを混ぜて動かし、**正解としては混ぜる前の行動を記録する。**
        こうすると台本は「ずれた場所から目標へ戻す」行動を自分で作るので、
        **成功した軌道だけを集めても復帰のしかたが学習データに入る。**
        混ぜずに集めた手本は、ずれた状態を一度も通らない
        （`docs/plan/08_training_data_creation_cautions.md` 4 節）。

        **決めたことが 3 つある。どれも変えると別物になる。**

        1. **なました後に混ぜる**（`_smoothed` は呼び出し側で先に済んでいる）。
           先に混ぜると `smoothing` の低い側でノイズが吸われ、**指定した
           大きさが実際に届く大きさと合わなくなる。**
        2. **混ぜた行動を `_last_command` に戻さない。** 戻すとノイズが次の
           ステップへ持ち越されて累積し、**毎ステップ独立ではなく片側へ
           流れる**（DART のノイズは各ステップ独立である）。
           `_smoothed` は混ぜる前の値で履歴を更新済みなので、ここで何もしなければ
           そのまま成り立つ。
        3. **指の開閉（`action[6]`）と手首の向き（`action[3:6]`）には混ぜない。**
           指に混ぜると掴んでいる物を無作為に落とすだけで、復帰は学べない。
           向きは台本が閉ループで直していない（`turn_arc` が固定値を入れるだけ）
           ので、混ぜるとずれが誰にも直されずに溜まり、**手本として誤った
           正解になる。** 手先の位置（`action[:3]`）は `follow` が観測から
           毎ステップ引き直すので、ここだけが直される。

        **振り切れている軸では片側にしか効かない。** `_velocity_command` は
        いちばん大きい軸をちょうど ±1 に揃えるので、運んでいる最中の主軸は
        すでに端にいる。そこへ足したノイズは、外へ出るぶんが ±1 で切られ、
        内へ入るぶんだけが残る。**つまり運搬中は「速さが揺らいで遅れる」
        働きになり、横へ押し出す働きは減速中と目標の近くで強く出る。**
        これは環境側が受け取る範囲そのものなので避けられない（切らなくても
        robosuite の制御器が同じところで切る）。**σ を上げても主軸方向の
        ずれは増えない**ことを、成功率の落ち方を読むときに思い出すこと。
        """
        if self.action_noise <= 0.0:
            return action
        noisy = action.copy()
        noisy[:3] = np.clip(
            noisy[:3] + self._rng.normal(0.0, self.action_noise, size=3),
            -1.0, 1.0,      # 環境が受け取る範囲。混ぜると 1 を越えうる
        )
        return noisy

    def _take_step(self, action: np.ndarray) -> bool:
        """1 ステップ進める。ステップ数の上限に達したら False を返す。"""
        index = self.record.steps
        self.record.ee_positions.append(self.eef_position.copy())
        self.record.states.append(self._observation_state())
        self.record.actions.append(action.astype(np.float32))
        # **観測とノイズ無しの行動を記録してから、混ぜたほうを送る。**
        action = self._noisy(action)
        self.record.executed_actions.append(action.astype(np.float32))
        if self._front_buffer is not None and index < self.max_steps:
            # **学習データは 180 度回した向き**である（LeRobot の同名の処理）。
            # 確保済みの置き場へ直に書く（この代入そのものが複製になる）。
            self._front_buffer[index] = self.obs["agentview_image"][::-1, ::-1]
            self._wrist_buffer[index] = (
                self.obs["robot0_eye_in_hand_image"][::-1, ::-1])

        self.obs, _, _, _ = self.env.env.step(action)
        self.record.steps += 1
        if len(self._watched_keys):
            current = np.stack([self.obs[key] for key in self._watched_keys])
            moved = float(np.abs(current - self._watched_origin).sum(axis=1).max())
            self.record.max_displacement = max(self.record.max_displacement,
                                               moved)
        return self.record.steps < self.max_steps

    def attach_images(self) -> None:
        """溜めた画像を、ステップ数ぶんに切って結果へ渡す。**複製しない。**"""
        if self._front_buffer is None:
            return
        count = min(self.record.steps, self.max_steps)
        self.record.front = self._front_buffer[:count]
        self.record.wrist = self._wrist_buffer[:count]

    def move_to(self, label: str, target: np.ndarray | list[float],
                gripper: float, tolerance: float = 0.005, budget: int = 60
                ) -> None:
        """1 点へ寄せて、そこで止まる。**位置を合わせたい所に使う。**

        `follow` に中継点 1 つを渡すのと同じである。制御の中身を 2 つ持つと
        必ず片方が古くなるので、こちらは呼び名だけを与える薄い包みにしてある。

        **ここは意図的になまさない**（`smoothing` を渡さない）。指令をなますと
        指令が小さくなる終盤で這い、3.8cm 降りるのに 25 ステップ使って届かなくなる。
        滑らかにするのは運ぶ区間だけでよい。
        """
        self.follow(label, [target], gripper, blend_radius=0.0,
                    tolerance=tolerance, budget=budget,
                    brake_distance=BRAKE_WITHIN_STOP)

    def follow(self, label: str, waypoints: list[Any], gripper: float,
               blend_radius: float, tolerance: float = 0.006,
               budget: int = 90, full_speed_within: float = BRAKE_WITHIN_STOP,
               smoothing: float = 1.0,
               brake_distance: float = BRAKE_WITHIN_PASS) -> None:
        """中継点を**止まらずに**繋いで通る。**運ぶ区間に使う。**

        `blend_radius` まで近づいたら、そこへ着く前に次の中継点へ目標を
        移す。**角を丸めて通るので減速も方向転換もしない。** 最後の 1 点
        だけは `tolerance` まで詰める。

        Parameters
        ----------
        blend_radius : float
            途中の中継点をどれだけ手前で見切るか。**大きいほど滑らかだが、
            角が内側に膨らむ。** 障害物の際を通るときは小さくする。
        brake_distance : float
            最後の 1 点の手前、どれだけの距離をかけて減速するか。
            **途中で止まらない代わりに、終点には速度を残したまま着く。**
            そのまま物を離すと引きずって置き場所から外れるので、ここで
            長めに減速させる（短いと狙いから 2.4cm ずれた）。
        """
        used = self.record.steps
        targets = [np.asarray(point, dtype=np.float64) for point in waypoints]
        index = 0
        for _ in range(budget):
            delta = targets[index] - self.eef_position
            distance = float(np.linalg.norm(delta))
            is_last = index == len(targets) - 1
            if distance < (tolerance if is_last else blend_radius):
                if is_last:
                    break
                index += 1
                continue
            action = np.zeros(7)
            action[:3] = self._smoothed(
                self._velocity_command(
                    delta, brake_distance if is_last else full_speed_within),
                smoothing)
            action[6] = gripper
            if not self._take_step(action):
                break
        error = float(np.linalg.norm(targets[-1] - self.eef_position))
        self.phases.append((label, self.record.steps - used, error))

    def joint_anchor(self, name: str) -> np.ndarray:
        """関節の回転の中心。**什器は初期配置ごとに数 mm ずれる**ので実行時に読む。"""
        sim = self.env.env.sim
        return np.asarray(sim.data.xanchor[sim.model.joint_name2id(name)])

    def turn_arc(self, label: str, centre: np.ndarray, step: TurnKnob) -> None:
        """掴んだつまみを、円弧に沿って動かしながら手首も回す。

        **`follow` を使わない。** あちらは中継点に近づいたら次へ飛ぶので、
        回す速さが場面ごとに変わってしまう。ここは**1 ステップに中継点 1 つ**を
        割り当てて、手先の進みと手首の回りを揃える。
        """
        used = self.record.steps
        for index in range(step.turn_steps):
            angle = (index + 1) / step.turn_steps * step.sweep
            target = np.array([centre[0] + step.radius * np.cos(angle),
                               centre[1] + step.radius * np.sin(angle),
                               step.height])
            action = np.zeros(7)
            action[:3] = self._velocity_command(target - self.eef_position,
                                                BRAKE_WITHIN_STOP)
            action[5] = step.yaw
            action[6] = GRIPPER_CLOSE
            if not self._take_step(action):
                break
        self.phases.append((label, self.record.steps - used, 0.0))

    def hold(self, label: str, gripper: float, count: int) -> None:
        """腕を止めたまま指だけ動かす。"""
        used = self.record.steps
        self._last_command = np.zeros(3)      # 止まるので指令の履歴も戻す
        for _ in range(count):
            action = np.zeros(7)
            action[6] = gripper
            if not self._take_step(action):
                break
        self.phases.append((label, self.record.steps - used, 0.0))


def _run_pick_place(rollout: _Rollout, step: PickPlace, plan: TaskPlan) -> None:
    """1 つの物体を掴んで置く。

    **止まってよい所と、止まってはいけない所を分ける。**
    位置を合わせたいのは「掴む」と「置く」の 2 点だけで、そこへ行くまでの
    経路はどこを通ってもよい。**経路の途中で止まるとガタつきになる**ので、
    そこは `follow` で一続きに通す。

    ```
    出発 ──弧──▶ 鍋の真上 ─▶ 掴む位置   ← ここだけ正確に止まる
                            掴む
          ──────弧──────▶ 置く手前      ← 途中で一度も止まらない
                          ▶ 置く位置     ← ここだけ正確に止まる
                            離す
          ──弧──▶ 逃げる
    ```
    """
    name = step.target
    place_xy = rollout.site_position(step.place_site)[:2] + np.asarray(
        step.place_offset)
    position = rollout.object_position(name)

    # 近づく途中。**真上の点は通過点**で、止まる必要がない。
    above = [position[0], position[1], position[2] + step.approach_dz]
    rollout.follow(f"{name} 近づく", [above, [position[0], position[1],
                                             position[2] + step.hover_dz]],
                   GRIPPER_OPEN, blend_radius=plan.blend_radius,
                   budget=plan.budget_move, smoothing=plan.smoothing)

    # **寄っている間に物体が動いていることがある**ので位置を取り直す。
    position = rollout.object_position(name)
    rollout.move_to(f"{name} 掴む位置へ",
                    [position[0], position[1], position[2] + step.grasp_dz],
                    GRIPPER_OPEN, tolerance=0.004, budget=plan.budget_descend)
    rollout.hold(f"{name} 掴む", GRIPPER_CLOSE, plan.close_steps)

    # 掴んでから置く手前まで。**持ち上げと運びを 1 つの弧にする。**
    # 直に運び先へ向かうと什器にぶつけるので、まず持ち上げる点を経由する。
    #
    # **弧はここで終える。** 置く位置の真上まで斜めに降りると、
    # 先に置いた物体に横から当たって落とす。**最後は真下へまっすぐ降ろす。**
    rollout.follow(
        f"{name} 運ぶ",
        [[position[0], position[1], plan.carry_z],
         [place_xy[0], place_xy[1], plan.carry_z]],
        GRIPPER_CLOSE, blend_radius=plan.blend_radius,
        budget=plan.budget_lift + plan.budget_move, smoothing=plan.smoothing)

    rollout.move_to(f"{name} 置く", [place_xy[0], place_xy[1], plan.place_z],
                    GRIPPER_CLOSE, tolerance=0.004, budget=plan.budget_descend)
    rollout.hold(f"{name} 落ち着かせる", GRIPPER_CLOSE, plan.settle_steps)
    rollout.hold(f"{name} 離す", GRIPPER_OPEN, plan.open_steps)
    rollout.follow(f"{name} 逃げる",
                   [[place_xy[0], place_xy[1], plan.retreat_z]],
                   GRIPPER_OPEN, blend_radius=plan.blend_radius,
                   tolerance=0.01, budget=plan.budget_lift,
                   smoothing=plan.smoothing)


def _run_turn_knob(rollout: _Rollout, step: TurnKnob, plan: TaskPlan) -> None:
    """つまみを掴んで回し、離して逃げる。

    ```
    出発 ─▶ 設定値の上   ← 構える
           ▶ 掴む高さ     ← ここだけ正確に止まる
             掴む
           ⤿ 円弧に沿って動かしながら手首を回す   ← ここで点く
             離す
           ▲ 真上へ逃げる
    ```
    """
    centre = rollout.joint_anchor(step.joint)
    grip_at = np.array([centre[0] + step.radius, centre[1], step.height])

    rollout.move_to("つまみへ構える", grip_at + [0.0, 0.0, step.hover_dz],
                    GRIPPER_OPEN, tolerance=0.006, budget=plan.budget_move)
    rollout.move_to("つまみを掴む位置へ", grip_at, GRIPPER_OPEN,
                    tolerance=0.004, budget=plan.budget_descend)
    rollout.hold("つまみを掴む", GRIPPER_CLOSE, plan.close_steps)
    rollout.turn_arc("設定値を回す", centre, step)
    rollout.hold("設定値を離す", GRIPPER_OPEN, plan.open_steps)
    rollout.move_to("設定値から逃げる",
                    [rollout.eef_position[0], rollout.eef_position[1],
                     plan.retreat_z],
                    GRIPPER_OPEN, tolerance=0.01, budget=plan.budget_lift)


# **動きの種類ごとに実行の中身を引く。** 種類が増えても `run_episode` は変わらない。
_RUNNERS: dict[type, Any] = {PickPlace: _run_pick_place, TurnKnob: _run_turn_knob}


def run_episode(env: Any, init_state: np.ndarray, plan: TaskPlan,
                max_steps: int = 300, record_images: bool = False,
                verbose: bool = False, action_noise: float = 0.0,
                noise_seed: int | None = None) -> EpisodeRecord:
    """台本を 1 本ぶん実行し、採点と同じ基準で成否を付ける。

    Parameters
    ----------
    env : Any
        `libero.libero.envs.env_wrapper.ControlEnv`。
    init_state : np.ndarray
        `.pruned_init` から取り出した 1 本ぶんの初期状態。
    plan : TaskPlan
        実行する台本。
    max_steps : int
        ステップ数の上限。**本番の打ち切りは 300 である。**
    record_images : bool
        観測画像も溜めるか。**描画が要るので遅くなる。**
    verbose : bool
        段ごとのステップ数と誤差を印字するか。
    action_noise : float
        送る行動の手先の位置（`action[:3]`）に混ぜる正規ノイズの標準偏差。
        **0.0 なら混ぜない（今までと同じ挙動である）。** 記録する行動は
        混ぜる前のままなので、**これを上げると「ずれてから戻す」手本になる**
        （`_Rollout._noisy` に決めたことを書いた）。
    noise_seed : int | None
        ノイズの種。**本ごとに違う値を渡すこと。** 同じ値だと初期配置が
        違っても同じノイズ列を混ぜてしまい、ずれ方の種類が増えない。

    Returns
    -------
    record : EpisodeRecord
        ステップ数・成否・行動。画像は `rollout.images()` から取る。
    """
    rollout = _Rollout(env, init_state, max_steps, record_images,
                       action_noise=action_noise,
                       rng=np.random.default_rng(noise_seed))
    for step in plan.steps:
        _RUNNERS[type(step)](rollout, step, plan)
    rollout.hold("整定", GRIPPER_OPEN, plan.final_settle_steps)

    record = rollout.record
    record.goal_reached = bool(env.check_success())
    rollout.attach_images()

    if verbose:
        for label, count, error in rollout.phases:
            print(f"   {label:24s} {count:3d} 歩  誤差 {error * 100:5.2f} cm")
        print(f"   合計 {record.steps} 歩  ゴール {record.goal_reached} "
              f"衝突 {record.collided}")
    return record


def _bddl_path(libero_root: Path, suite: str, variant: str) -> Path:
    return libero_root / "bddl_files" / suite / f"{variant}.bddl"


def _init_states(libero_root: Path, suite: str, task: str) -> np.ndarray:
    """初期状態の一覧。**変種ではなく元タスクの名前で引く**（1 タスク 1 file）。"""
    import torch

    path = libero_root / "init_files" / suite / f"{task}.pruned_init"
    return np.asarray(torch.load(path, weights_only=False))


def _make_env(bddl_file: Path, with_images: bool) -> Any:
    """評価と同じ環境を作る。画像が要らないときは描画を切って速くする。

    `control_freq` は評価側と同じ経路（環境変数 `LIBERO_CONTROL_FREQ`）から
    読む。**書き写すと、制御周期を変えて試した回と食い違う。**
    """
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=with_images,
        has_offscreen_renderer=with_images,
        has_renderer=False,
        control_freq=int(os.environ.get("LIBERO_CONTROL_FREQ", "20")),
        horizon=2000,
        ignore_done=True,
        camera_names=list(CAMERAS),
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
    )


def reset_to(env: Any, init_state: np.ndarray) -> dict[str, np.ndarray]:
    """初期状態を書き戻して、採点ハーネスと同じ空撃ちまで済ませる。

    **`measure` と `verify`/`record` で手順が違うと、測った寸法と実際に
    動かす場面がずれる。** 1 か所に置いて両方から呼ぶ。

    空撃ちのステップ数は評価パイプラインに合わせている
    （`pipeline/rollout.py` が方策へ渡す前に `np.zeros` を 10 ステップ撃つ）。
    """
    env.reset()
    # **reset のたびに sim は作り直される。** 取り置くと壊れる。
    sim = env.env.sim
    sim.set_state_from_flattened(init_state)
    sim.forward()
    observation: dict[str, np.ndarray] = {}
    for _ in range(WARMUP_STEPS):
        observation, _, _, _ = env.env.step(np.zeros(7))
    return observation


def _resolve_plan(task: str, smoothing: float | None = None,
                  blend_radius: float | None = None,
                  suite: str | None = None) -> TaskPlan:
    """台本を引く。滑らかさの設定値だけ、その場で差し替えられる。

    **`measure` は台本より先に走る。** 寸法を測ってから台本を書くのに、
    台本が無いと測れないのでは順序が逆になる。`--suite` を渡した場合だけ、
    **空の台本**を作って場面を開けるようにする（`measure` 専用）。
    """
    if task not in TASK_PLANS:
        if suite is not None:
            return TaskPlan(suite=suite, steps=[], carry_z=0.0,
                            place_z=0.0, retreat_z=0.0)
        raise SystemExit(
            f"台本が無い: {task}\n  ある台本: {', '.join(sorted(TASK_PLANS))}\n"
            "  **測るだけなら `--suite libero_10` を渡す**（台本より先に測る）")
    plan = TASK_PLANS[task]
    if smoothing is not None:
        plan = replace(plan, smoothing=smoothing)
    if blend_radius is not None:
        plan = replace(plan, blend_radius=blend_radius)
    return plan


def _collidable_geoms(sim: Any) -> list[tuple[str, float, float, float]]:
    """当たり判定のある geom を（名前, 下端, 上端, 幅）で一覧にする。

    名前で引く表を 1 回だけ作る。**物体ごとに全 geom を走査すると、
    geom 500・物体 10 で 5,500 回になる。**

    見た目だけの geom は外す。**外さないと、コンロの飾りが上端に化けて
    運ぶ高さを無駄に上げてしまう**（実際に上端 0.9921 と 0.9652 の差が出た）。
    """
    return [
        (name,
         float(sim.data.geom_xpos[i][2] - sim.model.geom_size[i][2]),
         float(sim.data.geom_xpos[i][2] + sim.model.geom_size[i][2]),
         float(2 * max(sim.model.geom_size[i][0], sim.model.geom_size[i][1])))
        for i in range(sim.model.ngeom)
        if sim.model.geom_contype[i]
        and (name := sim.model.geom_id2name(i) or "")
    ]


def _report_gripper_span(sim: Any) -> None:
    """指の開き。**これより広い物体は、その幅の所では掴めない。**"""
    print("\n--- 指の開き ---")
    pads = [i for i in range(sim.model.ngeom)
            if (sim.model.geom_id2name(i) or "").endswith("_pad_collision")]
    if len(pads) != 2:
        print("  指のパッドが 2 つ見つからない。**掴める幅は自分で測ること**")
        return
    gap = float(np.linalg.norm(sim.data.geom_xpos[pads[0]]
                               - sim.data.geom_xpos[pads[1]]))
    print(f"  指の間隔 {gap * 100:.1f} cm  **これより広い物体は掴めない**")


def _report_collision_extents(env: Any, sim: Any) -> None:
    """物体と什器の当たり判定。**運ぶ高さと掴む高さを決める値。**"""
    print("\n--- 物体・什器の当たり判定 ---")
    print("  **運ぶ高さは、什器の上端＋物体の背丈より高く取る。**")
    geoms = _collidable_geoms(sim)
    for name, body_id in env.env.obj_body_id.items():
        # 什器は子の body に geom がぶら下がるので、**body ではなく名前で拾う。**
        # body で拾うとコンロの上端が 1 つも出ず、運搬中にぶつける。
        extents = [geom for geom in geoms if geom[0].startswith(name)]
        if not extents:
            continue
        origin_z = float(sim.data.body_xpos[body_id][2])
        low = min(extent[1] for extent in extents)
        high = max(extent[2] for extent in extents)
        widest = max(extent[3] for extent in extents)
        print(f"  {name:16s} 原点 z {origin_z:.4f}  当たる高さ {low:.4f}〜{high:.4f}"
              f"（原点から {low - origin_z:+.4f}〜{high - origin_z:+.4f}）"
              f"  いちばん広い所 {widest * 100:.1f} cm")


def _report_goal_sites(env: Any, sim: Any) -> None:
    """ゴール判定に使われる枠。**見た目で決めず、この枠の中へ置く。**"""
    print("\n--- ゴール判定の枠 ---")
    for site_name in env.env.object_sites_dict:
        if "init_region" in site_name:
            continue
        position = sim.data.get_site_xpos(site_name)
        size = getattr(env.env.object_sites_dict[site_name], "size", None)
        print(f"  {site_name:34s} 中心 {np.round(position, 4)} 半幅 {size}")


def _report_articulation(env: Any, sim: Any) -> None:
    """関節を持つ什器（コンロの設定値・引き出し）の状態と、点く境目を出す。

    **「点ける」は関節の角度で判定される。** `flat_stove` なら
    `qpos >= 0.5`（`libero/libero/envs/objects/articulated_objects.py` の
    `default_turnon_ranges` の下限）で、**どの関節でもよい**（`turn_on` は
    どれか 1 つが超えれば真を返す）。**見た目ではなくこの値を狙う。**
    """
    print("\n--- 関節（設定値・扉）---")
    found = False
    for name in env.env.obj_body_id:
        obj = env.env.get_object(name) if hasattr(env.env, "get_object") else None
        joints = list(getattr(obj, "joints", None) or [])
        for joint in joints:
            if not isinstance(joint, str):
                continue
            try:
                address = sim.model.get_joint_qpos_addr(joint)
            except Exception:  # noqa: BLE001 - 名前が無いだけ
                continue
            # **`int` で判定すると取りこぼす。** mujoco は numpy の整数を返す
            # （自由関節など複数自由度のときは範囲の対を返すので、そこは飛ばす）。
            if not isinstance(address, (int, np.integer)):
                continue
            found = True
            index = sim.model.joint_name2id(joint)
            limited = bool(sim.model.jnt_limited[index])
            low, high = sim.model.jnt_range[index]
            ranges = (getattr(obj, "object_properties", {})
                      .get("articulation", {}))
            print(f"  {joint:34s} いま {sim.data.qpos[address]:+.4f}"
                  f"  可動 {low:+.3f}〜{high:+.3f}{'' if limited else '（制限なし）'}")
            for key, value in ranges.items():
                print(f"      {key}: {value}")
    if not found:
        print("  関節を持つ物体が無い（掴んで運ぶだけの場面）")


def _setup(args: argparse.Namespace, with_images: bool
           ) -> tuple[TaskPlan, str, Any, np.ndarray]:
    """3 つの口に共通の下ごしらえ。台本・変種名・環境・初期状態を揃える。

    **`--variant` 省略時は元タスクそのもの**という決まりを 1 か所に置く。
    """
    plan = _resolve_plan(args.task, args.smoothing, args.blend_radius,
                         getattr(args, "suite", None))
    variant = args.variant or args.task
    env = _make_env(_bddl_path(args.libero_root, plan.suite, variant),
                    with_images)
    return plan, variant, env, _init_states(args.libero_root, plan.suite,
                                            args.task)


def command_measure(args: argparse.Namespace) -> None:
    """台本を書く前に、場面の寸法を測って印字する。

    **推測で台本を書かない。** 指の開き・物体の幅・什器の上端は
    ここで出た値をそのまま使う。
    """
    plan, variant, env, init_states = _setup(args, with_images=False)
    observation = reset_to(env, init_states[0])
    sim = env.env.sim

    print(f"=== {variant} ===")
    print(f"動かしてよい物体（obj_of_interest）: {env.obj_of_interest}")
    print(f"手先の初期位置: {np.round(observation['robot0_eef_pos'], 4)}")
    _report_gripper_span(sim)
    _report_collision_extents(env, sim)
    _report_articulation(env, sim)
    _report_goal_sites(env, sim)
    print(f"\n初期配置 {len(init_states)} 通り")


def command_verify(args: argparse.Namespace) -> None:
    """初期配置すべてで台本が通るかを数える。**描画しないので速い。**"""
    plan, variant, env, init_states = _setup(args, with_images=False)
    if args.episodes:
        init_states = init_states[args.episodes]

    print(f"=== {variant} — {len(init_states)} 通りの初期配置 ===")
    if args.action_noise > 0.0:
        print(f"  行動ノイズ σ={args.action_noise}（送る行動だけに混ぜる）")
    started = time.time()
    lengths: list[int] = []
    failures: list[tuple[int, int, bool]] = []
    measured: list[dict[str, float]] = []
    for index, state in enumerate(init_states):
        record = run_episode(env, state, plan, args.max_steps,
                             verbose=args.verbose and index == 0,
                             action_noise=args.action_noise,
                             noise_seed=args.noise_seed + index)
        lengths.append(record.steps)
        if not record.success:
            failures.append((index, record.steps, record.collided))
        else:
            measured.append(smoothness(record.ee_positions))

    succeeded = len(lengths) - len(failures)
    print(f"成功 {succeeded}/{len(lengths)}  "
          f"ステップ数 中央値 {int(statistics.median(lengths))} "
          f"最小 {min(lengths)} 最大 {max(lengths)}  "
          f"（上限 {args.max_steps}）")
    if failures:
        print(f"  失敗（番号, ステップ数, 衝突）: {failures[:20]}")
    if measured:
        middle = {key: statistics.median(m[key] for m in measured)
                  for key in ("avg_jerk", "rms_jerk", "sparc", "path_length")}
        print("  滑らかさ（採点と同じ式・中央値）  "
              f"平均ジャーク {middle['avg_jerk']:.2f}"
              f"  RMS ジャーク {middle['rms_jerk']:.2f}"
              f"  SPARC {middle['sparc']:.2f}"
              f"  手先の移動距離 {middle['path_length']:.2f} m")
    print(f"  {time.time() - started:.0f} 秒")


def command_record(args: argparse.Namespace) -> None:
    """学習データとして書き出す。**描画するので GPU を使うこと。**

    **`--action-noise` を付けたときも、失敗した本は既定で捨てる。**
    これは DART のやり方と合っている——欲しいのは「ずれたが戻って成功した」
    軌道であって、戻れずに終わった軌道ではない。**捨てた本の数は
    そのまま σ が大きすぎる合図**なので、印字を読むこと。

    Generates:
      - <output>/<変種>_ep<n>.npz（`--action-noise` を付けると `_n<σ>` が入る）
    """
    plan, variant, env, init_states = _setup(args, with_images=True)
    indices = args.episodes if args.episodes else list(range(len(init_states)))

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"=== {variant} — {len(indices)} 本を書き出す ===")
    print(f"  MUJOCO_GL={os.environ.get('MUJOCO_GL', '(未設定)')}")
    if args.action_noise > 0.0:
        print(f"  行動ノイズ σ={args.action_noise}"
              f"（種 {args.noise_seed}＋本の番号）。"
              "**書き出す行動は混ぜる前である**")
    for index in indices:
        started = time.time()
        record = run_episode(env, init_states[index], plan, args.max_steps,
                             record_images=True,
                             action_noise=args.action_noise,
                             noise_seed=args.noise_seed + index)
        if not record.success and not args.keep_failures:
            print(f"  ep{index:02d} 失敗（{record.steps} 歩）—— 書き出さない")
            continue
        # **ノイズの大きさを名前に入れる。** 入れないと、σ を変えて作った組が
        # 同じ名前になって上書きし合う（同じ変種・同じ番号で作るため）。
        tag = "" if args.action_noise <= 0.0 else f"_n{args.action_noise:g}"
        destination = args.output / f"{variant}{tag}_ep{index:03d}.npz"
        # **圧縮しない。** これは動画に再エンコードするだけの中継物で、
        # 画像 118 MB を deflate しても 1.7 倍しか縮まないのに 1 本 4.3 秒
        # かかる（50 本で 3.6 分）。disk を使うほうが安い。
        np.savez(
            destination,
            state=np.stack(record.states),
            action=np.stack(record.actions),
            front=record.front,
            wrist=record.wrist,
            success=record.success,
            steps=record.steps,
            instruction=env.language_instruction,
            # **学習に使うのは `action` のほうである。** 下の 2 つは
            # 「ノイズが本当に効いていたか」を後から確かめるためだけに残す。
            # 鍵が増えても `SavedEpisode` 経由で読む側は影響を受けない。
            action_executed=np.stack(record.executed_actions),
            action_noise=record.action_noise,
        )
        print(f"  ep{index:02d} {record.steps} 歩  成功 {record.success}  "
              f"{time.time() - started:.0f} 秒  → {destination.name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("measure", "verify", "record"),
                        help="measure=寸法を測る / verify=通るか数える / "
                             "record=学習データに書き出す")
    parser.add_argument("--task", required=True,
                        help="元タスクの名前（bddl から摂動の接尾辞を落としたもの）")
    parser.add_argument("--variant", default=None,
                        help="実際に使う変種の bddl 名。省略時は元タスクそのもの")
    parser.add_argument("--suite", default=None,
                        help="**台本を書く前に measure だけ回すときに渡す**"
                             "（例 libero_10）。台本があるときは要らない")
    parser.add_argument("--libero-root", type=Path, default=_DEFAULT_LIBERO_ROOT,
                        help=f"LIBERO-plus の libero/libero（既定 {_DEFAULT_LIBERO_ROOT}）")
    parser.add_argument("--episodes", default=None,
                        help="初期配置の番号をカンマ区切りで。省略時は全部")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="ステップ数の上限（既定 300。本番の打ち切りと同じ）")
    parser.add_argument("--output", type=Path, default=Path("outputs/scripted"),
                        help="record の書き出し先")
    parser.add_argument("--keep-failures", action="store_true",
                        help="失敗した回も書き出す（見て直すため）")
    parser.add_argument("--smoothing", type=float, default=None,
                        help="指令をなます強さ（1.0 でなまさない）。台本の値を上書きする")
    parser.add_argument("--action-noise", type=float, default=0.0,
                        help="送る行動の手先の位置に混ぜる正規ノイズの標準偏差"
                             "（行動の範囲は ±1）。**記録する行動は混ぜる前の"
                             "ままなので、ずれてから戻す手本になる**（DART）。"
                             "0.0 で今までと同じ。**大きすぎると成功率が落ちて"
                             "逆効果になる**ので、先に verify で振って決めること")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="ノイズの種の基準値。本ごとに番号を足して使う")
    parser.add_argument("--blend-radius", type=float, default=None,
                        help="通過点を見切る距離 [m]。台本の値を上書きする")
    parser.add_argument("--verbose", action="store_true",
                        help="1 本目だけ段ごとのステップ数を出す")
    args = parser.parse_args()
    if args.episodes:
        args.episodes = [int(x) for x in args.episodes.split(",")]
    return args


def main() -> None:
    args = _parse_args()
    {"measure": command_measure,
     "verify": command_verify,
     "record": command_record}[args.command](args)


if __name__ == "__main__":
    main()
