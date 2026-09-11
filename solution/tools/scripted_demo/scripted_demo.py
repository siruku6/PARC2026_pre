#!/usr/bin/env python3
"""手本より速い軌道を、シミュレータの中で scripted policy から作る。

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

    # 1. 場面の寸法を測る（新しいタスクの scripted policy を書く前に必ずやる）
    python3 solution/tools/scripted_demo/scripted_demo.py measure \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove

    # 2. 初期配置すべてで通るか数える（描画しないので 1 本 2.4 秒）
    python3 solution/tools/scripted_demo/scripted_demo.py verify \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove

    # 3. 学習データとして書き出す（描画するので 1 本 100 秒。**並列で回す**）
    MUJOCO_GL=osmesa python3 solution/tools/scripted_demo/scripted_demo.py record \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \\
        --variant KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_2 \\
        --episodes 0,1,2 --output /out/scripted/

    # 4. **ずれてから戻す手本にする**（本選 Track2 向け。DART のやり方）
    #    送る行動にだけノイズを混ぜ、正解としては混ぜる前を記録する。
    #    **σ は大きすぎると成功率が落ちて逆効果になる**ので、
    #    描画しない verify で振ってから決める（1 本 2.4 秒）。
    for sigma in 0 0.02 0.05 0.10; do
      python3 solution/tools/scripted_demo/scripted_demo.py verify \\
        --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \\
        --action-noise $sigma
    done

**描画は CPU（`osmesa`）で行う。** 手元の GPU マシンの vGPU は
`NVIDIA Virtual Compute Server` で OpenGL に対応せず、`egl` を指定すると
**空の場面は 0.4 ms で描けるのに、実際の場面で止まる。**
速くする手は並列だけである（`--episodes` を分けて同時に流す。手順は
`docs/howto/05_scripted_demos.md` の 3 節）。

**「fixture」はコンロ・棚・引き出しのように、シーンに据え置かれた物である**
（LIBERO のコードの呼び名。掴んで運ぶ対象の物体とは別で、自由関節を持たない）。

**scripted policy を書くときに必ず先に測ること**（`measure` が出す）。

| 測るもの | なぜ | moka pots で実際に踏んだ穴 |
|---|---|---|
| 指の開き 対 物体の幅 | 開かなければ掴めない | 本体は 5cm 角、指は 4.8cm しか開かない。**掴めるのは蓋のつまみだけ** |
| 当たり判定のある fixture の上端 | 運搬中にぶつける | コンロ上面 0.9305 に対し運搬中の鍋の底が 0.922。8mm 潜っていた |
| 1 ステップで進む距離 | ステップ数の見積もり | OSC の追従の限界で **1 ステップ 1.25cm**。行動の上限 5cm には届かない |

Reads:
  - <libero-root>/bddl_files/<suite>/<variant>.bddl
  - <libero-root>/init_files/<suite>/<元タスク>.pruned_init
Generates:
  - 標準出力
  - `record` のときのみ <output>/<variant>_ep<n>.npz（観測・行動・成否）
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

# 指の開き [m]。**実測**（`solution/tools/probe/gripper_frame.py`）。
# **掴む対象の水平の張り出しがこの半分を超えると、縁をまたぐしかない。**
FINGER_SPAN = 0.0481

# 掴む方位を選ぶとき、**これだけ空いていれば十分**とみなす距離 [m]。
# **余裕をひたすら最大にすると、腕が遠回りして歩数が伸びる**——実測で
# 余裕 8.8 cm の方位を選んだ結果、同じ丼の課題が 151 歩から 273 歩になり、
# 上限 300 歩に当たってゴールに届かなくなった。**一定以上は同点にして、
# 残りは「ロボットの基部に近い側」で決める。**
# **隙間の狭い場面では上げたほうがよい**ので、`GRASP_CLEARANCE_ENOUGH` で振れる。
GRASP_CLEARANCE_ENOUGH = float(os.environ.get("GRASP_CLEARANCE_ENOUGH", 0.030))

# 方位を選ぶとき、fixture から最低これだけ離れていることを求める [m]。
# **fixture は採点の衝突判定の対象外**（観測に `_pos` を持たない）だが、
# ぶつけると腕が逸れて、そのあとの動きが全部狂う。
FIXTURE_MARGIN = 0.020

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
        fixture は初期配置ごとに数 mm ずれるので、**絶対座標で書くと variant で外れる。**
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
class RimPickPlace:
    """**指より広い物**を、縁をまたいで掴んで置く 1 まとまり。

    **なぜ `PickPlace` と別なのか。**あちらは物体の真上から降りて掴む前提で、
    **物体の幅が指の開きより狭いときだけ成り立つ。**丼は水平の張り出しが
    5.65 cm あり、指は 4.81 cm しか開かない（実測）。だから手首は丼の軸から
    5.65 cm 外へ出し、**縁の壁を 2 本の指で挟む**しかない。

    **この「外へ出す向き」が自由変数である。**丼は Z 軸まわりに回転対称なので
    どの方位からまたいでも掴めるのに、**方位によって隣の物との余裕が
    数 mm から 10 cm まで変わる。**`pick up the black bowl next to the cookie box`
    では丼とクッキー箱の隙間が 3.42 cm しかなく、箱の側からまたぐと
    外側の指の余裕は 2 mm ほどになる——人の手本を再生すると
    **1 mm を超えた 6 本すべてで指が箱に触れていた**
    （`solution/tools/contact_clean/contact_phase.py`）。
    **空いている方位は 209 度あるので、選べば当たらない。**

    Attributes
    ----------
    target : str
        掴む物体の観測名。**Z 軸まわりに回転対称な物にだけ使える**
        （方位を勝手に選ぶので、向きに意味がある物では壊れる）。
    place_site : str | None
        置く先の基準にする site。`place_object` と**どちらか一方**を指定する。
    place_object : str | None
        置く先の基準にする物体の観測名。**動く物（皿など）はこちらを使う**
        ——site は初期配置ごとにずれるが、物体の位置は観測から正確に読める。
    place_offset : tuple[float, float]
        基準から見た、置く先の水平方向のずれ。
    grasp_dz : float
        物体の原点から見て、どの高さで縁を挟むか。**縁の上端より下、
        パッドの背丈（1.6 cm）が縁にかかる高さに取る。**
    approach_dz : float
        降りる前に構える高さ。**寄りながらここまで手首を回す。**
    hover_dz : float
        **降りるのを 2 段に割る中間の高さ。**指が縁に近づく前に手首を回しきる
        ため、ここまでは回しながら降り、ここから下は真下へまっすぐ降りる。
    lift_dz : float
        掴んだあと、**真上へ**上げる高さ。
    descend_speed : float
        **縁を挟む位置へ降りるときの指令の上限。**台の上に載っている物では、
        勢いよく降りると指が器に当たった衝撃で器が台の上を滑る。
    lift_speed : float
        持ち上げるときの指令の上限。**1.0 のまま上げると隣の物へ押し付ける。**
    grasp_radius : float | None
        手首を物体の軸から何 m 外へ出すか。**None なら `rim_radius` が
        掴む高さの「縁の壁の中心」を測る**（張り出しではない）。
    clearance_band : tuple[float, float]
        物体の原点から見て、どの高さの帯で方位の空き具合を見るか。
    """

    target: str
    place_site: str | None = None
    place_object: str | None = None
    place_offset: tuple[float, float] = (0.0, 0.0)
    grasp_dz: float = 0.042
    approach_dz: float = 0.14
    hover_dz: float = 0.07
    lift_dz: float = 0.10
    descend_speed: float = 1.0
    lift_speed: float = 0.45
    grasp_radius: float | None = None
    clearance_band: tuple[float, float] = (-0.02, 0.10)

    def __post_init__(self) -> None:
        if (self.place_site is None) == (self.place_object is None):
            raise ValueError(
                "place_site と place_object は**どちらか一方だけ**を指定する")


@dataclass
class TurnKnob:
    """コンロの火力つまみを掴んで回す。
    **「点ける」は角度で判定される。**

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
        回る関節の名前。**回転の中心はこの関節から実行時に読む**（fixture は
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
    """1 タスクぶんの scripted policy。

    Attributes
    ----------
    suite : str
        `libero_10` など、bddl の置き場所。
    steps : list[PickPlace | RimPickPlace | TurnKnob]
        前から順に実行する。**動きの種類ごとに実行の中身が違う**
        （`_RUNNERS` で引く）。
    carry_z : float
        運ぶときの手先の高さ。**fixture の上端より、物体の背丈のぶん高く取る。**
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
        膨らむ。** fixture の際を通る経路では小さくする。
    cruise : float
        **巡航中に出す指令の大きさ**（1.0 が既定で、いちばん大きい軸を振り切る）。
        **振り切ったままにすると、指令が上限の 1 点に積み上がる**——配布データが
        上限付近に居るのは 7.79% なのに、既定のままだと 35.5% になった
        （`docs/reference/33_gradient_norm_and_action_saturation.md`）。
        **配布データの並進指令の中央は 0.603** なので、そこへ寄せる。
    command_envelope : bool
        **配布データから測った「封筒」に指令を収めるか**
        （`solution/tools/motion/envelope.py`）。既定の False は今までどおりで、
        `_velocity_command` の出す ±1.0 をそのまま送る。**True にすると
        並進 ±0.9375・回転 ±0.30 に収め、1 コマの変化量も p99 で抑える。**

        **なぜ要るか。**自作した手本を配布データと混ぜて学習する以上、
        **指令の分布が違うとそこだけ別物になって学習が解けない**
        （`docs/reference/33_gradient_norm_and_action_saturation.md`）。
        とくに手首を回す `action[5]` は、配布データでは ±0.375 が端なのに、
        素朴に比例制御で書くと ±1.0 まで出てしまう。
    smoothing : float
        指令を前回からなます強さ。**既定の 1.0 はなまさない。**
        向きを保つ指令と角の丸めだけでジャークは 18.3 → 12.3 まで下がり、
        **なましを足すと滑らかさはほとんど変わらないままステップ数だけ増えた**
        （0.6 で 234 ステップ・0.35 で 264 ステップ）。細かく動かしたいときだけ下げる。
    """

    suite: str
    steps: list[PickPlace | RimPickPlace | TurnKnob]
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
    cruise: float = 1.0
    command_envelope: bool = False
    smoothing: float = 1.0


# **scripted policy はここに足していく。** 鍵は元タスクの名前（bddl から摂動の接尾辞を
# 落としたもの）。摂動が見た目だけなら（照明・背景）、同じ scripted policy がそのまま通る。
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
    # ---- 丼の 5 課題。**評価で失点の 65% を出している家族である**
    # （`docs/plan/24_contact_avoidance_redesign.md`）。丼は Z 軸まわりに
    # 回転対称なので、**縁をまたぐ方位を実行時に選べる**——`RimPickPlace` が
    # 余裕のいちばん大きい方位を選ぶので、ここに方位は書かない。
    # 高さは実測（丼 当たる高さ 0.8864〜0.9508・原点 z 0.8984、
    # クッキー箱の上端 0.9507、皿 0.8751〜0.9380）。
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate": TaskPlan(
        suite="libero_spatial",
        # 隙間 3.46 cm・空いている方位 209 度。**指が箱に当たるのが 15/15**
        steps=[RimPickPlace(
            "akita_black_bowl_1", place_object="plate_1",
            grasp_dz=0.042,       # 手先 0.940。パッド 1.6 cm が縁の壁にかかる
            approach_dz=0.15,
            lift_dz=0.10,
            lift_speed=1.0)],    # 台に載っていないので絞る理由が無い
        carry_z=1.05,     # 丼の底が 0.996。クッキー箱の上端 0.9507 を越える
        place_z=0.972,    # 丼の底が 0.918。皿の縁の上端 0.9380 の内側へ降りる
        retreat_z=1.08,
        blend_radius=0.030,
        cruise=0.85,             # 上限の 1 点への積み上がりを消す（35.5% → 0%）
        command_envelope=True,   # 配布データと同じ指令の形にする
        # **止まる歩数を削る。**指令が 0 の 1 点に積み上がると学習が解けない
        # （配布データの停止コマは 0.96%。既定のままだと 24.2% になった）。
        close_steps=10, settle_steps=4, open_steps=6, final_settle_steps=4,
        budget_move=60,   # 出発から丼まで 35 cm ある（45 歩では届かなかった）
        budget_descend=30,
    ),
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate": TaskPlan(
        suite="libero_spatial",
        # 隙間 1.96 cm・空いている方位 314 度
        steps=[RimPickPlace(
            "akita_black_bowl_1", place_object="plate_1",
            grasp_dz=0.042,       # 手先 0.940。パッド 1.6 cm が縁の壁にかかる
            approach_dz=0.15,
            lift_dz=0.10,
            lift_speed=1.0)],    # 台に載っていないので絞る理由が無い
        carry_z=1.05,     # 丼の底が 0.996。クッキー箱の上端 0.9507 を越える
        place_z=0.972,    # 丼の底が 0.918。皿の縁の上端 0.9380 の内側へ降りる
        retreat_z=1.08,
        blend_radius=0.030,
        cruise=0.85,             # 上限の 1 点への積み上がりを消す（35.5% → 0%）
        command_envelope=True,   # 配布データと同じ指令の形にする
        # **止まる歩数を削る。**指令が 0 の 1 点に積み上がると学習が解けない
        # （配布データの停止コマは 0.96%。既定のままだと 24.2% になった）。
        close_steps=10, settle_steps=4, open_steps=6, final_settle_steps=4,
        budget_move=60,   # 出発から丼まで 35 cm ある（45 歩では届かなかった）
        budget_descend=30,
    ),
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": TaskPlan(
        suite="libero_spatial",
        # 皿との隙間 1.17 cm・空いている方位 125 度
        steps=[RimPickPlace(
            "akita_black_bowl_1", place_object="plate_1",
            grasp_dz=0.042,       # 手先 0.940。パッド 1.6 cm が縁の壁にかかる
            approach_dz=0.15,
            lift_dz=0.10,
            lift_speed=1.0)],    # 台に載っていないので絞る理由が無い
        carry_z=1.05,     # 丼の底が 0.996。クッキー箱の上端 0.9507 を越える
        place_z=0.972,    # 丼の底が 0.918。皿の縁の上端 0.9380 の内側へ降りる
        retreat_z=1.08,
        blend_radius=0.030,
        cruise=0.85,             # 上限の 1 点への積み上がりを消す（35.5% → 0%）
        command_envelope=True,   # 配布データと同じ指令の形にする
        # **止まる歩数を削る。**指令が 0 の 1 点に積み上がると学習が解けない
        # （配布データの停止コマは 0.96%。既定のままだと 24.2% になった）。
        close_steps=10, settle_steps=4, open_steps=6, final_settle_steps=4,
        budget_move=60,   # 出発から丼まで 35 cm ある（45 歩では届かなかった）
        budget_descend=30,
    ),
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": TaskPlan(
        suite="libero_spatial",
        # **箱の上に載っている。**方位では直らないので、真上へゆっくり上げる
        steps=[RimPickPlace(
            "akita_black_bowl_1", place_object="plate_1",
            grasp_dz=0.042,       # 手先 0.940。パッド 1.6 cm が縁の壁にかかる
            approach_dz=0.20,
            descend_speed=0.35,   # 器に当たった衝撃で台の上を滑る
            lift_dz=0.14,         # 載っている台から確実に離す
            lift_speed=0.35)],
        carry_z=1.12,     # **台の上から掴むので手先がもともと高い。**丼の底 1.066
        place_z=0.972,    # 丼の底が 0.918。皿の縁の上端 0.9380 の内側へ降りる
        retreat_z=1.08,
        blend_radius=0.030,
        cruise=0.85,             # 上限の 1 点への積み上がりを消す（35.5% → 0%）
        command_envelope=True,   # 配布データと同じ指令の形にする
        # **止まる歩数を削る。**指令が 0 の 1 点に積み上がると学習が解けない
        # （配布データの停止コマは 0.96%。既定のままだと 24.2% になった）。
        close_steps=10, settle_steps=4, open_steps=6, final_settle_steps=4,
        budget_move=60,   # 出発から丼まで 35 cm ある（45 歩では届かなかった）
        budget_descend=30,
    ),
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": TaskPlan(
        suite="libero_spatial",
        # **ラメキンの上に載っている。**手本 20/20 が汚れ、damage の 50% が運搬中
        steps=[RimPickPlace(
            "akita_black_bowl_1", place_object="plate_1",
            grasp_dz=0.042,       # 手先 0.940。パッド 1.6 cm が縁の壁にかかる
            approach_dz=0.20,
            descend_speed=0.35,   # 器に当たった衝撃で台の上を滑る
            lift_dz=0.14,         # 載っている台から確実に離す
            lift_speed=0.35)],
        carry_z=1.12,     # **台の上から掴むので手先がもともと高い。**丼の底 1.066
        place_z=0.972,    # 丼の底が 0.918。皿の縁の上端 0.9380 の内側へ降りる
        retreat_z=1.08,
        blend_radius=0.030,
        cruise=0.85,             # 上限の 1 点への積み上がりを消す（35.5% → 0%）
        command_envelope=True,   # 配布データと同じ指令の形にする
        # **止まる歩数を削る。**指令が 0 の 1 点に積み上がると学習が解けない
        # （配布データの停止コマは 0.96%。既定のままだと 24.2% になった）。
        close_steps=10, settle_steps=4, open_steps=6, final_settle_steps=4,
        budget_move=60,   # 出発から丼まで 35 cm ある（45 歩では届かなかった）
        budget_descend=30,
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
    # **どの物体をいつ動かしたか。**`max_displacement` だけでは「衝突した」しか
    # 分からず、**打つ手が決まらない**（掴む前か、運搬中か、どの物か）。
    object_displacements: dict[str, float] = field(default_factory=dict)
    first_violation: tuple[str, int] | None = None
    collision_threshold_m: float = 0.001
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    # **実際にシミュレータへ送った行動。** `--action-noise` を使ったときだけ
    # `actions` と違う（学習データに入れるのは `actions` のほうである）。
    # 送ったほうも残すのは、**ノイズが本当に効いていたかを後から確かめる**ため。
    executed_actions: list[np.ndarray] = field(default_factory=list)
    # scripted policy に混ぜたノイズの大きさ（標準偏差）。0.0 なら混ぜていない。
    action_noise: float = 0.0
    # **一度だけ大きく妨害した記録**（`--disturb-scale`）。0.0 なら妨害していない。
    # 何を・いつ・どちらへ・何歩やったかを残す（後から条件別に選び直せるように）。
    disturb_scale: float = 0.0
    disturb_start: int = -1
    disturb_steps: int = 0
    disturb_direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_positions: list[np.ndarray] = field(default_factory=list)
    # **その歩までの「対象外の物体のずれの最大」**（m）。毎歩ぶん残す。
    # `max_displacement` は最後の 1 つしか持たないので、動画に焼いたり
    # 「どの歩で 1 mm を超えたか」を見るのに足りない。
    displacements: list[float] = field(default_factory=list)
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
        # **`--disturb-scale` より前に書いた npz にはこの 3 つが無い。**
        self.disturb_scale = (float(self._payload["disturb_scale"])
                              if "disturb_scale" in self._payload else 0.0)
        self.disturb_start = (int(self._payload["disturb_start"])
                              if "disturb_start" in self._payload else -1)
        self.disturb_steps = (int(self._payload["disturb_steps"])
                              if "disturb_steps" in self._payload else 0)

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
                 rng: np.random.Generator | None = None,
                 disturb_scale: float = 0.0, disturb_steps: int = 10,
                 disturb_at: tuple[int, int] | None = None,
                 envelope: Any = None, cruise: float = 1.0) -> None:
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
        # **封筒は「送る前」に当てる。**記録する行動と送る行動が食い違うと、
        # 学習の正解が実際に起きた動きと合わなくなる。
        self._envelope = envelope
        self._last_sent = np.zeros(6)
        self.cruise = float(cruise)
        self.record.action_noise = self.action_noise

        # **妨害の中身は 1 歩も進める前に引く。** ここで引けば、
        # 毎ステップのノイズを何回引いたかに関係なく同じ種から同じ妨害が出る。
        self.disturb_scale = float(disturb_scale)
        self.disturb_steps = int(disturb_steps)
        if self.disturb_scale > 0.0 and disturb_at is not None:
            low, high = disturb_at
            self.disturb_start = int(self._rng.integers(low, high + 1))
            drawn = self._rng.normal(size=3)
            norm = float(np.linalg.norm(drawn))
            self.disturb_direction = drawn / (norm if norm > 0.0 else 1.0)
        else:
            self.disturb_start = -1
            self.disturb_direction = np.zeros(3)
        self.record.disturb_scale = self.disturb_scale
        self.record.disturb_start = self.disturb_start
        self.record.disturb_steps = (self.disturb_steps
                                     if self.disturb_start >= 0 else 0)
        self.record.disturb_direction = self.disturb_direction

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

    @functools.cached_property
    def _pad_geoms(self) -> tuple[int, int]:
        """指のパッドの geom 番号。**指の向きはここから測る。**

        `robot0_eef_quat` からも手首の回転は読めるが、**指がその回転の
        どの軸に並ぶかは実装依存**なので、パッドの実座標から測る。
        """
        sim = self.env.env.sim
        pads = tuple(i for i in range(sim.model.ngeom)
                     if (sim.model.geom_id2name(i) or "").endswith("_pad_collision"))
        if len(pads) != 2:
            raise RuntimeError(
                f"指のパッドが 2 つ見つからない（{len(pads)} 個）。"
                "**掴む向きを選ぶには 2 つ要る**")
        return pads       # type: ignore[return-value]

    @property
    def finger_azimuth(self) -> float:
        """指が並ぶ向き [度]。**180 度回しても同じ向きである**（指は 2 本）。"""
        sim = self.env.env.sim
        first, second = self._pad_geoms
        axis = sim.data.geom_xpos[second] - sim.data.geom_xpos[first]
        return float(np.degrees(np.arctan2(axis[1], axis[0])))

    def object_extent(self, name: str) -> float:
        """物体の水平の張り出し [m]。**原点から測り直す。**

        `geom_size` をそのまま読むと**geom 1 つの幅**しか出ない。丼のように
        本体が小さな箱の集まりでできている物では、それは全体の幅にならない
        （丼は geom 1 つ 3.1 cm だが、全体では 5.65 cm 張り出す）。
        """
        sim = self.env.env.sim
        centre = self.object_position(name)
        geoms = [i for i in range(sim.model.ngeom)
                 if sim.model.geom_contype[i]
                 and (sim.model.geom_id2name(i) or "").startswith(name)]
        if not geoms:
            raise RuntimeError(f"{name} の当たり判定が無い")
        return max(
            float(np.linalg.norm(sim.data.geom_xpos[i][:2] - centre[:2])
                  + max(sim.model.geom_size[i][0], sim.model.geom_size[i][1]))
            for i in geoms)

    def rim_radius(self, name: str, height: float) -> float:
        """その高さで、**縁の壁の中心**が物体の軸から何 m の所にあるか。

        **ここを外すと掴めない。**丼の縁の壁は半径 3.78〜5.65 cm の間にあり、
        厚みは 1.87 cm ある（実測。`probe/_bowl_profile.py`）。指の開きは
        4.81 cm なので、**手首を壁の中心 4.71 cm に置くと、内側の指は 2.31 cm・
        外側の指は 7.11 cm から閉じ始め、壁のどちら側にも 1.46 cm ずつで届く。**
        左右が同時に当たるので、閉じる力が打ち消し合う。

        **手首を内側へ置くと壊れる。**4.0 cm に置くと外側の指が 0.75 cm で
        壁に届くのに、内側の指はまだ 1.43 cm 手前にある。**片側だけで押すので
        掴めず、丼を内側へ押しのける**——実測で 1.6 cm 押して掴み損ねた
        （50 配置のうち 16 本がこれで、接触は 0 のままゴールに届かなかった）。

        **「いちばん外の輪」の中心を返す。** 外周（中心＋半径）がいちばん
        大きい geom を選び、その**中心までの距離**を返す。
        """
        sim = self.env.env.sim
        centre = self.object_position(name)
        best: tuple[float, float] | None = None
        for index in range(sim.model.ngeom):
            if not sim.model.geom_contype[index]:
                continue
            if not (sim.model.geom_id2name(index) or "").startswith(name):
                continue
            position = sim.data.geom_xpos[index]
            half = float(sim.model.geom_size[index][2])
            if not (position[2] - half <= height <= position[2] + half):
                continue
            distance = float(np.linalg.norm(position[:2] - centre[:2]))
            width = float(max(sim.model.geom_size[index][0],
                              sim.model.geom_size[index][1]))
            if best is None or distance + width > best[0]:
                best = (distance + width, distance)
        if best is None:
            raise RuntimeError(f"{name} に高さ {height:.4f} を覆う当たり判定が無い")
        return best[1]

    def choose_grasp_azimuth(self, target: str, radius: float,
                             z_low: float, z_high: float
                             ) -> tuple[float, float]:
        """縁をまたぐ方位を選ぶ。**回転対称な物にだけ使える。**

        **選べる理由。**丼は Z 軸まわりに回転対称なので、どの方位から縁を
        またいでも同じように掴める。**だから方位は自由変数である。**

        **選ばないと当たる理由。**手首は物体の軸から `radius` 外へ出て、
        **外側の指はさらに 2.4 cm 外へ張り出す。**隣の物との隙間が 3.42 cm
        しかない場面では、隣の側からまたぐと余裕が数 mm になる。

        **測る余裕は 2 種類に分ける。**
        採点の衝突判定は「観測に `_pos` を持ち、`obj_of_interest` でない物体」
        だけを見る（`pipeline/rollout.py`）。**fixture は判定の対象外**なので、
        こちらを最大化するのは筋が悪い。**判定される物体との余裕を最大化し、
        fixture は「`FIXTURE_MARGIN` 以上離れていること」という条件にする。**

        **同じだけ空いているなら、ロボットの基部に近い方位を選ぶ。**縁をまたぐと
        **置くときの手首の位置も同じ向きへ外れる**ので、遠い方位は掴む側と置く側の
        両方を遠くする。回す量は最後の同点崩しにしか使わない。

        **余裕は頭打ちにする。**`GRASP_CLEARANCE_ENOUGH` を超えたぶんは
        同点として扱い、**ロボットの基部に近い方位**を選ぶ。そうしないと
        腕が遠回りして歩数が伸び、時間の点を落とす（実測 151 → 273 歩）。

        Returns
        -------
        tuple[float, float]
            選んだ方位 [度] と、そのときの**判定される物体との余裕** [m]。
        """
        sim = self.env.env.sim
        centre = self.object_position(target)
        watched = tuple(key[:-4] for key in self._watched_keys)
        ignore = ("robot0", "gripper0", target, "table", "floor", "wall") + watched
        base = np.asarray(sim.data.body_xpos[sim.model.body_name2id("robot0_base")]) \
            if "robot0_base" in sim.model.body_names else np.zeros(3)
        toward_base = float(np.degrees(np.arctan2(*(base[:2] - centre[:2])[::-1])))
        now = self.finger_azimuth

        best: tuple[float, float, float, float] | None = None
        for degree in range(0, 360, 5):
            angle = np.radians(float(degree))
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            # 手首・外側の指・内側の指。**外側の指がいちばん外へ張り出す。**
            points = [centre + offset * direction
                      for offset in (radius, radius + 0.5 * FINGER_SPAN,
                                     radius - 0.5 * FINGER_SPAN)]
            scored = min(_horizontal_clearance(sim, point, z_low, z_high,
                                               keep=watched)
                         for point in points)
            allowed = min(_horizontal_clearance(sim, point, z_low, z_high,
                                                skip=ignore)
                          for point in points)
            gap = abs((degree - toward_base + 180.0) % 360.0 - 180.0)
            # **手首を何度回すことになるか。**指は 2 本で対称なので ±90 度に折り返す。
            turn = abs((degree - now + 90.0) % 180.0 - 90.0)
            # 条件を満たすものを優先し、次に余裕、次に**基部へ近い側**、
            # 最後に手首を回さずに済む側。
            # **基部へ近い側が効くのは、掴む側と置く側の両方を決めるからである。**
            # 縁をまたぐと掴んだ物は手首から `radius` 外れるので、**置くときの
            # 手首の位置も皿の中心から同じ向きへ `radius` 外れる。**手首の回転を
            # 優先して遠い方位を選ぶと、置く位置が腕の届く範囲の端へ出て
            # ゴールに届かなくなった（実測 46/50 → 2/8）。
            # **余裕は `GRASP_CLEARANCE_ENOUGH` で頭打ちにする**——それ以上は
            # 当たらないことに変わりが無く、遠い方位を選ぶと歩数だけ増える。
            # **手首を回す速さは封筒で 1 歩 1.6 度に制限される**ので、90 度回すと
            # それだけで 53 歩かかる。しかもその間は腕が止まり、**指令が 0 の
            # 1 点に積み上がる**（reference/33）。回さずに済むなら回さない。
            key = (1.0 if allowed >= FIXTURE_MARGIN else 0.0,
                   min(scored, GRASP_CLEARANCE_ENOUGH), -gap, -turn)
            if best is None or key > best[0:4]:
                best = key + (float(degree),)      # type: ignore[assignment]
        assert best is not None
        return float(best[4]), float(best[1])      # type: ignore[index]

    def turn_wrist(self, label: str, azimuth: float, gripper: float,
                   budget: int = 45, tolerance: float = 4.0) -> None:
        """指の並ぶ向きだけを回す。**腕は動かさない。**

        `action[5]` は向きの差分指令で、**1 歩あたり 3.1 度しか回らない**
        （実測。`probe/gripper_frame.py`）。だから**降りる前に回しきる**——
        降りながら回すと、回り切る前に指が隣の物へ当たる。

        **指は 2 本で対称なので、誤差は ±90 度に折り返す。** 折り返さないと
        180 度回すのに 58 歩使って、同じ形の掴みにしかならない。
        """
        used = self.record.steps
        for _ in range(budget):
            error = (azimuth - self.finger_azimuth + 90.0) % 180.0 - 90.0
            if abs(error) < tolerance:
                break
            action = np.zeros(7)
            action[5] = float(np.clip(error / 12.0, -1.0, 1.0))
            action[6] = gripper
            if not self._take_step(action):
                break
        left = abs((azimuth - self.finger_azimuth + 90.0) % 180.0 - 90.0)
        # **`phases` の 3 つめは m で表示される**ので、度をそのまま入れると
        # 1.6 度が「163 cm」と読める。残りは名前のほうへ書く。
        self.phases.append((f"{label}・残り {left:.1f} 度",
                            self.record.steps - used, 0.0))

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

        **`cruise` はその上限そのものを下げる。**振り切ったままにすると指令が
        上限の 1 点に積み上がり、配布データと分布が食い違う（`TaskPlan.cruise`）。
        """
        distance = float(np.linalg.norm(delta))
        largest = float(np.max(np.abs(delta)))
        if distance < 1e-9 or largest < 1e-9:
            return np.zeros(3)
        return delta / largest * self.cruise * min(1.0, distance / full_speed_within)

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
        こうすると scripted policy は「ずれた場所から目標へ戻す」行動を自分で作るので、
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
           向きは scripted policy が閉ループで直していない（`turn_arc` が固定値を入れるだけ）
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

    def _disturbed(self, action: np.ndarray) -> np.ndarray:
        """決めた区間だけ、送る行動を大きく横へ押しのける。**記録は押す前である。**

        `_noisy` は毎ステップ独立に小さく揺らすので、**連続した微小なずれ**になる。
        こちらは**1 本につき一度だけ、数歩まとめて大きく押す**ので、
        「途中で一度失敗し、そこから戻る」形の手本になる。scripted policy は閉ループなので、
        押しのけられた先から目標へ戻す行動を自分で作る（`_noisy` と同じ仕組み）。

        **決めたことが 4 つある。どれも変えると別物になる。**

        1. **足すのではなく上書きする。** `_velocity_command` はいちばん大きい軸を
           ±1 に揃えるので、**足すと外へ出るぶんが切られて狙った大きさが届かない**
           （`_noisy` の docstring に書いた性質）。上書きなら確実にずれる。
        2. **向きは 1 本につき 1 つ引き、妨害中は固定する。** 毎ステップ引き直すと
           行きと戻りが打ち消し合って正味の変位が小さくなり、**「一度失敗」ではなく
           「揺れ続ける」**になる。
        3. **指の開閉（`action[6]`）と手首の向き（`action[3:6]`）には触らない。**
           `_noisy` と同じ理由である ── 指に触ると物を落とすだけで復帰は学べず、
           向きは scripted policy が閉ループで直していないので誤った正解になる。
        4. **開始する歩は本ごとに範囲から引く。** ご要望の「異なるタイミングで
           一度失敗」を、1 つのコマンドで本数ぶん作るためである。段の途中で押すか
           境目で押すかによって、崩れ方の種類が変わる。

        **戻れなかった本は `record` が既定で捨てる。** 捨てた本の割合が上がったら
        `--disturb-scale` か `--disturb-steps` が大きすぎる合図である。
        """
        if self.disturb_scale <= 0.0 or self.disturb_start < 0:
            return action
        step = self.record.steps
        if not (self.disturb_start <= step
                < self.disturb_start + self.disturb_steps):
            return action
        pushed = action.copy()
        pushed[:3] = np.clip(self.disturb_scale * self.disturb_direction,
                             -1.0, 1.0)   # 環境が受け取る範囲
        return pushed

    def _enveloped(self, action: np.ndarray) -> np.ndarray:
        """指令を配布データの封筒へ収める。**大きさと、1 コマの変化量の両方。**

        **大きさだけ切っても足りない。**上限ちょうどの 1 点に積み上がった
        教師信号は学習で解けない（[33](../../../docs/reference/33_gradient_norm_and_action_saturation.md)）。
        1 コマの変化量まで抑えると、上限に触れる前に立ち上がりが鈍るので、
        端へ張り付く割合そのものが下がる。

        **0 へ落ちるときも同じ速さで落とす。** 指令が急に 0 になる点は、
        勾配の急変という意味で上限の 1 点と変わらない（reference/35）。
        """
        limits = np.concatenate([
            np.full(3, self._envelope.limit_pos),
            np.full(3, self._envelope.limit_rot)])
        rates = np.concatenate([
            np.full(3, self._envelope.accel_pos),
            np.full(3, self._envelope.accel_rot)])
        wanted = np.clip(np.asarray(action[:6], dtype=np.float64), -limits, limits)
        change = np.clip(wanted - self._last_sent, -rates, rates)
        self._last_sent = self._last_sent + change
        shaped = action.copy()
        shaped[:6] = self._last_sent
        return shaped

    def _take_step(self, action: np.ndarray) -> bool:
        """1 ステップ進める。ステップ数の上限に達したら False を返す。"""
        if self._envelope is not None:
            action = self._enveloped(action)
        index = self.record.steps
        self.record.ee_positions.append(self.eef_position.copy())
        self.record.states.append(self._observation_state())
        self.record.actions.append(action.astype(np.float32))
        # **観測とノイズ無しの行動を記録してから、混ぜたほうを送る。**
        action = self._disturbed(self._noisy(action))
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
            each = np.abs(current - self._watched_origin).sum(axis=1)
            moved = float(each.max())
            self.record.max_displacement = max(self.record.max_displacement,
                                               moved)
            for position, key in enumerate(self._watched_keys):
                name = key[:-4]
                value = float(each[position])
                if value > self.record.object_displacements.get(name, 0.0):
                    self.record.object_displacements[name] = value
                if (self.record.first_violation is None
                        and value > self.record.collision_threshold_m):
                    self.record.first_violation = (name, self.record.steps)
        self.record.displacements.append(self.record.max_displacement)
        return self.record.steps < self.max_steps

    def attach_images(self) -> None:
        """溜めた画像を、ステップ数ぶんに切って結果へ渡す。**複製しない。**"""
        if self._front_buffer is None:
            return
        count = min(self.record.steps, self.max_steps)
        self.record.front = self._front_buffer[:count]
        self.record.wrist = self._wrist_buffer[:count]

    def move_to(self, label: str, target: np.ndarray | list[float],
                gripper: float, tolerance: float = 0.005, budget: int = 60,
                yaw_target: float | None = None) -> None:
        """1 点へ寄せて、そこで止まる。**位置を合わせたい所に使う。**

        `follow` に中継点 1 つを渡すのと同じである。制御の中身を 2 つ持つと
        必ず片方が古くなるので、こちらは呼び名だけを与える薄い包みにしてある。

        **ここは意図的になまさない**（`smoothing` を渡さない）。指令をなますと
        指令が小さくなる終盤で這い、3.8cm 降りるのに 25 ステップ使って届かなくなる。
        滑らかにするのは運ぶ区間だけでよい。
        """
        self.follow(label, [target], gripper, blend_radius=0.0,
                    tolerance=tolerance, budget=budget,
                    brake_distance=BRAKE_WITHIN_STOP, yaw_target=yaw_target)

    def follow(self, label: str, waypoints: list[Any], gripper: float,
               blend_radius: float, tolerance: float = 0.006,
               budget: int = 90, full_speed_within: float = BRAKE_WITHIN_STOP,
               smoothing: float = 1.0,
               brake_distance: float = BRAKE_WITHIN_PASS,
               speed_cap: float = 1.0, yaw_target: float | None = None) -> None:
        """中継点を**止まらずに**繋いで通る。**運ぶ区間に使う。**

        `blend_radius` まで近づいたら、そこへ着く前に次の中継点へ目標を
        移す。**角を丸めて通るので減速も方向転換もしない。** 最後の 1 点
        だけは `tolerance` まで詰める。

        Parameters
        ----------
        blend_radius : float
            途中の中継点をどれだけ手前で見切るか。**大きいほど滑らかだが、
            角が内側に膨らむ。** 障害物の際を通るときは小さくする。
        yaw_target : float | None
            指の並ぶ向きの目標 [度]。**渡すと、動きながら同時に手首を回す。**
            封筒に収めると手首は 1 歩 1.6 度しか回らないので、**移動が終わって
            から回すと 50 歩以上かかる。**移動と同時なら追加の歩数は要らない。
        speed_cap : float
            指令の上限。**1.0 が既定で、絞らない。**縁を挟んだ物を真上へ
            上げるときのように、**速く動かすと隣の物へ押し付ける区間だけ**
            絞る。
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
            action[:3] = speed_cap * self._smoothed(
                self._velocity_command(
                    delta, brake_distance if is_last else full_speed_within),
                smoothing)
            if yaw_target is not None:
                turn = (yaw_target - self.finger_azimuth + 90.0) % 180.0 - 90.0
                action[5] = float(np.clip(turn / 12.0, -1.0, 1.0))
            action[6] = gripper
            if not self._take_step(action):
                break
        error = float(np.linalg.norm(targets[-1] - self.eef_position))
        self.phases.append((label, self.record.steps - used, error))

    def joint_anchor(self, name: str) -> np.ndarray:
        """関節の回転の中心。**fixture は初期配置ごとに数 mm ずれる**ので実行時に読む。"""
        sim = self.env.env.sim
        return np.asarray(sim.data.xanchor[sim.model.joint_name2id(name)])

    def turn_arc(self, label: str, centre: np.ndarray, step: TurnKnob) -> None:
        """掴んだコンロの火力つまみを、円弧に沿って動かしながら手首も回す。

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

    def hold_at(self, label: str, gripper: float, count: int,
                yaw_target: float | None = None) -> None:
        """**いまの姿勢を閉ループで保ったまま**、指だけ動かす。

        **`hold` との違いは、指令が厳密な 0 にならないことである。**
        あちらは `np.zeros(7)` を送るので、指を閉じる区間ぶんだけ
        **指令が 0 の 1 点に積み上がる**——配布データの停止コマは 0.96% なのに、
        `hold` だけで 14.4% になった。1 点に集中した教師信号は flow matching が
        出せないので、上限の 1 点と同じ理由で学習が解けない
        （`docs/reference/33_gradient_norm_and_action_saturation.md`）。

        **測って外した。使わないこと。**指令の形は直る（停止コマ 14.40% →
        1.71%・停止の塊 3.4 → 2.2 個/本）が、**掴めなくなる**——指が縁を挟む
        あいだ、閉じる反力で腕が押されるのを毎歩打ち消しに行くので、
        **制御が振動して器を押しのける**（並進の符号反転が 0.38% → 2.30%）。
        50 配置で 246/250 だった成否が、10 配置で 2/10・3/10 に落ちた。
        **手を止める区間は本当に止めるしかない。**

        残した理由は 2 つある。**同じ案を次のセッションが思いつくため**と、
        **押し付けずに保つ場面（掴んでいない区間の整定）には使えるため**である。
        """
        used = self.record.steps
        target = self.eef_position.copy()
        for _ in range(count):
            action = np.zeros(7)
            action[:3] = self._velocity_command(target - self.eef_position,
                                                BRAKE_WITHIN_STOP)
            if yaw_target is not None:
                turn = (yaw_target - self.finger_azimuth + 90.0) % 180.0 - 90.0
                action[5] = float(np.clip(turn / 12.0, -1.0, 1.0))
            action[6] = gripper
            if not self._take_step(action):
                break
        self.phases.append((label, self.record.steps - used,
                            float(np.linalg.norm(target - self.eef_position))))

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
    # 直に運び先へ向かうと fixture にぶつけるので、まず持ち上げる点を経由する。
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


def _run_rim_pick_place(rollout: _Rollout, step: RimPickPlace,
                        plan: TaskPlan) -> None:
    """**縁をまたいで**掴み、**掴んだ物の中心**を置き先の中心へ合わせて置く。

    ```
    出発 ─▶ 選んだ方位の真上   ← 何にも当たらない高さ
            ↻ 手首を回す        ← **降りる前に回しきる**
            ▼ 真下へ降りる      ← 縁を 2 本の指で挟む
              掴む
            ▲ 真上へゆっくり    ← **横へ動かすと隣の物へ押し付ける**
          ──弧──▶ 置く手前
            ▼ 置く              ← **手首ではなく、掴んだ物の中心**を合わせる
              離す
            ▲ 逃げる
    ```

    **`PickPlace` と違うのは 3 か所だけである。**

    | 違い | なぜ |
    |---|---|
    | 掴む位置を**方位から決める** | 物体が指より広く、縁をまたぐしかないため。**方位は自由変数** |
    | 掴んだあと**真上へゆっくり上げる** | 横へ動かすと隣の物へ押し付ける。実測で damage の 85〜88% が掴む区間 |
    | 置く位置を**掴んだ物の中心**で合わせる | 縁をまたぐと物の中心が手首から 5.65 cm 外れる。合わせないと皿の縁に落ちる |
    """
    target = step.target
    centre = rollout.object_position(target)
    # **縁の壁の中心を、掴む高さで測る。**張り出し（外周）ではない——外周に
    # 手首を置くと内側の指が壁に乗ってしまう（`rim_radius` の説明）。
    radius = (step.grasp_radius if step.grasp_radius is not None
              else rollout.rim_radius(target, centre[2] + step.grasp_dz))
    azimuth, clearance = rollout.choose_grasp_azimuth(
        target, radius,
        centre[2] + step.clearance_band[0], centre[2] + step.clearance_band[1])
    angle = np.radians(azimuth)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])

    wrist = centre + radius * direction
    # **寄りながら手首も回す。**封筒に収めると手首は 1 歩 1.6 度しか回らないので、
    # 寄り終わってから回すと 50 歩以上かかる。**同時なら追加の歩数は要らない。**
    rollout.follow(f"{target} 掴む所の上へ（手首も {azimuth:.0f} 度へ）",
                   [[wrist[0], wrist[1], centre[2] + step.approach_dz]],
                   GRIPPER_OPEN, blend_radius=0.0, tolerance=0.006,
                   budget=plan.budget_move, smoothing=plan.smoothing,
                   yaw_target=azimuth)
    # **手首を回すためだけに腕を止めない。**止まっている区間は指令が 0 の 1 点に
    # 積み上がり、上限の 1 点と同じ理由で学習が解けない（reference/33・35）。
    # **大きく残っているときだけ**その場で詰め、あとは降りながら回す。
    # **寄っている間に物体が動いていることがある**ので位置を取り直す。
    centre = rollout.object_position(target)
    wrist = centre + radius * direction
    # **降りるのを 2 段に割る。**上の段は回しながら降り、下の段は真下だけにする
    # ——指が縁に近づいてから回すと、回り切る前に隣の物へ当たる。
    rollout.follow(f"{target} 構える（手首も {azimuth:.0f} 度へ）",
                   [[wrist[0], wrist[1], centre[2] + step.hover_dz]],
                   GRIPPER_OPEN, blend_radius=0.0, tolerance=0.006,
                   budget=plan.budget_descend, yaw_target=azimuth)
    left = abs((azimuth - rollout.finger_azimuth + 90.0) % 180.0 - 90.0)
    if left > 6.0:
        rollout.turn_wrist(f"{target} 手首の残りを詰める", azimuth,
                           GRIPPER_OPEN, budget=30, tolerance=5.0)
    rollout.follow(f"{target} 縁を挟む位置へ（余裕 {clearance * 100:.1f} cm）",
                   [[wrist[0], wrist[1], centre[2] + step.grasp_dz]],
                   GRIPPER_OPEN, blend_radius=0.0, tolerance=0.004,
                   budget=int(round(plan.budget_descend / step.descend_speed)),
                   brake_distance=BRAKE_WITHIN_STOP,
                   speed_cap=step.descend_speed)
    # **ここは `hold_at` を使わない。**閉ループで保つと閉じる反力に対して
    # 振動し、器を押しのけて掴めなくなる（`hold_at` の説明）。
    rollout.hold(f"{target} 掴む", GRIPPER_CLOSE, plan.close_steps)

    here = rollout.eef_position.copy()
    # **指令を絞るぶん、同じ距離に歩数が要る。**絞ったまま同じ budget で
    # 走らせると上がりきらず、そのあとの運搬が低い所を通る（実測で
    # 10 cm 上げるつもりが 4.5 cm しか上がらなかった）。
    rollout.follow(f"{target} 真上へ上げる",
                   [[here[0], here[1], here[2] + step.lift_dz]],
                   GRIPPER_CLOSE, blend_radius=0.0, tolerance=0.006,
                   budget=int(round(plan.budget_lift / step.lift_speed)),
                   speed_cap=step.lift_speed, yaw_target=azimuth)

    # **手首ではなく、掴んだ物の中心を置き先へ合わせる。**
    base = (rollout.site_position(step.place_site)
            if step.place_site is not None
            else rollout.object_position(step.place_object))
    held = rollout.object_position(target)[:2] - rollout.eef_position[:2]
    place_xy = base[:2] + np.asarray(step.place_offset) - held

    rollout.follow(f"{target} 運ぶ",
                   [[here[0], here[1], plan.carry_z],
                    [place_xy[0], place_xy[1], plan.carry_z]],
                   GRIPPER_CLOSE, blend_radius=plan.blend_radius,
                   budget=plan.budget_lift + plan.budget_move,
                   smoothing=plan.smoothing, yaw_target=azimuth)
    rollout.move_to(f"{target} 置く", [place_xy[0], place_xy[1], plan.place_z],
                    GRIPPER_CLOSE, tolerance=0.004, budget=plan.budget_descend,
                    yaw_target=azimuth)
    rollout.hold(f"{target} 落ち着かせる", GRIPPER_CLOSE, plan.settle_steps)
    rollout.hold(f"{target} 離す", GRIPPER_OPEN, plan.open_steps)
    rollout.follow(f"{target} 逃げる",
                   [[place_xy[0], place_xy[1], plan.retreat_z]],
                   GRIPPER_OPEN, blend_radius=plan.blend_radius,
                   tolerance=0.01, budget=plan.budget_lift,
                   smoothing=plan.smoothing)


def _run_turn_knob(rollout: _Rollout, step: TurnKnob, plan: TaskPlan) -> None:
    """コンロの火力つまみを掴んで回し、離して逃げる。

    ```
    出発 ─▶ コンロの火力つまみの上   ← 構える
           ▶ 掴む高さ     ← ここだけ正確に止まる
             掴む
           ⤿ 円弧に沿って動かしながら手首を回す   ← ここで点く
             離す
           ▲ 真上へ逃げる
    ```
    """
    centre = rollout.joint_anchor(step.joint)
    grip_at = np.array([centre[0] + step.radius, centre[1], step.height])

    rollout.move_to("コンロの火力つまみへ構える", grip_at + [0.0, 0.0, step.hover_dz],
                    GRIPPER_OPEN, tolerance=0.006, budget=plan.budget_move)
    rollout.move_to("コンロの火力つまみを掴む位置へ", grip_at, GRIPPER_OPEN,
                    tolerance=0.004, budget=plan.budget_descend)
    rollout.hold("コンロの火力つまみを掴む", GRIPPER_CLOSE, plan.close_steps)
    rollout.turn_arc("コンロの火力つまみを回す", centre, step)
    rollout.hold("コンロの火力つまみを離す", GRIPPER_OPEN, plan.open_steps)
    rollout.move_to("コンロの火力つまみから逃げる",
                    [rollout.eef_position[0], rollout.eef_position[1],
                     plan.retreat_z],
                    GRIPPER_OPEN, tolerance=0.01, budget=plan.budget_lift)


# **動きの種類ごとに実行の中身を引く。** 種類が増えても `run_episode` は変わらない。
_RUNNERS: dict[type, Any] = {PickPlace: _run_pick_place,
                             RimPickPlace: _run_rim_pick_place,
                             TurnKnob: _run_turn_knob}


def _envelope_for(plan: TaskPlan) -> Any:
    """計画が求めるなら、配布データの封筒を読んで返す。

    **import はここで行う。**`motion` は別の入り口（`solution/tools/motion`）に
    あり、`PYTHONPATH` に入っていない場に置かれることがある。**封筒を使わない
    計画まで import で落とすと、moka pot の 2 課題が動かなくなる。**
    """
    if not plan.command_envelope:
        return None
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from motion.envelope import envelope_from_environment

    return envelope_from_environment()


def run_episode(env: Any, init_state: np.ndarray, plan: TaskPlan,
                max_steps: int = 300, record_images: bool = False,
                verbose: bool = False, action_noise: float = 0.0,
                noise_seed: int | None = None, disturb_scale: float = 0.0,
                disturb_steps: int = 10,
                disturb_at: tuple[int, int] | None = None) -> EpisodeRecord:
    """scripted policy を 1 本ぶん実行し、採点と同じ基準で成否を付ける。

    Parameters
    ----------
    env : Any
        `libero.libero.envs.env_wrapper.ControlEnv`。
    init_state : np.ndarray
        `.pruned_init` から取り出した 1 本ぶんの初期状態。
    plan : TaskPlan
        実行する scripted policy。
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
    disturb_scale : float
        **1 本につき一度だけ**、送る行動の手先の位置を押しのける大きさ。
        0.0 なら妨害しない。`action_noise` とは別物で、あちらは毎ステップ
        独立に小さく揺らす（`_Rollout._disturbed` に決めたことを書いた）。
    disturb_steps : int
        妨害を続ける歩数。
    disturb_at : tuple[int, int] | None
        妨害を始める歩の範囲。**この範囲から本ごとに 1 つ引く**ので、
        本数ぶん違うタイミングで失敗する手本になる。

    Returns
    -------
    record : EpisodeRecord
        ステップ数・成否・行動。画像は `rollout.images()` から取る。
    """
    rollout = _Rollout(env, init_state, max_steps, record_images,
                       action_noise=action_noise,
                       rng=np.random.default_rng(noise_seed),
                       disturb_scale=disturb_scale,
                       disturb_steps=disturb_steps,
                       disturb_at=disturb_at,
                       envelope=_envelope_for(plan), cruise=plan.cruise)
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


def _init_states(libero_root: Path, suite: str, task: str,
                 from_demos: Path | None = None) -> np.ndarray:
    """初期状態の一覧。**variant ではなく元タスクの名前で引く**（1 タスク 1 file）。

    `from_demos` を渡すと、**人の手本が持っている置き場所**を使う
    （`work/artifacts/libero_states/<suite>/<task>_demo.npz` の `states[0]`）。

    **なぜ 2 通り要るか。**`.pruned_init` は採点器が使う 50 通りで、
    **そこで通ることが成否の基準**である。一方、配布データの置き場所は
    1 タスク 50 本ぶんあり、`.pruned_init` とは別の並びである
    （2,000 デモで 1,681 通り。[reference/25 の 8.9]）。
    **学習データを作るときは両方から作るほうが、置き場所の散らばりが増える。**
    """
    if from_demos is not None:
        with np.load(from_demos / suite / f"{task}_demo.npz",
                     allow_pickle=False) as bundle:
            count = len({key.split("/")[0] for key in bundle.files})
            return np.stack([bundle[f"demo_{index}/states"][0]
                             for index in range(count)])
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
    """scripted policy を引く。滑らかさの設定値だけ、その場で差し替えられる。

    **`measure` は scripted policy より先に走る。** 寸法を測ってから scripted policy を書くのに、
    scripted policy が無いと測れないのでは順序が逆になる。`--suite` を渡した場合だけ、
    **空の scripted policy**を作って場面を開けるようにする（`measure` 専用）。
    """
    if task not in TASK_PLANS:
        if suite is not None:
            return TaskPlan(suite=suite, steps=[], carry_z=0.0,
                            place_z=0.0, retreat_z=0.0)
        raise SystemExit(
            f"scripted policy が無い: {task}\n  ある scripted policy: {', '.join(sorted(TASK_PLANS))}\n"
            "  **測るだけなら `--suite libero_10` を渡す**（scripted policy より先に測る）")
    plan = TASK_PLANS[task]
    if smoothing is not None:
        plan = replace(plan, smoothing=smoothing)
    if blend_radius is not None:
        plan = replace(plan, blend_radius=blend_radius)
    return plan


def _horizontal_clearance(sim: Any, point: np.ndarray, z_low: float,
                          z_high: float, keep: tuple[str, ...] = (),
                          skip: tuple[str, ...] = ()) -> float:
    """点から、当たり判定のある geom の面までの**水平距離の最小値** [m]。

    **箱を円で近似している。**方位を選ぶだけなら十分である——うまく選べば
    余裕は数 mm と数 cm の違いになるので、mm の精度は要らない。

    `keep` を渡すとその名前で始まる geom **だけ**を見る。渡さないときは
    `skip` で始まるものを除いて全部見る。**採点で判定される物体と、
    判定されない fixture を別々に測るために 2 通り要る。**
    """
    best = float("inf")
    for index in range(sim.model.ngeom):
        if not sim.model.geom_contype[index]:
            continue
        name = sim.model.geom_id2name(index) or ""
        if not name:
            continue
        if keep:
            if not name.startswith(keep):
                continue
        elif name.startswith(skip):
            continue
        position = sim.data.geom_xpos[index]
        half = float(sim.model.geom_size[index][2])
        if position[2] + half < z_low or position[2] - half > z_high:
            continue
        radius = float(max(sim.model.geom_size[index][0],
                           sim.model.geom_size[index][1]))
        best = min(best, float(np.linalg.norm(position[:2] - point[:2])) - radius)
    return best


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
    """物体と fixture の当たり判定。**運ぶ高さと掴む高さを決める値。**"""
    print("\n--- 物体・fixture の当たり判定 ---")
    print("  **運ぶ高さは、fixture の上端＋物体の背丈より高く取る。**")
    geoms = _collidable_geoms(sim)
    for name, body_id in env.env.obj_body_id.items():
        # fixture は子の body に geom がぶら下がるので、**body ではなく名前で拾う。**
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
    """関節を持つ fixture（コンロの火力つまみ・引き出し）の状態と、点く境目を出す。

    **「点ける」は関節の角度で判定される。** `flat_stove` なら
    `qpos >= 0.5`（`libero/libero/envs/objects/articulated_objects.py` の
    `default_turnon_ranges` の下限）で、**どの関節でもよい**（`turn_on` は
    どれか 1 つが超えれば真を返す）。**見た目ではなくこの値を狙う。**
    """
    print("\n--- 関節（コンロの火力つまみ・扉）---")
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
    """3 つの口に共通の下ごしらえ。scripted policy・variant 名・環境・初期状態を揃える。

    **`--variant` 省略時は元タスクそのもの**という決まりを 1 か所に置く。
    """
    plan = _resolve_plan(args.task, args.smoothing, args.blend_radius,
                         getattr(args, "suite", None))
    variant = args.variant or args.task
    env = _make_env(_bddl_path(args.libero_root, plan.suite, variant),
                    with_images)
    return plan, variant, env, _init_states(args.libero_root, plan.suite,
                                            args.task,
                                            getattr(args, "init_from_demos", None))


def command_measure(args: argparse.Namespace) -> None:
    """scripted policy を書く前に、場面の寸法を測って印字する。

    **推測で scripted policy を書かない。** 指の開き・物体の幅・fixture の上端は
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


def _report_disturbance(env: Any, plan: TaskPlan, init_states: np.ndarray,
                        args: argparse.Namespace) -> None:
    """妨害が軌道をどれだけ外したかを、同じ本の対で測る。

    **成功率では妨害の効き具合が読めない。** 10 本では 1〜3 本の違いが
    偶然と区別できず、実際に 10 歩と 30 歩で成功率・失敗した本の番号まで
    一致した（2026-08-23）。**そこで同じ初期配置・同じ種で「妨害あり」と
    「妨害なし」を 2 回走らせ、同じ歩どうしの手先の距離を直接測る。**

    ここで見るのは 3 つである。

    | 見るもの | 意味 |
    |---|---|
    | 最大のずれ | 妨害でどこまで軌道から外れたか。**これが小さいなら妨害は効いていない** |
    | 最後のずれ | **互いの最終歩どうし**で比べた距離。戻れていれば 0 に近い |
    | 遅れ | 復帰に使った歩数。**打ち切り 300 への余裕を食う分**である |

    **「同じ歩どうし」で最後を比べてはいけない**（2026-08-23 に一度やって誤った）。
    妨害された本は復帰に歩数を使うので遅れる。短いほうの最終歩で比べると、
    **妨害ありのほうはまだ手前の段にいる**ので、戻れていても大きな差に見える。
    """
    print(f"  **妨害の効き具合を測る**（1 本を 2 回走らせるので時間は 2 倍）")
    rows: list[tuple[int, int, float, float, bool, bool]] = []
    for index, state in enumerate(init_states):
        with_hit = run_episode(env, state, plan, args.max_steps,
                               action_noise=args.action_noise,
                               noise_seed=args.noise_seed + index,
                               disturb_scale=args.disturb_scale,
                               disturb_steps=args.disturb_steps,
                               disturb_at=args.disturb_at)
        without = run_episode(env, state, plan, args.max_steps,
                              action_noise=args.action_noise,
                              noise_seed=args.noise_seed + index,
                              disturb_scale=0.0)
        a = np.asarray(with_hit.ee_positions)
        b = np.asarray(without.ee_positions)
        shared = min(len(a), len(b))
        gap = np.linalg.norm(a[:shared] - b[:shared], axis=1)
        # **最後は互いの最終歩どうしで比べる。** 同じ歩どうしだと、遅れている
        # ぶんが「戻れていない」に見える（この docstring に書いた誤り）。
        ending = float(np.linalg.norm(a[-1] - b[-1]))
        rows.append((index, with_hit.disturb_start, float(gap.max()),
                     ending, with_hit.steps - without.steps,
                     with_hit.success, without.success))
    print(f"  {'本':>3} {'妨害開始':>8} {'最大のずれ':>10} {'最後のずれ':>10}"
          f" {'遅れ':>6} {'妨害あり':>8} {'妨害なし':>8}")
    for index, start, peak, ending, delay, ok, base_ok in rows:
        print(f"  {index:3d} {start:8d} {peak * 100:9.1f}cm {ending * 100:9.1f}cm"
              f" {delay:+5d}歩 {'成功' if ok else '**失敗**':>8}"
              f" {'成功' if base_ok else '失敗':>8}")
    peaks = np.array([r[2] for r in rows])
    endings = np.array([r[3] for r in rows])
    delays = np.array([r[4] for r in rows])
    kept = sum(1 for r in rows if r[5])
    print(f"  **最大のずれ 中央 {np.median(peaks) * 100:.1f}cm "
          f"（最小 {peaks.min() * 100:.1f} 最大 {peaks.max() * 100:.1f}）**")
    print(f"  **最後のずれ 中央 {np.median(endings) * 100:.1f}cm** "
          f"（最小 {endings.min() * 100:.1f} 最大 {endings.max() * 100:.1f}）"
          "  ← 0 に近ければ戻れている")
    print(f"  遅れ 中央 {int(np.median(delays)):+d} 歩"
          f"（最小 {delays.min():+d} 最大 {delays.max():+d}）")
    print(f"  **残る本 {kept}/{len(rows)}**"
          f"（エピソードの作成は 1 本 95 秒。捨てる割合がそのまま無駄になる）")


def command_verify(args: argparse.Namespace) -> None:
    """初期配置すべてで scripted policy が通るかを数える。**描画しないので速い。**"""
    plan, variant, env, init_states = _setup(args, with_images=False)
    if args.episodes:
        init_states = init_states[args.episodes]

    print(f"=== {variant} — {len(init_states)} 通りの初期配置 ===")
    if args.action_noise > 0.0:
        print(f"  行動ノイズ σ={args.action_noise}（送る行動だけに混ぜる）")
    if args.disturb_scale > 0.0 and args.disturb_at is not None:
        print(f"  妨害 大きさ={args.disturb_scale} × {args.disturb_steps} 歩"
              f"（開始 {args.disturb_at[0]}〜{args.disturb_at[1]} 歩）")
    if args.disturb_report and args.disturb_scale > 0.0:
        _report_disturbance(env, plan, init_states, args)
        return
    started = time.time()
    lengths: list[int] = []
    failures: list[tuple[int, int, bool]] = []
    measured: list[dict[str, float]] = []
    saved_actions: list[np.ndarray] = []
    culprits: dict[str, list[tuple[int, int, float]]] = {}
    for index, state in enumerate(init_states):
        record = run_episode(env, state, plan, args.max_steps,
                             verbose=args.verbose and index == 0,
                             action_noise=args.action_noise,
                             noise_seed=args.noise_seed + index,
                             disturb_scale=args.disturb_scale,
                             disturb_steps=args.disturb_steps,
                             disturb_at=args.disturb_at)
        lengths.append(record.steps)
        if args.save_actions:
            saved_actions.append(np.asarray(record.actions, dtype=np.float32))
        if not record.success:
            failures.append((index, record.steps, record.collided))
            if record.collided and record.first_violation is not None:
                name, when = record.first_violation
                culprits[name] = culprits.get(name, [])
                culprits[name].append((when, record.steps,
                                       record.object_displacements[name]))
        else:
            measured.append(smoothness(record.ee_positions))

    if args.save_actions:
        # **指令の形を測るために、行動だけを書き出す。**画像は要らないので
        # `record` と違って描画しない（`motion/report_smoothness.py` の入力）。
        args.save_actions.mkdir(parents=True, exist_ok=True)
        for index, actions in enumerate(saved_actions):
            np.savez_compressed(args.save_actions / f"{variant}_ep{index:03d}.npz",
                                action=actions)
        print(f"  行動を {len(saved_actions)} 本 {args.save_actions} へ書いた")

    succeeded = len(lengths) - len(failures)
    print(f"成功 {succeeded}/{len(lengths)}  "
          f"ステップ数 中央値 {int(statistics.median(lengths))} "
          f"最小 {min(lengths)} 最大 {max(lengths)}  "
          f"（上限 {args.max_steps}）")
    if failures:
        print(f"  失敗（番号, ステップ数, 衝突）: {failures[:20]}")
    if culprits:
        print("  **何をいつ動かしたか**（1 mm を超えた最初の歩／全体の歩数）")
        for name, hits in sorted(culprits.items(), key=lambda x: -len(x[1])):
            share = statistics.median(h[0] / max(h[1], 1) for h in hits)
            print(f"    {name:<32}{len(hits):>3} 本  "
                  f"最初の歩 中央 {int(statistics.median(h[0] for h in hits)):>3}"
                  f"（本の {100 * share:.0f}% 地点）  "
                  f"変位 中央 {1000 * statistics.median(h[2] for h in hits):.1f} mm")
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
      - <output>/<variant>_ep<n>.npz（`--action-noise` を付けると `_n<σ>` が入る）
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
    if args.disturb_scale > 0.0 and args.disturb_at is not None:
        print(f"  妨害 大きさ={args.disturb_scale} × {args.disturb_steps} 歩、"
              f"開始は {args.disturb_at[0]}〜{args.disturb_at[1]} 歩から本ごとに引く。"
              "**書き出す行動は押す前である**")
    for index in indices:
        started = time.time()
        record = run_episode(env, init_states[index], plan, args.max_steps,
                             record_images=True,
                             action_noise=args.action_noise,
                             noise_seed=args.noise_seed + index,
                             disturb_scale=args.disturb_scale,
                             disturb_steps=args.disturb_steps,
                             disturb_at=args.disturb_at)
        if not record.success and not args.keep_failures:
            print(f"  ep{index:02d} 失敗（{record.steps} 歩）—— 書き出さない")
            continue
        # **ノイズの大きさを名前に入れる。** 入れないと、σ を変えて作った組が
        # 同じ名前になって上書きし合う（同じ variant・同じ番号で作るため）。
        tag = "" if args.action_noise <= 0.0 else f"_n{args.action_noise:g}"
        # **初期状態の出どころも名前に入れる。**`.pruned_init` の 50 通りと
        # 手本の 50 通りは**番号が独立**なので、入れないと同じ名前で上書きし合う。
        if getattr(args, "init_from_demos", None) is not None:
            tag += "_di"
        # **妨害の大きさも名前に入れる。** 妨害あり／なしを同じ置き場に
        # 混ぜて作るので、入れないと同じ名前になって上書きし合う。
        if args.disturb_scale > 0.0:
            tag += f"_d{args.disturb_scale:g}"
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
            # **何を・いつ・どちらへ妨害したかを残す。** 残さないと、
            # 後から「妨害あり／なし」で選び直すことも、混合比を測ることも
            # できない（08_training_data_creation_cautions.md 2.11 節）。
            disturb_scale=record.disturb_scale,
            disturb_start=record.disturb_start,
            disturb_steps=record.disturb_steps,
            disturb_direction=record.disturb_direction,
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
                        help="実際に使う variant の bddl 名。省略時は元タスクそのもの")
    parser.add_argument("--suite", default=None,
                        help="**scripted policy を書く前に measure だけ回すときに渡す**"
                             "（例 libero_10）。scripted policy があるときは要らない")
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
                        help="指令をなます強さ（1.0 でなまさない）。scripted policy の値を上書きする")
    parser.add_argument("--action-noise", type=float, default=0.0,
                        help="送る行動の手先の位置に混ぜる正規ノイズの標準偏差"
                             "（行動の範囲は ±1）。**記録する行動は混ぜる前の"
                             "ままなので、ずれてから戻す手本になる**（DART）。"
                             "0.0 で今までと同じ。**大きすぎると成功率が落ちて"
                             "逆効果になる**ので、先に verify で振って決めること")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="ノイズの種の基準値。本ごとに番号を足して使う")
    parser.add_argument("--disturb-scale", type=float, default=0.0,
                        help="**1 本につき一度だけ**手先を押しのける大きさ"
                             "（行動の範囲は ±1）。`--action-noise` と違い、"
                             "毎ステップではなく**数歩まとめて大きく**押すので、"
                             "「途中で一度失敗して戻る」手本になる。"
                             "0.0 で妨害しない")
    parser.add_argument("--disturb-steps", type=int, default=10,
                        help="妨害を続ける歩数（既定 10）")
    parser.add_argument("--disturb-report", action="store_true",
                        help="**妨害が実際に軌道をどれだけ外したかを測る**"
                             "（verify のみ）。同じ本を妨害あり／なしで 2 回走らせ、"
                             "同じ歩どうしの手先の距離を比べる。**成功率では"
                             "妨害の効き具合が読めない**ので、大きさと歩数を"
                             "決めるときはこちらを見る。時間は 2 倍になる")
    parser.add_argument("--disturb-at", default=None,
                        help="妨害を始める歩。`N` で固定、`LO,HI` なら"
                             "**その範囲から本ごとに引く**（タイミングを"
                             "散らすにはこちら）。省略時は妨害しない")
    parser.add_argument("--blend-radius", type=float, default=None,
                        help="通過点を見切る距離 [m]。scripted policy の値を上書きする")
    parser.add_argument("--init-from-demos", type=Path, default=None, metavar="DIR",
                        help="初期状態を人の手本の置き場所から取る"
                             "（`work/artifacts/libero_states`）。**省くと "
                             "`.pruned_init` の 50 通り**（採点器と同じ）")
    parser.add_argument("--save-actions", type=Path, default=None, metavar="DIR",
                        help="verify の行動列を npz で書き出す（描画しない）。"
                             "**指令の形を `motion/report_smoothness.py` で測るため**")
    parser.add_argument("--verbose", action="store_true",
                        help="1 本目だけ段ごとのステップ数を出す")
    args = parser.parse_args()
    if args.episodes:
        args.episodes = [int(x) for x in args.episodes.split(",")]
    if args.disturb_at:
        bounds = [int(x) for x in str(args.disturb_at).split(",")]
        # `N` は「その歩に固定」＝範囲の幅が 0 と同じである
        args.disturb_at = (bounds[0], bounds[-1])
        if args.disturb_at[0] > args.disturb_at[1]:
            raise SystemExit("--disturb-at は LO,HI の順で渡すこと")
    else:
        args.disturb_at = None
    return args


def main() -> None:
    args = _parse_args()
    {"measure": command_measure,
     "verify": command_verify,
     "record": command_record}[args.command](args)


if __name__ == "__main__":
    main()
