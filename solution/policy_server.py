"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

# 提出物の版。**採点ログの先頭に出るので、スコアと構成を後から結び付けられる。**
# 以前は「どの zip がどのスコアだったか」をハッシュと記憶で辿っていた。
#
#   major … 重みが変わったとき（追加学習の反映など）
#   minor … 推論の挙動が変わったとき（既定値の変更、後処理の追加）
#   patch … 挙動が変わらない修正
#
# リーダーボードの履歴との対応:
#   0.1.0  素の重み・後処理なし（n_action_steps=50）      スコア 0.05832
#   0.2.0  n_action_steps=10                              スコア 0.096
#   0.3.0  n_action_steps=5 ＋ EMA 0.3                    スコア 0.1409
#   1.0.0  ＋ LoRA 追加学習 20,000 ステップ（推論の設定は 0.3.0 と同じ）
#          → 本番 0.016。**0.3.0 の 0.1409 から 8.8 倍の悪化**
#   2.0.0  重みを λ=0.25 のブレンド（素 75% ＋ 1.0.0 の学習分 25%）に差し替え。
#          推論の設定は 0.3.0 / 1.0.0 と同じままなので、**0.1409 との差は重みだけ**。
#          目的はスコアではなく、1.0.0 の悪化がどこまで薄めれば消えるかを測ること。
#          観測の受け口は 1.0.0 と同じ（front/wrist・state 8・empty_cameras=0）
#          だが、**受け口の違いは推論では打ち消される**ので、この違いが原因と取り違える恐れはない。
#          → docs/reference/10_observation_interface.md
#          → **本番 0.09507。** ただし本番は 8 本のみで、うち 4 本は素・1.0.0・
#            2.0.0 すべてで 300 ステップに張り付く。差は 1 本ぶんで、0.1409 との優劣は
#            判定できない。明確に違うのは 1.0.0（0.016）だけ。
#   3.0.0  重みを weak3x（苦手タスク 3 倍 ＋ 手本の 300 ステップ超を落とす学習）に差し替え。
#          推論の設定は 2.0.0 と同じなので、**2.0.0 との差は重みだけ**。
#          weak3x は 1.0.0 と同じ全学習（λ=1.0 相当）だが、手元 84 件で 1.0.0 の
#          滑らかさの劣化（ジャーク +13.6%・空回り 2 倍）が素の水準まで戻っている。
#          → これで本番崩壊の 2 説を切り分ける。0.14 付近なら「滑らかさが犯人・
#            もう直った」、0.016 付近なら「重みのずれ自体が毒（滑らかさは無関係）」。
#          → **本番 0.144（成功 3/8・係数 0.384）。** 素 0.1409 と同水準で H1 を裏づけ。
#            学習を含む（＝資格を満たす）重みが初めて素の水準に乗った。8/14 の土台。
#   3.1.0  weak3x のまま n_action_steps を 5 → 2 に下げる（重みは 3.0.0 と同じ）。
#          n の 50→10→5 が 0.058→0.096→0.1409 と単調改善だった延長。再計画が
#          2.5 倍の頻度になり、300 ステップで終われない 4 本のどれかを崩せる見込み。
#          推論のみの変更＝minor。時間は n=2 でも約 585 秒（上限の 16%）。
#   3.2.0  weak3x のまま samples を 1 → 3（n は 5 に戻す＝3.0.0 から 1 変数）。
#          3 本引いて姿勢 6 次元の medoid を選ぶ。外れ chunk を捨て、滑らかさの
#          係数（3.0.0 で 0.384）を上げる向き。成功本数は動きにくい。時間は約 700 秒。
#          → **未提出。** n=2（3.1.0）が n=5 に勝ったため、n=5 の上で samples を
#            測る価値が下がった。samples は勝った n の上で測り直す。
#   3.1.0  → **本番 0.166。** weak3x 0.144 を上回り、資格ありの最高。天井の 1 本
#            （ep3）を 300→202 に解いた。n の 50→10→5→2 が単調改善の続き。
#   3.3.0  weak3x のまま n_action_steps を 1（3.1.0 からもう 1 つ下げる。samples=1）。
#          → **本番 0.080。悪化した。n=2 が谷底と確定。**
#            早く終わった回が 4 → 3 本に減り、係数も落ちた（0.443 → 0.32 前後）。
#            **第 7 回 講義資料part1.pdf p.10 が機構を説明している** ——「k が短すぎる
#            （k=2）／チャンク境界のたびに分布から別のモードをサンプルしうる →
#            mode-jumping・jerky な動き」。n=1 は1 ステップごとが境界になる。
#   3.4.0  n を 2（勝った値）に戻し、samples を 1 → 3。**3.1.0 から 1 変数。**
#          EMA（出したあとにならす）は既に効いていたのに n=1 が崩れたので、
#          **「跳んだモードを引かない」側**から攻める。3 本引いて姿勢 6 次元の
#          medoid を選ぶので、外れたモードを掴んだ回を捨てられる。
#          時間は約 1,500 秒（環境 129 秒 ＋ 推論 911/2×3。上限 3,600 秒の 42%）。
#   3.4.0  → **本番 0.168。** 0.166（3.1.0）から 0.002 しか動かなかった。
#          **1 エピソードが 0.047 にあたるので、これは 0.04 本ぶん＝差が無い。**
#          しかも合計時間が 564 秒 → 1,523 秒（上限の 16% → 42%）になる。
#          **samples の軸を閉じ、既定を 1 に戻す。** 同時に「n=2 が谷底」も
#          取り消す（n=5 との差 0.022 は 0.5 本ぶんで測れていない）。
#          → docs/strategy/05_実測結果.md「本番 8 回分を並べ直した」
#   3.5.0  指示文の正規化を入れた。**3.1.0（0.166）から 1 変数**
#          （n=2・samples=1・EMA 0.3 は 3.1.0 と同じ。重みも weak3x で同じ）。
#          **評価が渡してくる文は、学習データの書き方と違う。** 配布された 4 タスク
#          （compe/t1/T1_TASKS.csv）が渡すのは
#          `Pick the akita black bowl in the top layer of the wooden cabinet …` で、
#          学習データの `task` 列は
#          `pick up the black bowl in the top drawer of the wooden cabinet …` である。
#          **本番の 2 カテゴリで渡ってくる文は 40 種しかない**（変種 7,280 件が
#          この 40 種を使い回す）。**そのうち 25 種は学習の書き方とずれている。**
#          学習データの 40 種のうち最も近いものへ寄せると、**40 種すべてを
#          正しく寄せ、誤りは 0 だった**（変種 7,280 件すべて。実測は
#          solution/tools/suite/check_instruction_normalization.py）。
#          運営 Q&A No.29 が「小文字化、"Pick the"→"pick up the" のような言い換えの
#          統一」を名指しで許容している。No.31 の埋め込み最近傍も許容だが、
#          **既定では使わない**（下の _USE_EMBEDDING）。
#   3.5.0  → **本番 0.181。過去最高。**重みを 1 ビットも変えずに、8 回すべてで
#          300 ステップに張り付いていた ep2 が 94 ステップ、ep7 が 155 ステップで終わるようになった
#          （早く終わった回 4 → 5 本）。運営 Q&A No.26 が「評価タスク集合・各
#          エピソードの初期状態・乱数シードは固定」と明言しているので、この比較は
#          同条件である。**8 本すべてが normalize=lexical**（exact は 0 件）＝
#          **本番の指示文は 8 本とも学習の書き方と違っていた。言い換えの軸は閉じた。**
#   3.6.0  n_action_steps を 2 → 5。**3.5.0 から 1 変数**（重み・正規化・EMA・
#          samples は据え置き）。**狙いはスコアではなく打ち切りの余裕**である。
#          n=2 は 1 本 300 ステップで 90 秒かかり、**39 本で 1 時間の上限に当たる**。
#          n=5 なら 48 秒で 74 本まで持つ。**予選最終評価の件数は非公開**で、
#          運営は「リーダーボードとは総 Trial 数が異なる」と明言している
#          （リーダーボードは 8 本）。根拠は下の n_action_steps の節。
#          **同時に reset の点数の印字を落とした**（挙動は変わらない。理由は
#          MyPolicy.reset の注記）。
#   3.6.0  → **本番 0.184。過去最高。しかも合計 272 秒（3.5.0 の 487 秒の 56%）。**
#          1 ステップあたり 0.154 秒で、上の見積もり 0.16 秒と一致した。
#          **0.181 との差 0.003 は 0.06 本ぶん＝同じ。同じ点を 56% の時間で取った。**
#          1 時間の上限に対しては 78〜105 本まで持つ（n=2 は 39〜58 本）。**n=5 に確定。**
#          ステップ数は [300, 96, 300, 300, 136, 142, 300, 192]。3.5.0 で解けた ep7 が
#          戻り、代わりに ep8 が 300 → 192 になった。**早く終わった回は 5 → 4 本に
#          減ったのにスコアは上がった。ステップ数から成功本数は読めない**の再確認である。
#   3.6.1  **`reset` の印字を完全に落とした。挙動は変わらない（patch）。**
#          3.6.0 では決め方（`lexical` など）だけ残していたが、**8 本すべてが
#          `lexical` だったのは今回そうだっただけで、次も本数を割らない保証は無い。**
#          正規化が効いているかは起動時の `normalize=on` で分かるので困らない。
#   3.7.0  n_action_steps を 5 → 8。**3.6.1 から 1 変数**（重み・正規化・EMA・
#          samples は据え置き。印字も 3.6.1 のまま）。
#          **2 → 5 で点が落ちなかったので、同じ向きにもう一歩だけ伸ばす。**
#          1 ステップあたりは `0.065 + 0.464 / 8 = 0.123 秒`の見込みで、
#          **300 ステップの 1 本が 37 秒 → 1 時間で 97 本**（n=5 は 48 秒・74 本）。
#          `chunk_size` は 50 なので 8 は範囲内である。
#          **講義資料（第7回 part1 p.10）は一般に k=16〜32 と書く。**
#          n=2 が良かったのはモデルが弱く誤差の蓄積を頻繁に打ち切る必要が
#          あったためと読めるので、**正規化で 2 本が解けた今、上げ側を試す価値がある。**
#          → **落ちたら 3.6.1（0.184）に戻す。** 3.6.1 は最終提出の床として残す。
#   3.8.0  **π0.5 のチェックポイントも同じサーバーで動くようにした。**
#          切り替えはチェックポイント自身の `config.json` の `type` を見るだけで、
#          **SmolVLA の道は 1 行も変えていない**（保険の提出物を壊さないため。
#          `type` が `smolvla` なら 3.7.0 と同じ経路をそのまま通る）。
#          足したものは 3 つだけである。
#          - `_build_pi05`：**RAM 18GB を要求しない読み込み**（理由はその docstring）
#          - `n_action_steps` の既定を π0.5 では 10 に（学習時の値。SmolVLA は 8 のまま）
#          - π0.5 では埋め込みでの指示文の寄せを切る（SmolVLA の VLM を触る実装で、
#            π0.5 には無い。文字列での寄せはそのまま効く）
#   3.9.0  **π0.5 の `n_action_steps` の既定を 10 → 16 にした。ここだけの 1 変数。**
#          手元の 36 件で 10 → 16 は成功 24→25 件・衝突 7→6 件・
#          平均ジャーク 4.300→3.699（−14%）・固まった回 2→0 と**4 つとも good 側**。
#          **採点の総合スコアは成功率だけでなく滑らかさも重み付きで入る**
#          （`docs/reference/17_scoring_formula.md`。採点環境は秘匿の
#          `total_score_config.json` を持ち込み、本番のスコアはどれも 0.125 の
#          倍数にならない）。**滑らかさの効き目は手元では測れず、
#          リーダーボードに出すことだけが測る方法である。**
#          3.8.0（n=10）が 0.177 なので、**これは 1 変数だけ違う比較になる**
#          （採点タスクと乱数種は提出をまたいで固定＝運営 Q&A No.26）。
#          → **落ちたら 3.8.0 でも 3.6.1（0.184）でもなく、数字を見て決める。**
#   3.9.0  → **本番 0.264。過去最高を 0.080 上回った。**ステップ数は
#          [90, 102, 225, 188, 216, 123, 96, 102] で**8 本すべてが 300 ステップより前に
#          終わった**（n=10 では ep3 と ep7 が 300 に張り付いていた）。
#          **`pipeline/rollout.py` は `done` だけで打ち切る**（衝突では止まらない）
#          ため、**300 ステップより前に終わった＝ゴールに届いた**と読める。
#          つまり**リーダーボードの 4 タスクは全部解けており、残りは
#          1mm の失格と滑らかさ・ステップ数の配点である。**
#   3.10.0 **`PARC_BLEND_STEPS` の既定を 0 → 4 にした。3.9.0 から 1 変数。**
#          chunk の継ぎ目の最初の 4 ステップだけ直前の行動へ寄せる。
#          手元 36 件で成功 25 → 28（0.694 → 0.778。**過去最高。weak3x 0.722 も
#          上回った**）、方策の衝突率 0.111 → 0.083、平均ジャーク 3.699 → 3.320
#          （−10%）、SPARC は −2.595 → −2.603（横ばい）。
#          **「型 D（課題ができていない）は設定値では動かない」という読みを
#          訂正した** ——型 D の 5 件のうち 4 件がこれだけで解けた。
#          出典 work/gpu_eval/0814_172608_pi05_24k_blend4_n16_ep1_max300_blend4。
#          **blend8 は行き過ぎ**（同じ 12 件で成功 7 対 9・3.10.0 が直した
#          tb_13 を壊した・長い手本の 1 件が 300 ステップに張り付いた）。**山は 4。**
#   4.0.0  **重みを RunPod のフルパラメータ学習 3,000 ステップに差し替えた**
#          （`work/artifacts/pi05_rp3000`）。段 1 の 24,000 ステップの続きを、
#          **画像を見る側まで解いて**batch 64・lr 5e-6→2.5e-6 で 3,000 ステップ。
#          `blend_steps=4` は 3.10.0 から引き継ぐので、**3.10.0 との差は重みだけ**。
#          テンソルの並びは段 1 と完全一致（813 個・型も形も差なし）なので
#          **コードは 1 行も変えていない＝リハーサルの心配が無い。**
#          段 1 は 18,000 ステップ以降 loss 0.2375→0.2352→0.2360 で頭打ちだったのが、
#          この 3,000 ステップで 0.2294 → 0.2100（−11%）まで下がり、まだ下がっている。
#   4.1.0  **`PARC_SAMPLES` > 1 を π0.5 でも使えるようにした（既定は 1 なので
#          挙動は 4.0.0 と同じ）。** π0.5 の `select_action` は
#          `predict_action_chunk` を呼ぶので、包む先をそちらにした。
#          **同時に medoid の距離を「実際に実行するステップ数」までに切った。**
#          chunk は 50 ステップぶん作られるのに実行するのは n 歩だけなので、
#          50 ステップで選ぶと**捨てる区間が選択を奪う**（作った例で再現した）。
#          版 3.4.0 で samples=3 に差が出なかったのは、**n=2 のとき距離の 96% が
#          実行しない行動から来ていた**ためという疑いが濃い。
__version__ = "4.5.1"

import argparse
import math
import re
from collections import deque
from abc import ABC, abstractmethod
from typing import Callable

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================

# 行動 7 次元のうち 0〜5 は EEF の並進・回転の差分。6 はグリッパで、
# 逆正規化統計の min/max がちょうど ±1、std 0.999 のバンバン制御である。
# 後処理で均してよいのは前半 6 次元だけ。
_POSE_DIMS = slice(0, 6)
# 手先の並進の差分（dx, dy, dz）と、向きの差分（droll, dpitch, dyaw）。
# 運営資料 11 ページの並び `[dx, dy, dz, droll, dpitch, dyaw, gripper]` に対応する。
_TRANSLATION_DIMS = slice(0, 3)
_ROTATION_DIMS = slice(3, 6)

# 寄せ先。**学習データ（lerobot/libero_plus）の指示文 40 種そのままである。**
# 出し方は `python3 solution/tools/suite/instruction_targets.py --emit python`。
# 2 つの独立な資料が一致することを確かめてある —— LIBERO-plus の
# `task_classification.json` から取った素のタスク名と、学習データのエピソードの
# メタデータ（`task` 列）。**40 種にならなければ道具がその場で止まる。**
_TRAINING_INSTRUCTIONS = (
    "open the middle drawer of the cabinet",
    "open the top drawer and put the bowl inside",
    "pick up the alphabet soup and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the book and place it in the back compartment of the caddy",
    "pick up the butter and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the orange juice and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "push the plate to the front of the stove",
    "put both moka pots on the stove",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "put the cream cheese in the bowl",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put the wine bottle on the rack",
    "put the wine bottle on top of the cabinet",
    "put the yellow and white mug in the microwave and close it",
    "turn on the stove",
    "turn on the stove and put the moka pot on it",
)

# 語の切り出し。小文字化してから英数字の並びだけを拾う。
_WORD_RE = re.compile(r"[a-z0-9]+")

# **語の重なりで寄せると決める合格ライン。** LIBERO-plus の実物 1,597 種
# （変種 10,030 件）に対して測って決めた
# （`solution/tools/suite/check_instruction_normalization.py --sweep`）。
#
# | 合格ライン | 本番の 2 カテゴリ（40 種） | 言い換え（1,527 種） |
# |---|---|---|
# | **0.50 / 差 0.05（採用）** | **40 種すべて正しい・誤り 0** | 468 正しい・誤り 4 |
# | 0.40 / 差 0.05 | 40 種すべて正しい・誤り 0 | **709 正しい**・誤り 15 |
# | 0.00 / 差 0.00（合格ラインなし） | 40 種すべて正しい・誤り 0 | 992・**誤り 535** |
# | 0.80 / 差 0.05 | **33 種しか寄せられない** | 13・誤り 0 |
#
# **本番の 2 カテゴリは 0.70 まで 40/40・誤り 0 で動かない。** 選び方が効くのは
# 言い換えだけである。**誤って別のタスクの文へ寄せるのが最悪の失敗**なので
# （いま解けている回を壊しうる）、狭い側の 0.50 で出した。
#
# **【8/13・決着】0.50 のままでよい。** 版 3.5.0 の採点ログで **8 本すべてが
# `lexical`（`unchanged` は 0 件）**だった ——**本番で渡る文は 8 本とも 0.50 を
# 超えて寄っている。**緩める必要が無い。**この読み取りに使った印字は版 3.6.1 で
# 落としたので、今後この値を採点ログから決め直すことはできない。**
_MIN_LEXICAL = 0.50
_MIN_LEXICAL_MARGIN = 0.05

# **埋め込み最近傍（Q&A No.31）は既定で使わない。**
# 言い換えは物の名前を避けるので語の重なりでは届かない（1,527 種のうち寄せられて
# いるのは 468 種で、残り 1,055 種は触らないまま）。
# 同梱の SmolVLM で意味を見れば届く見込みがあるが、
# **手元（macOS）に torch が無く精度を測れていない。** 測らずに入れると
# 「別のタスクの文へ寄せる」危険が読めないので、**測ってから既定を変える。**
# 測り方は docs/howto/03_gpu_eval.md の「指示文の正規化の精度を測る」。
_USE_EMBEDDING = False
_MIN_SIMILARITY = 0.90
_MIN_SIMILARITY_MARGIN = 0.02
# 意味で 1 番になっても、語がまるで重なっていなければ寄せない（両方を要求する）。
_MIN_EMBEDDING_LEXICAL = 0.20


class InstructionNormalizer:
    """入ってきた指示文を、学習データの言い方のうち最も近いものへ寄せる。

    **モデルは 40 種の言い方しか見ていない。** 評価はそれを書き換えて渡してくる
    ので、入り口で戻す。**壊す経路が無い仕掛けである** —— 学習と同じ文が来たら
    小文字化だけして自分に写り、迷ったときは何もしない。

    寄せ方は 2 通り。**どちらも「全タスク共通・決定論的・タスク別分岐なし」**で、
    運営 Q&A No.29 の許容条件に合わせてある。

    1. **同じ文なら、そのまま**（小文字化・空白の詰め・末尾の句点落としだけ）
    2. **語の重なりで最も近い 1 つへ寄せる。** 珍しい語を重く見る（下の `_idf`）

    埋め込み最近傍（Q&A No.31）は `attach_embeddings` で足せる。
    **足さなければ 1〜2 だけで動く**ので、torch も VLM も要らない。

    Parameters
    ----------
    targets : tuple[str, ...]
        寄せ先。既定は学習データの 40 種。
    min_lexical : float
        語の重なりで寄せると決める下限。
    min_lexical_margin : float
        1 番と 2 番の差の下限。**迷ったら寄せない。**
    """

    def __init__(
        self,
        targets: tuple[str, ...] = _TRAINING_INSTRUCTIONS,
        *,
        min_lexical: float = _MIN_LEXICAL,
        min_lexical_margin: float = _MIN_LEXICAL_MARGIN,
    ) -> None:
        self._targets = tuple(targets)
        self._min_lexical = min_lexical
        self._min_lexical_margin = min_lexical_margin

        # 同じ文が来たときの受け口。**寄せ先そのものを返す**ので、
        # 大文字や末尾の句点も同時に落ちる。
        self._exact = {self.canonical(target): target for target in self._targets}

        # **珍しい語を重く見る。** `bowl` や `plate` は 40 種の多くに出るので
        # 効かせても区別できないが、`ramekin` や `drawer` は 1 つに絞れる。
        # これが無いと `… in the top drawer of the wooden cabinet …` と
        # `… on the wooden cabinet …` の差が消える（実測で取り違えが起きた）。
        frequency: dict[str, int] = {}
        for target in self._targets:
            for word in set(_WORD_RE.findall(target)):
                frequency[word] = frequency.get(word, 0) + 1
        total = len(self._targets)
        self._idf = {
            word: math.log((total + 1.0) / (count + 1.0)) + 1.0
            for word, count in frequency.items()
        }
        # 40 種に無い語は最も珍しいものとして扱う。**長い言い換えの点数が
        # 下がる向き**なので、迷いを寄せない側へ倒せる。
        self._unknown_idf = math.log(total + 1.0) + 1.0
        self._target_vectors = [self._vector(target) for target in self._targets]

        self._embeddings = None
        self._embed: Callable[[list[str]], np.ndarray] | None = None

    @property
    def uses_embeddings(self) -> bool:
        """意味で寄せる道（Q&A No.31）が入っているか。"""
        return self._embeddings is not None

    @staticmethod
    def canonical(text: str) -> str:
        """比べる前の下ごしらえ。小文字化・空白の詰め・末尾の句点落とし。"""
        return " ".join(text.lower().split()).rstrip(" .!")

    def _words(self, text: str) -> set[str]:
        """語に切る。**知らない語は、複数形を単数形に直して 1 度だけ試す。**

        評価は `cookies box` と書くが学習データは `cookie box` である。
        `s` を落とすだけで一致するので、**40 種に無い語のときに限って**試す。
        知っている語には触らないので、`pots` や `plates` の意味は変わらない。
        """
        words = set()
        for word in _WORD_RE.findall(text):
            if word not in self._idf and word.endswith("s") and word[:-1] in self._idf:
                word = word[:-1]
            words.add(word)
        return words

    def _vector(self, text: str) -> dict[str, float]:
        """文を、珍しさで重み付けした長さ 1 のベクトルにする。"""
        weights = {
            word: self._idf.get(word, self._unknown_idf)
            for word in self._words(text)
        }
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if norm == 0.0:
            return {}
        return {word: value / norm for word, value in weights.items()}

    def _lexical_scores(self, text: str) -> list[float]:
        """寄せ先 40 種それぞれとの、語の重なりの点数（0〜1）。"""
        vector = self._vector(text)
        return [
            sum(value * target.get(word, 0.0) for word, value in vector.items())
            for target in self._target_vectors
        ]

    def attach_embeddings(self, embed: Callable[[list[str]], np.ndarray]) -> None:
        """埋め込みで意味を見る道を足す（Q&A No.31）。

        Parameters
        ----------
        embed : Callable[[list[str]], np.ndarray]
            文の一覧を受け取り、**長さ 1 に正規化した**行ベクトルの並び
            (N, D) を返すもの。40 種はここで 1 度だけ埋め込む。
        """
        self._embed = embed
        self._embeddings = np.asarray(embed(list(self._targets)), dtype=np.float32)
        if self._embeddings.shape[0] != len(self._targets):
            raise ValueError(
                f"埋め込みの本数が寄せ先と合いません: {self._embeddings.shape}"
            )

    def normalize(self, text: str) -> tuple[str, str, float]:
        """指示文を寄せる。

        Parameters
        ----------
        text : str
            `/reset` で渡された指示文。

        Returns
        -------
        instruction : str
            モデルへ渡す文。寄せられなければ小文字化した入力そのまま。
        how : str
            どの決め方で寄せたか（`exact` / `lexical` / `nearest` / `unchanged`）。
            **採点ログに出して、本番で何が起きたかを後から読むために使う。**
        score : float
            そのときの点数。`unchanged` のときは 1 番の点数。
        """
        cleaned = self.canonical(text)
        if not cleaned:
            return text, "empty", 0.0

        hit = self._exact.get(cleaned)
        if hit is not None:
            return hit, "exact", 1.0

        scores = self._lexical_scores(cleaned)
        order = sorted(range(len(scores)), key=lambda index: -scores[index])
        best, second = order[0], order[1]
        margin = scores[best] - scores[second]
        if scores[best] >= self._min_lexical and margin >= self._min_lexical_margin:
            return self._targets[best], "lexical", scores[best]

        if self._embeddings is not None and self._embed is not None:
            similarity = self._embeddings @ np.asarray(
                self._embed([cleaned]), dtype=np.float32
            )[0]
            ranked = np.argsort(-similarity)
            top, runner = int(ranked[0]), int(ranked[1])
            if (
                similarity[top] >= _MIN_SIMILARITY
                and similarity[top] - similarity[runner] >= _MIN_SIMILARITY_MARGIN
                and scores[top] >= _MIN_EMBEDDING_LEXICAL
            ):
                return self._targets[top], "nearest", float(similarity[top])

        return cleaned, "unchanged", scores[best]


def _no_weight_init():
    """重みのランダム初期化を飛ばす入れ物を返す（π0.5 の起動を速くするため）。

    **採点ハーネスはサーバーが 120 秒で繋がらないと諦める**
    （`pipeline/remote_policy.py` の `wait_for_server(timeout=120.0)`）。
    π0.5 は 41 億個のパラメータを持ち、**その乱数での初期化だけで CPU 1 コア
    300 秒級**かかる（手元の実測。読み込み全体は 352 秒だった）。

    **初期化した値は 1 つも使わない。** 直後に `model.safetensors` の値で
    全部上書きし、1 つでも欠けていれば `_build_pi05` が落とすからである。
    だから初期化そのものを飛ばす。確保だけは行われるので、メモリの山は変わらない。

    **飛ばして安全なことは測って確かめてある**（2026-08-14、CPU 1 コア）。

    | 測ったもの | 飛ばす | 飛ばさない |
    |---|---|---|
    | 組み立ての時間 | **3.2 秒** | 264.3 秒 |
    | チェックポイントに無い buffer 6 個 | **6 個とも完全に一致** | （基準） |

    つまり rotary の `inv_freq` のように模型自身が計算して作る値は、
    初期化を飛ばしても同じものが入る。念のため `_build_pi05` 側でも
    有限かどうかを見ている。

    `transformers` が持っている仕掛けを使う。**版によって置き場所が違い、
    採点環境の版は分からない**ので、2 か所を順に探し、どちらも無ければ
    「何もしない入れ物」を返す（その場合は遅いだけで、動きは変わらない）。

    | 版 | 置き場 |
    |---|---|
    | 5.x | `transformers.initialization.no_init_weights`（手元の 5.5.4 で確認） |
    | 4.x | `transformers.modeling_utils.no_init_weights` |

    Returns
    -------
    context : 入れ物
        `with` に渡せるもの。
    """
    import contextlib
    import importlib

    for module_name in ("transformers.initialization", "transformers.modeling_utils"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        factory = getattr(module, "no_init_weights", None)
        if factory is not None:
            return factory()

    print("[parc] no_init_weights が見つからない。初期化を飛ばさずに読む"
          "（遅いだけで挙動は同じ）", flush=True)
    return contextlib.nullcontext()


class MyPolicy(BasePolicy):
    """同梱した LeRobot v0.6.0 で SmolVLA、または π0.5 を動かす。

    **どちらを動かすかは `model_weights/config.json` の `type` で決まる**
    （版 3.8.0）。置いた重みが SmolVLA なら、3.7.0 までと 1 行も違わない道を
    通る。π0.5 のときだけ `_build_pi05` を通り、埋め込みでの指示文の寄せを
    切る。**どちらの場合も `model_weights/vlm/` からトークナイザを読む**
    （SmolVLA は SmolVLM の、π0.5 は paligemma のもの。置き場を共通に
    してあるので、評価スクリプトも提出物の組み立ても分岐しない）。

    採点環境は Python 3.10 固定で LeRobot（requires-python >= 3.12）を
    pip install できないため、ソースを vendor/ に同梱して sys.path から読む。

    評価パイプラインが渡す観測は LIBERO の生の形式であり、モデルが学習した
    lerobot/libero_plus の形式とは異なる。差分は次の 3 点で、いずれも
    LeRobot 側の LiberoProcessorStep が行う変換に合わせている。

    - 画像を 180 度回転する（LIBERO のカメラ向きとデータセットの規約が違う）
    - state は [eef_pos(3), eef_quat を軸角に変換(3), gripper_qpos(2)] の 8 次元
    - action はモデル出力を postprocessor で逆正規化したものをそのまま返す

    推論は chunk 単位で走る（chunk_size=50）。select_action が内部キューを
    管理するため、重い推論は 50 ステップに 1 回だけ発生する。
    """

    _CHW_SCALE = 255.0

    def __init__(self):
        import os
        import sys
        from pathlib import Path

        here = Path(__file__).resolve().parent

        # 採点環境は外部通信が遮断される。Hub へ取りに行かせない。
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(here / ".hf_home"))

        # 同梱した LeRobot を import 可能にする。
        # vendor/lerobot ではなく vendor を入れること。パッケージ自身を
        # sys.path に載せると lerobot/types.py が標準ライブラリの types を
        # 隠して循環 import になる。
        sys.path.insert(0, str(here / "vendor"))

        import torch

        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.env_processor import LiberoProcessorStep

        self._torch = torch
        weights = here / "model_weights"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # **`from_pretrained` より前に factory を import しておくこと。**
        # 設定の型は「選択肢クラス」として登録されており、登録は
        # `lerobot.policies.factory` を読み込んだ時に起きる。順番を入れ替えると
        # π0.5 のチェックポイントで
        # `Couldn't find a choice class for 'pi05'` になって起動できない（実測）。
        config = PreTrainedConfig.from_pretrained(weights)
        config.device = self._device

        # **どちらの模型を積んでいるかは、チェックポイント自身に聞く。**
        # 置いてある重みが SmolVLA なら 3.7.0 と同じ道をそのまま通る（版 3.8.0）。
        self._kind = str(config.type)

        if self._kind != "pi05":
            # VLM の重みはチェックポイントに含まれる。True のままだと Hub へ
            # 取りに行き、通信が遮断された採点環境では起動に失敗する。
            config.load_vlm_weights = False
            # 設定とトークナイザは同梱物から読む（本体の重みは含めていない）。
            config.vlm_model_name = str(weights / "vlm")

        # 1 回の推論で作った chunk のうち、何ステップ分を実行するか。
        #
        # チェックポイントの値は 50 で、chunk を丸ごと開ループで実行する。
        # **10 に縮めると成功率が 0.0625 → 0.90 に跳ねる**（実測。
        # docs/strategy/05_実測結果.md）。方策は各時点の判断自体は正しく、
        # 失敗の原因は開ループ 50 ステップの間に誤差が積み上がることだった。
        # 再計画を 5 倍の頻度で入れると、その蓄積が毎回打ち切られる。
        #
        # 本番採点でも効いた。50 → 10 でスコアが 0.058 → 0.096 になっている。
        #
        # 小さくしすぎると、毎ステップ独立に flow matching の乱数を引くため
        # 軌道がガタつく恐れがある。`PARC_EMA_ALPHA` がそれを打ち消す。
        #
        # **版 3.6.0 で 2 → 5 に戻した。理由はスコアではなく打ち切りである。**
        # 採点 10 回ぶんの秒数とステップ数から、1 ステップあたりの時間は
        # `0.065 秒 + 0.464 秒 / n` に乗る（n=1 で 0.529・n=2 で 0.30・n=5 で 0.16。
        # 実測と 3 点とも一致）。予選 PDF p.12 の**「Track 評価に 1 時間以上かかる
        # 場合はタイムアウト」**に当てはめると、1 本 300 ステップの最悪で
        #
        #     n=2 → 90 秒/本 → **39 本で打ち切り**
        #     n=5 → 48 秒/本 → **74 本まで持つ**
        #
        # **予選最終評価の件数は非公開で、運営は「リーダーボードとは総 Trial 数が
        # 異なる」と明言している**（リーダーボードは 8 本）。一方 n=2 と n=5 の
        # スコア差は 0.166 対 0.144＝**0.5 本ぶん**で、1 本＝約 0.047 の分解能では
        # 区別できない。**測れない 0.5 本のために 0 になる目を買わない。**
        #
        # 講義資料（`docs/第7回 講義資料part1.pdf` p.10）も一般には k=16〜32 と書く。
        # n=2 が良かったのはモデルが弱く誤差の蓄積を頻繁に打ち切る必要があったため
        # と読めるので、**素性が上がれば n は上げる側が自然**でもある。
        #
        # **【8/13 実測】n=5 は本番 0.184・272 秒。1 ステップ 0.154 秒で上の式と一致した。**
        # n=2（0.181・487 秒）と点は同じで、時間は 56%。**上げても落ちなかった。**
        #
        # **版 3.7.0 で 8 まで伸ばす。** 見込みは 1 ステップ 0.123 秒 → 300 ステップの 1 本が
        # 37 秒 → **1 時間で 97 本**（n=5 は 48 秒・74 本）。`chunk_size` は 50 なので
        # 8 は範囲内。**落ちたら 3.6.1（n=5・0.184）に戻す。**
        #
        # **π0.5 の既定は 16 である（版 3.9.0 で 10 から上げた）。**
        # 手元の 36 件で 10 → 16 は成功 24→25 件・衝突 7→6 件・
        # 平均ジャーク 4.300→3.699（−14%）・固まった回 2→0 と**4 つとも good 側**で、
        # **採点の総合スコアには滑らかさも重み付きで入る**
        # （`docs/reference/17_scoring_formula.md`）。`chunk_size` は 50 なので
        # 16 は範囲内。1 ステップの見込みは 0.084 秒で、`/act` の 10 秒に対して桁違いに余裕。
        # SmolVLA の既定は 8 のままである。設定値の名前は共通なので、
        # `PARC_N_ACTION_STEPS` を渡せばどちらも同じように上下できる。
        config.n_action_steps = int(os.environ.get(
            "PARC_N_ACTION_STEPS", "16" if self._kind == "pi05" else "8"
        ))

        if self._kind == "pi05":
            # flow matching を何回刻むか。**1 回の推論の秒数はほぼこれに比例する。**
            # `n_action_steps` は「推論と推論の間隔」を変えるだけで 1 回の秒数は
            # 変えないので、**`/act` の 10 秒制限に効くのはこちらだけ**である。
            # 既定はチェックポイントの値（10）のまま。減らしたときの質は未測定。
            # **既定を 20 にした（版 3.12.0）。** 手元 36 件で
            # ゴール到達 30 → 34（上限は 34。落下物 2 件は誰も取れない）、
            # 打ち切り張り付き 6 → 2、**両方が成功した 23 件のステップ数は ±0（+0.9%）**。
            # **秒数だけ払ってステップ数を買う形**で、blend/RTC のように遅くならない
            # （blend4 はステップ数 +29% で本番 0.264 → 0.169 に落ちた）。
            # 本番の見込みは約 150 秒で、1 時間の上限に 24 倍の余裕がある。
            steps = os.environ.get("PARC_NUM_INFERENCE_STEPS", "10")
            if steps:
                config.num_inference_steps = int(steps)
            self._policy = self._build_pi05(config, weights)
        else:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            self._policy = SmolVLAPolicy.from_pretrained(weights, config=config)

        # ============================================================
        # 推論時量子化スイープ（候補 C4: bnb / C5: torchao。work/quant_sweep/
        # CANDIDATES.md）。**`QUANT_SERVE_MODE` が空のときは何もしない no-op**
        # ——本番・通常の評価は必ずここを未設定で通るので、既定の挙動は
        # 1 バイトも変わらない。C1〜C3（学習時候補）と違い、こちらは
        # **すでにフル精度で読み終えた重みに後掛けする**（再学習が要らない）。
        #
        # 実体は solution/tools/quant/quant_serve_bnb.py / quant_serve_torchao.py。
        # ロジックをここへ埋め込まず薄い分岐だけにしてあるのは、
        # solution/train/experiments/quantization/lerobot_train_with_quant.py（C1〜C3 側の分岐）と
        # 同じ流儀に揃えるため。
        #
        # `solution/tools/*.py` はこのファイル自身とは別ディレクトリにある
        # （提出物の一部ではなく実験用のツールのため）。実行のされ方によって
        # 隣にある場所が変わるので両方を通しておく——
        #   - run_docker_eval.sh / run_gpu_eval.sh 経由: quant_serve_*.py を
        #     policy_server.py と同じ「serve」ディレクトリへコピー・マウントする
        #     ので `here` 直下にある
        #   - リポジトリ内から直接 `python solution/policy_server.py` する場合は
        #     `here / "tools"`（= solution/tools）にある
        quant_mode = os.environ.get("QUANT_SERVE_MODE", "")
        if quant_mode:
            if self._kind != "pi05":
                raise RuntimeError(
                    f"QUANT_SERVE_MODE={quant_mode!r} は π0.5 専用の実験的候補で"
                    f"ある（work/quant_sweep/CANDIDATES.md の C4/C5）。"
                    f"SmolVLA（config.type={self._kind!r}）では未実装。"
                )
            sys.path.insert(0, str(here / "tools"))
            if quant_mode == "bnb_vision4bit":
                import quant_serve_bnb

                quant_serve_bnb.apply(self._policy, self._device)
            elif quant_mode == "torchao_int8":
                import quant_serve_torchao

                quant_serve_torchao.apply(self._policy, self._device)
            else:
                raise ValueError(
                    f"QUANT_SERVE_MODE の値が不明: {quant_mode!r}"
                    "（bnb_vision4bit / torchao_int8 のいずれか）"
                )

        self._policy.to(self._device)
        self._policy.eval()

        # tokenizer_processor は保存時の設定に Hub の repo 名が焼き込まれている。
        # 上書きしないと起動時に Hub へ取りに行き、通信のない採点環境で失敗する。
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            config,
            str(weights),
            preprocessor_overrides={
                "device_processor": {"device": self._device},
                "tokenizer_processor": {"tokenizer_name": str(weights / "vlm")},
            },
        )
        self._env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

        # 行動の後処理。運営 Q&A No.10 で「VLA が生成した Action を、全タスク
        # 共通のルールで整える処理」は禁止事項に含まれないと明言されている。
        # 使うのはモデル出力と過去の行動だけで、タスクや seed は参照しない。
        #
        # **EMA の既定は 0.3。** 直前の行動を 3 割混ぜて動きをならす。
        # 80 エピソードの実測で、成功率 0.875 → 0.888（誤差の範囲）を保ったまま
        # avg_cartesian_jerk が 4.5161 → 3.5335（−22%）、rms が −16% になった。
        # 衝突率も 0.087 → 0.075。懸念した「惰性で行き過ぎて接触が増える」は
        # 起きていない。n を小さくしたときのガタつきを抑える役目も兼ねる。
        #
        # **版 3.10.0 で blend_steps の既定を 0 → 4 にした（π0.5・n=16 で実測）。**
        # chunk の継ぎ目の最初の 4 ステップだけ直前の行動へ寄せる。手元 36 件で
        # 成功 25 → 28（0.694 → 0.778。過去最高）、方策の衝突率 0.111 → 0.083、
        # avg_cartesian_jerk 3.699 → 3.320（−10%）、SPARC は −2.595 → −2.603（横ばい）。
        # 出典 work/gpu_eval/0814_172608_pi05_24k_blend4_n16_ep1_max300_blend4/
        # summary_compare.txt。**採点環境は環境変数を渡さないので、既定を変えるしかない。**
        self._blend_steps = int(os.environ.get("PARC_BLEND_STEPS", "0"))
        self._ema_alpha = float(os.environ.get("PARC_EMA_ALPHA", "0.3"))

        # **行動に掛ける倍率。既定は 1.0（＝何もしない）である。**
        # 運営 PDF の「よくいただいたご質問と回答」が
        # **「全タスク共通・決定論的な action の後段補正 → 禁止事項に該当しません」**
        # と明言しているので、全タスク・全ステップに同じ定数を掛けるこの形は使える。
        #
        # 並進を上げると 1 歩で進む距離が増えて**歩数が減る**。回転を下げると
        # 本選で採点指標に入った**手先の回転の総量**が下がる。
        # **どちらも既定のままなら提出時の挙動は 1 ビットも変わらない。**
        # **既定は 1.0（何も掛けない）に戻した。**1.15 で出した版 4.5.0 は
        # Track1 が 0.370 → 0.136、Track2 が 0.154 → 0.035 と両方下がった
        # （2026-08-26 の採点。倍率と種以外の設定は 4.2.0 と同一）。
        # Track1 の歩数が 1,039 → 1,398 に増えており、行き過ぎて戻る動きが
        # 増えたと読める。**上げる向きは打ち切る。**
        self._gain_translation = float(os.environ.get("PARC_GAIN_TRANSLATION", "1.0"))
        self._gain_rotation = float(os.environ.get("PARC_GAIN_ROTATION", "1.0"))
        self._n_action_steps = int(self._policy.config.n_action_steps)

        # **`PARC_SEED` を渡すと、エピソードごとに乱数を固定する（既定は固定しない）。**
        # 行動は flow matching で作るので、毎回ノイズを引き直している
        # （`_get_action_chunk` に `noise=None` を渡す＝モデル側が引く）。**同じ設定で
        # 回し直すと結果が変わる。** 手元の 49 回はすべて 1 タスク 1 エピソードで、
        # この揺れ幅を測ったことがない。
        #
        # **これは「2 つの設定を比べられるようにする」ものではない。** 種を揃えても、
        # `blend_steps` のように行動そのものを書き換える設定は 16 歩目以降で軌道を
        # 変えてしまい、300 歩の境界にいたタスクはどちらへも転ぶ。**設定の効果と
        # 境界での転びを分けるには、種を変えた複数回が要る。** 種を揃える意味は、
        # 複数回で比べるときに対応が取れてばらつきが小さくなる点にある。
        #
        # 種はエピソードごとに `PARC_SEED + 通し番号` にする。**同じ値にすると全
        # エピソードが同じ軌道になり、揺れ幅が 0 に見えてしまう。**
        # **採点環境は環境変数を渡さないので、既定（固定しない）が提出時の挙動である。**
        # **【2026-08-26・4.5.0】既定を "0" にした（それまでは空＝固定しない）。**
        # 種は当たり外れを作らない ── 分布を変えずに 1 つの引きを選ぶだけなので、
        # 種の有無で期待値は動かない。**得られるのは再現性である。**
        # 同じ zip を出し直せば同じ点が返り、設定を 1 つ変えた 2 本は
        # **同じ引きの上で**比べられる。`PARC_SEED=` と空で渡せば従来どおり固定しない。
        seed_text = os.environ.get("PARC_SEED", "0")
        self._seed = int(seed_text) if seed_text else None
        self._episode_index = 0

        # chunk を何回引くか。1 なら素の挙動。
        # 版 3.4.0 で 3 にして本番で測ったが、0.166 → 0.168（0.04 本ぶん）で
        # **差が無く、時間だけ 2.7 倍になった。版 3.5.0 で 1 に戻した。**
        #
        # **【8/14 20:00】その測定は n=2 と n=5 のときのものである。**
        # n=2 は 2 ステップごとに引き直すので、外れを引いてもすぐ打ち消される。
        # **外れを 16 ステップぶん引きずるのは n が大きいときだけ**で、そこは未測定。
        # π0.5 は n=16 なので、軸を開け直して測る。
        self._samples = int(os.environ.get("PARC_SAMPLES", "1"))
        if self._samples > 1:
            self._install_chunk_sampler()

        # RTC（Real-Time Chunking）。**既定は切ってある。**
        # `blend_steps` が「出た行動を後からならす」のに対し、RTC は
        # **継ぎ目で前の計画と一致するように生成させる**（塗り足しと同じ考え方）。
        # blend4 が効いた＝継ぎ目が本物、という裏づけがあるので筋は良い。
        # **ただし一度も採点環境で走っていない道**なので、
        # 例外が出たら `blend_steps` の道へ戻す受け皿を付けてある（_get_action_rtc）。
        self._rtc_enabled = os.environ.get("PARC_RTC", "0") not in ("0", "", "false")
        self._rtc_horizon = int(
            os.environ.get("PARC_RTC_HORIZON", "4")
        )
        self._rtc_guidance = float(os.environ.get("PARC_RTC_GUIDANCE", "10.0"))
        self._rtc_queue: deque = deque()
        self._rtc_previous_tail = None
        self._rtc_fell_back = False
        if self._rtc_enabled:
            self._install_rtc()

        self._previous_action: np.ndarray | None = None
        self._step_index = 0

        # 指示文の正規化（版 3.5.0。先頭のコメント）。
        # **既定は入り**。`PARC_NORMALIZE_INSTRUCTION=0` で切れるので、
        # 手元の評価で「入り」と「切り」を 1 変数で比べられる。
        self._normalizer: InstructionNormalizer | None = None
        if os.environ.get("PARC_NORMALIZE_INSTRUCTION", "1") != "0":
            self._normalizer = InstructionNormalizer()
            # **π0.5 では埋め込みでの寄せを切る。** `_embed_instructions` は
            # SmolVLA の `model.vlm_with_expert` を直に触る実装で、π0.5 には
            # その属性が無い。文字列での寄せ（lexical）はそのまま効く。
            use_embedding = self._kind != "pi05" and os.environ.get(
                "PARC_NORM_EMBED", "1" if _USE_EMBEDDING else "0"
            ) != "0"
            if use_embedding:
                self._normalizer.attach_embeddings(self._embed_instructions)

        # **採点ログに構成を残す。** 以前は「どの設定で採点されたか」がログから
        # 分からず、所要時間から逆算して推定するしかなかった。ここに出しておけば
        # 提出物とスコアの対応が後から確実に辿れる。
        print(
            f"[parc] policy_server {__version__}  model={self._kind}  "
            f"device={self._device}  "
            f"num_inference_steps={getattr(self._policy.config, 'num_inference_steps', '-')}  "
            f"n_action_steps={self._n_action_steps}  ema_alpha={self._ema_alpha}  "
            f"blend_steps={self._blend_steps}  samples={self._samples}  "
            f"gain=({self._gain_translation},{self._gain_rotation})  "
            f"seed={self._seed if self._seed is not None else 'off'}  "
            f"rtc={f'on(horizon={self._rtc_horizon})' if self._rtc_enabled else 'off'}  "
            f"normalize={self._normalize_mode()}  "
            f"quant_serve={quant_mode or 'off'}"
            + (f"({os.environ.get('QUANT_SERVE_SCOPE', 'vision_only')})" if quant_mode else ""),
            flush=True,
        )

        self._instruction = ""
        self._warm_up()

    def _build_pi05(self, config, weights):
        """π0.5 のチェックポイントを、**RAM 18GB を要求せずに**読む。

        LeRobot 本体の `PI05Policy.from_pretrained` は 2 か所で山を作る。

        | 段階 | 要る量 | なぜ |
        |---|---|---|
        | 組み立て | 16.6 GB | PaliGemma を**fp32 で作ってから**bf16 へ落とす作りである |
        | 読み込み | ＋8.7 GB | 8.7 GB を丸ごと辞書にしてから写すため、模型と辞書が同時に生きる |

        **採点環境の host RAM は非公開である。** 足りなければ起動に失敗し、
        推論の中身と関係なく 0 点になる。そこで 2 つ変える。

        1. 最初から bf16 の既定 dtype で組む（fp32 の 16.6 GB を通らない）
        2. safetensors を**1 テンソルずつ** `param.data` へ差し替える。
           **代入なのでチェックポイント側の dtype がそのまま入り**、
           fp32 で保存されている vision_tower や layernorm は fp32 に戻る
           ——**出来上がりは本体の道と同じ dtype の並びになる。**

        ピークは 10 GB 前後に下がる。同時に生きるのは模型＋最大 1 テンソル
        （0.98 GB の embed_tokens）だけである。

        **読めたことを数で確かめる。** 本体の `from_pretrained` は読み込みに
        失敗しても `Returning model without loading pretrained weights` と出して
        **初期化したままの模型を返す**ので、起動しただけでは学習の成果を
        使っているとは言えない。1 つでも欠けたらここで落とす。

        Parameters
        ----------
        config : PreTrainedConfig
            チェックポイントから読んだ設定（`device` と `n_action_steps` は
            呼び出し側で入れてある）。
        weights : Path
            `model_weights` ディレクトリ。

        Returns
        -------
        policy : PI05Policy
            重みを写し終えた方策（`to` と `eval` は呼び出し側で行う）。

        Reads:
          - <weights>/model.safetensors
        """
        import time

        from safetensors import safe_open

        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        torch = self._torch
        started = time.perf_counter()

        # 学習コンテナの中のパスが残っている。π0.5 は PaliGemma を設定から
        # 組むので参照されないが、誤解を生むので消しておく。
        config.pretrained_path = None

        default_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        try:
            with _no_weight_init():
                policy = PI05Policy(config)
        finally:
            torch.set_default_dtype(previous_dtype)

        parameters = dict(policy.named_parameters())
        targets = {**parameters, **dict(policy.named_buffers())}
        copied: set[str] = set()
        with safe_open(weights / "model.safetensors", framework="pt", device="cpu") as reader:
            for key in reader.keys():
                target = targets.get(key)
                if target is None:
                    continue
                tensor = reader.get_tensor(key)
                if tuple(tensor.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"重みの形が合わない: {key} "
                        f"{tuple(tensor.shape)} != {tuple(target.shape)}"
                    )
                target.data = tensor
                copied.add(key)

        missing = sorted(set(parameters) - copied)
        if missing:
            raise RuntimeError(
                f"チェックポイントに無い重みが {len(missing)} 個ある"
                f"（初期値のまま推論することになる）: {missing[:3]}"
            )

        # **初期化を飛ばした代償の確認である。** チェックポイントに入っていない
        # buffer（rotary の inv_freq など 6 個）は、模型自身の計算で作られるので
        # 初期化を飛ばしても正しく埋まる——**はずである。** もし `torch.nn.init`
        # 経由で作られていれば、飛ばしたぶん**中身が未定義のまま**残る。
        # 数が少ないのでここで見ておく。重み本体は全部上書き済みなので見ない。
        broken = [name for name, buffer in policy.named_buffers()
                  if name not in copied and buffer.numel() > 0
                  and not torch.isfinite(buffer).all()]
        if broken:
            raise RuntimeError(
                f"初期化を飛ばしたせいで壊れた buffer がある: {broken[:3]}"
            )

        print(f"[parc] pi05 の重みを {len(copied)} 個読んだ"
              f"（{time.perf_counter() - started:.1f} 秒）", flush=True)
        return policy

    def _embed_instructions(self, texts: list[str]) -> np.ndarray:
        """同梱した SmolVLM の言語側で文を埋め込む（長さ 1 に正規化して返す）。

        **外から何も取ってこない。** 使うのはチェックポイントに入っている
        VLM の言語モデル（16 層に切り詰められている）と、同梱した
        `model_weights/vlm/` のトークナイザだけである。

        平均を取る対象は最終層の隠れ状態で、**詰め物（padding）は数えない。**

        Parameters
        ----------
        texts : list[str]
            埋め込む文。

        Returns
        -------
        embeddings : np.ndarray
            (N, D) float32。行ごとに長さ 1。
        """
        torch = self._torch
        vlm_with_expert = self._policy.model.vlm_with_expert
        tokenizer = vlm_with_expert.processor.tokenizer
        text_model = vlm_with_expert.get_vlm_model().text_model

        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self._policy.config.tokenizer_max_length),
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self._device)
        mask = tokens["attention_mask"].to(self._device)

        with torch.no_grad():
            hidden = text_model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return pooled.detach().to("cpu", dtype=torch.float32).numpy()

    def _install_chunk_sampler(self) -> None:
        """chunk を複数回引き、最も典型的な 1 本を選ぶようにする。

        flow matching は確率的で、**同じ観測でも引くたびに違う行動列が出る**。
        実際、同じ初期状態を与えても成功する回と失敗する回がある。
        複数引いて中心的なものを採れば、外れを引く確率を下げられる。

        平均ではなく **medoid**（他との距離の総和が最小の実サンプル）を使う。
        平均には 2 つの問題がある。

        - 右へ避ける軌道と左へ避ける軌道を混ぜると、まっすぐ障害物へ向かう
          「どちらでもない」行動になりうる（モード間の平均）
        - グリッパは ±1 のバンバン制御なので、混ぜると中間値になり掴めない

        medoid は必ず実際に引いた 1 本を返すため、どちらも起きない。

        **包む先はモデルによって違う。**

        - SmolVLA: `select_action` が `_get_action_chunk` を直接呼ぶ
        - π0.5: `select_action` が `predict_action_chunk` を呼ぶ
          （`modeling_pi05.py` の `select_action`）

        **距離は「実際に実行するステップ数」だけで測る。** chunk は 50 ステップぶん作られるが、
        実行するのは `n_action_steps` 歩（π0.5 の既定 16）だけである。
        50 ステップで選ぶと、**捨てる 34 ステップぶんが選択を左右してしまう。**
        版 3.4.0 で samples=3 に差が出なかったのは、これが原因である疑いが濃い
        （当時は n=2 で、**50 ステップのうち 2 ステップしか使わないのに 50 ステップで選んでいた**）。
        """
        policy = self._policy
        torch = self._torch
        count = self._samples
        horizon = self._n_action_steps

        def pick_medoid(chunks):
            """引いた複数の chunk から、他との距離の総和が最小の 1 本を返す。"""
            # 距離は姿勢 6 次元だけで測る。グリッパは ±1 に振れるため、
            # 含めるとそこだけで距離が決まってしまう。
            # さらに、実行するステップ数までに切る（上の docstring を参照）。
            flat = chunks[:, :, :horizon, _POSE_DIMS].reshape(count, -1)
            distances = torch.cdist(flat, flat).sum(dim=1)
            return chunks[int(distances.argmin())]

        if self._kind == "pi05":
            original_predict = policy.predict_action_chunk

            def sampled_predict(batch, **kwargs):
                # **`.clone()` が要る（2026-08-22、compile_model=true の実機検証で発見）。**
                # `sample_actions` を `torch.compile` した状態だと、CUDA graphs に
                # 裏打ちされた静的バッファを返すことがある。clone せずに複数回呼んだ
                # 結果を後から `torch.stack` すると、2回目以降の呼び出しが1回目の
                # 出力バッファを上書きし、`RuntimeError: accessing tensor output of
                # CUDAGraphs that has been overwritten by a subsequent run` で落ちる。
                # `compile_model=false`（これまでの既定）では静的バッファを使わないため
                # 顕在化していなかった——このバグは compile_model 系の変更としてここに
                # 記録する（work/quant_sweep/PROGRESS.md 参照）。
                chunks = torch.stack(
                    [original_predict(batch, **kwargs).clone() for _ in range(count)]
                )
                return pick_medoid(chunks)

            policy.predict_action_chunk = sampled_predict
            return

        original_chunk = policy._get_action_chunk

        def sampled_chunk(batch, noise=None, **kwargs):
            chunks = torch.stack(
                [original_chunk(batch, noise, **kwargs) for _ in range(count)]
            )
            return pick_medoid(chunks)

        policy._get_action_chunk = sampled_chunk

    def _install_rtc(self) -> None:
        """継ぎ目を「後からならす」のでなく「つながるように生成させる」ようにする。

        `blend_steps` は出てきた行動を後から前の行動へ寄せる。RTC は
        **生成の途中（ノイズ除去の各段）で、前の計画に一致する向きへ押す。**
        重みは 0 番目が 1.0 で、`execution_horizon` 歩目で 0 まで直線的に落ちる
        （`modeling_rtc.py` の `get_prefix_weights`）。つまり
        **継ぎ目の直後だけ強く前の計画に従い、先へ行くほど自由になる。**

        チェックポイントの `config.json` には `rtc_config` が入っていないため、
        ここで作って `init_rtc_processor` を呼ぶ。この 1 つの呼び出しで
        方策側と内側のモデル（`PI05Pytorch`）の両方に処理器が入る
        （`modeling_pi05.py` の `init_rtc_processor`）。
        """
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        self._policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=self._rtc_horizon,
            max_guidance_weight=self._rtc_guidance,
        )
        self._policy.init_rtc_processor()

    def _get_action_rtc(self, batch: dict):
        """RTC で 1 ステップぶんの行動を返す。前の chunk の未実行部分を手がかりにする。

        `select_action` は RTC を `assert` で止めるので、待ち行列をここで持つ。
        本家と同じ形（要素は `(batch, action_dim)`）にしてある。

        **手がかりに渡すのは「前の chunk のうち実行しなかった分」である。**
        chunk は 50 ステップぶん作られるのに実行するのは `n_action_steps` 歩だけなので、
        残りはそのまま「次に何をするつもりだったか」になる。

        `inference_delay` は 0 である。**この評価では推論中にロボットが動かない**
        （`/act` を待って 1 ステップ進む同期の作りなので、推論の間に消費されるステップ数が無い）。

        Parameters
        ----------
        batch : dict
            `_build_batch` が作った、モデルへ渡す観測。

        Returns
        -------
        action : Tensor
            形 `(batch, action_dim)` の 1 ステップぶんの行動。
        """
        if not self._rtc_queue:
            kwargs = {
                "inference_delay": 0,
                "execution_horizon": self._rtc_horizon,
            }
            if self._rtc_previous_tail is not None:
                kwargs["prev_chunk_left_over"] = self._rtc_previous_tail

            chunk = self._policy.predict_action_chunk(batch, **kwargs)
            executed = self._n_action_steps
            # 実行しない残りを次回の手がかりに取っておく。
            self._rtc_previous_tail = chunk[:, executed:, :].detach()
            self._rtc_queue.extend(chunk[:, :executed, :].transpose(0, 1))

        return self._rtc_queue.popleft()

    def _fall_back_from_rtc(self, error: Exception) -> None:
        """RTC で例外が出たら、以降は `blend_steps` の道へ戻す。

        **一度も採点環境で走っていない道なので、落ちても 0 点にしない。**
        `rtc_config.enabled` を偽にすると `select_action` の `assert` が通り、
        3.10.0 と同じ挙動になる。
        """
        self._rtc_enabled = False
        self._rtc_fell_back = True
        self._rtc_queue.clear()
        self._rtc_previous_tail = None
        if getattr(self._policy.config, "rtc_config", None) is not None:
            self._policy.config.rtc_config.enabled = False
        self._policy.reset()
        print(
            f"[parc] rtc 失敗のため blend_steps の道へ戻す: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        batch = self._build_batch(obs)

        with self._torch.no_grad():
            if self._rtc_enabled:
                try:
                    action = self._get_action_rtc(batch)
                except Exception as error:  # noqa: BLE001
                    self._fall_back_from_rtc(error)
                    action = self._policy.select_action(batch)
            else:
                action = self._policy.select_action(batch)

        action = self._postprocessor(action)
        action = np.asarray(
            action[0].detach().to("cpu").numpy(), dtype=np.float32
        ).reshape(7)
        return self._smooth(action)

    def _smooth(self, action: np.ndarray) -> np.ndarray:
        """姿勢 6 次元だけを均す。グリッパ（7 次元目）は素通しする。

        chunk 境界では、新しい観測から作り直した行動が前ステップと不連続に
        なりうる。これが jerk / SPARC を悪化させる。境界の直後だけ前の行動を
        混ぜて段差を消す。逆正規化統計は次のとおりで、次元ごとにスケールが
        大きく違うため、絶対量ではなく凸結合（スケール不変）で扱う。

        | 次元 | 中身 | min/max | std |
        |---|---|---|---|
        | 0-2 | EEF 並進の差分 | ±0.9375 | 0.34〜0.45 |
        | 3-5 | EEF 回転の差分 | ±0.36 | 0.04〜0.08 |
        | 6 | グリッパ | **ちょうど ±1.0** | **0.999** |

        6 次元目は min/max がちょうど ±1、std がほぼ 1 のバンバン制御である。
        均すと中間値になり、掴めなくなる。**触ってはいけない。**

        Parameters
        ----------
        action : np.ndarray
            逆正規化済みの行動 (7,)。

        Returns
        -------
        smoothed : np.ndarray
            後処理後の行動 (7,)。無効時は入力をそのまま返す。
        """
        # **倍率は EMA より前に掛ける。** EMA も継ぎ目の混ぜも線形なので
        # 出力は「後で掛ける」場合と厳密に同じだが、**前で掛けると入力と
        # `_previous_action` の単位がそろう。**「後で掛けて、掛けた値を
        # `_previous_action` に保存する」実装だと毎ステップ倍率が掛かって発散するので、
        # その罠を構造的に避けている。
        if self._gain_translation != 1.0 or self._gain_rotation != 1.0:
            action = action.copy()
            action[_TRANSLATION_DIMS] *= self._gain_translation
            action[_ROTATION_DIMS] *= self._gain_rotation
            # **各軸を独立に ±1 で切ると、進む向きが変わってしまう**
            # （環境が受け取れる範囲は ±1.0。配布物の乱数方策が
            # `np.random.uniform(-1, 1)` を返している）。3 軸をまとめて縮めて向きを保つ。
            for dims in (_TRANSLATION_DIMS, _ROTATION_DIMS):
                peak = float(np.max(np.abs(action[dims])))
                if peak > 1.0:
                    action[dims] = action[dims] / peak

        enabled = self._blend_steps > 0 or self._ema_alpha > 0.0

        if not enabled or self._previous_action is None:
            self._previous_action = action
            self._step_index += 1
            return action

        smoothed = action.copy()
        previous = self._previous_action

        if self._ema_alpha > 0.0:
            smoothed[_POSE_DIMS] = (
                self._ema_alpha * previous[_POSE_DIMS]
                + (1.0 - self._ema_alpha) * smoothed[_POSE_DIMS]
            )

        # chunk 境界（step_index が n_action_steps の倍数）の直後だけ効かせる。
        offset = self._step_index % self._n_action_steps
        if offset < self._blend_steps:
            weight = (self._blend_steps - offset) / (self._blend_steps + 1.0)
            smoothed[_POSE_DIMS] = (
                weight * previous[_POSE_DIMS] + (1.0 - weight) * smoothed[_POSE_DIMS]
            )

        self._previous_action = smoothed
        self._step_index += 1
        return smoothed

    def _normalize_mode(self) -> str:
        """採点ログに出す、正規化の構成。"""
        if self._normalizer is None:
            return "off"
        return "on+embed" if self._normalizer.uses_embeddings else "on"

    def reset(self, instruction: str = "") -> None:
        # **エピソードごとには何も出さない。** 指示文も、寄せた決め方も、点数も。
        #
        # 版 3.5.0 は点数まで出していた。**それが漏れになっていた。** 採点 8 本の
        # 点数が 0.851, 0.851, 0.819, 0.819, 0.972, 0.972, 0.902, 0.902 と並び、
        # **4 つの値が 2 回ずつ ＝ 4 タスク × 2 エピソード**と読めたうえ、
        # 手元の 10,030 変種と突き合わせれば**タスク名まで特定できてしまう。**
        # 運営 Q&A No.33 は「運営の採点(本番評価)で得られた観測・結果を利用する
        # こと」を禁じている。**特定できる状態そのものを残さない。**
        #
        # 版 3.6.0 では決め方（`lexical` など）だけ残していたが、**版 3.6.1 で
        # それも落とした。** 8 本すべてが `lexical` だったので本数は割れないが、
        # **「割れない」は今回そうだっただけで、次の採点でも割れない保証は無い。**
        #
        # **正規化が効いているかは起動時の 1 行で分かる**（`normalize=on`）ので、
        # 落として困らない。
        if self._normalizer is None:
            self._instruction = instruction
        else:
            self._instruction = self._normalizer.normalize(instruction)[0]

        # **種を渡されているときだけ、このエピソードの乱数を固定する。**
        # 既定（`PARC_SEED` 無し）では何もしない＝提出時の挙動は変わらない。
        # 何のためかは `__init__` の `PARC_SEED` の注記にある。
        if self._seed is not None:
            import torch

            seed = self._seed + self._episode_index
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32))
        self._episode_index += 1

        # chunk のキャッシュを捨てる。忘れると前エピソードの action を出し続ける。
        self._policy.reset()
        self._previous_action = None
        self._step_index = 0
        # RTC の持ち物も捨てる。**前エピソードの計画を手がかりにしてはいけない。**
        self._rtc_queue.clear()
        self._rtc_previous_tail = None

    def _build_batch(self, obs: dict[str, np.ndarray]) -> dict:
        """評価パイプラインの観測を、モデルが期待する batch へ変換する。"""
        observation = {
            "observation.images.front": self._to_image_tensor(obs["agentview_image"]),
            "observation.images.wrist": self._to_image_tensor(
                obs["robot0_eye_in_hand_image"]
            ),
            "observation.robot_state": {
                "eef": {
                    "pos": self._to_batched_tensor(obs["robot0_eef_pos"]),
                    "quat": self._to_batched_tensor(obs["robot0_eef_quat"]),
                },
                "gripper": {
                    "qpos": self._to_batched_tensor(obs["robot0_gripper_qpos"]),
                },
            },
            "task": [self._instruction],
        }

        # 画像の 180 度回転と 8 次元 state の組み立て。
        observation = self._env_preprocessor(observation)
        # rename / batch / tokenize / device / normalize。
        return self._preprocessor(observation)

    def _to_image_tensor(self, image: np.ndarray):
        """(H, W, 3) uint8 を (1, 3, H, W) float32 [0, 1] へ変換する。"""
        tensor = self._torch.from_numpy(np.ascontiguousarray(image))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(dtype=self._torch.float32) / self._CHW_SCALE

    def _to_batched_tensor(self, values: np.ndarray):
        """(N,) を (1, N) float32 へ変換する。"""
        tensor = self._torch.from_numpy(np.ascontiguousarray(values))
        return tensor.to(dtype=self._torch.float32).unsqueeze(0)

    def _warm_up(self) -> None:
        """ダミー入力で 1 回推論し、初回 /act が 10 秒に触れないようにする。

        起動 120 秒の枠を消費するが、超過するとトラック全体が 0 点になる
        /act の 10 秒制限を守る方が優先度が高い。
        """
        camera = 128
        dummy = {
            "agentview_image": np.zeros((camera, camera, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((camera, camera, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7, dtype=np.float32),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }
        self.reset(instruction="warm up")
        self.get_action(dummy)
        self.reset(instruction="")


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
