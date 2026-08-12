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
#   1.0.0  ＋ LoRA 追加学習 20,000 歩（推論の設定は 0.3.0 と同じ）
#          → 本番 0.016。**0.3.0 の 0.1409 から 8.8 倍の悪化**
#   2.0.0  重みを λ=0.25 のブレンド（素 75% ＋ 1.0.0 の学習分 25%）に差し替え。
#          推論の設定は 0.3.0 / 1.0.0 と同じままなので、**0.1409 との差は重みだけ**。
#          目的はスコアではなく、1.0.0 の悪化がどこまで薄めれば消えるかを測ること。
#          観測の受け口は 1.0.0 と同じ（front/wrist・state 8・empty_cameras=0）
#          だが、**受け口の違いは推論では打ち消される**ので交絡しない。
#          → docs/reference/10_observation_interface.md
#          → **本番 0.09507。** ただし本番は 8 本のみで、うち 4 本は素・1.0.0・
#            2.0.0 すべてで 300 歩に張り付く。差は 1 本ぶんで、0.1409 との優劣は
#            判定できない。明確に違うのは 1.0.0（0.016）だけ。
#   3.0.0  重みを weak3x（苦手タスク 3 倍 ＋ 手本の 300 歩超を落とす学習）に差し替え。
#          推論の設定は 2.0.0 と同じなので、**2.0.0 との差は重みだけ**。
#          weak3x は 1.0.0 と同じ全学習（λ=1.0 相当）だが、手元 84 件で 1.0.0 の
#          滑らかさの劣化（ジャーク +13.6%・空回り 2 倍）が素の水準まで戻っている。
#          → これで本番崩壊の 2 説を切り分ける。0.14 付近なら「滑らかさが犯人・
#            もう直った」、0.016 付近なら「重みのずれ自体が毒（滑らかさは無関係）」。
#          → **本番 0.144（成功 3/8・係数 0.384）。** 素 0.1409 と同水準で H1 を裏づけ。
#            学習を含む（＝資格を満たす）重みが初めて素の水準に乗った。8/14 の土台。
#   3.1.0  weak3x のまま n_action_steps を 5 → 2 に下げる（重みは 3.0.0 と同じ）。
#          n の 50→10→5 が 0.058→0.096→0.1409 と単調改善だった延長。再計画が
#          2.5 倍の頻度になり、300 歩で終われない 4 本のどれかを崩せる見込み。
#          推論のみの変更＝minor。時間は n=2 でも約 585 秒（上限の 16%）。
#   3.2.0  weak3x のまま samples を 1 → 3（n は 5 に戻す＝3.0.0 から 1 変数）。
#          3 本引いて姿勢 6 次元の medoid を選ぶ。外れ chunk を捨て、滑らかさの
#          係数（3.0.0 で 0.384）を上げる向き。成功本数は動きにくい。時間は約 700 秒。
#          → **未提出。** n=2（3.1.0）が n=5 に勝ったため、n=5 の上で samples を
#            測る価値が下がった。samples は勝った n の上で測り直す。
#   3.1.0  → **本番 0.166。** weak3x 0.144 を上回り、資格ありの最高。天井の 1 本
#            （ep3）を 300→202 に解いた。n の 50→10→5→2 が単調改善の続き。
#   3.3.0  weak3x のまま n_action_steps を 1（3.1.0 からさらに 1 段。samples=1）。
#          毎ステップ再計画。単調改善（…→5→2）の次の点。EMA 0.3 でガタつきを抑える。
#          時間は約 1,040 秒（上限の 29%）。ここで反転すれば n=2 が谷底と分かる。
__version__ = "3.3.0"

import argparse
from abc import ABC, abstractmethod

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


class MyPolicy(BasePolicy):
    """同梱した LeRobot v0.6.0 で SmolVLA を動かす。

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
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.env_processor import LiberoProcessorStep

        self._torch = torch
        weights = here / "model_weights"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        config = PreTrainedConfig.from_pretrained(weights)
        config.device = self._device
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
        # **既定は 5。** 本番の実測 2 点から合計時間は
        # `環境 129.4 秒 + 推論 911.2 秒 / n` と分解でき、n=5 でも 312 秒
        # （上限 3,600 秒の 8.7%）にとどまる。n=1 ですら 1,041 秒で収まるため、
        # 時間制限は縮めない理由にならない。
        #
        # 小さくしすぎると、毎ステップ独立に flow matching の乱数を引くため
        # 軌道がガタつく恐れがある。`PARC_EMA_ALPHA` がそれを打ち消す。
        # 版 3.3.0 で既定を 1 に（n の単調改善の続き。詳細は先頭のコメント）。
        config.n_action_steps = int(os.environ.get("PARC_N_ACTION_STEPS", "1"))

        self._policy = SmolVLAPolicy.from_pretrained(weights, config=config)
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
        self._blend_steps = int(os.environ.get("PARC_BLEND_STEPS", "0"))
        self._ema_alpha = float(os.environ.get("PARC_EMA_ALPHA", "0.3"))
        self._n_action_steps = int(self._policy.config.n_action_steps)

        # chunk を何回引くか。1 なら素の挙動。
        self._samples = int(os.environ.get("PARC_SAMPLES", "1"))
        if self._samples > 1:
            self._install_chunk_sampler()

        self._previous_action: np.ndarray | None = None
        self._step_index = 0

        # **採点ログに構成を残す。** 以前は「どの設定で採点されたか」がログから
        # 分からず、所要時間から逆算して推定するしかなかった。ここに出しておけば
        # 提出物とスコアの対応が後から確実に辿れる。
        print(
            f"[parc] policy_server {__version__}  device={self._device}  "
            f"n_action_steps={self._n_action_steps}  ema_alpha={self._ema_alpha}  "
            f"blend_steps={self._blend_steps}  samples={self._samples}",
            flush=True,
        )

        self._instruction = ""
        self._warm_up()

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

        `select_action` は `predict_action_chunk` ではなく
        `_get_action_chunk` を直接呼ぶため、包むのはそちらである。
        """
        policy = self._policy
        torch = self._torch
        original = policy._get_action_chunk
        count = self._samples

        def sampled_chunk(batch, noise=None, **kwargs):
            chunks = torch.stack(
                [original(batch, noise, **kwargs) for _ in range(count)]
            )
            # 距離は姿勢 6 次元だけで測る。グリッパは ±1 に振れるため、
            # 含めるとそこだけで距離が決まってしまう。
            flat = chunks[..., _POSE_DIMS].reshape(count, -1)
            distances = torch.cdist(flat, flat).sum(dim=1)
            return chunks[int(distances.argmin())]

        policy._get_action_chunk = sampled_chunk

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        batch = self._build_batch(obs)

        with self._torch.no_grad():
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

    def reset(self, instruction: str = "") -> None:
        self._instruction = instruction
        # chunk のキャッシュを捨てる。忘れると前エピソードの action を出し続ける。
        self._policy.reset()
        self._previous_action = None
        self._step_index = 0

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
