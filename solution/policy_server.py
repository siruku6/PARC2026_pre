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
        # 既定はチェックポイントの値（50）で、chunk を丸ごと開ループで実行する。
        # 縮めると再計画の頻度が上がって誤差の蓄積に強くなる代わりに、
        # 推論回数が増えてトラック全体 1 時間の制限に効く。
        # 環境変数が無ければ一切触らないため、既定の挙動は変わらない。
        n_action_steps = os.environ.get("PARC_N_ACTION_STEPS")
        if n_action_steps:
            config.n_action_steps = int(n_action_steps)

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

        self._instruction = ""
        self._warm_up()

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        batch = self._build_batch(obs)

        with self._torch.no_grad():
            action = self._policy.select_action(batch)

        action = self._postprocessor(action)
        return np.asarray(
            action[0].detach().to("cpu").numpy(), dtype=np.float32
        ).reshape(7)

    def reset(self, instruction: str = "") -> None:
        self._instruction = instruction
        # chunk のキャッシュを捨てる。忘れると前エピソードの action を出し続ける。
        self._policy.reset()

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
