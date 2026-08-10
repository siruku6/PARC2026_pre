"""ロールアウトを mp4 に撮る。**採点ハーネス側に差し込んで使う。**

本番の 8 本中 4 本が、2 回とも 300 歩を使い切ってゴールに届いていない。
**止まっているのか、往復しているのか、掴めずに空振りしているのか——
成功率と衝突率では区別できない。** それを見るために撮る。

配布キットの `pipeline/rollout.py` の 3 か所へ
[patch_kit_for_variants.py](patch_kit_for_variants.py) が呼び出しを差し込み、
このファイル自身を `<kit>/pipeline/rollout_recorder.py` へ複製する。
**提出物（`solution/policy_server.py`）には一切手を入れない。** 採点環境は
ファイルを返さないので、提出物側に入れても映像は持ち帰れない。

環境変数で切り替える。**`PARC_RECORD_DIR` が空なら完全に無効**で、
`add()` は即座に返る。

| 変数 | 既定 | 意味 |
|---|---|---|
| `PARC_RECORD_DIR` | （空） | 出力先。**空なら録画しない** |
| `PARC_RECORD_WHICH` | `failures` | `failures`＝失敗だけ／`all`＝全部 |
| `PARC_RECORD_FPS` | `20` | `control_freq` が 20 なので既定で実時間 |
| `PARC_RECORD_SCALE` | `3` | 128 px を何倍に伸ばすか（文字を読むため） |
| `PARC_RECORD_MAX` | `40` | 撮る本数の上限。ディスクの保険 |

使い方（コンテナ内・評価を回すシェルで）::

    export PARC_RECORD_DIR=/out/videos
    bash solution/tools/run_local_eval.sh

**撮れる映像はモデルに渡るのと同じ向き**（180 度回転後）である。評価環境の
生画像は上下逆で、回転して初めて学習データと揃う。詳細は
`docs/strategy/05_実測結果.md`「観測変換の正しさ」。

**評価に相乗りさせて構わない費用である**（手元の実測）。

| 測ったもの | 値 |
|---|---|
| 1 歩の上乗せ | **0.33 ms**（本番の `env.step` 39 ms に対して 0.8%） |
| メモリの山 | 31 MB（300 歩・2 カメラ） |
| 書き出し | 1 エピソードにつき約 2 秒 |
| 84 タスク全部撮った場合 | 90 分の測定に **+2.9 分** |

Reads:
  - 環境変数（上の表）
  - 採点ハーネスが渡す観測（`agentview_image` / `robot0_eye_in_hand_image`）
Generates:
  - <PARC_RECORD_DIR>/<結末>_<タスク名>_ep<N>_<歩数>steps.mp4
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 観測のどのキーを、どの見出しで並べるか。左が広い視野、右が手先。
_CAMERAS: tuple[tuple[str, str], ...] = (
    ("agentview_image", "front"),
    ("robot0_eye_in_hand_image", "wrist"),
)

# 出発点からこれ未満しか離れていなければ、どこへも行けていないと見る（m）。
_STALL_DISPLACEMENT_M = 0.03

# そのうえで、通った距離がこれ未満なら「静止」、超えていれば「往復」と呼ぶ（m）。
# **比だけで決めると誤る。** 変位がほぼ 0 のとき「経路長 > 10 × 変位」は
# 常に真になり、1 mm の震えしかない回まで往復に見えてしまう。
_WANDER_PATH_M = 0.5

_TEXT_COLOR = (255, 255, 255)
_SHADOW_COLOR = (0, 0, 0)
_OUTCOME_COLORS: dict[str, tuple[int, int, int]] = {
    # cv2 は BGR で書く。
    "成功": (80, 200, 80),
    "衝突": (60, 60, 230),
    "未完了": (60, 170, 240),
}

# **映像に焼く文字は ASCII に限る。** キットが固定している
# opencv-python-headless 4.11.0.86 の `putText` は Hershey フォントしか持たず、
# 日本語を 1 文字ずつ `?` に置き換える（実測。「衝突」→「??????」）。
# ファイル名とログは UTF-8 のまま日本語で出す — そちらは化けない。
_OUTCOME_LABELS: dict[str, str] = {
    "成功": "SUCCESS",
    "衝突": "COLLIDED",
    "未完了": "TIMEOUT",
}
_MOTION_LABELS: dict[str, str] = {
    "静止": "  stalled",
    "往復": "  wandering",
}


def _is_enabled() -> bool:
    return bool(os.environ.get("PARC_RECORD_DIR", "").strip())


class _Recorder:
    """1 エピソード分のフレームを貯め、結末が分かってから mp4 にする。

    **書き出しを最後に回すのは、結末をファイル名と映像に焼くためである。**
    「未完了」だけ見たいときに、開いて確かめる手間をなくす。

    **貯めるのは 128 px のまま**で、拡大と文字入れは書き出すときにやる。
    3 倍に拡大してから持つと 300 歩で 265 MB になり、評価と同じマシンで
    並べて回すには重い。等倍なら 29 MB で収まる（実測）。
    """

    def __init__(self, task_name: str, episode_id: int, output_dir: Path):
        self._task_name = task_name
        self._episode_id = episode_id
        self._output_dir = output_dir
        self._frames: list[np.ndarray] = []
        self._eef_positions: list[np.ndarray] = []
        self._gripper_actions: list[float] = []
        self._camera_labels: list[str] = [label for _key, label in _CAMERAS]

    def add(self, obs: dict[str, np.ndarray], action: np.ndarray) -> None:
        """1 ステップ分のフレームを貯める。**例外を外へ出さない。**"""
        try:
            self._frames.append(self._compose(obs))
            self._eef_positions.append(
                np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64).copy()
            )
            self._gripper_actions.append(float(np.asarray(action).reshape(-1)[6]))
        except Exception as error:                       # noqa: BLE001
            _disable(f"フレームの作成に失敗した: {error}")

    def _compose(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """2 つのカメラを横に並べ、等倍の 1 フレームを作る。

        **拡大も文字入れもここではやらない。** 300 歩ぶんをメモリに持つので、
        貯める段では 128 px のままにしておく。
        """
        import cv2

        panels = []
        # **実際に入っていたカメラだけ**を覚える。片方が欠けた観測でも、
        # 見出しが panel とずれないようにする。
        self._camera_labels = []
        for key, label in _CAMERAS:
            image = obs.get(key)
            if image is None:
                continue
            self._camera_labels.append(label)
            # 評価環境の生画像は上下逆。回すとモデルへ渡るものと同じ向きになる
            # （LiberoProcessorStep の `torch.flip(img, dims=[2,3])` と一致する
            # ことを確認済み）。
            panel = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[::-1, ::-1])
            panels.append(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        if not panels:
            raise KeyError(f"画像のキーが観測に無い: {sorted(obs)}")
        return np.hstack(panels)

    def finalize(self, outcome: str, offenders: dict[str, float]) -> Path | None:
        """結末が決まったので mp4 を書く。撮らない回は None を返す。

        Parameters
        ----------
        outcome : str
            「成功」「衝突」「未完了」のいずれか。
        offenders : dict[str, float]
            ぶつけた物体 → 動いた距離（m）。衝突以外では空。

        Returns
        -------
        path : Path | None
            書き出した mp4。撮らなかった場合は None。
        """
        try:
            return self._write(outcome, offenders)
        except Exception as error:                       # noqa: BLE001
            _disable(f"mp4 の書き出しに失敗した: {error}")
            return None

    def _write(self, outcome: str, offenders: dict[str, float]) -> Path | None:
        import cv2

        which = os.environ.get("PARC_RECORD_WHICH", "failures").strip().lower()
        if which != "all" and outcome == "成功":
            return None
        if not self._frames:
            return None

        global _written
        limit = int(os.environ.get("PARC_RECORD_MAX", "40"))
        if _written >= limit:
            if _written == limit:
                logger.info(
                    "  [録画] 上限 %d 本に達した。以降は撮らない"
                    "（PARC_RECORD_MAX で変えられる）", limit,
                )
                _written += 1                            # この案内を 1 度だけ出す
            return None

        # 動きの要約。**止まっているのか往復しているのかは、経路長と変位の
        # 食い違いで分かる。**
        summary = self._motion_summary()

        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / (
            f"{outcome}_{self._task_name[:70]}"
            f"_ep{self._episode_id}_{len(self._frames)}steps.mp4"
        )

        scale = max(1, int(os.environ.get("PARC_RECORD_SCALE", "3")))
        height, width = self._frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(os.environ.get("PARC_RECORD_FPS", "20")),
            (width * scale, height * scale),
        )
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter が開けない: {path}")
        try:
            for index, frame in enumerate(self._frames):
                writer.write(
                    self._annotate(frame, index, outcome, offenders, summary, scale)
                )
        finally:
            writer.release()

        _written += 1
        logger.info(
            "  [録画] %s  %s", path.name, summary,
        )
        return path

    def _annotate(
        self,
        frame: np.ndarray,
        index: int,
        outcome: str,
        offenders: dict[str, float],
        summary: str,
        scale: int,
    ) -> np.ndarray:
        """等倍のフレームを拡大し、歩数・グリッパ・結末を焼く。

        **映像だけでは掴む瞬間が分からない。** グリッパの指令は ±1 の
        バンバン制御なので、閉じているかどうかを文字で出す。拡大を文字入れの
        前に済ませるのは、**引き伸ばした文字がぼやけないようにする**ためである。
        """
        import cv2

        annotated = cv2.resize(
            frame, (frame.shape[1] * scale, frame.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        # カメラの見出しは、並べた各パネルの左上に置く。
        panel_width = annotated.shape[1] // len(self._camera_labels)
        for offset, label in enumerate(self._camera_labels):
            _put_text(annotated, label, (6 + offset * panel_width, 18))

        total = len(self._frames)
        closing = self._gripper_actions[index] > 0
        _put_text(
            annotated,
            f"{index + 1:3d}/{total}  grip {'CLOSE' if closing else 'open '}",
            (6, annotated.shape[0] - 30),
        )
        _put_text(annotated, summary, (6, annotated.shape[0] - 10))

        # 結末は最後の 1 秒だけ、色を変えて大きく出す。
        fps = int(os.environ.get("PARC_RECORD_FPS", "20"))
        if index >= total - fps:
            text = _OUTCOME_LABELS.get(outcome, outcome)
            if offenders:
                worst = max(offenders.items(), key=lambda kv: kv[1])
                text += f" {worst[0]} {worst[1] * 1000:.0f}mm"
            cv2.putText(
                annotated, text, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                _OUTCOME_COLORS.get(outcome, _TEXT_COLOR), 2, cv2.LINE_AA,
            )
        return annotated

    def _motion_summary(self) -> str:
        """手先の動きを 1 行にまとめる。

        **経路長と変位を並べると、失敗の型が分かれる。**

        | 経路長 | 変位 | 読み方 |
        |---|---|---|
        | 小 | 小 | その場で止まっている |
        | 大 | 小 | 往復している・空振りしている |
        | 大 | 大 | 動いてはいるが目標に届いていない |
        """
        if len(self._eef_positions) < 2:
            return ""
        track = np.asarray(self._eef_positions)
        path_length = float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())
        displacement = float(np.linalg.norm(track[-1] - track[0]))
        grips = sum(
            1 for a, b in zip(self._gripper_actions, self._gripper_actions[1:])
            if (a > 0) != (b > 0)
        )
        note = ""
        if displacement < _STALL_DISPLACEMENT_M:
            note = "往復" if path_length > _WANDER_PATH_M else "静止"
        return (
            f"path {path_length:.2f}m  disp {displacement:.2f}m  grip x{grips}"
            + _MOTION_LABELS.get(note, "")
        )


def _put_text(frame: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """影を付けて 1 行書く。**影が無いと明るい背景で読めない。**"""
    import cv2

    for color, thickness in ((_SHADOW_COLOR, 3), (_TEXT_COLOR, 1)):
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            color, thickness, cv2.LINE_AA,
        )


# 撮った本数と、無効化した理由。**評価を止めないため、失敗しても例外にしない。**
_written = 0
_disabled_reason: str | None = None


def _disable(reason: str) -> None:
    """録画を諦める。**理由は 1 度だけ出す。**

    90 分の評価の途中で録画が壊れても、**評価そのものは完走させる。**
    毎ステップ同じ警告を吐くとログが埋まって `[結果]` 行が読めなくなる。
    """
    global _disabled_reason
    if _disabled_reason is None:
        _disabled_reason = reason
        logger.warning("  [録画] 中止した: %s（評価は続ける）", reason)
    os.environ["PARC_RECORD_DIR"] = ""


def begin_episode(task_name: str, episode_id: int) -> Any:
    """エピソードの録画を始める。無効なら None を返す。

    `pipeline/rollout.py` の `_run_episode` の先頭から呼ばれる。
    **返り値が None のときは `add()` も呼ばれない**ので、無効時の費用は
    この関数の env 参照 1 回だけである。

    Parameters
    ----------
    task_name : str
        タスク（変種）の名前。ファイル名に使う。
    episode_id : int
        エピソード番号。

    Returns
    -------
    recorder : _Recorder | None
        録画する場合は記録器。しない場合は None。
    """
    if not _is_enabled():
        return None
    try:
        import cv2                                       # noqa: F401
    except Exception as error:                           # noqa: BLE001
        _disable(f"cv2 を import できない: {error}")
        return None
    return _Recorder(task_name, episode_id, Path(os.environ["PARC_RECORD_DIR"]))
