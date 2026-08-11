"""1 エピソードの動きを、mp4 と 1 行の要約に残す。**採点ハーネス側に差し込んで使う。**

本番の 8 本中 4 本が、2 回とも 300 歩を使い切ってゴールに届いていない。
**止まっているのか、往復しているのか、掴めずに空振りしているのか——
成功率と衝突率では区別できない。** それを見るために撮る。

**映像と `[結果]` 行の両方をここで作る。** 動きの判定（`静止` / `往復`）に使う
閾値が 1 か所で済むからである。映像は既定で失敗した回・最大 40 本までしか
残らないので、**84 件すべての動きはログでしか分からない。** 2 つに分けて
書くと、判定の基準が食い違ったまま気づけない。

配布キットの `pipeline/rollout.py` の数か所へ
[patch_kit_for_variants.py](patch_kit_for_variants.py) が呼び出しを差し込み、
このファイル自身を `<kit>/pipeline/rollout_recorder.py` へ複製する。
**提出物（`solution/policy_server.py`）には一切手を入れない。** 採点環境は
ファイルを返さないので、提出物側に入れても映像は持ち帰れない。

**numpy が無くても読み込める。** `measure_motion` と `format_result_line` は
標準ライブラリだけで書いてあり、numpy と cv2 は使う関数の中で import する。
手元（numpy を入れていない環境）でも `[結果]` 行の書式を試験できる
（[tests/test_rollout_report_line.py](../../tests/test_rollout_report_line.py)）。

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
  - `[結果]` 行の文字列（書き出しは呼び出し側の logger）
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 観測のどのキーを、どの見出しで並べるか。左が広い視野、右が手先。
_CAMERAS: tuple[tuple[str, str], ...] = (
    ("agentview_image", "front"),
    ("robot0_eye_in_hand_image", "wrist"),
)

# 「どこへも行けていない」と見る変位の上限（m）。出発点からこれ未満。
_STALL_DISPLACEMENT_M = 0.03

# 変位が小さいまま、通った距離がこれを超えていれば「往復」（m）。
#
# **絶対値と比を両方使う。** どちらか片方だけでは取り違える。
#
# | 決め方 | 取り違え |
# |---|---|
# | 比だけ（経路長 > 10 × 変位） | 変位がほぼ 0 のとき常に真。**1 mm の震えも往復に見える** |
# | 絶対値だけ（経路長 > 0.5 m） | **60 歩で 0.32 m 行き来した回を「静止」と呼んだ**（実測） |
#
# 1 エピソードの経路長は 300 歩で平均 1.12 m（[05_実測結果.md]）なので、
# 0.10 m は「1 割ほど動いた」に当たり、震え（0.00 m）とは明確に分かれる。
_WANDER_PATH_M = 0.10

# 変位が上の下限を超えていても、経路長がその何倍もあれば往復と見る。
#
# **これが無いと、大きく往復した回が両方の網を抜ける。** 300 歩で 2.37 m
# 行き来して変位 0.03 m という回が「ラベル無し」になった（実測）。
# 目標へ向かって進んだ回は経路長と変位が同じ桁に収まるので、
# **8 倍は「行って戻った」を捉えるのに十分**である。
_WANDER_PATH_RATIO = 8.0

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


def _gripper_command(signal: Any) -> float:
    """1 歩ぶんの行動から、グリッパへの指令を取り出す。

    **行動ベクトルでも float でも受ける。** 映像側は貯める段で float に
    してしまっており、`[結果]` 行側は行動ベクトルをそのまま持っている。
    どちらからも同じ関数を通せるようにしておく。

    LIBERO の行動は 7 次元（並進 3・回転 3・グリッパ 1）で、**グリッパは
    最後の要素**である。指令は ±1 のバンバン制御なので、符号だけを見る。
    """
    try:
        return float(signal[-1])
    except (TypeError, IndexError, KeyError):
        return float(signal)


def _distance(start: Any, end: Any) -> float:
    """3 次元の 2 点間の距離。**numpy を使わない。**"""
    return math.sqrt(
        sum((float(end[axis]) - float(start[axis])) ** 2 for axis in range(3))
    )


def measure_motion(
    eef_positions: list[Any], gripper_signals: list[Any]
) -> dict[str, Any]:
    """手先の動きを数え、失敗の型を表すラベルを付ける。

    **経路長と変位を並べると、失敗の型が分かれる。**

    | 経路長 | 変位 | 読み方 | ラベル |
    |---|---|---|---|
    | 小 | 小 | その場で止まっている | `静止` |
    | 大 | 小 | 往復している・空振りしている | `往復` |
    | 大 | 大 | 動いてはいるが目標に届いていない | なし |

    **「大きく往復」を落とさないため、比も見る。** 変位が上限を超えていても、
    経路長がその 8 倍以上なら行って戻っただけである（実測で 2.37 m / 0.03 m
    という回が絶対値の網だけでは漏れた）。

    Parameters
    ----------
    eef_positions : list
        手先の位置の列。要素は長さ 3 で添字が引ければよい（numpy 配列・tuple）。
    gripper_signals : list
        各歩の行動ベクトルか、グリッパ指令の float の列。

    Returns
    -------
    motion : dict
        `path_m` / `displacement_m` / `grip_changes` / `note` を持つ辞書。
        `note` は `静止` `往復` または空文字列。
        **2 歩未満なら空の辞書**（何も言えないので、呼び出し側で省く）。
    """
    if len(eef_positions) < 2:
        return {}

    path_length = sum(
        _distance(start, end)
        for start, end in zip(eef_positions, eef_positions[1:])
    )
    displacement = _distance(eef_positions[0], eef_positions[-1])

    grips = [_gripper_command(signal) for signal in gripper_signals]
    grip_changes = sum(
        1 for before, after in zip(grips, grips[1:]) if (before > 0) != (after > 0)
    )

    note = ""
    if displacement < _STALL_DISPLACEMENT_M:
        note = "往復" if path_length > _WANDER_PATH_M else "静止"
    elif path_length > _WANDER_PATH_RATIO * displacement:
        note = "往復"

    return {
        "path_m": path_length,
        "displacement_m": displacement,
        "grip_changes": grip_changes,
        "note": note,
    }


def format_result_line(
    task_name: str,
    episode_id: int,
    total_steps: int,
    goal_reached: bool,
    eef_positions: list[Any],
    gripper_signals: list[Any],
    first_collision_step: int | None,
    offenders: dict[str, float],
) -> str:
    """1 エピソードの結末を 1 行にまとめる。

    **`ゴール` を結末と別に出す。** 採点の成功は「LIBERO の成功条件を
    満たし、**かつ**対象外の物体を 1 mm 超動かしていない」ことなので
    （[pipeline/rollout.py](../../pipeline/rollout.py) の `success = bool(done)
    and not collided`）、成功率だけを見ると
    **「課題ができない」のと「できたが物にぶつけた」の区別が付かない。**
    前者は視覚と到達の話、後者は軌道の話で、**打つ手が正反対になる。**

    **経路長と変位も全件に出す。** 映像は既定で失敗した回・最大 40 本までしか
    残らないので、84 件すべての「止まっているか往復しているか」はここでしか
    分からない。

    Parameters
    ----------
    task_name, episode_id, total_steps
        そのまま書き出す。
    goal_reached : bool
        LIBERO の成功条件を満たしたか（`env.step` の `done`）。
    eef_positions, gripper_signals
        `measure_motion` に渡すもの。
    first_collision_step : int | None
        初めて 1 mm を超えた歩。衝突していなければ None。
    offenders : dict[str, float]
        ぶつけた物体 → 動いた距離（m）。衝突していなければ空。

    Returns
    -------
    line : str
        `[結果] <タスク> ep<N>: <結末> <歩数> steps  ゴール 到達  経路 …` の 1 行。
        **先頭の書式は変えない。** [summarize_gpu_eval.py](summarize_gpu_eval.py) と
        [summarize_smoke_test.py](summarize_smoke_test.py) が読む。
    """
    outcome = "成功" if (goal_reached and not offenders) else (
        "衝突" if offenders else "未完了"
    )

    notes = [f"ゴール {'到達' if goal_reached else '未達'}"]

    motion = measure_motion(eef_positions, gripper_signals)
    if motion:
        notes.append(f"経路 {motion['path_m']:.2f}m")
        notes.append(f"変位 {motion['displacement_m']:.2f}m")
        notes.append(f"握り x{motion['grip_changes']}")
        if motion["note"]:
            notes.append(str(motion["note"]))

    if first_collision_step is not None:
        notes.append(f"初衝突 {first_collision_step} 歩")

    if offenders:
        notes.append("ぶつけた物体: " + ", ".join(
            f"{name}={value * 1000:.1f}mm"
            for name, value in sorted(offenders.items(), key=lambda pair: -pair[1])
        ))

    return (
        f"[結果] {task_name} ep{episode_id}: {outcome} {total_steps} steps  "
        + "  ".join(notes)
    )


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
        self._frames: list[Any] = []
        self._eef_positions: list[tuple[float, ...]] = []
        self._gripper_actions: list[float] = []
        self._camera_labels: list[str] = [label for _key, label in _CAMERAS]

    def add(self, obs: dict[str, Any], action: Any) -> None:
        """1 ステップ分のフレームを貯める。**例外を外へ出さない。**"""
        try:
            self._frames.append(self._compose(obs))
            position = obs.get("robot0_eef_pos")
            self._eef_positions.append(
                (0.0, 0.0, 0.0) if position is None
                else tuple(float(value) for value in position)
            )
            self._gripper_actions.append(_gripper_command(action))
        except Exception as error:                       # noqa: BLE001
            _disable(f"フレームの作成に失敗した: {error}")

    def _compose(self, obs: dict[str, Any]) -> Any:
        """2 つのカメラを横に並べ、等倍の 1 フレームを作る。

        **拡大も文字入れもここではやらない。** 300 歩ぶんをメモリに持つので、
        貯める段では 128 px のままにしておく。
        """
        import cv2
        import numpy as np

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
        """手先の動きを、映像に焼く 1 行にする。

        **数え方は `measure_motion` に置いてある。** `[結果]` 行と同じ関数を
        通すことで、映像のラベルとログのラベルが食い違わないようにする。
        ここでやるのは**日本語のラベルを ASCII に直すこと**だけである
        （キットが固定している opencv 4.11 の `putText` は日本語を `?` にする）。
        """
        motion = measure_motion(self._eef_positions, self._gripper_actions)
        if not motion:
            return ""
        return (
            f"path {motion['path_m']:.2f}m  disp {motion['displacement_m']:.2f}m"
            f"  grip x{motion['grip_changes']}"
            + _MOTION_LABELS.get(str(motion["note"]), "")
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
