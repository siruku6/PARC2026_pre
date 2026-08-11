"""`[結果]` 行を、書く側と読む側の両方から確かめる。

**書式が 1 か所で決まっていても、読み手がずれれば意味が無い。**
[rollout_recorder.py](../solution/tools/rollout_recorder.py) の
`format_result_line` が書いた行を、
[summarize_gpu_eval.py](../solution/tools/summarize_gpu_eval.py) の
`_parse_log` にそのまま食わせて、項目が復元できることを見る。

**採点環境でしか分からないことを、手元で確かめられるようにするための試験である。**
90 分の測定を回してから「集計が読めていなかった」と気づくと、その回は
数字が出ない（2026-08-10 に 84 件のうち 26 件で止まったのと同じ損である）。

動きのラベル（`静止` / `往復`）の 5 例は、実物の映像で確かめた値である
（`docs/talk/open_items.md` の「動きの判定は 5 通りで確かめた」）。
**閾値を触ったらここが落ちる。**
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "solution" / "tools"))

import rollout_recorder as recorder  # noqa: E402


def _load_summarizer() -> object:
    """`summarize_gpu_eval.py` を読み込む。**`-` を含まないが実行形式なので直接読む。**"""
    path = _ROOT / "solution" / "tools" / "summarize_gpu_eval.py"
    spec = importlib.util.spec_from_file_location("summarize_gpu_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = _load_summarizer()


def _line(steps: int, track: list, grips: list) -> str:
    """1 エピソード分の行を作る。書式は本物の関数に任せる。"""
    return recorder.format_result_line(
        "pick_up_the_black_bowl_light_49", 0, steps, True,
        track, grips, 108, {"ketchup_1": 0.5446},
    )


def _straight(steps: int, distance: float) -> list:
    """まっすぐ distance だけ進む軌跡。**点は steps + 1 個**になる。"""
    return [(distance * index / steps, 0.0, 0.0) for index in range(steps + 1)]


# --------------------------------------------------------- 動きのラベル

@pytest.mark.parametrize("path_m,displacement_m,expected", [
    (0.00, 0.00, "静止"),      # 震えだけ
    (0.32, 0.02, "往復"),      # 小さく往復
    (2.37, 0.03, "往復"),      # 大きく往復（比の条件でだけ捕まる）
    (0.29, 0.29, ""),          # まっすぐ進む
    (0.30, 0.24, ""),          # 進んで少し戻る
])
def test_motion_label_matches_measured_cases(
    path_m: float, displacement_m: float, expected: str,
) -> None:
    """実物で確かめた 5 例と同じラベルが付く。"""
    # 行って戻る軌跡を作る。往路 (path + disp) / 2、復路 (path - disp) / 2。
    forward = (path_m + displacement_m) / 2
    track = [(0.0, 0.0, 0.0), (forward, 0.0, 0.0), (displacement_m, 0.0, 0.0)]
    motion = recorder.measure_motion(track, [0.0, 0.0, 0.0])
    assert motion["path_m"] == pytest.approx(path_m, abs=1e-9)
    assert motion["displacement_m"] == pytest.approx(displacement_m, abs=1e-9)
    assert motion["note"] == expected


def test_motion_needs_two_steps() -> None:
    """1 歩では何も言わない。**0 と言わせない。**"""
    assert recorder.measure_motion([(0.0, 0.0, 0.0)], [0.0]) == {}


def test_gripper_command_accepts_vector_and_float() -> None:
    """行動ベクトルでも float でも、最後の要素をグリッパとして読む。"""
    assert recorder._gripper_command([0, 0, 0, 0, 0, 0, -1.0]) == -1.0
    assert recorder._gripper_command(1.0) == 1.0


def test_grip_changes_counts_sign_flips() -> None:
    """握り直した回数は符号の変わった回数である。"""
    motion = recorder.measure_motion(
        _straight(4, 0.4), [-1.0, -1.0, 1.0, 1.0, -1.0],
    )
    assert motion["grip_changes"] == 2


# --------------------------------------------------------- 書式と読み取り

def test_written_line_is_parsed_back(tmp_path: Path) -> None:
    """書いた行から、結末・ゴール到達・動き・初衝突・ぶつけた物体が戻る。"""
    log = tmp_path / "eval.log"
    log.write_text(
        "=== タスク 1/1: x ===\n"
        "INFO:  " + _line(137, _straight(137, 1.10), [[0] * 6 + [-1]] * 137) + "\n",
        encoding="utf-8",
    )

    episodes, exceptions, declared = summarize._parse_log(log)

    assert (len(episodes), exceptions, declared) == (1, [], 1)
    episode = episodes[0]
    assert episode["task"] == "pick_up_the_black_bowl_light_49"
    assert episode["outcome"] == "衝突"           # ゴールに着いたが失格
    assert episode["steps"] == 137
    assert episode["goal"] is True
    assert episode["path_m"] == pytest.approx(1.10, abs=0.005)
    assert episode["displacement_m"] == pytest.approx(1.10, abs=0.005)
    assert episode["first_collision_step"] == 108
    assert episode["offenders"] == {"ketchup_1": 544.6}


def test_motion_label_survives_the_round_trip(tmp_path: Path) -> None:
    """`静止` が付いた行を読み戻せる。**警告の材料になる項目である。**"""
    line = recorder.format_result_line(
        "task_tb_1", 0, 300, False,
        [(0.0, 0.0, 0.0), (0.001, 0.0, 0.0)], [1.0, 1.0], None, {},
    )
    log = tmp_path / "eval.log"
    log.write_text("INFO:  " + line + "\n", encoding="utf-8")

    episode = summarize._parse_log(log)[0][0]

    assert episode["motion"] == "静止"
    assert episode["outcome"] == "未完了"
    assert "first_collision_step" not in episode
    assert episode["offenders"] == {}


def test_old_log_has_no_new_keys(tmp_path: Path) -> None:
    """2026-08-11 より前の行も読める。**無い項目は鍵ごと無い。**"""
    log = tmp_path / "eval.log"
    log.write_text(
        "INFO:  [結果] task_light_1 ep0: 未完了 300 steps\n", encoding="utf-8"
    )

    episode = summarize._parse_log(log)[0][0]

    assert episode["steps"] == 300
    for key in ("goal", "path_m", "motion", "first_collision_step"):
        assert key not in episode


def test_goal_is_estimated_from_steps_for_old_logs() -> None:
    """記録が無ければ、打ち切りに届いたかどうかで推定する。"""
    assert summarize._goal_reached({"steps": 137}, 300) is True
    assert summarize._goal_reached({"steps": 300}, 300) is False
    # 記録があるほうを優先する（歩数と食い違っても記録が正しい）。
    assert summarize._goal_reached({"steps": 300, "goal": True}, 300) is True


# --------------------------------------------------------- 摂動カテゴリ

def test_tail_suffix_wins_over_a_table_in_the_middle() -> None:
    """`_table_` を名前の途中に含む `_tb_` の変種を取り違えない。

    LIBERO-plus の 2,218 件のうち 23 件がこの形で、含むかどうかで見ると
    `table` 613 / `tb` 463 が 636 / 440 にずれる。
    """
    name = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_tb_1"
    assert summarize._appearance_suffix(name) == "tb"
    assert summarize._category_of(name, {}) == "Background Textures"


def test_view_and_initstate_names_are_not_guessed() -> None:
    """視点・指示文・初期姿勢は名前から分けられない。**当てずに「不明」と言う。**"""
    name = "turn_on_the_stove_view_0_0_100_2_352_initstate_1"
    assert summarize._category_of(name, {}) == summarize._AMBIGUOUS_CATEGORY


def test_csv_wins_over_the_name() -> None:
    """CSV があればそちらを使う（分類表が唯一の正解である）。"""
    name = "turn_on_the_stove_view_0_0_100_2_352_initstate_1"
    known = {name: "Language Instructions"}
    assert summarize._category_of(name, known) == "Language Instructions"
