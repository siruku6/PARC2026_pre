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
from types import ModuleType
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "solution" / "tools"))

import rollout_recorder as recorder  # noqa: E402


def _load_summarizer() -> Any:
    """`summarize_gpu_eval.py` を読み込む。**`-` を含まないが実行形式なので直接読む。**

    戻り値の型を `Any` にしてある。`ModuleType` だと、型検査器が
    **`_parse_log` などの private な名前をすべて「無い」と言う**
    （動的に読んだモジュールの中身は静的には見えない）。
    """
    path = _ROOT / "solution" / "tools" / "summarize_gpu_eval.py"
    spec = importlib.util.spec_from_file_location("summarize_gpu_eval", path)
    assert spec is not None and spec.loader is not None, f"読み込めない: {path}"
    module: ModuleType = importlib.util.module_from_spec(spec)
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


def test_froze_label_is_counted(tmp_path: Path) -> None:
    """`固まった` が数えられる。**8/11 に正規表現から漏れていた。**

    lora20k の `eval.log` には 8 件あったのに、集計は `静止|往復` しか見て
    おらず「ラベル無し」に数えていた。**打ち切りの中身を読む唯一のラベル**
    なので、落とすと未完了 22 件の正体が分からなくなる。
    """
    # 前半で動いてから、終盤 50 歩だけ止まる軌跡。全体の変位では捕まらない。
    track = [(0.30 * index / 250, 0.0, 0.0) for index in range(251)]
    track += [(0.30, 0.0, 0.0)] * 50
    line = recorder.format_result_line(
        "task_light_1", 0, 300, False, track, [-1.0] * 301, None, {},
    )
    log = tmp_path / "eval.log"
    log.write_text("INFO:  " + line + "\n", encoding="utf-8")

    episode = summarize._parse_log(log)[0][0]

    assert episode["motion"] == "固まった"
    assert episode["tail_path_m"] == pytest.approx(0.0, abs=0.005)


def test_grip_toggles_are_read_back(tmp_path: Path) -> None:
    """`握り xN` が読み戻せる。**本番の悪化を検知できた唯一の指標である。**

    84 件の検証用データでは成功率に差が出なかった（0.476 対 0.500）のに、
    **グリッパを 20 回以上開閉した回は 13% から 27% へ増えていた。**
    """
    grips = [1.0 if index % 2 else -1.0 for index in range(56)]
    line = recorder.format_result_line(
        "task_tb_6", 0, 55, False, _straight(55, 0.58), grips, None, {},
    )
    log = tmp_path / "eval.log"
    log.write_text("INFO:  " + line + "\n", encoding="utf-8")

    episode = summarize._parse_log(log)[0][0]

    assert episode["grip_toggles"] == 55
    # **20 回以上なので「掴み直しで空回り」に数えられる側である。**
    assert episode["grip_toggles"] >= summarize._GRIP_CHATTER_THRESHOLD


def test_froze_is_absent_for_old_logs(tmp_path: Path) -> None:
    """終盤の記録が無いログでは `固まった率` を出さない。

    **0 を入れると「固まった回が無かった」と読めてしまう。**
    素の重みの回（8/11 午前）は録画側が古く、ラベル自体が存在しない。
    """
    log = tmp_path / "eval.log"
    log.write_text(
        "INFO:  [結果] task_light_1 ep0: 未完了 300 steps  ゴール 未達  "
        "経路 1.10m  変位 0.25m  握り x3\n",
        encoding="utf-8",
    )
    episodes = summarize._parse_log(log)[0]

    assert "tail_path_m" not in episodes[0]
    assert episodes[0]["grip_toggles"] == 3


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


# ------------------------------------------------- 衝突の切り分け（環境 / 方策）

def _episode(task: str, offenders: dict[str, float], **extra: object) -> dict:
    """`_parse_log` が返す形の 1 件を組み立てる。"""
    return {"task": task, "episode": 0, "outcome": "衝突", "steps": 300,
            "offenders": offenders, **extra}


def test_the_same_displacement_three_times_is_blamed_on_the_environment() -> None:
    """**3 回そろって初めて環境由来と見る。** 2 回では偶然がありうる。

    実例は 2026-08-10 の `ketchup_1=544.6mm` が 6 回そろった回である。
    摂動が違う回で変位が 0.1mm も違わないのは方策の振る舞いではない。
    """
    twice = [_episode(f"t{i}", {"ketchup_1": 544.6}) for i in range(2)]
    assert summarize._environment_offenders(twice) == set()

    thrice = [_episode(f"t{i}", {"ketchup_1": 544.6}) for i in range(3)]
    assert summarize._environment_offenders(thrice) == {("ketchup_1", "544.6")}


def test_a_slightly_different_displacement_stays_the_policys_fault() -> None:
    """**0.1mm でも違えば別扱い。** 方策がぶつけると同じ値にはならない。"""
    episodes = [
        _episode("t0", {"bowl_1": 8.5}),
        _episode("t1", {"bowl_1": 10.5}),
        _episode("t2", {"bowl_1": 9.5}),
    ]
    assert summarize._environment_offenders(episodes) == set()


def test_one_unique_offender_makes_the_episode_the_policys_fault() -> None:
    """環境由来と方策由来が同じ回に混ざったら、**方策由来として数える。**

    衝突対策で拾える回を取りこぼさないため、判定は「1 つでも方策由来が
    あるか」で行う。
    """
    episodes = [_episode(f"t{i}", {"ketchup_1": 544.6}) for i in range(3)]
    episodes.append(_episode("t3", {"ketchup_1": 544.6, "bowl_1": 12.3}))
    environment = summarize._environment_offenders(episodes)

    assert summarize._policy_offenders(episodes[0], environment) == {}
    assert summarize._policy_offenders(episodes[3], environment) == {"bowl_1": 12.3}


def test_collision_source_report_splits_the_two_rates() -> None:
    """報告に「方策の衝突率」と「環境を含む衝突率」の両方が出る。"""
    episodes = [_episode(f"t{i}", {"ketchup_1": 544.6}) for i in range(3)]
    episodes.append(_episode("t3", {"bowl_1": 12.3}))
    episodes.append(_episode("t4", {}, outcome="成功"))
    lines: list[str] = []
    summarize._report_collision_source(lines, episodes)
    report = "\n".join(lines)

    # **方策の衝突率だけを強調してある**（対策の効き目を測る数字なので）。
    assert "方策の衝突率      0.200**  (1/5)" in report
    assert "環境を含む衝突率    0.800  (4/5)" in report
    assert "ketchup_1=544.6mm" in report
    # 環境由来だけの回は 3 件（ケチャップの 3 本）、方策由来は 1 件。
    assert "環境由来だけ               3" in report
    assert "方策由来を含む           1**" in report


# --------------------------------------------------------- 並べて比べる

def test_comparison_names_the_rows_it_could_not_show() -> None:
    """**出せなかった行を黙って落とさない。**

    行が無いのを「差が無い」と読み違えるのが、いちばん危ない誤読である。
    滑らかさの指標は本番の悪化 8.8 倍のうち 4.4 倍を占めていた。
    """
    mine = {"n": 8, "success": 0.5, "jerk": 3.7, "by_category": {}}
    theirs = {"n": 8, "success": 0.25, "by_category": {}}   # jerk が無い
    lines: list[str] = []
    summarize._report_comparison(lines, mine, theirs, "相手")
    report = "\n".join(lines)

    assert "平均ジャーク" in report            # 出せなかった項目として名前が出る
    assert "**「差が無い」ではない。**" in report
    assert "+0.250" in report                  # 成功率の差は出ている


def test_comparison_says_which_direction_is_better() -> None:
    """**衝突率は下がるほうが良い。** 差の符号だけでは判断できない。"""
    assert "良くなった" in summarize._format_delta(0.1, 0.3, higher_is_better=False)
    assert "悪くなった" in summarize._format_delta(0.3, 0.1, higher_is_better=False)
    assert "良くなった" in summarize._format_delta(0.7, 0.5, higher_is_better=True)
    assert "同じ" in summarize._format_delta(0.5, 0.5, higher_is_better=True)


def test_padding_counts_full_width_characters_as_two() -> None:
    """表がずれないこと。`f"{'項目':<28}"` は文字数で詰めるので合わない。"""
    assert len(summarize._pad("項目", 28)) == 2 + 24      # 全角 2 文字 = 4 桁
    assert len(summarize._pad("abc", 28)) == 3 + 25
    assert summarize._pad("とても長い見出しなので詰められない", 4).endswith("い")
