#!/usr/bin/env python3
"""配布キットを、検証用スイートで測れる状態に書き換える。

**キットは変種のうち 6,287 件を開けない。** カメラ視点・ロボット初期姿勢・
観測ノイズの摂動は変種名そのものに数値として埋め込まれており、その名前の
bddl は存在しない（`libero/libero/envs/env_wrapper.py` が `_view_` で切って
元タスクの bddl を開く仕組みになっている）。キットの 2 か所がこれを知らない。

| 直す場所 | 何が起きていたか |
|---|---|
| `compe/t1/register.py` | 初期状態ファイルを正規表現で探す。摂動の付け方はカテゴリごとに違うため取りこぼす |
| `pipeline/environment.py` | 衝突判定のため bddl を自分で開く。`_view_` を切らないので `FileNotFoundError` |
| `pipeline/rollout.py` | エピソードごとの結末を出さない。**何にぶつかったのかが分からない** |
| `pipeline/rollout.py` | **ゴール到達と成功を分けない。** 「できない」のと「できたがぶつけた」が同じ 0 に見える |
| `pipeline/rollout.py` | 映像を残さない。**打ち切りに張り付いた回で何が起きたか見られない** |

2 番目の例外はトラック全体を止めるが、`pipeline.py` がトラックごとに握って
`_errored_track_score` を書くため、**終了コード 0 で正常終了に見える。**
2026-08-10 に 84 タスクのはずが 3 タスクで終わり、気づけなかった。

3 番目は落ちないが、**成功・衝突・未完了の区別が付かない。** 配布キットは
タスク単位の成功率しか出さず、衝突判定に使った `object_max_disp`
（どの物体がどれだけ動いたか）を判定の直後に捨てる。判定式には触らず、
そのあとに `logger.info` を 1 つ足す。

4 番目は打つ手を左右する。採点の成功は「LIBERO の成功条件を満たし、**かつ**
対象外の物体を 1mm 超動かしていない」ことなので、成功率だけでは
**視覚と到達の問題なのか、軌道の問題なのかが分からない。** ゴール到達
（`done`）・初めて 1mm を超えた歩・手先の経路長と変位を同じ行に出す。
**行の中身は [rollout_recorder.py](rollout_recorder.py) の
`format_result_line` にある**（ここに書式を置くと、直すたびにパッチを
差し替えることになる）。

5 番目は数字では埋まらない穴である。**本番の 8 本中 4 本が 300 ステップを
使い切って届いていない**のに、止まっているのか往復しているのか掴めずに
空振りしているのかが分からない。録画の本体も
[rollout_recorder.py](rollout_recorder.py) にあり、ここから
`pipeline/` へ複製する。**`PARC_RECORD_DIR` が空なら完全に無効**なので、
当てたままにしておいて構わない。

**何度実行してもよい。** すでに当たっていれば「適用済み」と出て何もしない。
**中身を変えた版を、古い版が当たったままのキットへ当てても壊れない。**
古い版は `_OBSOLETE` として剥がしてから当て直す（下）。

    python3 solution/tools/patch_kit_for_variants.py --kit-dir /workspace

Reads:
  - <kit-dir>/compe/t1/register.py
  - <kit-dir>/pipeline/environment.py
  - <kit-dir>/pipeline/rollout.py
  - solution/tools/rollout_recorder.py
Generates:
  - 上の 3 ファイル（上書き）
  - <kit-dir>/pipeline/rollout_recorder.py（複製）
"""

from __future__ import annotations

import argparse
from pathlib import Path

# `[結果]` 行を出す `logger` 呼び出しの目印。**1 本だけであることを最後に確かめる。**
# 2 本になると集計（summarize_gpu_eval.py）が同じエピソードを二重に数える。
_RESULT_LINE_MARKER = '"  %s", _format_result('

# (対象ファイル, 説明, 置換前, 置換後)
_PATCHES: tuple[tuple[str, str, str, str], ...] = (
    (
        "compe/t1/register.py",
        "初期状態ファイルを CSV の指定から取る",
        '            init_states_file=f"{r[\'task_id\']}.pruned_init",',
        '            init_states_file=r.get("init_states_file")\n'
        '            or f"{r[\'task_id\']}.pruned_init",',
    ),
    (
        "compe/t1/register.py",
        "初期状態が別階層にある摂動（物体を増やすもの）を開けるようにする",
        "def _init_states_for(task, get_libero_path):\n"
        "    fn = task.init_states_file\n"
        "    root = get_libero_path(\"init_states\")\n"
        "    stem, ext = os.path.splitext(fn)\n"
        "    stripped = _SUFFIX_RE.sub(\"\", stem) + ext\n"
        "    p = os.path.join(root, task.problem_folder, stripped)\n"
        "    return torch.load(p, weights_only=False)",
        "def _init_states_for(task, get_libero_path):\n"
        "    fn = task.init_states_file\n"
        "    root = get_libero_path(\"init_states\")\n"
        "    # CSV が `/` を含む相対パスを書いていれば、接尾辞を剥がさず\n"
        "    # そのまま開く。**物体が増える摂動（`_add_` / `_level`）は\n"
        "    # 変種名そのものの初期状態を持っており**、元タスクの状態を\n"
        "    # 当てると自由度が合わずに落ちる。\n"
        "    #   ValueError: could not broadcast input array\n"
        "    #               from shape (23,) into shape (49,)\n"
        "    # 本家の benchmark/__init__.py も `_add_` と `_level` のときだけ\n"
        "    # `libero_newobj/<suite>/` を見て、読んだあと reshape している\n"
        "    # （`get_task_init_states`）。**problem_folder には畳めない** —\n"
        "    # あちらは bddl の置き場としても使われるため。\n"
        "    if \"/\" in fn:\n"
        "        states = torch.load(os.path.join(root, fn), weights_only=False)\n"
        "        return states.reshape(1, -1)\n"
        "    stem, ext = os.path.splitext(fn)\n"
        "    stripped = _SUFFIX_RE.sub(\"\", stem) + ext\n"
        "    p = os.path.join(root, task.problem_folder, stripped)\n"
        "    return torch.load(p, weights_only=False)",
    ),
    (
        "pipeline/rollout.py",
        "1 タスクの例外でトラック全体を止めない",
        "            result = self.evaluate_task(policy, task_info, perturbation)\n"
        "            results.append(result)",
        "            # **1 タスクの例外で残り全部を捨てない。**\n"
        "            # ここが素のままだと、例外が evaluate_tasks を抜けて\n"
        "            # pipeline.py の握りに入り、そのトラックが 0 点になる。\n"
        "            # 2026-08-10 に 84 タスクの 26 件目で落ち、**残り 58 件が\n"
        "            # 1 度も試されなかった**（1 時間の測定が無駄になった）。\n"
        "            try:\n"
        "                result = self.evaluate_task(policy, task_info, perturbation)\n"
        "            except Exception as error:\n"
        "                logger.error(\n"
        '                    "  [結果] %s: 例外で測れず (%s: %s)",\n'
        "                    task_info.name, type(error).__name__, error,\n"
        "                )\n"
        "                continue\n"
        "            results.append(result)",
    ),
    (
        "pipeline/environment.py",
        "衝突判定の bddl を `_view_` より前で探す",
        "def parse_obj_of_interest(bddl_file: str | Path) -> set[str]:\n"
        "    txt = Path(bddl_file).read_text()",
        "def parse_obj_of_interest(bddl_file: str | Path) -> set[str]:\n"
        "    # LIBERO-plus の 6,287 変種は自分の名前の bddl を持たない。摂動は\n"
        "    # 変種名に数値として埋め込まれており、実体は `_view_` より前にある\n"
        "    # （envs/env_wrapper.py が環境生成時に同じ切り方をしている）。\n"
        "    path = Path(bddl_file)\n"
        '    if not path.is_file() and "_view_" in path.name:\n'
        '        path = path.with_name(path.name.split("_view_")[0] + ".bddl")\n'
        "    txt = path.read_text()",
    ),
    (
        "pipeline/rollout.py",
        "初めて 1mm を超えた歩を覚える枠を用意する",
        "        object_max_disp: dict[str, float] = {}\n",
        "        object_max_disp: dict[str, float] = {}\n"
        "        # **初めて 1mm を超えた歩**を覚える。衝突したかどうかだけでは\n"
        "        # 打つ手が決まらない。掴む前（前半）に当てているなら近づき方の話、\n"
        "        # 置くとき（後半）なら手離しの話である。\n"
        "        first_collision_step: int | None = None\n",
    ),
    (
        "pipeline/rollout.py",
        "1mm を超えた歩を記録する",
        "            for name, p0 in object_init_pos.items():\n"
        "                cur = obs.get(name + \"_pos\")\n"
        "                if cur is not None:\n"
        "                    d = float(np.sum(np.abs(np.asarray(cur) - p0)))\n"
        "                    if d > object_max_disp.get(name, 0.0):\n"
        "                        object_max_disp[name] = d\n",
        "            for name, p0 in object_init_pos.items():\n"
        "                cur = obs.get(name + \"_pos\")\n"
        "                if cur is not None:\n"
        "                    d = float(np.sum(np.abs(np.asarray(cur) - p0)))\n"
        "                    if d > object_max_disp.get(name, 0.0):\n"
        "                        object_max_disp[name] = d\n"
        "            # 判定と同じ式で、**最初に超えた歩だけ**を覚える。物体は 3 個\n"
        "            # 前後なので費用は無視できる（超えたあとは条件式に入らない）。\n"
        "            if first_collision_step is None and any(\n"
        "                d > collision_threshold for d in object_max_disp.values()\n"
        "            ):\n"
        "                first_collision_step = step + 1\n",
    ),
    (
        "pipeline/rollout.py",
        "エピソードごとの結末・ゴール到達・動き・ぶつけた物体をログに出す",
        "        collided = any(d > collision_threshold for d in object_max_disp.values())\n",
        "        collided = any(d > collision_threshold for d in object_max_disp.values())\n"
        "        # 判定式には触らず、その直後に 1 行足すだけにする。\n"
        "        # object_max_disp は判定の直後に捨てられるため、**何にぶつかったのかは\n"
        "        # ここで書き出さないと永久に分からない。**\n"
        "        #\n"
        "        # **行の中身は rollout_recorder.format_result_line にある。**\n"
        "        # ここに書式を持たせると、直すたびにこのパッチを差し替えることに\n"
        "        # なり、当たったままのキットに二重適用する危険が出る。\n"
        "        # 要約に失敗しても評価は続ける（**90 分をログの整形で落とさない**）。\n"
        "        try:\n"
        "            from .rollout_recorder import format_result_line as _format_result\n"
        "            logger.info(\"  %s\", _format_result(\n"
        "                task_info.name, episode_id, total_steps, bool(done),\n"
        "                ee_positions, actions_log, first_collision_step,\n"
        "                {\n"
        "                    k: v for k, v in object_max_disp.items()\n"
        "                    if v > collision_threshold\n"
        "                },\n"
        "            ))\n"
        "        except Exception as _summary_error:\n"
        "            logger.warning(\n"
        '                "  [結果] %s ep%d: %s %d steps  （要約に失敗: %s）",\n'
        "                task_info.name, episode_id,\n"
        '                "衝突" if collided else ("成功" if done else "未完了"),\n'
        "                total_steps, _summary_error,\n"
        "            )\n",
    ),
    (
        "pipeline/rollout.py",
        "録画を始める（PARC_RECORD_DIR が空なら何もしない）",
        "        start_time = time.time()\n"
        "        joint_positions: list[np.ndarray] = []\n",
        "        # **映像を残す。** 打ち切りに張り付いた回で、止まっているのか\n"
        "        # 往復しているのか掴めずに空振りしているのかは、成功率と\n"
        "        # 衝突率では区別できない。`PARC_RECORD_DIR` が空なら無効で、\n"
        "        # ここは None を返すだけである（以降の呼び出しも起きない）。\n"
        "        # import に失敗しても評価は続ける。**録画のために 90 分の\n"
        "        # 測定を落とさない。**\n"
        "        try:\n"
        "            from .rollout_recorder import begin_episode as _begin_recording\n"
        "            _recorder = _begin_recording(task_info.name, episode_id)\n"
        "        except Exception:\n"
        "            _recorder = None\n"
        "        start_time = time.time()\n"
        "        joint_positions: list[np.ndarray] = []\n",
    ),
    (
        "pipeline/rollout.py",
        "1 ステップごとに、方策が見た画像と出した行動を貯める",
        "            action = self.env_manager.apply_action_noise(action, perturbation)\n",
        "            action = self.env_manager.apply_action_noise(action, perturbation)\n"
        "\n"
        "            # **方策が実際に見た観測**（摂動を掛けたあと）を渡す。\n"
        "            # env.step のあとの obs は次の歩のものなので、ここで貯める。\n"
        "            if _recorder is not None:\n"
        "                _recorder.add(obs_for_policy, action)\n",
    ),
    (
        "pipeline/rollout.py",
        "結末が決まってから mp4 を書く",
        "        return EpisodeResult(\n"
        "            task_name=task_info.name,\n",
        "        # 結末をファイル名と映像に焼くため、書き出しはここまで待つ。\n"
        "        # **既定では失敗した回だけ残す**（PARC_RECORD_WHICH=all で全部）。\n"
        "        if _recorder is not None:\n"
        "            _recorder.finalize(\n"
        '                "成功" if success else ("衝突" if collided else "未完了"),\n'
        "                {\n"
        "                    k: v for k, v in object_max_disp.items()\n"
        "                    if v > collision_threshold\n"
        "                },\n"
        "            )\n"
        "\n"
        "        return EpisodeResult(\n"
        "            task_name=task_info.name,\n",
    ),
)

# パッチだけでは足りず、キットへ丸ごと置く必要があるファイル。
# `pipeline/` の中に置くのは、パッチが `from .rollout_recorder import` と
# 相対 import で読むためである。
_COPIES: tuple[tuple[str, str], ...] = (
    ("rollout_recorder.py", "pipeline/rollout_recorder.py"),
)

# 剥がしてから当て直すもの。**(対象ファイル, 説明, 古い版の差し込み文)**
#
# **キットは使い回される。** `run_gpu_eval.sh` は `work/eval-kit` が在れば
# 作り直さないので、パッチの中身を変えると「古い版が当たったキット」に
# 新しい版を当てることになる。置換前の目印（`collided = any(...)`）は
# 置換後にも残るため、`_apply` は**二重に当ててしまう。** その結果
# `[結果]` 行が 1 エピソードにつき 2 本出て、集計が二重に数える。
#
# 消す文は 1 文字も違ってはいけないので、**残っていないことを最後に
# `_RESULT_LINE_MARKER` の本数で確かめる**（そちらは文字列一致に頼らない）。
_OBSOLETE: tuple[tuple[str, str, str], ...] = (
    (
        "pipeline/rollout.py",
        "2026-08-11 より前の `[結果]` 行（書式をこのファイルが持っていた版）",
        "        # 判定式には触らず、その直後に 1 行足すだけにする。\n"
        "        # object_max_disp は判定の直後に捨てられるため、**何にぶつかったのかは\n"
        "        # ここで書き出さないと永久に分からない。**\n"
        "        _offenders = {\n"
        "            k: v for k, v in object_max_disp.items() if v > collision_threshold\n"
        "        }\n"
        "        logger.info(\n"
        '            "  [結果] %s ep%d: %s %d steps%s",\n'
        "            task_info.name, episode_id,\n"
        '            "成功" if (bool(done) and not _offenders)\n'
        '            else ("衝突" if _offenders else "未完了"),\n'
        "            total_steps,\n"
        '            ("  ぶつけた物体: " + ", ".join(\n'
        '                f"{k}={v * 1000:.1f}mm"\n'
        "                for k, v in sorted(_offenders.items(), key=lambda kv: -kv[1])\n"
        '            )) if _offenders else "",\n'
        "        )\n",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kit-dir", type=Path, required=True,
                        help="配布キットの場所（Colab なら /content/PARC2026_pre）")
    return parser.parse_args()


def is_applied(kit_dir: Path) -> bool:
    """キットにパッチが当たっているかを返す。

    **旗で受け取らずに実物を見る**ために公開している。「当てたつもり」で
    当たっていない状態を検査側が信じてしまうと、82 分の評価が 4 行目で
    落ちるのを見逃す。判定には置換後の文字列そのものを使うので、
    パッチの定義と食い違うことがない。

    **録画のファイル（`_COPIES`）は判定に入れない。** 測れるかどうかを
    左右しないので、無くても「当たっている」と答える。
    """
    return all(
        (kit_dir / relative_path).is_file()
        and after in (kit_dir / relative_path).read_text(encoding="utf-8")
        for relative_path, _, _, after in _PATCHES
    )


def _copy(source: Path, destination: Path) -> str:
    """付随ファイルをキットへ置き、何をしたかを返す。

    **中身が同じでも上書きする。** 手元で録画のコードを直したのに、キットに
    古いものが残っていると、直したはずの挙動が出ないまま原因を別の場所に
    探しに行くことになる（F セルの版番号と同じ理由）。

    **無くても例外にしない。** 録画は任意の機能であり、Colab では
    `/content/` への置き忘れが起こりうる。ここで落とすと、**評価の下準備
    （手順 2）ごと止まって本題の測定ができなくなる。** 差し込んだ側は
    import に失敗しても `None` を返すので、録画されないだけで済む。
    """
    if not source.is_file():
        return (f"**無い**  {source.name} を置けなかった。"
                "録画は無効のまま評価は走る")
    existed = destination.is_file()
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return ("置き換えた" if existed else "置いた  ") + f"  {destination.name}"


def _apply(path: Path, description: str, before: str, after: str) -> str:
    """1 か所を書き換え、何をしたかを返す。"""
    if not path.is_file():
        raise FileNotFoundError(f"{path} が見つかりません")

    source = path.read_text(encoding="utf-8")
    if after in source:
        return f"適用済み  {description}"
    if before not in source:
        raise RuntimeError(
            f"{path} にアンカーが見つかりません。キットの版を確認してください。\n"
            f"  探した文字列: {before.splitlines()[0]!r}"
        )
    path.write_text(source.replace(before, after, 1), encoding="utf-8")
    return f"適用した  {description}"


def _strip(path: Path, description: str, obsolete: str) -> str:
    """古い版の差し込みを剥がし、何をしたかを返す。

    **無ければ何もしない。** 新しく作ったキットには当たっていないので、
    見つからないのが普通である。
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} が見つかりません")

    source = path.read_text(encoding="utf-8")
    if obsolete not in source:
        return f"無し      {description}"
    path.write_text(source.replace(obsolete, "", 1), encoding="utf-8")
    return f"剥がした  {description}"


def _verify_single_result_line(kit_dir: Path) -> None:
    """`[結果]` 行を出す箇所が 1 つだけであることを確かめる。

    **文字列一致に頼らない最後の関門である。** `_OBSOLETE` の文が 1 文字でも
    違えば古い版は残り、`[結果]` 行が 1 エピソードに 2 本出る。集計は行を
    数えるので、**成功率は変わらないまま件数が倍になり**、84 件のはずが
    168 件になって「そろった」と読めてしまう。
    """
    path = kit_dir / "pipeline/rollout.py"
    count = path.read_text(encoding="utf-8").count(_RESULT_LINE_MARKER)
    if count == 1:
        return
    raise RuntimeError(
        f"{path} の `[結果]` 行が {count} 箇所ある（1 であるべき）。\n"
        "  古い版が残っている恐れがあります。キットを作り直してください:\n"
        f"    rm -rf {kit_dir}"
    )


def _refuse_own_repository(kit_dir: Path) -> None:
    """自分のリポジトリを書き換えようとしたら止める。

    **`compe/` と `pipeline/` は配布キットのファイルであり、提出物では
    未改変のまま保つ必要がある。** このリポジトリにも同じ 3 ファイルが
    追跡されているので、`--kit-dir .` のような打ち間違いで汚れてしまう。
    実際の呼び出し先（Docker の `/workspace`、Colab の clone）は
    使い捨てなので、この制限に当たらない。

    **この台本は `/content/` へ 1 枚だけコピーして使われる**（Colab）。
    そのとき `parents` は `/content` と `/` の 2 つしかないので、
    `parents[2]` は `IndexError` になる。**リポジトリの外に置かれている
    ということは、守るべきリポジトリも無い**ので、その場合は何もしない。
    """
    parents = Path(__file__).resolve().parents
    if len(parents) < 3:
        return                                   # リポジトリの外に置かれている
    repository_root = parents[2]
    if kit_dir.resolve() == repository_root:
        raise SystemExit(
            f"このリポジトリ自身（{repository_root}）は書き換えません。\n"
            "  配布キットは使い捨ての場所に置いてください"
            "（Docker なら /workspace、Colab なら /content/PARC2026_pre）。"
        )


def main() -> None:
    args = _parse_args()
    _refuse_own_repository(args.kit_dir)
    print(f"[patch] キット: {args.kit_dir}")
    # **剥がすのを先にやる。** 古い版が残っていると、置換前の目印が
    # そのまま在るために新しい版が二重に当たる。
    for relative_path, description, obsolete in _OBSOLETE:
        result = _strip(args.kit_dir / relative_path, description, obsolete)
        print(f"  {result}  ({relative_path})")
    for relative_path, description, before, after in _PATCHES:
        result = _apply(args.kit_dir / relative_path, description, before, after)
        print(f"  {result}  ({relative_path})")
    tools_dir = Path(__file__).resolve().parent
    for source_name, destination in _COPIES:
        result = _copy(tools_dir / source_name, args.kit_dir / destination)
        print(f"  {result}  ({destination})")
    _verify_single_result_line(args.kit_dir)
    print("[patch] 完了")


if __name__ == "__main__":
    main()
