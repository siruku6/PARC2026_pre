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

2 番目の例外はトラック全体を止めるが、`pipeline.py` がトラックごとに握って
`_errored_track_score` を書くため、**終了コード 0 で正常終了に見える。**
2026-08-10 に 84 タスクのはずが 3 タスクで終わり、気づけなかった。

3 番目は落ちないが、**成功・衝突・未完了の区別が付かない。** 配布キットは
タスク単位の成功率しか出さず、衝突判定に使った `object_max_disp`
（どの物体がどれだけ動いたか）を判定の直後に捨てる。判定式には触らず、
そのあとに `logger.info` を 1 つ足す。

**何度実行してもよい。** すでに当たっていれば「適用済み」と出て何もしない。

    python3 solution/tools/patch_kit_for_variants.py --kit-dir /workspace

Reads:
  - <kit-dir>/compe/t1/register.py
  - <kit-dir>/pipeline/environment.py
  - <kit-dir>/pipeline/rollout.py
Generates:
  - 同じ 3 ファイル（上書き）
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
        "エピソードごとの結末と、ぶつけた物体をログに出す",
        "        collided = any(d > collision_threshold for d in object_max_disp.values())\n",
        "        collided = any(d > collision_threshold for d in object_max_disp.values())\n"
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


def main() -> None:
    args = _parse_args()
    print(f"[patch] キット: {args.kit_dir}")
    for relative_path, description, before, after in _PATCHES:
        result = _apply(args.kit_dir / relative_path, description, before, after)
        print(f"  {result}  ({relative_path})")
    print("[patch] 完了")


if __name__ == "__main__":
    main()
