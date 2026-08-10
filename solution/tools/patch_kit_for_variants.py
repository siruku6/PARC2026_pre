#!/usr/bin/env python3
"""配布キットを、LIBERO-plus の摂動つき変種でも動くように書き換える。

**キットは変種のうち 6,287 件を開けない。** カメラ視点・ロボット初期姿勢・
観測ノイズの摂動は変種名そのものに数値として埋め込まれており、その名前の
bddl は存在しない（`libero/libero/envs/env_wrapper.py` が `_view_` で切って
元タスクの bddl を開く仕組みになっている）。キットの 2 か所がこれを知らない。

| 直す場所 | 何が起きていたか |
|---|---|
| `compe/t1/register.py` | 初期状態ファイルを正規表現で探す。摂動の付け方はカテゴリごとに違うため取りこぼす |
| `pipeline/environment.py` | 衝突判定のため bddl を自分で開く。`_view_` を切らないので `FileNotFoundError` |

後者の例外はトラック全体を止めるが、`pipeline.py` がトラックごとに握って
`_errored_track_score` を書くため、**終了コード 0 で正常終了に見える。**
2026-08-10 に 84 タスクのはずが 3 タスクで終わり、気づけなかった。

**何度実行してもよい。** すでに当たっていれば「適用済み」と出て何もしない。

    python3 solution/tools/patch_kit_for_variants.py --kit-dir /workspace

Reads:
  - <kit-dir>/compe/t1/register.py
  - <kit-dir>/pipeline/environment.py
Generates:
  - 同じ 2 ファイル（上書き）
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
