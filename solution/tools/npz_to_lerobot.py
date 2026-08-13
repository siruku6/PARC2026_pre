#!/usr/bin/env python3
"""台本で作った軌道を、そのまま学習に載せられる形式に変換する。

[scripted_demo.py](scripted_demo.py) の `record` が書くのは npz である。
**シミュレータを動かす環境に lerobot を入れずに済ませる**ためで、変換は
lerobot が入っている側（Colab か学習用の venv）でこちらが行う。

出来上がりは `lerobot/libero_plus` と**同じ規約の LeRobot v3.0 データセット**
である（fps 20・画像 256×256・`observation.state` 8 次元・`action` 7 次元）。
形式を手で書き起こさず `LeRobotDataset` に作らせているので、列の並びや
`meta/` の中身がずれることがない。

    # 1. 作った分だけで 1 つのデータセットにする
    python3 solution/tools/npz_to_lerobot.py build \\
        --npz-dir outputs/scripted --output outputs/scripted_lerobot \\
        --repo-id local/scripted_moka

    # 2. 学習が読めるかを確かめる（**流す前にここで捕まえる**）
    python3 solution/tools/npz_to_lerobot.py inspect \
        --root outputs/scripted_lerobot --repo-id local/scripted_moka

    # 3. 学習を回す（作った分だけ。**提出物にはならない**）
    lerobot-train --dataset.repo_id=local/scripted_moka \\
        --dataset.root=outputs/scripted_lerobot ...

    # 4a. 混ぜる相手と列の書き方を揃える（**lerobot の版が違うと食い違う**）
    python3 solution/tools/npz_to_lerobot.py align \\
        --base ~/.cache/huggingface/lerobot/lerobot/libero_plus \\
        --add outputs/scripted_lerobot

    # 4b. 既存の学習データに混ぜる（提出候補を作るときはこちら）
    python3 solution/tools/npz_to_lerobot.py merge \\
        --base ~/.cache/huggingface/lerobot/lerobot/libero_plus \\
        --add outputs/scripted_lerobot \\
        --output outputs/libero_plus_with_scripted

**作った分だけで学習したものは提出できない。** 1 タスクぶんの手本しか
無いので、残り 39 タスクの性能が落ちる（実際 8/11 の追加学習は本番
0.1409 → 0.016 に崩れた）。**`build` は「台本を真似られるか」を見るため、
`merge` が提出候補を作るためのもの**である。

Reads:
  - <npz-dir>/*.npz（`scripted_demo.py record` の出力）
Generates:
  - <output>/ … LeRobot v3.0 のデータセット一式（data/ videos/ meta/）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# npz の取り決めは書き手と同じファイルに置いてある。**写しを作らない。**
# `scripted_demo` は lerobot を import しないので、ここから読んで安全である。
#
# **標準入力から流し込んで実行すると `__file__` が使えない**
# （`docker exec -i ... python - <引数> < この file` の形）。その場合は
# 道具の置き場を PYTHONPATH に入れて渡す。
if Path(__file__).suffix == ".py":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from scripted_demo import SavedEpisode  # noqa: E402
except ModuleNotFoundError as error:  # pragma: no cover - 起動の仕方の問題
    raise SystemExit(
        f"{error}\n  scripted_demo.py が import できない。"
        "**道具の置き場を PYTHONPATH に入れて実行すること**\n"
        "  例: PYTHONPATH=/path/to/solution/tools python - check ... < npz_to_lerobot.py"
    ) from error

# `lerobot/libero_plus` の `meta/info.json` と同じ宣言にする。
# **ここがずれると、混ぜたときに列が合わずに読めない。**
FPS = 20
IMAGE_SHAPE = (256, 256, 3)
STATE_DIM = 8
ACTION_DIM = 7
ROBOT_TYPE = "panda"

FEATURES: dict[str, dict[str, Any]] = {
    "observation.images.front": {
        "dtype": "video", "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist": {
        "dtype": "video", "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32", "shape": (STATE_DIM,),
        "names": [f"state_{i}" for i in range(STATE_DIM)],
    },
    "action": {
        "dtype": "float32", "shape": (ACTION_DIM,),
        "names": [f"action_{i}" for i in range(ACTION_DIM)],
    },
}


def _check_shapes(episode: SavedEpisode) -> None:
    """宣言した形と食い違っていないか確かめる。

    **食い違ったまま書くと、学習時ではなく読み込み時に落ちる。**
    どの npz が悪いのかが分からなくなるので、ここで名前を付けて止める。

    画像はヘッダだけ読む。**画素を伸長しないので、50 本でも一瞬で済む。**
    """
    name = episode.path.name
    length = len(episode.state)
    for key, expected in (("state", (STATE_DIM,)), ("action", (ACTION_DIM,))):
        array = getattr(episode, key)
        if len(array) != length:
            raise SystemExit(
                f"{name}: {key} のコマ数が {len(array)}、state は {length}")
        if tuple(array.shape[1:]) != tuple(expected):
            raise SystemExit(f"{name}: {key} の形が {tuple(array.shape[1:])}、"
                             f"宣言は {tuple(expected)}")
    for key in ("front", "wrist"):
        shape, dtype = episode.image_spec(key)
        if shape[0] != length:
            raise SystemExit(
                f"{name}: {key} のコマ数が {shape[0]}、state は {length}")
        if tuple(shape[1:]) != IMAGE_SHAPE:
            raise SystemExit(f"{name}: {key} の形が {tuple(shape[1:])}、"
                             f"宣言は {IMAGE_SHAPE}")
        if dtype != np.uint8:
            raise SystemExit(f"{name}: {key} が uint8 でない（{dtype}）")


def command_check(args: argparse.Namespace) -> None:
    """npz が変換先の宣言と合っているかだけを、lerobot 無しで確かめる。

    **`build` は Python 3.12 が要る**（lerobot が PEP 695 の型構文を使うため、
    3.10 のシミュレータ環境では構文エラーになる）。**書き出した直後に形の
    食い違いを見つけられないと、Colab まで持って行ってから落ちる。**
    そこで、形と型と長さの確認だけをここで済ませられるようにしてある。

    `--against` に本物の `meta/info.json` を渡すと、**宣言そのものを
    突き合わせる**（HuggingFace から取ってきたものでよい）。

    Reads:
      - <npz-dir>/*.npz
      - <against>（任意。混ぜたい相手の meta/info.json）
    """
    paths = sorted(args.npz_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"npz が 1 つも無い: {args.npz_dir}")

    mismatched: list[str] = []
    if args.against is not None:
        reference = json.loads(args.against.read_text())["features"]
        print(f"=== 宣言の突き合わせ（{args.against}）===")
        for name, feature in FEATURES.items():
            other = reference.get(name)
            if other is None:
                print(f"  {name}: **相手に無い**")
                mismatched.append(name)
                continue
            same = tuple(other["shape"]) == tuple(feature["shape"])
            if not same:
                mismatched.append(name)
            print(f"  {name:28s} {tuple(feature['shape'])} 対 "
                  f"{tuple(other['shape'])}  {'一致' if same else '**不一致**'}")

    print(f"\n=== npz {len(paths)} 本 ===")
    total, kept = 0, 0
    for path in paths:
        episode = SavedEpisode(path)
        _check_shapes(episode)
        total += len(episode.state)
        kept += episode.success
        shape, dtype = episode.image_spec("front")
        print(f"  {path.name}: {len(episode.state)} コマ  "
              f"成功 {episode.success}  "
              f"state {episode.state.dtype} {episode.state.shape[1:]}  "
              f"action {episode.action.dtype} {episode.action.shape[1:]}  "
              f"画像 {dtype} {tuple(shape[1:])}")

    if mismatched:
        # **「食い違いは無い」と出して正常終了してはいけない。**
        # ここで止めないと、merge まで進んでから落ちる。
        raise SystemExit(
            f"\n宣言が相手と食い違っている: {', '.join(mismatched)}\n"
            "  **FEATURES を相手の meta/info.json に合わせること。**")
    print(f"\n形の食い違いは無い。成功 {kept}/{len(paths)} 本・{total} コマ。")
    print("**書き出しは Python 3.12 の環境で `build` を実行する。**")


def command_build(args: argparse.Namespace) -> None:
    """npz を集めて 1 つの LeRobot データセットにする。

    Generates:
      - <output>/data/ <output>/videos/ <output>/meta/
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    paths = sorted(args.npz_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"npz が 1 つも無い: {args.npz_dir}")
    _prepare_output(args.output)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, features=FEATURES,
        root=args.output, robot_type=ROBOT_TYPE, use_videos=True,
    )

    total_frames, written = 0, 0
    for path in paths:
        episode = SavedEpisode(path)
        # **画像に触る前に成否を見る。** 捨てる回のために 118 MB を
        # 伸長するのは丸損である（半分失敗なら 50 本で 2.9 GB）。
        if not episode.success and not args.keep_failures:
            print(f"  {path.name}: 失敗した回なので入れない")
            continue
        _check_shapes(episode)
        front, wrist = episode.front, episode.wrist
        state = episode.state.astype(np.float32, copy=False)
        action = episode.action.astype(np.float32, copy=False)
        for index in range(len(state)):
            dataset.add_frame({
                "observation.images.front": front[index],
                "observation.images.wrist": wrist[index],
                "observation.state": state[index],
                "action": action[index],
                "task": episode.instruction,
            })
        dataset.save_episode()
        total_frames += len(state)
        written += 1
        print(f"  {path.name}: {len(state)} コマ  「{episode.instruction}」")

    dataset.finalize()
    # **本数は自分で数えた値を出す。** `meta.episodes` は読み出しのときだけ
    # 遅延で読み込まれる欄で、書き込みモードでは `None` のままである
    # （`len(None)` で落ちた）。確定した宣言は `_print_summary` が
    # 書き出された `meta/info.json` から読む。
    print(f"\n{written} 本 / {total_frames} コマ → {args.output}")
    _print_summary(args.output)


def command_inspect(args: argparse.Namespace) -> None:
    """書き出したデータセットを lerobot に読ませて、中身を 1 コマ確かめる。

    **ファイルが揃っていることと、学習が読めることは別である。**
    `build` が終わっても、動画の符号化が壊れていれば読み出しで初めて落ちる
    （LeRobot の自動フォールバックは torchcodec が「入っているか」しか見ない）。
    **学習を流す前にここで捕まえる。**

    Reads:
      - <root>/ … build か merge が書いたデータセット
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    meta = dataset.meta
    print(f"=== {args.root} ===")
    print(f"  {meta.total_episodes} 本 / {meta.total_frames} コマ / "
          f"{meta.total_tasks} タスク  fps {meta.fps}")
    print(f"  指示文: {sorted(meta.tasks.index.tolist())[:5]}")

    print("\n--- 動画 ---")
    for path in sorted((args.root / "videos").rglob("*.mp4")):
        print(f"  {path.relative_to(args.root)}  "
              f"{path.stat().st_size / 1e6:.1f} MB")

    print("\n--- 1 コマ目を実際に読む（ここが通れば学習も読める）---")
    item = dataset[0]
    for key in sorted(item):
        value = item[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else "-"
        print(f"  {key:34s} {str(getattr(value, 'dtype', type(value).__name__)):16s} {shape}")

    for name, feature in FEATURES.items():
        if name not in item:
            print(f"  **{name} が読み出しに出てこない**")
            continue
        got = tuple(item[name].shape)
        want = tuple(feature["shape"])
        # 画像は (C, H, W) で返る。**宣言の (H, W, C) と並びが違う。**
        if feature["dtype"] == "video":
            want = (want[2], want[0], want[1])
        if got != want:
            print(f"  **{name} の形が {got}、宣言は {want}**")


def _prepare_output(path: Path) -> None:
    """書き出し先を空けておく。

    **lerobot は書き出し先が既にあると作らない**（`mkdir(exist_ok=False)`）。
    ところが `docker run -v "$PWD/work/merged:/merged"` と書くと、**docker が
    先に作ってしまう**ので必ず `FileExistsError` になる。空なら消して先へ進む。

    消せないときは、その場所が**マウントの取り付け先そのもの**である
    （コンテナの中からは外せない）。**親を渡して、その下の未作成の名前を
    書き出し先にする**しかないので、そう伝える。
    """
    if not path.exists():
        return
    if any(path.iterdir()):
        raise SystemExit(
            f"{path} に中身がある。**消すか、別の名前を渡すこと。**")
    try:
        path.rmdir()
    except OSError as error:
        raise SystemExit(
            f"{path} は空だが消せない（{error}）。\n"
            "  **docker の -v がこの場所を直接指している。** 親を渡して、\n"
            "  書き出し先はその下の未作成の名前にする:\n"
            '    -v "$PWD/work:/work" … --output /work/merged') from error
    print(f"  空の {path} を消した（docker が作ったもの）")


def _feature_shape(feature: dict[str, Any]) -> tuple:
    """比べるべき中身だけを取り出す。**飾りは見ない。**"""
    return (feature.get("dtype"), tuple(feature.get("shape", ())),
            tuple(feature.get("names") or ()))


def command_align(args: argparse.Namespace) -> None:
    """混ぜる相手と `features` の書き方を揃える。

    **中身が同じでも、書き方が違うと混ぜられない。** `merge_datasets` は
    `features` の辞書が一致することを求める（動画の符号化設定だけは無視する）。
    ところが lerobot は版が上がるたびに飾りを変えるので、**新しい版で作った
    データセットは、古い版で作られた `lerobot/libero_plus` と一致しない。**
    実際に食い違ったのは 2 つで、どちらも意味は同じである。

    | | こちら（v0.6.0 で作った） | libero_plus（古い版） |
    |---|---|---|
    | 非動画の列 | `fps` が無い | `'fps': 20` がある |
    | 動画の `info` | `is_depth_map` | `video.is_depth_map` |

    そこで**相手の `features` をそのまま写す。** 写す前に
    **`dtype`・`shape`・`names` が一致することを確かめる**ので、
    本当に形が違うときは止まる。

    Reads:
      - <base>/meta/info.json（写す元）
      - <add>/meta/info.json（写す先）
    Generates:
      - <add>/meta/info.json（書き換える）
      - <add>/meta/info.json.bak（書き換える前のもの）
    """
    base_info = json.loads((args.base / "meta" / "info.json").read_text())
    add_path = args.add / "meta" / "info.json"
    add_info = json.loads(add_path.read_text())

    for key in ("fps", "robot_type"):
        if base_info.get(key) != add_info.get(key):
            raise SystemExit(
                f"{key} が違う: こちら {add_info.get(key)}、相手 {base_info.get(key)}\n"
                "  **これは飾りではない。混ぜられない。**")

    base_features, add_features = base_info["features"], add_info["features"]
    missing = set(add_features) - set(base_features)
    if missing:
        raise SystemExit(f"相手に無い列がある: {sorted(missing)}")

    changed = []
    for name, feature in add_features.items():
        ours, theirs = _feature_shape(feature), _feature_shape(base_features[name])
        if ours != theirs:
            raise SystemExit(
                f"{name} の形が違う: こちら {ours}、相手 {theirs}\n"
                "  **これは飾りではない。データを作り直すこと。**")
        if feature != base_features[name]:
            changed.append(name)

    print(f"=== {args.add} の features を {args.base} に合わせる ===")
    for name in changed:
        print(f"  {name}: 形は同じ。飾りだけ書き換える")
    if not changed:
        print("  食い違いは無い。書き換えない")
        return

    add_path.with_suffix(".json.bak").write_text(json.dumps(add_info, indent=4))
    add_info["features"] = {name: base_features[name] for name in add_features}
    add_path.write_text(json.dumps(add_info, indent=4))
    print(f"\n{len(changed)} 列を書き換えた。**元は {add_path.with_suffix('.json.bak').name} にある。**")


def command_merge(args: argparse.Namespace) -> None:
    """既存のデータセットの後ろに、作った分を足す。

    **エピソード番号は振り直される。** `--dataset.episodes` で混ぜ方を
    選べるように、足した分の番号を最後に印字する。

    Generates:
      - <output>/ … 2 つを繋いだデータセット
    """
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _prepare_output(args.output)
    base = LeRobotDataset(args.base_repo_id, root=args.base)
    added = LeRobotDataset(args.add_repo_id, root=args.add)
    print(f"元 {base.meta.total_episodes} 本 ＋ 作った分 "
          f"{added.meta.total_episodes} 本")

    try:
        merged = merge_datasets([base, added], output_repo_id=args.output_repo_id,
                                output_dir=args.output)
    except ValueError as error:
        if "Same features is expected" not in str(error):
            raise
        # **形ではなく書き方の食い違いである。** lerobot の版が上がって
        # `features` の飾りが変わったために起きる。`align` が揃える。
        raise SystemExit(
            f"{error}\n\n"
            "**列の書き方が食い違っている（形ではない）。先に揃えること:**\n"
            f"  python npz_to_lerobot.py align --base {args.base} --add {args.add}"
        ) from error
    first = base.meta.total_episodes
    last = merged.meta.total_episodes - 1
    print(f"\n合計 {merged.meta.total_episodes} 本 → {args.output}")
    print(f"**作った分のエピソード番号は {first}〜{last}** である。")
    print(f"  作った分だけで学習: --dataset.episodes=[{first}..{last}]")
    _print_summary(args.output)


def _print_summary(root: Path) -> None:
    """書き出されたものの宣言を印字する。**混ぜる前に目視で突き合わせる。**

    数えるのは `meta/info.json` である。**書き出しが本当に終わったかは、
    残ったファイルから読むのが確実**で、生きているオブジェクトの欄には
    書き込みモードでは入っていないものがある。
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    print(f"  {info['total_episodes']} 本 / {info['total_frames']} コマ / "
          f"{info['total_tasks']} タスク が書けている")
    print(f"  codebase_version {info['codebase_version']}  fps {info['fps']}  "
          f"robot_type {info.get('robot_type')}")
    for name, feature in info["features"].items():
        if name.startswith("observation") or name == "action":
            print(f"    {name:28s} {feature['dtype']:8s} {feature['shape']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="npz からデータセットを作る")
    build.add_argument("--npz-dir", type=Path, required=True,
                       help="scripted_demo.py record の出力先")
    build.add_argument("--output", type=Path, required=True,
                       help="データセットの書き出し先")
    build.add_argument("--repo-id", default="local/scripted_demos",
                       help="データセットの名前（学習時に渡すもの）")
    build.add_argument("--keep-failures", action="store_true",
                       help="失敗した回も入れる（**通常は入れない**）")

    check = subparsers.add_parser(
        "check", help="npz が変換先の宣言と合っているか（lerobot 不要）")
    check.add_argument("--npz-dir", type=Path, required=True)
    check.add_argument("--against", type=Path, default=None,
                       help="混ぜたい相手の meta/info.json")

    inspect_parser = subparsers.add_parser(
        "inspect", help="書き出したものを lerobot に読ませて確かめる")
    inspect_parser.add_argument("--root", type=Path, required=True,
                                help="build か merge の出力先")
    inspect_parser.add_argument("--repo-id", default="local/scripted_demos")

    align = subparsers.add_parser(
        "align", help="混ぜる相手と列の書き方を揃える（merge の前に 1 度）")
    align.add_argument("--base", type=Path, required=True,
                       help="写す元（混ぜたい相手のデータセット）")
    align.add_argument("--add", type=Path, required=True,
                       help="写す先（build の出力）")

    merge = subparsers.add_parser("merge", help="既存のデータセットに足す")
    merge.add_argument("--base", type=Path, required=True,
                       help="元のデータセットの場所")
    merge.add_argument("--base-repo-id", default="lerobot/libero_plus")
    merge.add_argument("--add", type=Path, required=True,
                       help="足すデータセットの場所（build の出力）")
    merge.add_argument("--add-repo-id", default="local/scripted_demos")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--output-repo-id", default="local/libero_plus_scripted")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    {"check": command_check, "build": command_build, "align": command_align,
     "inspect": command_inspect, "merge": command_merge}[args.command](args)


if __name__ == "__main__":
    main()
