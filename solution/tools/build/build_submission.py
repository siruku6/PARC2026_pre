#!/usr/bin/env python3
"""提出用の zip を組み立てる。

提出物は 2 か所から集める。**自分で書いたコードは solution/ が正**であり、
生成物（重み・同梱した LeRobot）だけが work/artifacts/ にある。両者を
コピーで同期させると必ずどちらかが古くなるため、zip 化のときに直接束ねる。

Omnicampus へ出す zip は `policy_server.py` が**ルート直下**に来る構造でなければ
ならない。フォルダごと固めた zip は自動で 1 階層降ろされるが、採点環境によっては
即エラーになると配布 README が警告しているため、ここで構造を保証する。

重みは圧縮が効かず zip 化が遅くなるだけなので、無圧縮（ZIP_STORED）で固める。

実行はどのディレクトリからでもよい。

重みだけは `--weights` で別の場所から取れる（zip の中の名前は
`model_weights/` に固定される）。**π0.5 と SmolVLA を入れ替えて出すときに、
本物のディレクトリを退避させずに済ませるため**である。

    python3 solution/tools/build/build_submission.py \\
        --weights work/artifacts/pi05_rp3000 \\
        --output work/submission_pi05.zip

Reads:
  - solution/policy_server.py
  - solution/requirements.txt
  - work/artifacts/model_weights/**（`--weights` で差し替え可）
  - work/artifacts/vendor/**
Generates:
  - work/submission.zip
  - work/submission.sha256（レポートに記載するハッシュ値）
"""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path

# solution/tools/build/build_submission.py の 3 階層上がリポジトリルート。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SOLUTION = _REPO_ROOT / "solution"
_DEFAULT_ARTIFACTS = _REPO_ROOT / "work" / "artifacts"
_DEFAULT_OUTPUT = _REPO_ROOT / "work" / "submission.zip"

# solution/ から取るファイル（zip のルート直下に置く）。
# `geometry_graft.py` は幾何の接ぎ木（`open_items.md` の F25）の形の定義である。**学習側と同じ
# ファイルを同梱する**——形が食い違うと学習した重みが意味を失う。幾何を接いでいない
# 重みでも `policy_server.py` は読み込まないので、入れておいて害は無い。
_SOLUTION_FILES = ("policy_server.py", "requirements.txt", "geometry_graft.py")

# work/artifacts/ から取るディレクトリ（名前をそのまま zip 内のパスにする）。
_ARTIFACT_DIRS = ("model_weights", "vendor")

# 実行時に生成されるもの。含めるとサイズが膨らむだけで意味がない。
# **`.cache` は `snapshot_download` が重みの隣に作るダウンロード履歴である。**
# 8/14 の 3.6.1 の組み直しで `model_weights/vlm/.cache/huggingface/` の 14 個が
# 紛れ込み、8/13 に検査を通した zip と中身が変わってしまった。**推論には
# 一切使わないので必ず落とす。**
_EXCLUDE_PARTS = frozenset(
    {"__pycache__", ".hf_home", ".ipynb_checkpoints", ".DS_Store", ".cache"}
)
_EXCLUDE_SUFFIXES = frozenset({".pyc"})

# 採点環境では `pip install lerobot` が失敗する（依存の evdev がビルドできない）。
# 他の参加者が実際にこれで採点エラーになっている。方式 A は vendor/lerobot を
# 同梱するので不要だが、うっかり書き足すと採点 1 回分を失う。
_FORBIDDEN_REQUIREMENTS = frozenset({"lerobot", "evdev"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--solution",
        type=Path,
        default=_DEFAULT_SOLUTION,
        help="自分で書いたコードの置き場（既定: solution/）",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=_DEFAULT_ARTIFACTS,
        help="重みと同梱ソースの置き場（既定: work/artifacts/）",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="重みの置き場（既定: <artifacts>/model_weights）。"
             "**zip の中では必ず model_weights/ という名前になる。** "
             "model_weights_pi05_024000 のように別名で置いた重みを、"
             "本物を退避させずにそのまま固めるためにある",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="出力する zip（既定: work/submission.zip）",
    )
    return parser.parse_args()


def _check_requirements(path: Path) -> None:
    """採点環境で pip install に失敗するパッケージが宣言されていないか確かめる。

    `lerobot` は依存の `evdev` がビルドできず、採点環境ではインストールが
    失敗する。`validate_submission.py` は依存解決まで見ないため**ローカルでは
    PASS したまま本番で採点エラーになる**。ここで止める。

    Parameters
    ----------
    path : Path
        提出物の requirements.txt。

    Raises
    ------
    ValueError
        採点環境で入らないパッケージが宣言されている場合。
    """
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue

        name = re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
        if name in _FORBIDDEN_REQUIREMENTS:
            raise ValueError(
                f"{path}:{line_number} に {name} が宣言されています。"
                "採点環境ではインストールに失敗します（evdev のビルドが通らない）。"
                "LeRobot は vendor/ に同梱するため、ここに書く必要はありません。"
            )


def _collect_entries(solution: Path, artifacts: Path,
                     weights: Path | None = None) -> list[tuple[Path, str]]:
    """zip に入れる (実ファイル, zip 内のパス) の一覧を作る。

    足りないものがあれば、zip を作る前に例外にする。中身が欠けた zip を
    提出してしまうと、採点 1 回分を無駄にすることになる。

    Parameters
    ----------
    solution : Path
        `policy_server.py` と `requirements.txt` の置き場。
    artifacts : Path
        同梱ソース（`vendor/`）の置き場。`weights` を省いたときは
        重みもここの `model_weights/` から取る。
    weights : Path | None
        重みの置き場。**zip の中の名前は `model_weights/` に固定する。**
        `MyPolicy` が自分の隣の `model_weights` しか見ないためである。

    Returns
    -------
    entries : list[tuple[Path, str]]
        (実ファイル, zip 内のパス) の一覧。
    """
    entries: list[tuple[Path, str]] = []

    for name in _SOLUTION_FILES:
        path = solution / name
        if not path.is_file():
            raise FileNotFoundError(f"必須ファイルがありません: {path}")
        entries.append((path, name))

    sources = {name: artifacts / name for name in _ARTIFACT_DIRS}
    if weights is not None:
        sources["model_weights"] = weights

    for name, directory in sources.items():
        if not directory.is_dir():
            raise FileNotFoundError(
                f"生成物がありません: {directory}"
                "（vendor_lerobot.py の実行や重みの配置が済んでいるか確認してください）"
            )
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if _EXCLUDE_PARTS & set(path.parts):
                continue
            if path.suffix in _EXCLUDE_SUFFIXES:
                continue
            entries.append((path, f"{name}/{path.relative_to(directory)}"))

    return entries


def _write_zip(output: Path, entries: list[tuple[Path, str]]) -> None:
    """ルート直下構造を保ったまま無圧縮で固める。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path, arcname in entries:
            archive.write(path, arcname=arcname)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_zip(output: Path, entries: list[tuple[Path, str]]) -> None:
    """書き終えた zip を開き直し、中身が意図どおりかを確かめる。

    過去に「zip の中の requirements.txt だけ古い」という事故を起こしている。
    提出は 1 日 1 回しかできないため、組み立てた直後にここで潰す。

    Parameters
    ----------
    output : Path
        書き出した zip。
    entries : list[tuple[Path, str]]
        `_collect_entries` が返した (実ファイル, zip 内のパス) の一覧。

    Raises
    ------
    ValueError
        エントリの集合が一致しない、テキストの中身が食い違う、
        あるいは必須の構成要素が欠けている場合。
    """
    expected = {arcname: path for path, arcname in entries}

    with zipfile.ZipFile(output) as archive:
        actual = {info.filename: info.file_size for info in archive.infolist() if not info.is_dir()}

        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
            raise ValueError(f"zip の中身が一致しません（不足 {missing[:3]} / 余分 {extra[:3]}）")

        # 自分で書いたコードは中身まで突き合わせる。ここが古いのが最も危ない。
        for arcname in _SOLUTION_FILES:
            if archive.read(arcname) != expected[arcname].read_bytes():
                raise ValueError(f"zip の中の {arcname} が {expected[arcname]} と一致しません")

        # 重みとトークナイザは、あるべきものが欠けていないかだけ見る。
        for required in ("model_weights/model.safetensors", "model_weights/config.json"):
            if required not in actual:
                raise ValueError(f"zip に {required} がありません")
        if not any(name.startswith("model_weights/vlm/") for name in actual):
            raise ValueError(
                "zip に model_weights/vlm/ がありません"
                "（トークナイザが無いと採点環境で Hub へ取りに行って失敗します）"
            )
        if not any(name.startswith("vendor/lerobot/") for name in actual):
            raise ValueError("zip に vendor/lerobot/ がありません")
        # 幾何の接ぎ木が入った重みなら、VGGT のソースも要る（pip では入れられない）。
        if any(name.startswith("vendor/vggt/") for name in actual):
            if not any(name == "geometry_graft.py" for name in actual):
                raise ValueError(
                    "zip に vendor/vggt/ があるのに geometry_graft.py がありません"
                    "（接ぎ木の形が無いと VGGT を組めません）"
                )

    print(f"[build] 検証     {len(expected)} エントリ一致、必須構成あり")


def main() -> None:
    args = _parse_args()
    solution: Path = args.solution.resolve()
    artifacts: Path = args.artifacts.resolve()
    output: Path = args.output.resolve()

    weights: Path | None = args.weights.resolve() if args.weights else None

    _check_requirements(solution / "requirements.txt")
    entries = _collect_entries(solution, artifacts, weights)
    _write_zip(output, entries)
    _verify_zip(output, entries)

    checksum = _sha256(output)
    checksum_path = output.with_suffix(".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")

    print(f"[build] コード   {solution}")
    print(f"[build] 生成物   {artifacts}")
    print(f"[build] 重み     {weights or artifacts / 'model_weights'}")
    print(f"[build] 出力     {output}")
    print(f"[build] {len(entries)} files / {output.stat().st_size / 1e6:.0f} MB")
    print(f"[build] sha256 {checksum}")
    print(f"[build] ハッシュを保存: {checksum_path}")


if __name__ == "__main__":
    main()
