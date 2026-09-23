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

組み終わると、`docs/reference/19_weights_catalog.md`（実験台帳）への記載を促す
メッセージを出す。**行を自動で書きはしない**——重みの出自の系譜や、前の版との
1 変数の比較はこのスクリプトの引数からは分からず、空の行を先に作ると
「書いたつもりになる」落とし穴になる（2026-09-13）。メッセージには
sha256・大きさ・重みのパス・zip の中の設定など、**ここで分かる事実だけ**を埋める。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

# `patch_zip_policy_server.py` の `_INTERESTING`（設定を抜き出す正規表現）を再利用する。
# 同じディレクトリのスクリプトなので、実行時の sys.path[0] に頼らず明示的に足す。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_zip_policy_server import _INTERESTING  # noqa: E402

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

# **ライセンス全文を同梱する**（予選のレポート 4 節の宿題。Apache License 2.0 第 4 条 (a) は
# 頒布物に LICENSE 全文を入れることを求めており、予選の提出物はこれを欠いていた）。
# `solution/licenses/` から zip の `licenses/` へそのまま入れる。
_LICENSE_DIR_NAME = "licenses"
_LICENSE_FILES = ("LICENSE-Apache-2.0-lerobot.txt", "LICENSE-MIT-LIBERO.txt")

# **NOTICE は模型で中身が違う**（π0.5 の土台は Gemma Terms of Use、MolmoAct2 は Apache 2.0）。
# `model_weights/config.json` の `type` で選び、zip の中では `NOTICE.md` の 1 つになる。
_NOTICE_BY_KIND = {"pi05": "NOTICE-pi05.md", "smolvla": "NOTICE-pi05.md", "molmoact2": "NOTICE-molmoact2.md"}

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

    license_dir = solution / _LICENSE_DIR_NAME
    for name in _LICENSE_FILES:
        path = license_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"ライセンス全文がありません: {path}")
        entries.append((path, f"{_LICENSE_DIR_NAME}/{name}"))

    weights_dir = weights if weights is not None else artifacts / "model_weights"
    notice = license_dir / _notice_name_for(weights_dir)
    if not notice.is_file():
        raise FileNotFoundError(f"NOTICE がありません: {notice}")
    entries.append((notice, "NOTICE.md"))

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


# **模型ごとに、`vlm/` に入るべきトークナイザが違う。**
# π0.5 の土台は PaliGemma / Gemma、SmolVLA の土台は SmolVLM2 である。
#
# **有無だけを見ていたせいで、2026-09-17 の 22.0.0 は SmolVLM2 のトークナイザを
# 積んだまま検査を通り、本選 1 の Track1 で 0.075 になった**（正しい重みでの値は
# 0.498。docs/reference/19_weights_catalog.md の 6.21 節）。**方策は起動して動くが、
# 指示文が別のトークン列になるので言語の条件づけを失う**——8 本中 7 本が打ち切りに
# 張り付いた。**中身まで見ないと、この事故はここを素通りする。**
_TOKENIZER_CLASS_BY_POLICY = {"pi05": "GemmaTokenizer", "pi0": "GemmaTokenizer",
                              "smolvla": "GPT2Tokenizer"}


def _verify_tokenizer_family(archive: zipfile.ZipFile, actual: dict[str, int]) -> None:
    """`model_weights/vlm/` のトークナイザが、その模型のものかを見る。"""
    if not any(name.startswith("model_weights/vlm/") for name in actual):
        return  # MolmoAct2 は `vlm/` を使わない

    policy = json.loads(archive.read("model_weights/config.json")).get("type")
    want = _TOKENIZER_CLASS_BY_POLICY.get(policy)
    if want is None:
        return  # 知らない模型には口を出さない

    arcname = "model_weights/vlm/tokenizer_config.json"
    if arcname not in actual:
        raise ValueError(f"zip に {arcname} がありません（トークナイザの素性を確かめられません）")

    got = json.loads(archive.read(arcname)).get("tokenizer_class")
    if got != want:
        files = sorted(
            name.removeprefix("model_weights/vlm/")
            for name in actual
            if name.startswith("model_weights/vlm/")
        )
        raise ValueError(
            f"model_weights/vlm/ のトークナイザが {policy} のものではありません"
            f"（{got}。{want} であるべきです）。"
            f"入っている {len(files)} ファイル: {files}。"
            "**別の模型の vlm/ を写していないか確かめてください**"
            "——work/artifacts/model_weights/vlm は SmolVLA 時代の置き場で、"
            "π0.5 に写すと動くのに課題を解かない zip になります"
            "（docs/howto/02_build_submission.md の 0-3 節）"
        )


def _notice_name_for(weights: Path) -> str:
    """重みの `config.json` の `type` から、同梱する NOTICE を選ぶ。

    **読み違えると、載っていない土台のライセンスを名乗ることになる。**
    分からないときは止める——提出物の権利表示を推測で埋めない。
    """
    config = weights / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"重みの設定がありません: {config}")
    kind = str(json.loads(config.read_text()).get("type", ""))
    try:
        return _NOTICE_BY_KIND[kind]
    except KeyError:
        raise ValueError(
            f"NOTICE を選べません（config.json の type が {kind!r}）。"
            f"solution/licenses/ に NOTICE を足して _NOTICE_BY_KIND へ登録すること"
        ) from None


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
        # **トークナイザの同梱先は模型で違う。**π0.5・SmolVLA は `model_weights/vlm/`、
        # MolmoAct2 は `model_weights/molmoact2_base_small/` である
        # （`policy_server.py` の `_build_molmoact2`。MolmoAct2 には `vlm/` 同梱の
        # 仕組みが無い）。**どちらも無ければ採点環境が Hub へ取りに行って失敗する。**
        _tokenizer_dirs = ("model_weights/vlm/", "model_weights/molmoact2_base_small/")
        if not any(name.startswith(_tokenizer_dirs) for name in actual):
            raise ValueError(
                "zip に model_weights/vlm/ も model_weights/molmoact2_base_small/ も"
                "ありません（トークナイザが無いと採点環境で Hub へ取りに行って失敗します）"
            )
        _verify_tokenizer_family(archive, actual)
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


def _extract_version(solution: Path) -> str:
    """組んだ `policy_server.py` の `__version__` を読む。台帳の見出しに使う。"""
    text = (solution / "policy_server.py").read_text(encoding="utf-8")
    match = re.search(r'__version__ = "([^"]+)"', text)
    return match.group(1) if match else "?"


def _extract_settings(solution: Path) -> list[str]:
    """`_INTERESTING`（`patch_zip_policy_server.py` と共通）で設定行を抜き出す。"""
    text = (solution / "policy_server.py").read_text(encoding="utf-8")
    lines: list[str] = []
    for pattern in _INTERESTING:
        for m in re.finditer(pattern, text, re.M):
            lines.append(m.group(0).strip())
    return lines


def _print_catalog_reminder(
    *, version: str, weights: Path | None, output: Path,
    checksum: str, entries: int, size_mb: float, settings: list[str],
) -> None:
    """台帳（`docs/reference/19_weights_catalog.md`）への記載を促す。

    **ここで自動で行を書かない。** 台帳に要る事実の半分（重みの出自の系譜、
    何を変えたか、データの内容）はこのスクリプトの引数からは分からず、
    空の行を先に作ると「書いたつもりになる」落とし穴になる
    （2026-09-13、ユーザの指摘）。**代わりに、ここで分かる事実だけを
    埋めて、書く作業そのものは急かすだけにする。**

    Parameters
    ----------
    version : str
        `__version__`。台帳の版番号の見出しに使う。
    weights : Path | None
        `--weights` で渡された重みの置き場。**出自の系譜はこれだけでは分からない**
        （学習の設定・出発点・歩数は `train_config.json` を読むか、学習側の記録から
        補うこと）。
    settings : list[str]
        `_INTERESTING` で抜き出した設定行。前の版との 1 変数の比較に使える。
    """
    print()
    print("=" * 60)
    print(f"[build] 台帳への記載を忘れずに —— docs/reference/19_weights_catalog.md")
    print(f"[build]   版        : {version}")
    print(f"[build]   重み      : {weights or '（既定の model_weights/）'}")
    print(f"[build]   sha256    : {checksum}")
    print(f"[build]   大きさ    : {entries} エントリ・{size_mb:.0f} MB")
    print(f"[build]   出力      : {output}")
    print("[build]   中の設定  :")
    for line in settings:
        print(f"[build]     {line}")
    print("[build]")
    print("[build]   この呼び出しだけでは分からず、自分で書く必要があるもの:")
    print("[build]     - 重みの出自（どの学習・何ステップ・何から出発したか）")
    print("[build]     - 前の版との 1 変数の比較（設定／重み／データのどれを変えたか）")
    print("[build]     - 提出したら、出した先のトラックと日付")
    print("=" * 60)


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

    _print_catalog_reminder(
        version=_extract_version(solution),
        weights=weights,
        output=output,
        checksum=checksum,
        entries=len(entries),
        size_mb=output.stat().st_size / 1e6,
        settings=_extract_settings(solution),
    )


if __name__ == "__main__":
    main()
