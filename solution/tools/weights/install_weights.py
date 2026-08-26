#!/usr/bin/env python3
"""Colab で追加学習したマージ済みモデルを、提出物の重みとして取り込む。

配布ノートブックの**マージセル**（`## 9. LoRAをマージしてモデル全体を保存する`）が
出力するディレクトリを入力に取る。そこには `vlm/` が**含まれない**ため、
今ある `model_weights/vlm/`（SmolVLM の設定とトークナイザ）を引き継ぐ必要がある。
これを手作業でやると必ず取りこぼすので、ここで機械的に行う。

取り込む前に、提出 1 回分を無駄にする類の事故を検出する。

- LoRA パラメータがマージされずに残っていないか
- テンソルの構成が今の重みと一致しているか（`MyPolicy` が読めるか）
- 必須ファイルが揃っているか

**既存の重みは削除せず `.bak` へ退避する。** 追加学習の結果が悪かったときに
素の重みへ戻せなければ、1 日 1 回の提出枠を守れない。

実行はどのディレクトリからでもよい。

    python solution/tools/weights/install_weights.py ~/Downloads/merged_model
    python solution/tools/weights/install_weights.py ~/Downloads/merged_model --dry-run

Reads:
  - <source>/**（マージ済みモデル）
  - work/artifacts/model_weights/vlm/**（引き継ぐ）
Generates:
  - work/artifacts/model_weights/（`vlm/` を除いて置き換え）
  - work/artifacts/model_weights.bak/（直前の内容を退避）
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# solution/tools/weights/install_weights.py の 3 階層上がリポジトリルート。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DESTINATION = _REPO_ROOT / "work" / "artifacts" / "model_weights"

# マージ済みモデルに必ずあるべきファイル。1 つでも欠ければ取り込まない。
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)

# 既存の重みから引き継ぐディレクトリ。マージセルは出力しない。
_INHERITED_DIRS = ("vlm",)

_EXCLUDE_PARTS = frozenset({"__pycache__", ".ipynb_checkpoints", ".DS_Store"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source",
        type=Path,
        help="マージ済みモデルのディレクトリ（Colab から持ち帰ったもの）",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=_DEFAULT_DESTINATION,
        help="取り込み先（既定: work/artifacts/model_weights）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="検査だけ行い、ファイルは書き換えない",
    )
    parser.add_argument(
        "--allow-key-change",
        action="store_true",
        help="テンソル構成が変わっていても取り込む（通常は使わない）",
    )
    return parser.parse_args()


def _read_safetensors_keys(path: Path) -> set[str]:
    """safetensors のヘッダだけを読み、テンソル名の集合を返す。

    先頭 8 バイトがヘッダ長（リトルエンディアン）、続く区間が JSON という
    形式なので、900MB を読み込まずに構成を確認できる。torch にも
    safetensors パッケージにも依存しない。

    Parameters
    ----------
    path : Path
        safetensors ファイル。

    Returns
    -------
    keys : set[str]
        テンソル名の集合。`__metadata__` は含めない。
    """
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if not 0 < header_size < (1 << 30):
            raise ValueError(f"safetensors のヘッダ長が異常です: {path} ({header_size})")
        header = json.loads(stream.read(header_size))

    return {name for name in header if name != "__metadata__"}


def _check_source(source: Path) -> None:
    """マージ済みモデルとして体裁が整っているかを確認する。"""
    if not source.is_dir():
        raise NotADirectoryError(f"取り込み元が見つかりません: {source}")

    missing = [name for name in _REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"マージ済みモデルに必須ファイルがありません: {', '.join(missing)}"
            "（マージセルが最後まで実行されたか確認してください）"
        )

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("use_peft"):
        raise ValueError(
            "config.json の use_peft が真のままです。LoRA がマージされていません。"
        )


def _check_no_lora(keys: set[str]) -> None:
    """LoRA パラメータが残っていないことを確認する。

    マージセル自身も同じ検査をしているが、別経路で作ったチェックポイントを
    渡される場合に備えてここでも見る。残っていると `MyPolicy` が読めない。
    """
    remaining = sorted(name for name in keys if "lora_" in name.lower())
    if remaining:
        raise ValueError(
            f"LoRA パラメータが {len(remaining)} 個残っています: {remaining[:3]} …"
        )


def _compare_keys(new_keys: set[str], old_keys: set[str]) -> tuple[list[str], list[str]]:
    """テンソル構成の差分を (増えたもの, 消えたもの) で返す。"""
    return sorted(new_keys - old_keys), sorted(old_keys - new_keys)


def _copy_tree(source: Path, destination: Path) -> int:
    """除外対象を飛ばしてディレクトリを複製し、コピーしたファイル数を返す。"""
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or _EXCLUDE_PARTS & set(path.parts):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def _build_staging(source: Path, destination: Path, staging: Path) -> None:
    """取り込み後の内容を、退避前に別ディレクトリへ組み立てる。

    先に組み立ててから入れ替えることで、途中で失敗しても既存の重みが
    半端な状態にならない。
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for path in sorted(source.iterdir()):
        if path.name in _INHERITED_DIRS or _EXCLUDE_PARTS & {path.name}:
            continue
        if path.is_file():
            shutil.copy2(path, staging / path.name)
        elif path.is_dir():
            _copy_tree(path, staging / path.name)

    for name in _INHERITED_DIRS:
        inherited = destination / name
        if not inherited.is_dir():
            raise FileNotFoundError(
                f"引き継ぐディレクトリがありません: {inherited}"
                "（素の重みを一度配置してから実行してください）"
            )
        _copy_tree(inherited, staging / name)


def _swap_in(destination: Path, staging: Path) -> Path:
    """既存を .bak へ退避し、組み立てたものを本番の場所へ入れ替える。"""
    backup = destination.with_name(destination.name + ".bak")
    if backup.exists():
        shutil.rmtree(backup)
    destination.rename(backup)
    staging.rename(destination)
    return backup


def _print_plan(source: Path, destination: Path, added: list[str], removed: list[str]) -> None:
    print(f"[install] 取り込み元 {source}")
    print(f"[install] 取り込み先 {destination}")
    print(f"[install] 引き継ぐ   {', '.join(_INHERITED_DIRS)}/")

    if not added and not removed:
        print("[install] テンソル構成は既存と一致しています")
        return

    print(f"[install] 警告: テンソル構成が変わっています (+{len(added)} / -{len(removed)})")
    for name in added[:5]:
        print(f"[install]   + {name}")
    for name in removed[:5]:
        print(f"[install]   - {name}")


def main() -> None:
    args = _parse_args()
    source: Path = args.source.resolve()
    destination: Path = args.destination.resolve()

    _check_source(source)

    new_keys = _read_safetensors_keys(source / "model.safetensors")
    _check_no_lora(new_keys)

    existing_weights = destination / "model.safetensors"
    if existing_weights.is_file():
        added, removed = _compare_keys(new_keys, _read_safetensors_keys(existing_weights))
    else:
        added, removed = [], []

    _print_plan(source, destination, added, removed)

    if (added or removed) and not args.allow_key_change:
        raise ValueError(
            "テンソル構成が既存と異なります。MyPolicy が読めない可能性が高いため中止しました。"
            "意図した変更であれば --allow-key-change を付けてください。"
        )

    if args.dry_run:
        print("[install] --dry-run のため書き換えていません")
        return

    staging = destination.with_name(destination.name + ".new")
    _build_staging(source, destination, staging)
    backup = _swap_in(destination, staging)

    total_bytes = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    print(f"[install] 退避     {backup}")
    print(f"[install] 完了     {total_bytes / 1e6:.0f} MB")
    print("[install] 次に: python solution/tools/build/build_submission.py")


if __name__ == "__main__":
    main()
