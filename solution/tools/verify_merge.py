#!/usr/bin/env python3
"""マージ済みモデルが「学習の成果を含み、かつ壊れていない」ことを確かめる。

`merge.py` の検査は **LoRA が残っていないこと**しか見ていない。これは
「畳み込みに失敗して LoRA が消えただけ」の場合も通ってしまう。**初期重みを
そのまま出しても合格する**ため、学習が反映されたかどうかは分からない。

このスクリプトは初期重みと 1 テンソルずつ比べ、**どれだけ変わったか**を出す。

    variance を見る観点は 3 つある。

      1. 変わったテンソルが LoRA の対象（attention の射影）に集中しているか
      2. 変化量が小さすぎないか（＝ほぼ初期重みのまま）
      3. 変化量が大きすぎないか（＝壊れている、NaN・Inf を含む）

使い方::

    # GPU マシンで（初期重みは HF から取る。7 GB 程度の通信）
    python3 solution/tools/verify_merge.py /out/smolvla_lora_all_suites/merged_model

    # 通信を避けたい場合、手元にある初期重みを指してもよい
    python3 solution/tools/verify_merge.py <merged> --base-model <初期重みのディレクトリ>

Reads:
  - <merged>/model.safetensors, <merged>/config.json
  - lerobot/smolvla_libero_plus の model.safetensors（HF、既定）
Generates:
  - 標準出力に判定結果（異常があれば終了コード 1）
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

_BASE_MODEL_REPO = "lerobot/smolvla_libero_plus"
_BASE_MODEL_REVISION = "7bb70aa5bc92b82c9239142775d3a173103567ff"

# LoRA が入る層の名前。ここ以外が動いていたら、マージの経路がおかしい。
# LeRobot の LORA 既定は attention の射影に当たる。
_EXPECTED_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

# 相対変化量がこれ未満なら「学習が反映されていない」と見る。
# LoRA r=16 / alpha=16 を 15,000 歩まわした場合、対象層は 1e-3 の桁で動く。
_MIN_RELATIVE_CHANGE = 1e-5

# これを超えたら壊れている可能性を疑う（初期重みと同じ大きさだけ動いた）。
_MAX_RELATIVE_CHANGE = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("merged", type=Path, help="マージ済みモデルのディレクトリ")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="初期重みのディレクトリ。省略すると HuggingFace から取る",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="変化量の大きいテンソルを何件表示するか（既定 10）",
    )
    return parser.parse_args()


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    """safetensors のヘッダと、本体の開始位置を返す。

    先頭 8 バイトがヘッダ長（リトルエンディアン）、続く区間が JSON である。
    900 MB を読み込まずにテンソルの構成と位置が分かる。

    Parameters
    ----------
    path : Path
        safetensors ファイル。

    Returns
    -------
    header : dict
        テンソル名 → {dtype, shape, data_offsets} の辞書（`__metadata__` を除く）。
    body_offset : int
        本体（テンソルの実データ）が始まるバイト位置。
    """
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size < (1 << 30):
            raise ValueError(f"safetensors のヘッダ長が異常です: {path} ({header_size})")
        header = json.loads(stream.read(header_size))

    return ({name: spec for name, spec in header.items() if name != "__metadata__"},
            8 + header_size)


_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),   # numpy に bfloat16 が無いので、後で f4 へ広げる
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def _read_tensor(path: Path, spec: dict, body_offset: int) -> np.ndarray:
    """1 テンソルだけをファイルから読み、float64 の 1 次元配列で返す。

    比較にしか使わないので形は落とす。bfloat16 は numpy に型が無いため、
    16 bit を上位に置いた float32 として復元する（下位 16 bit は 0）。

    Parameters
    ----------
    path : Path
        safetensors ファイル。
    spec : dict
        ヘッダの 1 エントリ。`dtype` と `data_offsets` を使う。
    body_offset : int
        本体の開始位置。

    Returns
    -------
    values : np.ndarray
        float64 の 1 次元配列。
    """
    dtype_name = spec["dtype"]
    if dtype_name not in _DTYPES:
        raise ValueError(f"未対応の dtype です: {dtype_name}")

    start, end = spec["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(body_offset + start)
        raw = stream.read(end - start)

    values = np.frombuffer(raw, dtype=_DTYPES[dtype_name])
    if dtype_name == "BF16":
        # bfloat16 は float32 の上位 16 bit と同じ並びである。
        widened = np.zeros(values.size, dtype="<u4")
        widened |= values.astype("<u4") << 16
        values = widened.view("<f4")

    return values.astype(np.float64)


def _resolve_base_model(argument: Path | None) -> Path:
    """初期重みの safetensors を返す。指定が無ければ HuggingFace から取る。"""
    if argument is not None:
        path = argument / "model.safetensors" if argument.is_dir() else argument
        if not path.is_file():
            raise FileNotFoundError(f"初期重みが見つかりません: {path}")
        return path

    from huggingface_hub import hf_hub_download

    print(f"[verify] 初期重みを取得: {_BASE_MODEL_REPO}@{_BASE_MODEL_REVISION[:8]}")
    return Path(hf_hub_download(
        _BASE_MODEL_REPO, "model.safetensors", revision=_BASE_MODEL_REVISION,
    ))


def _check_config(merged: Path) -> list[str]:
    """config.json が提出物として読める状態かを見て、問題を並べて返す。"""
    problems: list[str] = []
    config_path = merged / "config.json"
    if not config_path.is_file():
        return [f"config.json がありません: {config_path}"]

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("use_peft"):
        problems.append("use_peft が真のままです。LoRA が畳み込まれていません")
    if config.get("pretrained_path"):
        problems.append(
            f"pretrained_path が残っています（{config['pretrained_path']}）。"
            "提出物では解決できないパスになります"
        )
    if config.get("load_vlm_weights"):
        problems.append("load_vlm_weights が真です。提出物は同梱の VLM を使います")
    return problems


def _compare(merged_path: Path, base_path: Path) -> tuple[list[tuple[str, float, float]],
                                                          list[str], list[str]]:
    """テンソルごとの変化量を測る。

    Returns
    -------
    changes : list[tuple[str, float, float]]
        (テンソル名, 相対変化量, 最大絶対差) を変化量の降順で。変化 0 は含めない。
    unhealthy : list[str]
        NaN または Inf を含むテンソルの名前。
    shape_mismatch : list[str]
        初期重みと大きさが違うテンソルの名前。
    """
    merged_header, merged_offset = _read_safetensors_header(merged_path)
    base_header, base_offset = _read_safetensors_header(base_path)

    changes: list[tuple[str, float, float]] = []
    unhealthy: list[str] = []
    shape_mismatch: list[str] = []

    for name, spec in merged_header.items():
        merged_values = _read_tensor(merged_path, spec, merged_offset)

        if not np.all(np.isfinite(merged_values)):
            unhealthy.append(name)
            continue

        base_spec = base_header.get(name)
        if base_spec is None:
            continue   # 初期重みに無いものは _compare_keys 側の話なので触らない

        base_values = _read_tensor(base_path, base_spec, base_offset)
        if base_values.size != merged_values.size:
            shape_mismatch.append(name)
            continue

        difference = np.abs(merged_values - base_values)
        max_difference = float(difference.max())
        if max_difference == 0.0:
            continue

        base_scale = float(np.abs(base_values).max())
        relative = max_difference / base_scale if base_scale > 0 else float("inf")
        changes.append((name, relative, max_difference))

    changes.sort(key=lambda entry: -entry[1])
    return changes, unhealthy, shape_mismatch


def _classify(changes: list[tuple[str, float, float]]) -> tuple[int, int]:
    """変化したテンソルを、LoRA の対象かどうかで数える。"""
    on_target = sum(
        1 for name, _, _ in changes
        if any(target in name for target in _EXPECTED_LORA_TARGETS)
    )
    return on_target, len(changes) - on_target


def _report(changes: list[tuple[str, float, float]], top: int) -> None:
    """変化量の大きいものを表で出す。"""
    print(f"\n変化したテンソル {len(changes)} 個のうち、大きい順に {min(top, len(changes))} 件")
    print(f"  {'相対変化':>10}  {'最大絶対差':>12}  テンソル名")
    for name, relative, absolute in changes[:top]:
        print(f"  {relative:>10.2e}  {absolute:>12.2e}  {name}")


def main() -> None:
    args = _parse_args()
    merged_dir: Path = args.merged.resolve()
    merged_path = merged_dir / "model.safetensors"
    if not merged_path.is_file():
        raise SystemExit(f"エラー: {merged_path} がありません")

    problems = _check_config(merged_dir)
    base_path = _resolve_base_model(args.base_model)

    print(f"[verify] マージ済み {merged_path}")
    print(f"[verify] 初期重み   {base_path}")

    changes, unhealthy, shape_mismatch = _compare(merged_path, base_path)
    on_target, off_target = _classify(changes)

    if changes:
        _report(changes, args.top)

    largest = changes[0][1] if changes else 0.0

    print("\n判定")
    print(f"  変化したテンソル      {len(changes)} 個"
          f"（LoRA の対象 {on_target} / 対象外 {off_target}）")
    print(f"  いちばん大きい相対変化 {largest:.2e}")

    if unhealthy:
        problems.append(
            f"NaN または Inf を含むテンソルが {len(unhealthy)} 個: {unhealthy[:3]} …"
        )
    if shape_mismatch:
        problems.append(
            f"初期重みと大きさが違うテンソルが {len(shape_mismatch)} 個: {shape_mismatch[:3]} …"
        )
    if not changes:
        problems.append(
            "初期重みと 1 ビットも違いません。**学習の成果が入っていません。**"
            "マージ対象のチェックポイントを間違えていないか確認してください"
        )
    elif largest < _MIN_RELATIVE_CHANGE:
        problems.append(
            f"変化が小さすぎます（{largest:.2e} < {_MIN_RELATIVE_CHANGE:.0e}）。"
            "学習が数歩しか進んでいないか、別のチェックポイントを見ています"
        )
    elif largest > _MAX_RELATIVE_CHANGE:
        problems.append(
            f"変化が大きすぎます（{largest:.2e} > {_MAX_RELATIVE_CHANGE:.0e}）。"
            "重みが壊れている可能性があります"
        )
    if changes and on_target == 0:
        problems.append(
            "LoRA の対象（q_proj / k_proj / v_proj / o_proj）が 1 つも変わっていません。"
            "LoRA の当たり先が想定と違います"
        )

    if problems:
        print("\n**問題あり**")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("\n  正常。学習の成果が入っており、壊れてもいない。")
    print("\n次に: python3 solution/tools/install_weights.py "
          f"{merged_dir} --dry-run")


if __name__ == "__main__":
    main()
