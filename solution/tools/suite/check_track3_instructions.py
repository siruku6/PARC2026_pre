#!/usr/bin/env python3
"""**Track3 の指示文が、正規化で逆の意味に書き換わらないかを確かめる。**

## なぜこの確認が要るか

**寄せ先は順方向の 40 文をハードコードしたものである**（`policy_server.py` の
`_TRAINING_INSTRUCTIONS`）。**逆向きデータで足した 3 文は入っていない。**
Track3 の 4 課題はどれも「取る・下ろす・出す」なので、
**しきい値が低いと「皿から下ろす」が「皿に載せる」へ寄ってしまう。**

**実際に起きた**——版 5.0.1（しきい値 0.50）で two mugs の 1 文が 0.607 で誤って寄り、
逆向きの課題が順方向に書き換わっていた（[knowledge/15](../../../docs/knowledge/15_instruction_normalizer_inverted_a_task.md)）。
**逆向きデータを学ばせた重みでは、これは「点が下がる」ではなく「正反対を命じる」である。**

## 何を出すか

| 出るもの | 意味 |
|---|---|
| 既定（引数なし） | **いまの `_MIN_LEXICAL` で 4 文が書き換わるか。**1 つでも書き換わったら**終了コード 1** |
| `--sweep` | しきい値を振って、どこから書き換わり始めるかの表 |

**終了コードで落ちる道具である。**提出物を組む前の関門として打つ。

## 使い方

    python3 solution/tools/suite/check_track3_instructions.py
    python3 solution/tools/suite/check_track3_instructions.py --sweep

Reads:
  - <キット>/compe/t3/assets/bddl_files/libero_t3/*.bddl の `(:language …)`
  - solution/policy_server.py（`InstructionNormalizer` を借りる）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from check_instruction_normalization import _load_normalizer_class  # noqa: E402

# bddl の `(:language …)` から文を取る。**Track3 には課題 CSV が無く**、
# キットの `compe/t3/register.py` もここから読んでいる（`_bddl_language`）。
LANGUAGE_RE = re.compile(r"\(:language\s+([^()]+?)\s*\)")
# 既定のキット。`run_final_eval.sh` の `KIT_ROOT` と同じ「リポジトリの隣」である。
REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_KIT = REPOSITORY.parent / "b_PARC2026_final"
SWEEP = (0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.99)


def track3_instructions(kit: Path) -> list[tuple[str, str]]:
    """Track3 の 4 課題の (課題名, 指示文) を bddl から読む。"""
    directory = kit / "compe/t3/assets/bddl_files/libero_t3"
    if not directory.is_dir():
        raise SystemExit(f"**Track3 の bddl が無い: {directory}**")
    found = []
    for path in sorted(directory.glob("*.bddl")):
        match = LANGUAGE_RE.search(path.read_text())
        if match:
            found.append((path.stem, match.group(1)))
    if not found:
        raise SystemExit(f"**`(:language …)` が 1 つも読めなかった: {directory}**")
    return found


def rewrites(normalizer_class, instructions, threshold: float) -> list[tuple[str, str]]:
    """`threshold` で書き換わった (課題名, 寄せ先) だけを返す。"""
    normalizer = normalizer_class(min_lexical=threshold)
    changed = []
    for name, sentence in instructions:
        result = normalizer.normalize(sentence)
        text = result[0] if isinstance(result, (tuple, list)) else result
        if text.strip().lower() != sentence.strip().lower():
            changed.append((name, text))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--sweep", action="store_true",
                        help="しきい値を振って表を出す（関門としては使わない）")
    args = parser.parse_args()

    normalizer_class, _has_numpy = _load_normalizer_class()
    import policy_server  # `_load_normalizer_class` が読み込んだもの

    instructions = track3_instructions(args.kit)
    print(f"Track3 の指示文 {len(instructions)} 文・寄せ先 "
          f"{len(policy_server._TRAINING_INSTRUCTIONS)} 文（**すべて順方向**）")

    if args.sweep:
        print()
        print(f"  {'しきい値':>8}  書き換わった文")
        for threshold in SWEEP:
            changed = rewrites(normalizer_class, instructions, threshold)
            mark = "**危険**" if changed else "安全"
            print(f"  {threshold:>8.2f}  {len(changed)}/{len(instructions)}  {mark}")
            for name, text in changed:
                print(f"            {name}\n              → {text}")
        return

    threshold = policy_server._MIN_LEXICAL
    changed = rewrites(normalizer_class, instructions, threshold)
    print(f"いまの _MIN_LEXICAL = {threshold}")
    if changed:
        print()
        for name, text in changed:
            print(f"  **書き換わった** {name}")
            print(f"    → {text}")
        raise SystemExit(
            f"\n**{len(changed)} 文が書き換わる。この設定で Track3 を出してはいけない。**\n"
            "  `--sweep` でどこから安全になるか見る")
    print(f"  **{len(instructions)} 文すべてそのまま渡る。**Track3 に出してよい")


if __name__ == "__main__":
    main()
