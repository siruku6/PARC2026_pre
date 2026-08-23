#!/usr/bin/env python3
"""指示文の正規化が、どれだけ正しく寄せられるかを測る。

**提出物に入っている `InstructionNormalizer` そのものを呼ぶ。** 別に書き写した
実装ではないので、しきい値を変えたらこの数字も動く。

答えは分かっている。LIBERO-plus の変種名から素のタスク名が取れるので
（`make_holdout_split.py` が配布ノートブックと突き合わせて自己検査している）、
**「渡ってくる文」と「本来の学習データの文」の対が 10,030 件そろう。**

    python3 solution/tools/suite/check_instruction_normalization.py            # 表を出す
    python3 solution/tools/suite/check_instruction_normalization.py --sweep    # しきい値を振る
    python3 solution/tools/suite/check_instruction_normalization.py --embed    # 意味で寄せる道も測る

**`--embed` には torch と重みが要る**（VLM の言語側を実際に動かすため）。
手元（macOS）には torch が無いので、Docker で回す。
手順は [docs/howto/03_gpu_eval.md](../../docs/howto/03_gpu_eval.md) にある。

## 何を見て採否を決めるのか

| 見るもの | 意味 |
|---|---|
| **誤って別のタスクへ寄せた数** | **これが 0 でなければ入れない。** いま解けている回を壊しうる |
| 正しく寄せた数 | 増えるほど、モデルが学習した言い方を受け取れる |
| 触らなかった数 | 迷ったので何もしなかった。**害は無い** |

**本番の摂動は Background Textures と Light Conditions の 2 つ**と運営 Q&A No.21 が
答えているので、その 2 つを別立てで出す。**言い換え（Language Instructions）は
本番に含まれない可能性が高い**が、含まれたときに効くかどうかも一緒に出す。

Reads:
  - <libero-plus>/libero/libero/benchmark/task_classification.json
  - <libero-plus>/libero/libero/bddl_files/**/*.bddl
  - solution/policy_server.py（`InstructionNormalizer` を借りる）
Generates:
  - 標準出力（表）
  - <output>（`--json` のとき。判定の内訳）
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruction_targets import build_pairs, training_instructions  # noqa: E402

# **本番に含まれる摂動**（運営 Q&A No.21 の回答と、配布された 4 タスク）。
_PRODUCTION_CATEGORIES = frozenset({"Background Textures", "Light Conditions"})
_LANGUAGE_CATEGORY = "Language Instructions"

# `policy_server.py` を読むために最低限あればよいもの。**サーバーは動かさない。**
_SERVER_ONLY_MODULES = ("msgpack", "uvicorn", "fastapi")


def _load_normalizer_class():
    """提出物から `InstructionNormalizer` を借りる。

    **`policy_server.py` はサーバーでもあるので、HTTP 側の依存を import する。**
    語の重なりで寄せる道はそれらを 1 つも使わないため、無い環境では
    **形だけの代役**を置いて読み込む。**動かすのは正規化の部分だけである。**

    Returns
    -------
    normalizer_class : type
        `InstructionNormalizer`。
    has_numpy : bool
        本物の numpy があるか（無ければ意味で寄せる道は測れない）。
    """
    has_numpy = True
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        has_numpy = False
        stub = types.ModuleType("numpy")
        # 関数の注釈（`-> np.ndarray` など）を読むためだけに要る。
        stub.ndarray = type("ndarray", (), {})
        stub.float32 = float
        sys.modules["numpy"] = stub

    for name in _SERVER_ONLY_MODULES:
        try:
            __import__(name)
        except ModuleNotFoundError:
            stub = types.ModuleType(name)
            if name == "fastapi":
                # `@app.get(...)` / `@app.post(...)` を素通しさせる。
                # **経路の中身は呼ばない**（サーバーとして動かさないため）。
                def _route(*_args, **_kwargs):
                    return lambda function: function

                stub.FastAPI = type("FastAPI", (), {
                    "__init__": lambda self, **kwargs: None,
                    "get": _route,
                    "post": _route,
                })
                stub.Request = type("Request", (), {})
                stub.Response = type("Response", (), {})
            sys.modules[name] = stub

    sys.path.insert(0, str(_REPO_ROOT / "solution"))
    import policy_server  # noqa: E402

    return policy_server.InstructionNormalizer, has_numpy


def _group_of(categories: set[str]) -> str:
    """その文がどの束に属するか（本番 / 言い換え / その他）。"""
    if _LANGUAGE_CATEGORY in categories:
        return "言い換え"
    if categories & _PRODUCTION_CATEGORIES:
        return "本番の 2 カテゴリ"
    return "その他の摂動"


def _collapse(pairs: list[dict]) -> list[dict]:
    """同じ文をまとめる。**変種の数も持たせる**（重み付けに使う）。"""
    merged: dict[tuple[str, str], dict] = {}
    for pair in pairs:
        key = (pair["delivered"], pair["target"])
        entry = merged.setdefault(key, {
            "delivered": pair["delivered"],
            "target": pair["target"],
            "categories": set(),
            "variants": 0,
        })
        entry["categories"].add(pair["category"])
        entry["variants"] += 1
    return list(merged.values())


def _judge(normalizer, entries: list[dict]) -> list[dict]:
    """1 件ずつ寄せて、結果を「正しい / 誤り / 触らない」に分ける。"""
    judged = []
    for entry in entries:
        result, how, score = normalizer.normalize(entry["delivered"])
        changed = how in ("exact", "lexical", "nearest")
        if not changed:
            verdict = "触らない"
        elif result == entry["target"]:
            verdict = "正しい"
        else:
            verdict = "誤り"
        judged.append({**entry, "how": how, "score": score,
                       "result": result, "verdict": verdict,
                       "group": _group_of(entry["categories"])})
    return judged


def _print_table(judged: list[dict]) -> dict[str, collections.Counter]:
    """束ごとに、判定の内訳を出す。"""
    by_group: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    weighted: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item in judged:
        by_group[item["group"]][item["verdict"]] += 1
        by_group[item["group"]]["文の種類"] += 1
        weighted[item["group"]][item["verdict"]] += item["variants"]
        weighted[item["group"]]["変種"] += item["variants"]

    order = ("本番の 2 カテゴリ", "その他の摂動", "言い換え")
    print()
    print("| 束 | 文の種類 | 寄せた（正しい） | **誤り** | 触らない |")
    print("|---|---|---|---|---|")
    for group in order:
        counts = by_group.get(group)
        if not counts:
            continue
        print(f"| {group} | {counts['文の種類']} | {counts['正しい']} | "
              f"**{counts['誤り']}** | {counts['触らない']} |")
    print()
    print("**変種の数で重み付けすると**（同じ文が複数の変種に出るため）")
    print()
    print("| 束 | 変種 | 寄せた（正しい） | **誤り** | 触らない |")
    print("|---|---|---|---|---|")
    for group in order:
        counts = weighted.get(group)
        if not counts:
            continue
        print(f"| {group} | {counts['変種']} | {counts['正しい']} | "
              f"**{counts['誤り']}** | {counts['触らない']} |")

    print()
    how_counts = collections.Counter((item["group"], item["how"]) for item in judged)
    print("**どの段で決まったか**")
    for (group, how), count in sorted(how_counts.items()):
        print(f"  {group:<14} {how:<10} {count}")
    return by_group


def _print_mistakes(judged: list[dict], limit: int) -> None:
    """誤って寄せたものを並べる。**採否はここを読んで決める。**"""
    mistakes = [item for item in judged if item["verdict"] == "誤り"]
    if not mistakes:
        print("\n**誤って別のタスクへ寄せたものは無い。**")
        return
    print(f"\n**誤って寄せたもの {len(mistakes)} 件**（点数の高い順に {limit} 件まで）")
    for item in sorted(mistakes, key=lambda entry: -entry["score"])[:limit]:
        print(f"  [{item['group']}] score={item['score']:.3f} ({item['how']})")
        print(f"    渡る文 {item['delivered']!r}")
        print(f"    寄せ先 {item['result']!r}")
        print(f"    正解   {item['target']!r}")


def _sweep(normalizer_class, targets, entries: list[dict]) -> None:
    """しきい値を振って、寄せた数と誤りの数の釣り合いを見る。"""
    print("\n**しきい値を振る**（語の重なりの下限 × 1 番と 2 番の差の下限）")
    print()
    print("| 下限 | 差 | 本番 2 カテゴリ 正/誤 | 言い換え 正/誤 |")
    print("|---|---|---|---|")
    for minimum in (0.0, 0.40, 0.50, 0.60, 0.70, 0.80):
        for margin in (0.0, 0.03, 0.05, 0.10):
            normalizer = normalizer_class(
                targets, min_lexical=minimum, min_lexical_margin=margin
            )
            judged = _judge(normalizer, entries)
            counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
            for item in judged:
                counts[item["group"]][item["verdict"]] += 1
            production = counts["本番の 2 カテゴリ"]
            language = counts["言い換え"]
            print(f"| {minimum:.2f} | {margin:.2f} | "
                  f"{production['正しい']}/{production['誤り']} | "
                  f"{language['正しい']}/{language['誤り']} |")


def _attach_embeddings(normalizer, artifacts: Path) -> None:
    """提出物と同じ道筋で、意味で寄せる道を足す。

    **`MyPolicy` を実際に立ち上げる**（重みを読む。1〜2 分かかる）。
    別に書いた埋め込みでは、提出物の挙動を測ったことにならない。

    **`MyPolicy` は自分の隣に `model_weights/` と `vendor/` がある前提**である
    （提出物の zip ではそうなる）。リポジトリでは `work/artifacts/` にあるので、
    **提出物と同じ形をした作業場を作ってから読み込む。**
    `policy_server.py` は実体をコピーする —— `__file__` を `resolve()` して
    隣を探すため、**シンボリックリンクでは元の場所に戻ってしまう。**

    Parameters
    ----------
    normalizer : InstructionNormalizer
        埋め込みを足す相手。
    artifacts : Path
        `model_weights/` と `vendor/` の親（既定 `work/artifacts`）。
    """
    import importlib.util
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix="parc_norm_"))
    shutil.copy2(_REPO_ROOT / "solution/policy_server.py", staging / "policy_server.py")
    for name in ("model_weights", "vendor"):
        source = (artifacts / name).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"提出物の構成要素がありません: {source}")
        (staging / name).symlink_to(source)

    spec = importlib.util.spec_from_file_location(
        "policy_server_staged", staging / "policy_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print(f"[check] 方策を立ち上げる（{staging}）")
    policy = module.MyPolicy()
    normalizer.attach_embeddings(policy._embed_instructions)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--libero-plus", type=Path, default=_REPO_ROOT / "work/LIBERO-plus",
        help="LIBERO-plus の clone 先",
    )
    parser.add_argument("--sweep", action="store_true", help="しきい値を振る")
    parser.add_argument(
        "--embed", action="store_true",
        help="意味で寄せる道も測る（torch と重みが要る。Docker で回す）",
    )
    parser.add_argument(
        "--artifacts", type=Path, default=_REPO_ROOT / "work/artifacts",
        help="--embed のとき読む model_weights/ と vendor/ の親",
    )
    parser.add_argument("--mistakes", type=int, default=10, help="誤りを何件出すか")
    parser.add_argument("--json", type=Path, default=None, help="内訳の書き出し先")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    normalizer_class, has_numpy = _load_normalizer_class()

    pairs = build_pairs(args.libero_plus)
    targets = training_instructions(pairs)
    entries = _collapse(pairs)
    print(f"[check] 変種 {len(pairs)} 件 / 渡ってくる文 {len(entries)} 種 / "
          f"寄せ先 {len(targets)} 種")

    normalizer = normalizer_class(targets)
    if args.embed:
        if not has_numpy:
            raise RuntimeError(
                "numpy が無いので意味で寄せる道は測れません。"
                "Docker で回してください（docs/howto/03_gpu_eval.md）"
            )
        _attach_embeddings(normalizer, args.artifacts)
        print("[check] 意味で寄せる道: 入り（同梱の SmolVLM の言語側）")
    else:
        print("[check] 意味で寄せる道: 切り"
              "（測るには --embed。torch が要る）")

    judged = _judge(normalizer, entries)
    _print_table(judged)
    _print_mistakes(judged, args.mistakes)

    if args.sweep:
        _sweep(normalizer_class, targets, entries)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [{k: (sorted(v) if isinstance(v, set) else v) for k, v in item.items()}
                 for item in judged],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n[check] 書き出し {args.json}")

    # **本番の 2 カテゴリで 1 件でも誤って寄せたら、落とす。**
    # そこはいま点が取れている場所なので、壊す変更を黙って通さない。
    wrong = [item for item in judged
             if item["verdict"] == "誤り" and item["group"] == "本番の 2 カテゴリ"]
    if wrong:
        raise SystemExit(
            f"[check] 本番の 2 カテゴリで {len(wrong)} 件を誤って寄せています。"
            "しきい値を上げるか、寄せ方を直してください"
        )


if __name__ == "__main__":
    main()
