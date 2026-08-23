#!/usr/bin/env python3
"""重みを退避先の HuggingFace リポジトリへ上げ、索引に 1 行足す。

**既定の上げ先はもう存在しない。`--repo-id` を必ず渡すこと。**
`siruku6/parc2026-archive` は 8/18 に消した（commit `c3b8f1e`）。付けずに
実行すると `RepositoryNotFoundError` で落ちる —— ここが捕まえているのは
`EntryNotFoundError` だけである。**既定値を直していないのは、8/18 に
「重み 1 つごとに 1 リポジトリ」へ変えたので、既定にすべき 1 つが
原理的に無いためである。**

**索引の役目も、もうここには無い。** `index.json` に替わって
`docs/reference/19_weights_catalog.md` が索引そのものになっている。
8/18 以降に作った重みは `PROVENANCE.md` ではなく `INFERENCE.md`
（推論の設定）を持つので、**このツールを既存のリポジトリへ向けると
2 つの形が混ざる。**

**それでも残してあるのは、下の歯止めのためである。**

**このツールを通す限り、「どれがどれだか分からない重み」は退避先に増えない。**
`train_config.json` を持たない重みは受け付けないためである（`--no-train-config`
と `--note` を両方付けたときだけ、理由付きで例外にできる）。

`work/artifacts/model_weights_*` は **書き換えない**。あそこへファイルを足すと
`build_submission.py` がそれを zip に入れてしまう（除外規則に `.md` も `.json`
も無い）。記録は退避先の側だけに持たせる。

索引の一次資料は `index.json` である。`README.md` はそこから毎回作り直すので、
手で直しても次回の実行で消える。**直すなら `index.json` を直す。**

    python solution/tools/weights/archive_push.py work/artifacts/model_weights_weak3x \\
        --as smolvla/weak3x_20000 \\
        --train-config work/train2/weak3x_max300/checkpoints/020000/pretrained_model/train_config.json \\
        --source work/train2/weak3x_max300/checkpoints/020000 \\
        --score 0.722 --score-n 36 \\
        --score-run 0814_101939_weak3x_n8_ep1_max300
    python solution/tools/weights/archive_push.py ... --dry-run   # 何を上げるかだけ見る

Reads:
  - <source-dir>/**（上げる重み一式）
  - <--train-config>（設定の一次資料。重みの隣に無いとき明示する）
  - 退避先の index.json
Generates:
  - 退避先の <--as>/**（重み一式 + train_config.json + PROVENANCE.md）
  - 退避先の index.json（1 行追加）
  - 退避先の README.md（index.json から作り直す）
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError

# solution/tools/weights/archive_push.py の 3 階層上がリポジトリルート。
_REPO_ROOT = Path(__file__).resolve().parents[3]

# **このリポジトリは 8/18 に消してある。** 残してあるのは「何を既定にすべきか
# 決まっていない」ことを見えるようにするためで、実行時は `--repo-id` で指す。
_DEFAULT_REPO_ID = "siruku6/parc2026-archive"

# 索引の一次資料と、そこから作る読みもの。
_INDEX_PATH = "index.json"
_README_PATH = "README.md"

# 設定の一次資料。これが無い重みは受け付けない。
_TRAIN_CONFIG_NAME = "train_config.json"

# 実行時に生成されるだけのもの。上げても意味がない。
# `build_submission.py` の除外規則と同じ考えで揃えてある。
_EXCLUDE_PARTS = frozenset(
    {"__pycache__", ".hf_home", ".ipynb_checkpoints", ".DS_Store", ".cache"}
)
_EXCLUDE_SUFFIXES = frozenset({".pyc"})

# 重みの同一性を確かめる相手。ここが一致すれば「同じ重み」と言い切れる。
_IDENTITY_FILE = "model.safetensors"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source",
        type=Path,
        help="上げる重みのディレクトリ（例: work/artifacts/model_weights_weak3x）",
    )
    parser.add_argument(
        "--as",
        dest="folder",
        required=True,
        help="退避先での置き場所（例: smolvla/weak3x_20000）。"
             "**実行名とステップ数が分かる名前にする。**",
    )
    parser.add_argument(
        "--repo-id",
        default=_DEFAULT_REPO_ID,
        help="退避先のリポジトリ。**必ず渡すこと。** "
             f"既定の {_DEFAULT_REPO_ID} は 8/18 に消してあるので、"
             "省くと RepositoryNotFoundError で落ちる",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help=f"設定の一次資料。重みの隣に {_TRAIN_CONFIG_NAME} が無いとき、"
             "たどった学習実行のものを指す",
    )
    parser.add_argument(
        "--no-train-config",
        action="store_true",
        help="設定の一次資料が手元に無いことを承知で上げる。"
             "**--note で理由を書かないと通らない。**",
    )
    parser.add_argument(
        "--origin",
        default=None,
        help="重みを作った学習実行の場所（既定: --train-config の親、"
             "無ければ source そのもの）",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="この重みが何ステップ目か。既定では --origin の数字だけのディレクトリ名から"
             "読む。**train_config.json の steps は run の予定総ステップ数であって"
             "これではない。**",
    )
    parser.add_argument("--score", default=None, help="評価の成功率（例: 0.722）")
    parser.add_argument("--score-n", default=None, help="評価した件数（例: 36）")
    parser.add_argument(
        "--score-run",
        default=None,
        help="その成功率を出した work/gpu_eval/ の実行名",
    )
    parser.add_argument("--note", default=None, help="索引に残す短い説明")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="何を上げるかだけ表示し、退避先へは書かない",
    )
    return parser.parse_args()


def _collect_files(directory: Path) -> list[tuple[Path, str]]:
    """上げるファイルを、退避先での相対パスと組にして集める。

    Parameters
    ----------
    directory : Path
        重みのディレクトリ。

    Returns
    -------
    entries : list[tuple[Path, str]]
        (手元のパス, ディレクトリ内での相対パス) の並び。
    """
    entries: list[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if _EXCLUDE_PARTS & set(path.parts):
            continue
        if path.suffix in _EXCLUDE_SUFFIXES:
            continue
        entries.append((path, str(path.relative_to(directory))))
    return entries


def _hash_file(path: Path) -> str:
    """ファイル全体の sha256 を返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_train_config(source: Path, args: argparse.Namespace) -> Path | None:
    """設定の一次資料を決める。無いまま上げようとしていれば止める。

    Parameters
    ----------
    source : Path
        重みのディレクトリ。
    args : argparse.Namespace
        `--train-config` / `--no-train-config` / `--note` を見る。

    Returns
    -------
    train_config : Path | None
        一緒に上げる `train_config.json`。`--no-train-config` のときは None。
    """
    beside = source / _TRAIN_CONFIG_NAME
    if args.train_config is not None:
        if not args.train_config.is_file():
            raise FileNotFoundError(f"設定が見つかりません: {args.train_config}")
        return args.train_config.resolve()
    if beside.is_file():
        return beside
    if args.no_train_config:
        if not args.note:
            raise SystemExit(
                f"{_TRAIN_CONFIG_NAME} 無しで上げるなら --note で理由を書いてください。"
                "\n  書かないと、退避先でも「どれがどれだか分からない重み」になります。"
            )
        return None
    raise SystemExit(
        f"{_TRAIN_CONFIG_NAME} がありません: {beside}\n"
        "  たどった学習実行のものを --train-config で指してください。\n"
        f"  手元に一次資料が無いなら --no-train-config と --note を両方付けてください。"
    )


def _load_index(api: HfApi, repo_id: str) -> list[dict]:
    """退避先の索引を読む。まだ無ければ空の索引を返す。"""
    try:
        path = api.hf_hub_download(repo_id=repo_id, filename=_INDEX_PATH)
    except EntryNotFoundError:
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_entry(args: argparse.Namespace, source: Path,
                 train_config: Path | None, identity: str,
                 total_bytes: int) -> dict:
    """索引に足す 1 行を組む。

    Parameters
    ----------
    args : argparse.Namespace
        コマンド行の指定。
    source : Path
        重みのディレクトリ。
    train_config : Path | None
        一緒に上げる設定の一次資料。
    identity : str
        `model.safetensors` の sha256。
    total_bytes : int
        上げるファイルの合計サイズ。

    Returns
    -------
    entry : dict
        `index.json` に並べる 1 件。
    """
    origin = args.origin
    if origin is None:
        origin = str(train_config.parent) if train_config else str(source)
    planned_steps: int | None = None
    dataset: str | None = None
    base: str | None = None
    if train_config is not None:
        config = json.loads(train_config.read_text(encoding="utf-8"))
        planned_steps = config.get("steps")
        dataset = (config.get("dataset") or {}).get("repo_id")
        base = (config.get("policy") or {}).get("pretrained_path")
    step = args.step if args.step is not None else _read_step(origin)
    return {
        "folder": args.folder,
        "local_path": str(_relative_to_repo(source)),
        "origin": str(_relative_to_repo(Path(origin))),
        "step": step,
        "planned_steps": planned_steps,
        "dataset": dataset,
        "base_model": base,
        "has_train_config": train_config is not None,
        "sha256_model": identity,
        "total_bytes": total_bytes,
        "score": args.score,
        "score_n": args.score_n,
        "score_run": args.score_run,
        "note": args.note,
        "pushed_on": date.today().isoformat(),
    }


def _read_step(origin: str) -> int | None:
    """学習実行の場所から、この重みが何ステップ目かを読む。

    lerobot は `checkpoints/024000/` のようにステップ数をディレクトリ名にする。
    **`train_config.json` の `steps` は run の予定総ステップ数**であって、途中で
    取り出した重みのステップ数ではない。混同すると 500 ステップの重みが 24000 ステップに見える。

    Parameters
    ----------
    origin : str
        重みを作った学習実行の場所。

    Returns
    -------
    step : int | None
        数字だけのディレクトリ名が見つかればその値。無ければ None。
    """
    for part in reversed(Path(origin).parts):
        if part.isdigit():
            return int(part)
    return None


def _format_steps(entry: dict) -> str:
    """ステップ数の欄を、予定総ステップ数と混ざらない書き方に整える。"""
    step = entry.get("step")
    planned = entry.get("planned_steps")
    if step is not None and planned is not None:
        return str(step) if step == planned else f"{step} / 予定 {planned}"
    if step is not None:
        return str(step)
    if planned is not None:
        return f"不明 / 予定 {planned}"
    return "—"


def _relative_to_repo(path: Path) -> Path:
    """リポジトリルートからの相対パスにする。外にあるならそのまま返す。"""
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return resolved


def _render_provenance(entry: dict) -> str:
    """この重みが何であるかを、フォルダの中に置く文章として組む。"""
    lines = [
        f"# {entry['folder']}",
        "",
        f"- 上げた日: {entry['pushed_on']}",
        f"- 手元の置き場: `{entry['local_path']}`",
        f"- 作った学習実行: `{entry['origin']}`",
        f"- `model.safetensors` の sha256: `{entry['sha256_model']}`",
    ]
    if entry.get("step") is not None:
        lines.append(f"- この重みは {entry['step']} 歩目")
    if entry.get("planned_steps") is not None:
        lines.append(f"- その学習実行の予定総ステップ数: {entry['planned_steps']}")
    if entry["dataset"]:
        lines.append(f"- 学習データ: `{entry['dataset']}`")
    if entry["base_model"]:
        lines.append(f"- 初期重み: `{entry['base_model']}`")
    if entry["score"]:
        measured = f"- 評価: 成功率 {entry['score']}"
        if entry["score_n"]:
            measured += f"（n={entry['score_n']}）"
        if entry["score_run"]:
            measured += f" 実測は `work/gpu_eval/{entry['score_run']}/summary.txt`"
        lines.append(measured)
    if not entry["has_train_config"]:
        lines += [
            "",
            f"**{_TRAIN_CONFIG_NAME} が無い。** 設定の一次資料が手元に残って"
            "いないため、上のステップ数や学習データは埋まっていない。",
        ]
    if entry["note"]:
        lines += ["", entry["note"]]
    lines += [
        "",
        "この文章は `solution/tools/weights/archive_push.py` が生成した。手で直さない。",
    ]
    return "\n".join(lines) + "\n"


def _load_front_matter(api: HfApi, repo_id: str) -> str:
    """既にあるモデルカードの YAML 前付けを読む。

    README は毎回作り直すので、ライセンス表記のように**人が書いた前付けは
    そのまま持ち越さないと消える。**前付けが無ければ空文字を返す。
    """
    try:
        path = api.hf_hub_download(repo_id=repo_id, filename=_README_PATH)
    except EntryNotFoundError:
        return ""
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return ""
    end = text.find("\n---\n", 3)
    if end == -1:
        return ""
    return text[: end + len("\n---\n")] + "\n"


def _render_readme(index: list[dict], repo_id: str,
                   front_matter: str = "") -> str:
    """索引を、一覧して選べる表として組み直す。"""
    header = [
        f"# {repo_id}",
        "",
        "学習した重みの退避先。**`solution/tools/weights/archive_push.py` だけが"
        "ここへ書く。**",
        "",
        "一次資料は [index.json](index.json) である。この README はそこから"
        "毎回作り直すので、**手で直しても次回の実行で消える。**",
        "",
        "各フォルダには重み一式のほかに `train_config.json`（設定の一次資料）と"
        "`PROVENANCE.md`（何であるかの説明）が入っている。",
        "",
        "**成功率は実行ごとに分母が違う。** 数字だけを並べて比べると誤る。"
        "必ず右の実測ログを見て、同じ条件どうしか確かめること。",
        "",
        "ステップ数の欄が `500 / 予定 24000` の形になっているものは、**24000 ステップ回す"
        "つもりだった学習の 500 ステップ目**である。`train_config.json` の `steps` は"
        "予定の方なので、そこだけ見ると取り違える。",
        "",
        "| フォルダ | ステップ数 | 学習データ | 成功率 | 実測ログ | 手元の置き場 | 設定 | 上げた日 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for entry in sorted(index, key=lambda item: item["folder"]):
        score = "—"
        if entry.get("score"):
            score = str(entry["score"])
            if entry.get("score_n"):
                score += f" (n={entry['score_n']})"
        rows.append(
            "| [{folder}]({folder}/) | {steps} | {dataset} | {score} | {run} "
            "| `{local}` | {config} | {pushed} |".format(
                folder=entry["folder"],
                steps=_format_steps(entry),
                dataset=f"`{entry['dataset']}`" if entry.get("dataset") else "—",
                score=score,
                run=(f"`work/gpu_eval/{entry['score_run']}/`"
                     if entry.get("score_run") else "—"),
                local=entry["local_path"],
                config="✅" if entry.get("has_train_config") else "**なし**",
                pushed=entry["pushed_on"],
            )
        )
    footer = [
        "",
        "## 上げ方",
        "",
        "```bash",
        "python solution/tools/weights/archive_push.py <重みのディレクトリ> \\",
        "    --as <実行名>/<ステップ数> \\",
        "    --train-config <たどった学習実行>/train_config.json \\",
        "    --score <成功率> --score-n <件数> --score-run <gpu_eval の実行名>",
        "```",
        "",
        f"`{_TRAIN_CONFIG_NAME}` を渡さないと**弾かれる**。"
        "これが「どれがどれだか分からない」を防ぐ唯一の仕掛けである。",
    ]
    return front_matter + "\n".join(header + rows + footer) + "\n"


def _print_plan(entry: dict, files: list[tuple[Path, str]],
                train_config: Path | None) -> None:
    """上げる内容の要約を出す。"""
    print(f"[archive] 退避先の置き場  {entry['folder']}")
    print(f"[archive] 手元の置き場    {entry['local_path']}")
    print(f"[archive] 作った学習実行  {entry['origin']}")
    print(f"[archive] ステップ数            {_format_steps(entry)}")
    print(f"[archive] 学習データ      {entry['dataset']}")
    print(f"[archive] sha256          {entry['sha256_model']}")
    print(f"[archive] 設定            "
          f"{train_config if train_config else '**なし（--no-train-config）**'}")
    print(f"[archive] ファイル        {len(files)} 個 / "
          f"{entry['total_bytes'] / 1e9:.2f} GB")


def _make_operations(files: list[tuple[Path, str]], folder: str,
                     train_config: Path | None, entry: dict,
                     index: list[dict], repo_id: str,
                     front_matter: str = ""
                     ) -> list[CommitOperationAdd]:
    """1 回の書き込みにまとめる操作を組む。

    Parameters
    ----------
    files : list[tuple[Path, str]]
        重みのファイルと、ディレクトリ内での相対パス。
    folder : str
        退避先での置き場所。
    train_config : Path | None
        一緒に置く設定の一次資料。
    entry : dict
        索引に足す 1 件。
    index : list[dict]
        追加後の索引全体。
    repo_id : str
        退避先のリポジトリ。
    front_matter : str
        既にあるモデルカードの YAML 前付け。ライセンス表記を消さないために
        そのまま先頭へ戻す。

    Returns
    -------
    operations : list[CommitOperationAdd]
        重み・設定・説明・索引・README をまとめた操作の並び。
    """
    operations = [
        CommitOperationAdd(path_in_repo=f"{folder}/{name}", path_or_fileobj=str(path))
        for path, name in files
    ]
    if train_config is not None:
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{folder}/{_TRAIN_CONFIG_NAME}",
                path_or_fileobj=str(train_config),
            )
        )
    operations.append(
        CommitOperationAdd(
            path_in_repo=f"{folder}/PROVENANCE.md",
            path_or_fileobj=_render_provenance(entry).encode("utf-8"),
        )
    )
    operations.append(
        CommitOperationAdd(
            path_in_repo=_INDEX_PATH,
            path_or_fileobj=json.dumps(
                index, ensure_ascii=False, indent=2
            ).encode("utf-8"),
        )
    )
    operations.append(
        CommitOperationAdd(
            path_in_repo=_README_PATH,
            path_or_fileobj=_render_readme(
                index, repo_id, front_matter
            ).encode("utf-8"),
        )
    )
    return operations


def main() -> None:
    args = _parse_args()
    source: Path = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"重みのディレクトリがありません: {source}")

    train_config = _check_train_config(source, args)
    files = _collect_files(source)
    identity_path = source / _IDENTITY_FILE
    if not identity_path.is_file():
        raise FileNotFoundError(f"{_IDENTITY_FILE} がありません: {identity_path}")

    print(f"[archive] {_IDENTITY_FILE} の sha256 を計算しています…")
    identity = _hash_file(identity_path)
    total_bytes = sum(path.stat().st_size for path, _ in files)

    entry = _build_entry(args, source, train_config, identity, total_bytes)
    _print_plan(entry, files, train_config)

    if args.dry_run:
        print("[archive] --dry-run のため、退避先へは書きませんでした。")
        return

    api = HfApi()
    index = [item for item in _load_index(api, args.repo_id)
             if item["folder"] != args.folder]
    index.append(entry)

    operations = _make_operations(
        files, args.folder, train_config, entry, index, args.repo_id,
        _load_front_matter(api, args.repo_id),
    )
    print(f"[archive] {len(operations)} 個を 1 回で書き込みます…")
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"archive {args.folder}",
    )
    print(f"[archive] 完了  https://huggingface.co/{args.repo_id}/tree/main/{args.folder}")


if __name__ == "__main__":
    main()
