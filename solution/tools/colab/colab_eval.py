#!/usr/bin/env python3
"""Colab で 84 件の検証用データを評価する手順を、セルから関数へ畳んだもの。

**貼り替える対象を無くすために作った。** 手順が 10 セル・1,400 行あると、
どのセルが新しいのか分からなくなる。8/11 に 3 度その事故が起きた
（`patch_kit_for_variants.py` の修正版が `work/colab_upload/` にしか無かった／
`N_EPISODES` の既定が 20 のままだった／確かめたい行が警告で押し出された）。

**セルの中の値を書き換える形もやめた。** 編集するのは
`colab_eval_config.py` だけで、この関数群は**呼ぶたびに読み直す**。
上げ直して実行し直せば効く（ランタイムの再起動は要らない）。

畳んだことで、手順の順番に由来する事故が構造的に消える。

===================================  ==========================================
セルでやっていたときの事故            なぜ起きなくなるか
===================================  ==========================================
CHUNK を書き換えて手順 6 を流し忘れ    `run(chunk=2)` が CSV も出力先も同時に決める
手順 8 を飛ばして手順 9 を実行         1 回の呼び出しで評価と集計を通す
退避先の名前と中身が食い違う           名前を作るのも中身を書くのも同じ関数
`MAX_TASKS = 5` の戻し忘れ            疎通確認は `run(chunk=1, max_tasks=5)`
===================================  ==========================================

使い方（ノートブック側はこれだけ）::

    import importlib, sys
    sys.path.insert(0, "/content")
    import colab_eval as ce
    importlib.reload(ce)              # 上げ直したときのため

    ce.prepare()          # Drive をつなぐ・上がっているか確かめる・EGL を直す
    ce.setup_kit()        # clone・setup.sh・パッチ
    ce.prepare_weights()  # Drive から複製、または HF から取る（ハッシュを出す）
    ce.setup_policy()     # 方策サーバーの依存と配置
    ce.build_suite()      # 検証用スイートへ差し替え、塊に割る
    ce.run(chunk=1)       # 評価 → 集計 → Drive へ退避
    ce.run(chunk=2)
    ce.run(chunk=3)

    ce.status()           # どの塊が終わっているか

Reads:
  - /content/colab_eval_config.py（**編集するのはここだけ**）
  - /content/patch_kit_for_variants.py, /content/rollout_recorder.py
  - /content/policy_server.py, /content/make_validation_suite.py
  - /content/check_validation_suite.py, /content/summarize_rollout_videos.py
  - /content/VAL_TASKS.csv（あれば使う。前の回と同じ 84 件で比べるため）
Generates:
  - /content/PARC2026_pre/（配布キット）, /content/model_weights/
  - /content/submission/, /content/chunks/VAL_*.csv
  - /content/results/chunk<N>/server_8000.json, /content/setup_logs/*.log
  - /content/drive/MyDrive/parc2026/eval/live/*.results.txt（1 本ごと）
  - /content/drive/MyDrive/parc2026/eval/<日時>_<設定>/（塊ごとに一式）
"""

from __future__ import annotations

import ctypes
import csv
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# **版。** 画面に出るこの数字と、リポジトリのこのファイルの数字が合っていなければ
# 古いものを上げている。貼り替えの取り違えは、これでしか止められない。
__version__ = "1.1.0"

CONTENT = Path("/content")
KIT_DIR = CONTENT / "PARC2026_pre"
KIT_TASKS = KIT_DIR / "compe/t1/T1_TASKS.csv"
KIT_TASKS_BACKUP = KIT_DIR / "compe/t1/T1_TASKS.csv.example"
LOG_DIR = CONTENT / "setup_logs"
WEIGHTS_DIR = CONTENT / "model_weights"
SUBMISSION_DIR = CONTENT / "submission"
SERVER = SUBMISSION_DIR / "policy_server.py"
LEROBOT_DIR = CONTENT / "lerobot"
LEROBOT_SRC = LEROBOT_DIR / "src"
CHUNK_DIR = CONTENT / "chunks"
VAL_TASKS = CONTENT / "VAL_TASKS.csv"
SERVER_LOG = CONTENT / "server.log"
DRIVE_ROOT = Path("/content/drive/MyDrive")
EVAL_ROOT = DRIVE_ROOT / "parc2026/eval"

LEROBOT_TAG = "v0.6.0"


@dataclass(frozen=True)
class PolicySpec:
    """模型の種類ごとに違うところだけを集めた表。

    **どちらを積むかは重みを落とす前に決まっていないといけない。** lerobot の
    extra（`[smolvla]` か `[pi]`）も、必須ファイルの名前も、重みを見る前に要る。
    だから `config.json` から後で読む形にはできず、`colab_eval_config.py` の
    `POLICY_KIND` で明示してもらう。
    """

    kind: str
    weights_repo: str
    weights_revision: str | None
    vlm_repo: str | None
    lerobot_extra: str
    normalizer_file: str
    default_n_action_steps: str
    patch_bf16: bool
    known_digests: dict[str, str]


POLICY_SPECS: dict[str, PolicySpec] = {
    # Track1。8/11 に 84 件で実証済みの経路。
    "smolvla": PolicySpec(
        kind="smolvla",
        weights_repo="lerobot/smolvla_libero_plus",
        weights_revision="7bb70aa5bc92b82c9239142775d3a173103567ff",
        # VLM 本体の重みはチェックポイントに含まれる。設定とトークナイザだけ別に落とす。
        vlm_repo="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        lerobot_extra="smolvla",
        normalizer_file="policy_preprocessor_step_5_normalizer_processor.safetensors",
        default_n_action_steps="5",
        # bf16 非対応の GPU（T4 など）では fp16 へ書き換える。**SmolVLA 専用の処置**
        # で、π0.5 には効かない（書き換える対象のファイルが smolvla/ の下にある）。
        patch_bf16=True,
        known_digests={
            "a67a056a": "素の重み（8/11 に評価済み。成功率 0.476）",
            "2cd9faab": "lora20k（4 回目の提出。本番 0.016。評価済み）",
            "952171f9": "λ=0.25（素の重み ＋ 学習で動いた分 × 0.25）",
        },
    ),
    # Track2。**重みは HF の非公開リポジトリに `vlm/` 込みで置いてある**ので、
    # Drive を経由せず `snapshot_download` で引ける（docs/reference/19_weights_catalog.md）。
    # 非公開なので `hf auth login` か HF_TOKEN が要る。
    "pi05": PolicySpec(
        kind="pi05",
        weights_repo="siruku6/pi05_stage1_24000",
        weights_revision=None,
        vlm_repo=None,
        # π0.5 は `[pi]`（π0/π0.5 が要る依存一式）。手元の学習イメージと同じ
        # 組み合わせにしてある（solution/train/Dockerfile.pi05 の v0.6.0 + [training,pi]）。
        lerobot_extra="pi",
        # **SmolVLA は step_5 だが π0.5 は step_2 である。**
        # 実物: work/artifacts/model_weights_pi05_024000/
        normalizer_file="policy_preprocessor_step_2_normalizer_processor.safetensors",
        # policy_server.py は π0.5 のときだけ 16 を既定にする（版 3.9.0）。
        default_n_action_steps="16",
        patch_bf16=False,
        known_digests={
            "d46bffcf": "pi05/full_local_024000（1 回目の学習 24,000 歩。本番 0.264・手元36件 0.667〜0.722）",
            "0c5d542e": "pi05/full_runpod_003000（予選の最終提出。本番 0.485・手元36件 0.750）",
            "89fc796d": "C0 baseline 16,000 歩（量子化スイープの対照群）",
        },
    ),
}

# 手元の collect_colab_files.py が集めて、手で /content/ へ置くもの。
UPLOADED_FILES: tuple[str, ...] = (
    "colab_eval_config.py",
    "patch_kit_for_variants.py",
    "rollout_recorder.py",
    "policy_server.py",
    "make_validation_suite.py",
    "check_validation_suite.py",
    "summarize_rollout_videos.py",
)

# 推論に必須のファイル。train_config.json は学習スクリプトが書くもので、
# マージ済みモデル（save_pretrained の出力）には無いため必須にしない。
# **正規化の safetensors だけは模型によって名前が違う**ので `PolicySpec` から足す。
_COMMON_WEIGHT_FILES: frozenset[str] = frozenset({
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
})

_HEARTBEAT_SEC = 60


# ============================================================
# 設定と、共通の道具
# ============================================================


def _load_config() -> ModuleType:
    """編集用の設定ファイルを**毎回読み直して**返す。

    読み直すのは、`colab_eval_config.py` を上げ直しただけで効かせるため。
    import したまま使うと、直した値が反映されずに前の設定で 51 分回すことになる。

    Reads:
      - /content/colab_eval_config.py
    """
    try:
        import colab_eval_config
    except ModuleNotFoundError as error:  # pragma: no cover - Colab 専用
        raise RuntimeError(
            "colab_eval_config.py が無い。collect_colab_files.py で集めて "
            "/content/ へ上げること"
        ) from error
    return importlib.reload(colab_eval_config)


def _setting(config: ModuleType, name: str) -> object:
    """設定を 1 つ取り出す。無ければ**何を足すか名指しで**止める。"""
    if not hasattr(config, name):
        raise RuntimeError(
            f"colab_eval_config.py に {name} が無い。"
            "古い設定ファイルを上げている可能性がある"
        )
    return getattr(config, name)


def _spec(config: ModuleType) -> PolicySpec:
    """どの模型を積むかを設定から決め、その差分表を返す。

    Raises
    ------
    RuntimeError
        知らない種類が書かれているとき。**黙って Track1 の既定で走らせない**
        —— 別の模型の必須ファイル一覧で照合すると、通ってしまった後に
        「重みが足りない」ではなく推論時の別のエラーになって原因が遠くなる。
    """
    kind = str(_setting(config, "POLICY_KIND")).strip()
    if kind not in POLICY_SPECS:
        raise RuntimeError(
            f"POLICY_KIND={kind!r} は知らない。使えるのは "
            f"{sorted(POLICY_SPECS)} である"
        )
    return POLICY_SPECS[kind]


def _print_header(step: str) -> None:
    """どの版のどの手順を実行しているかを、必ず 1 行目に出す。"""
    print(f"[colab_eval {__version__}] {step}")


def _shell(
    command: str,
    *,
    check: bool = True,
    tail: int = 30,
    log: str | None = None,
    cwd: Path | None = None,
) -> int:
    """シェルコマンドを 1 つ実行し、失敗したときだけ末尾を長く見せる。

    Parameters
    ----------
    command : str
        `shell=True` で渡す文字列。
    check : bool
        真なら失敗時に RuntimeError を投げる。**握りつぶさない**のが既定。
    tail : int
        失敗したときに画面へ出す末尾の行数。
    log : str | None
        全文を残すファイル名。`/content/setup_logs/` の下に置く。
    cwd : Path | None
        実行するディレクトリ。

    Returns
    -------
    returncode : int
        終了コード。

    Generates:
      - /content/setup_logs/<log>
    """
    print(f"$ {command}")
    LOG_DIR.mkdir(exist_ok=True)
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        cwd=str(cwd) if cwd else None,
    )
    output = result.stdout + result.stderr

    if log:
        (LOG_DIR / log).write_text(output, encoding="utf-8")

    lines = output.strip().splitlines()
    if result.returncode != 0:
        print(f"--- 失敗 (returncode={result.returncode}) 末尾 {tail} 行 ---")
        print("\n".join(lines[-tail:]))
        if log:
            print(f"--- 全文: {LOG_DIR / log} ---")
        if check:
            raise RuntimeError(f"失敗 ({result.returncode}): {command}")
    else:
        print("\n".join(lines[-4:]) if lines else "(出力なし)")
    return result.returncode


def _grep_log(name: str, needle: str) -> list[str]:
    """ログから確かめたい行だけを抜く。

    `_shell` が出すのは末尾 4 行だけで、そこは robosuite と gym の警告で埋まる。
    **確かめたい行が押し出されて見えない**という事故を 8/11 に踏んだので、
    名指しで拾って出し直す。
    """
    path = LOG_DIR / name
    if not path.is_file():
        return []
    return [
        line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if needle in line
    ]


def _mount_drive() -> bool:
    """Drive をつなぐ。つながっていれば何もしない。

    Returns
    -------
    mounted : bool
        使える状態かどうか。
    """
    if (DRIVE_ROOT).is_dir():
        return True
    try:  # pragma: no cover - Colab 専用
        from google.colab import drive

        drive.mount(str(DRIVE_ROOT.parent))
    except Exception as error:  # pragma: no cover - Colab 専用
        print(f"**Drive をつなげない: {error}**")
        return False
    return DRIVE_ROOT.is_dir()


def _digest(path: Path) -> str:
    """ファイルの sha256 を**少しずつ読んで**求める。

    `read_bytes()` で一気に読むと、π0.5 の `model.safetensors`（9.4 GB）を
    載せた時点で Colab の RAM（T4 のランタイムは 12.7 GB）が尽きて、
    プロセスが SIGKILL される。SmolVLA の 912 MB では起きなかった事故である。

    Reads:
      - path（1 MiB ずつ）
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(path: Path, message: str) -> Path:
    """前の手順が作ったはずのものを確かめる。無ければ**どの手順か名指しで**止める。"""
    if not path.exists():
        raise RuntimeError(f"{path} が無い。{message}")
    return path


def _count_tasks(csv_path: Path) -> int:
    """タスク定義 CSV の行数（見出しを除く）を返す。"""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return max(0, sum(1 for _ in stream) - 1)


# ============================================================
# 1. 準備
# ============================================================


def prepare() -> None:
    """評価を始められる状態にする —— 上がっているか、Drive、GPU と EGL。

    **EGL の登録がここで決まる。** NVIDIA の登録が無いと MuJoCo は Mesa の
    ソフトウェア実装を掴み、**例外も警告も出さずに 11 倍遅くなる。**

    Reads:
      - /content/<UPLOADED_FILES>
    Generates:
      - /usr/share/glvnd/egl_vendor.d/10_nvidia.json
    """
    _print_header("1. 準備")

    missing = [name for name in UPLOADED_FILES if not (CONTENT / name).is_file()]
    print(f"上がっているか: {'不足 ' + ', '.join(missing) if missing else '7 件そろっている'}")
    if missing:
        raise RuntimeError(
            "上のファイルを /content/ へドラッグしてから進むこと"
            "（手元で collect_colab_files.py を実行して集める）"
        )

    print(
        "VAL_TASKS.csv: "
        + ("あり（前の回と同じ 84 件で比べられる）" if VAL_TASKS.is_file()
           else "**無い。build_suite() が作り直す**")
    )

    # **先につないでおく。** 後回しにすると run() の途中で認証を求められて止まる。
    print(f"Drive: {'つながった' if _mount_drive() else '**つながっていない**'}")

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"GPU: {gpu or '**出ない。ランタイムのタイプが GPU になっていない**'}")
    print(f"python: {sys.version.split()[0]}")
    if not gpu:
        raise RuntimeError("ランタイムのタイプを GPU に変更してから進むこと")

    try:
        ctypes.CDLL("libEGL.so.1")
        print("libEGL: あり")
    except OSError:
        print("libEGL: 無し → setup_kit() の apt で入る")

    # **ここが Colab の最大の罠。** libglvnd は「ローダ」でしかなく、どのベンダ
    # 実装を使うかは egl_vendor.d の JSON が決める。
    icd = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if icd.is_file():
        print(f"{icd}: 既にある")
    else:
        icd.parent.mkdir(parents=True, exist_ok=True)
        icd.write_text(
            '{\n'
            '    "file_format_version" : "1.0.0",\n'
            '    "ICD" : {\n'
            '        "library_path" : "libEGL_nvidia.so.0"\n'
            '    }\n'
            '}\n'
        )
        print(f"{icd}: 作成した（NVIDIA の EGL を使わせる）")


# ============================================================
# 2. 配布キット
# ============================================================


def _kit_python() -> str:
    """評価パイプライン側の Python を決める。

    キットが gym==0.25.2 / robosuite==1.4.0 という古い版を固定しているため、
    素性の分かっている 3.10 の方が安全。**採点環境への忠実性の話ではない**
    （それは Docker 側で済んでいる）。
    """
    return shutil.which("python3.10") or sys.executable


def setup_kit() -> None:
    """配布キットを取得し、`setup.sh` を通してパッチを当てる（5〜10 分）。

    **キットは `git clone` で取るため、リポジトリ側の修正は入っていない。**
    パッチは `/content/patch_kit_for_variants.py` が当てる —— Docker で
    確かめたものとまったく同じコードが当たる。

    Reads:
      - /content/patch_kit_for_variants.py, /content/rollout_recorder.py
    Generates:
      - /content/PARC2026_pre/（clone と venv）
      - /content/setup_logs/{apt_*,setup,patch,verify}.log
    """
    _print_header("2. 配布キットのセットアップ")
    _require(CONTENT / "patch_kit_for_variants.py", "prepare() を先に通すこと")

    _shell("apt-get -qq update", log="apt_update.log")
    _shell(
        "apt-get -qq install -y libosmesa6 libegl1 libglfw3 libglew-dev "
        "libmagickwand-dev unzip build-essential",
        log="apt_system.log",
    )
    _shell(
        "apt-get -qq install -y python3.10 python3.10-venv python3.10-dev",
        check=False, log="apt_python310.log",
    )

    kit_python = _kit_python()
    print(f"\nキット側の Python: {kit_python}")
    if "3.10" not in kit_python:
        print("  python3.10 が使えないため Colab の Python を使う。"
              "gym / robosuite が古いため、後段で落ちたらログを見ること。")

    if not KIT_DIR.exists():
        _shell(
            f"git clone --depth 1 https://github.com/matsuolab/PARC2026_pre {KIT_DIR}"
        )

    # Colab は MPLBACKEND に matplotlib_inline を設定している。キットの venv には
    # そのパッケージが無いため、libero が matplotlib を import した瞬間に落ちる。
    # 評価は画面表示をしないので、以降ずっと Agg にしておく。
    os.environ["MPLBACKEND"] = "Agg"

    _shell(f"PYTHON={kit_python} bash setup.sh", tail=40, log="setup.log", cwd=KIT_DIR)

    # torch 2.11.0 は setuptools<82 を要求するが、Colab の venv には新しいものが
    # 入る。警告のまま放置すると import で落ちることがあるため揃えておく。
    _shell(f"{KIT_DIR}/venv/bin/pip install -q 'setuptools<82'", check=False)

    _shell(
        f"{kit_python} /content/patch_kit_for_variants.py --kit-dir {KIT_DIR}",
        tail=10, log="patch.log",
    )

    # setup.sh 自身の確認は MPLBACKEND のせいで落ちるため、ここでやり直す。
    _shell(
        f'MPLBACKEND=Agg PYTHONPATH="{KIT_DIR}/LIBERO-plus:{KIT_DIR}:{KIT_DIR}/compe" '
        f'{KIT_DIR}/venv/bin/python -c '
        '"import torch, libero.libero.benchmark; '
        'from compe.t1 import register_t1; register_t1(); '
        "print('suite 登録 OK / torch', torch.__version__)\"",
        tail=20, log="verify.log",
    )

    verified = _grep_log("verify.log", "suite 登録 OK")
    print("\n".join(verified) if verified
          else "**suite 登録 OK が出ていない。** setup_logs/verify.log を見ること")
    if not verified:
        raise RuntimeError("キットの動作確認が通っていない")

    applied = [line for line in _grep_log("patch.log", "") if line.startswith("  ")]
    if applied:
        print(f"パッチ {len(applied)} 件:")
        print("\n".join(applied))

    print("キットのセットアップ完了")


# ============================================================
# 3. 重み
# ============================================================


def prepare_weights() -> None:
    """評価する重みを `/content/model_weights` に用意し、**ハッシュで名指しする**。

    **3 つとも 912 MB で、大きさでは見分けられない。** 取り違えると 2.6 時間を
    捨てるので、設定の `EXPECT_DIGEST` と突き合わせて違えば止める。

    Reads:
      - /content/colab_eval_config.py
      - Drive の DRIVE_WEIGHTS_DIR、または HF の素の重み
    Generates:
      - /content/model_weights/（本体と vlm/ のトークナイザ）
    """
    _print_header("3. 評価する重み")
    config = _load_config()
    spec = _spec(config)
    source_kind = str(_setting(config, "WEIGHTS_SOURCE")).strip().lower()
    expect = str(_setting(config, "EXPECT_DIGEST")).strip().lower()

    from huggingface_hub import snapshot_download

    if source_kind == "drive":
        source = Path(str(_setting(config, "DRIVE_WEIGHTS_DIR")))
        if not _mount_drive():
            raise RuntimeError("Drive がつながらないので重みを持ってこられない")
        if not source.is_dir():
            raise RuntimeError(
                f"{source} が無い。Drive の置き場と "
                "colab_eval_config.py の DRIVE_WEIGHTS_DIR が合っているか確かめること"
            )
        _shell(f"rm -rf {WEIGHTS_DIR} && cp -r '{source}' {WEIGHTS_DIR}")
    elif source_kind == "hf":
        # SmolVLA のリポジトリには eval/（評価動画 402 ファイル・25.6 MB）が入って
        # いる。推論には不要なので除外する。π0.5 側は `vlm/` が同梱なので、
        # ここで落としたものだけで揃う。
        snapshot_download(
            spec.weights_repo,
            revision=spec.weights_revision,
            local_dir=str(WEIGHTS_DIR),
            ignore_patterns=["eval/*", "eval/**"],
        )
    else:
        raise RuntimeError(
            f"WEIGHTS_SOURCE={source_kind!r} は知らない。'hf' か 'drive' である"
        )

    # policy_server.py は model_weights/vlm/ からトークナイザを読む（π0.5 も同じ。
    # solution/policy_server.py の tokenizer_processor の上書き）。**π0.5 の
    # リポジトリは vlm/ を同梱しているので別に落とさない**（spec.vlm_repo=None）。
    if spec.vlm_repo:
        snapshot_download(
            spec.vlm_repo,
            local_dir=str(WEIGHTS_DIR / "vlm"),
            allow_patterns=["*.json", "*.txt", "*.model", "tokenizer*"],
        )

    present = {path.name for path in WEIGHTS_DIR.iterdir() if path.is_file()}
    missing = (_COMMON_WEIGHT_FILES | {spec.normalizer_file}) - present
    if missing:
        raise RuntimeError(f"重みが足りない: {sorted(missing)}")
    _require(WEIGHTS_DIR / "vlm/tokenizer_config.json", "vlm のトークナイザが無い")

    config_json = json.loads((WEIGHTS_DIR / "config.json").read_text())
    if config_json.get("use_peft"):
        raise RuntimeError("use_peft が真。LoRA がマージされていない重みである")
    if str(config_json.get("type")) != spec.kind:
        raise RuntimeError(
            f"POLICY_KIND={spec.kind} だが、落ちてきた重みは "
            f"{config_json.get('type')} である。設定か置き場を直すこと"
        )

    total = sum(path.stat().st_size for path in WEIGHTS_DIR.rglob("*") if path.is_file())
    print(f"重みの準備完了: {total / 1e6:.0f} MB  （{spec.kind}・{source_kind}）")

    digest = _digest(WEIGHTS_DIR / "model.safetensors")
    head = digest[:8]
    print(f"model.safetensors: {digest[:16]}  … "
          f"{spec.known_digests.get(head, '**記録に無い重み**')}")
    if expect and head != expect:
        raise RuntimeError(
            f"意図した重みでない。EXPECT_DIGEST={expect} だが実物は {head} "
            f"（{spec.known_digests.get(head, '記録に無い重み')}）。"
            "設定か置き場を直すこと"
        )


# ============================================================
# 4. 方策サーバー
# ============================================================


def setup_policy() -> None:
    """方策サーバーの依存を入れ、`policy_server.py` を専用ディレクトリへ置く。

    LeRobot は**学習用ノートブックと同じ手順**で入れる。手順を変えると学習時と
    推論時でモデルの挙動が変わり、比較の意味が無くなる。

    Reads:
      - /content/policy_server.py
    Generates:
      - /content/lerobot/（v0.6.0 の clone）
      - /content/submission/policy_server.py と model_weights への symlink
    """
    _print_header("4. 方策サーバーの準備")
    _require(WEIGHTS_DIR, "prepare_weights() を先に通すこと")
    spec = _spec(_load_config())

    import torch

    # av が要る理由は非自明なので記録しておく。lerobot/policies/factory.py が
    # groot の設定を import し、その連鎖で groot/processor_groot.py に到達する。
    # そこは `datasets` が入っていると通る分岐で、lerobot/datasets/__init__.py が
    # require_package("av") を呼ぶ。**Colab には datasets が既定で入っている。**
    # 版を固定するのは v0.6.0 が av>=15,<16 を宣言しているため（17 系だと
    # pyav_utils.py が使う av.option が無く AttributeError になる）。
    _shell(f"{sys.executable} -m pip install -q fastapi uvicorn msgpack "
           "'av>=15.0.0,<16.0.0'")

    if not LEROBOT_DIR.exists():
        _shell(f"{sys.executable} -m pip uninstall -y -q lerobot torchao", check=False)
        _shell(f"git clone --quiet --depth 1 --branch {LEROBOT_TAG} "
               f"https://github.com/huggingface/lerobot.git {LEROBOT_DIR}")

        # bf16 非対応の GPU（T4 など）では fp16 に書き換える。
        # 学習用ノートブックが同じ置換をしており、揃えないと挙動が変わる。
        if not torch.cuda.is_bf16_supported():
            if spec.patch_bf16:
                smolvlm = LEROBOT_SRC / "lerobot/policies/smolvla/smolvlm_with_expert.py"
                source = smolvlm.read_text(encoding="utf-8")
                smolvlm.write_text(
                    source.replace('torch_dtype="bfloat16",', 'torch_dtype="float16",', 1),
                    encoding="utf-8",
                )
                print("bf16 非対応の GPU のため fp16 へ書き換えた")
            else:
                # **π0.5 には効く書き換えを用意していない。** dtype は
                # チェックポイントの config.json（`"dtype": "bfloat16"`）と
                # policy_server.py の `_build_pi05` が決めており、fp16 を選ぶ道が
                # そもそも無い（bfloat16 でなければ float32 になる分岐で、
                # float32 の π0.5 は T4 の 15 GB に載らない）。
                # **止めずに進む。** 動くかどうかは疎通確認（run(max_tasks=5)）で
                # 分かる。ここで落とすと、その確認自体ができない。
                print("**この GPU は bf16 非対応だが、π0.5 は bf16 のまま進む。**\n"
                      "  疎通確認（ce.run(chunk=1, max_tasks=5)）で動くかを見ること。\n"
                      "  落ちる場合は cc 8.0 以上の GPU（L4・A100 など）が要る。")

        _shell(f"{sys.executable} -m pip install -q --upgrade "
               f"-e '{LEROBOT_DIR}[{spec.lerobot_extra}]'")
        _shell(f"{sys.executable} -m pip uninstall -y -q torchao", check=False)

    # clone 先 /content/lerobot には __init__.py が無いため、`import lerobot` が
    # これを名前空間パッケージとして掴み lerobot.__file__ が None になる。
    # 実体のある src/ を sys.path の先頭へ置く。**別プロセスには効かない**ので、
    # サーバー起動時は PYTHONPATH で同じことをする（_server_env）。
    for name in [module for module in sys.modules
                 if module == "lerobot" or module.startswith("lerobot.")]:
        del sys.modules[name]
    sys.path = [entry for entry in sys.path
                if entry not in {str(LEROBOT_DIR), str(LEROBOT_SRC)}]
    sys.path.insert(0, str(LEROBOT_SRC))
    importlib.invalidate_caches()

    import lerobot

    if lerobot.__file__ is None:
        raise RuntimeError("lerobot が名前空間パッケージとして読まれている")
    if LEROBOT_SRC.resolve() not in Path(lerobot.__file__).resolve().parents:
        raise RuntimeError(f"LeRobot の読込先が正しくない: {lerobot.__file__}")
    print(f"lerobot: {Path(lerobot.__file__).parent}")
    print(f"bf16 対応: {torch.cuda.is_bf16_supported()}")

    # /content 直下のままだと、サーバーの別プロセスで `import lerobot` が
    # clone 先の /content/lerobot（__init__.py 無し）を掴む。隣に lerobot/ が
    # 無いディレクトリへ移す。
    SUBMISSION_DIR.mkdir(exist_ok=True)
    shutil.copy2(_require(CONTENT / "policy_server.py", "prepare() を先に通すこと"), SERVER)

    weights_link = SUBMISSION_DIR / "model_weights"
    if not weights_link.exists():
        weights_link.symlink_to(WEIGHTS_DIR)
    if (SUBMISSION_DIR / "lerobot").exists():
        raise RuntimeError("submission/ に lerobot/ があると import が壊れる")

    print(f"配置 OK: {SERVER}")
    print(f"  重み: {sorted(path.name for path in weights_link.iterdir())[:4]} …")


# ============================================================
# 5. 評価するタスク
# ============================================================


def build_suite() -> None:
    """検証用スイートへ差し替え、Colab の切断に耐える大きさへ割る。

    **example 4 件での成功率は本番を予測できない**（0.875 に対し本番 0.096）。
    LIBERO-plus の 10,030 変種から example 以外を選び、7 カテゴリを全部入れる。
    **前の回の CSV が /content/VAL_TASKS.csv にあれば、作り直さずにそれを使う**
    —— 1 件でも違うタスクが混ざると、1 件ずつ対にした比較が成り立たない。

    Reads:
      - /content/make_validation_suite.py, /content/check_validation_suite.py
      - /content/VAL_TASKS.csv（あれば）
    Generates:
      - /content/VAL_TASKS.csv, /content/chunks/VAL_1.csv 〜 VAL_<CHUNKS>.csv
      - /content/PARC2026_pre/compe/t1/T1_TASKS.csv（.example に退避してから差し替え）
    """
    _print_header("5. 検証用スイートと小分け")
    config = _load_config()
    _require(KIT_DIR, "setup_kit() を先に通すこと")
    kit_python = _kit_python()

    # パッチが当たっているかを実物で見る。「当てたつもり」を見逃さない。
    _shell(f"{kit_python} /content/patch_kit_for_variants.py --kit-dir {KIT_DIR}",
           tail=10, log="patch.log")

    if VAL_TASKS.is_file():
        print(f"既にある {VAL_TASKS} を使う（前の回と同じ {_count_tasks(VAL_TASKS)} 件）")
    else:
        per_cell = int(_setting(config, "PER_CELL"))
        _shell(
            f'PYTHONPATH="{KIT_DIR}/LIBERO-plus:{KIT_DIR}:{KIT_DIR}/compe" '
            f'{KIT_DIR}/venv/bin/python /content/make_validation_suite.py '
            f'--libero-plus {KIT_DIR}/LIBERO-plus '
            f'--exclude {KIT_TASKS} '
            f'--output {VAL_TASKS} --per-cell {per_cell}',
            tail=60, log="val_suite.log",
        )

    # **元の CSV は退避する。** 比較のために example へ戻すことがある。
    if not KIT_TASKS_BACKUP.exists():
        shutil.copy(KIT_TASKS, KIT_TASKS_BACKUP)
    shutil.copy(VAL_TASKS, KIT_TASKS)

    with KIT_TASKS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    print(f"\n差し替えた: {len(rows)} タスク（example は {KIT_TASKS_BACKUP.name} に退避）")
    for row in rows[:7]:
        print(f"  {row['suite']:16s} {row['category']:22s} {row['task_id'][:44]}")

    # **82 分回してから「4 行目で落ちていた」と分かるのは高すぎる。**
    # 配布キットと LIBERO-plus の実際のコードと同じ手順でパスを組み立てて、
    # 開けない行をここで名指しする。
    _shell(f'{KIT_DIR}/venv/bin/python /content/check_validation_suite.py '
           f'--csv {VAL_TASKS} --libero-plus {KIT_DIR}/LIBERO-plus '
           f'--kit-dir {KIT_DIR}',
           tail=30, log="val_check.log")

    _split_into_chunks(int(_setting(config, "CHUNKS")))


def _split_into_chunks(chunks: int) -> None:
    """タスクを塊へ割る。**どの塊にも全部のカテゴリが入る**ことを確かめる。

    元の CSV はカテゴリ巡回順（84 件版なら先頭 7 件が 7 カテゴリ）なので、続きを
    そのまま切り出せば偏らない。**カテゴリの数は CSV 自身から数える** ——
    Track2 の 36 件（`compe/val/EVAL_TASKS.csv`）は Background Textures と
    Light Conditions の 2 つしか無く、7 で決め打ちすると毎回誤警告が出る。

    Generates:
      - /content/chunks/VAL_1.csv 〜 VAL_<chunks>.csv
    """
    with VAL_TASKS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    header, body = rows[0], rows[1:]
    category = header.index("category")

    if CHUNK_DIR.exists():
        shutil.rmtree(CHUNK_DIR)          # 前回の割り方が残っていると混ざる
    CHUNK_DIR.mkdir(parents=True)

    size = -(-len(body) // chunks)        # 切り上げ
    expected_kinds = len({row[category] for row in body})
    print()
    for index in range(chunks):
        part = body[index * size:(index + 1) * size]
        path = CHUNK_DIR / f"VAL_{index + 1}.csv"
        with path.open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink)
            writer.writerow(header)
            writer.writerows(part)
        kinds = len({row[category] for row in part})
        note = ("" if kinds == expected_kinds
                else f"  ← **{expected_kinds} カテゴリのうち {kinds} しか入っていない**")
        print(f"{path.name}: {len(part)} 件 / {kinds} カテゴリ{note}")
    print(f"\nrun(chunk=1) から run(chunk={chunks}) まで、順に回す")


# ============================================================
# 6〜9. 評価する（1 回の呼び出しで、設定・実行・集計・退避を通す）
# ============================================================


@dataclass(frozen=True)
class EvalPlan:
    """1 回の評価の条件。**名前と中身を同じ 1 か所で決める**ための入れ物。"""

    chunk: int | None
    n_episodes: int
    max_steps: int
    max_tasks: int | None
    port: int
    policy_env: dict[str, str]
    tasks_csv: Path
    results_dir: Path
    record_dir: Path | None
    record_which: str
    record_max: int
    label: str


def _resolve_plan(
    config: ModuleType,
    chunk: int | None,
    max_tasks: int | None,
    n_episodes: int | None,
) -> EvalPlan:
    """設定と引数から条件を決め、**キットが読む CSV をそれに合わせる**。

    セルでやっていたときは CSV の差し替えと出力先の切り替えが別々の操作で、
    片方を忘れると別の 28 件の結果を同じ場所へ書いていた。ここで同時に決める。

    Generates:
      - /content/PARC2026_pre/compe/t1/T1_TASKS.csv（塊の CSV で上書き）
    """
    policy_env = {
        key: str(value)
        for key, value in dict(_setting(config, "POLICY_ENV")).items() if value
    }
    record_dir_setting = _setting(config, "RECORD_DIR")

    if chunk is None:
        tasks_csv = _require(VAL_TASKS, "build_suite() を先に通すこと")
        results_dir = CONTENT / "results"
    else:
        tasks_csv = CHUNK_DIR / f"VAL_{chunk}.csv"
        if not tasks_csv.is_file():
            # **どの番号が渡せるのかを名指しで出す。** 「無い」だけだと、
            # 割った数（設定の CHUNKS）を超えて渡したのか、build_suite() を
            # 通していないのかが区別できない。
            available = sorted(
                int(path.stem.removeprefix("VAL_"))
                for path in CHUNK_DIR.glob("VAL_*.csv")
            )
            raise RuntimeError(
                f"chunk={chunk} は無い。"
                + (f"渡せるのは {available} である"
                   if available else "build_suite() を先に通すこと")
            )
        results_dir = CONTENT / "results" / f"chunk{chunk}"
    shutil.copy(tasks_csv, KIT_TASKS)

    # 既定値は模型によって違う（policy_server.py は π0.5 のとき 16、SmolVLA は 5）。
    # ここを取り違えると、n16 で回した結果に n5 という名前が付く。
    parts = [f"n{policy_env.get('PARC_N_ACTION_STEPS', _spec(config).default_n_action_steps)}",
             f"ep{n_episodes if n_episodes else _setting(config, 'N_EPISODES')}",
             f"chunk{chunk}" if chunk else "all"]
    if max_tasks:
        # **疎通確認を本番と間違えないように名前へ残す。** 5 件の結果が
        # 84 件の結果と同じ名前で並ぶのが、いちばん危ない。
        parts.append(f"smoke{max_tasks}")
    for key, value in sorted(policy_env.items()):
        if key != "PARC_N_ACTION_STEPS":
            parts.append(f"{key.replace('PARC_', '').lower()}{value}".replace(".", ""))

    # **録画も塊ごとに分ける。** 1 つのディレクトリへ溜めると、退避のときに
    # 前の塊の動画も一緒に写り、どの 28 件で撮れたのか分からなくなる。
    record_dir = None
    if record_dir_setting:
        record_dir = Path(str(record_dir_setting))
        if chunk:
            record_dir = record_dir / f"chunk{chunk}"

    return EvalPlan(
        chunk=chunk,
        n_episodes=int(n_episodes if n_episodes else _setting(config, "N_EPISODES")),
        max_steps=int(_setting(config, "MAX_STEPS")),
        # **設定ファイルには置かない。** セルの `MAX_TASKS = 5` を戻し忘れて
        # 84 件のはずが 5 件で走る事故が起きたので、呼ぶときだけ渡す形にした。
        max_tasks=max_tasks,
        port=int(_setting(config, "PORT")),
        policy_env=policy_env,
        tasks_csv=tasks_csv,
        results_dir=results_dir,
        record_dir=record_dir,
        record_which=str(_setting(config, "RECORD_WHICH")),
        record_max=int(_setting(config, "RECORD_MAX")),
        label="_".join(parts),
    )


def _server_env(plan: EvalPlan) -> dict[str, str]:
    """方策サーバー側の環境変数を作る。"""
    env = dict(os.environ)
    env.update(plan.policy_env)
    # policy_server.py は setdefault で offline を立てる。Colab は通信できるので
    # 明示的に解除する。**採点環境は遮断されるため、その確認は Docker 側で行う。**
    env["HF_HUB_OFFLINE"] = "0"
    env["TRANSFORMERS_OFFLINE"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # 別プロセスなので、ノートブック側の sys.path 操作は効かない。
    env["PYTHONPATH"] = str(LEROBOT_SRC)
    return env


def _eval_env(plan: EvalPlan) -> dict[str, str]:
    """評価パイプライン側の環境変数を作る。"""
    env = dict(os.environ)
    env.update({
        "MUJOCO_GL": "egl",          # GPU 描画。prepare() のベンダ設定と対で効く
        "MPLBACKEND": "Agg",         # Colab の既定値だと libero の import が落ちる
        "PYTHONUNBUFFERED": "1",     # 無いと出力が溜まり、止まったように見える
        "PYTHONPATH": f"{KIT_DIR}/LIBERO-plus:{KIT_DIR}:{KIT_DIR}/compe",
        "LIBERO_ROOT": f"{KIT_DIR}/LIBERO-plus",
    })
    if plan.record_dir:
        env["PARC_RECORD_DIR"] = str(plan.record_dir)
        env["PARC_RECORD_WHICH"] = plan.record_which
        env["PARC_RECORD_MAX"] = str(plan.record_max)
    return env


def _health_answers(port: int) -> bool:
    """そのポートで誰かが /health に答えるか。"""
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
    except Exception:
        return False
    return True


def stop_server(port: int = 8000) -> None:
    """居残っている方策サーバーを止める。

    **前のサーバーが生きたまま次を起動すると、古い重みで測ってしまう。**
    新しいサーバーはポートを掴めずに死ぬが、`/health` には古い方が答えるため、
    評価はそのまま進む。run() は起動前にこれを検出して止まる。
    """
    _shell(f"pkill -f 'policy_server.py --port {port}'", check=False)
    time.sleep(2)
    print(f"ポート {port}: {'**まだ答えている**' if _health_answers(port) else '空いた'}")


def _start_server(plan: EvalPlan) -> subprocess.Popen[bytes]:
    """方策サーバーを起こし、**意図した構成で立っているかを画面で確かめる**。

    Generates:
      - /content/server.log
    """
    if _health_answers(plan.port):
        raise RuntimeError(
            f"ポート {plan.port} で既に誰かが /health に答えている。前のサーバーが"
            "居残っていると**古い重みで測る**ことになる。stop_server() を実行すること"
        )

    log = SERVER_LOG.open("wb")
    server = subprocess.Popen(
        [sys.executable, str(_require(SERVER, "setup_policy() を先に通すこと")),
         "--port", str(plan.port)],
        cwd=str(SUBMISSION_DIR), env=_server_env(plan),
        stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"サーバー起動 (pid={server.pid})  設定値: {plan.policy_env or '既定'}")

    for elapsed in range(300):
        if _health_answers(plan.port):
            print(f"/health 応答（{elapsed} 秒）")
            break
        if server.poll() is not None:
            print(SERVER_LOG.read_text(errors="replace")[-3000:])
            raise RuntimeError("サーバーが起動に失敗した")
        time.sleep(1)
    else:
        server.terminate()
        raise RuntimeError("300 秒待っても /health が応答しない")

    # **版と実効設定はサーバーのログにしか出ない。** ここで拾って画面に出す。
    # MyPolicy() は uvicorn.run より前に作られるので、/health が答える時点で
    # この行は必ず書かれている。**出ないのは、古い policy_server.py を置いたか、
    # 別のサーバーが答えているとき。**
    marks = [line for line in SERVER_LOG.read_text(errors="replace").splitlines()
             if line.startswith("[parc]")]
    if not marks:
        server.terminate()
        raise RuntimeError(
            "[parc] の行が出ていない。古い policy_server.py を置いているか、"
            "別のサーバーが答えている。評価しても構成が分からないので止める"
        )
    print("\n".join(marks))
    return server


def _run_pipeline(plan: EvalPlan, server: subprocess.Popen[bytes]) -> Path:
    """評価パイプラインを回し、**結果行を 1 本ごとに Drive へ**書く。

    画面に出すのは結果行と 60 秒ごとの心拍だけにする。Colab の出力はブラウザ側に
    すべて保持されるため、数万行出すと Chrome が応答しなくなり、カーネル側にも
    積み上がってメモリ不足で評価ごと落ちる。

    Returns
    -------
    eval_log : Path
        全文ログの場所。

    Generates:
      - /content/setup_logs/eval_<日時>_chunk<N>.log と .results.txt
      - /content/drive/MyDrive/parc2026/eval/live/<日時>_chunk<N>.results.txt
    """
    stamp = time.strftime("%m%d_%H%M%S")
    eval_log = LOG_DIR / f"eval_{stamp}_{plan.label}.log"

    # **/content はランタイムが切れると消える。** 退避を待つと、落ちた回の
    # 40 分ぶんが丸ごと失われる（8/11 に失った）。1 行ずつ flush する。
    live_log = None
    if _mount_drive():
        live_dir = EVAL_ROOT / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        live_log = (live_dir / f"{stamp}_{plan.label}.results.txt").open(
            "w", encoding="utf-8"
        )
        print(f"逐次退避: {live_log.name}")
    else:
        print("**Drive へ逐次退避できない。** 落ちたらこの回は失われる")

    started = time.time()
    done = collided = succeeded = 0
    last_beat = started
    process = None

    try:
        process = subprocess.Popen(
            [f"{KIT_DIR}/venv/bin/python", "-m", "pipeline",
             "--server-url", f"http://localhost:{plan.port}",
             "--track", "track1",
             "--n-episodes", str(plan.n_episodes),
             "--max-steps", str(plan.max_steps),
             *(["--max-tasks", str(plan.max_tasks)] if plan.max_tasks else []),
             "--output-dir", str(plan.results_dir)],
            # --verbose は付けない。root logger が DEBUG になり、urllib3 が
            # /act 1 回ごとに 1 行出すため数万行になる。エピソードごとの結果は
            # パッチが logger.info で出すので、DEBUG に落とす必要が無い。
            cwd=str(KIT_DIR), env=_eval_env(plan),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        with eval_log.open("w", encoding="utf-8") as sink:
            for line in process.stdout:
                sink.write(line)
                if "[結果]" in line:
                    done += 1
                    collided += "衝突" in line
                    succeeded += "成功" in line
                    if live_log is not None:
                        live_log.write(line)
                        live_log.flush()          # **落ちても直前まで残す**
                    print(f"[{time.time() - started:5.0f}s] {line.rstrip()[-110:]}")
                    last_beat = time.time()
                elif time.time() - last_beat >= _HEARTBEAT_SEC:
                    print(f"[{time.time() - started:5.0f}s] 実行中… "
                          f"{done} エピソード完了（成功 {succeeded} / 衝突 {collided}）")
                    last_beat = time.time()
        process.wait()
    finally:
        server.terminate()
        if live_log is not None:
            live_log.close()

    code = process.returncode if process else -1
    print(f"\n所要 {time.time() - started:.0f} 秒  終了コード {code}")
    if code != 0:
        print("--- ログの末尾 ---")
        print(eval_log.read_text(encoding="utf-8", errors="replace")[-3000:])
        if code == -9:
            print("**終了コード -9 は SIGKILL。** Colab がメモリ不足で殺した")
    return eval_log


def _wilson_interval(rate: float, count: int) -> tuple[float, float]:
    """二項分布の 95% 信頼区間。20 本と 80 本では判定できる差が違う。"""
    z = 1.96
    denominator = 1 + z**2 / count
    center = (rate + z**2 / (2 * count)) / denominator
    margin = z * math.sqrt(
        rate * (1 - rate) / count + z**2 / (4 * count**2)
    ) / denominator
    return center - margin, center + margin


def _result_json(plan: EvalPlan) -> Path | None:
    """完走時に書かれる結果 JSON を探す。

    名前はサーバーの URL から決まる（`server_8000.json`）。ポートを変えたときに
    見つけられなくなるので、まず名前で探し、無ければその出力先の JSON を拾う。
    """
    named = plan.results_dir / f"server_{plan.port}.json"
    if named.is_file():
        return named
    found = sorted(plan.results_dir.glob("*.json")) if plan.results_dir.is_dir() else []
    return found[0] if found else None


def _print_metrics(plan: EvalPlan, eval_log: Path) -> dict[str, float]:
    """完走時の JSON から、採点と同じ指標を出す。

    **JSON があっても中身が結果とは限らない。** トラックが例外で落ちると
    `_errored_track_score` が `{"error": 1.0, "n_tasks": 0.0}` だけを書き出す。
    鍵の有無を見ずに読むと KeyError になり、**本当の原因である例外の行を
    探しに行けない。**

    Reads:
      - <results_dir>/server_8000.json
    """
    result_json = _result_json(plan)
    if result_json is None:
        print(f"\nJSON が無い（{plan.results_dir}）。評価が完走していない")
        return {}

    track = json.loads(result_json.read_text())["tracks"][0]
    overall = track["overall_metrics"]
    if "n_total_episodes" not in overall:
        print("\n**評価が例外で落ちている。** JSON はエラー記録だけを持つ。")
        print(f"  overall_metrics = {overall}")
        text = eval_log.read_text(encoding="utf-8", errors="replace")
        marks = [line for line in text.splitlines()
                 if any(needle in line for needle in
                        ("Traceback", "Error", "失敗", "例外"))]
        print(f"\n  ログ中のエラー行（末尾 15 行）: 全 {len(marks)} 件")
        for line in marks[-15:]:
            print(f"    {line[:160]}")
        print(f"\n  全文: {eval_log}")
        return {}

    count, rate = overall["n_total_episodes"], overall["mean_success_rate"]
    low, high = _wilson_interval(rate, count)
    print(f"\n成功率   {rate:.3f}   95%CI [{low:.3f}, {high:.3f}]  n={count}")
    print(f"衝突率   {overall['mean_collision_rate']:.3f}\n")
    for task in track["tasks"]:
        print(f"  成功 {task['success_rate']:.2f}  衝突 "
              f"{task['metrics']['collision_rate']:.2f}  {task['task_name'][:50]}")
    print()
    for key in ("mean_avg_cartesian_jerk", "mean_rms_cartesian_jerk",
                "mean_sparc", "mean_cartesian_path_length"):
        print(f"  {key:32s} {overall[key]:9.4f}")
    return {
        "success_rate": rate,
        "collision_rate": overall["mean_collision_rate"],
        "n_episodes": count,
    }


def _print_log_summary(plan: EvalPlan, eval_log: Path) -> Path | None:
    """ログの結果行から、**何にぶつかったか**と摂動カテゴリ別を出す。

    JSON には「衝突したか」しか入らない。ぶつけた物体はパッチが出すログ行に
    しかなく、そこが次の一手を決める。**途中で落ちた回でもここは読める。**

    Returns
    -------
    results_txt : Path | None
        結果行だけを抜いた小さいファイル。行が無ければ None。

    Generates:
      - <eval_log>.results.txt
    """
    lines = [line for line
             in eval_log.read_text(encoding="utf-8", errors="replace").splitlines()
             if "[結果]" in line]
    if not lines:
        print("\n**[結果] 行が 0 件。** パッチを当てずに回している")
        return None

    # 全文は数 MB あって持ち帰るのが億劫になる。結果行だけなら数十 KB。
    results_txt = eval_log.with_suffix(".results.txt")
    results_txt.write_text("\n".join(lines), encoding="utf-8")

    outcomes: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = {}
    for line in lines:
        found = re.search(r"\[結果\] (\S+) ep(\d+): (\S+)", line)
        if not found:
            continue
        name, _episode, outcome = found.groups()
        outcomes[outcome] += 1
        by_task.setdefault(name, Counter())[outcome] += 1
        for obj, _millimeter in re.findall(r"(\w+)=([\d.]+)mm", line):
            objects[obj] += 1

    total = sum(outcomes.values())
    print(f"\n=== ログから {total} エピソード分 ===")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome:6s} {count:3d}  ({count / total:.1%})")

    if objects:
        print("\nぶつけた物体（衝突したエピソード数）")
        for obj, count in objects.most_common():
            print(f"  {count:3d}  {obj}")

    # **どのカテゴリが弱いかが分かれば、学習で何を厚くすべきかが決まる。**
    category_of: dict[str, str] = {}
    with plan.tasks_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("category"):
                category_of[row["task_id"]] = row["category"]

    if category_of:
        per_category: dict[str, Counter[str]] = {}
        for name, slot in by_task.items():
            per_category.setdefault(category_of.get(name, "（不明）"), Counter()).update(slot)
        print("\n摂動カテゴリ別")
        for key in sorted(per_category):
            slot = per_category[key]
            here = sum(slot.values())
            print(f"  {key:24s} 成功 {slot.get('成功', 0) / here:.2f}  ({here} 本)  "
                  + " ".join(f"{name} {count}" for name, count in slot.most_common()))

    print("\nタスク別")
    for name, slot in sorted(by_task.items()):
        print(f"  {name[:46]:46s} "
              + " ".join(f"{key} {count}" for key, count in slot.most_common()))
    return results_txt


def _summarize_videos(plan: EvalPlan, eval_log: Path) -> None:
    """撮れた動画を**見るべき順に並べ**、コマ並べ画像を作る。

    打ち切りに張り付いた回が先頭に来るので、再生する前に当たりを付けられる。
    """
    if not plan.record_dir:
        return
    if not (plan.record_dir.is_dir() and any(plan.record_dir.glob("*.mp4"))):
        print(f"\n**動画が 1 本も無い（{plan.record_dir}）。**"
              " 失敗した回が無かったか、パッチが当たっていない")
        for line in eval_log.read_text(errors="replace").splitlines():
            if "[録画]" in line:
                print(f"    {line.strip()}")
        return

    summarize = CONTENT / "summarize_rollout_videos.py"
    print()
    if summarize.is_file():
        # キットの venv で回す。cv2 はそちらに入っている。
        subprocess.run([f"{KIT_DIR}/venv/bin/python", str(summarize),
                        str(plan.record_dir)], check=False)
    else:
        print(f"**{summarize} が無い。** 一覧だけ出す:")
        for path in sorted(plan.record_dir.glob("*.mp4")):
            print(f"  {path.stat().st_size / 1e3:8.0f} KB  {path.name}")


def _archive(plan: EvalPlan, eval_log: Path, results_txt: Path | None) -> Path | None:
    """一式を Drive へ退避する。**名前は条件から作るので付け忘れが無い。**

    Generates:
      - /content/drive/MyDrive/parc2026/eval/<日時>_<設定>/
    """
    if not _mount_drive():
        print("\n**Drive へ退避できない。** 左のファイルペインから手で落とすこと:")
        print(f"  {eval_log}")
        return None

    destination = EVAL_ROOT / f"{time.strftime('%m%d_%H%M%S')}_{plan.label}"
    destination.mkdir(parents=True, exist_ok=True)
    for path in (_result_json(plan), results_txt, eval_log, SERVER_LOG):
        if path and path.exists():
            shutil.copy(path, destination / path.name)

    # **動画も一緒に。** 忘れると 51 分の測定の映像が切断で消える。
    if plan.record_dir and plan.record_dir.is_dir():
        videos = (sorted(plan.record_dir.glob("*.mp4"))
                  + sorted(plan.record_dir.glob("sheets/*.png")))
        if videos:
            video_destination = destination / "videos"
            video_destination.mkdir(exist_ok=True)
            for path in videos:
                shutil.copy(path, video_destination / path.name)
            print(f"  動画 {len(videos)} 件を {video_destination} へ")

    print(f"\n=== Drive へ退避した ===\n  {destination}")
    for path in sorted(destination.iterdir()):
        if path.is_dir():
            print(f"  {'':8s}    {path.name}/  ({len(list(path.iterdir()))} 件)")
        else:
            print(f"  {path.stat().st_size / 1e3:8.0f} KB  {path.name}")
    return destination


def run(
    chunk: int | None = None,
    *,
    max_tasks: int | None = None,
    n_episodes: int | None = None,
) -> dict[str, object]:
    """1 つの塊を評価し、集計して Drive へ退避する（28 件で約 51 分）。

    **設定・実行・集計・退避を 1 回の呼び出しで通す。** 別々の操作だったころは、
    片方を忘れて前の塊の結果を新しい名前で退避する事故が起きた。

    Parameters
    ----------
    chunk : int | None
        `build_suite()` が作った塊の番号。None なら全件を一度に回す
        （切断が無い環境向け）。
    max_tasks : int | None
        **疎通確認のときだけ渡す。** 5 を渡すと 5 タスクで終わる。設定ファイルを
        書き換えないので**戻し忘れが起きない**。退避先の名前に `smoke5` が付く。
    n_episodes : int | None
        1 タスクあたりの本数。既定は設定ファイルの `N_EPISODES`（1）。

    Returns
    -------
    summary : dict[str, object]
        成功率・衝突率・エピソード数・退避先。完走していなければ空に近い。

    Reads:
      - /content/colab_eval_config.py, /content/chunks/VAL_<chunk>.csv
    Generates:
      - <results_dir>/server_8000.json, /content/setup_logs/eval_*.log
      - /content/drive/MyDrive/parc2026/eval/{live/,<日時>_<設定>/}
    """
    _print_header(f"評価する  chunk={chunk if chunk else 'なし（全件）'}")
    config = _load_config()
    plan = _resolve_plan(config, chunk, max_tasks, n_episodes)

    print(f"  タスク  {_count_tasks(plan.tasks_csv)} 件（{plan.tasks_csv}）")
    print(f"  1 タスクあたり {plan.n_episodes} 本 / 打ち切り {plan.max_steps} 歩"
          + (f" / **疎通確認 {plan.max_tasks} タスクだけ**" if plan.max_tasks else ""))
    print(f"  設定値  {plan.policy_env or '既定（n_action_steps=5 / ema_alpha=0.3）'}")
    print(f"  出力先  {plan.results_dir}")
    print(f"  録画    {plan.record_dir or '（撮らない）'}")
    print(f"  名前    {plan.label}")

    # **1 本目が 200 秒を超えていたら EGL が効いていない。** 100〜130 秒が想定。
    server = _start_server(plan)
    eval_log = _run_pipeline(plan, server)

    metrics = _print_metrics(plan, eval_log)
    results_txt = _print_log_summary(plan, eval_log)
    _summarize_videos(plan, eval_log)
    destination = _archive(plan, eval_log, results_txt)

    print("\n**成功率と衝突率を先に見ること。** ここが下がったら、"
          "jerk や SPARC がどれだけ良くても採用しない")
    return {**metrics, "label": plan.label, "log": str(eval_log),
            "archive": str(destination) if destination else None}


# ============================================================
# どこまで進んだか
# ============================================================


def status() -> None:
    """どの塊が終わっているかを、**残っている物から**読む。

    ランタイムが落ちると `/content` は消える。Drive の `live/` だけは
    1 本ごとに書いているので、そこが唯一の進捗記録になる。

    Reads:
      - /content/results/chunk*/server_8000.json
      - /content/drive/MyDrive/parc2026/eval/live/*.results.txt
    """
    _print_header("どこまで進んだか")

    print("\n--- 完走した塊（/content。ランタイムが切れると消える）---")
    found = sorted((CONTENT / "results").glob("chunk*/*.json"))
    if not found:
        print("  無い（run() が完走していない）")
    for path in found:
        overall = json.loads(path.read_text())["tracks"][0]["overall_metrics"]
        stamp = time.strftime("%m/%d %H:%M", time.localtime(path.stat().st_mtime))
        if "n_total_episodes" in overall:
            print(f"  {path.parent.name:8s} {overall['n_total_episodes']:3d} 本  "
                  f"成功 {overall['mean_success_rate']:.3f}  {stamp}")
        else:
            print(f"  {path.parent.name:8s} **例外で落ちている**  {stamp}")

    print("\n--- 1 本ごとの記録（Drive。落ちても残る）---")
    live_dir = EVAL_ROOT / "live"
    if not (_mount_drive() and live_dir.is_dir()):
        print("  無い（Drive につながっていないか、まだ 1 度も回していない）")
        return
    for path in sorted(live_dir.glob("*.results.txt")):
        rows = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
        stamp = time.strftime("%m/%d %H:%M", time.localtime(path.stat().st_mtime))
        print(f"  {rows:3d} 本  {stamp}  {path.name}")
    print("\n  **時刻は Colab の既定の UTC。** 日本時間より 9 時間前に見える")
