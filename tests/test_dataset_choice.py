"""学習データの差し替え（`--dataset-repo` / `--dataset-revision`）の試験。

**打つのは 1 行である**（本物の torch と lerobot が要るので、常駐コンテナで動かす）::

    bash solution/tools/train/run_train_tests.sh

**確かめるのは 3 つ。**

1. **何も渡さなければ、今までと 1 文字も変わらないこと。**
2. **配布データへ向けたとき、`lerobot/libero_plus` の版が付いて行かないこと**
   ——版はあちらのコミットなので、別のデータへ渡すと落ちるか別物を取る。
3. **`--dataset-root` を忘れたら、その場で止まること**——忘れると番号選びが
   Hub の 14,347 本を読み、**足したぶんが 1 本も選ばれないまま学習が始まる**
   （エラーが出ないので気付けない）。
"""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="本物の torch が要る（常駐コンテナで動かす）")

_TRAIN = Path(__file__).resolve().parent.parent / "solution" / "train"
sys.path.insert(0, str(_TRAIN))

pytest.importorskip("lerobot.scripts.lerobot_train",
                    reason="本物の lerobot が要る（常駐コンテナで動かす）")

_BASE = [
    "--run-name", "t", "--steps", "2500", "--batch-size", "64",
    "--num-workers", "8", "--learning-rate", "5e-6", "--save-freq", "500",
    "--freeze-vision-encoder", "false", "--train-expert-only", "false",
    "--camera-viewpoint-augment",
]

_DEFAULT_REPO = "lerobot/libero_plus"
_DEFAULT_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
_COMBINED = "lerobot/libero_combined_20hz"


def _build(extra):
    """`train_pi05.py` に引数を渡して、組まれたコマンドを返す。"""
    import train_pi05

    for var in ("CAMERA_DROPOUT_PROB", "CAMERA_DROPOUT_KEY"):
        os.environ.pop(var, None)
    sys.argv = ["train_pi05.py", *_BASE, *extra]
    args = train_pi05._parse_args()
    train_pi05._check_dataset_choice(args)
    return args, train_pi05._build_command(
        args, base_model="/w/001500", episode_indices=None, output_dir="/out/t")


def _value_of(command, flag):
    """`--dataset.repo_id=X` のような 1 語から X を取り出す。無ければ None。"""
    hits = [c.split("=", 1)[1] for c in command if c.startswith(flag + "=")]
    return hits[0] if hits else None


class TestDefault:
    def test_unchanged_when_nothing_is_passed(self):
        """渡さなければ、今まで（8.0.0・12.0.0）と同じ名前と版が付くこと。"""
        _, command = _build([])
        assert _value_of(command, "--dataset.repo_id") == _DEFAULT_REPO
        assert _value_of(command, "--dataset.revision") == _DEFAULT_REVISION
        assert _value_of(command, "--dataset.root") is None

    def test_root_alone_keeps_the_default_revision(self):
        """`--dataset-root` だけなら、名前も版も既定のままであること。

        **既にある使い方を壊さない**——幾何の実験などは
        既定のデータを手元に置いて `--dataset-root` だけを渡している。
        """
        _, command = _build(["--dataset-root", "/data/libero_plus"])
        assert _value_of(command, "--dataset.repo_id") == _DEFAULT_REPO
        assert _value_of(command, "--dataset.revision") == _DEFAULT_REVISION
        assert _value_of(command, "--dataset.root") == "/data/libero_plus"


class TestCombined:
    def test_revision_is_dropped_for_another_dataset(self):
        """**別のデータには、既定の版を付けないこと。**"""
        _, command = _build(["--dataset-repo", _COMBINED,
                             "--dataset-root", "/data/combined"])
        assert _value_of(command, "--dataset.repo_id") == _COMBINED
        assert _value_of(command, "--dataset.revision") is None
        assert _value_of(command, "--dataset.root") == "/data/combined"

    def test_an_explicit_revision_is_kept(self):
        """明示した版は、別のデータでもそのまま渡ること。"""
        _, command = _build(["--dataset-repo", _COMBINED,
                             "--dataset-root", "/data/combined",
                             "--dataset-revision", "abc123"])
        assert _value_of(command, "--dataset.revision") == "abc123"

    def test_stops_when_the_root_is_missing(self):
        """**`--dataset-root` を忘れたら、その場で止まること。**

        止めないと、番号選びが Hub の `lerobot/libero_plus`（14,347 本）を読み、
        足した 5,186 本が 1 本も選ばれないまま学習が走る。
        """
        with pytest.raises(SystemExit) as caught:
            _build(["--dataset-repo", _COMBINED])
        text = str(caught.value)
        assert "--dataset-root" in text          # 直し方を教える
        assert "1 本も選ばれない" in text        # なぜ駄目かを教える

    def test_only_the_dataset_changes(self):
        """**データ以外は 1 語も変わらないこと。**1 変数を保つための試験である。"""
        _, plain = _build([])
        _, moved = _build(["--dataset-repo", _COMBINED,
                           "--dataset-root", "/data/combined"])
        strip = lambda c: [x for x in c if not x.startswith("--dataset.")]
        assert strip(moved) == strip(plain)
