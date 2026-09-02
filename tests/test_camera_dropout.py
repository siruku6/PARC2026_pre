"""場面カメラを確率で潰す工夫（`experiments/camera_dropout/`）の試験。

**打つのは 1 行である**（本物の torch と lerobot が要るので、常駐コンテナで動かす）::

    bash solution/tools/train/run_camera_dropout_test.sh

**この機械の venv には torch も lerobot も入っていないので、そのまま
`pytest tests/` を打つと全部 skip になる**（それが分かるように理由を出してある）。

試験は 2 つに分かれる。

`TestPatch`
    潰す本体（`camera_dropout_patch.apply`）。**確率どおりに潰れるか・手首の
    カメラを触っていないか・列の名前が違うときにその場で止まるか。**
`TestCommand`
    `train_pi05.py` が組むコマンド。**引数を渡さなければ 1 文字も変わらないか・
    渡したときに先頭だけが替わるか。**

`original` はパッチを当てる前の `DatasetReader.get_item` を掴むので、
**先に stub を置いてから apply() を呼ぶ**——そうすると本物のデータセットを
用意せずに、包んでいる側の振る舞いだけを試せる。
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="本物の torch が要る（常駐コンテナで動かす）")

_TRAIN = Path(__file__).resolve().parent.parent / "solution" / "train"
sys.path.insert(0, str(_TRAIN))

pytest.importorskip(
    "lerobot.datasets.dataset_reader",
    reason="本物の lerobot が要る（常駐コンテナで動かす）",
)

from lerobot.datasets.dataset_reader import DatasetReader  # noqa: E402

FRONT = "observation.images.front"
WRIST = "observation.images.wrist"


def _make_item():
    return {
        FRONT: torch.full((3, 8, 8), 0.7),
        WRIST: torch.full((3, 8, 8), 0.4),
        "action": torch.zeros(7),
    }


@pytest.fixture
def patched():
    """`(確率, 列の名前)` を受け取り、stub の上にパッチを当てた get_item を返す。

    後片付けで `DatasetReader.get_item` と環境変数を元に戻す——**戻さないと
    同じ試験の後続が、前の試験のパッチを掴んだままになる。**
    """
    saved = DatasetReader.get_item
    saved_env = {k: os.environ.get(k)
                 for k in ("CAMERA_DROPOUT_PROB", "CAMERA_DROPOUT_KEY")}

    def build(probability, key=FRONT):
        os.environ["CAMERA_DROPOUT_PROB"] = str(probability)
        os.environ["CAMERA_DROPOUT_KEY"] = key
        DatasetReader.get_item = lambda self, idx: _make_item()
        sys.modules.pop("experiments.camera_dropout.camera_dropout_patch", None)
        module = importlib.import_module(
            "experiments.camera_dropout.camera_dropout_patch")
        module.apply()
        return DatasetReader.get_item

    yield build

    DatasetReader.get_item = saved
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class TestPatch:
    def test_drops_at_the_requested_rate(self, patched):
        """4,000 コマで、実測の割合が指定に寄ること。"""
        get_item = patched(0.5)
        torch.manual_seed(0)
        dropped = sum(bool((get_item(None, i)[FRONT] == 0).all())
                      for i in range(4000))
        assert 0.47 < dropped / 4000 < 0.53

    def test_leaves_the_wrist_camera_alone(self, patched):
        """**手首のカメラは絶対に触らない**——Track2 でずれるのは場面側だけである。

        あわせて、潰さなかったコマの場面カメラも元の値のままであること。
        """
        get_item = patched(0.5)
        torch.manual_seed(0)
        for i in range(500):
            item = get_item(None, i)
            assert bool((item[WRIST] == 0.4).all())
            assert (bool((item[FRONT] == 0).all())
                    or bool((item[FRONT] == 0.7).all()))

    def test_drops_every_frame_at_probability_one(self, patched):
        get_item = patched(1.0)
        torch.manual_seed(7)
        assert all(bool((get_item(None, i)[FRONT] == 0).all())
                   for i in range(50))

    def test_is_reproducible_for_a_fixed_seed(self, patched):
        """`--seed` を固定すれば同じ並びになること（torch の乱数を使っている）。"""
        runs = []
        for _ in range(2):
            get_item = patched(0.5)
            torch.manual_seed(1234)
            runs.append([bool((get_item(None, i)[FRONT] == 0).all())
                         for i in range(200)])
        assert runs[0] == runs[1]

    def test_stops_when_the_column_name_is_wrong(self, patched):
        """**黙って効かないのを防ぐ。**

        列の名前が違っていても学習自体は通ってしまい、4 時間 46 分後に
        「効かなかった」という誤った結論だけが残る。だからその場で止める。
        """
        get_item = patched(0.5, key="observation.images.nosuch")
        with pytest.raises(SystemExit) as caught:
            get_item(None, 0)
        text = str(caught.value)
        assert FRONT in text and WRIST in text     # 在る列を教える
        assert "--scene-camera-key" in text        # 直し方を教える

    def test_does_nothing_when_the_probability_is_zero(self):
        """既定（0.0）では包まないこと——渡さなければ何も変わらない。"""
        saved = DatasetReader.get_item
        try:
            DatasetReader.get_item = lambda self, idx: _make_item()
            before = DatasetReader.get_item
            os.environ.pop("CAMERA_DROPOUT_PROB", None)
            sys.modules.pop("experiments.camera_dropout.camera_dropout_patch",
                            None)
            module = importlib.import_module(
                "experiments.camera_dropout.camera_dropout_patch")
            module.apply()
            assert DatasetReader.get_item is before
        finally:
            DatasetReader.get_item = saved


# 8.0.0 を作った学習と同じ引数（`work/train/pi05_ladder2500.log` の 8 行目）。
_BASE = [
    "--run-name", "t", "--steps", "2500", "--batch-size", "64",
    "--num-workers", "8", "--learning-rate", "5e-6", "--save-freq", "500",
    "--freeze-vision-encoder", "false", "--train-expert-only", "false",
    "--camera-viewpoint-augment",
]


def _build(extra):
    """`train_pi05.py` に引数を渡して、組まれたコマンドと args を返す。"""
    import train_pi05

    for var in ("CAMERA_DROPOUT_PROB", "CAMERA_DROPOUT_KEY"):
        os.environ.pop(var, None)
    sys.argv = ["train_pi05.py", *_BASE, *extra]
    args = train_pi05._parse_args()
    train_pi05.camera_dropout.set_environment(args)
    train_pi05.camera_dropout.validate(args)
    command = train_pi05._build_command(
        args, base_model="/w/001500", episode_indices=None, output_dir="/out/t")
    return args, command


class TestCommand:
    def test_unchanged_when_the_option_is_absent(self):
        """渡さなければ、今まで（8.0.0）と同じコマンドが組まれること。"""
        _, command = _build([])
        assert command[0] == "lerobot-train"
        assert "CAMERA_DROPOUT_PROB" not in os.environ
        assert not [c for c in command if "scene" in c]

    def test_only_the_head_changes_when_selected(self):
        """**先頭だけが替わり、lerobot-train へ渡す引数は 1 つも増えない。**

        潰すのは DataLoader の中なので、lerobot-train から見た設定は
        変わらないのが正しい。

        比べる位置がずれる点に注意——`["lerobot-train"]` は 1 語だが、
        ラッパーは `["python", "<パス>"]` の 2 語である。
        """
        _, plain = _build([])
        _, dropped = _build(["--scene-camera-dropout", "0.5"])
        assert dropped[0] == "python"
        assert dropped[1].endswith("lerobot_train_with_camera_dropout.py")
        assert dropped[2:] == plain[1:]
        assert os.environ["CAMERA_DROPOUT_PROB"] == "0.5"
        assert os.environ["CAMERA_DROPOUT_KEY"] == FRONT

    def test_describe_shows_the_value(self):
        import train_pi05

        args, _ = _build(["--scene-camera-dropout", "0.5"])
        assert "0.5" in train_pi05.camera_dropout.describe(args)

    def test_rejects_a_probability_above_one(self):
        """データを読み込む前に止めること（気づくのが学習開始後だと遅い）。"""
        with pytest.raises(SystemExit):
            _build(["--scene-camera-dropout", "1.5"])

    def test_the_column_name_can_be_replaced(self):
        _build(["--scene-camera-dropout", "0.3",
                "--scene-camera-key", WRIST])
        assert os.environ["CAMERA_DROPOUT_KEY"] == WRIST
