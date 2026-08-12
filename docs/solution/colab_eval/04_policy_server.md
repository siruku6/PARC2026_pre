# 4. 方策サーバーを立てる準備

> **いまは [`ce.setup_policy()`](../../../solution/tools/colab_eval.py) が全部やる。**
> **コードの正はそちらである。** このページは**何が起きているかと、その理由**を
> 残す。**`PARC_*` の 4 つのつまみの表（下）はここにしかない。**

Colab のシステム Python（3.12）に、`policy_server.py` が要るものだけ足す。
LeRobot は学習用ノートブックと同じ手順で入れる。

```python
# === 方策サーバー側の依存 ===
# 学習用ノートブックと同じ手順で LeRobot v0.6.0 を入れる。手順を変えると
# 学習時と推論時でモデルの挙動が変わり、比較の意味が無くなる。
import shutil, sys
from pathlib import Path

import torch

LEROBOT_TAG = "v0.6.0"
LEROBOT_DIR = Path("/content/lerobot")

# av が要る理由は非自明なので記録しておく。
# lerobot/policies/factory.py は groot の設定を import し、その連鎖で
# groot/processor_groot.py に到達する。そこは
#     if TYPE_CHECKING or _datasets_available:
#         from lerobot.datasets.lerobot_dataset import LeRobotDataset
# という分岐になっており、`datasets` が入っていると通る。すると
# lerobot/datasets/__init__.py が require_package("av") を呼ぶ。
# **Colab には datasets が既定で入っているため、av も必要になる。**
# 提出物の環境には datasets が無いので分岐がスキップされ、av は要らない。
#
# バージョンを固定するのは、LeRobot v0.6.0 が av>=15,<16 を宣言しているため。
# 素で入れると 17 系になり、lerobot/datasets/pyav_utils.py が使う
# av.option が無くなっていて AttributeError になる。
run("pip install -q fastapi uvicorn msgpack 'av>=15.0.0,<16.0.0'")

if not LEROBOT_DIR.exists():
    run(f"pip uninstall -y -q lerobot torchao", check=False)
    run(f"git clone --quiet --depth 1 --branch {LEROBOT_TAG} "
        f"https://github.com/huggingface/lerobot.git {LEROBOT_DIR}")

    # bf16 非対応の GPU（T4 など）では fp16 に書き換える。
    # 学習用ノートブックが同じ置換をしており、揃えないと挙動が変わる。
    if not torch.cuda.is_bf16_supported():
        smolvlm = LEROBOT_DIR / "src/lerobot/policies/smolvla/smolvlm_with_expert.py"
        source = smolvlm.read_text(encoding="utf-8")
        smolvlm.write_text(
            source.replace('torch_dtype="bfloat16",', 'torch_dtype="float16",', 1),
            encoding="utf-8",
        )
        print("bf16 非対応の GPU のため fp16 へ書き換えました")

    run(f"pip install -q --upgrade -e '{LEROBOT_DIR}[smolvla]'")
    run("pip uninstall -y -q torchao", check=False)

# clone 先 /content/lerobot には __init__.py が無いため、`import lerobot` が
# これを名前空間パッケージとして掴んでしまい lerobot.__file__ が None になる。
# 学習用ノートブックと同じく、実体のある src/ を sys.path の先頭へ置く。
LEROBOT_SRC = LEROBOT_DIR / "src"

for name in [m for m in sys.modules if m == "lerobot" or m.startswith("lerobot.")]:
    del sys.modules[name]

sys.path = [p for p in sys.path if p not in {str(LEROBOT_DIR), str(LEROBOT_SRC)}]
sys.path.insert(0, str(LEROBOT_SRC))

import importlib
importlib.invalidate_caches()

import lerobot
assert lerobot.__file__ is not None, "lerobot が名前空間パッケージとして読まれています"
assert LEROBOT_SRC.resolve() in Path(lerobot.__file__).resolve().parents, \
    f"LeRobot の読込先が正しくありません: {lerobot.__file__}"

print("lerobot:", Path(lerobot.__file__).parent)
print("bf16 対応:", torch.cuda.is_bf16_supported())
```

> **`torch.cuda.is_bf16_supported()` が False なら、必ず上の書き換えを通すこと。**
> T4 は bf16 非対応である。L4（本番と同じ）や A100 なら不要。
> 学習を T4 で回して評価を L4 で回すと、この一点だけ条件が変わる。

> **`lerobot.__file__` が `None` になるのは、clone 先の `/content/lerobot` が
> `__init__.py` を持たないためである。** `/content` が `sys.path` に入っていると
> そこが名前空間パッケージとして拾われ、実体の `src/lerobot` に届かない。
> 上の 3 行（`sys.modules` の掃除・`sys.path` の差し替え・`invalidate_caches`）が
> それを避けている。**この対処は別プロセスには効かない**ので、
> サーバー起動時にも同じ手当てが要る（手順 6 で `PYTHONPATH` に入れている）。

## それでも `av` 関連で落ちるとき

`lerobot.datasets` の中で別の不整合が出る場合は、**`datasets` を消して分岐そのものを
通らなくする**方が確実である。提出物の環境と同じ状態になり、起動も速くなる。

```python
run("pip uninstall -y -q datasets", check=False)

# 分岐が閉じたことを確認する
check = (
    f"PYTHONPATH={LEROBOT_SRC} python -c "
    "\"from lerobot.utils.import_utils import _datasets_available; "
    "print('datasets available:', _datasets_available); "
    "from lerobot.policies.factory import make_pre_post_processors; "
    "print('factory の import: 成功')\""
)
run(check, tail=20)
```

`datasets available: False` と出れば、`lerobot.datasets` は読まれない。
この評価では `datasets` を使わないので消して問題ない。

## どの `policy_server.py` を使うか

**リポジトリの [policy_server.py](../../../solution/policy_server.py) をそのまま使う。**
提出した zip の中身と同一のファイルであり、書き換えは要らない。

| | |
|---|---|
| 場所 | `solution/policy_server.py` |
| 提出物との関係 | **完全に同一**。zip はこのファイルをルート直下に入れているだけ |
| Colab での違い | `vendor/` が無いので、同梱版ではなくインストール済みの LeRobot を使う |
| つまみ | `PARC_*` の 4 つ。**`PARC_N_ACTION_STEPS` と `PARC_EMA_ALPHA` は既定で有効**（下表） |

## `PARC_*` の 4 つのつまみ

`policy_server.py` が読む環境変数はこれで全部である
（`HF_HUB_OFFLINE` などのキャッシュ関連を除く）。**`POLICY_ENV` に書かなければ
既定値が使われる。** 4 つは独立で、組み合わせても構わない。

| 変数 | 既定 | 何をするか | いつ効くか |
|---|---|---|---|
| **`PARC_N_ACTION_STEPS`** | **`5`** | 1 回の推論で作る 50 ステップ分の chunk のうち、**何ステップ実行してから作り直すか**。小さいほど観測を見直す頻度が上がり、開ループ中の誤差の蓄積が打ち切られる。代償は推論回数（1 ステップあたり `38.6 ms + 850 ms / n`） | **常時。既定で有効** |
| **`PARC_EMA_ALPHA`** | **`0.3`** | 出力した行動に**直前の行動をこの割合だけ混ぜる**。`0.3` なら「3 割は前の行動」。動きの行き過ぎが減り、軌道が滑らかになる。**並進と回転の 6 次元だけ**で、グリッパは触らない（中間値になると掴めなくなるため） | **常時。既定で有効** |
| `PARC_BLEND_STEPS` | `0`（無効） | **chunk の継ぎ目の直後だけ**、直前の行動へ寄せて段差を消す。`3` なら継ぎ目から 3 ステップ、重みを 3/4 → 2/4 → 1/4 と減らしながら混ぜる。EMA と違い**常時はかからない**。こちらも 6 次元のみ | 再計画の頻度を上げて jerk が悪化したとき |
| `PARC_SAMPLES` | `1`（無効） | chunk を**この本数だけ引き直し、互いに最も近いもの（medoid）を採用**する。平均ではなく実在の 1 本を選ぶので、グリッパが中間値にならない。外れた chunk を捨てられる代わりに**推論コストがこの倍数になる** | 上の 3 つを詰めたあと |

**`PARC_EMA_ALPHA` と `PARC_SAMPLES` は別物である。** 前者は出力した行動を
時間方向にならす処理、後者は同じ観測から chunk を複数引いて 1 本選ぶ処理で、
効く場所も計算コストも違う。

> 運営 Q&A No.10 で「VLA が生成した Action を、全タスク共通のルールで整える処理」は
> 禁止事項に含まれないと明言されている。4 つとも**モデル出力と過去の行動しか
> 参照しない**（タスク名・seed・シミュレータの内部状態は見ない）。

**`policy_server.py` は手順 1-1 で `/content/` へ上げてある。**
下のセルが、それを専用ディレクトリへ移す。

```python
# === policy_server.py を専用ディレクトリへ配置する ===
# /content 直下のままだと、サーバーの別プロセスで `import lerobot` が
# clone 先の /content/lerobot（__init__.py 無し）を掴んでしまう。
# スクリプトの置き場所は sys.path の先頭に入るため、隣に lerobot/ が
# 無いディレクトリへ移しておく。
from pathlib import Path
import shutil

SUBMISSION_DIR = Path("/content/submission")
SUBMISSION_DIR.mkdir(exist_ok=True)
SERVER = SUBMISSION_DIR / "policy_server.py"

uploaded = Path("/content/policy_server.py")
if uploaded.is_file():
    shutil.copy2(uploaded, SERVER)
assert SERVER.is_file(), "policy_server.py を /content/ に置いてください"

# model_weights は policy_server.py と同じ階層に無ければならない。
weights_link = SUBMISSION_DIR / "model_weights"
if not weights_link.exists():
    weights_link.symlink_to(WEIGHTS_DIR)

assert not (SUBMISSION_DIR / "lerobot").exists(), \
    "この階層に lerobot/ があると import が壊れます"
print("配置 OK:", SERVER)
print("  重み:", sorted(p.name for p in weights_link.iterdir())[:4], "…")
```

---

---

**次は [5. 評価するタスクを決める](05_suite.md) へ。**
