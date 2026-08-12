# 1. 準備 — Drive をつなぎ、ファイルを置き、GPU と EGL を確認する

> **いまは [`ce.prepare()`](../../../solution/tools/colab_eval.py) が全部やる。**
> **コードの正はそちらである。** このページは**何が起きているかと、その理由**を
> 残す。下のセルは、関数を使わずに貼り付けで進めるときの控えである。

新しいノートブックを作り、**ランタイムのタイプを GPU に変更**してから始める。

## 1-1. 手元から 7 ファイルを上げる（**先にこれ。手順 2 で必要になる**）

**手順 2 は `/content/patch_kit_for_variants.py` を呼ぶ。**
**上げていないと手順 2 で落ちる。**

手元でこれを走らせる。

```bash
# 前の回と同じ 84 件で比べるなら、その CSV も一緒に集める
python3 solution/tools/collect_colab_files.py \
    --val-tasks outputs/0811_101824_base_n5_ep1_max300/VAL_TASKS.csv
open work/colab_upload
```

**中身をまとめて Colab の左のファイルペインへドラッグする。** 置き場所は
**`/content/` の直下**（手順 4-2 が `policy_server.py` を専用ディレクトリへ移す）。

| 上がるもの | 使うのは |
|---|---|
| `patch_kit_for_variants.py` | **手順 2**（これが無いと落ちる） |
| `rollout_recorder.py` | 手順 2（上のパッチがキットへ複製する） |
| `policy_server.py` | 手順 4-2 |
| `make_validation_suite.py` | 手順 5-1 |
| `check_validation_suite.py` | 手順 5-1 |
| `summarize_rollout_videos.py` | 手順 9 |
| **`VAL_TASKS.csv`**（`--val-tasks` を渡したとき） | **手順 5-1。**前の回と同じ 84 件で比べられる |

**毎回コピーを作り直すので、リポジトリを直したのに古いものを上げてしまう
取り違えが起きない。**

```python
# === 上がっているか確かめる ===
from pathlib import Path

need = ["patch_kit_for_variants.py", "rollout_recorder.py", "policy_server.py",
        "make_validation_suite.py", "check_validation_suite.py",
        "summarize_rollout_videos.py"]
missing = [n for n in need if not Path(f"/content/{n}").is_file()]
print("不足:", missing if missing else "無し")
print("VAL_TASKS.csv:", "あり（前の回と同じ 84 件で比べられる）"
      if Path("/content/VAL_TASKS.csv").is_file() else "**無い。手順 5-1 で作り直す**")
assert not missing, "上のファイルを /content/ へドラッグしてから進むこと"
```

## 1-2. Drive をつなぐ

**評価の結果はすべて Drive へ残す。** `/content` はランタイムが切れると消える。
**先につないでおけば、手順 8 が `[結果]` 行を 1 本ごとに書き込める。**

```python
# === Drive をつなぐ ===
# **初回は対話認証がある。** 後回しにすると、手順 8 の途中で認証を求められて
# 評価が止まる。**ここで済ませておく。**
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
assert Path("/content/drive/MyDrive").is_dir(), "Drive がつながっていない"
print("Drive: つながった")
```

## 1-3. GPU と EGL を確認する

```python
# === GPU と EGL の確認 ===
# MuJoCo は EGL を使うと GPU で描画する。ここがソフトウェア実装に落ちると
# シミュレーションが 11 倍遅くなり、この手順の意味が無くなる。
import ctypes, subprocess, sys
from pathlib import Path

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print("python:", sys.version.split()[0])

try:
    ctypes.CDLL("libEGL.so.1")
    print("libEGL: あり")
except OSError:
    print("libEGL: 無し → 次のセルの apt で入る")

# **ここが Colab の最大の罠。** libglvnd は「ローダ」でしかなく、
# どのベンダ実装を使うかは egl_vendor.d の JSON が決める。NVIDIA の登録が
# 無いと Mesa のソフトウェア実装を掴み、**例外も警告も出さずに 11 倍遅くなる。**
icd = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
if not icd.is_file():
    icd.parent.mkdir(parents=True, exist_ok=True)
    icd.write_text(
        '{\n'
        '    "file_format_version" : "1.0.0",\n'
        '    "ICD" : {\n'
        '        "library_path" : "libEGL_nvidia.so.0"\n'
        '    }\n'
        '}\n'
    )
    print(f"{icd} を作成した（NVIDIA の EGL を使わせる）")
else:
    print(f"{icd} は既にある")
```

GPU 名が出なければ**ランタイムのタイプが GPU になっていない**。先に直すこと。

---

---

**次は [2. 配布キットをセットアップする](02_kit.md) へ。**
