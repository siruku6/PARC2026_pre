# 1. GPU と EGL を確認する

新しいノートブックを作り、**ランタイムのタイプを GPU に変更**してから最初のセルに貼る。

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
