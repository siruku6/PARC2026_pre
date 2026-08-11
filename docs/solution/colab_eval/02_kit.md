# 2. 配布キットを取得してセットアップする

**5〜10 分かかる。** `setup.sh` をそのまま使う（本番と同じ手順）。
キットは公開リポジトリなので clone できる。

```python
# === 配布キットのセットアップ ===
# setup.sh は python3.10 を前提にしている。Colab は 3.12 なので apt で 3.10 を足す。
# 評価パイプライン側は 3.10 の venv に閉じ込め、方策側は Colab の 3.12 を使う。
#
# 失敗したときに原因が分かるよう、出力は多めに残す。
import os, subprocess, shutil
from pathlib import Path

KIT_DIR = Path("/content/PARC2026_pre")
LOG_DIR = Path("/content/setup_logs"); LOG_DIR.mkdir(exist_ok=True)

def run(command: str, check: bool = True, tail: int = 30, log: str | None = None) -> int:
    """シェルコマンドを実行し、出力の末尾を表示する。全文はログに残す。"""
    print(f"$ {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
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

run("apt-get -qq update", log="apt_update.log")
run("apt-get -qq install -y libosmesa6 libegl1 libglfw3 libglew-dev "
    "libmagickwand-dev unzip build-essential", log="apt_system.log")

# python3.10 が入るなら使う。入らなければ Colab の Python（3.12）で作る。
# ここは採点環境への忠実性の話ではない（それは Docker 側で済んでいる）。
# キットが gym==0.25.2 / robosuite==1.4.0 という古い版を固定しているため、
# 素性の分かっている 3.10 の方が安全というだけである。
import sys
run("apt-get -qq install -y python3.10 python3.10-venv python3.10-dev",
    check=False, log="apt_python310.log")

KIT_PYTHON = shutil.which("python3.10") or sys.executable
print(f"\nキット側の Python: {KIT_PYTHON}")
if "3.10" not in KIT_PYTHON:
    print("  python3.10 が使えないため Colab の Python を使う。"
          "gym / robosuite が古いため、後段で落ちたらログを見ること。")

if not KIT_DIR.exists():
    run(f"git clone --depth 1 https://github.com/matsuolab/PARC2026_pre {KIT_DIR}")

os.chdir(KIT_DIR)

# Colab は MPLBACKEND に matplotlib_inline を設定している。キットの venv には
# そのパッケージが無いため、libero が matplotlib を import した瞬間に
# 「'module://matplotlib_inline.backend_inline' is not a valid value for backend」
# で落ちる。評価は画面表示をしないので、以降ずっと Agg にしておく。
os.environ["MPLBACKEND"] = "Agg"

run(f"PYTHON={KIT_PYTHON} bash setup.sh", tail=40, log="setup.log")

# torch 2.11.0 は setuptools<82 を要求するが、Colab の venv には新しいものが入る。
# 警告のまま放置すると import で落ちることがあるため揃えておく。
run(f"{KIT_DIR}/venv/bin/pip install -q 'setuptools<82'", check=False)

# --- キットにパッチを当てる ---
# **3 か所ある。** 中身は patch_kit_for_variants.py に集約してあるので、
# Docker で確かめたものとまったく同じコードが当たる。
#   1. エピソードごとの結末をログに出す（`[結果]` 行。これが無いと
#      成功・衝突・未完了の区別が付かず、何にぶつかったかも分からない）
#   2. 初期状態ファイルを CSV の指定から取る（検証用スイート用）
#   3. 衝突判定の bddl を `_view_` より前で探す（同上）
#
# **example の 4 タスクで評価するときも 1 は要る。** 2 と 3 は検証用スイートの
# ときだけ効くが、当てておいて害は無い。
run(f"{KIT_PYTHON} /content/patch_kit_for_variants.py --kit-dir {KIT_DIR}", tail=10)

# 動作確認。setup.sh 自身の確認は MPLBACKEND のせいで落ちるため、ここでやり直す。
verify = (
    f'MPLBACKEND=Agg PYTHONPATH="{KIT_DIR}/LIBERO-plus:{KIT_DIR}:{KIT_DIR}/compe" '
    f'{KIT_DIR}/venv/bin/python -c '
    '"import torch, libero.libero.benchmark; '
    'from compe.t1 import register_t1; register_t1(); '
    'print(\'suite 登録 OK / torch\', torch.__version__)"'
)
run(verify, tail=20, log="verify.log")
print("キットのセットアップ完了")
```

> **`setup.sh` の最後の「動作確認（suite 登録）」は Colab では必ず失敗する。**
> `MPLBACKEND` が原因であり、セットアップ自体（`1/5`〜`5/5`）が完了していれば問題ない。
> 上のセルは `setup.sh` を通したあとに、環境変数を直して確認をやり直している。
> `setup.sh` は再実行しても venv・clone・アセットを作り直さないため、
> 失敗後にこのセルをもう一度流すだけでよい。

> **どちらの Python でも `setup.sh` は `venv` を作る。** 方策側（Colab の 3.12 本体）とは
> 別の環境になるため、`numpy==1.26.4` が方策側へ漏れることはない。
> **2 プロセス構成は Python の版に関係なく必要である。**

## `setup.sh` が失敗したとき

失敗すると末尾 40 行が表示され、全文が `/content/setup_logs/setup.log` に残る。
**`[setup] N/5` のどこまで進んだかで切り分ける。**

| 止まった段階 | 何が起きているか | 対処 |
|---|---|---|
| `1/5 venv + 依存` | `python3.10` が無い、または pip が失敗 | 下記「python3.10 が入らないとき」 |
| `2/5 LIBERO-plus / LIBERO の取得` | git clone の失敗 | 再実行する（一時的な失敗が多い） |
| `3/5 パッチ` | ほぼ起きない | ログを見る |
| `4/5 アセット` | `assets.zip`（HF）の取得や `unzip` の失敗 | `unzip` が入っているか確認して再実行 |
| `5/5 libero 設定` | ほぼ起きない | ログを見る |

止まった段階が分からないときは次で確認する。

```python
print((LOG_DIR / "setup.log").read_text()[-4000:])
```

## 依存のインストールで落ちるとき（`1/5` で止まる）

`gym==0.25.2` や `robosuite==1.4.0` は 2022 年前後の版で、
**新しい Python では wheel が無い / ビルドに失敗する**ことがある。
どのパッケージで落ちたかはログの末尾に出る。

```python
print((LOG_DIR / "setup.log").read_text()[-4000:])
```

`python3.10` が使えていればまず起きない。Colab の Python（3.12）で落ちた場合は、
`python3.10` を入れられないか先に確認する価値がある。

```python
run("apt-cache policy python3.10", check=False)
```

`Candidate: (none)` なら apt に無い。その場合は deadsnakes を足す手もあるが、
**そこまでするより先に、落ちたパッケージ 1 つだけを新しい版で入れて
`setup.sh` を再実行する**方が速いことが多い。

> **`setup.sh` は `~/.libero/config.yaml` を書き換える。** Colab は使い捨てなので問題ないが、
> ローカルで同じことをすると既存の LIBERO 設定が `.bak` に退避される。

## ログのパッチについて

上のセルの最後で `pipeline/rollout.py` にログ行を 1 つ足している。
**判定にも採点にも触らない。** 得られるのはこれである。

```
  [結果] put_the_bowl_on_the_stove_light_11 ep3: 衝突 214 steps  ぶつけた物体: wooden_cabinet=3.7mm
  [結果] put_the_bowl_on_the_stove_light_11 ep4: 成功 96 steps
```

| 無いと困ること | |
|---|---|
| **何にぶつかったのか分からない** | 衝突は失敗の大半を占める。対象物が分からないと次の手が決まらない |
| **45 分間まったく出力が変わらない** | 進んでいるのか止まったのか区別できない |

> **キットは `git clone` で取ってくるため、リポジトリ側の修正は反映されない。**
> このセルで当てる必要がある。B を再実行してもパッチは消えない
> （`git clone` は `KIT_DIR` が無いときだけ走る)。2 回目以降は
> `適用済み` と出るだけで、二重には当たらない。

---

---

**次は [3. 評価する重みを用意する](03_weights.md) へ。**
