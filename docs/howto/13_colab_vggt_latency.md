# 13. Colab の L4 で VGGT の推論時間だけを測る

**この 1 枚だけで実行できる。** セルを 4 つ順に貼るだけで、他のファイルは要らない。

| 節 | 中身 |
|---|---|
| [なぜ測るのか](#なぜ測るのか) | 手元は H100、採点は L4。**残っている仮定はこれ 1 つ** |
| [ランタイムの選び方](#ランタイムは-l4-を選ぶ他の-gpu-では答えにならない) | **L4 でなければ意味が無い** |
| [貼るセル 4 つ](#貼るセルは-4-つ) | そのままコピーできる形 |
| [読み方](#結果の読み方) | 何 ms なら通るか |
| [所要と費用](#所要と費用) | 5 分・無料枠で足りる |

---

## なぜ測るのか

**採点環境は L4 1 枚（VRAM 24 GB）で、手元の測定は H100-20C で取った。**
[plan/16](../plan/16_finals_initiative_priority.md) 7.3 節の実測はこうである。

| | 値 |
|---|---|
| 1 問い合わせの上限 | **10 秒。1 回でも超えるとそのトラックが 0 点** |
| いまの π0.5 の推論（本番設定 `num_inference_steps=20`・`samples=3`） | 1.5〜2.4 秒 |
| **残り** | **7.6〜8.5 秒** |
| **VGGT が足す分（H100-20C・518×518・2 枚・aggregator のみ）** | **72 ms** |
| 何倍まで遅くなってよいか | **105 倍** |

**L4 は H100 の 4〜8 倍遅いと見込んでいるが、これは見積もりであって実測ではない。**
**このセルはその見積もりを実測に置き換えるためのものである。**

**π0.5 は載せない。** 載せると RAM が足りずランタイムごと落ちる
（[11_colab_eval.md](11_colab_eval.md) の 1 節で 8/24 に確認済み）。
**VGGT だけなら重み 4.69 GB で、L4 の 24 GB にも Colab の RAM にも収まる。**

---

## ランタイムは L4 を選ぶ。他の GPU では答えにならない

**「ランタイムのタイプを変更」で GPU を `L4` にする。** T4 や A100 を引くと、
測った数字が採点環境の数字にならない。**セル 1 が L4 でなければ止まるようにしてある。**

---

## 貼るセルは 4 つ

### セル 1 —— ランタイムの確認（これだけは必ず最初）

```python
# 役割: L4 を引けたかの確認だけ。ここで止まったらランタイムを変更して最初からやり直す。
import subprocess

info = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
print(info)
assert "L4" in info, f"L4 ではない: {info}。ランタイムのタイプを L4 に変更する"
```

### セル 2 —— 依存を入れる（インストールとパッチはここに全部置く）

```python
# 役割: vggt の導入だけ。torch は Colab のものをそのまま使う（入れ直すとランタイム再起動が要る）。
# vggt は numpy 1.26 を要求して落としにくるので、入れたあとに numpy を戻す。
!pip install --quiet "git+https://github.com/facebookresearch/vggt.git"
!pip install --quiet --upgrade "numpy>=2.0,<2.3"

import numpy, torch
print("torch", torch.__version__, "/ numpy", numpy.__version__)
from vggt.models.vggt import VGGT
print("vggt を import できた")
```

### セル 3 —— 設定（変えるのはこのセルだけ）

```python
# 役割: つまみだけ。実行はセル 4。ここと実行を混ぜると、直したつもりの値が使われない。
MODEL_ID   = "facebook/VGGT-1B"   # 商用可の版は門があるので、測るのは非商用版でよい（構造は同じ）
IMAGE_SIZE = 518                  # VGGT の既定。14 の倍数でないと落ちる（256 は不可、252 は可）
N_VIEWS    = 2                    # front と wrist で 1 問い合わせ
WARMUP     = 3
ITERS      = 20
```

### セル 4 —— 実行と結果の書き出し

```python
# 役割: 実行と出力の後始末を全部ここで持つ。
# 使うのは aggregator だけである。Evo-0（arXiv 2507.00416）が取り出すのは
# 最終層のトークンで、深度・点群・カメラ姿勢の頭は動かさなくてよい。
import json, time, pathlib, torch
from vggt.models.vggt import VGGT

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
model = VGGT.from_pretrained(MODEL_ID).to("cuda").eval()
load_sec = time.perf_counter() - t0
weights_gb = torch.cuda.max_memory_allocated() / 1024**3

images = torch.rand(1, N_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(WARMUP):
        model.aggregator(images)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.aggregator(images)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

times.sort()
result = {
    "gpu": torch.cuda.get_device_name(0),
    "model": MODEL_ID,
    "image_size": IMAGE_SIZE,
    "n_views": N_VIEWS,
    "iters": ITERS,
    "load_sec": round(load_sec, 1),
    "weights_gb": round(weights_gb, 2),
    "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "median_ms": round(times[len(times) // 2] * 1000),
    "max_ms": round(times[-1] * 1000),
}
print(json.dumps(result, ensure_ascii=False, indent=2))

# H100-20C の実測 72 ms（518×518・2 枚・aggregator のみ）に対する比
print(f"H100-20C 比: {result['median_ms'] / 72:.1f} 倍")
print(f"残り 7.6 秒に対する余裕: {7600 / result['median_ms']:.0f} 倍")

# 生の結果を残す。ランタイムが切れても手元に貼れるよう、画面にも JSON を出してある。
out = pathlib.Path(f"/content/vggt_latency_{result['gpu'].replace(' ', '_')}_{IMAGE_SIZE}.json")
out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print("書き出した:", out)
```

---

## 結果の読み方

**見るのは `median_ms` の 1 つだけである。**

| `median_ms` | 判定 |
|---|---|
| **1,000 未満** | **通る。**残り 7.6 秒に対して 7 倍以上の余裕がある。**見込みどおり** |
| 1,000〜3,000 | 通るが余裕が 2.5〜7 倍。**`IMAGE_SIZE` を 252 に落とすことを検討する**（H100 では 72 → 63 ms） |
| **3,000 以上** | **見込みが外れている。**π0.5 の推論 2.4 秒と足すと 5.4 秒を超え、ばらつき次第で 10 秒に触れる。**接ぐ手そのものを考え直す** |

**`peak_gb` も見る。** 5 GB 前後なら、採点環境の L4（24 GB）で π0.5 の約 10 GB と
同居できる。**10 GB を超えていたら報告してほしい。**

---

## 所要と費用

| | |
|---|---|
| セル 1〜4 の合計 | **5 分程度**（うち重みの取得が 1〜2 分） |
| 費用 | **無料枠で足りる。**GPU の占有は数分 |
| 落ちても失うもの | 無い。**やり直せばよいだけ**で、学習も評価も走らせていない |

**報告してほしいのは、セル 4 が出力する JSON をそのまま貼ったものである。**

---

## あとで見る資料

| 知りたいこと | ファイル |
|---|---|
| なぜ VGGT なのか・どの重みを使うのか | [plan/16](../plan/16_finals_initiative_priority.md) 7 節 |
| 手元（H100-20C）で測った同じ数字 | [plan/16](../plan/16_finals_initiative_priority.md) 7.3 節 |
| 同じ測定を手元で回す道具 | [measure_vggt_latency.py](../../solution/tools/analysis/measure_vggt_latency.py) |
| Colab で π0.5 の評価が回らない理由 | [11_colab_eval.md](11_colab_eval.md) 1 節 |
