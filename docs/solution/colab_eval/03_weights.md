# 3. 評価する重みを用意する

> **いまは [`ce.prepare_weights()`](../../../solution/tools/colab_eval.py) が全部やる。**
> どちらを評価するかは
> [colab_eval_config.py](../../../solution/tools/colab_eval_config.py) の
> `USE_FINETUNED` / `DRIVE_WEIGHTS_DIR` / **`EXPECT_DIGEST`** で決める。
> **`EXPECT_DIGEST` と実物が違えば、読み込む前に止まる。**
> このページは**何が起きているかと、その理由**を残す。

**素の重み**と**手元で作った重み**のどちらを評価するかで分岐する。
まず素の重みで基準を取り、そのあと比べるのが順序である。

素の重みは **`0.05832` を出した提出物とまったく同じもの**なので、
ここで出る数字は本番スコアと直接比べられる。

## どれをどこへ置くか

| 評価するもの | Colab への持ち込み方 |
|---|---|
| **素の重み** | **何もしない。**`USE_FINETUNED = False` で HF から落ちる（Drive も不要） |
| **手元で作った重み**（学習後・λ で混ぜたもの） | **Drive へ置いて、その場所を `DRIVE_MERGED_DIR` に書く** |

**Drive はランタイムが落ちても消えない。** 一度上げれば、やり直しのときも
そのまま使える（**912 MB を上げ直さなくてよい**）。

### 手元から Drive へ上げる

**ブラウザで Drive を開き、フォルダごとドラッグする。**
置き場所は決めておく —— `MyDrive/parc2026/FOR_COLAB_EVAL/<名前>/`。

```bash
# 上げるのはこのディレクトリ丸ごと（16 ファイル・912 MB）
open work/artifacts/model_weights_lam025
```

**`vlm/` を含めて丸ごと上げる。** `blend_weights.py` と
`install_weights.py` は必要なファイルを全部揃えて出している。

### 上げたあと、中身が思ったものか確かめる

**3 つとも 912 MB で同じ大きさなので、大きさでは見分けられない。**
**ハッシュで見る。**

```python
!shasum -a 256 /content/model_weights/model.safetensors
```

| 先頭 16 桁 | 何の重みか |
|---|---|
| `a67a056ab0761afa` | 素の重み（8/11 に 84 件で評価済み。成功率 0.476） |
| `2cd9faabb421ce22` | lora20k（4 回目の提出。本番 0.016。評価済み） |
| `952171f908eeae99` | **λ=0.25**（素の重み ＋ 学習で動いた分 × 0.25） |

**`USE_FINETUNED = False` のまま流すと、素の重みが落ちてくる。**
**既に評価済みのものを 2.6 時間かけて測り直すことになるので、必ず確かめる。**

```python
# === 評価する重みを用意する ===
# USE_FINETUNED = False なら HF から素の重みを取る（Drive 不要）。
# True なら Drive のマージ済みモデルを使う。
#
# **どちらを評価するか。** True なら Drive に置いたものを使う。
#
# 置き場の例（上の表を参照）:
#   /content/drive/MyDrive/parc2026/FOR_COLAB_EVAL/model_weights_lam025
#     … 手元の work/artifacts/model_weights_lam025 を丸ごと上げたもの
#   /content/drive/MyDrive/parc2026/<RUN_NAME>/merged_model
#     … COLAB.md のマージセルが書き出す先（Colab で学習したとき）
#
# **False のままだと素の重みを測ることになる。** 既に評価済みなので、
# 手元で作ったものを測るつもりなら必ず True にして場所を書く。
USE_FINETUNED = True
DRIVE_MERGED_DIR = "/content/drive/MyDrive/parc2026/FOR_COLAB_EVAL/model_weights_lam025"

from pathlib import Path
from huggingface_hub import snapshot_download

WEIGHTS_DIR = Path("/content/model_weights")

if USE_FINETUNED:
    from google.colab import drive
    drive.mount("/content/drive")
    source = Path(DRIVE_MERGED_DIR)
    assert source.is_dir(), (
        f"{source} がありません。COLAB.md のセル E（マージ済みモデルを Drive へ）"
        "まで実行されているか確認してください"
    )
    run(f"rm -rf {WEIGHTS_DIR} && cp -r '{source}' {WEIGHTS_DIR}")
else:
    # このリポジトリには eval/（評価動画 402 ファイル・25.6 MB）が入っている。
    # 推論には不要なので除外する。落とすのはルート直下の 7 ファイルだけでよい。
    snapshot_download(
        "lerobot/smolvla_libero_plus",
        revision="7bb70aa5bc92b82c9239142775d3a173103567ff",
        local_dir=str(WEIGHTS_DIR),
        ignore_patterns=["eval/*", "eval/**"],
    )

# policy_server.py は model_weights/vlm/ からトークナイザを読む。
# VLM 本体の重みはポリシー側のチェックポイントに含まれるので、
# 設定とトークナイザだけで足りる（*.safetensors は落とさない）。
snapshot_download(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    local_dir=str(WEIGHTS_DIR / "vlm"),
    allow_patterns=["*.json", "*.txt", "*.model", "tokenizer*"],
)

# 推論に必須のファイル。train_config.json は学習スクリプトが書くもので、
# マージ済みモデル（save_pretrained の出力）には無いため必須にしない。
required = {
    "config.json", "model.safetensors",
    "policy_preprocessor.json", "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
}
present = {p.name for p in WEIGHTS_DIR.iterdir() if p.is_file()}
missing = required - present
assert not missing, f"重みが足りません: {missing}"
assert (WEIGHTS_DIR / "vlm" / "tokenizer_config.json").is_file(), "vlm のトークナイザがありません"

import json
config = json.loads((WEIGHTS_DIR / "config.json").read_text())
assert not config.get("use_peft"), "use_peft が真です。LoRA がマージされていません"

total = sum(p.stat().st_size for p in WEIGHTS_DIR.rglob("*") if p.is_file())
print(f"重みの準備完了: {total / 1e6:.0f} MB  （追加学習: {'あり' if USE_FINETUNED else 'なし'}）")

# **どの重みかをハッシュで出す。** 大きさは 3 つとも 912 MB で見分けられない。
# **既に評価済みのものを 2.6 時間かけて測り直す事故を、ここで止める。**
import hashlib
digest = hashlib.sha256((WEIGHTS_DIR / "model.safetensors").read_bytes()).hexdigest()
known = {
    "a67a056ab0761afa": "素の重み（8/11 に評価済み。成功率 0.476）",
    "2cd9faabb421ce22": "lora20k（4 回目の提出。本番 0.016。評価済み）",
    "952171f908eeae99": "λ=0.25",
}
print(f"model.safetensors: {digest[:16]}  … {known.get(digest[:16], '**記録に無い重み**')}")
```

> 提出物の `model_weights/` はこのセルと**同じ中身**である
> （ルート直下の 6 ファイル ＋ `train_config.json` ＋ `vlm/`）。
> 素の重みでは `README.md` と `.gitattributes` が余分に落ちるが、
> 読み込みには影響しない。

---

---

**次は [4. 方策サーバーを立てる準備](04_policy_server.md) へ。**
