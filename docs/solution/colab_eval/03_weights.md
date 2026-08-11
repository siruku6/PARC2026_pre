# 3. 評価する重みを用意する

**素の重み**と**追加学習した重み**のどちらを評価するかで分岐する。まず素の重みで
基準を取り、そのあと学習後と比べるのが順序である。

素の重みは **`0.05832` を出した提出物とまったく同じもの**なので、
ここで出る数字は本番スコアと直接比べられる。

```python
# === 評価する重みを用意する ===
# USE_FINETUNED = False なら HF から素の重みを取る（Drive 不要）。
# True なら Drive のマージ済みモデルを使う。
#
# DRIVE_MERGED_DIR は COLAB.md のセル E が書き出す先である。
#   merged_on_drive = DRIVE_RUN_DIR / "merged_model"
# つまり  /content/drive/MyDrive/parc2026/<RUN_NAME>/merged_model  になる。
# <RUN_NAME> は学習セルの同期ログに出ている名前をそのまま使う。
# **マージセルを実行するまで存在しない。** 学習中は False のままにすること。
USE_FINETUNED = False
DRIVE_MERGED_DIR = "/content/drive/MyDrive/parc2026/<RUN_NAME>/merged_model"

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
print(sorted(present))
```

> 提出物の `model_weights/` はこのセルと**同じ中身**である
> （ルート直下の 6 ファイル ＋ `train_config.json` ＋ `vlm/`）。
> 素の重みでは `README.md` と `.gitattributes` が余分に落ちるが、
> 読み込みには影響しない。

---

---

**次は [4. 方策サーバーを立てる準備](04_policy_server.md) へ。**
