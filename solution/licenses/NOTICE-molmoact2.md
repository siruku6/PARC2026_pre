# NOTICE

この提出物に含まれる第三者の成果物と、そのライセンスを示す。
**ライセンス全文は同梱の `licenses/` にある。**

## 1. 同梱している第三者コード

| 何 | 版 | 権利者 | ライセンス | 全文 |
|---|---|---|---|---|
| LeRobot（`vendor/lerobot/`） | v0.6.0 | The Hugging Face team | **Apache License 2.0** | `licenses/LICENSE-Apache-2.0-lerobot.txt` |

**各ファイルの著作権表示とライセンスヘッダは、配布元のまま保持している。**

**変更したファイルが 1 つある**（Apache License 2.0 第 4 条 (b) に基づく告知）。

| ファイル | 変更の内容 |
|---|---|
| `vendor/lerobot/policies/molmoact2/modeling_molmoact2.py` | 2 か所を変更した。(1) `_load_hf_model`——`checkpoint_path` に safetensors が無い場合は素の重みを読まず、設定だけからモデルを組む分岐を足した（提出物の容量を 2 倍にしないため）。(2) `_load_as_safetensor`——PEFT が付ける名前（`base_model.model.` / `.base_layer.`）に合わせて保存済み重みを読み直す分岐を足した |

## 2. 重み

| 何 | 権利者 | ライセンス |
|---|---|---|
| `model_weights/`（この提出物の方策の重み） | 出発点は **`allenai/MolmoAct2-LIBERO-LeRobot`**（Allen Institute for AI）。**そこへ参加者が LoRA を学習して載せたもの** | 出発点のモデルカードの表示は **Apache License 2.0**。全文は `licenses/LICENSE-Apache-2.0-lerobot.txt` と同一の文面である（Apache License 2.0 の本文は権利者によらず同じ） |

## 3. 学習に使ったデータ

| 何 | 権利者 | ライセンス | 全文 |
|---|---|---|---|
| LIBERO / LIBERO-plus 由来の実演データ | Lifelong Robot Learning ほか | **MIT License** | `licenses/LICENSE-MIT-LIBERO.txt` |

**学習データそのものはこの提出物に含まれていない**（含まれるのは学習の結果の重みだけである）。
出典を明らかにするために記載する。
