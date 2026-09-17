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
| `vendor/lerobot/policies/molmoact2/modeling_molmoact2.py` | 重みの読み込みを 2 か所変更した。**この提出物は π0.5 を使うので、この経路は動かない** |

## 2. 重み

**Gemma is provided under and subject to the Gemma Terms of Use found at
ai.google.dev/gemma/terms**

| 何 | 権利者 | ライセンス |
|---|---|---|
| `model_weights/`（この提出物の方策の重み） | 出発点は **`lerobot/pi05_libero_base`**。**そこへ参加者が追加学習したもの**で、Gemma Terms of Use にいう Model Derivative にあたる | **Gemma Terms of Use**（`lerobot/pi05_libero_base` のモデルカードの表示は `license: gemma`） |

**Gemma Terms of Use の全文は同梱していない。**上記 URL が正本である。
利用にあたっては同 Terms の Section 3.2（Use Restrictions）と
Gemma Prohibited Use Policy（ai.google.dev/gemma/prohibited_use_policy）が適用される。

## 3. 学習に使ったデータ

| 何 | 権利者 | ライセンス | 全文 |
|---|---|---|---|
| LIBERO / LIBERO-plus 由来の実演データ | Lifelong Robot Learning ほか | **MIT License** | `licenses/LICENSE-MIT-LIBERO.txt` |

**学習データそのものはこの提出物に含まれていない**（含まれるのは学習の結果の重みだけである）。
出典を明らかにするために記載する。
