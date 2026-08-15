# 02. モデル情報（レポート必須欄 2）

**すべて提出物そのものから取った値である。**レポートの記載と提出物に差があると
本選の途中で失格になる（予選 PDF p23）ので、**記憶やこのリポジトリの `.md` から
写さない。**

## 値

| 欄 | 値 |
|---|---|
| モデル名およびバージョン | **`policy_server` 4.2.0**（`solution/policy_server.py` の `__version__`。採点ログの先頭行に出る） |
| ベースモデル名 | **π₀.₅（Pi05）** —— Physical Intelligence の Vision-Language-Action モデルの LeRobot 実装 |
| ベース重みの入手元 | **Hugging Face `lerobot/pi05_libero_base`** |
| 使用した重みの revision | **`a217bfd3b14673cf2ce597e69997ab21866438dd`** |
| **提出 zip の sha256** | **`5cf6b9836f46805ba10bdde93ead02913c65d050adfb20d6800a3788b4cdb9ee`** |
| 提出 zip のファイル名 | `submission_4.2.0_rp3000_blend0.zip`（541 ファイル・9,378 MB） |
| モデル構成 | π₀.₅（VLM は PaliGemma 系、行動側は flow matching の action expert）。**パラメータ 4.14B**・safetensors のテンソル **813 個** |
| 推論方式 | **flow matching**。ノイズ除去 **10 段**（`num_inference_steps`） |
| **Action chunk** | **有り。chunk は 50 歩ぶん生成し、先頭 16 歩だけ実行して作り直す** |

## Action chunk の書き方に注意

**設定ファイルとサーバーの既定が食い違う。**レポートには**実際に動いた値**を書く。

| | 値 | どこ |
|---|---|---|
| `chunk_size`（生成する長さ） | **50** | `work/artifacts/pi05_rp3000/config.json` |
| `n_action_steps`（設定ファイル） | 10 | 同上 |
| **`n_action_steps`（実際に動いた値）** | **16** | `policy_server.py` が `PARC_N_ACTION_STEPS` の既定 16 で上書きする。**採点ログの先頭行に出ている** |

```
[parc] policy_server 4.2.0  model=pi05  device=cuda  num_inference_steps=10
       n_action_steps=16  ema_alpha=0.3  blend_steps=0  samples=1  rtc=off  normalize=on
```

**この 1 行が、提出物の設定を採点環境側から証明している。**レポートにはこの行を
そのまま載せるのが、失格条項に対していちばん安全である。

## 推論時の後処理（工夫点の欄にも効く）

| 後処理 | 既定 | 効果 |
|---|---|---|
| EMA 平滑化 | `ema_alpha=0.3` | 直前の行動を 3 割混ぜる。姿勢 6 次元のみ。**グリッパ（7 次元目）は素通し** |
| 指示文の正規化 | `normalize=on` | 評価が渡す文を学習データの 40 種のうち最も近いものへ寄せる |
| chunk 継ぎ目の平滑化 | **`blend_steps=0`（切ってある）** | 入れると悪化した（[06](06_step_count_drives_score.md)） |
| chunk の複数サンプリング | **`samples=1`（切ってある）** | — |
| RTC | **`rtc=off`** | 入れると悪化した（[06](06_step_count_drives_score.md)） |

## 確かめ直すコマンド

```bash
# ベース重みの revision と、そのライセンス
.venv/bin/python3 -c "
from huggingface_hub import HfApi
i = HfApi().repo_info('lerobot/pi05_libero_base',
                      revision='a217bfd3b14673cf2ce597e69997ab21866438dd')
print(i.sha, i.card_data.get('license'))"

# 提出した重みの構成
python3 -c "
import json; c = json.load(open('work/artifacts/pi05_rp3000/config.json'))
print({k: c[k] for k in ('type','chunk_size','n_action_steps','num_inference_steps')})"

# 実際に動いた設定（採点ログから）
grep -m1 'policy_server' outputs/260814_230032_pi0.5_4.2.0.md

# 提出 zip のハッシュ（Mac 側）
shasum -a 256 submission_4.2.0_rp3000_blend0.zip
```

## 未取得

| 欄 | 状態 |
|---|---|
| `model.safetensors` 単体の sha256 | 計算中（9.35 GB）。**zip のハッシュがあれば必須欄は満たせる**が、あると親切 |
| パラメータ数 4.14B の 1 次資料 | 学習ログの出力から取る。**まだ引用元を特定していない** |
