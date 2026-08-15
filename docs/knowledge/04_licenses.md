# 04. 権利関係（レポート必須欄 4）

**必須欄でありながら、8/15 まで一度も調べていなかった欄である。**
失格条項（予選 PDF p23）が掛かるので、値は 1 次資料からしか取らない。

## 値

| 欄 | 値 | 出典 |
|---|---|---|
| **ベースモデルのライセンス** | **Gemma Terms of Use**（HF のカードで `license: gemma`） | HF `lerobot/pi05_libero_base` rev `a217bfd3…` のカード metadata |
| **学習データのライセンス** | **MIT License** | HF `Sylvest/LIBERO-plus`（原典・作者本人の配布）のカードに `license: mit` |
| **第三者コードのライセンス** | **LeRobot v0.6.0 — Apache License 2.0** | 同梱ソースのファイル冒頭のヘッダ（487 ファイル中 **477** に保持） |

## 【8/15 訂正】学習データは MIT である —— 見る場所が 3 つに分かれていた

**「ライセンス未記載」は誤りだった。**同じ LIBERO-plus でも、配布物が 3 つあり、
**表記があるのは原典の HuggingFace だけ**である。

| どこ | ライセンス |
|---|---|
| **`Sylvest/LIBERO-plus`（HF・作者本人の配布）** | **`license: mit`。これが原典** |
| `sylvestf/LIBERO-plus`（GitHub・コード） | LICENSE ファイルが無い |
| `lerobot/libero_plus`（HF・我々が学習に使ったもの） | カードに `license` 欄が無い（再配布時に書き写されていない） |

**運営の `THIRD_PARTY_LICENSES.md` は GitHub 側だけを見ている。**そのため
「未記載」と書かれているが、**原典の HF には MIT が明示されている。**

```bash
.venv/bin/python3 -c "
from huggingface_hub import HfApi
print(HfApi().repo_info('Sylvest/LIBERO-plus', repo_type='dataset').card_data.get('license'))"
```

**教訓: ライセンスは「使った配布物」だけでなく「原典」まで遡って確認する。**
再配布された形にはライセンスが写されていないことがある。

## 運営の記載（GitHub 側を見たもの）

`THIRD_PARTY_LICENSES.md`（配布リポジトリ直下・`init` コミット・2026-07-30・運営作成）に
そのまま書いてある。**推測ではない。**

| ソフトウェア | 取得元 | ライセンス |
|---|---|---|
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | **MIT**（Copyright (c) 2023 Lifelong Robot Learning） |
| **LIBERO-plus** | https://github.com/sylvestf/LIBERO-plus | **ライセンス未記載** |

> 「明示的なライセンスが付与されていない著作物は、既定では著作権者に全権利が
> 留保される点に留意すること。（中略）なお LIBERO-plus は LIBERO（MIT）を基盤とする
> 派生物であり、LIBERO 由来部分には LIBERO の MIT License が及ぶ。」

**我々が学習に使ったのは `lerobot/libero_plus`**（LIBERO-plus を LeRobot 形式に
直したもの）で、**こちらのカードにもライセンス欄が無い。**上流と同じ状態である。

**レポートにはこの状態をそのまま書く。**「MIT」と書いてはいけない —— LIBERO 由来の
部分に MIT が及ぶだけで、LIBERO-plus 全体が MIT なのではない。

## 提出物に入っている第三者コード

**zip の 541 エントリのうち 527 が `vendor/`＝同梱した LeRobot である。**

| 中身 | 件数 | ライセンス |
|---|---|---|
| `vendor/lerobot/`（LeRobot v0.6.0 のソース） | 527 | **Apache-2.0** |
| `model_weights/`（重みと paligemma のトークナイザ） | 12 | 重みは Gemma Terms of Use |
| `policy_server.py` / `requirements.txt` | 2 | 自作 |

**`requirements.txt` が入れるもの**（採点環境で `pip install` される）:
safetensors / draccus / transformers / num2words / accelerate / gymnasium /
torchvision / typing_extensions / fastapi / uvicorn / msgpack / numpy。
**採点イメージに入っているものは意図的に書いていない**（torch など）。

## 見つけた不備 —— **zip に LICENSE ファイルが入っていない**

```bash
# 確かめ方
python3 -c "
import zipfile
z = zipfile.ZipFile('<提出 zip>')
print([n for n in z.namelist() if 'licen' in n.lower()] or 'LICENSE は 1 つも無い')"
```

**Apache-2.0 は再配布時にライセンス本文の同梱を求める（第 4 条）。**
提出物にはファイル単位の著作権表示とライセンスヘッダは残っている（477/487）が、
**`LICENSE` ファイルそのものが入っていない。**

| | |
|---|---|
| いま直せるか | **直せない。**予選最終提出の締切（8/14 23:59）は過ぎている |
| レポートにどう書くか | **入っている状態をそのまま書く。**「ヘッダは保持、LICENSE ファイルは非同梱」 |
| 次に活かすこと | **提出物を組む工程に「LICENSE を入れる」を組み込む**（成果物 3 の項目） |

## 確かめ直すコマンド

```bash
# ベースモデルのライセンス
.venv/bin/python3 -c "
from huggingface_hub import HfApi
i = HfApi().repo_info('lerobot/pi05_libero_base',
                      revision='a217bfd3b14673cf2ce597e69997ab21866438dd')
print(i.card_data.get('license'))"

# 学習データにライセンス欄があるか
.venv/bin/python3 -c "
from huggingface_hub import HfApi
i = HfApi().repo_info('lerobot/libero_plus', repo_type='dataset',
                      revision='f3f49f426d75030177b18778374005bc12ccd588')
print('license:', i.card_data.get('license') if i.card_data else None)"

# 運営の記載
cat THIRD_PARTY_LICENSES.md

# 同梱コードのヘッダ保持率
grep -rl "Apache License, Version 2.0" work/artifacts/vendor/lerobot --include="*.py" | wc -l
find work/artifacts/vendor -name "*.py" | wc -l
```

## 人間側 / Claude 側

**人間側**: **権利関係は「最後に埋める欄」ではなく「最初に確認する欄」である。**
今回は必須欄だと気づいたのが締切の 2 日前だった。使う重みとデータを決めた時点で、
`license` を控えておけば 5 分で済む。

**Claude 側**: 配布物に `THIRD_PARTY_LICENSES.md` のような運営作成のファイルが
無いかを、**リポジトリ直下を最初に `ls` して確かめる。**今回はこれが答えを
持っていたのに、締切間際まで開いていなかった。
