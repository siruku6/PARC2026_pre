# 03. レポートに書く値

**予選ルール PDF p.14 の必須項目を、そのまま埋められる形で集めた。**
未提出だとランキング上位でも本選に行けない（PDF p.23）。

**学習を回した時点で記録しないと後から復元できない項目がある。**
だから提出のたびにここを更新する。
`?` は**まだ埋まっていない**もので、[open_items.md](../talk/open_items.md) の 16 に対応する。

対象は **4 回目の提出（版 1.0.0、8/11）** である。

---

## モデル情報

| 項目 | 値 | 出典・確認方法 |
|---|---|---|
| モデル名・バージョン | `policy_server 1.0.0` | [policy_server.py](../../solution/policy_server.py) の `__version__` |
| ベースモデル名 | `lerobot/smolvla_libero_plus` | 配布ノートブックの `BASE_MODEL_REPO` |
| ベース重みの入手元 | HuggingFace Hub | 同上 |
| 使用した重みの revision | `7bb70aa5bc92b82c9239142775d3a173103567ff` | 配布ノートブックの `BASE_MODEL_REVISION` |
| **提出チェックポイントのハッシュ** | `2cd9faabb421ce22adadd7a85760afa46047f572293f979a5092f823fae18604` | 下のコマンド |
| 提出 zip のハッシュ | `420aff5dfa955c2f40dabd3d3b07aee5ab6088811266317483bd77a53da57ada` | `work/submission.sha256` |
| モデル構成 | SmolVLA（SmolVLM2-500M-Video-Instruct の前半 16 層 ＋ action expert） | [01_base_model.md](01_base_model.md) |
| 推論方式 | HTTP サーバー（`/health` `/reset` `/act`）。flow matching で行動列を生成 | [policy_server.py](../../solution/policy_server.py) |
| **action chunk の有無と chunk 長** | **有り。`chunk_size=50`** | `config.json` |
| 1 回に実行するステップ数 | **`n_action_steps=5`**（`chunk_size` の 1/10 だけ実行して作り直す） | `policy_server.py` が起動時に上書き |
| 推論時の後処理 | **EMA `alpha=0.3`**（並進・回転の 6 次元のみ。グリッパは素通し） | 同上 |

ハッシュの取り方。

```bash
shasum -a 256 work/artifacts/model_weights/model.safetensors
cat work/submission.sha256
```

> **`config.json` の `n_action_steps` は 50 のままである。**
> `policy_server.py` が起動時に 5 で上書きするため、**実際に効いているのは 5** である。
> レポートには実効値を書く。

## 学習情報

| 項目 | 値 | 出典・確認方法 |
|---|---|---|
| 学習方法 | **LoRA**（`peft.method_type=LORA`） | [train.py](../../solution/train/train.py) |
| Full FT・LoRA・adapter の区分 | **LoRA** | 同上 |
| 学習対象パラメータ | `lm_expert` の attention 射影（`q_proj` / `v_proj`） | **実測。** 素の重みと比べて変化した 37 テンソルのうち 32 個が該当 |
| **ステップ数** | **20,000** | `--steps 20000`（素の重みと同数） |
| LoRA の階数 / スケール | `r=16` / `alpha=16` | [train.py](../../solution/train/train.py) の既定 |
| 学習率 | `3e-4` → `3e-5`（コサイン減衰、warmup 100 ステップ） | 同上 |
| バッチサイズ | **8** | `--batch-size 8`（**素の重みは 32**） |
| 使用データ | `lerobot/libero_plus`（LIBERO-plus の成功軌道） | 配布ノートブックの `DATASET_REPO` |
| データの revision | `f3f49f426d75030177b18778374005bc12ccd588` | 同上 |
| タスク数 | **40**（`libero_plus` の全タスク） | `--episodes-per-task 5` で選んだ |
| エピソード数 | **200**（40 × 5） | `episodes.json` に実際の一覧がある |
| 画像拡張 | **有効** | 当時の既定が有効だった |
| 独自収集・生成データ | **無し。** 配布データセットのみ | — |

実行したコマンドは 1 行である。**残りは当時の `train.py` の既定値**である。

```bash
python /workspace/train.py --steps 20000 --batch-size 8
```

> **8/11 に `train.py` の既定値を入れ替えた。上の表は 4 回目の提出の記録なので、
> いまの既定とは違う。** 旗も `--no-image-transforms` から
> `--image-transforms`（既定は無効）へ変わった。
> **最終提出で重みを差し替えたら、この表を作り直すこと。**
> 新しい既定は [../howto/01_retrain.md](../howto/01_retrain.md) の表にある。

> **タスク数 40 とエピソード数 200 は導いた値である。** `lerobot/libero_plus`
> のタスク数が 40 で（[01_base_model.md](01_base_model.md)）、
> `--episodes-per-task` の既定が 5 だからである。
> **提出前に実物で確かめること。**
>
> ```bash
> head -20 work/train/smolvla_lora_all_suites/episodes.json
> ```

> **`merged_model/` に `train_config.json` が入っていない。**
> 素の重みには付いてくる（`model_weights.bak/train_config.json`）が、
> マージの出力には含まれない。**学習条件は上の表と GPU マシンの記録が唯一の情報源**である。

## 権利関係

| 項目 | 値 | 出典 |
|---|---|---|
| ベースモデルのライセンス | **Apache-2.0** | `lerobot/smolvla_libero_plus` の HuggingFace ページ |
| 学習データのライセンス | **Apache-2.0**（`lerobot/libero_plus`） | 同 HuggingFace ページ |
| **提出物に含まれる第三者コードのライセンス** | **LeRobot v0.6.0（Apache-2.0）を `vendor/lerobot/` に同梱** | [THIRD_PARTY_LICENSES.md](../../THIRD_PARTY_LICENSES.md) |

**第三者コードの同梱は必ず該当する。** [02_環境戦略.md](../strategy/02_環境戦略.md) の
方式 A（LeRobot のソースを提出物へ同梱）を採っているためで、
**運営 Q&A No.36 で明示的に許可されている。** ライセンス表記は提出物に含めてある。

## リーダーボードの履歴

| 版 | 重み | 推論設定 | zip の sha256 | スコア |
|---|---|---|---|---|
| 0.1.0 | 素 | n=50 | `f80abc7b…` | 0.05832 |
| 0.2.0 | 素 | n=10 | `8ed91671…` | 0.096 |
| **0.3.0** | **素** | **n=5 ＋ EMA 0.3** | `c1685507…` | **0.1409 ← 最良** |
| 0.3.0（版を上げる前） | LoRA 20,000 ステップ | 0.3.0 と同じ | `80e8711e…` | **0.016** |
| 1.0.0 | 同じ重み ＋ 版の行 | 同上 | `420aff5d…` | 未提出 |

**推論の設定を揃えて重みだけを変えたので、0.1409 → 0.016 は学習の効果である。**
**8.8 倍の悪化。** 詳細と原因の見立ては [05_実測結果.md](../strategy/05_実測結果.md)。

> **レポートに書くのは、最終提出に使う構成である。**
> いまの最良は **0.3.0（素の重み）** なので、**このままでは
> 「独自学習を行った」という要件（Q&A No.36）との整合を説明する必要がある。**
> 4 回目で学習した重みを提出した事実は残るが、
> **最終提出の構成をどうするかは別に決めること**（[open_items.md](../talk/open_items.md) の 17・18）。

## 更新の手順

提出のたびにこの 3 つを更新する。

1. **`__version__` を上げる**（重みが変わったら major、推論の挙動なら minor）
2. **ハッシュを取り直す**（上のコマンド 2 本）
3. **リーダーボードの表に行を足す**（スコアが出たら埋める）
