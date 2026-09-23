# 24. 公開している 18 件のうち、条件に当たるのは **MolmoAct2 の 1 件だけ**である

**2026-09-17 に、[talk/inter_sessions/035](../talk/inter_sessions/035_license_provenance_check.md)
の依頼で 3 つの配布元を読んで調べた。**

> **これが分かると何が決まるか。****非公開へ戻さなければならないものは 1 件も無い。**
> ただし `siruku6/molmoact2_lora_v1_1` は、**ライセンスの記載が無いリポジトリから
> 取った 6 ファイル**を同梱したまま公開されており、**そのモデルカード自体が上がっていない**。
> ここだけは手を入れる。

| 節 | 中身 |
|---|---|
| [1 結論](#1-結論) | 18 件の判定 |
| [2 3 つの配布元に何と書いてあるか](#2-3-つの配布元に何と書いてあるか) | 原文の引用と出どころ |
| [3 依頼の前提が 2 つ違っていた](#3-依頼の前提が-2-つ違っていた) | 土台は Apache 2.0 ではない・公開は 6 件ではない |
| [4 手を入れるもの](#4-手を入れるもの) | 3 つ。どれも公開を止めずに済む |
| [5 気づいたが実行しなかったこと](#5-気づいたが実行しなかったこと) | |

---

## 1. 結論

**法律の判断はしていない。**「どこに何と書いてあるか」と、それに照らした形式上の過不足だけである。

| 公開しているもの | 判定 | 根拠 |
|---|---|---|
| `siruku6/libero_plus_bowl5`（データ・14,085 本） | **条件付きで可** | 中身は LIBERO 由来のデモ（**CC BY 4.0**——表示すれば再配布できる）と LIBERO-plus 由来（作者自身が HF で **MIT** と書いている）と自作分。**再配布は認められているが、カードが `license: mit` だけで、CC BY 4.0 が要求する出典表示が無い** |
| `siruku6/libero_plus_bowl5_initpose`（データ・16,820 本） | **条件付きで可** | 同上 |
| `siruku6/libero_combined_bowl5_initpose`（データ・21,642 本） | **条件付きで可** | 同上。足した `libero_combined_20hz` の中身も LIBERO の標準スイートと `libero_90` で、出どころは同じ CC BY 4.0 である（本選ルール PDF p.20 が、配布データの中身を `IPEC-COMMUNITY/`・`Sylvest/`・`lerobot/` の公開データセットだと書いている） |
| π0.5 の重み 11 件（`pi05_*`） | **このままでよい** | 土台 `lerobot/pi05_libero_base` は **Apache 2.0 ではなく Gemma 利用規約**である。11 件とも `license: gemma` を宣言し、規約 3.1 が要求する一文（`Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms`）をカードに載せている。**`NOTICE` ファイルが無いものが 4 件**あるが、同じ一文はカードにある |
| `siruku6/smolvla-libero-plus-finetunes`（重み） | **このままでよい** | 土台 `lerobot/smolvla_libero_plus` が Apache 2.0、宣言も `apache-2.0` で一致 |
| **`siruku6/molmoact2_lora_v1_1`（重み）** | **条件付きで可——ここだけ手を入れる** | 2 つある。①**モデルカードが上がっていない**（リポジトリに `README.md` が無く、ライセンス欄が空である）。②同梱している `molmoact2_base_small/` の 6 ファイルは `allenai/MolmoAct2-LIBERO`（revision `0d24a92`）から取ったもので、**このリポジトリにはライセンスの記載が無い**。土台の重み `allenai/MolmoAct2-LIBERO-LeRobot` が Apache 2.0 なのとは別のリポジトリである |
| `siruku6/jaist_mix_v2`（データ） | **このままでよい** | 中身は自作の逆向きデモである |
| `siruku6/llm-jp-3-13b-it_lora2`（重み・2024 年） | **今回の範囲外** | 本選と無関係の古い公開物 |

**非公開のまま**：`libero_combined_bowl5_initpose_l90clean`・`pi05_bowl5_full`・`pi05_code`・
`lerobot_init_pose_all`。**戻す必要のあるものは無い。**

---

## 2. 3 つの配布元に何と書いてあるか

| 出どころ | ライセンス | 原文 | 場所 |
|---|---|---|---|
| **LIBERO**（本体） | コード **MIT**／**データセット CC BY 4.0** | `MIT License / Copyright (c) 2023 Lifelong Robot Learning`（LICENSE）。README の License 表に `\| Codebase \| MIT License \|` と `\| Datasets \| Creative Commons Attribution 4.0 International (CC BY 4.0) \|` | `https://github.com/Lifelong-Robot-Learning/LIBERO` の `LICENSE` と `README.md`（9/17 に取得） |
| **LIBERO-plus** | **GitHub は無記載／作者の HF は MIT** | GitHub の clone（commit `4976dc3`）に `LICENSE` ファイルは無く、README にもライセンスの記載は無い。一方、作者自身の配布物 `Sylvest/LIBERO-plus` と `Sylvest/libero_plus_lerobot` のカードには `license: mit` とある | 手元の clone `work/LIBERO-plus`／`https://huggingface.co/datasets/Sylvest/LIBERO-plus/raw/main/README.md`（9/17） |
| **`lerobot/libero_plus`**（実際に学習に使った側） | **無記載** | カード（`README.md`）そのものが無く（HTTP 404）、`cardData.license` も空である。**素性は Sylvest 版と同じデータなので、上流の MIT を頼りにする形になる** | `https://huggingface.co/api/datasets/lerobot/libero_plus`（9/17） |
| **本選ルール PDF v1.1** | **外部公開についての記載は無い** | 権利に触れるのは 3 か所だけである。p.22 レポート記載事項に「**権利関係／ベースモデルのライセンス／学習データのライセンス／提出物に含まれる第三者コードのライセンス**」。p.27 データの使用ルールに「**使用するデータについて，出所，ライセンス，生成方法および使用目的を説明できる必要があります**」「**他参加者の非公開データ，利用権限のないデータ，法令・ライセンスに違反するデータは利用できません**」。p.32 の Q&A に「**LIBERO / LIBERO-plus のデータを学習に使用 → 問題ありません**」。**参加者が作った重みやデータセットを外部へ公開してよいかは、禁止事項（p.28〜29）にも注意事項（p.30〜31）にも書かれていない** | `docs/materials/PARC2026開発コンペティション_本選_v1.1.pdf` |
| **Gemma 利用規約**（π0.5 の土台） | Gemma Terms of Use | 3.1 が再配布側に求めるのは、①3.2 の使用制限を契約条項として含めること ②規約の写しを受領者に渡すこと ③変更したファイルにその旨を目立つ形で示すこと ④「Notice」テキストファイルに `Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms` を入れること | `https://ai.google.dev/gemma/terms` の 3.1（9/17） |

**PDF の配布リポジトリは招待制である**（p.16〜17 に GitHub アカウント登録フォーム経由の招待手順）。
**招待制のリポジトリのコードそのものを公開することは想定されていない**と読めるが、
**禁止とは書かれていない。**いま公開しているものにコードは含まれていない（`siruku6/pi05_code` は非公開）。

---

## 3. 依頼の前提が 2 つ違っていた

| 035 の記述 | 実際 | どう影響するか |
|---|---|---|
| π0.5 の土台のライセンスは調べる対象に入っていなかった（表には MolmoAct2 の Apache 2.0 だけがある） | **`lerobot/pi05_libero_base` は `license: gemma`** である。Apache 2.0 のような「表示すれば自由」ではなく、**使用制限を下流へ引き継がせる**規約である | **既に 11 件とも `license: gemma` を宣言して一文も載せてある**ので、結果として問題は無い。ただし**規約が求めているのは「Notice テキストファイル」で、4 件はカード本文にしか無い** |
| 公開済みは 6 件（データ 3・π0.5 の重み 14・MolmoAct2 1） | **公開は models 14 件・datasets 4 件の計 18 件**である。models 14 件のうち π0.5 は 11 件で、残りは `smolvla-libero-plus-finetunes`・`llm-jp-3-13b-it_lora2`（2024 年・本選と無関係）・`molmoact2_lora_v1_1` である。datasets には `jaist_mix_v2`（自作の逆向きデモ）も入っている | 判定の対象が 3 件増えたが、**増えた 3 件はどれも「このままでよい」である** |

**035 の 4.2 が「`lerobot/libero_plus` は記載なし」と書いたのは正しい。**
**足りなかったのは、上流の作者本人（`Sylvest/*`）が MIT で出している**という 1 段先である。

---

## 4. 手を入れるもの

**どれも公開を止めずに済む。**

1. **`siruku6/molmoact2_lora_v1_1` のモデルカードを上げる。**
   原稿は `work/train/molmoact2_lora_v1_1/publish/`（`README.md`・`LICENSE`・`NOTICE`）に
   出来ているが、**HF 側のリポジトリには上がっていない**（ファイル一覧に `README.md` が無い）。
2. **同梱している `molmoact2_base_small/` の 6 ファイルの扱いを決める。**
   出どころ `allenai/MolmoAct2-LIBERO` に**ライセンスの記載が無い**。取れる道は 3 つある——
   ①同梱をやめて、使う側に `huggingface-cli download` させる（[howto/20](../howto/20_train_molmoact2_local.md) の 0-2 と同じ手順）
   ②そのまま置き、カードに「出どころは `allenai/MolmoAct2-LIBERO` rev `0d24a92`・**ライセンス無記載**」と書く
   ③Ai2 に問い合わせる。**①なら条件の話が消えるが、評価する機械での手数が 1 つ増える。**
3. **データ 3 件のカードに、LIBERO の表示を足す。**
   CC BY 4.0 は表示（出典・ライセンス・変更した旨）を求めている。いまのカードは
   `license: mit` だけで、**LIBERO への言及も、変更した旨も、書式としては入っていない**
   （本文には作り方が詳しく書いてあるので、足すのは 3 行である）。

---

## 5. 気づいたが実行しなかったこと

- **`lerobot/libero_combined_20hz` は HF から見えない**（API が 401 を返す・公開検索にも出ない）。
  配布された tar の中身は読めるが、**HF 側の配布条件は確かめようが無い。**上で CC BY 4.0 に
  寄せて判定したのは、PDF p.20 が中身を LIBERO 標準スイートと `libero_90` だと書いているからである。
- **このマシンには PDF を読む道具が入っていない**（`pypdf` は `pip install` が拒否され、
  `poppler-utils` も無い）。**代わりに `gs -sDEVICE=txtwrite` で本文を取り出した**——
  Ghostscript は入っている。035 の 5 節が書いた手順は、この機械では通らない。
- **リポジトリ直下の `THIRD_PARTY_LICENSES.md`（予選の自己評価キット向け）は、LIBERO-plus を
  「ライセンス未記載」とだけ書いている。**作者の HF が MIT だという 1 段先を足せるが、
  あれは予選キットの表記なので今回は触っていない。
