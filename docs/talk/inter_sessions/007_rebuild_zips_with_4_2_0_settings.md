# 007 1,500 歩と 1,000 歩の zip を、4.2.0 と同じ推論設定で作り直す

> **状態: 依頼中** ／ 受け側: 未定 ／ 更新: 2026-08-28

## 背景（先に読んでほしい）

**このコンペで何をしているか。** PARC2026 の本選 1 である。ロボット（Franka Emika Panda）に
自然言語の指示を与え、LIBERO-plus のシミュレータの中で物を掴んで運ばせる。
こちらの方策は **π0.5**（4,143,404,816 パラメータ）。
`solution/policy_server.py` が画像 2 枚と状態を受け取り、行動列を返す。
**提出は 1 日 1 トラック 1 回だけ。**

| トラック | **こちらの点** | 最高点 |
|---|---|---|
| Track1 | **0.370**（版 4.2.0） | 0.65 |
| Track2 | 0.207（版 4.3.0）／0.154（版 4.2.0） | 0.53 |
| Track3 | 0.000 | 0.000（全参加者） |

### 何のために作り直すのか

**8/28 に RunPod で 1,500 歩のフルパラメータ学習を回した。**`rp3000` から続けて
batch 64・lr 5e-6・カメラ拡張あり。**8/17 以来はじめて画像側を開けた学習**である
（`num_learnable_params=4143404816` をログで確認済み）。**loss は単調に下がった**——
500 歩 0.2532 → 1,000 歩 0.2463 → 1,500 歩 **0.2407**（−4.9%）。

**その重みを採点側で測りたい。比べる相手は Track1 の 0.370（版 4.2.0）である。**
**だから推論の設定を 4.2.0 と 1 ビット違わず揃える。**そうしないと
「重みが効いたのか設定が効いたのか」が読めない。

### すでに作った 3 本には、揃っていない点が 1 つある

**HF `siruku6/parc2026-submission` に 3 本ある。**中身は zip の索引を読んで特定した。

| zip | 中身 | `_MIN_LEXICAL` |
|---|---|---|
| `submission_5.0.0.zip` | 001500 | **0.65** |
| `submission_6.0.0.zip` | 001000 | **0.65** |
| `submission_6.0.1_500steps.zip` | 000500 | **0.65** |

**3 本の `policy_server.py` は同一**（CRC `cef8f7e2`）で、`__version__` は `"4.6.0"`。
設定はこうだった。

| 項目 | **4.2.0** | **作った 3 本** | |
|---|---|---|---|
| `num_inference_steps` | 10 | 10 | 一致 |
| `n_action_steps` | 16 | 16 | 一致 |
| `blend_steps` | 0 | 0 | 一致 |
| `ema_alpha` | 0.3 | 0.3 | 一致 |
| `samples` | 1 | 1 | 一致 |
| 倍率（並進／回転） | 1.0 / 1.0 | 1.0 / 1.0 | 一致 |
| RTC | off | off | 一致 |
| **`_MIN_LEXICAL`（指示文の正規化のしきい値）** | **0.50** | **0.65** | **違う** |
| 種 | off | 0（固定） | **違うが、これは揃えない**（下） |

### しきい値の差は「効かない」ではなかった

**私（依頼した側）は最初「Track1 の 4 文と Track2 の 8 文で出力が完全一致するので
実質何もしない」と書いた。それは配布された example しか見ていない結論だった。**

**LIBERO-plus の全 bddl から指示文 2,124 種を集めて測り直すと、
503 種（24%）で 0.50 と 0.65 の出力が違う。**

```
元   : Turn on the stove and position the moka pot on it just like we do every morning
0.50 : turn on the stove and put the moka pot on it        ← 学習で見た文へ寄る
0.65 : turn on the stove and position the moka pot on it just like we do every morning
```

**0.50 の方が正しい向きである。**一致度 0.50〜0.60 台の言い換えが、
0.65 では寄せられずに冗長なまま渡っている。

**カテゴリ別に見ると、危ないところが 1 つに絞れる。**

| カテゴリ | 出力が違う | 全体 |
|---|---|---|
| Background Textures | **0** | 2,152 |
| Light Conditions | **0** | 2,284 |
| Objects Layout | **0** | 3,050 |
| **言い換え（Language Instructions）** | **503** | — |

**つまり「採点される集合に言い換えが含まれるか」で全部が決まる。**
配布された example には入っていないが、**採点されるのは非公開のタスクである。**
**「example に無い」を「出ない」と読み替えてはいけない**——手元の 36 行で 0.713 が
出たのに本番が 0.141 だった前例がある。

**種（`PARC_SEED`）は 0 のまま揃えない。** コードのコメントにある理由のとおり、
**種は分布を変えずに 1 つの引きを選ぶだけなので期待値を動かさない。**
固定してある方が、同じ zip を出し直したときに同じ点が返るので比較に有利である。

### しきい値 0.65 は消さない。別の 1 変数として残す

**0.65 へ上げたのは 8/27 のバグ修正である。**0.50 だと Track3 の 1 文が
意味の逆に書き換わっていた（「皿から下ろす」が「皿に載せる」になる）。

**だから 0.65 は捨てない。**リポジトリの `solution/policy_server.py` は
**0.65 のまま残し、今回の zip の中だけを 0.50 にする。**
**Track3 のバグ修正は、重みの変更とは別の 1 変数として後日測る。**

---

## 何をしてほしいか

**`001500` と `001000` の 2 本を、`_MIN_LEXICAL = 0.50` で zip にして
HF へ上げてほしい。**

| 版 | 重み | しきい値 | ファイル名 |
|---|---|---|---|
| **5.1.0** | **001500** | **0.50** | `submission_5.1.0.zip` |
| **6.1.0** | **001000** | **0.50** | `submission_6.1.0.zip` |

**5.x が 1,500 歩・6.x が 1,000 歩という既存の対応を保ち、マイナーを上げて
設定の変更を示す。**
**古い `5.0.0` / `6.0.0` / `6.0.1_500steps` は消さないこと**——
台帳（[../../reference/19_weights_catalog.md](../../reference/19_weights_catalog.md)）に
sha256 で記録済みなので、上書きすると対応が崩れる。

成果物は **(1) HF に上がった 2 本と sha256**、**(2) 下の「報告の書き方」の表**である。

---

## いま止まっているところ

**会社の RTX6000（`~/src/PARC2026_pre`）で、重みの取り込みが落ちた。**

```
$ python solution/tools/weights/install_weights.py work/artifacts/trial1500/001500
[install] 取り込み元 .../work/artifacts/trial1500/001500
[install] 取り込み先 .../work/artifacts/model_weights
[install] 引き継ぐ   vlm/
[install] テンソル構成は既存と一致しています
FileNotFoundError: 引き継ぐディレクトリがありません:
  .../work/artifacts/model_weights/vlm（素の重みを一度配置してから実行してください）
```

**原因は分かっている。**`install_weights.py:51` の
`_INHERITED_DIRS = ("vlm",)` により、**`vlm/` は取り込み元から写さず、
取り込み先にあるものを引き継ぐ**作りになっている（`_build_staging`）。
**取り込み先の `work/artifacts/model_weights/vlm` が無い。**

**`テンソル構成は既存と一致しています` が先に出ているので、
`model_weights/` 自体はあり、π0.5 の重みが入っている。無いのは `vlm/` だけである。**
**学習のチェックポイントには `vlm/` が付かない**ので、そこから写して作ったのだと思われる。

### 直し方 —— 落としてきた側に `vlm/` がある

**`hf download` した `001500` には `vlm/` が入っている**（HF のファイル一覧で確認済み。
`added_tokens.json`・`special_tokens_map.json`・`tokenizer.json`・`tokenizer_config.json`
の 4 つ・**合わせて 17 MB**）。**それを取り込み先へ置けば通る。**

```bash
ls work/artifacts/model_weights/                      # vlm が無いことを確かめる
ls work/artifacts/trial1500/001500/vlm/               # 4 ファイルあることを確かめる
cp -r work/artifacts/trial1500/001500/vlm work/artifacts/model_weights/vlm
```

**`cp` にすること。`ln -s` では駄目である**（提出物に実体が入らない）。

### howto/02 のケース 0 には従わないこと（罠）

**[../../howto/02_build_submission.md](../../howto/02_build_submission.md) の
「0-3. 重みを置く」は SmolVLA 時代のままである。**
あの手順は `vlm/` に **`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`** の
トークナイザを入れる。**π0.5 のトークナイザは PaliGemma / Gemma 系なので、
あれを入れると別物になる。**

**気づいたことを 1 行、報告に書いてほしい**——あの節は π0.5 向けに直す必要がある。
**ただし直すのはこの依頼の範囲外。**

---

## やること

1. **`git pull`。** `solution/tools/build/patch_zip_policy_server.py` が入っている
   （8/28 に足した。下の 4 で使う）。

2. **`vlm/` を置いて、重みを取り込む。**

   ```bash
   cd ~/src/PARC2026_pre && git pull
   cp -r work/artifacts/trial1500/001500/vlm work/artifacts/model_weights/vlm

   python solution/tools/weights/install_weights.py \
       work/artifacts/trial1500/001500 --dry-run
   python solution/tools/weights/install_weights.py \
       work/artifacts/trial1500/001500
   ```

3. **zip を組む。**

   ```bash
   work/venv310/bin/python solution/tools/build/build_submission.py
   ```

   **`[build] 検証` の行が出ることを必ず確かめる。**出ないビルドは使わない。

4. **zip の中の 1 行だけを書き換える。**
   `policy_server.py` は zip に**無圧縮で入っている**ので、
   **9.4 GB を組み直さずにバイトを差し替えて CRC を直せる。**

   ```bash
   python3 solution/tools/build/patch_zip_policy_server.py work/submission.zip \
       --replace '_MIN_LEXICAL = 0.65' '_MIN_LEXICAL = 0.50' \
       --replace '__version__ = "4.6.0"' '__version__ = "5.1.0"' \
       --in-place
   ```

   **道具は書き換え前と後の設定を両方印字し、`zipfile` で読み直して
   `testzip()` まで通す。**通らなければ書かない。
   **`policy_server.py` が無圧縮でなければその場で止まる**——
   そのときは 2 に戻り、**リポジトリの `policy_server.py` を直接
   `_MIN_LEXICAL = 0.50` に書き換えて組み直し、そのあと `git checkout` で戻す。**

5. **名前を付けて sha256 を残し、HF へ上げる。**

   ```bash
   mv work/submission.zip work/submission_5.1.0.zip
   sha256sum work/submission_5.1.0.zip | tee work/submission_5.1.0.sha256
   hf upload siruku6/parc2026-submission \
       work/submission_5.1.0.zip submission_5.1.0.zip
   ```

6. **`001000` でも同じことをする。** 先に落とす（**約 17 分**）。

   ```bash
   hf download siruku6/pi05_trial1500 --include '001000/*' \
       --local-dir work/artifacts/trial1500
   python solution/tools/weights/install_weights.py \
       work/artifacts/trial1500/001000
   work/venv310/bin/python solution/tools/build/build_submission.py
   python3 solution/tools/build/patch_zip_policy_server.py work/submission.zip \
       --replace '_MIN_LEXICAL = 0.65' '_MIN_LEXICAL = 0.50' \
       --replace '__version__ = "4.6.0"' '__version__ = "6.1.0"' \
       --in-place
   mv work/submission.zip work/submission_6.1.0.zip
   sha256sum work/submission_6.1.0.zip | tee work/submission_6.1.0.sha256
   hf upload siruku6/parc2026-submission \
       work/submission_6.1.0.zip submission_6.1.0.zip
   ```

   **`install_weights.py` は `.bak` を上書きする。**2 回続けて走らせると
   1 回目の `.bak` が消える。**元の `rp3000` に戻すときは HF から取り直すこと**
   （`siruku6/pi05_full_runpod`）。

7. **リポジトリを汚していないことを確かめる。**

   ```bash
   git status --short
   grep -n '^_MIN_LEXICAL = ' solution/policy_server.py    # 0.65 のままであること
   ```

---

## できあがりの条件

- HF に `submission_5.1.0.zip` と `submission_6.1.0.zip` がある
- **どちらも中の `_MIN_LEXICAL` が `0.50`、`__version__` が `5.1.0` / `6.1.0`**
  （`patch_zip_policy_server.py --show` で確かめられる）
- sha256 が手元のファイルに残っている
- **リポジトリの `solution/policy_server.py` は `0.65` のまま**
- 古い 3 本が HF に残っている

---

## 報告の書き方

**表と数値だけでよい。**

1. **2 本の zip の表**——版・重み・バイト数・sha256・`--show` で出た設定。
2. **`vlm/` の件**——何を置いて通ったか。
3. **howto/02 のケース 0 が π0.5 向けでないこと**を 1 行。
4. **想定と違ったところ。**とくに `patch_zip_policy_server.py` が使えたかどうか
   （無圧縮だったか）。

---

## やらなくてよいこと

- **提出。**ユーザがブラウザで出す。
- **`policy_server.py` の恒久的な変更。**しきい値 0.65 はリポジトリに残す。
  **Track3 のバグ修正は別の 1 変数として後日測る。**
- **`000500` の zip。**いまは 2 本でよい。
- **評価。**手元の評価は別セッションが RTX6000 で回す。
- **howto/02 のケース 0 の書き直し。**報告に 1 行書くだけでよい。
- **`NOW.md` の書き換え。** 書き換えるのはユーザである。

---

## 判断が必要になったら

**この依頼の目的は「1,500 歩の重みを、4.2.0 の 0.370 と 1 変数で比べられる形にする」ことである。**

**その目的に照らして、より確実なやり方を思いついたら変えてよい。**
たとえば `patch_zip_policy_server.py` が使えなければ、
**リポジトリを一時的に書き換えて組み、`git checkout` で戻す**方が確実である。
**変えた場合は理由を報告に書くこと。**

**次の 2 つは動かさないでほしい。**

| 動かさないもの | なぜ |
|---|---|
| **`_MIN_LEXICAL` 以外の設定** | 1 変数にするための依頼である。他を触ると目的が消える |
| **古い 5.0.0 / 6.0.0 / 6.0.1 を消さないこと** | 台帳が sha256 で対応を記録している |

---

## 報告

（受け側がここに書く）
