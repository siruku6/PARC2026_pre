# 007 1,500 歩の zip を、4.2.0 と同じ推論設定で作り直す

> **状態: 依頼中** ／ 受け側: 未定 ／ 更新: 2026-08-28

**2026-08-28 に範囲を狭めた。**もとは 1,500 歩（版 5.1.0）と 1,000 歩（版 6.1.0）の
2 本を作る依頼だった。**いまは 1,500 歩の 1 本だけでよい。**
**版番号も 5.1.0 ではなく `5.0.0` にして、HF にある古い `submission_5.0.0.zip` を
置き換える**（古い 5.0.0 はしきい値を揃えないまま作ってしまったものなので、残さない）。

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

**`001500` の 1 本を、`_MIN_LEXICAL = 0.50` で zip にして HF へ上げてほしい。**

| 版 | 重み | しきい値 | ファイル名 |
|---|---|---|---|
| **5.0.0** | **001500** | **0.50** | `submission_5.0.0.zip`（**HF の同名を置き換える**） |

**5.x が 1,500 歩という既存の対応を保つ。**版番号は上げない——
**古い 5.0.0 はしきい値を揃え忘れて作ったものなので、番号ごと作り直す。**

**HF の `submission_5.0.0.zip` は上書きしてよい。** ただし
**`submission_6.0.0.zip` と `submission_6.0.1_500steps.zip` は消さないこと。**

**上書きすると、8/28 に Track2 へ出した 5.0.0 の点に対応する zip が無くなる。**
それは承知のうえの依頼である（sha256 は台帳に残るので、対応そのものは辿れる）。
**台帳の 5.0.0 の行を、受け側で書き換えてほしい**（下の「やること」の最後）。

成果物は **(1) HF に上がった 1 本と sha256**、**(2) 下の「報告の書き方」の表**である。

---

## いま止まっているところ

**会社の RTX6000（`~/src/PARC2026_pre`）で 2 つ落ちた。両方とも直し方は分かっている。**

| 落ちたもの | 直し方 |
|---|---|
| `install_weights.py`——`model_weights/vlm` が無い | **取り込み自体をやめる。**`build_submission.py --weights work/artifacts/trial1500/001500` で 001500 を直に指す（`vlm/` は 001500 に同梱されている） |
| `build_submission.py`——`work/artifacts/vendor` が無い | **LeRobot v0.6.0 のソースを取って `vendor_lerobot.py` を走らせる**（13 MB・3 分。「やること」の手順 2） |

### 1 つ目 —— 重みの取り込みが落ちた

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

### 直し方（`model_weights/` を入れ替える道を採るときだけ）—— 落としてきた側に `vlm/` がある

**`hf download` した `001500` には `vlm/` が入っている**（HF のファイル一覧で確認済み。
`added_tokens.json`・`special_tokens_map.json`・`tokenizer.json`・`tokenizer_config.json`
の 4 つ・**合わせて 17 MB**）。**それを取り込み先へ置けば通る。**

```bash
ls work/artifacts/model_weights/                      # vlm が無いことを確かめる
ls work/artifacts/trial1500/001500/vlm/               # 4 ファイルあることを確かめる
cp -r work/artifacts/trial1500/001500/vlm work/artifacts/model_weights/vlm
```

**`cp` にすること。`ln -s` では駄目である**（提出物に実体が入らない）。

**`vlm/` は zip を組む側でも要る。**`build_submission.py:248` が
`model_weights/vlm/` で始まる項目が 1 つも無ければ検証で落ちる
（「トークナイザが無いと採点環境で Hub へ取りに行って失敗します」）。
**取り込みを飛ばしても、`vlm/` は避けて通れない。**

### 取り込みは要らない —— `--weights` で 001500 を直に指す（こちらを推す）

**`build_submission.py --weights <ディレクトリ>` は、渡したディレクトリを
そのまま zip の `model_weights/` にする**（`build_submission.py:168-169`。
zip の中の名前は `model_weights/` に固定される）。
**`001500` には `vlm/` が同梱されているので、これだけで組める。**

```bash
work/venv310/bin/python solution/tools/build/build_submission.py \
    --weights work/artifacts/trial1500/001500
```

**`install_weights.py` の 3 つの検査は、001500 についてはもう通っている。**
検査は `main()` の中で `_build_staging` より**前**に走る（`install_weights.py:217-234`）。

| 検査 | 何を見るか | 001500 の結果 |
|---|---|---|
| `_check_source` | 必須 4 ファイルがあること・`config.json` の `use_peft` が真でないこと | **通った**（落ちずに次へ進んでいる） |
| `_check_no_lora` | LoRA のテンソルが残っていないこと | **通った**（同上） |
| `_compare_keys` | テンソルの構成が既存と一致すること | **通った**（`テンソル構成は既存と一致しています` が印字された） |

**落ちたのは検査の後ろの `_build_staging` である。**上のエラーの出力で
3 つとも通っていることが読める。**だから `--weights` で組んでも失うものは無い。**

**加えて、9.4 GB の複製 2 つ**（`model_weights.new/` と `model_weights.bak/`）**を作らずに済む。**
`install_weights.py` は `work/artifacts/model_weights.bak/` を**上書きする**ので、
入れ替えると元の `rp3000` は HF から取り直すことになる（`siruku6/pi05_full_runpod`）。

**`model_weights/` を入れ替える必要はこの依頼には無い。**手元の評価（板の F38）は
`WEIGHTS=` で重みの置き場を直に渡すので、`model_weights/` が `rp3000` のままでも困らない。

### 2 つ目 —— zip を組む側で `vendor/` が無かった

**`--weights` で組もうとしたら、重みではなく同梱ソースで落ちた**（8/28）。

```
$ python solution/tools/build/build_submission.py \
    --weights work/artifacts/trial1500/001500
FileNotFoundError: 生成物がありません:
  /home/ubuntu/src/PARC2026_pre/work/artifacts/vendor（vendor_lerobot.py の実行や重みの配置が済んでいるか確認してください）
```

**`build_submission.py:55` の `_ARTIFACT_DIRS = ("model_weights", "vendor")` により、
`vendor/` は `--weights` では差し替えられない。**必ず `work/artifacts/vendor/` から取る。
**RTX6000 ではまだ一度も作っていない**（zip はこれまで別のマシンで組んでいた）。
**作り方は「やること」の手順 2 にある。13 MB・3 分。**

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

1. **`git pull` と、揃っているかの確認。**
   `solution/tools/build/patch_zip_policy_server.py` が入っている
   （8/28 に足した。手順 4 で使う）。

   ```bash
   cd ~/src/PARC2026_pre && git pull
   ls work/artifacts/trial1500/001500/vlm/     # 4 ファイルあることを確かめる
   ls -d work/artifacts/vendor/lerobot         # 無ければ手順 2 が要る
   ```

2. **`work/artifacts/vendor/lerobot` が無ければ作る**（あれば飛ばす）。
   **8/28 に RTX6000 で無いことが分かった**——`build_submission.py` は
   `model_weights` と `vendor` の 2 つを `work/artifacts/` から取るので
   （`build_submission.py:55` の `_ARTIFACT_DIRS`）、`vendor/` が無いと
   `生成物がありません: .../work/artifacts/vendor` で落ちる。

   **`vendor/` は LeRobot v0.6.0 のソースを Python 3.10 向けに書き換えたものである。**
   採点環境は 3.10 固定だが、LeRobot v0.6.0 は `requires-python >= 3.12` を宣言して
   いるので `pip install` できない。**13 MB・3 分で作れる。**

   ```bash
   python3 -c "
   import io, tarfile, urllib.request, pathlib, shutil
   url = 'https://codeload.github.com/huggingface/lerobot/tar.gz/refs/tags/v0.6.0'
   data = urllib.request.urlopen(url, timeout=300).read()
   dest = pathlib.Path('work'); target = dest / 'lerobot-src'
   if target.exists(): shutil.rmtree(target)
   with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
       root = tf.getnames()[0].split('/')[0]; tf.extractall(dest)
   (dest / root).rename(target)
   "

   python3 solution/tools/build/vendor_lerobot.py    # 3.10 があれば --python310 を付ける
   ```

   **`--python310 <3.10 の python>` を付けると、書き換えた結果が 3.10 で構文として
   通ることまで検証する。省くと検証だけを飛ばす**（`vendor_lerobot.py:159-164`）。
   **VGGT は要らない**——`vendor/vggt/` が zip に入っていなければ検証は VGGT を求めない
   （`build_submission.py:256`）。001500 に幾何の接ぎ木は入っていない。

3. **zip を組む。重みは 001500 を直に指す。**

   ```bash
   python3 solution/tools/build/build_submission.py \
       --weights work/artifacts/trial1500/001500
   ```

   **`[build] 検証` の行が出ることを必ず確かめる。**出ないビルドは使わない。
   （`work/venv310/bin/python` があるならそちらでよいが、
   `build_submission.py` は標準ライブラリだけで動くので `python3` で足りる。）
   **`model_weights/` は入れ替えない**（理由は上の「取り込みは要らない」）。
   入れ替える道を採るなら
   `cp -r work/artifacts/trial1500/001500/vlm work/artifacts/model_weights/vlm` の後に
   `python solution/tools/weights/install_weights.py work/artifacts/trial1500/001500`
   を `--dry-run` 付きで一度・付けずに一度走らせ、`--weights` を外して組む。

4. **zip の中の 2 行だけを書き換える。**
   `policy_server.py` は zip に**無圧縮で入っている**ので、
   **9.4 GB を組み直さずにバイトを差し替えて CRC を直せる。**

   ```bash
   python3 solution/tools/build/patch_zip_policy_server.py work/submission.zip \
       --replace '_MIN_LEXICAL = 0.65' '_MIN_LEXICAL = 0.50' \
       --replace '__version__ = "4.6.0"' '__version__ = "5.0.0"' \
       --in-place
   ```

   **道具は書き換え前と後の設定を両方印字し、`zipfile` で読み直して
   `testzip()` まで通す。**通らなければ書かない。
   **`policy_server.py` が無圧縮でなければその場で止まる**——
   そのときは zip を組む手順に戻り、**リポジトリの `policy_server.py` を直接
   `_MIN_LEXICAL = 0.50` と `__version__ = "5.0.0"` に書き換えて組み直し、
   そのあと `git checkout` で戻す。**

5. **名前を付けて sha256 を取り直し、HF へ上げる。**

   ```bash
   mv work/submission.zip work/submission_5.0.0.zip
   rm -f work/submission.sha256
   sha256sum work/submission_5.0.0.zip | tee work/submission_5.0.0.sha256
   hf upload siruku6/parc2026-submission \
       work/submission_5.0.0.zip submission_5.0.0.zip
   ```

   **`build_submission.py` は自分で `work/submission.sha256` を書く**
   （`build_submission.py:280-281`）。**それは書き換える前の zip の値なので消す。**
   台帳に載せるのは `sha256sum` で取り直した方である。

   **同名なので HF 側は上書きになる。**上げ終わったら
   `hf download siruku6/parc2026-submission submission_5.0.0.zip --local-dir /tmp` などで
   **新しい sha256 が返ることまで確かめなくてよい**（9.4 GB を落とし直す価値が無い）。
   代わりに **`hf upload` の出力に出るリビジョンの URL を報告に貼ってほしい。**

6. **リポジトリを汚していないことを確かめる。**

   ```bash
   git status --short
   grep -n '^_MIN_LEXICAL = ' solution/policy_server.py    # 0.65 のままであること
   ```

7. **台帳の 5.0.0 の行を書き換える。**
   [../../reference/19_weights_catalog.md](../../reference/19_weights_catalog.md) に
   5.0.0 が 3 か所ある（「版ごとに何を変えたか」の表・「版ごとの設定」の表・
   「8/28 の RunPod 1,500 歩」の sha256 表）。
   **`正規化` を `入り（0.50）` に、sha256 を新しい値に書き換える。**
   **古い sha256 は消さず、`8/28 に Track2 へ出した版（0.65）` と添えて同じ行に残す**
   ——採点側に返っている点は、そちらの zip のものである。

---

## できあがりの条件

- HF の `submission_5.0.0.zip` が新しいものに替わっている
- **中の `_MIN_LEXICAL` が `0.50`、`__version__` が `5.0.0`**
  （`patch_zip_policy_server.py --show` で確かめられる）
- sha256 が `work/submission_5.0.0.sha256` に残っている
- **リポジトリの `solution/policy_server.py` は `0.65` のまま**
- `submission_6.0.0.zip` と `submission_6.0.1_500steps.zip` が HF に残っている
- 台帳の 5.0.0 の 3 か所が書き換わっている

---

## 報告の書き方

**表と数値だけでよい。**

1. **zip の表**——版・重み・バイト数・sha256・`--show` で出た設定。
2. **重みをどう渡したか**——`--weights` で 001500 を直に指したか、
   `model_weights/` を入れ替えたか。
3. **`vendor/` の件**——作ったか、既にあったか。`--python310` を付けたか。
4. **howto/02 のケース 0 が π0.5 向けでないこと**を 1 行。
5. **想定と違ったところ。**とくに `patch_zip_policy_server.py` が使えたかどうか
   （無圧縮だったか）。

---

## やらなくてよいこと

- **提出。**ユーザがブラウザで出す。
- **`policy_server.py` の恒久的な変更。**しきい値 0.65 はリポジトリに残す。
  **Track3 のバグ修正は別の 1 変数として後日測る。**
- **`001000` と `000500` の zip。**いまは 1,500 歩の 1 本だけでよい
  （**8/28 に範囲を狭めた。**もとは 1,000 歩も作る依頼だった）。
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
| **`6.0.0` と `6.0.1_500steps` を消さないこと** | 台帳が sha256 で対応を記録している。**上書きしてよいのは `5.0.0` だけ** |

---

## 報告

（受け側がここに書く）
