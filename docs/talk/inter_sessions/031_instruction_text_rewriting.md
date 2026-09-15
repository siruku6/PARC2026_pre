# 031. 指示文の書き換えが適切かを判定する —— **寄せ先が正しいか・寄せてよいか**

> **状態: 依頼中** ／ 受け側: 未定 ／ 更新: 2026-09-12
>
> **9/12 に [026](026_contact_demo_cleanup.md)・[030](030_data_side_levers_review.md) の
> 流れから切り出した。**穴そのものと検査の道具は
> [knowledge/23](../../knowledge/23_self_made_demos_need_the_instruction_the_model_sees.md)
> に書いた。**この依頼は「書き換えの中身が正しいか」を人が判定できる形にすることである。**

## 背景（先に読んでほしい）

**このコンペで何をしているか。** PARC2026 の本選である。ロボットアーム
（Franka Emika Panda）に**言葉で指示して**机の上の物を操作させる方策を作り、
シミュレータ（LIBERO / LIBERO-plus）の中で採点される。こちらの方策は公開の基盤モデル
`lerobot/pi05_libero_base` を微調整した π0.5 で、`solution/policy_server.py` が
HTTP で採点器とやり取りする。

**トラックは 3 つあり、対策が違う。**

| トラック | 中身 | こちらの最高点 | リーダーボード最高 |
|---|---|---|---|
| **Track1** | 単一課題・見え方の摂動 | **0.492**（版 5.0.1・8/28） | 0.65 |
| **Track2** | 単一課題・摂動がより強い | **0.261**（版 8.0.0・9/1） | 0.53 |
| **Track3** | **逆向き・未知の言い回し** | **0.000**（9/11 の 14.0.0 は採点待ち） | 0.09603 |

**この依頼が効くのは 3 トラックすべてである。**指示文の書き換えは
`policy_server.py` の入り口にあり、**どのトラックでも全エピソードが通る。**

**何が起きているか。** 採点器が送ってくる指示文は、学習データの言い方と違う。
`InstructionNormalizer` はそれを**学習データの言い方へ寄せてからモデルへ渡す。**
予選ではこれが効いた（版 3.1.0 → 3.5.0 で 0.166 → 0.181・歩数 1,922 → 1,617）。
運営 Q&A の No.29／No.31 で認められた処理である。

**すでに分かっていること・すでに外したこと。**

- **8/27 に 1 件の誤りを見つけて、しきい値を 0.50 → 0.65 に上げた**
  （[knowledge/15](../../knowledge/15_instruction_normalizer_inverted_a_task.md)）。
  Track3 の 1 文（2 つのマグを**皿から下ろす**）が 0.607 で
  「**皿に載せろ**」へ寄っていた。**語の重なりだけで測ると、否定や方向を見ないので
  逆の文がいちばん近くなりうる。**
- **9/1 の版 5.2.0 と 9/11 の版 14.0.0 では、Track3 向けにしきい値を 0.99 にした**
  （＝どの文にも当たらなくなる＝寄せを実質切ってある）。
  **いまの `solution/policy_server.py` は 0.65 である。**
  **つまり「提出した zip ごとにしきい値が違う」状態が続いている。**
- **9/12 に、この処理には裏の要件があることが分かった**——
  **自作した学習データは「寄せた先の言い方」で入っていないと評価で 1 度も引かれない**
  （[knowledge/23](../../knowledge/23_self_made_demos_need_the_instruction_the_model_sees.md)）。
  摂動下再生 1,009 本のうち 626 本（62%）がこれを踏んでいた。

**まだ誰も見ていないのがここである。**「寄せた／寄せない」の判定はしきい値で測ったが、
**寄せた先の文が意味として正しいかを、16 文すべてについて人が読んで確かめたことがない。**
8/27 に見つけた 1 件は**偶然、配布キットの課題定義を読んでいて気づいた**ものである。

## 何をしてほしいか

**「採点器が送る文 → モデルが実際に見る文」の対応表を、出典付きで 1 枚にする。**
**そのうえで、1 行ごとに「この書き換えは妥当か」を判定できる材料を添える。**

成果物は**表 1 枚**である（`docs/reference/` に新しいファイルを 1 つ）。
**判定そのものはユーザが行う。**受け側は判定できる形に並べるところまでやる。

## なぜ要るか

**この表が無いと、しきい値をいくつにするかが決められない。**

いま 0.65 と 0.99 が併存していて、**どちらが正しいかを決める根拠が「1 件の誤りを
見つけた」しか無い。**しきい値を上げれば寄せる回数が減って安全側だが、
**寄せることで得ていた利益（予選で歩数 1,922 → 1,617）も減る。**
**どの文が得をしていて、どの文が損をしているかを 1 行ずつ見ないと、この取引は測れない。**

**もう 1 つ、学習データ側の要件が決まる。**寄せ先が変われば、
自作した手本が持つべき指示文も変わる。
いま [howto/16](../../howto/16_bowl_regen_train_eval.md) で丼 5 課題を作り直していて、
**そこへ入れる文はこの表の右列で決まる。**

## 前提となる事実（**リンクではなく本文に書き写す**）

### 文はこの 4 か所にある

| # | 何の文か | 場所 |
|---|---|---|
| **1** | **採点器が送る文**（Track1・2） | `~/src/b_PARC2026_final/compe/t1/T1_TASKS.csv`・`t2/T2_TASKS.csv` の **`instruction` 列** |
| **2** | **採点器が送る文**（Track3） | `~/src/b_PARC2026_final/compe/t3/assets/bddl_files/libero_t3/*.bddl` の **`(:language …)`** |
| **3** | **寄せ先の候補**（学習データの言い方・40 種） | `work/artifacts/libero_plus/meta/tasks.parquet` の **`task` 列** |
| **4** | **寄せる実装** | `solution/policy_server.py` の `InstructionNormalizer`（`_MIN_LEXICAL = 0.65`・`_MIN_LEXICAL_MARGIN = 0.05`。324〜325 行） |

**1 と 2 は合わせて 16 文である**（Track1 が 4・Track2 が 8・Track3 が 4）。
**ただしこれは配布されている example であって、本番のタスク集合ではない。**
`compe/t3/README.md` に「本番評価は非公開タスクを含む別のタスクセットで実施される」とある。

### 9/12 に測った 16 文の結果（**これはもう出ている。作り直さなくてよい**）

**Track1・Track2 の 12 文はすべて書き換えられ、寄せ先はすべて配布データの 40 種にある。**

| 出典 | 採点器が送る文 | モデルが見る文 | 判定 |
|---|---|---|---|
| `t1/T1_TASKS.csv:2` | Pick the akita black bowl **in the top layer of the wooden cabinet** and place it on the plate | pick up the black bowl **in the top drawer of the wooden cabinet** and place it on the plate | lexical **0.708** |
| `t1/T1_TASKS.csv:3` | **Pick** the tomato sauce and place it in the basket | **pick up** the tomato sauce and place it in the basket | lexical 0.970 |
| `t1/T1_TASKS.csv:4` | **Pick** the milk and place it in the basket | **pick up** the milk and place it in the basket | lexical 0.963 |
| `t1/T1_TASKS.csv:5` | Put the bowl on the stove | put the bowl on the stove | **exact** 1.000 |
| `t2/T2_TASKS.csv:2` | **Pick** the orange juice and place it in the basket | **pick up** the orange juice and place it in the basket | lexical 0.975 |
| `t2/T2_TASKS.csv:3` | Put the cream cheese **on** the bowl | put the cream cheese **in** the bowl | lexical **0.902** |
| `t2/T2_TASKS.csv:4` | **Pick** the ketchup and place it in the basket | **pick up** the ketchup and place it in the basket | lexical 0.963 |
| `t2/T2_TASKS.csv:5` | **Pick** the salad dressing and place it in the basket | **pick up** the salad dressing and place it in the basket | lexical 0.975 |
| `t2/T2_TASKS.csv:6` | Pick the **akita** black bowl on the **cookies** box and place it on the plate | pick up the black bowl on the **cookie** box and place it on the plate | lexical **0.805** |
| `t2/T2_TASKS.csv:7` | Turn on the stove | turn on the stove | **exact** 1.000 |
| `t2/T2_TASKS.csv:8` | Pick the **akita** black bowl on the ramekin and place it on the plate | pick up the black bowl on the ramekin and place it on the plate | lexical **0.762** |
| `t2/T2_TASKS.csv:9` | Pick the **akita** black bowl between the plate and the ramekin and place it on the plate | pick up the black bowl between the plate and the ramekin and place it on the plate | lexical **0.818** |

**Track3 の 4 文はすべて素通りする**（しきい値 0.65 に届かない）。

| 出典 | 文 | 語の重なり |
|---|---|---|
| `take_the_alphabet_soup_out_of_the_tray_…bddl:3` | take the alphabet soup out of the tray and place it on the table in front of the ketchup | 0.402 |
| `take_the_red_mug_off_the_left_plate_…bddl:3` | take the red mug off the left plate and put it back on the table behind the other two mugs | 0.317 |
| `take_the_two_mugs_off_the_plates_…bddl:3` | take the white mug off the left plate and the yellow and white mug off the right plate, then place … | **0.607** ← **0.50 だと誤って寄る文** |
| `take_the_yellow_book_off_the_shelf_…bddl:3` | take the yellow book off the top of the shelf and put it back on the table on the left | 0.36 |

### 逆向きの学習データは、この 4 文と完全一致している

`work/artifacts/reverse_lerobot_v2`（397 本）・`work/artifacts/jaist_mix_v2_x8`（6,176 本）の
指示文は**Track3 の bddl と 1 文字も違わない。**
**配布データの 40 種に無いのは正しい**——寄らないので、採点器の文がそのまま届くからである。

### 同じ表を出し直すコマンド

```bash
cd ~/src/PARC2026_pre
python - <<'PY'
import sys, types, pathlib, csv, re
for name in ("msgpack","fastapi","uvicorn","torch"):
    m = types.ModuleType(name); m.__getattr__ = lambda n: None
    sys.modules.setdefault(name, m)
sys.modules["fastapi"].FastAPI = sys.modules["fastapi"].Request = \
    sys.modules["fastapi"].Response = object
src = pathlib.Path("solution/policy_server.py").read_text()
ns = {"__name__": "x"}
exec(compile(src[:src.index("class MyPolicy")], "policy_server.py", "exec"), ns)
N = ns["InstructionNormalizer"]()
base = "/home/ubuntu/src/b_PARC2026_final/compe"
rows = []
for t in ("t1","t2"):
    with open(f"{base}/{t}/T{t[1]}_TASKS.csv") as h:
        for i, r in enumerate(csv.DictReader(h), start=2):
            if r.get("instruction"): rows.append((f"{t}:{i}", r["instruction"]))
for p in sorted(pathlib.Path(f"{base}/t3/assets/bddl_files/libero_t3").glob("*.bddl")):
    for i, line in enumerate(p.read_text().splitlines(), start=1):
        if ":language" in line:
            rows.append((f"{p.name}:{i}",
                         re.sub(r"^\s*\(:language\s+|\)\s*$", "", line)))
for where, raw in rows:
    out, how, score = N.normalize(raw)[:3]
    print(f"{where}\n  IN : {raw}\n  OUT: {out}\n  {how} {score:.3f}\n")
PY
```

**学習データ側の検査はもう道具になっている。**

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/scripted_demo/check_dataset_instructions.py \
    <データセット> --grader-texts work/regen/grader_texts.txt
```

## やること

1. **上の 16 行を `docs/reference/` の新しいファイルに移す**（`36_instruction_rewriting_table.md` など）。
   **出典の行番号を落とさないこと。**ユーザがファイルを開いて突き合わせられることが目的である。

2. **1 行ごとに「何が変わったか」を語の単位で書く。**
   いまの表は文を丸ごと並べているだけで、**どの語が入れ替わったかが目で追えない。**
   例: `akita black bowl` → `black bowl`（形容詞が落ちた）・
   `cookies box` → `cookie box`（単複）・`on the bowl` → `in the bowl`（**前置詞。位置関係が変わる**）・
   `top layer` → `top drawer`（**物の呼び名が変わる**）。

3. **「意味が変わったか」を 3 段階で分類する。**
   **これが判定の材料である。**
   - **無害**（大文字小文字・`pick`→`pick up`・単複）
   - **呼び名が変わる**（`akita black bowl`→`black bowl`・`top layer`→`top drawer`）
   - **関係が変わる**（`on the bowl`→`in the bowl` のような前置詞・方向・否定）
   **3 番目に入るものは 1 行ずつ根拠を書く**——そのシーンで実際にどちらが正しいかを、
   bddl の `:goal` を読んで確かめる（`compe/t2/.../*.bddl` か LIBERO-plus 側）。

4. **しきい値を動かしたときに何が変わるかを表にする。**
   `solution/tools/suite/check_instruction_normalization.py --sweep` が既にある。
   **0.50 / 0.65 / 0.99 の 3 点**について、
   **16 文のそれぞれが「寄る／寄らない」どちらになるか**を並べる。
   **いま提出済みの zip は Track3 が 0.99・それ以外が 0.65 である**ことを表に明記する。

5. **本番タスクが未知であることをどう扱うかを 1 段落書く。**
   16 文は example であって本番の集合ではない。
   **「16 文で誤りゼロ」が本番で何を保証するのか・しないのか**を書く。

## できあがりの条件

- [ ] `docs/reference/` に表 1 枚があり、**16 行すべてに出典（ファイル名と行番号）が付いている**
- [ ] **各行に「変わった語」と「3 段階の分類」がある**
- [ ] **「関係が変わる」に分類した行は、bddl の `:goal` を読んだ根拠が付いている**
- [ ] しきい値 0.50 / 0.65 / 0.99 の比較表がある
- [ ] **上のコマンドをそのまま打てば同じ表が出る**
- [ ] `knowledge/23` と `knowledge/15` から新しいファイルへの参照が張ってある

## やらなくてよいこと

- **しきい値を変更しないこと。**この依頼は材料を作るところまでで、**決めるのはユーザである。**
- **`solution/policy_server.py` を書き換えないこと。**
- **提出物（zip）を組まないこと。**
- **学習を回さないこと。**
- **1,527 種の言い換えの束**（`check_instruction_normalization.py` が持っている）
  **を全部並べないこと。**見るのは**採点器が実際に送る 16 文**である。
  言い換えの束は「寄せ過ぎ／寄せ足りない」の傾向を見るためだけに使う。

## 報告の書き方

**表と数値がほしい。長い説明は要らない。**

1. 完成したファイルのパス
2. **3 段階の分類の内訳**（無害 N 件・呼び名 N 件・関係 N 件）
3. **「関係が変わる」に入った行の一覧と、その根拠 1 行ずつ**
4. しきい値 3 点で判定が変わる行が何行あったか
5. **途中で測り方を変えたなら、その理由**

## 判断が必要になったら

**この依頼は「指示文を寄せるしきい値をいくつにするか」というユーザの判断を変えるためのものである。**
いま 0.65 と 0.99 が併存していて、**根拠は「1 件の誤りを見つけた」しか無い。**

**その目的に照らして、より良い測り方を思いついたら変えてよい。**
たとえば「語の重なり」以外の物差し（前置詞と否定だけを別に見る、など）で
分類したほうが判定しやすいなら、そうしてよい。**変えた場合は理由を報告に書くこと。**

**逆に、次の 2 つは変えないでほしい。**
**① 出典の行番号を付けること**（ユーザが自分で突き合わせるための表である）。
**② 16 文だけを見ること**（本番の集合ではないと分かったうえで、それでも
この 16 文が唯一の一次資料である）。
