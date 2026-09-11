# 23. 自作した手本は、**評価でモデルが実際に見る文**で入れないと 1 度も引かれない

**2026-09-12 に、摂動下再生 1,009 本を学習データに足そうとして見つけた。
1,009 本のうち 626 本（62%）が、評価で一度も渡らない指示文の下に入っていた。**

> **これが分かると何が決まるか。****自作したデータを足す作業すべてに、検査が 1 つ増える。**
> 見落としても学習は成功し、損失も下がり、エラーも出ない。
> **出るのは「作ったのに効かない」という結果だけで、原因がデータ量や学習の刻みに見える。**

| 節 | 中身 |
|---|---|
| [1 何が起きるか](#1-何が起きるか) | 方策が見る文は、採点器が送った文ではない |
| [2 規則](#2-規則配布-40-種に合わせるではない) | **「配布 40 種に合わせる」ではない** |
| [3 検査のしかた](#3-検査のしかた) | 道具 1 つ・1 秒 |
| [4 手元のデータを検査した結果](#4-手元のデータを検査した結果) | 逆向きデータは無事・摂動下再生は 62% が死んでいた |
| [5 直し方](#5-直し方) | `--normalize-instructions` |

---

## 1. 何が起きるか

**`solution/policy_server.py` の `InstructionNormalizer` は、採点器から来た文を
学習データの言い方へ寄せてからモデルへ渡す**（[15](15_instruction_normalizer_inverted_a_task.md)）。
これは運営 Q&A No.29／No.31 で認められた処理である。

```
採点器が送る     Pick the akita black bowl on the cookies box and place it on the plate
                  │  InstructionNormalizer（lexical 0.81）
                  ▼
モデルが見る     pick up the black bowl on the cookie box and place it on the plate
```

**したがって学習データ側は「寄せた先の言い方」で入っていなければ、
その手本は評価で 1 度も引かれない。** bddl の言い方のまま足すと、
モデルにとって**41 番目の課題**になる。

**自作したデータは既定で必ずずれる。**`scripted_demo.py` の `record` が残すのは
`env.language_instruction`（＝bddl の `:language`）で、
**配布データの `task` 列とは綴りが違う**からである。

## 2. 規則——「配布 40 種に合わせる」ではない

**正しくは「評価でモデルが実際に見る文に合わせる」である。**
正規化はしきい値（0.65・[15 の 3 節](15_instruction_normalizer_inverted_a_task.md#3-なぜ-065-なのか)）を
**超えたときしか書き換えない。**超えない課題では採点器の文がそのまま届く。

| 課題 | 採点器が送る文 | 正規化 | **学習データが持つべき文** |
|---|---|---|---|
| Track1・Track2 | bddl の言い方 | **寄る**（実測 0.71〜1.00） | **配布データの言い方** |
| Track3（逆向き） | bddl の言い方 | **寄らない**（実測 0.32〜0.40） | **bddl の言い方そのまま** |

**この 2 つは逆向きの要求である。**片方の規則をもう片方に当てると壊れる。
**しきい値を動かすと、学習データが持つべき文も変わる**——
この 2 つは同じ数字で結ばれている。

## 3. 検査のしかた

```bash
# 採点器が送る文を書き出す（配布キットから。1 行 1 文）
{ for t in t1 t2; do python3 -c "
import csv
p='~/src/b_PARC2026_final/compe/$t/T${t#t}_TASKS.csv'
[print(r['instruction']) for r in csv.DictReader(open(p)) if r.get('instruction')]
"; done
  grep -h ':language' ~/src/b_PARC2026_final/compe/t3/assets/bddl_files/libero_t3/*.bddl \
    | sed 's/^ *(:language //; s/)$//'; } > work/regen/grader_texts.txt

IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/scripted_demo/check_dataset_instructions.py \
    <データセット> ... --grader-texts work/regen/grader_texts.txt
```

**ずれが 1 つでもあれば終了コード 1 を返す。**データセットを組む手順に挟める。

**`--grader-texts` を省くと基準が配布データの 40 種だけになる。**
Track3 のように寄らない文を持つデータセットは、それだけだと誤って「ずれている」と出る。
**配布キットはリポジトリの外にあるのでコンテナからは見えない。**
`--grader-dir` を渡しても 0 件になるが、**そこは黙って進まず落ちる**ようにしてある。

## 4. 手元のデータを検査した結果

**2026-09-12 の実測。**

| データセット | 種 / 本 | 判定 |
|---|---|---|
| `work/artifacts/reverse_lerobot_v2` | 3 / 397 | **OK**。3 文とも Track3 の bddl と完全一致 |
| `work/artifacts/jaist_mix_v2_x8` | 43 / 6,176 | **OK** |
| `work/regen/lerobot_batch1` | 38 / 1,009 | **24 種・626 本（62%）がずれている** |

**逆向きデータは無事だった。**配布 40 種に無い 3 文を持っているが、
**それは採点器が Track3 でそのまま送る文である。**
「配布に無いから直す」と判断していたら、**正しいものを壊していた。**

**摂動下再生は 62% が死んでいた。**内訳は
`pick the …`→`pick up the …`（`pick up` の脱落）、
`akita black bowl`→`black bowl`、`cookies box`→`cookie box`、
`top layer of the drawer`→`top drawer of the cabinet` など、
**すべて寄せれば配布データの言い方に着く**種類のずれである。

## 5. 直し方

```bash
python3 solution/tools/scripted_demo/npz_to_lerobot.py build \
    --npz-dir <npz> --output <出力> --repo-id <id> --normalize-instructions
```

**寄せる実装は `policy_server.py` のものをそのまま呼ぶ。**
書き写すと必ず片方が古くなる（読み込みは
`solution/tools/suite/check_instruction_normalization.py` の
`_load_normalizer_class()` を借りる——`policy_server.py` は `msgpack` を
import するので、偽の module を差さないとコンテナの中で止まる）。

**`none` で寄らなかった文が 1 つでもあれば止めること。**寄っていない＝評価で引かれない。

## Claude 側

**「データを作った」で終わりにしない。**作ったデータが**評価の経路に載るか**を
別の物差しで確かめる。今回は「学習が通った・損失が下がった」まで正常で、
**誰も引かないことだけが分からなかった。**
**沈黙して失敗する経路は、作った直後に 1 回測る。**

## 人間側

**推論側に入れた前処理は、学習データ側の要件になる。**
`InstructionNormalizer` は推論を良くするために入れたものだが、
**入れた瞬間に「学習データはこの言い方で持て」という制約が生まれた。**
**前処理を足したら、その逆要件を書き留める。**
