# 13. 学習に使わなかった実演から評価集合を作れるか

**手元の 84 件が本番の順序を当てられなかったのは、学習に使ったタスクを検証にも
使っていたからである。** そこで「学習が引かなかった実演」から評価集合を作った。
**その根拠と、言えること・言えないことをここに置く。**

結果として `compe/val/EVAL_TASKS.csv`（36 件）ができた。
作り方の手順は [../howto/04_proxy_eval.md](../howto/04_proxy_eval.md)、
回し方は [../howto/03_gpu_eval.md](../howto/03_gpu_eval.md) にある。

> **出典。**
>
> | 出典 | 何が分かるか |
> |---|---|
> | `outputs/train2/weak3x_max300/episodes.json` | **その学習が使ったエピソード番号**（`train.py` が書く） |
> | `outputs/tmp/libero_plus_episodes.parquet` | `lerobot/libero_plus` の `meta/episodes/*.parquet` を 1 つにしたもの |
> | `work/LIBERO-plus/.../task_classification.json` | 変種 10,030 件の名前・カテゴリ・難易度 |
> | 運営 Q&A CSV No.21 / No.33 | 採点の摂動カテゴリ、自己評価の使用可否 |
>
> **`solution/` 以下の説明文と `docs/` の他の `.md` は出典にしていない。**

## 手元でメタデータを読む方法

**Mac には numpy も pandas も入っていない。** `pyarrow` だけが
`work/pylibs` にあり、システムの `python3` から読める。

```bash
PYTHONPATH=work/pylibs python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('outputs/tmp/libero_plus_episodes.parquet')
print(t.num_rows, '行', t.num_columns, '列')
print([n for n in t.column_names if not n.startswith(('videos/', 'data/'))][:8])
"
```

62 列あり、判定に使えるのは `episode_index`・`tasks`（指示文）・`length`（手本の歩数）と
`stats/` で始まる統計量（画像 2 台・状態・行動の min/max/mean/std）である。

## 数えた結果

| | 本数 | |
|---|---|---|
| データセット全体 | **14,347** | 指示文 40 種 |
| その学習が使った | **4,609** | `episodes.json` の `episodes` |
| **使わなかった** | **9,738** | |
| うち**使ったものと見た目が同じ** | 7,396（75.9%） | **使えない** |
| うち**使ったものに無い見た目** | **2,342（24.1%）** | |
| さらに**手本が 300 歩以下** | **2,068** | **これが候補**（36 指示文） |

見た目は `stats/observation.images.front/mean` を 3 台のチャンネルで平均し、
**0.001 刻みで丸めた値**を場面の識別子として使った。

### なぜ 4 分の 3 が使えないのか

**間引きの仕組みそのものが原因である。** `train.py` の
`_choose_spread_by_value` は**明るさで並べて等間隔に残す**ので、
**落ちた側は残った側と明るさが交互に並ぶ。** 見た目が近いものが未使用に回る。

さらに `--emphasize-tasks` で強調した 3 タスクは
**エピソードを全部使っている**ので、長さで落ちた分を除けば未使用が無い。

## 「未使用だから未知」とは言えない —— 変種と対応付けられない

**評価はデータセットの実演を再生するのではなく、LIBERO-plus の変種を環境として
走らせる。** そこを繋げるかを確かめたが、繋がらなかった。

| 調べたこと | 結果 |
|---|---|
| データセットは変種を区別しているか | **していない。** `stats/task_index` は **40 種**しかなく、指示文と 1 対 1 |
| 変種の数と実演の数は対応するか | **しない。** 変種 10,030 に対し実演 14,347。指示文ごとの比は **0.58〜2.95（中央 1.45）** で整数にならない |

**データセットは変種の一部を不揃いな本数で収録している。**
したがって「この変種は学習に使っていない」とは言えない。

| | |
|---|---|
| **言える** | その指示文には、**学習が 1 度も引かなかった実演があり、その見た目は学習に使った実演のどれとも違う** |
| **言えない** | 選んだ変種そのものが学習に入っていないこと |

**強めるなら**、変種の初期画面を描いて明るさを測り、
`EVAL_TASKS.meta.json` の `seen_brightness` の外側にあるものだけを残す。
描画には GPU マシンが要る。

## 同じ指示文の中でも場面は違う（確かめた）

**「同じタスクなら配置も指示も同じでは」という懸念は当たらなかった。**
1 つの指示文の中に、明るさで見て**約 94 通りの見た目**がある（中央 385 本）。

| 指示文 | 本数 | 明るさの幅 | 0.001 刻みの種類 |
|---|---|---|---|
| `turn on the stove and put the moka pot on it` | 390 | 0.179〜0.718 | 128 |
| `pick up the chocolate pudding and place it in the basket` | 500 | 0.100〜0.676 | 115 |
| `open the top drawer and put the bowl inside` | 200 | 0.163〜0.544 | 75 |

**約 4 本が同じ明るさを共有する**形なので、見た目の変種ごとに初期状態が
数本ある構造だと読める。**エピソードは同じ場面の焼き直しではない。**

## 落とした指示文

**未使用に未知の見た目が無い 4 種を外した**（長さ 300 歩以下の条件も含めた判定）。
長さの条件を外して数えると、未知の見た目が 1 本も無いのは 2 種だけである。

そのうち 1 つは**丸ごと学習されていない**が、それでも使えない。

```
put both moka pots on the stove
  146 本すべて未使用 / 手本の歩数 最小 341・中央 396・最大 505
  300 歩以下: 0 本
```

`--max-demo-steps 300` で全部落ちたためである。**タスク単位の完全な
ホールドアウトが手に入った形だが、手本が最短 341 歩なので、300 歩で打ち切る
採点では解ける見込みが薄い。** 全方策が 0 になり判別に使えない。

**長さで落ちた 401 本も同じ理由で使いにくい**（うち 274 本は見た目も未知だが、
いずれも手本が 300 歩超）。

## タスク単位で外すのは採らなかった

**指示文を丸ごと外すと学習データが減る。** 1 タスクあたり 370〜445 本あるので、
40 種のうち 8 種を外すと**およそ 2 割**が消える。
**エピソード単位なら、その代償がずっと小さい。**

`train.py --holdout-tasks` と `make_holdout_split.py`（`compe/val/HOLDOUT_TASKS.csv`）は
タスク単位で外す道具として残っている。**いまの 36 件はそちらを使っていない。**

## 出発点の重みは全タスクを見ている（直せない）

素の重みは同じ `lerobot/libero_plus` 系のデータで 20,000 歩学習済みである
（重みに同梱の `train_config.json`）。**LIBERO-plus のどの場面も、
素の重みにとっては未知ではない。**

したがってこの 36 件で測れるのは
**「こちらが足した学習が、覚えていない場面でも効くか」**であって、
素との絶対的な比較が綺麗になるわけではない。
**ただし本番も同じ非対称を持っている**（本番も「全部見た素」と比べている）ので、
忠実さの点では問題にならない。

## ルール上の位置

**問題ない。** 運営 Q&A No.33 の回答が線を引いている。

> 禁止対象の「評価観測を保存しての後続学習」は、**運営の採点（本番評価）で
> 得られた観測・結果を利用すること**を指します。ご自身の環境での実行は
> これに含まれません（…）ご自身の環境での自己評価の結果（タスク別成功率等）を
> 「何を学習するか」の選定に使うことは問題ありません。

選定の根拠は学習の記録と LIBERO-plus であり、採点環境の情報は使っていない。
**採点ログからタスクを特定して選定に使うのは禁止側である**ので、やらない。

## この集合が固定である条件

**`EVAL_TASKS.csv` は「どの学習に対して未使用か」で決まる。**
いまのものは `outputs/train2/weak3x_max300/episodes.json`（weak3x の回）を根拠に
作った。**別の学習の重みを測るなら、その記録で作り直す必要がある。**

根拠は `compe/val/EVAL_TASKS.meta.json` の `record` に書かれている。
**スコアを見たあとに作り直すと選び直しになる**ので、
**先に確定させて git に入れてある。**

## 確認しなおす手順

```bash
# 使った・使わなかったの内訳と、見た目が未知の本数
PYTHONPATH=work/pylibs python3 - <<'PY'
import json, statistics as st
from collections import defaultdict
import pyarrow.parquet as pq

table = pq.read_table('outputs/tmp/libero_plus_episodes.parquet', columns=[
    'episode_index', 'tasks', 'length', 'stats/observation.images.front/mean'])

def flat(value):
    if isinstance(value, (int, float)):
        return [float(value)]
    return [n for item in value for n in flat(item)]

used = set(json.load(
    open('outputs/train2/weak3x_max300/episodes.json'))['episodes'])
grouped = defaultdict(lambda: {'used': [], 'unused': []})
for index, task, length, stat in zip(
        table.column('episode_index').to_pylist(),
        table.column('tasks').to_pylist(),
        table.column('length').to_pylist(),
        table.column('stats/observation.images.front/mean').to_pylist()):
    name = task if isinstance(task, str) else str(task[0])
    key = 'used' if int(index) in used else 'unused'
    grouped[name][key].append((st.fmean(flat(stat)), int(length)))

novel = same = candidates = 0
for split in grouped.values():
    seen = {round(b, 3) for b, _ in split['used']}
    for brightness, length in split['unused']:
        if round(brightness, 3) in seen:
            same += 1
        else:
            novel += 1
            candidates += length <= 300
print(f'使った {len(used):,} / 見た目が同じ {same:,} / 未知 {novel:,} '
      f'/ 未知かつ 300 歩以下 {candidates:,}')
PY

# 変種と実演が対応しないこと
PYTHONPATH=work/pylibs:solution/tools python3 - <<'PY'
import json, statistics as st
from collections import Counter, defaultdict
import pyarrow.parquet as pq
from make_holdout_split import base_task, instruction

classification = json.load(open(
    'work/LIBERO-plus/libero/libero/benchmark/task_classification.json'))
variants = defaultdict(int)
for items in classification.values():
    for item in items:
        variants[instruction(base_task(item['name']))] += 1

table = pq.read_table('outputs/tmp/libero_plus_episodes.parquet',
                      columns=['tasks'])
episodes = Counter(
    value if isinstance(value, str) else str(value[0])
    for value in table.column('tasks').to_pylist())
ratios = [episodes[name] / variants[name] for name in episodes if variants[name]]
print(f'変種 {sum(variants.values()):,} / 実演 {sum(episodes.values()):,}')
print(f'ep/変種  最小 {min(ratios):.2f} 中央 {st.median(ratios):.2f} '
      f'最大 {max(ratios):.2f}  → 整数にならない')
PY
```

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| ~~36 件に判別力があるか~~ | **済み（8/13）。素の重みで 21/36 = 0.583**（打ち切りは 8 件だけ）。**天井も床もない**（[../howto/03_gpu_eval.md](../howto/03_gpu_eval.md)）。**残るのは順序の検定** ——LoRA 20k を同じつまみで回して、本番の 0.1409 > 0.016 を再現するか |
| **摂動カテゴリ別の表が読めるか** | **読めない。**1 指示文 1 変種なので、タスクの難しさと分けられない（[../plan/04_training_targets.md](../plan/04_training_targets.md) の 5 節）。**引き算ができるのは 84 件だけ** |
| 選んだ変種が本当に未知か | 変種の初期画面を描いて明るさを測り、`seen_brightness` の外か見る |
| 明るさ以外の軸 | `stats/observation.state` の min/max で初期姿勢や動きの範囲を比べる。**未着手** |
| 2 つの学習データが同じ軌道か | `pepijn223/libero_plus_lerobot` と `lerobot/libero_plus` の `meta/info.json` を突き合わせる |
