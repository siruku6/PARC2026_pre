# 026 接触が起きているデモを配布データから外し、外す前と比べる

> **状態: 着手（掃除済みデータセットまで出来た。次は学習 1 本）** ／
> 受け側: このセッション ／ 更新: 2026-09-09
>
> **9/9 に、ユーザの指示で 019 とは別に立てた。**019 は「再生しても再現しない
> デモを外す」で、こちらは**「対象外の物体に触れているデモを外す」**である。
> **物差しが違うので別の依頼にした。**
>
> **決まった方針と出来上がったデータセットは末尾の 12 節にある。**
> **`on the ramekin` は本選 1 では扱わない**——[027](027_ramekin_task_unwinnable.md) へ切り出した。

## 背景（先に読んでほしい）

**このコンペで何をしているか。** PARC2026 の本選 1 である。ロボットアーム
（Franka Emika Panda）に言葉で指示して机の上の物を操作させる方策を作り、
シミュレータ（LIBERO / LIBERO-plus）の中で採点される。こちらの方策は
公開の基盤モデル π0.5 を追加学習したものである。**締切は 9/17（木）23:59。**

### この依頼が効くのは Track2 である

| トラック | 中身 | これまでの最高点 | 順位 | 他の参加者の最高 |
|---|---|---|---|---|
| Track1 | 学習済みタスク・背景と照明が変わる | 0.492（版 5.0.1） | 36 位 | 0.65 |
| **Track2** | 学習済みタスク・視点・初期姿勢・物の配置・雑音が変わる | **0.261（版 8.0.0）** | **91 位** | 0.53 |
| Track3 | 見たことのないタスク | 0.000 | — | 0.000（ほぼ全員） |

**Track2 で落としているのは「届く本数」ではなく「届き方」である。**
8 本のうち 6 本は目標に届いていて、成功と数えられたのは 3〜4 本しかない。
**2〜3 本が衝突で消えている**（`talk/open_items.md` の `F60`。
[reference/19](../../reference/19_weights_catalog.md) の「300 未満は『目標に届いた』であり…」）。

**衝突の門は厳しい。**採点は、**目当て以外の物が 1 mm 動いた時点で失格**にする
（`threshold_m` 0.001）。物理が落ち着くのを待った後の位置が基準になる
（`pipeline/rollout.py`）。

**そこで疑うのが学習データそのものである。**手本のデモが運ぶ途中で皿やトレイを
押しているなら、**方策は「押しても良い」を教わっている**ことになる。

### 測る対象のデータ

`work/artifacts/libero_plus`（= Hugging Face の `lerobot/libero_plus`）。
**14,347 本・2,238,036 コマ・指示文 40 種**、LeRobot v3.0、20 Hz、15 GB。
**配布 tar の `libero_combined_20hz`（19,533 本・111 課題）は使わない。**
比べる相手を揃えるためで、`libero_90` が入っていないほうで良い（ユーザの指定）。

### すでに分かっていること・すでに外したこと

| 分かっていること | どこ |
|---|---|
| 配布データに成功フラグは無い。1 本ずつ確かめるしかない | [019](019_forward_replay_failure_exclusion.md) の 2.4 |
| どのエピソードがどの摂動版かを示す列はどこにも無い | [reference/06](../../reference/06_training_data_composition.md) |
| 「順方向 goal が立ったか」では選別できない（述語の相手が噛み合わない課題がある） | [019](019_forward_replay_failure_exclusion.md) の 2.2 |
| **予選の項目 21「衝突しない手本だけを選ぶ」は、着手の条件を書いただけで止まっている** | `talk/open_items_notes.md` |
| 配布データを 111 課題へ増やす道は打ち切り済み（3 トラックとも下がった） | [260906_briefing.md](../260906_briefing.md) の 2 節 |

## 何をしてほしいか

**2 つである。**

1. **`work/artifacts/libero_plus` から「対象外の物体に触れているデモ」を外した
   データセットを作る。**
2. **`lerobot/pi05_libero_base` から 24,000 歩の action expert 学習を、
   掃除前と掃除後で 1 本ずつ回し、接触率が変わるかを見る。**

**ただし 4 節の「やること」の 5 番で一度止めて、外れる割合を報告する。**
そこから先（外す単位・対照を置くか）はユーザが決める。

## なぜ要るか

**Track2 の残り 0.12〜0.19 点を取りにいく手が、いまひとつも無い**（`F60` が
「未着手（何を変えるかがまだ無い）」のまま）。**この依頼はその 1 つ目の手である。**

**そして「効いた／効かない」がはっきり出る形になっている。**24,000 歩の学習を
データだけ変えて 2 本回し、**接触率を比べる。**変わらなければ、この道は打ち切って
別の手に移れる。

## 前提となる事実（2026-09-09 に実測した）

### 1. 14,347 本の中身は、**1,681 個の物理を見た目だけ変えて焼き直したもの**である

全エピソードの `action` 列と `observation.state` 列を突き合わせた。

| 数えたもの | 値 |
|---|---|
| エピソード | 14,347 |
| **行動列が 1 ビットも違わない組** | **1,681** |
| **状態列（手先・向き・指 2 本、全コマ）が 1 ビットも違わない組** | **1,681**（行動列の組と完全に一致） |
| 1 コマ目の状態の種類 | 1,681 |
| 1 組あたりのエピソード | 10 本が 1,175 組・5 本が 477 組（平均 8.5） |
| 1 指示文あたりの組 | 28〜50 |

**指の開き具合まで 1 ビットも違わない。**物の置き場所が違えば掴む瞬間に指の
閉じ方が変わるので、**組の中では物の配置も同じ**と言える。**違うのは画だけである**
（カメラ位置・照明・質感・雑音）。

**帰結が 2 つ。**

- **接触の判定は 1,681 回で済む。**14,347 回は要らない。
- **1 組を外すと平均 8.5 本が消える。**刻みはこの粗さになる。

### 2. **配布データからは接触を判定できない**——物体の位置が 1 列も無い

`meta/info.json` の列は `observation.state`（8 次元 ＝ 手先位置 3・向き 3・指 2）と
`action`（7 次元）だけである。物がどこにあるかはどこにも入っていない。

### 3. LIBERO 本家の hdf5 には、**MuJoCo の状態がまるごと**入っている

`yifengzhu-hf/LIBERO-datasets` の `<スイート>/<課題>_demo.hdf5`。
1 デモにつき `states`（79〜110 次元。課題によって物の数が違う）と `actions` がある。
**再生をやり直す必要が無く、記録そのものから全物体の姿勢が読める。**

| | 値 |
|---|---|
| 要るスイート | `libero_goal`・`libero_object`・`libero_spatial`・`libero_10`（40 課題。`libero_plus` の 40 指示文と一致する） |
| 大きさ | 33.78 GB（画像込み） |
| 落とす速さ | 78 MB/s の実測（0.45 GB が 5.8 秒） |
| **抜き出したあとの大きさ** | **21 課題で 63 MB**（`states` と `actions` だけ写す） |

### 4. 対応付けは通る——**1 課題で 50 組すべてが本家デモと一致した**

`turn on the stove`（`libero_goal`）で確かめた。

| 突き合わせ方 | 一致 |
|---|---|
| そのまま比べる | 47 / 50 |
| **両側から「前 6 次元がすべて 0 のコマ」を落としてから比べる** | **50 / 50** |

**落ちるコマは `no_noops` の処理で消えたものである。**片側にだけ掛けると
49 / 50 で止まる（`demo_25` は本家 90 コマ・配布 90 コマだが、中の 1 コマだけ
扱いが違う）。**両側に掛けるのが正しい。**

### 5. 使える機械とディスク

| | |
|---|---|
| この機械 | JAIST サーバ（`gpu-422591`）。H100-20C（20 GB）・vCPU 16 |
| **`docker run` は通らない**（このセッションの権限） | 既に起きている `parc2026_verify` と `quant_train` への `docker exec` は通る |
| 空きディスク | **53 GB** |
| 学習 | **VRAM 20 GB なので action expert だけ（693M）**。全パラメータは RunPod か Blackwell |

## やること

> **1〜7 は 9/9 に実行済みである。残っているのは 8（学習）だけ。**
> **打ち直す必要は無い**——出来上がっているものは次のとおりで、結果は
> このファイルの「報告」以降（1〜13 節）にある。
>
> | 何 | どこ | 中身 |
> |---|---|---|
> | 1 の出力 | `work/artifacts/libero_states/` | 40 課題 / 2,000 デモ / 338,575 コマ |
> | **2 の対応表** | **`work/artifacts/libero_states/demo_map.json`** | 1,681 組 ↔ 本家デモ（**100% 対応**） |
> | 3・4 の判定 | `work/artifacts/libero_states/contact_scores.json` | 1 組 1 行・物差し 2 つ |
> | 6 の残す一覧 | 同 `keep_cleaned.json` / `keep_boost.json` | 12,216 本 / 90 本 |
> | **7 の掃除済みデータ** | **`work/artifacts/libero_plus_clean`** | **12,396 本 / 1,940,083 コマ / 39 課題** |
>
> **各段のコマンドは報告の 1 節と 12.3 節にそのまま置いてある。**

### 1. 本家 hdf5 から `states` を抜く —— **済（9/9）**

```bash
cd /home/ubuntu/src/PARC2026_pre
.venv/bin/python solution/tools/contact_clean/fetch_libero_states.py \
    --out work/artifacts/libero_states
```

**1 課題ぶんを落とす → 抜く → hdf5 を消す、を 40 回まわす。**
どの時点でもディスクは 2.1 GB しか使わない。何度打ってもよい（済んだ課題は飛ばす）。

### 2. 1,681 組と本家デモの対応表を作り、**カバー率を出す** —— **済（9/9）**

```bash
.venv/bin/python solution/tools/contact_clean/map_demos.py \
    --dataset work/artifacts/libero_plus --states work/artifacts/libero_states \
    --out work/artifacts/libero_states/demo_map.json
```

**中でやっていること。**配布データの `action` 列をエピソードごとに突き合わせて
**同じものを 1 組にまとめ**（14,347 本 → 1,681 組）、本家 hdf5 の 2,000 デモの
`actions` と**行動列そのものを鍵にして**引き当てる。**両側から「前 6 次元が
すべて 0 のコマ」を落としてから突き合わせる**（前提の 4）。

**出すもの。** 1 組 1 行の表——`libero_plus` の指示文 / 代表エピソード番号 /
その組に属するエピソード番号の一覧 / 本家の `<スイート>/<課題>` と `demo_<番号>` /
落ちたコマ数。**そして「対応が付かなかった組が何組か」。**

**対応が付かない組が出たら、その割合を報告に書く。**付かない組は掃除の判定が
できないので、**残す側に倒す**（消さない）。

### 3. `states` から物体の姿勢を読む

**MuJoCo の状態ベクトルのどこがどの物体かは、env を作らないと決まらない。**
`parc2026_verify` の中で bddl から env を組み、`sim.model` の関節の並びを引く。
**物理は 1 歩も進めない。描画もしない**ので、EGL の無いこの機械でも通る。

```bash
docker exec parc2026_verify python /workspace/<置いたパス>/read_object_poses.py ...
```

**確かめること。**抜いた `states` の 1 コマ目を env に流し込み、
**`libero_plus` の `observation.state` の 1 コマ目（手先の位置）と一致するか。**
ここがずれたら、対応付けか状態の読み方のどちらかが間違っている。

### 4. 2 つの物差しで接触を数える

**どちらを採るかは決めない。両方を出す**（ユーザの指定）。

| 物差し | 中身 |
|---|---|
| **(a) 1 mm** | **対象外の物体が、デモの間に 1 mm 以上動いたか。**採点と同じ物差し |
| **(b) 触れた** | **腕（と掴んでいる物）が対象外の物体に触れたか。**(a) より厳しい |

**「対象外」の決め方も報告に書くこと。** bddl の goal に名前が出る物と、
それを載せる相手（皿・トレイ・引き出し）をどう扱うかで数が変わる。
`compe/val/EVAL_TASKS.csv` に `n_nontarget_objects` 列があるので、
**採点側の数え方と突き合わせてから決める。**

### 5. **ここで止めて、外れる割合を報告する**

| 出すもの | なぜ |
|---|---|
| 物差し (a)(b) それぞれで、外れる組の数と、消えるエピソードの数 | **外す単位と対照の要否がここで決まる** |
| 指示文ごとの内訳 | 特定の課題だけが落ちるなら、掃除ではなく課題の問題である |
| いちばん動いた物体と、その量の分布 | 1 mm という境目が妥当かを見る |

### 6. 残す番号の一覧でデータセットを切り出す口を足す

**いまある `subset` は「課題ごとに等間隔で N 本」しかできない**
（`solution/tools/scripted_demo/npz_to_lerobot.py` の `command_subset`）。
中で呼んでいる `split_dataset` は番号の一覧をそのまま受けるので、
**`--keep-file` を足すだけで済む。**

### 7. 掃除済みデータセットを作り、確かめる

```bash
python3 solution/train/verify_dataset.py <掃除済み> --expect "<本数> <コマ数> 40"
```

**`--norm` と `--rev-tasks` は省いてよい。**どちらも既定が `None` で、渡さなければ
その項目を飛ばす（`verify_dataset.py` の `main`）。**引っかかるのは 6 番の動画の
確認だけである**——中で `docker run` を打つので、このセッションでは通らない
（145 行目）。**`--skip-video` で飛ばすと学習で初めて落ちる**ので、動画は
`docker exec parc2026_verify` で `av` を呼ぶ形に書き換える。

### 8. 学習を 2 本回して比べる

| | 中身 |
|---|---|
| 出発点 | **`lerobot/pi05_libero_base`**（ユーザの指定） |
| 歩数 | **24,000**・action expert だけ（693M） |
| 変えるもの | **データだけ。**掃除前（14,347 本）と掃除後 |
| 見るもの | **接触率。**成功率だけでは打つ手が決まらない |

**正規化統計は据え置かない。**`pi05_libero_base` は normalizer に統計を持って
いないので、**最初の学習が尺度を決める**（[reference/29](../../reference/29_training_data_requirements.md) の 6 節）。
**据え置きを打つのは「統計を持っている重みから続ける」ときだけ**で、
これまでに打ったのは `pi05_proto20` と `pi05_t3rev` の 2 本しかない。

**ただし、掃除前と掃除後で統計がずれると「尺度の違う 2 本」を比べることになる。**
**掃除で std がどれだけ動くかを先に出し、大きければ両方を掃除前の統計に揃える。**
耐えられた実績は ±8% しかない（同 6.3 節）。

## できあがりの条件

| # | 何が揃ったら終わりか |
|---|---|
| 1 | 1,681 組と本家デモの対応表があり、**対応が付かない組の数が書いてある** |
| 2 | 1 組 1 行で、物差し (a)(b) それぞれの判定が付いた表がある |
| 3 | 外れる組・消えるエピソードの数が、全体と指示文ごとに出ている |
| 4 | 掃除済みデータセットが `verify_dataset.py` を通っている |
| 5 | 24,000 歩の学習 2 本の**接触率**が並んでいる |

## やらなくてよいこと

- **逆向きデータ（`work/probe_017/reverse_lerobot_all` の 389 本）は触らない。**
  あちらは成否で既に絞ってある（[019](019_forward_replay_failure_exclusion.md) の 4.2）
- **`libero_90` と配布 tar の `libero_combined_20hz` は使わない**（ユーザの指定）
- **シムでの再生はしない。**記録された `states` を読む道が通るかぎり、
  再生には誤りが入りうる（ユーザの指定）
- **`--dataset.episodes` で本数を絞る道は採らない。**データセットとして切り出す

## 報告の書き方

**表と数値がほしい。長い説明は要らない。**

1. 対応付けのカバー率（組・エピソードの両方で）
2. 物差し (a)(b) ごとの、外れる組数・消えるエピソード数・全体に対する割合
3. 指示文 40 種の内訳（外れる割合の高い順に 10 行と、低い順に 3 行）
4. 掃除前後の `meta/stats.json` の std の比（`action` 7 次元・`observation.state` 8 次元）
5. 学習 2 本の接触率・成功率・打ち切り率
6. **実測値を出したコマンドかファイルの場所**

## そのままでは使えなくなる設定ファイル（先に洗い出した）

| ファイル | 何が起きるか |
|---|---|
| `solution/train/jaist_dataset_build.sh` の `WANT_SRC="14347 2238036 40"` | 掃除後は本数が変わるので**必ずここで止まる** |
| `solution/train/jaist_launch.sh` の `stats.json.bak` 必須の確認 | 据え置きを打たない今回は `.bak` が無いので**起動前に止まる** |
| `solution/train/train_pi05.py` の `_DATASET_REPO` / `_DATASET_REVISION` | 掃除後のデータは Hub に無い。**`--dataset-repo` と `--dataset-root` を両方渡さないと、14,347 本ぶんの番号で選ばれて掃除が効かないまま学習が走る**（エラーにならない。[reference/25](../../reference/25_dataset_inventory.md) の 8.2） |
| `work/episode_lists/*.json`・`outputs/train2/*/episodes.json` | エピソード番号が振り直されるので、番号で指す指定は全部ずれる |
| `compe/val/EVAL_TASKS.csv`（36 件） | 「その学習が使わなかったエピソード」から作ってある（[reference/13](../../reference/13_holdout_episodes.md)）。母集団が変われば作り直し |
| `solution/train/verify_dataset.py` の**動画の確認（6 番）** | 中で `docker run` を打つので、このセッションでは通らない（145 行目）。**`--norm` と `--rev-tasks` は省けるので問題ない**（既定が `None`） |

## 判断が必要になったら

**この依頼が変える判断は 1 つである。**「Track2 の衝突を、学習データ側で減らせるか」。
**その目的に照らして、より良い測り方を思いついたら変えてよい。**変えた場合は理由を
報告に書くこと。

**ユーザに決めてもらうこと（4 節の 5 で止まったあと）。**

| # | 決めること | 材料 |
|---|---|---|
| 1 | **物差しは (a) 1 mm か (b) 触れたか** | 5 節で出す、それぞれの外れる割合 |
| 2 | **外す単位は組でよいか** | 1 組で平均 8.5 本が消える |
| 3 | **同じ大きさを無作為に抜いた対照を並べるか** | 消える本数が多ければ要る。GPU が 1 本ぶん増える |

## 関連

| 何 | どこ |
|---|---|
| 再現しないデモを外す話（**物差しが違う別の依頼**） | [019](019_forward_replay_failure_exclusion.md) |
| 学習データ 14,347 本の内訳 | [reference/06](../../reference/06_training_data_composition.md) |
| データセットの棚卸し（`libero_plus` と配布 tar の関係） | [reference/25](../../reference/25_dataset_inventory.md) |
| 正規化統計の据え置き | [reference/29](../../reference/29_training_data_requirements.md) の 6 節 |
| 衝突で落ちている本数の根拠 | [reference/19](../../reference/19_weights_catalog.md)・`talk/open_items.md` の `F60` |

---

# 報告（2026-09-09・やること 1〜5 まで）

> **やること 5 の「ここで止めて報告する」に到達した。**判断が要る 3 件は末尾に置いた。

## 1. 対応づけは 100% 通った

| | 値 |
|---|---|
| 配布データ | 14,347 本 → **1,681 組** |
| 本家から抜いたデモ | 2,000（40 課題） |
| **対応が付いた組** | **1,681 / 1,681（100.0%）** |
| **対応が付いたエピソード** | **14,347 / 14,347（100.0%）** |
| 行動列は一致したが指示文が食い違った組 | **0** |
| 本家にあって配布に使われていないデモ | 319 |

**再生は 1 回もしていない。**打ったのは次の 3 つで、道具はすべて
`solution/tools/contact_clean/` に置いた。

```bash
.venv/bin/python solution/tools/contact_clean/fetch_libero_states.py \
    --out work/artifacts/libero_states
.venv/bin/python solution/tools/contact_clean/map_demos.py \
    --dataset work/artifacts/libero_plus --states work/artifacts/libero_states \
    --out work/artifacts/libero_states/demo_map.json
# 判定は parc2026_verify の中（docker cp は通らないので tar を標準入力へ流し込む）
docker exec parc2026_verify python /tmp/contact_clean/score_contact.py \
    --states /tmp/contact_clean/libero_states \
    --map /tmp/contact_clean/libero_states/demo_map.json \
    --bddl-root /workspace/LIBERO-plus/libero/libero/bddl_files \
    --out /tmp/contact_clean/contact_scores.json
```

**状態の次元は 40 課題すべてで env と一致した**（`1 + nq + nv`。79〜110 次元）。
**判定にかかったのは 40 課題で 2 分 27 秒**である。

## 2. 基準にするコマは 5 以降ならどこでも同じ

物体は少し浮いた位置に置かれていて最初の数コマで落ちる。**0 コマ目を基準にすると
その落下を接触と数えてしまう**（002 が空撃ちで見た 111 mm と同じ現象）。

| 基準コマ | 1 mm を超えた組 | 消えるエピソード | 割合 |
|---|---|---|---|
| 0 | 786 | 6,866 | 47.9% |
| **5** | 258 | 2,151 | 15.0% |
| **10** | **256** | **2,131** | **14.9%** |
| **20** | 256 | 2,131 | 14.9% |

**5 以降は動かないので、基準は 10 コマ目に決めた。**

## 3. 2 つの物差しは、ほぼ同じものを指した

| 物差し | 組 | エピソード | 割合 |
|---|---|---|---|
| **(a) 対象外の物体が 1 mm を超えて動いた** | 256 | 2,131 | 14.9% |
| **(b) 腕か対象物が、対象外の物体に触れた** | 245 | 2,036 | 14.2% |
| (a) か (b) のどちらか | **273** | **2,266** | **15.8%** |

**両方に当たる組が 228。(a) だけが 28、(b) だけが 17。**
**どちらを採ってもほぼ同じ集合が落ちる。**

## 4. **1 mm という境目は、谷底に置かれている**

いちばん動いた量の分布（基準 10 コマ目・3 軸の絶対値の和・mm）。

| 50%点 | 75%点 | 85%点 | 90%点 | 95%点 | 99%点 | 最大 |
|---|---|---|---|---|---|---|
| 0.00 | 0.01 | 1.13 | 5.16 | 19.78 | 60.72 | 245.47 |

**4 分の 3 のデモは 0.01 mm しか動かしていない。**触っていないということである。
**境目を動かしても外れる本数はさほど変わらない。**

| 境目 | 0.5 mm | **1 mm** | 2 mm | 5 mm | 10 mm | 50 mm |
|---|---|---|---|---|---|---|
| 外れるエピソード | 15.3% | **14.9%** | 13.2% | 9.7% | 6.7% | 1.2% |

## 5. **外れ方は課題に強く偏っている**——2 つの指示文は丸ごと消える

40 指示文のうち **17 は 1 本も外れない。**一方で上位 5 つに全体の 56.3% が集まる。

| 指示文 | 対象外の数 | 外れる組 | 外れる本 | 割合 |
|---|---|---|---|---|
| pick up the black bowl **on the cookie box** and place it on the plate | 3 | 44/44 | 385/385 | **100.0%** |
| pick up the black bowl **on the ramekin** and place it on the plate | 3 | 39/39 | 295/295 | **100.0%** |
| **push the plate** to the front of the stove | 3 | 31/36 | 201/241 | 83.4% |
| pick up the black bowl **next to the cookie box** and place it on the plate | 3 | 30/47 | 285/445 | 64.0% |
| open the top drawer and put the bowl inside | 3 | 17/32 | 110/200 | 55.0% |
| put both the cream cheese box and the butter in the basket | 5 | 22/49 | 215/480 | 44.8% |

**上の 2 つは「掴む相手が対象外の物体の上に載っている」課題である。**

```
pick up the black bowl on the ramekin …
  対象   : akita_black_bowl_1, plate_1
  対象外 : akita_black_bowl_2, cookies_1, glazed_rim_porcelain_ramekin_1
```

**黒い椀はラメキンの上に載っているので、持ち上げれば必ずラメキンが動く。**
**採点の規則どおりに読むと、この課題は成功しようがない。**
`push the plate` も、皿を机の上で滑らせる課題なので同じ筋である。

**動かされている物体の上位。** `cookies_1` 66 組・`cream_cheese_1` 51 組・
`glazed_rim_porcelain_ramekin_1` 42 組・`milk_1` 21 組・`akita_black_bowl_2` 17 組。

### 5.1 **これは `F5`（Track2 で落ちる 2 本）の片方の答えでもある**

`compe/val/TRACK2_VAL72.csv` に入っている 2 課題を照らすと、こうなる。

| 評価に入っている課題 | 手本のうち対象外を動かした組 |
|---|---|
| `pick_up_the_black_bowl_on_the_cookie_box_…` | **44 / 44（配布 385 本すべて）** |
| `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_…` | **0 / 43** |

**`cookie box` は方策の出来と関係なく落ちる。**人が操作した手本ですら 1 本残らず
`cookies_1` を動かしているので、**採点の規則どおりなら成功しようがない。**
**`ramekin` の側は別の理由なので、動画で見るならそちらだけでよい。**

## 6. 外し方は 2 通りある

| | 全部外す | **割合 5 割超の 5 指示文は掃除の対象にしない** |
|---|---|---|
| 消えるエピソード | **2,266（15.8%）** | **990（6.9%）** |
| 残るエピソード | 12,081 | 13,357 |
| **残る指示文** | **38**（2 つが丸ごと消える） | **40** |

**「全部外す」は指示文を 2 つ失う。**1 課題ぶんの学習データが丸ごと無くなる形になる。

## 7. 正規化統計は動くが、耐えられた幅の内側である

掃除前の全 14,347 本を 1.000 としたときの std の比。

| | action でいちばん動く | state でいちばん動く |
|---|---|---|
| 全部外す | **4.7%** | **6.5%** |
| 5 指示文は残す | 2.3% | 2.9% |

**どちらも過去に耐えられた ±8% の内側である**（[reference/29](../../reference/29_training_data_requirements.md) の 6.3）。
**据え置きは要らないと見ている。**ただし比較の 2 本で尺度が違うことに変わりはないので、
そこが気になるなら両方を掃除前の統計に揃える手はある。

## 8. 判断が要る 3 件

| # | 決めること | 材料 |
|---|---|---|
| **1** | **物差しは (a) か (b) か「どちらか」か** | 3 節。**どれを採ってもほぼ同じ集合**なので、いちばん素直な「どちらか」を勧める |
| **2** | **指示文が 2 つ丸ごと消えるのを許すか** | 6 節。**許さないなら消えるのは 6.9%** で、対照（同じ大きさの無作為抜き取り）を置く値打ちは下がる |
| **3** | **対照を並べて回すか** | 消えるのが 6.9% なら、量の効きは小さいと見てよい。15.8% なら置いたほうがよい |

## 9. まだやっていないこと（やること 6〜8）

- `npz_to_lerobot.py subset` に「残す番号の一覧を渡す口」を足す
- 掃除済みデータセットを書き出して `verify_dataset.py` を通す
- `lerobot/pi05_libero_base` から 24,000 歩 × 2 本（**1 本 約 9 時間 15 分**）

---

## 10. 追記（9/9）——物差しを **(a) 1 mm** に決めた場合の数字

**ユーザが物差しを (a) に決めた。**そのうえで「指示文そのものが消えてしまう課題
だけを掃除の対象から外す」とどうなるかを数え直した。

**まず訂正が 1 つ。**「丸ごと消える指示文は 2 つ」と書いたのは
**「(a) か (b) のどちらか」で外した場合**の数である。
**物差しを (a) だけにすると、丸ごと消えるのは 1 つになる。**

| 指示文 | (a) で外れる組 |
|---|---|
| pick up the black bowl **on the ramekin** and place it on the plate | **39 / 39 → 丸ごと消える** |
| pick up the black bowl **on the cookie box** and place it on the plate | 38 / 44（6 組 50 本が残る） |

### 外し方 2 通りの比較（物差しは (a)・基準コマ 10・境目 1 mm）

| | 全部外す | **`on the ramekin` だけ免除** |
|---|---|---|
| 消える組 | 256 | **217** |
| **消えるエピソード** | **2,131（14.9%）** | **1,836（12.8%）** |
| 残るエピソード | 12,216 | **12,511** |
| **残る指示文** | **39 / 40** | **40 / 40** |

### ただし、指示文 2 つが極端に痩せる

免除したうえで掃除すると、残る本数はこうなる。

| 指示文 | 掃除前 | **掃除後** |
|---|---|---|
| push the plate to the front of the stove | 241 | **40** |
| pick up the black bowl on the cookie box and place it on the plate | 385 | **50** |
| open the top drawer and put the bowl inside | 200 | 105 |
| （参考）いちばん多い指示文 | 500 | 500 |

**40 本と 50 本は、他の指示文の 10 分の 1 である。**
**1 課題あたりの学習量が薄まると点が落ちることは、9/5 の 13.0.0 で実測されている**
（40 課題 → 111 課題で 3 トラックとも下がった。[260906_briefing.md](../260906_briefing.md) の 2 節）。
**「消えはしないが痩せる」課題をどう扱うかは、まだ決めていない。**

**選べる手は 3 つある。**

| 手 | 中身 |
|---|---|
| **そのまま痩せさせる** | いちばん素直。**痩せた 2 課題が落ちたら、それはそれで測れる** |
| **残り本数に下限を置く**（例: 100 本） | 下限を割る指示文は掃除の対象から外す。**免除が 3 つに増える**（`on the ramekin`・`push the plate`・`on the cookie box`）→ 消えるのは 1,435 本（10.0%） |
| **痩せたぶんをコピーで埋める** | 残った清潔なデモを複製して本数を戻す。**同じ手本を何度も見せることになる** |

### この道は他の参加者も採っている

`docs/materials/slack_talks/cleaning_dirty_training_data.md` に、
**「replay して衝突したかどうか、何ミリ動いたかを分類してデータセットを作っている」**
という発言がある。**やっていることは同じで、こちらは replay ではなく記録された
`states` を読んでいるぶん、再生の誤りが入らない。**

---

## 11. 追記（9/9）——痩せる 2 課題の「典型的な失敗」を動画にした

**数字だけでは何が起きているのか読めない**ので、記録された `states` をそのまま
描画した。**物理は 1 歩も進めていない**ので、映っているのは記録そのものである。
道具は [`render_contact_videos.py`](../../../solution/tools/contact_clean/render_contact_videos.py)。

**選んだのは 1 課題につき 3 本。**外れた組のうち**ずれが中央のもの**（`typical`）と
**いちばん大きいもの**（`worst`）、それに**外れなかった組**（`kept`）である。
**`kept` を入れたのは、掃除して残る側がどう見えるかが判断の材料だからである。**

置き場は `work/artifacts/contact_videos/`（8 本・384×384・20 fps）。
**画面の下に「いちばん動く対象外の物体の名前」と「基準コマからのずれ（mm）」を
焼いてある。**1 mm を超えると赤字で `OVER 1mm` が出る。

| 課題 | 種別 | デモ | いちばん動く物体 | ずれ |
|---|---|---|---|---|
| push the plate to the front of the stove | typical | `demo_27` | `cream_cheese_1` | **38.92 mm** |
| 〃 | worst | `demo_0` | `cream_cheese_1` | **88.51 mm** |
| 〃 | **kept** | `demo_2` | `akita_black_bowl_1` | 0.00 mm |
| pick up the black bowl **on the cookie box** … | typical | `demo_21` | `cookies_1` | **2.29 mm** |
| 〃 | worst | `demo_37` | `cookies_1` | **7.88 mm** |
| 〃 | **kept** | `demo_0` | `cookies_1` | 0.12 mm |
| （参考）pick up the black bowl **on the ramekin** … | typical | `demo_6` | `glazed_rim_porcelain_ramekin_1` | **10.99 mm** |
| 〃 | worst | `demo_14` | 〃 | **245.47 mm** |

### 見えたこと

| 課題 | 何が起きているか |
|---|---|
| **push the plate** | 皿を机の上で**手前から奥（コンロの前）へ滑らせる**課題である。皿の進む先に**青い箱（`cream_cheese_1`）が置かれていて、皿がそれを押す。**押した量が 39〜89 mm と大きい |
| **on the cookie box** | 黒い椀が**クッキー箱の上に載っている。**持ち上げるときに箱がわずかに動く。**量は 2〜8 mm と小さい**——「ぶつけた」というより「載っていたものを外した」ぶんである |
| （参考）**on the ramekin** | 椀がラメキンの上に載っていて、**持ち上げるとラメキンごと持ち上がる回がある**（最悪の回は 245 mm）。この課題が 39/39 で全滅するのはこれが理由 |

### この 3 つは性質が違う

- **`push the plate` は避けられる。**皿の通り道に箱があるかどうかは初期配置しだいで、
  **40 本（16.6%）は 1 mm も動かさずに終えている。**掃除して残る側は「箱を避けて
  押す手本」なので、**教える中身としては正しい。**
  **ずれの分布が二山になっているのが証拠である**——36 組のうち 5 組は **0.00 mm**
  ちょうどで、残りは **9.4 mm 以上**に跳ぶ。境目のあたりに 1 組も無い。
- **`on the cookie box` と `on the ramekin` は避けられない。**掴む相手が対象外の
  物体の上に載っているので、**持ち上げれば必ず動く。**
  `on the cookie box` で残る 6 組は、**たまたま 1 mm に届かなかった回**である
  （0.12〜1.0 mm）。**きれいな手本ではなく、境目の下に落ちた回**である。

---

## 12. 方針が決まった（9/9）——掃除済みデータセットを作った

### 12.1 決まったこと

| # | 決めたこと | 中身 |
|---|---|---|
| 1 | **物差しは (a) 1 mm** | 対象外の物体が 1 mm を超えて動いた組を外す。基準は 10 コマ目 |
| 2 | **痩せる 2 課題は痩せさせる。ただし残ったぶんを 3 倍にする** | `push the plate` 40 → **120 本**、`on the cookie box` 50 → **150 本** |
| 3 | **丸ごと消える 1 課題は消す** | `on the ramekin`（295 本）。**本選 1 では手当てしない。**[027](027_ramekin_task_unwinnable.md) へ切り出して**本選 2 か 3 で対応する** |
| 4 | 対照（同じ大きさの無作為抜き取り） | **保留。**まず掃除済みで 1 本回してから考える |

**「全部外す」と「免除する」で迷っていた点は、3 番目の道で片が付いた**——
**避けられる失敗（`push the plate`）は掃除し、避けられない課題（`on the ramekin`）は
課題ごと落とす。**中間の `on the cookie box` は掃除して 3 倍に戻す。

### 12.2 出来上がったもの

**`work/artifacts/libero_plus_clean`**（実体は `~/.cache/huggingface/parc_work/` にあり、
`work/artifacts/` からは symlink で見える。理由は 12.4）。

| | 掃除前 | **掃除後** |
|---|---|---|
| エピソード | 14,347 | **12,396** |
| コマ | 2,238,036 | **1,940,083** |
| 指示文 | 40 | **39** |
| 実体のエピソード | 14,347 | 12,216（**コピーで足した 180 本**を含めて 12,396） |

**確かめたこと。**

| 見たもの | 結果 |
|---|---|
| `verify_dataset.py`（動画の復号を除く 6 項目） | **すべて通った** |
| 元データと突き合わせ（無作為 6 本 × 3 コマ） | `observation.state`・`action`・**画 2 台とも最大差 0**。指示文も一致 |
| 動画の復号（無作為 300 本 × 3 コマ ＝ 900 コマ） | **落ちたもの 0** |
| 3 倍にしたぶん | 同じ動画区間を指す本が**ちょうど 3 本ずつ 90 組** |
| `on the ramekin` | **0 本**（消えている） |
| 正規化統計の動き | action の std が最大 **3.8%**・state が最大 **5.5%**。耐えられた ±8% の内側 |

### 12.3 打ったコマンド

```bash
# 残す番号の一覧を作る（掃除後の全体と、3 倍にする 2 課題ぶん）
.venv/bin/python solution/tools/contact_clean/summarize_contact.py \
    --scores work/artifacts/libero_states/contact_scores.json \
    --baseline 10 --rule displacement \
    --keep-out work/artifacts/libero_states/keep_cleaned.json \
    --boost-instruction "push the plate to the front of the stove" \
    --boost-instruction "pick up the black bowl on the cookie box and place it on the plate" \
    --boost-out work/artifacts/libero_states/keep_boost.json

# データセットを組み直す（--keep-file を 3 回渡すと、2 回目以降がコピーになる）
.venv/bin/python solution/tools/contact_clean/rebuild_dataset.py \
    --root ~/.cache/huggingface/parc_work/libero_plus \
    --output ~/.cache/huggingface/parc_work/libero_plus_clean \
    --keep-file work/artifacts/libero_states/keep_cleaned.json \
    --keep-file work/artifacts/libero_states/keep_boost.json \
    --keep-file work/artifacts/libero_states/keep_boost.json

.venv/bin/python solution/train/verify_dataset.py work/artifacts/libero_plus_clean \
    --expect "12396 1940083 39" --skip-video
```

### 12.4 踏んだ穴 3 つ

| 穴 | 何が起きたか |
|---|---|
| **lerobot の `split_dataset` は動画を AV1 で再エンコードし直す** | `dataset_tools._keep_episodes_from_video_with_av`。**90 本で数十分**、12,216 本なら 100 時間を超える。**そこで [`rebuild_dataset.py`](../../../solution/tools/contact_clean/rebuild_dataset.py) を書いた**——LeRobot v3 は 1 本の mp4 に何十本も詰めて開始時刻で切り出す形なので（`dataset_reader` の `shifted_query_ts = from_timestamp + ts`）、**動画はハードリンクで置き、メタデータだけ組み直せばよい。**ディスクも増えない |
| **`meta/tasks.parquet` は指示文を pandas の索引として書く** | 列として書くと `load_tasks` が引けず、学習データの `task` が数字のまま出る（`io_utils.load_tasks` が `tasks.index.name = "task"` を当てている） |
| **`docker run` も `docker cp` もこのセッションでは通らない** | 使えるのは `docker exec` だけ。データは `tar` を標準入力へ流し込んで運び、**元データは `~/.cache/huggingface` へハードリンクして** `quant_train` の `/hf_home` から見えるようにした。**`work/artifacts/libero_plus` は symlink になっている** |

### 12.5 次にやること

**[8 節（やること 8）](#8-学習を-2-本回して比べる)の学習に進める。
ただし回すのは 1 本でよい——掃除前の 1 本は 8/14 に済んでいる（13 節）。**

| | 中身 |
|---|---|
| 出発点 | `lerobot/pi05_libero_base` |
| 歩数 | 24,000・action expert だけ（693M）・**1 本 約 9 時間 15 分** |
| 変えるもの | **データだけ。**掃除前（`work/artifacts/libero_plus`・14,347 本）と掃除後（`libero_plus_clean`・12,396 本） |
| 見るもの | **接触率** |

**`jaist_launch.sh` は `stats.json.bak` が無いと起動前に止まる。**据え置きは
打たない方針なので、**その確認を外すか、`--no-freeze-check` のような口を足す。**

---

## 13. 追記（9/9）——**掃除前の 1 本はもう存在する。回すのは 1 本でよい**

**「24,000 歩を 2 本」と書いていたが、片方は 8/14 に回してある。**
`pi05_full_local`（[reference/19](../../reference/19_weights_catalog.md) の学習の記録）である。

| | `pi05_full_local`（8/14・**掃除前**） | **これから回す 1 本（掃除後）** |
|---|---|---|
| 出発点 | `lerobot/pi05_libero_base` rev `a217bfd3` | **同じ** |
| データ | `lerobot/libero_plus`（**14,347 本すべて**。`episodes: None`） | **`libero_plus_clean`（12,396 本）** |
| 歩数 | 24,000 | 同じ |
| batch | 16 | 同じ |
| 学習率 | **2.5e-5 → 2.5e-6** | 同じ |
| 動かす範囲 | action expert だけ（693M） | 同じ |
| カメラ拡張 | **なし**（この旗は 8/26 に足したもので、8/14 の実行には入っていない） | **なし** |
| 機械 | JAIST H100-20C | 同じ |
| 所要 | **9 時間 31 分** | 同程度 |
| 重み | `work/artifacts/model_weights_pi05_024000`・HF `siruku6/pi05_stage1_24000` | — |

**出典は `work/train/pi05_full_local.log` の 1〜2 行目**（`lerobot-train` の
コマンドがそのまま残っている）。**`--dataset.image_transforms` は付いておらず、
`episodes` は `None`** である。

**設定は `train_config.json` で 1 つずつ突き合わせた**
（`~/.cache/huggingface/hub/models--siruku6--pi05_stage1_24000/snapshots/*/train_config.json`。
**重み本体は落ちていないが、この 1 ファイルだけは手元にある**）。
`seed 42`・`n_action_steps 10`・`empty_cameras 1`・`image_transforms.enable false`
まで一致を確認した。

### 打つコマンド —— **夜中に回す**（9/9・ユーザの判断）

**GPU の順番は「[reference/32](../../reference/32_current_tasks.md) の D（Track3 の
v2 学習・3〜4 時間）が先、こちらは夜中」である。**1 枚を取り合うため。

```bash
cd ~/src/PARC2026_pre
RUN=pi05_clean24k WEIGHTS=base \
  DATASET=$HOME/.cache/huggingface/parc_work/libero_plus_clean \
  DATASET_REPO_ID=local/libero_plus_clean \
  STEPS=24000 LR=2.5e-5 SAVE_FREQ=8000 CAMERA_AUGMENT=0 FREEZE_CHECK=0 \
  setsid nohup bash solution/train/jaist_launch.sh \
  < /dev/null > ~/launch_pi05_clean24k.log 2>&1 &
disown
```

**ssh を切っても続く形にしてある。**学習そのものは `docker run -d` なので元から
切れても走るが、**起こしたあとの確認（最初の 20 歩まで最大 30 分、前面で待つ）は
切れると出力ごと消える。**だから 3 つ付けた。

| 付けたもの | 何のためか |
|---|---|
| `setsid nohup … &` | ssh が切れたときの SIGHUP から外す |
| `< /dev/null` | 背面のまま端末から読もうとして止まるのを防ぐ |
| `> ~/launch_pi05_clean24k.log` | 確認の結果をファイルに残す |

**切ったあとの見かた。**

**学習ログはファイルにも出ている。**`train_pi05.py` が標準出力を二重化して
`$OUT/<RUN>.log` へ書く（`_log_path`）。**run のディレクトリの「中」ではなく「隣」に
置いてある**——中に置くと `lerobot-train` が「出力先が空でない」と言って始まらないため。
**コンテナを消しても残るので、`docker logs` よりこちらが確実である。**

```bash
tail -f work/train/pi05_clean24k.log | tr '\r' '\n'   # 進み具合（本命）
grep step: work/train/pi05_clean24k.log | tr '\r' '\n' | tail -3
cat ~/launch_pi05_clean24k.log                         # 起動時の確認（693422112 / 12396 本）
docker stop -t 60 parc-pi05_clean24k                   # 止める
```

**`tr '\r' '\n'` を挟むのは進捗バーのせいである。**tqdm が復帰文字で同じ行を
上書きするので、そのまま読むと 1 行に見える。

**`DATASET` は symlink ではなく実体を渡す。**`work/artifacts/libero_plus_clean` は
`~/.cache/huggingface/parc_work/` への symlink なので、docker の `-v` に渡すと
解決のされ方が版で変わりうる。

**先に空きを確かめること。**このコマンドは 3 個＋最終で約 36 GB 使う。
**D の学習（`SAVE_FREQ=2000` で 4 個 ＝ 36 GB）と同時には置けない**ので、
**D のチェックポイントを間引いてから起こす**（`solution/train/prune_checkpoints.sh`）。

**`jaist_launch.sh` に 3 つ足した**（9/9）。

| 渡すもの | 何をするか |
|---|---|
| `WEIGHTS=base` | `--init-from` を付けず、`train_pi05.py` の既定（`pi05_libero_base` rev `a217bfd3`）から始める |
| `CAMERA_AUGMENT=0` | `--camera-viewpoint-augment` を外す。**8/14 の 1 本と揃えるため** |
| `FREEZE_CHECK=0` | 「正規化が据え置かれているか」の確認を飛ばす。**base から始める学習には据え置く相手が無い**（ユーザ承認・9/9） |

**`SAVE_FREQ=8000` にするのは容量のためである。**1 個 約 9 GB で、空きは 52 GB。
8,000 なら 3 個＋最終で 36 GB に収まる。

### まだ足りないもの —— **掃除前の側の接触率を測っていない**

**比べるのは接触率だが、`pi05_full_local` ではそれを測っていない。**
**重みは HF `siruku6/pi05_stage1_24000` にあるので落として測り直す**（約 9 GB）。
**測り方は両方で揃える**——`object_disp.jsonl` を出す評価を同じ集合で回し、
`solution/tools/eval/` の集計で衝突率を出す。

---

# 14. 追記（9/10）——学習が終わった。**RTX6000 で測るための下ごしらえ**

## 14.1 出来た重み

**24,000 歩を 9 時間 14 分で走り切った**（終了コード 0）。

| | |
|---|---|
| 置き場所 | `work/train/pi05_clean24k/checkpoints/{008000,016000,024000}` |
| 大きさ | 1 個 8.8 GB（`pretrained_model` だけ。全部で 33 GB） |
| 最後の loss | **0.225**（`grdn` 0.887） |
| ログ | `work/train/pi05_clean24k.log` |

**`train_config.json` を読んで、対照（`pi05_full_local`）と揃っていることを確かめた。**

| | 値 | 対照と |
|---|---|---|
| 出発点 | `lerobot/pi05_libero_base` rev `a217bfd3` | 同じ |
| 歩数・batch・`seed` | 24,000・16・42 | 同じ |
| 学習率 | 2.5e-5 | 同じ |
| `n_action_steps`・`empty_cameras` | 10・1 | 同じ |
| `image_transforms.enable` | `false`（カメラ拡張なし） | 同じ |
| データ | **`local/libero_plus_clean`（12,396 本）** | **ここだけが違う**（14,347 本） |

## 14.2 対照は HF にあり、**`vlm/` まで入っている**

**`siruku6/pi05_stage1_24000`（公開・9.37 GB・14 個）**である。
`model.safetensors` と正規化の 4 ファイルに加えて `vlm/` の 4 ファイルが入っているので、
**落としてそのまま `WEIGHTS` に渡せる。**

## 14.3 JAIST 側でやること

**1. `vlm/` を隣に置く —— 済（9/10）。**
学習が出したチェックポイントには入っておらず、**無いと評価が 5 秒で落ちる**
（[howto/03](../../howto/03_gpu_eval.md) の RTX6000 の表）。`pi05_rp3000` から
**symlink ではなく copy** した。

```bash
cp -r work/artifacts/pi05_rp3000/vlm \
      work/train/pi05_clean24k/checkpoints/024000/pretrained_model/vlm
```

**2. HF へ上げる —— 未（ユーザの判断待ち）。**
RTX6000 へ渡す道はこれしかない（`pi05_t3rev_v2` も同じ道を通した）。

```bash
hf upload siruku6/pi05_clean24k \
    work/train/pi05_clean24k/checkpoints/024000/pretrained_model . \
    --repo-type model --private
```

**`hf` は `~/.local/bin/hf`（1.30.0）を使う。**リポジトリの `.venv/bin/hf` は
shebang が別の機械のパス（`/home/ubuntu/lab/...`）を指していて動かない。

## 14.4 RTX6000 側でやること

**機械は `eng003sasaki01`。**持ち込みは 8/26 に済んでいて、重み・`vendor/`・
方策イメージが揃っている（[howto/03](../../howto/03_gpu_eval.md)）。

| # | やること | 確かめ方 |
|---|---|---|
| 1 | `PARC2026_pre` を `git pull` する | **採点器の差し替えが古いと指標が 14 個しか出ない** |
| 2 | 隣に `b_PARC2026_final` があるか見る | 無ければ `KIT_ROOT=<実際の場所>` を渡す。**無いと `set -e` でいきなり終わる** |
| 3 | イメージ 2 つを確かめる | `docker image inspect parc2026-final parc2026-train-pi05`。サーバ版は初回に自動で派生する |
| 4 | `id -nG` に `docker` があるか見る | 無ければセッションが古い（[knowledge/16](../../knowledge/16_docker_group_needs_session_restart.md)） |
| 5 | 重みを 2 つ落とす | 下のコマンド。**合計 18.2 GB** |

```bash
cd ~/src/PARC2026_pre
hf download siruku6/pi05_clean24k     --local-dir work/artifacts/pi05_clean24k
hf download siruku6/pi05_stage1_24000 --local-dir work/artifacts/pi05_full_local
ls work/artifacts/pi05_clean24k/vlm work/artifacts/pi05_full_local/vlm   # 4 個ずつ
```

## 14.5 測り方 —— **2 本を同じ集合で回す**

**見るのは接触率である**（026 の「やること 8」）。**`TRACK2_VAL72.csv` で測る**——
12 課題 × 6 条件の 72 行で、[plan/16 の 3 節](../../plan/16_finals_initiative_priority.md)
が 8/28 に `pi05_rp3000` を同じ形で回している（216 本・64 分・合計点 0.6523・衝突率 0.2269）。
**だから 3 本目の比較相手が無料で付いてくる。**

```bash
cd ~/src/PARC2026_pre

# 掃除後（これから測る）
MUJOCO_GL=egl TASKS_CSV=compe/val/TRACK2_VAL72.csv TRACK_EPISODES=3 \
  WEIGHTS=$PWD/work/artifacts/pi05_clean24k LABEL=clean24k \
  bash solution/tools/eval/run_final_eval.sh track2

# 掃除前（対照。まだ 1 度も測っていない）
MUJOCO_GL=egl TASKS_CSV=compe/val/TRACK2_VAL72.csv TRACK_EPISODES=3 \
  WEIGHTS=$PWD/work/artifacts/pi05_full_local LABEL=full_local \
  bash solution/tools/eval/run_final_eval.sh track2
```

**推論の設定は何も渡さない。**`policy_server.py` の既定が `PARC_N_ACTION_STEPS=16`・
`PARC_EMA_ALPHA=0.3` で、8/28 の回と同じである（[policy_server.py:687](../../../solution/policy_server.py#L687)・
[:785](../../../solution/policy_server.py#L785)）。**渡すと逆にずれる。**

**同時に流さないこと。**方策サーバが 2 つ立って GPU を取り合う。

| | |
|---|---|
| 1 本あたり | **216 ロールアウト・約 64 分**（8/28 の実測） |
| 2 本で | **約 2 時間 10 分** |
| 出る場所 | `work/final_eval/<日時>_track2_<LABEL>/results/` |

## 14.6 出そろったら見るもの

```bash
python3 solution/tools/eval/summarize_object_disp.py work/final_eval/<...>_clean24k
python3 solution/tools/eval/compare_object_disp.py \
    work/final_eval/<...>_full_local work/final_eval/<...>_clean24k
python3 solution/tools/eval/compare_runs_by_task.py \
    work/final_eval/<...>_full_local work/final_eval/<...>_clean24k
```

**いちばん見たいのは `cookies_1` である。**8/28 に「取り戻せる 36 本のうち 20 本が
`cookies_1`」と出ており（[plan/16 の 3.2](../../plan/16_finals_initiative_priority.md)）、
**掃除で `on the cookie box` の 44 組すべてを外したのはまさにこの物体である**（026 の 5.1）。
**ここが動かなければ、掃除は接触率に効いていない。**
