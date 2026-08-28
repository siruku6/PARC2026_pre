# 引き継ぎ: π0.5の2段学習(2026-08-13 22:40 時点)

**この文書は会話をまたぐための覚書である。** 決めたこと・いま動いていること・
次に打つコマンドを、あとから読んで再開できる形で置く。
根拠は[docs/reference/15](../reference/archive/15_pi05_lr_schedule.md)(学習率)と
[docs/reference/16](../reference/16_weak3x_for_pi05.md)(データの選び方)にある。

## 1. 決めた方針——2段に分ける

**段1で全データに馴染ませ、段2でweak3xの選び方に絞って仕上げる。**

| | データ | 狙い |
|---|---|---|
| **段1(いま動いている)** | 全14,347本 | π0.5の出発点は素のLIBEROしか見ていないので、**libero_plusのばらつきに初めて馴染ませる** |
| **段2(やるか未定)** | weak3xと同じ4,609本 | 段1で馴染んだ重みに対して、**苦手3タスクの露出を増やす** |

**段1を挟むことが要点である。** weak3xをπ0.5にそのまま持ち込めない理由は、
出発点の重みが素のLIBEROでしか学習されていない(得意タスクを27%に間引くと、
まだ見ていない材料を7割捨てることになる)ことだった。
**段1が終われば、π0.5はSmolVLAの出発点と同じ立場になり、その反対理由が消える。**

**段1のチェックポイントが残るので保険が効く。** 段2が外れても段1に戻れる。
1本しか回せない条件でweak3xに賭けるより安全である。

## 2. いま動いていること(2026-08-13 22:40 実測)

| | 状態 |
|---|---|
| 学習コンテナ | `sad_sammet`(22:06 開始)。`python /opt/pi05/train_pi05.py --run-name pi05_full_local --steps 24000 --batch-size 16` |
| 進み | **1,260 / 24,000 ステップ**。`loss` 0.30前後、`lr` 2.5e-05(頂点に到達済み)、`smp/s` 12、`mem_gb` 16.27 |
| 終わる見込み | `24000×16÷12÷3600 ≈ 8.9時間` → **8/14 07:00 ごろ**(残り8時間44分と表示) |
| ログ | `work/train/pi05_full_local.log`(`tail -f`で追える)。`docker logs sad_sammet` の方は`--rm`で消える |
| 間引き | cronに登録済み。**5分ごと・5個残す**(下のディスクの注意) |
| チェックポイント | 1個 **11GB**(`pretrained_model` 8.8GB ＋ `training_state` 2.1GB)。`save_freq` 500 なので**約11分に1回**保存される |
| ディスク | 298GBのうち**81GB空き**。この学習が使っているのは22GB |

`lr` が既に2.5e-05なのは想定どおり([15](../reference/archive/15_pi05_lr_schedule.md)——
立ち上げは自動で800 ステップに縮み、そこが頂点)。

### ディスクの注意——cronは「5個残す」で登録されている

```
*/5 * * * * .../prune_checkpoints.sh .../work/train/pi05_full_local 5 --once --log .../prune.log
```

**5個 × 11GB = 55GB** で頭打ちになる(保存中の一瞬だけ6個=66GB)。
空き81GBなので**段1だけなら足りる**。

**ただし段2を始めると、段2の出力先にも別のcron行が要る。**
5個残しを2つ動かすと110GBになり**入らない**。段2を始めるときは、
**段1側のcron行を「2個残す」に落とすか、段1のディレクトリを別の場所へ退避すること。**

## 3. 段2を始めるのに足りないもの(実装済み。2026-08-13 22:50)

**両方とも書いて動作を確かめた。** ただし**イメージに入っていない**(3.3)。

### 3.1 `--init-from`——ローカルのチェックポイントから始める(実装した)

もともと`--policy.pretrained_path`は`_download_base_model()`(HFの
`lerobot/pi05_libero_base`)に固定されていた。`--resume`は**同じデータ・
同じステップ数の続き**しかできないので段2には使えない。そこで
[train_pi05.py](../../solution/train/train_pi05.py)に`--init-from`と
`_resolve_init_from()`を足した。

**中身の条件は整っていた。** 500 ステップのチェックポイントの`pretrained_model/`は
`config.json` ＋ `model.safetensors`(8.8GB) ＋ 正規化の統計を含む前後処理
(`policy_preprocessor*`・`policy_postprocessor*`)まで揃った
**そのまま`--policy.pretrained_path`に渡せる形**だった。

`checkpoints/<ステップ数>`(1階層浅い)を渡す間違いが起きやすいので、その場で
言い直すようにした。**4通り試して確かめた**(正しいパス・1階層浅い・
存在しないパス・`config.json`が無いディレクトリ)。

```
[train_pi05] --init-from に config.json と model.safetensors が無い: .../000500
  渡すのは checkpoints/<ステップ数>/pretrained_model である。
  もしかして: --init-from .../000500/pretrained_model
```

### 3.2 `--max-demo-steps`を単独で渡すと落ちるバグ(直した)

`train.py`の`_select_emphasized_episodes`が、`--emphasize-tasks`が空のときに
`0.0 / 0.0`を計算して`ZeroDivisionError`で止まっていた。

```python
share_now = weak_frames / (weak_frames + strong_frames)   # 強調が空なら 0.0
share_target = min(multiplier * share_now, 0.9)           # → 0.0
keep_ratio = weak_frames * (1.0 - share_target) / (share_target * strong_frames)
```

`if wanted:`で囲み、強調が無いときは`keep_ratio = 1.0`(間引かない)にした。
**`train.py`が本体なのでSmolVLA側にも同じ直しが効く**(pi05固有ではなかった)。

**直した本体を実際に呼んで確かめた**(2026-08-13)。

| 呼び方 | 結果 |
|---|---|
| `--max-demo-steps 300` だけ | **13,946本 / 2,096,321フレーム**(前は落ちていた) |
| 従来どおり苦手3タスクも付ける | **4,609本 / 699,411フレーム**(SmolVLAの記録と同じ＝壊していない) |

### 3.3 **イメージを作り直さないと段2で使えない**

Dockerfileが`COPY train.py train_pi05.py ... /opt/pi05/`でビルド時に焼き込むので、
**いま動いているイメージには`--init-from`が入っていない。**
動いている学習(段1)には影響しないが、**段2の前にどちらかが要る。**

| やり方 | コマンド |
|---|---|
| **A. 作り直す**(素直。段1と並行してやれる) | `docker build -t parc2026-train-pi05 -f solution/train/Dockerfile.pi05 solution/train` |
| **B. ホスト側の scripted policy を被せる**(作り直さない) | `docker run` に `-v "$PWD/solution/train:/opt/pi05_src:ro"` を足し、`python /opt/pi05_src/train_pi05.py ...` を呼ぶ |

Bが効くのは、`train_pi05.py`が`sys.path.insert(0, Path(__file__).resolve().parent)`で
**自分と同じディレクトリから`train.py`を輸入する**ためである(112〜122行)。
ディレクトリごと被せれば、直した2ファイルがまとめて差し替わる。
`/opt/pi05`ではなく別の名前に当てているのは、**イメージ側の元の scripted policy を残して
見比べられるようにするため**で、`/opt/pi05`に直接被せても動く。

**このマシンでは`docker build`の実行が許可されていないので、Claudeは
どちらも試していない。**

## 4. 段2を始めるときのコマンドと、2つの注意

**3.3のBを使う形で書く**(イメージを作り直した場合は`-v .../pi05_src` を外し、
`/opt/pi05_src/`を`/opt/pi05/`に読み替える)。

```bash
docker run -d --gpus all --rm -it \
    --user "$(id -u):$(id -g)" \
    -v "$PWD/work/train:/out" -v "$HOME/.cache/huggingface:/hf_home" \
    -v "$PWD/solution/train:/opt/pi05_src:ro" \
    parc2026-train-pi05 \
    python /opt/pi05_src/train_pi05.py --run-name pi05_weak3x_stage2 \
        --init-from /out/pi05_full_local/checkpoints/024000/pretrained_model \
        --steps 6000 --batch-size 16 --learning-rate 5e-6 \
        --max-demo-steps 300 \
        --emphasize-tasks \
          "put the bowl on top of the cabinet" \
          "put both the alphabet soup and the cream cheese box in the basket" \
          "pick up the black bowl next to the ramekin and place it on the plate"
```

- **`--init-from`はコンテナの中から見たパスで渡す。** `work/train`を`/out`に
  マウントしているので`/out/...`になる
- **`024000`は本当に残っているか確かめてから渡すこと。** 間引きのcronが
  新しい方から5個だけ残すので、最終歩の`024000`は残るが、
  **学習が完走しなければ番号が違う**(`ls work/train/pi05_full_local/checkpoints/`)
- **段2の出力先にもcronの行を足すこと**(下のディスクの注意)

### 注意1: 学習率を明示すること。既定のままだと10倍に焚き直される

**段1は2.5e-6で終わる。** 段2を既定のまま始めると頂点2.5e-5へ戻るので、
**終わった地点の10倍から再スタートする。段1で積んだものを崩しうるのはここである。**
`--learning-rate 5e-6`のように、終わった地点の2倍程度に抑える
(下限の`scheduler_decay_lr`は2.5e-6のままなので、5e-6→2.5e-6の緩やかな形になる)。

**形そのものは自動で縮む**([15](../reference/archive/15_pi05_lr_schedule.md))。
`--steps 6000`なら立ち上げ200 ステップ・6,000 ステップで下げきる。

### 注意2: 3,000 ステップでは差が測れない見込み

**「1コマを何回見るか」では見誤る。** 段2の狙いは苦手タスクの露出を増やすこと
なので、**苦手タスクに当たる引き回数**で比べる。段1(全部・24,000 ステップ)だけで
苦手3タスクには**約33,400回**当たる(苦手の割合8.7%)。段2はweak3xの4,609本で
苦手の割合が26.1%になるので:

| 段2のステップ数 | 苦手に当たる回数 | 段1比 | 所要 |
|---|---|---|---|
| 3,000 | 12,528 | +38% | 1.3h |
| **6,000** | **25,056** | **+75%** | **2.5h** |
| 10,000 | 41,760 | +125% | 4.2h |

**3,000 ステップだと段1の4割増しで、本番8本では差が測れない側に入りそうである。**
効き目を狙うなら6,000〜10,000 ステップ。

### `--max-demo-steps 300`はmokaのタスクを丸ごと消す

落ちるのは401本(2.8%)だけだが、`put both moka pots on the stove`が全滅する。
**手作業の手本を作っている相手そのもの**なので、手作業の分を混ぜる計画と
組み合わせると「実データのmokaが消えて手作業の分だけ残る」形になる。
狙うなら問題ないが、気づかずにそうなるのは避けたい。
外したいなら`--max-demo-steps`を付けない(3.2を直せば単独指定もできる)。

## 5. 時間の割り振り——本当の関門は推論側

締切は**8/14 23:59**(報告書は8/17)。

| | 所要 | 終わる時刻 |
|---|---|---|
| 段1(24,000 ステップ) | 8.9h | **8/14 07:00** |
| 段2(6,000 ステップ) | 2.5h | 8/14 09:30 |
| 残り | **約14h** | |

その残りで、π0.5用の`MyPolicy`を書き、**L4 24GB・10秒/推論**(規定PDF p.12)に
収まるか測り、zipにして、**さらに段1と段2のどちらを出すか評価で決める**必要がある。
**段2が意味を持つのは推論側が通ってからである。**

### 推論側は着手済み(2026-08-13 22:50時点)

**`solution/tools/viz/infer_pi05_once.py`(548行)がある**——学習途中の
チェックポイントで1手ぶんの行動が出るかを確かめる道具で、
(1)チェックポイントをそのまま読めるか (2)カメラ名が`front`/`wrist`へ
差し替わるか (3)ダミー観測から`(7,)`の有限な行動が出るか
(4)1推論が何秒か、の4点を見る。**まだgitに入っていない**(未追跡)。

**`solution/policy_server.py`はSmolVLAの提出物なので壊さないこと**——
π0.5用は別ファイルか、環境変数で分岐する形にする。

**段1の完走を待つ必要はない。** 推論時間の測定には**中身の良し悪しと関係なく
「π0.5の形をした重み」があれば足りる**ので、
`work/train/pi05_full_local_old/checkpoints/000500/pretrained_model/`(500 ステップ)で
進められる。

### gitに入っていない変更(2026-08-13 22:50)

| ファイル | 状態 |
|---|---|
| `solution/tools/viz/infer_pi05_once.py` | **未追跡。** 上記の推論確認の道具 |
| `solution/train/prune_checkpoints.sh` | **未コミット**(117行追加)。`--once`・`--log`・`--dry-run`・`flock`を足した版。**[docs/howto/07](../howto/07_train_pi05_local.md)がこの版を前提に書かれており、cronもこの版を呼んでいる**ので、入れ忘れると説明と実物がずれる |
| `compe/val/KNOB_TASKS.csv` | 未追跡。従来どおり意図的に外している |

### 段2をやるかの判断は 8/14 07:00 に

**推論側が通っているかを見てから決める。** 通っていなければ段2は捨てて、
段1のチェックポイントで提出まで持っていくことに時間を使う。

## 6. まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| 2段に分けた方が、段1だけより本番スコアで勝るか | 両方を評価して比べる以外に方法が無い。**時間が足りなければ測れない** |
| 段2の学習率5e-6が適切か | 根拠は「段1が終わる2.5e-6の2倍」だけで、測っていない |
| `--init-from`で渡したチェックポイントを`lerobot-train`が本当に読めるか | 実装したら`--smoke-test`相当(数十歩)で1度通す |
| π0.5にとっての苦手タスクは何か(強調3タスクはSmolVLAの評価で決めたもの) | π0.5のチェックポイントで`work/gpu_eval`を回し、`[結果]`行で落ちたタスクを名指しする |
| L4 24GBでの1推論の時間 | `policy_server.py`のπ0.5版を書いてから実測 |
| 24,000 ステップを完走したときの実際の所要時間 | ログの経過時間を見る(`smp/s`は12で、見積もりの10.5より速い) |
