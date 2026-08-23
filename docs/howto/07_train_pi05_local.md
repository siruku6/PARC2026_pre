# 07. π0.5をこのマシン(H100 20GB)で学習する

SmolVLAの提出は保険としてそのまま残し、**並行してπ0.5が間に合うか試す**。
この文書は、そのローカル側(H100 20GB 1枚)の本番学習を実際に回す手順である。
出発点の調査は[docs/reference/14_pi05_switch_feasibility.md](../reference/14_pi05_switch_feasibility.md)、
学習スクリプト自体は[solution/train/train_pi05.py](../../solution/train/train_pi05.py)を参照。

**RunPodを借りてフルパラメータ学習を試す線は[docs/howto/06](06_runpod_pi05.md)に
分けてある。この文書とは独立で、互いに関係しない。**

## 現在地(2026-08-13 時点)

| 済んでいること | 中身 |
|---|---|
| 学習コンテナが動く状態 | `parc2026-train-pi05`(=[Dockerfile.pi05](../../solution/train/Dockerfile.pi05))。権限まわりの4つの失敗は解消済み(下の「詰まりやすい点」) |
| gemmaライセンス同意と`hf auth login` | 済み。`~/.cache/huggingface`の所有者もこのユーザーに直した |
| smoke test(20 ステップ) | `End of training`まで通った |
| batch sizeの決定 | **16**(下の表の実測から) |
| チェックポイントの間引き | [prune_checkpoints.sh](../../solution/train/prune_checkpoints.sh)。放っておける形に作り直して実測済み。**cronに登録済み**(2026-08-13 22:06。`*/5 * * * *`・**5個残す**。下の例文の「1分ごと・2個」とは違う) |
| 本番学習(段1) | **回っている**(22:06 開始、`--steps 24000 --batch-size 16`)。`smp/s` 12 で**8/14 07:00 ごろ完走見込み** |
| チェックポイントがそのまま推論に使えること | **確認済み**(2026-08-13 23:00、500 ステップのチェックポイントで実測)。[infer_pi05_once.py](../../solution/tools/viz/infer_pi05_once.py)が重み813個を1つ残さず読み、`(7,)`の有限な行動を出した。マージ手順は要らない |
| カメラ名が`front`/`wrist`へ差し替わること | **確認済み**。出典はチェックポイント自身の`config.json`の`input_features`(`observation.images.front`/`wrist`、各[3,256,256]と`empty_camera_0`[3,224,224])。同じ値は上の実測の出力にも出る |

| 次にやること | 中身 |
|---|---|
| 1. ~~本番学習を始める~~ → **推論側を先に進める** | 段1が回っている間にやること。**500 ステップのチェックポイント(`work/train/pi05_full_local_old/checkpoints/000500/pretrained_model`)で足りる**——推論時間の測定は中身の良し悪しと無関係。下の3番と同じ作業 |
| 2. 手作業で作った画像を混ぜる | 効果が高いと分かっている追加の一手。`--dataset-root`で混ぜたデータセットを指す想定だが、**pi05側では未検証**(選択ロジックは`train.py`から再利用しているので通る見込み) |
| 3. 推論側を用意して実測する | `policy_server.py`にπ0.5用の`MyPolicy`を書き、**L4 24GB・10秒/推論**(規定PDF p.12)に収まるか測る。**SmolVLAの動いている提出物を壊さないよう、まだ触っていない** |
| 4. 提出 | 3が通ってスコアがSmolVLAを上回ったときだけ差し替える。上回らなければSmolVLAのままにする |
| 5. 段2(やるか未定) | 段1の重みを`--init-from`で読み直し、苦手タスクを厚くしたデータで6,000 ステップほど仕上げる。**やるかは段1が終わる8/14 07:00に、推論側が通っているかを見て決める** |

**段2の段取り・注意・時間の割り振りは
[docs/talk/handoff_pi05_stage2.md](../talk/handoff_pi05_stage2.md)にまとめてある。**
データの選び方の根拠は[docs/reference/16](../reference/16_weak3x_for_pi05.md)、
学習率は[docs/reference/15](../reference/15_pi05_lr_schedule.md)。


## batch sizeは16に決めた

学習スクリプト自身が出す`mem_gb`(GPUメモリのピーク使用量)を、batchを変えながら
20〜50 ステップだけ回して測った。

| batch | mem_gb | smp/s(1秒あたりのサンプル数) |
|---|---|---|
| 2 | 12.90 | 2〜4 |
| 8 | 13.99 | 9〜10 |
| 16 | 16.16〜16.27 | 10〜12 |

19.8GB中16GB強で、まだ3.5GBほど余裕がある。ただしbatch 8→16の増分(+2.3GB)は
batch 2→8の増分(+1.1GB)より大きく、このペースで24まで上げると18GB台に達し
OOMのリスクが増える一方、`smp/s`の伸びは頭打ちに近い(8→16で2割増程度)。
**伸びしろよりOOMのリスクの方が大きくなる境目**と判断し、**batch=16で確定**した。


## 本番の学習を始める

```bash
docker run -d --gpus all --rm -it \
    --user "$(id -u):$(id -g)" \
    -v "$PWD/work/train:/out" -v "$HOME/.cache/huggingface:/hf_home" \
    parc2026-train-pi05 \
    python /opt/pi05/train_pi05.py --run-name pi05_full_local --steps 24000 --batch-size 16
```

(イメージ名の後ろに書いた分は既定のCMDを丸ごと置き換えるので、`python ...`から
全部書く必要がある。スクリプトの置き場所が`/opt/pi05/`なのは、同じイメージを
RunPodでも使えるようにしてあるためである——[docs/howto/06](06_runpod_pi05.md)の3節。
ローカルで回すぶんには意味を持たない。)

24,000 ステップを選んだ理由は、`smp/s≈10.5`で見積もると`24000×16÷10.5÷3600≈10.2時間`と、
確保できる時間(10時間以上)に収まるためである。**ステップ数(`--steps`)は学習率の
下げ方(スケジュール)そのものを決める値なので、大きくしすぎて時間内に完走
できないと、学習率が下がりきらないまま終わってしまう**(SmolVLA側で同じ理由の
教訓がある: [docs/reference/11_training_schedule.md](../reference/11_training_schedule.md))。
確保できる時間が伸びるようなら、この`--steps`を上げてから最初から回し直すこと
(`--resume`ではステップ数を後から変えられない——`train.py`の`_build_resume_command`の
コメント参照)。

**学習を始めたら、続けて次の節の「間引き」も動かすこと。**


```bash
# 実測値
Training:   2%|██▌                       | 556/24000 [13:24<8:5
Training:   2%|██▌                       | 557/24000 [13:26<8:5
Training:   2%|██▌                       | 558/24000 [13:27<9:1
Training:   2%|██▌                       | 559/24000 [13:29<9:1
Training:   2%|██▌                       | 560/24000 [13:30<9:11:59,  1.41s/step]
```

## ログはどこに出るか

`train_pi05.py`が**自動で`work/train/<run-name>.log`に書き出す**(2026-08-13に
追加。`--log-file`で場所を変えられる、`--no-log`で切れる)。

```bash
tail -f work/train/pi05_full_local.log
```

```
[train_pi05] === 2026-08-13 22:10:03 開始 ===
[train_pi05] ログ: /out/pi05_full_local.log（`tail -f` で追える）
[train_pi05] GPU: NVIDIA H100-20C（19.8 GB）
[train_pi05] コマンド: lerobot-train --policy.type=pi05 ...
Training: 2%|██| 460/24000 [10:54<8:54:21] INFO step:460 loss:0.445 smp/s:12 mem_gb:16.27
```

- **`<run-name>`ディレクトリの「中」ではなく「隣」に置いている。** LeRobotは
  出力先ディレクトリが既にあると学習を始めずに落ちるため
  (`lerobot/configs/train.py`の`TrainPipelineConfig.validate`にある
  `if not self.resume and ... self.output_dir.is_dir(): raise FileExistsError`)、
  ログのためにディレクトリを先に作ると**それだけで学習が始まらなくなる**
- 進捗バーの書き換え(`\r`で上書きされる途中経過)はログに残さず、
  **20 ステップごと(`log_freq`)の1行だけ**を書く。端末側には素通しで流れるので
  進捗バーは普通に動く。素通しで保存すると24,000 ステップぶんで数十MBになる
- 追記なので、`--resume`で何度も再開しても前の分が消えない

### `-d`(バックグラウンド)で回したときの注意

`docker run -d`にすると端末には何も出ない。**ログファイルは上のとおり残るが、
それとは別に、dockerが持つログは`--rm`でコンテナが消えると一緒に消える。**

```bash
docker ps                                   # コンテナIDを調べる
docker logs -f <コンテナID>                  # dockerが持っている方を見る
docker logs --tail 2 <コンテナID> | tr '\r' '\n' | grep step:   # 今のステップ数だけ
```

(`tr '\r' '\n'`が要るのは、`docker logs`側は進捗バーの書き換えを全部持って
いて、`step:`の行がその後ろにくっついているためである。)

## 途中で止めて、回し直す

試運転を止めて本番として回し直すときの手順。**保存の直後に止めること**——
`docker stop`はSIGTERMを送って10秒待ち、返事が無ければSIGKILLで殺すので、
11GBを書いている途中に当たると**中途半端なチェックポイントが残る**。
ログに保存の行が出た直後を狙うか、`docker stop -t 60`で待ち時間を伸ばす。

```bash
docker ps                     # コンテナIDを確認
docker stop <コンテナID>       # --rm を付けているのでコンテナは消える
```

止めた時点の`checkpoints/`はそのまま残る。**ただし、残っていると次の学習が
始められない**——`train_pi05.py`が「出力先が既にある」で止める(LeRobot自身も
同じ理由で落ちる。上の「ログはどこに出るか」参照)。3つのどれかを選ぶ。

| やりたいこと | どうする |
|---|---|
| 途中から続ける | `--resume`を付けて同じ`--run-name`で実行する(ステップ数は変えられない) |
| 最初からやり直す。**前の重みは取っておきたい** | `mv work/train/pi05_full_local work/train/pi05_probe_step500`(ログも`mv work/train/pi05_full_local.log ...`)。名前を変えれば元の`--run-name`が空くので、そのまま回し直せる |
| 最初からやり直す。前の重みは要らない | `rm -rf work/train/pi05_full_local` |

**試運転で作った500 ステップのチェックポイントは、取っておくと使い道がある。**
スコアは出ないが、`policy_server.py`のπ0.5版を書いてL4での推論時間を測る
(次にやることの3番)には、**中身の良し悪しと関係なく「π0.5の形をした重み」が
あれば足りる**。本番の10時間を待たずに推論側の実装を進められる。

## チェックポイントを自動で間引く

`solution/train/prune_checkpoints.sh` が、`checkpoints/`直下の**数字だけのディレクトリ**を新しい方から指定数だけ残し、それより古いものを削除する。

### なぜ必要か(数字で)

`--steps 24000`・`--save-freq`の既定500だと、**保存は48回**起きる。1個11GBなので
**放っておくと528GB**になり、途中で必ずディスクが尽きる。最新2個だけ残せば
22GB前後で頭打ちになる(削除が追いつく前の一瞬だけ3個=33GB)。

保存の間隔は`500 ステップ×16÷10.5 smp/s ≈ 762秒`、つまり**約13分に1回**。
点検が1分おきでも十分間に合う。

### どのディレクトリを指定するのか

**引数に渡すのは`checkpoints`の1つ上、学習の`--run-name`のディレクトリである**
(スクリプトが中で`/checkpoints`を足して見に行く)。

| 渡すもの | 例 | スクリプトが実際に見る場所 |
|---|---|---|
| 第1引数: runディレクトリ | `work/train/pi05_full_local` | `work/train/pi05_full_local/checkpoints/` |
| 第2引数: 残す数(既定2) | `2` | 新しい方から2個を残し、それより古い**数字だけの名前**のディレクトリを消す |

`--run-name pi05_full_local`で学習し、`work/train`を`/out`にマウントしている
ので、ホストから見た場所は`work/train/pi05_full_local`になる。
**`checkpoints`まで含めて渡すと、`.../checkpoints/checkpoints`を探して
何もしないまま終わる**(エラーにもならないので気づきにくい)。

### 方法2: cronで1分ごとに1回だけ点検させる

常駐プロセスを持ちたくない場合(マシンを再起動しても効き続けてほしい場合)は、
`--once`を付けてcronから呼ぶ。**このマシンではcronが動いていることを確認済み**
(`pgrep -ax cron` → `/usr/sbin/cron -f -P`、`systemctl is-active cron` → `active`。
2026-08-13実測)。

#### 登録する

`crontab -e`(初回はエディタの選択を聞かれる)で次の1行を足して保存する。

```cron
* * * * * /home/ubuntu/lab/PARC2026_pre/solution/train/prune_checkpoints.sh /home/ubuntu/lab/PARC2026_pre/work/train/pi05_full_local 2 --once --log /home/ubuntu/lab/PARC2026_pre/work/train/pi05_full_local/prune.log
```

[cron を使用した時間指定](https://qiita.com/nemutas/items/3f5816eabbf0eda5e6a9)

#### 動いているか確かめる

```bash
crontab -l                                   # 登録された行を確認
sleep 70                                     # cronは1分刻みなので1分待つ
cat work/train/pi05_full_local/prune.log     # 消す物があれば行が増えている
```

**消す物が無い回は何も書かないので、ログが空のままでも異常ではない。**
cronがそもそも呼んだかどうかを直接見たいときは次を使う。

```bash
journalctl -u cron --since '-5 min' | grep prune_checkpoints
```

#### 解除する

```bash
crontab -l | grep -v prune_checkpoints.sh | crontab -   # この行だけ消す
crontab -l                                              # 消えたか確認
```

`crontab -e`で手で消してもよい。**`crontab -r`は登録を全部消してしまうので、
他の登録があるなら使わないこと。**

- **パスは全部絶対パスで書くこと。** cronの作業ディレクトリと`PATH`は
  ログイン時と違うため、相対パスだと動かない
- **`--log`も必ず付けること。** 付けないとcronが出力をメールで送ろうとして、
  このマシンには受け取る仕組みが無いため**ログがどこにも残らない**
- 前回の削除が1分以内に終わらず次の実行が重なっても、`flock`で二重に走らないよう
  弾いてある。方法1の常駐版と両方動かしても壊れない(後から来た方が黙って抜ける)
- **学習が終わったらcrontabの行を消すこと**(消し忘れると、次の別の学習でも
  勝手に間引かれる)


### 消される前に手で確かめたいとき

```bash
solution/train/prune_checkpoints.sh work/train/pi05_full_local 2 --once --dry-run
```

`--dry-run`は**消さずに、消す対象だけを表示する**。


### 間引きで気をつけていること

- **残す数は2個以上にする。** 1個にすると、保存中の1個を消してしまう競合と、
  LeRobotが張る`checkpoints/last`のリンク切れが起きうる(スクリプト側でも
  1未満は拒否し、1のときは警告を出す)
- 対象にするのは`checkpoints/`直下の**数字だけのディレクトリ**なので、
  `pretrained_model`や`last`(シンボリックリンク)には触らない
- 偽のチェックポイントを並べて、削除・二重起動の抑止・`checkpoints/`が
  まだ無い場合を通してある(2026-08-13実測)


## 詰まりやすい点(実機で実際に踏んだもの)

ローカルでこのDockerイメージを組み立てる過程で、4つの権限まわりの問題に
当たった。再発したときに同じ調査をしなくて済むよう記録しておく。

| 症状 | 原因 | 直し方 |
|---|---|---|
| `hf auth login`が`PermissionError`で落ちる | `~/.cache/huggingface`がroot所有(過去のroot実行の名残) | `sudo chown -R $(whoami):$(whoami) ~/.cache/huggingface` |
| `--user`を付けたら`/root/.cache/huggingface/token`で`Permission denied` | `/root`自体が`drwx------`で、非rootは中身の権限に関係なく通過できない | `HF_HOME`を`/root`配下ではない場所(`/hf_home`)にする。[Dockerfile.pi05](../../solution/train/Dockerfile.pi05)で対応済み |
| `getpass.getuser()`が`KeyError: getpwuid(): uid not found`で落ちる | `--user`のUIDが`/etc/passwd`に無い(torchの内部が呼んでいる) | `USER`/`LOGNAME`環境変数を設定しておく。同上のDockerfileで対応済み |
| チェックポイント保存の直前で`/out/<run-name>`に`Permission denied` | マウント先(`work/train`)がroot所有 | `sudo chown -R $(whoami):$(whoami) work/train` |

いずれも「過去にrootとして`docker run`した結果、ホスト側のディレクトリが
root所有のまま残る」という同じ根に行き着く。**`--user "$(id -u):$(id -g)"`を
最初から付けて回していれば起きなかった。**

もう1つ、引数の書き方でも1度詰まった。**`--steps`などスクリプトへの引数は、
必ずイメージ名より後ろに書く**(イメージ名より前はdockerへの引数として
解釈され、`unknown flag`で落ちる)。

## 学習の出発点(再確認するときの出典)

| 事柄 | 値 | 確かめ方 |
|---|---|---|
| 初期重み | `lerobot/pi05_libero_base` @ `a217bfd3b14673cf2ce597e69997ab21866438dd` | `curl -s https://huggingface.co/api/models/lerobot/pi05_libero_base` |
| 学習データ | `lerobot/libero_plus` @ `f3f49f426d75030177b18778374005bc12ccd588` | SmolVLA側(`train.py`)と同じもの |
| 正規化 | `{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}` | 既定のquantileだとq01/q99統計が無くて最初のバッチで落ちる |
| `chunk_size` / `n_action_steps` | 50 / 10 | `chunk_size`は学習後に変えてはいけない。`n_action_steps`は推論時に変えてよい([docs/reference/07](../reference/07_chunk_params.md)) |
| ライセンス | gemma(`google/paligemma-3b-pt-224`の規約に同意したアカウントで`hf auth login`) | 同意していないと初期重みの取得で止まる |

**値の根拠はすべて[docs/reference/14_pi05_switch_feasibility.md](../reference/14_pi05_switch_feasibility.md)に
出典付きで置いてある**(この文書の表は写しなので、食い違ったら14の方を信じること)。

## 学習が終わったら

`work/train/pi05_full_local/checkpoints/<最終ステップ>/pretrained_model/`に
提出可能な形の重みが残る(LoRAではなくフルパラメータ学習なので、`merge.py`に
相当するマージ手順は不要——そのまま`PI05Policy.from_pretrained()`で読める)。

次はこの重みを使ってpolicy_server.py側にπ0.5用の`MyPolicy`実装を用意し、
L4 24GB・10秒/推論の制約に収まるかを実測する段階に進む。

## 学習中の重みで推論を試す(GPUを使わずに)

学習が10時間走っている間、GPUは1枚しかないので推論には使えない。**CPUだけで
確かめられることは2つある。**どちらも学習に使ったのと同じイメージの中で動かす
(ホストにはtorchもlerobotも無い)。`--memory`を必ず付けて、はみ出したときに
死ぬのを推論側に限定すること——**付けないとOOM killerがいちばん大きいプロセス、
つまり学習本体を狙いうる。**

| 道具 | 何が分かるか | かかる時間(CPU 1コア実測) |
|---|---|---|
| [infer_pi05_once.py](../../solution/tools/viz/infer_pi05_once.py) | 重みを読めるか・カメラ名・1推論の秒数。観測は真っ黒なダミー | 読み込み302〜352秒＋1推論195〜209秒(学習と同時に回した2回の実測) |
| [replay_pi05_on_dataset.py](../../solution/tools/viz/replay_pi05_on_dataset.py) | **実データのフレームを流したときに何をしようとしているか。**正解の行動と並べた動画とcsvが出る | 上に同じ(`--mode chunk`なら推論は1回) |

```bash
# 実データ1フレームから先50 ステップを予測し、正解と並べた動画にする(推論1回・約9分)
docker run --rm -it --memory=13g --cpus=1 \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:/repo" -v "$HOME/.cache/huggingface:/hf_home" \
    parc2026-train-pi05 \
    python /repo/solution/tools/viz/replay_pi05_on_dataset.py \
        /repo/work/train/pi05_full_local_old/checkpoints/000500/pretrained_model \
        --task "pick up the black bowl" --output /repo/work/logs/replay
```

- **ピークは12.25GBだった**(500 ステップのチェックポイント・`--load-mode lowmem`)。
  `--memory=12g`では限界に張り付くので**13gにする**
- `--cpus=1`は学習を守るため。外すと推論は速くなるが、コアが2つしかないので
  学習が遅くなる
- **シミュレータは回さない。** 入力は常に学習データの実フレームなので、
  「その時点の判断が合っているか」だけが出る。誤差が積み上がって別の場所へ
  行く分は含まれない(ロールアウトの動画を撮るのは
  [rollout_recorder.py](../../solution/tools/eval/rollout_recorder.py)の役目で、
  そちらは評価コンテナとGPUが要る)
- `--dry-run`を付けると方策を読まずに動画の配管だけ試せる(数秒)
- **`--mode chunk`では、本番が実際に使う先頭10 ステップと、11 ステップ目以降(作り置きの外挿)を
  分けて数える。** 混ぜると外挿の崩れに引きずられて、実際より悪く見える
- 素のLIBERO(`lerobot/libero`)で見るときは
  `--dataset-root <展開先> --dataset-cameras image,image2`。**素のLIBEROは列名が
  `image`/`image2`**で、学習データの`front`/`wrist`とは違う。**このデータは手元に
  無い**(2026-08-13時点。HFに1.94GB・40タスク・1,693本)

### 500 ステップの時点で見えたこと(2026-08-13 23:48 実測)

エピソード2(`pick up the black bowl in the top drawer...`)・`--mode chunk`。
**本番が使う先頭10 ステップ**での値である(`work/logs/replay/replay_ep2_chunk.csv`)。

| 見たもの | 値 | 読み |
|---|---|---|
| 並進3軸の相関 | dx 0.94 / dy 0.93 / dz 0.87 | **行き先の向きはもう合っている** |
| 並進の大きさ | GTのdxが+0.92まで伸びるところ、予測は+0.58 | **6割ほどしか動かない。力加減が足りない** |
| 回転3軸 | GT自体がほぼ0(標準偏差0.006〜0.016)、予測は小さな下駄(step 0で dry −0.11) | 相関がマイナスなのはこの下駄のせいで、意味のある信号ではない |
| グリッパ | GT −1.00に対し −0.97〜−1.01 | 正しい(開いたまま) |

**最終チェックポイントで見るべきは「大きさが追いつくか」**である。向きの相関は
すでに0.9前後なので、そこが伸びる余地は小さい。同じエピソード・同じ`--mode chunk`で
撮り直して、先頭10 ステップの「誤差/標準偏差」が下がっているかを見れば足りる。

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| 24,000 ステップまで完走したときの実際の所要時間(`smp/s`は50 ステップの実測値で、長時間走らせると変わる可能性がある) | 学習ログの経過時間を見る |
| batch=16のまま学習データに手作業の画像を混ぜたときにメモリ・速度が変わるか | 混ぜたデータセットで同じsmoke-testを再実行する |
| 手作業の画像を`--dataset-root`で混ぜる形が、pi05でもコードを直さずに通るか | 少量(`--episodes-per-task 3`程度)で1度回してみる |
| L4 24GBでの1推論の時間(10秒に収まるか) | `policy_server.py`のπ0.5版を書いてから実測。**このマシンのCPU 1コアでは195秒**、重みの読み込みに302秒かかった(2026-08-13実測)。GPUでの値はまだ無い |
| 学習が進むと予測が正解に近づいていくか | [replay_pi05_on_dataset.py](../../solution/tools/viz/replay_pi05_on_dataset.py)で、実データのフレームを流して正解と並べた動画を作る(下の節) |
