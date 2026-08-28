# 13. RunPod を借りて回すときに効いたこと

**手順は [../howto/06_runpod_pi05.md](../howto/06_runpod_pi05.md) にある。**
ここに置くのは、**手順を追うだけでは分からなかった性質**である。
①〜⑤ は 2026-08-14 に A100 80GB PCIe を 1 台借りて 3,000 ステップ回したときに
実際に踏んだもので、**課金・ストレージ・見張りの 3 つ**に分かれる。
**⑥⑦⑧ は 2026-08-28 に、お試し学習用の Pod を作ったときに踏んだ**もので、
**Pod を作る画面・止める操作・`/workspace` へファイルを送る操作**の性質である。

## ① 「何日借りる」という設定は無い。止まるのは残高が尽きたとき

**時間貸しの料金は秒単位の従量課金だけである。**期限付きの契約は Savings Plan
だけで、これは **3 か月または 6 か月のみ**（前払い・返金不可）。1 日や 1 週間の
契約は無く、Instant Cluster にも期間を入れる欄が無い。

出典: [Pod pricing](https://docs.runpod.io/pods/pricing)・
[Billing](https://docs.runpod.io/accounts-billing/billing)（2026-08-14 に読んだ）。

**「1 日で借りた」という感覚の出どころは、Pod 管理画面の
`Estimated time left` である。**これは契約の残りではなく、
**いまの残高といまの時間単価から計算した、止まるまでの時間**である。

| 見たもの | 値 | 意味 |
|---|---|---|
| `Estimated time left` | **7 時間 3 分**（11:26 JST に見た） | **18:29 に止まる** |
| 締切から逆算した予定 | 20:00 に持ち帰りを始める | **1 時間半足りない** |

**伸ばし方は残高を足すことだけである。**Pod は止めずにそのまま続く。
入れる額は次で決まる。

    必要額 = 時間単価($/hr) × (使いたい終了時刻までの時間 + 予備 1 時間)

**残高が減ったときの知らせ（Low balance notification）と、
自動で補充する仕組み（Auto-pay、1 時間に 1 回まで）は、どちらも既定で切れている。**
使う分の上限（spend limit）の既定は全リソース合わせて $80/hr なので、
GPU 1 枚ならここには当たらない。

**教訓は、終わる時刻を 2 本持つことである。**学習が終わる時刻だけを見ていて、
残高が尽きる時刻を見ていなかった。**後者の方が先に来ていた。**

## ② 残高が $0 になったとき何が残るかは、ストレージの種類で決まる

| 借り方 | $0 になったとき | データ |
|---|---|---|
| Network Volume を付けている | **止まるだけ** | **残る** |
| 付けていない（Volume Disk だけ） | **Terminate される** | **復旧できない** |

出典: [Billing](https://docs.runpod.io/accounts-billing/billing)
（"Pods without a network volume are terminated, and their data can't be
recovered."）。**ただし $0 のままにすると保管料が払えず、Network Volume も
いずれ消される**とも書かれている。

**Network Volume は Pod を作るときにしか付けられない**（後から付け外しできない。
大きさは増やせるが減らせない。出典:
[Network volumes](https://docs.runpod.io/pods/storage/create-network-volumes)）。
**借りるときに決まってしまう性質なので、最初に決める。**

## ③ どちらを借りたかは `df` で判る。そして `df` では使用量が分からない

```bash
df -h /workspace
# mfs#ca-mtl-3.runpod.net:9421  755T  425T  330T  57%  /workspace
```

**`mfs#...runpod.net` かつ数百 TB と出たら、ネットワーク越しの共有ストレージ、
つまり Network Volume である**（Network Volume は Pod の既定のディスクを
置き換えて `/workspace` にマウントされる）。**755T はクラスタ全体の大きさで、
自分が 150 GB のうち何 GB 使ったかは、ここには出ない。**

```bash
du -sh /workspace/hf_home /workspace/out   # 自分の使用量はこちら
```

**借りた 150 GB を超えたらどうなるかは、公式ドキュメントに書かれていない。**
今回は超えなかったので、**確かめていない。**

## ④ チェックポイント 1 個は 25 GB。放っておくと 2,500 ステップで落ちる

| | 値 | 出どころ |
|---|---|---|
| 1 個の大きさ | **約 25 GB** | Pod のディスク使用量が、500 ステップの保存の直前 **30 GB**・直後 **55 GB**。その差 |
| 事前の見積もり | 21 GB | **2 割ほど小さく外していた** |
| 保存の間隔 | 500 ステップごと | `--save-freq` の既定値 |

**LeRobot は古いチェックポイントを消さない。**150 GB のうち学習開始時点で
30 GB（初期重み＋データセット）使っているので、
**30 ＋ 25 × 5 ＝ 155 GB となる 2,500 ステップの保存の途中で尽きる。**

**古い方を消して数を保つ見張り（間引き）を常駐させると、80 GB 前後で安定する。**
2 個残す設定なら、新しい 1 個を書いている一瞬だけ 3 個ぶん（105 GB）になる。

踏んだ細かいことが 3 つある。

1. **`crontab` は使えない。**イメージに cron を入れていないので
   `command not found` になる。`nohup ... &` の形で常駐させる。
2. **見張りが出すログの「ディスク空き」は、Network Volume では無意味な数字になる**
   （330T と出る）。**消す判断は個数で行っているので、動作そのものには影響しない。**
3. **同じ見張りを 2 回起動していた。**ログに「最新 3 個を残す」と
   「最新 2 個を残す」の起動行が並んで残っていた。二重起動は `flock` で弾く
   作りだが、**環境によって効かない。**`pgrep -af prune_checkpoints.sh` で
   実際に何個動いているかを見る。

## ⑤ 持ち帰りの経路も、借りるときの設定で決まる

| 経路 | 使える条件 | 実測 |
|---|---|---|
| `scp` | **SSH のみ。**Pod 作成時に「Expose TCP Ports」へ `22` を足し、Connect タブの公開 IP 側を使う | **1.7 MB/s。8.8 GB に 1 時間以上** |
| Hugging Face の非公開リポジトリ経由 | Pod の中でコマンドを打つだけ。**Web Terminal だけでも成立する** | **9.4 GB で 16 分** |

**Web Terminal ではファイルを取り出せない。**RunPod のサーバを経由する
Basic SSH でも `scp` は通らない。**`22` を公開し忘れると、`scp` の道は
借り直すまで戻ってこない。**（詰まったときの切り分けは
[../howto/08_hf_transfer_troubleshooting.md](../howto/08_hf_transfer_troubleshooting.md)。）

**走っている Pod を編集すると、コンテナが再起動する。**`/workspace` の中身は
残るが、**学習のプロセスは死ぬ**（出典:
[Storage types](https://docs.runpod.io/pods/storage/types) の
"Editing a running Pod resets it completely"）。**途中でディスクを増やす手は、
学習を殺す手でもある。**間引きで足りるなら触らない。

## ⑥ Docker Hub の同じ名前のイメージに、SSH が入っていない版が混じっている

**`sirukufarios/parc2026-train-pi05` の `v1` と `v2` は、中身が違う。**
`v1` を選んで起動コマンドを空にすると、Pod は同じ 1 行を延々と吐き続ける。

```
python: can't open file '/workspace/train_pi05.py': [Errno 2] No such file or directory
```

| タグ | push した時刻 | CMD | `openssh-server` と `runpod_ssh_start.sh` |
|---|---|---|---|
| `v1` | 08-13 17:40 JST | `python /workspace/train_pi05.py --smoke-test` | **入っていない** |
| **`v2`** | **08-14 07:27 JST** | `python /opt/pi05/train_pi05.py --smoke-test` | 入っている |

**`v1` は `Dockerfile.pi05` の最初のコミット（08-13 17:01）から作ったもので、
その後の 2 つの修正——CMD のパスを直したもの（18:00）とフル SSH を入れたもの
（19:07）——が入っていない。** 起動が繰り返す仕組みは 3 段である。

| 段 | 何が起きるか |
|---|---|
| 1 | 起動コマンドが空なので、イメージの CMD が動く |
| 2 | `/workspace` は Volume Disk がマウントで覆うので、**焼き込んだ `/workspace/train_pi05.py` は隠れて見えない**（③ の性質の裏返し） |
| 3 | **落ちたコンテナは RunPod が作り直す**ので、同じ 1 行が延々と出る |

**いちばん効くのは 3 段目である。** 起動コマンドが落ちると `sshd` も立たないので、
**SSH では中へ入れない。**直せる場所は Web コンソールの起動コマンドの欄だけになる。
**そして `v1` では、その欄に `bash /opt/pi05/runpod_ssh_start.sh` と書いても直らない**
——そのスクリプトが `v1` に入っていないので、文言が変わるだけで同じ繰り返しになる。
**`v1` を選んでしまったら、`v2` の Pod を作り直すしかない。**

**教訓は、イメージの名前だけを記録してタグを記録しなかったことである。**
[../resources/infra_and_docker.md](../resources/infra_and_docker.md) には
「2026-08-14 に push」とだけ書いてあり、それは `v2` を指していたが、
[../howto/06_runpod_pi05.md](../howto/06_runpod_pi05.md) の手順には `v1` と
書いてあった。**手順は 2026-08-28 に `v2` へ直した。**

## ⑦ Stop は GPU を手放す操作である。同じ GPU が戻る保証は無い

**「あとで Start すれば続きから」ではない。** Stop すると GPU は解放され、
**Start のときに空いていなければ GPU 0 枚の Pod として起動する**
（`/workspace` と環境は戻る）。出典:
[Manage Pods](https://docs.runpod.io/pods/manage-pods)
（"Stopping a Pod releases the GPU"・"You may be allocated zero GPUs if capacity
has changed."、2026-08-28 に読んだ）。**2026-08-28 に実際に取られた。**

| したいこと | Stop で足りるか |
|---|---|
| 課金を止める | **足りない。**GPU 代は止まるが、**Volume Disk の保管料は Stop 中も続く** |
| 設定を直して続ける | **危ない。**GPU を手放すので、直す前に取られる。**Pod を作り直す前提で動くほうが速い** |
| GPU の種類を変える | **できない。**Pod の GPU は後から変えられないので、作り直しになる |

**既存の Pod の GPU が取られた後にできるのは、空くのを待つか、Terminate して
作り直すかの 2 つだけである。** Network Volume を付けている場合、作り直す Pod は
**同じデータセンタでしか作れない**（② の性質）ので、**選べる GPU もその中に限られる。**

## ⑧ Network Volume へ `tar` で送ると、所有者の復元に失敗して止まる

**同じコマンドが `/opt/pi05` では通り、`/workspace` では落ちる**（2026-08-28 に踏んだ）。

```
tar: work/artifacts/vendor/vggt/heads/track_modules/__init__.py:
     Cannot change ownership to uid 1000, gid 1000: Operation not permitted
tar: Exiting with failure status due to previous errors
```

**`/opt/pi05` はコンテナ側の普通のファイルシステムだが、`/workspace` は MFS
（ネットワーク越しの共有ストレージ）である**（③）。root で展開していても、
**送り元の uid/gid をそこへ設定することはできない。**

| | 直し方 |
|---|---|
| 所有者だけ落ちる | `tar xzf - --no-same-owner -C <先>` |
| 権限も落ちる | さらに `--no-same-permissions` を足す |

**中身自体は展開されているので、`--no-same-owner` を付けて入れ直し、
ファイル数を送り元と突き合わせる**（`tar tzf - | grep -vE '/$' | wc -l` と
`find <先> -type f | wc -l`）。

## 測り直すコマンド

```bash
# Pod の中で
df -h /workspace                                  # ストレージの種類を見分ける
du -sh /workspace/hf_home /workspace/out          # 自分の使用量
du -sh /workspace/out/<run>/checkpoints/*         # 1 個の大きさ
pgrep -af prune_checkpoints.sh                    # 見張りが何個動いているか
tail /workspace/out/<run>_prune.log               # 「削除:」の行が出ているか
```

**残高と `Estimated time left` は画面でしか見られない。**`runpodctl billing` は
過去の履歴だけで、現在の残高は出さない。

## Claude 側 / 人間側

**Claude 側**: **時間の見積もりを 2 本作る。**学習が終わる時刻と、
残高が尽きる時刻である。今回は後者を計算していなかった。
**「消えるかどうか」を言う前に、`df -h /workspace` でストレージの種類を確かめる。**
種類を確かめずに「Terminate されて消える」と言ってしまい、
実際には Network Volume だったので**危険を過大に伝えた。**

**人間側**: **借りる画面で決まってしまうことが 3 つある**——
Network Volume を付けるか、`22` を公開するか、ディスクの大きさ。
**どれも後から直すには借り直しか再起動が要る**ので、
[../howto/06_runpod_pi05.md](../howto/06_runpod_pi05.md) の Pod 作成の節を
**作る前に**一度読む。

**両方**: **借りている時間の中で、いちばん読めないのは持ち帰りである。**
1.7 MB/s と 9.4 GB/16 分では、締切前の 1 時間の意味が変わる。
**本番の重みが出来るのを待たず、最初のチェックポイントで 1 回測る。**
