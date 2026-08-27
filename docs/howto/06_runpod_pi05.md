# 06. RunPodでπ0.5をフルパラメータ学習する

ローカル(H100 20GB)は`train_expert_only`で固定した縮小版しか載らない。RunPodで
80GBクラスのGPUを借りれば、公式レシピどおりのフルパラメータ学習(VLMも含めて
全部学習)を試せる。ローカルの学習はそのまま保険として並行して回し、こちらは
上振れ狙いの追加の一手として位置付ける。

いきなり数時間かけて本番を回すのではなく、**少量データで数十歩だけ動かして
「フルパラメータ学習が本当に通るか」を確かめてから**、バッチサイズを探り、
最後に本番のステップ数へ進む——という順で書く。

**2 回目の学習（フルパラメータ ＋ 画像拡張）の 1 回分の設計は
[../plan/15_runpod_fullparam_with_augment.md](../plan/15_runpod_fullparam_with_augment.md)
にある。**1 回目との差分・歩数と費用・立ち上がりの確認 3 点はそちら。
**このファイルは何度でも使う手順（イメージ・Pod・batch の探り方・間引き・持ち帰り）を持つ。**

## 0. 締切から逆算した時間割(2026-08-14 07:15 に書いた)

**提出は 8/14 23:59。残り約16時間45分。** この節は「確実に結果を出す」ために
**先に決めておくこと**をまとめる。以下はすべて実測値からの逆算で、
**推定の箇所はそう書いてある。**

| 後ろから積む | 見込み | 遅くともこの時刻に始まっている |
|---|---|---|
| 提出物を固めて出す | 30分 | 23:30 |
| **π0.5を評価する(36件)** | SmolVLAで**72分**が実測。π0.5は約9倍の模型なので**未測定。2倍なら2.4時間** | **21:00** |
| **持ち帰る(scp 8.7GB)** | **未測定。** 回線次第で20分〜2時間 | **20:00** |
| RunPodの学習 | 残り全部 | — |
| 準備(push・Pod作成・データ取得・batch探し) | **2〜3時間**(下の内訳) | 07:30 |

**つまりRunPodの学習に使えるのは、うまくいって9〜10時間である。**
[10節](#10-本番のフルパラメータ学習を始める)が挙げている30,000 ステップは
公式ドキュメントの値であって、**この時間に収まる保証は無い。ステップ数は
9節で`smp/s`を測ってから、下の式で決めること。**

    ステップ数 = 実測smp/s × 使える秒数 ÷ batch

**ステップ数を多すぎる側に外すと、途中で止めたときに学習率が下がりきらない。**
`--steps`は学習率の下げ方そのものを決める値で、`CosineDecayWithWarmup`は
指定したステップ数に合わせて形を伸縮させる。30,000 ステップの指定で15,000 ステップで止めると、
**下がりきる前の中途半端な重みしか残らない**(SmolVLA側で踏んだのと同じ失敗)。
**少なめに見積もって完走させる方が、必ず良い。**

### 準備にかかる時間の内訳(ここが読みにくい)

| やること | 量 | 備考 |
|---|---|---|
| イメージをDocker Hubへpush | **9.66GB**(`docker images`実測) | **このマシンの上り回線次第。ここが一番読めない** |
| RunPod側がpull | 同上 | データセンター内なので速い |
| 初期重み`pi05_libero_base` | **14GB**(HFキャッシュ実測) | |
| データセット`libero_plus` | **約30GB**(`~/.cache/huggingface/lerobot`実測。hub側15GB＋展開15GB) | **これが一番大きい** |

**pushの進み具合が遅いと分かった時点で、[6節の道B](#6-ポッドを作る道b)へ
切り替えられる**(公式テンプレート＋`runpod_setup.sh`。pushが要らない)。
**最初の1レイヤの速度を見て、30分以内に終わらなさそうなら切り替える**、
と決めておくとよい。

### Volume Diskは60GBでは足りない——**150GBにすること**

ローカルと同じ`train_expert_only`(固定した縮小版)ならチェックポイント1個は
11GBなので60GBでも回るが、**フルパラメータ学習では1個25GBになり、60GBでは
2個も置けずに落ちる**(この節に合わせて3節も150GBに直した)。

| | ローカル(expert only) | フルパラメータ(RunPodで実測) |
|---|---|---|
| 学習するパラメータ | 693M | 4.14B |
| `pretrained_model` | **8.7GB**(実測。全パラメータを保存するので変わらない) | 8.7GB(同じ) |
| `training_state` | **2.1GB**(実測。1パラメータあたり3.0バイト) | **約16GB**(下の25GBから8.7GBを引いた値。**直接は測っていない**) |
| **1個あたり** | **11GB** | **約25GB**(実測) |

**25GBの測り方(2026-08-14 11時台)**: Podのディスク使用量が、500 ステップの保存の
直前で**30GB**、保存の直後で**55GB**。その差が1個ぶんである。
**当初の推定は21GBだったので、2割ほど大きい方に外れた。**

Volume Diskの中身は、上と合わせてこうなる。

| 中身 | 量 |
|---|---|
| 初期重み＋データセット(`/workspace/hf_home`) | **合わせて30GB**(実測)。**見積もりは45GBだったので、こちらは小さい方に外れた** |
| 間引いて2個残したチェックポイント | 25GB×2＝**50GB** |
| 間引きが古い1個を消す前、一瞬だけ3個ある状態 | 25GB×3＝**75GB** |
| **最大** | **105GB** |

**150GBで足りる**(45GB余る)。$0.10/GB/月なので、1日なら数十円の差にしかならない。
**60GBではチェックポイントを2個置けずに落ちる。**

**`hf_home`の30GBの内訳(初期重み14GB＋データセット16GBのはず)は測っていない。**
気になるときは`du -sh /workspace/hf_home/*`で見る。
`training_state`と`pretrained_model`の分かれ方も同じく未実測で、
`du -sh /workspace/out/<run-name>/checkpoints/*/*`で見られる。

### 持ち帰りは、本番を待たずに一度試しておく

**8.7GBのscpが何分かかるかを、22時に初めて知るのは避ける。**
8節の20 ステップの確認でチェックポイントが1個できるので、**そのまま
12節のscpを実行して時間を測る。** これは同時に
**「フルパラメータ学習が吐いたチェックポイントを`policy_server.py`が
読めるか」の確認**にもなる(版3.8.0は`config.json`の`type`を見て
π0.5の道へ入る。`train_expert_only`が`false`でも中身の形は変わらない
はずだが、**確かめていない**)。

### CPUの数を見てから`--num-workers`を決める

このマシンは`nproc`が**2**しかなく、`--num-workers 2`でも
`data_s` 0.011〜0.020秒／`updt_s` 1.39秒と、データ待ちは無視できていた。
**80GBのGPUでbatchを64に上げると、1 ステップが短くなる分だけ動画の展開が
追いつかなくなりうる。**

```bash
nproc          # Pod のコア数を見て、--num-workers はその数-1 程度にする
```

学習ログの`data_s`が`updt_s`に近づいていたら、GPUがデータ待ちで
遊んでいる。`--num-workers`を上げる。

## 1. アカウントと料金の感覚

https://runpod.io でアカウントを作り、クレジットカードを登録する(使った分だけ
の従量課金)。2026年時点の目安は次の通り。

| GPU | 目安の時間単価 |
|---|---|
| A100 80GB | $1.4〜2.2/時間 |
| H100 80GB | $2.9〜3.0/時間 |

出典: [Spheron Blogの調査](https://www.spheron.network/blog/runpod-h100-pricing-2026/)、
[GridStackHubの実測](https://gridstackhub.ai/providers/runpod-a100-80gb)。**表示価格は
変わりうるので、実際にポッドを作る画面の金額を必ず確認すること。**

10〜12時間回すと、A100で$17〜26、H100で$35〜36程度になる計算。

**Secure CloudとCommunity Cloudの違い**: Community Cloudは安いが、他の利用者に
途中で追い出される(preemptされる)ことがある。**今回は10時間規模の無人実行**なので、
追い出されて数時間分が消えるのを避けるため、多少高くても**Secure Cloud**を選ぶこと。

## 2. マシンの選び方——「素のUbuntu + GPU」で自分でdockerを回すのは避ける

**RunPodの「Pod」自体が、既にコンテナである。**
仮想マシン(VM)を借りてその中を自由に使う、という形ではない
([Introduction to Containers](https://docs.runpod.io/tutorials/introduction/containers))。
そのため、素のUbuntuテンプレートのPodを借りて、その中で`docker build`/
`docker run --gpus all`のように**もう一段コンテナを入れ子にする(Docker-in-Docker)**
やり方は、CPU Podでは既に廃止されており、GPU Podでも「特権(privileged)」設定が
要るなど不安定である
([RunPodのブログ](https://www.runpod.io/blog/enhanced-cpu-pods-docker-network))。
**このマシンでこれまでやってきた「dockerで環境を作って回す」というやり方自体は、
RunPod上ではそのまま持ち込めない。**

代わりに使える道は2つある。

| 道 | やること | 向いている場面 |
|---|---|---|
| **A. 自分のイメージをそのままPodにする(推奨)** | [Dockerfile.pi05](../../solution/train/Dockerfile.pi05)をこのマシンでビルドし、Docker Hubへpushする。RunPod側は「そのイメージを実行するPod」を作るだけで、Pod自身はコンテナを入れ子にしない | ローカルで検証済みの環境をそのまま持っていきたいとき。**今回はこちら** |
| B. 公式テンプレート+スクリプトで直接installする | 公式の「RunPod PyTorch」テンプレートのPodの中で[runpod_setup.sh](../../solution/train/runpod_setup.sh)を実行し、pip/aptで直接環境を作る(dockerは使わない) | Docker Hubのアカウントを用意したくないとき |

**Aは4〜5節、Bは6節でポッドを作る。どちらを選んでも、7節(ライセンス同意)以降は
共通の手順になる。** ただし3節(永続化ストレージ)は、道A・道Bどちらでも
共通で必要な設定なので、先に読むこと。

## 3. 永続化ストレージの使い分け

**10〜12時間の無人実行の途中で接続が切れたり、誤ってPodを止めてしまったりすると、
何も対策していなければ学習結果ごと消える。** どこにデータを置くかを、
Podを作る前に決めておく。

RunPodのストレージは3種類ある。

| | Container Disk | Volume Disk(Pod付属) | Network Volume |
|---|---|---|---|
| **Stopで消えるか** | **消える** | 消えない | 消えない |
| **Terminateで消えるか** | 消える | 消える | **消えない** |
| **マウント先** | 指定不可 | **`/workspace`固定** | `/workspace`(Volume Diskの代わりに使う) |
| **作る操作** | Pod作成時に自動 | Pod作成時にサイズを指定するだけ | 別途「Storage」から先に作り、Pod作成時に選ぶ |
| **他のPodと共有** | 不可 | 不可 | 可(ただし作った場所のデータセンターに固定される) |
| **料金** | $0.10/GB/月 | 稼働中$0.10・停止中$0.20/GB/月(**止めている間の方が高い**) | $0.07/GB/月(稼働/停止同額) |

出典: [公式ドキュメント](https://docs.runpod.io/pods/storage/types)、
[Aquanodeの整理](https://www.aquanode.io/blog/runpod-volume-disk-vs-network-volume)。

**今回のように1回のPodを10〜12時間回して、終わったら持ち帰ってすぐ止める使い方
なら、Volume Diskで十分。** Network Volumeは複数Podで使い回す・長期間残す
場合向けで、今回はそこまでの要件が無い。

### サイズの目安

- **Container Disk: 20GB程度**(OS・pip/aptで入れたパッケージ分だけで足りる)
- **Volume Disk: 150GB。** フルパラメータ学習では最大105GB使う(実測。内訳は
  [0節の表](#volume-diskは60gbでは足りない150gbにすること))。
  `train_expert_only`の縮小版だけなら60GBでも足りるが、**このドキュメントが
  扱うフルパラメータ学習では落ちる**

### 使用量は`du`で見る——**Network Volumeでは`df`が使えない**

Network Volumeを付けた場合、`/workspace`はRunPodのネットワーク越しの共有
ストレージになる。**`df`はクラスタ全体の容量を出すので、自分が何GB使ったかは
分からない**(2026-08-14の実測。`mfs#ca-mtl-3.runpod.net:9421  755T  425T  330T`
と出た)。

```bash
du -sh /workspace/hf_home /workspace/out   # 自分の使用量はこちらで見る
```

11節の間引きスクリプトがログに出す「ディスク空き」も同じ理由で無意味な数字に
なるが、**削除の判断は個数で行っているので動作には影響しない。**

### Pod作成時に、環境変数でデータの置き場所を`/workspace`へ向ける

Volume Diskは`/workspace`に固定でマウントされる。**学習の出力先(`OUTPUT_ROOT`)
とHFキャッシュ(`HF_HOME`)を、Pod作成画面の「Environment Variables」で
明示的にその配下へ向けないと、Volume Diskを付けても永続化されない**
(既定のままだと道Aは`/out`と`/hf_home`、道Bは`/root/.cache/huggingface`に
書かれ続け、どちらも`/workspace`の外にあるため消える)。

```
OUTPUT_ROOT=/workspace/out
HF_HOME=/workspace/hf_home
```

**これは道A・道Bどちらでも必要。** 道Bはイメージを使わないので、Pod作成時の
環境変数として設定するか、`hf auth login`より前に`export OUTPUT_ROOT=...`
`export HF_HOME=...`しておく。

### スクリプト自体の置き場所は別問題

**上のデータの置き場所とは別に、学習スクリプト自体をどこに置くかにも注意が要る。**
道Aのイメージは、`train_pi05.py`などを`/workspace`ではなく**`/opt/pi05/`**に
置くようにしてある([Dockerfile.pi05](../../solution/train/Dockerfile.pi05)参照)。
**Volume Diskは`/workspace`に上書きでマウントされるため、そこにCOPYで
焼き込んだファイルは、マウントのされ方によっては見えなくなる可能性が
あるからである。** 道Bはファイルを起動後にscpで`/workspace`へ転送するので、
この問題は起きない(書き込む順番が逆——先にマウントされた場所へ後から書く)。

**以降のコマンド例で`/opt/pi05/`と書いている箇所は、道Bの場合`/workspace/`と
読み替えること。**

## 4. イメージをビルドしてpushする(道A)

このマシン(ローカル)で使ったのと同じ
[Dockerfile.pi05](../../solution/train/Dockerfile.pi05)を使う。**RunPod向けに
`openssh-server`と`tmux`を入れ足してある**(理由は5.3)。ローカルの`--user`回避
(`/hf_home`・`HOME=/tmp`)はRunPod側では要らなくなる(Podは既定でrootのまま
動くため)が、入っていても害はないのでそのまま流用できる。

```bash
docker build -t <Docker Hubのユーザー名>/parc2026-train-pi05:v1 \
    -f solution/train/Dockerfile.pi05 solution/train

docker login   # 初回だけ。Docker Hubのアカウントが要る(無料で作れる)
docker push <Docker Hubのユーザー名>/parc2026-train-pi05:v1
```

**RunPodはlinux/amd64のイメージを前提にしている。** このマシンがx86_64
(Ubuntu)であれば意識しなくてよいが、Apple Silicon等でビルドした場合は
`docker buildx build --platform linux/amd64 ...`が要る。

### ビルドできたか確かめる(pushの前に)

**RunPodに上げてから「sshdが立たない」と気付くと、Pod代を払いながら
調べ直すことになる。** ローカルで先に2つだけ確認する。

```bash
# sshdの設定が妥当か(ホスト鍵を作ってから-tで検査する。
# ホスト鍵はわざとイメージに焼いていないので、先に作る必要がある)
docker run --rm parc2026-train-pi05 \
    bash -c 'ssh-keygen -A >/dev/null && /usr/sbin/sshd -t && echo "sshd_config OK"'

# scpに必要なsftp-serverが入っているか
docker run --rm parc2026-train-pi05 ls -l /usr/lib/openssh/sftp-server
```

1つ目が`sshd_config OK`と出て、2つ目がファイルを表示すれば通っている。
(`--user`は付けない——RunPodではrootで動くので、そちらの条件で確かめる。)

## 5. ポッドを作る(道A)

### 5.1 先にRunPodへ公開鍵を登録する

**Podを作る前にやること。** RunPodは、アカウントに登録した公開鍵を
Pod起動時に`PUBLIC_KEY`という環境変数で中へ渡してくる。登録していないと
鍵が渡ってこないので、後述のsshdが立っても入れない。

```bash
# 鍵が無ければ作る(既に ~/.ssh/id_ed25519.pub があれば飛ばす)
ssh-keygen -t ed25519 -C "runpod"
cat ~/.ssh/id_ed25519.pub
```

出た1行を、RunPodの「Settings」→「SSH Public Keys」に貼る。

### 5.2 テンプレートとPodを作る

1. RunPodの「Templates」→「New Template」で、4節でpushしたイメージ名
   (`<ユーザー名>/parc2026-train-pi05:v1`)を指定する。
2. **Container Diskは20GB程度、Volume Diskは150GB**にする(3節)。
3. **「Environment Variables」に`OUTPUT_ROOT=/workspace/out`と
   `HF_HOME=/workspace/hf_home`を追加する**(3節)。
4. **「Expose TCP Ports」に`22`を追加する。** これが無いとRunPod側が
   SSH用のポートを割り当てないので、5.3のフルSSHが使えない。
5. **Container Start Commandを次で上書きする。**

   ```
   bash /opt/pi05/runpod_ssh_start.sh
   ```

   空のままだとイメージの既定CMD(`python /opt/pi05/train_pi05.py --smoke-test`)が
   デプロイ直後に動き出してしまうが、**7節のライセンス同意より前に初期重みの
   取得を試みて確実に失敗する。** 上のスクリプトは
   [runpod_ssh_start.sh](../../solution/train/runpod_ssh_start.sh)で、
   **sshdを起動してから`sleep infinity`で待つ**だけなので、学習は自分の手で
   始められる。
6. 「Deploy」→「GPU Pod」で、作ったテンプレートを選び、GPUは
   **A100 80GBまたはH100 80GB(Secure Cloud)**を選ぶ。

### 5.3 接続する

デプロイ後、Pod管理画面の「Connect」タブに2種類の入り口が出る。

| 入り口 | 使えること | 今回の使いどころ |
|---|---|---|
| Web Terminal(「Enable Web Terminal」→「Open Web Terminal」) | ブラウザからコマンドを打つだけ。**ファイル転送(scp)はできない** | 最初の様子見・短いコマンド |
| SSH(`ssh root@<IP> -p <port> -i ~/.ssh/id_ed25519`) | 鍵認証・**scp対応**。公式ドキュメントが長時間実行に推奨している方 | **本番の10〜12時間の学習と、12節の持ち帰り** |

**公式ドキュメントはWeb Terminalを「短いコマンド・デバッグ向け」、SSHを
「長時間動かすプロセス・確実な接続向け」と位置付けている**
([Connect to Pods](https://docs.runpod.io/pods/connect-to-a-pod))。
本番は10〜12時間動かすので、**SSHで入ること。**

RunPodのSSHには、公開IPを使う「Full SSH」と、RunPodのサーバを経由する
「Basic SSH」の2種類がある。**Basic SSHはscp/sftpに対応しない**ため、
チェックポイントを持ち帰るには公開IP側(Connectタブに出るIPとポートの組)を使う
([Use SSH](https://docs.runpod.io/pods/configuration/use-ssh))。

**入れなかったときの見どころ**: Pod管理画面の「Logs」に
`[runpod_ssh_start]`で始まる行が出ている。`PUBLIC_KEY が空です`という警告が
出ていれば5.1の鍵登録が漏れている(その場合はPodを作り直す)。

## 6. ポッドを作る(道B)

1. 「Deploy」→「GPU Pod」で、テンプレートは公式の**「RunPod PyTorch」系**
   (SSHが最初から使える状態で入っている)を選ぶ。
2. **Container Diskは20GB程度、Volume Diskは150GB**にする(3節)。
3. **「Environment Variables」に`OUTPUT_ROOT=/workspace/out`と
   `HF_HOME=/workspace/hf_home`を追加する**(3節)。設定し忘れた場合は、
   接続後に`export`で代用できる。
4. GPUは**A100 80GBまたはH100 80GB(Secure Cloud)**。
5. デプロイ後、「Connect」タブに出るSSHコマンド
   (`ssh root@<IP> -p <port> -i ~/.ssh/id_ed25519`)で接続する。鍵が無ければ
   5.1と同じ手順で作ってRunPodに登録する。**公式テンプレートはフルSSH
   (scp対応)が最初から設定済み**なので、道Aの5.2手順4・5(TCP 22の公開と
   起動コマンドの上書き)に相当する準備は要らない。
6. このリポジトリから4ファイルを転送する。

   ```bash
   scp -P <port> -i ~/.ssh/id_ed25519 \
       solution/train/train.py solution/train/train_pi05.py \
       solution/train/runpod_setup.sh solution/train/prune_checkpoints.sh \
       root@<IP>:/workspace/
   ```
7. ポッド側で依存関係を入れる。

   ```bash
   cd /workspace
   bash runpod_setup.sh
   ```

## 7. ライセンス同意とログイン

π0.5はgemmaライセンス配布。**道A・道Bどちらでも共通の手順。**

```bash
hf auth login
```

未同意なら https://huggingface.co/google/paligemma-3b-pt-224 を先に開いて
同意しておくこと。

## 8. 少量データで、まずフルパラメータ学習が通るか確かめる

**いきなり本番(14,347エピソード全部・30,000 ステップ)を回さない。** `--episodes-per-task`
(`train.py`から再利用している、`train_pi05.py`が既に対応済みの引数)で
データを絞り、**VLMを含めた全パラメータの学習ループが、そもそもエラーなく
1周するか**を数十秒で確認する。**以下は道Aの例(`/opt/pi05/`)。道Bなら
`/workspace/`と読み替える(3節)。**

```bash
python /opt/pi05/train_pi05.py --run-name pi05_small_check --steps 20 --batch-size 2 \
    --episodes-per-task 3 \
    --freeze-vision-encoder false --train-expert-only false
```

- `--episodes-per-task 3`: 40タスク×3本=120エピソードだけを使う。**ダウンロード量も
  学習データの読み込みも小さくなるので、フルパラメータ学習特有の問題
  (VLM側の勾配計算・保存するチェックポイントのサイズなど)を安く洗い出せる**。
  ローカルでbatch sizeを探ったときの`--steps 20`と同じ発想を、
  「フル学習に切り替えた」という変数だけ足して確かめている。
- ここで見るのは**エラーなく`End of training`まで到達するか**であって、
  スコアではない(ローカルのsmoke-testと同じ)。

通ったら、次はバッチサイズの探索に進む。

## 9. 適正なバッチサイズを探す

ローカルで使ったのと同じ考え方——**学習ログ自身が出す`mem_gb`を見比べる**。
公式ドキュメントは80GBでbatch_size=64を挙げているが、それが**本当にこの
環境で収まるか**、**もっと上げられる余地がないか**を実測する。

```bash
for bs in 32 64 96; do
    python /opt/pi05/train_pi05.py --run-name "pi05_bs_check_${bs}" \
        --steps 20 --batch-size "${bs}" --episodes-per-task 3 \
        --freeze-vision-encoder false --train-expert-only false
done
```

各回のログ末尾に出る`mem_gb`と`smp/s`を、ローカルのときと同じ形で表にする。

| batch | mem_gb | smp/s |
|---|---|---|
| 32 | | |
| 64 | | |
| 96 | | |

**読み方はローカルのときと同じ**([docs/howto/07](07_train_pi05_local.md)参照)。

- 2点(例: 32と64)から`fixed + k×batch`の直線を引き、80GBに対してどこまで
  伸ばせそうかを外挿する
- 外挿だけで決め打ちしない。**外挿より1段上**で必ず実測してから確定する
  (ローカルでも2→8→16と1段ずつ上げて確かめた)
- `smp/s`の伸びが頭打ちになってきたら、それ以上バッチを上げても
  「速くなる」効果は薄く、OOMのリスクだけが増える

**フルパラメータ学習は、固定する縮小版よりメモリの伸び方が急なはずである**
(オプティマイザの状態を持つパラメータが693Mから4.14B全部に増えるため)。
ローカルの数値(batch 2→8→16でmem_gbが12.90→13.99→16.27)をそのまま
当てはめられないことに注意。

## 10. 本番のフルパラメータ学習を始める

バッチサイズが決まったら、本番へ進む。**接続が切れてもプロセスが生き残るよう、
`tmux`の中で実行すること**(道Aのイメージには`tmux`を入れてある。道Bで
無ければ`apt-get install -y tmux`)。**入り口はWeb Terminalではなく
SSHにすること**(5.3の表)。

```bash
tmux new -s train
python /opt/pi05/train_pi05.py --run-name pi05_full --steps 30000 --batch-size 64 \
    --num-workers 8 --save-freq 5000 \
    --freeze-vision-encoder false --train-expert-only false
```

`--batch-size 64`は公式ドキュメント([lerobot/pi05](https://huggingface.co/docs/lerobot/en/pi05))
の値。9節の実測でもっと上げられると分かればそちらに変える。ステップ数(30000)は
公式ドキュメントがそのまま挙げている値——80GB GPU 1枚を前提にした数字。

`tmux`から抜けるには`Ctrl+b` → `d`(プロセスは動き続ける)。切断されても
再接続して`tmux attach -t train`で戻れる。

## 11. チェックポイントの間引き——**そのまま貼れる形**

チェックポイント1個は**フルパラメータ学習だと約25GB**(0節の実測)で、
LeRobotは古いものを自動で消さない。**放っておくと必ずディスクが尽きて
学習が落ちる。**

`--save-freq`の既定は500なので、500 ステップごとに25GB増える。Volume Disk 150GBで、
学習開始時点の使用量が30GB(初期重み＋データセット)なら、**2500 ステップの保存の
途中で尽きる**(30＋25×5＝155GB)。間引きを入れれば80GB前後で安定する。

### **RunPodでは`crontab`は使えない。`nohup`の形を使う**

ローカル(docs/howto/07)ではcronに登録しているが、**[Dockerfile.pi05](../../solution/train/Dockerfile.pi05)は
cronを入れていない**(`apt-get install`しているのは
`git build-essential ca-certificates ffmpeg`と`openssh-server
openssh-sftp-server tmux`だけ)。`crontab -e`を打っても
`command not found`になる。**Podは1回きりの使い捨てなので、
「再起動しても効き続ける」というcronの利点も要らない。**

### 貼るのはこの1行だけ(考えることは無い)

**本番の学習を始めた直後に、SSHの別の窓で1回貼る。**

```bash
nohup /opt/pi05/prune_checkpoints.sh /workspace/out/pi05_full 2 \
    --interval 60 --log /workspace/out/pi05_full_prune.log >/dev/null 2>&1 &
```

**この形が安全な理由を3つ。**

1. **`--log`の場所をrunディレクトリの「隣」にしてある。** スクリプトは
   `mkdir -p "$(dirname "$LOG_FILE")"`でログの置き場所を作るので、
   ログをrunディレクトリの「中」に指すと**そのディレクトリが先にできてしまい、
   学習が「出力先が既にある」で始まらなくなる**(`train_pi05.py`の`main`)。
   隣(`pi05_full_prune.log`)なら作られるのは`/workspace/out`だけで害が無い。
   **貼る順番を気にしなくてよくなるので、学習より先に貼っても壊れない。**
2. **`nohup ... &`なのでtmuxの窓を占有しない。** 切断しても生き残る。
3. **`--interval 60`。** 保存は`--save-freq 5000`なら数十分に1回なので、
   1分おきの点検で十分間に合う。

**`--run-name`を変えたら、2か所とも書き換えること**(`/workspace/out/<run-name>`と
`/workspace/out/<run-name>_prune.log`)。渡すのは`checkpoints`の**1つ上**である。

### 効いているか確かめる(30秒)

```bash
tail -f /workspace/out/pi05_full_prune.log     # 「監視開始:」の行が出ていれば動いている
pgrep -af prune_checkpoints.sh                 # プロセスが居るか
df -h /workspace                               # 空きが減り続けていないか
```

**消す物が無い回は何も書かない**ので、「監視開始:」の1行だけでも異常ではない。

### 止める

```bash
pkill -f prune_checkpoints.sh
```

**止めた時点の最新チェックポイントは残る。** 12節の持ち帰りが終わるまでは
止めなくてよい(2個残すので、最終歩は必ず残る)。

### 先に消える物を見たいとき

```bash
/opt/pi05/prune_checkpoints.sh /workspace/out/pi05_full 2 --once --dry-run
```

(3節の設定どおり`OUTPUT_ROOT=/workspace/out`にしていれば、この場所になる。
**`KEEP`は2未満にしないこと**——保存中の1個を消す競合と、
LeRobotが張る`checkpoints/last`のリンク切れが起きうる。)

## 12. 終わったら——チェックポイントを持ち帰る

**道A・道Bどちらでもscpで持ち帰れる**(道Aは5節でsshdを起動しているため。
道Bは公式テンプレートに最初から入っている)。このマシン側で実行する。

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r \
    root@<IP>:/workspace/out/pi05_full/checkpoints/030000/pretrained_model \
    ./work/train/pi05_full_runpod/
```

`<IP>`と`<port>`は、Connectタブに出る**公開IP側**(Full SSH)のもの。
RunPodのサーバを経由するBasic SSHのホスト名を使うとscpは動かない(5.3)。

**`pretrained_model`だけで8.7GBを転送するので回線次第で時間がかかる**
(`training_state`まで持ち帰る必要は無い。評価に使うのは`pretrained_model`である)。
途中で切れてやり直したくない
場合、あるいはscpが通らない場合は、Hugging Face Hubへ上げる方法もある
(7節で`hf auth login`は済んでいる前提)。

```bash
huggingface-cli upload <自分のHFユーザー名>/pi05_full_runpod \
    /workspace/out/pi05_full/checkpoints/030000/pretrained_model .
```

(`huggingface-cli`の代わりに新しい`hf upload`が使える版もある。どちらも
リポジトリが無ければ自動で作られる。**非公開にしたい場合は`--private`を付ける。**)

いずれの方法でも、**持ち帰りが終わってからポッドをStop(またはTerminate)して
課金を止めること。** Volume Diskは3節の表のとおりStopでは消えないが、
**Terminateすると消える**ので、Terminateする前には必ず持ち帰りを終えること。

## 13. 実際にやったこと —— **「30,000 ステップを最初から」は捨てた(2026-08-14)**

**このドキュメントの10節が書いている「最初から30,000 ステップ」は実行していない。**
9節の実測で前提が崩れたためである。**借りたA100は、この用途では
このマシンのH100-20Cより遅かった。**

| 動かした場所 | GPU | 構成 | batch | **`smp/s`** | `updt_s` | `mem_gb` |
|---|---|---|---|---|---|---|
| **このマシン**(段1) | H100-20C 20GB | 行動側だけ | 16 | **11.6** | 1.38 | 16.27 |
| RunPod | A100 80GB PCIe | 行動側だけ | 64 | **8.1** | 7.924 | 32.08 |
| **RunPod** | A100 80GB PCIe | **全部** | 64 | **6.2** | 10.289 | 48.76 |
| RunPod | A100 80GB PCIe | 全部 | 2 | 2 | 0.818 | 36.93 |
| RunPod | A100 80GB PCIe | 全部 | **128** | — | — | **足りない(75.77GBで落ちた)** |

**`smp/s`は「1秒あたり何コマ引けるか」で、これが学習に見せられる量の上限である。**
6.2 smp/sで10時間なら223,000コマ。**段1が既に384,000コマ引いているので、
最初から回しても段1に届かない。** ゆえに**段1の続きから足す形に変えた**
(`--init-from`。下のコマンド)。

```bash
python /opt/pi05/train_pi05.py --run-name pi05_full \
    --init-from /workspace/stage1 \
    --steps 3000 --batch-size 64 --num-workers 16 \
    --learning-rate 5e-6 --save-freq 500 \
    --freeze-vision-encoder false --train-expert-only false
```

**続きから始まっていることは、20 ステップ目の`loss`で分かる。** 最初から始めた回は
1.56〜2.25から下がり始めるが、この回は**`loss:0.234`**——段1が終わった水準で
始まった。ログの`All keys loaded successfully!`と合わせて2点で確かめられる。

### 上げすぎるとどこで落ちるか(実測した境界)

| 変えたこと | 結果 |
|---|---|
| batch 64・全部学習 | **通る。** `mem_gb` 48.76 |
| batch 128・全部学習 | **落ちる。** 3.78GiBを要求した時点で75.77GB使用済み。**batchに対してメモリの伸びが加速する**ので、64→128の直線外挿(61GB予想)は外れた |
| batch 64・行動側だけ・`--gradient-checkpointing false` | **落ちる。** 78.91GB。固定しても**画像側の前向き計算は必ず走る**ので、この節約は外せない |

### 学習率の下げ幅が段1と違う(**ここを読み違えると判断を誤る**)

`--learning-rate 5e-6`を渡すと`peak_lr`だけが変わり、**`decay_lr`は2.5e-6のまま
据え置かれる。** つまり**下がりきった値が始まりの半分**しかない。

    python3 -c "import json;print(json.load(open('<ckpt>/train_config.json'))['scheduler'])"
    # {'type': 'cosine_decay_with_warmup', 'num_warmup_steps': 1000,
    #  'num_decay_steps': 30000, 'peak_lr': 5e-06, 'decay_lr': 2.5e-06}

**表示は1000と30000だが、実際はステップ数に合わせて縮む**(ログの
`Scaling warmup: 1000 → 100, decay: 30000 → 3000`)。

| ステップ数 | 学習率 | 読み |
|---|---|---|
| 100 | 5.0e-6 | 暖機の終わり |
| 2,000 | 約3.2e-6 | **既に下がりきりに近い** |
| 3,000 | 2.5e-6 | 下がりきり |

**段1(2.5e-5→2.5e-6＝10分の1)とは違い、2倍の幅しか動かない。**
ゆえに**途中のステップ数のチェックポイントも、そのまま使える。**
0節が書いている「ステップ数を外すと下がりきらない」という警告は、
**`decay_lr`を据え置いた今回には当てはまらない。**

### 出来たチェックポイントは、段1のものと差し替えるだけで動く

**`model.safetensors`の中身の並びが段1と1つも違わない。**

| 見たこと | 結果 |
|---|---|
| 中身の個数 | **813個 対 813個** |
| 型と形 | **1つも違わない** |
| `config.json`の違い | `train_expert_only`と`freeze_vision_encoder`が`true`→`false`の**2つだけ**。どちらも学習のときだけ効く印である |

```bash
python3 - <<'EOF'
import json, struct
def keys(p):
    with open(p,'rb') as f:
        h = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
    h.pop('__metadata__', None); return h
a = keys('work/artifacts/model_weights_pi05_024000/model.safetensors')
b = keys('work/train/pi05_full_runpod/002000/model.safetensors')
print(len(a), len(b), set(a) ^ set(b), [k for k in a if a[k] != b[k]])
EOF
```

**したがって`policy_server.py`は1行も触らずに済む。**
取り込みは[install_pi05_weights.py](../../solution/tools/weights/install_pi05_weights.py)が
そのまま使える(トークナイザの`vlm/`を足すだけ)。

### 持ち帰りは`scp`ではなくHugging Face経由にした

**`scp`は1.7 MB/sしか出ず、8.8GBに1時間以上かかる。**
0節が「本番を待たずに一度試しておく」と書いたのは正しく、
**測った結果このやり方を捨てた。**

```bash
# Pod側(7節で hf auth login は済んでいる)
hf upload <HFユーザー名>/pi05_full_runpod \
    /workspace/out/pi05_full/checkpoints/002000/pretrained_model . --private

# このマシン側
hf download <HFユーザー名>/pi05_full_runpod \
    --local-dir work/train/pi05_full_runpod/002000
```

**9.4GBで16分。`scp`の4分の1以下である。**
しかも**中身が同じ部分は送り直さない**ので、ステップ数を変えて何度も持ち帰るときに差が開く。

> **`hf upload`が403で止まったら
> [08_hf_transfer_troubleshooting.md](08_hf_transfer_troubleshooting.md)。**
> 8/14の夜に実際に踏んだ。**`hf auth whoami`が通ることは書ける証拠にならない**
> ——`hf upload`は送る前に必ず「リポジトリを作る」APIを叩き、
> そこが権限で断られている。

### 学習ログの読み —— **段1は止まっていて、ここで頭打ちが破れた**

ログは**Podの`/workspace/out/pi05_full.log`**にある(`OUTPUT_ROOT`の直下、
run名の「隣」。`train_pi05.py`の`_resolve_log_path`)。**チェックポイントには
入らないので別に持ち帰る。** 3,000 ステップで45KBしかないので`scp`で十分である。

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
    root@<IP>:/workspace/out/pi05_full.log work/train/pi05_full_runpod.log

# 500 ステップごとの loss と grdn の平均を出す
grep -a "step:" work/train/pi05_full_runpod.log | sed 's/.*smpl:/smpl:/' | awk '
{ for(i=1;i<=NF;i++){ split($i,a,":"); v[a[1]]=a[2] }
  n++; blk=int((n-1)/25); ls[blk]+=v["loss"]; gs[blk]+=v["grdn"]; c[blk]++ }
END { for(b=0;b<=blk;b++) printf "%6d〜%-7d loss %.4f  grdn %.4f\n",
      b*500+1,(b+1)*500, ls[b]/c[b], gs[b]/c[b] }'
```

**段1は18,000 ステップを過ぎたところで下がるのをやめていた。**

| 段1のステップ数 | `loss` |
|---|---|
| 16,001〜18,000 | 0.2375 |
| 18,001〜20,000 | 0.2375 |
| 20,001〜22,000 | 0.2352 |
| **22,001〜24,000** | **0.2360(上がっている＝頭打ち)** |

そこへ画像を見る側を解いた結果である。

| RunPodのステップ数 | `loss` | `grdn` |
|---|---|---|
| 1〜500 | 0.2294 | 0.692 |
| 501〜1,000 | 0.2211 | 0.657 |
| 1,001〜1,500 | 0.2162 | 0.653 |
| 1,501〜2,000 | 0.2165 | 0.657 |
| 2,001〜2,500 | 0.2122 | 0.653 |
| **2,501〜3,000** | **0.2100** | **0.640** |

**0.236 → 0.210。11%下がり、3,000 ステップの時点でまだ下がり続けている。**
`grdn`は0.86 → 0.64(−26%)で、跳ねも`nan`も無い。**完走**(3,000/3,000・
8時間33分・`End of training`・終了コード0・`lr`は2.5e-06まで落ちきり・`epch:0.09`)。

**読み: 段1が抱えていた残りの誤差は、固定していた側にあった。**
下の重みの測定(画像側が0.34%動いた)と話が噛み合う。

> **`loss`の比べ方に1つ注意がある。** 段1はbatch 16、こちらはbatch 64なので、
> **`loss`は1回の推定の散らばりが違う**(64個の平均なので上下が小さい)。
> **期待値は比べられるが、「滑らかに見えるから安定した」とは読めない。**
>
> **そしてもっと大事な限界。学習の`loss`は本番スコアを予想する材料として
> いちばん弱い。** 手元の成功率でさえ本番を予想できないことが実測で分かっている
> ([02_build_submission.md](02_build_submission.md)の8/14 17:30の訂正)。
> `loss`が下がったのは**手本に近づいた**ことしか意味せず、
> **遅い手本にも近づいた**目が残る。**判断は評価と本番スコアで行う。**

### 画像を見る側が本当に動いたかを確かめる —— **`config.json`では足りない**

**`freeze_vision_encoder: false`は「そう指定した」記録であって、
「重みが動いた」証拠ではない。** 確かめるには**重みそのものを突き合わせる。**
素のπ0.5(`lerobot/pi05_libero_base`。14GBが`$HF_HOME`にある)を挟んで三つ巴で見る。

| | 素 → 段1 | 素 → **RunPod** |
|---|---|---|
| **画像を見る側**(437個・4.1億パラメータ) | **1ビットも違わない**＝**段1は本当に固定していた** | **437個すべて違う。動いた割合0.34%** |
| 画像→言語のつなぎ | 1ビットも違わない | **違う。1.02%** |

まとまりごとの動いた量(段1 → RunPod)。

| まとまり | 動いた割合(RMS) |
|---|---|
| **画像→言語のつなぎ** | **1.02%** |
| **画像を見る側 vision_tower** | **0.34%** |
| 言語側 gemma_2b | 0.08% |
| 行動を作る側 gemma_expert | 0.05% |

**画像側が行動側の4〜20倍動いている。** 段1で24,000 ステップぶん一度も触られなかった
側なので、そうなるのが筋である。

**「保存し直したから全部違って見えるだけ」ではない。** 動かなかった重みが
9個あり、その内訳が理屈と合う。

| 最後の層(`layers.17`)の中身 | 結果 |
|---|---|
| `k_proj` / `v_proj` | **動いた** |
| `q_proj` / `o_proj` / `mlp`の3つ | **動いていない** |
| `lm_head`(言語を出す頭)2つ | **動いていない** |

**行動を作る側が最後の層から受け取るのはKとVだけ**なので、同じ層の残りには
勾配が流れない。**流れる所だけが動き、流れない所は1ビットも動いていない**
——この比較は本物の学習を追えている。

```bash
timeout 400 python3 - <<'PY'
import json, struct
import numpy as np
def open_st(p):
    f=open(p,'rb'); n=struct.unpack('<Q',f.read(8))[0]
    h=json.loads(f.read(n)); h.pop('__metadata__',None); return f,8+n,h
def tens(f,o,m):
    s,e=m['data_offsets']; f.seek(o+s); b=f.read(e-s)
    if m['dtype']=='F32': return np.frombuffer(b,dtype=np.float32)
    return (np.frombuffer(b,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
f1,o1,h1 = open_st('work/artifacts/model_weights_pi05_024000/model.safetensors')
f2,o2,h2 = open_st('work/train/pi05_full_runpod/002000/model.safetensors')
ks=[k for k in h1 if 'vision_tower' in k]
num=den=0.0
for k in ks:
    a=tens(f1,o1,h1[k]); b=tens(f2,o2,h2[k])
    num+=float(((b-a)**2).sum()); den+=float((a**2).sum())
print(len(ks), f'{100*np.sqrt(num/den):.2f}%')
PY
```

**素の重みと比べるときは名前がずれる。** 素は`paligemma_with_expert.…`で、
学習が吐いた側は先頭に`model.`が付く(`model.paligemma_with_expert.…`)。
**先頭6文字を落とせば対応する。** 言語側まで全部比べると14GBを読むので数分かかる
——**画像側だけなら1分で済む。**

> **0.34%が「光に強くなるのに足りる量か」は、この数字からは分からない。**
> 動いたことは確実だが、動いた向きが良いかどうかは**評価しか答えない。**

### この重みで何を狙っているか

**段1は画像を見る側を固定して学習した。この回だけがそこを動かしている。**
36件の評価で分かった弱点のうち、**設定値では動かない7件**
(光への強さ・黒椀の見分け・2段階の課題。
[../reference/18_failure_breakdown.md](../reference/18_failure_breakdown.md))は
画像を見る側の話なので、**狙いはそこである。**

| | 段1(このマシン) | **RunPod** |
|---|---|---|
| 動かしたパラメータ | 693M | **4.14B** |
| `freeze_vision_encoder` | `true` | **`false`** |
| 引いたコマ | 384,000 | **＋192,000(段1の1.5倍になる)** |

**ただし192,000コマはデータ全体の0.086周にすぎない。** 効く保証は無く、
**元々持っていた見る力を忘れる目もある。** 判断は本番スコアで行う
([../talk/260814_today.md](../talk/260814_today.md)の18:10の節)。

## 確認できたこと(2026-08-14 に道Aで実際に動かして)

| 事柄 | 結果 |
|---|---|
| sshdとtmuxを足したDockerfileがビルドできるか | **できた。** Docker Hubへpushし、そのイメージのPodが動いた |
| Docker Hubのカスタムイメージが「New Template」でそのまま認識されるか | **された** |
| `/workspace`にボリュームを付けたとき、イメージの`/opt/pi05/`の中身が隠れないか | **隠れない。** Podの中で`/opt/pi05/prune_checkpoints.sh`をそのまま実行できた |
| フルパラメータ学習が実際に始まるか | **始まった。** 500 ステップまで到達し、チェックポイントが保存された(1個25GB) |
| **`--init-from`で段1の続きから始められるか** | **できた。** `All keys loaded successfully!`が出て、20 ステップ目の`loss`が**0.234**(最初から始めた回は1.56〜2.25)。3,000 ステップ・batch 64で約8時間30分 |
| 150GBで学習が最後まで持つか | **持った。**2,000 ステップの保存まで到達している。**ただし何個残す設定で回したかは記録していない**ので、間引きの余裕そのものは測れていない |
| `runpod_ssh_start.sh`で起動したコンテナにSSHで入れるか(公開鍵が`PUBLIC_KEY`で渡ってくるか、割り当てられたポートが22に向くか) | **入れた。** 「Expose TCP Ports」に`22`を足しておけば、Connectタブの公開IPとポートでそのまま入れる |

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| `sleep infinity`で待っているコンテナで、Web Terminalも問題なく開けるか | デプロイ後に実際に押してみる |
| **フルパラメータ学習が、行動側だけより実際にスコアで勝るか** | **これだけが残っている。**両方を`policy_server.py`経由で採点に出して比べる以外に確かめる方法が無い(手元の36件では順位が付かないことが既に分かっている) |

**上の表から消した項目**(13節で実測して決着した)。

| かつて未確認だったこと | 分かったこと |
|---|---|
| scpの実効速度 | **1.7 MB/s。8.8GBに1時間以上。**使うのをやめてHugging Face経由(16分)にした |
| 80GBでbatch_size=64が収まるか | **収まる**(`mem_gb` 48.76)。**128は落ちる**(75.77GB) |
| `hf upload`が8.7GB級で完走するか | **完走する。9.4GBで16分。**分割の設定は要らない |
| 出来たチェックポイントを`policy_server.py`が読めるか | **読める。**中身の並びが段1と完全に同じ(813個・型も形も一致) |
