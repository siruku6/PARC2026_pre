# 06. RunPodでπ0.5をフルパラメータ学習する

ローカル(H100 20GB)は`train_expert_only`で凍結した縮小版しか載らない。RunPodで
80GBクラスのGPUを借りれば、公式レシピどおりのフルパラメータ学習(VLMも含めて
全部学習)を試せる。ローカルの学習はそのまま保険として並行して回し、こちらは
上振れ狙いの追加の一手として位置付ける。

いきなり数時間かけて本番を回すのではなく、**少量データで数十歩だけ動かして
「フルパラメータ学習が本当に通るか」を確かめてから**、バッチサイズを探り、
最後に本番の歩数へ進む——という順で書く。

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
[10節](#10-本番のフルパラメータ学習を始める)が挙げている30,000歩は
公式ドキュメントの値であって、**この時間に収まる保証は無い。歩数は
9節で`smp/s`を測ってから、下の式で決めること。**

    歩数 = 実測smp/s × 使える秒数 ÷ batch

**歩数を多すぎる側に外すと、途中で止めたときに学習率が下がりきらない。**
`--steps`は学習率の下げ方そのものを決める値で、`CosineDecayWithWarmup`は
指定した歩数に合わせて形を伸縮させる。30,000歩の指定で15,000歩で止めると、
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

3節はローカルと同じ`train_expert_only`を前提に60GBと書いていたが、
**フルパラメータ学習ではチェックポイントが大きくなる。**

| | ローカル(expert only) | フルパラメータ(推定) |
|---|---|---|
| 学習するパラメータ | 693M | 4.14B |
| `pretrained_model` | **8.7GB**(実測。全パラメータを保存するので変わらない) | 8.7GB |
| `training_state` | **2.1GB**(実測。1パラメータあたり3.0バイト) | **約12GB**(同じ比率での推定) |
| **1個あたり** | **11GB** | **約21GB** |

**合計の見積もり: HFキャッシュ約45GB(データ30＋初期重み14)＋
チェックポイント2個で42GB ＝ 約87GB。** 余裕を見て**150GB**にする
($0.10/GB/月なので、1日なら数十円の差にしかならない)。

**`training_state`の12GBは推定である。** 8節の20歩の確認が終わった直後に
`du -sh /workspace/out/pi05_small_check/checkpoints/*/*` で実測し、
足りなければその場でVolumeを増やすこと。

### 持ち帰りは、本番を待たずに一度試しておく

**8.7GBのscpが何分かかるかを、22時に初めて知るのは避ける。**
8節の20歩の確認でチェックポイントが1個できるので、**そのまま
12節のscpを実行して時間を測る。** これは同時に
**「フルパラメータ学習が吐いたチェックポイントを`policy_server.py`が
読めるか」の確認**にもなる(版3.8.0は`config.json`の`type`を見て
π0.5の道へ入る。`train_expert_only`が`false`でも中身の形は変わらない
はずだが、**確かめていない**)。

### CPUの数を見てから`--num-workers`を決める

このマシンは`nproc`が**2**しかなく、`--num-workers 2`でも
`data_s` 0.011〜0.020秒／`updt_s` 1.39秒と、データ待ちは無視できていた。
**80GBのGPUでbatchを64に上げると、1歩が短くなる分だけ動画の展開が
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
- **Volume Disk: 60GB以上。** 内訳はおおよそ、初期重み(実測11GB)＋
  データセット(全部使うと約16GB。[train.py](../../solution/train/train.py)の
  docstringに実測値がある)＋間引いて残す直近2チェックポイント(11GB×2=22GB)
  ＋lerobot本体やpipのキャッシュ数GBで、合計55GB前後になる

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
2. **Container Diskは20GB程度、Volume Diskは60GB以上**にする(3節)。
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
2. **Container Diskは20GB程度、Volume Diskは60GB以上**にする(3節)。
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

**いきなり本番(14,347エピソード全部・30,000歩)を回さない。** `--episodes-per-task`
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

**フルパラメータ学習は、凍結する縮小版よりメモリの伸び方が急なはずである**
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
の値。9節の実測でもっと上げられると分かればそちらに変える。歩数(30000)は
公式ドキュメントがそのまま挙げている値——80GB GPU 1枚を前提にした数字。

`tmux`から抜けるには`Ctrl+b` → `d`(プロセスは動き続ける)。切断されても
再接続して`tmux attach -t train`で戻れる。

## 11. チェックポイントの間引き——**そのまま貼れる形**

チェックポイント1個は**フルパラメータ学習だと約21GB**(0節の推定)で、
LeRobotは古いものを自動で消さない。**放っておくと必ずディスクが尽きて
学習が落ちる。**

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

**11GBを転送するので回線次第で時間がかかる。** 途中で切れてやり直したくない
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

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| sshdとtmuxを足したDockerfileが実際にビルドできるか(このマシンでは`docker build`の実行が許可されていないため未確認) | 4節のコマンドでビルドしたあと、下の「ビルドできたか確かめる」を実行する |
| Docker Hubへpushしたカスタムイメージが、RunPodの「New Template」画面でそのまま認識されるか | 実際にテンプレートを作って試す |
| Volume Diskを`/workspace`に付けたとき、イメージの`/opt/pi05/`の中身(`/workspace`の外)が本当に影響を受けないか | 実際にPodを作って`ls /opt/pi05/`を確認する |
| `runpod_ssh_start.sh`で起動したコンテナに、実際にSSHで入れるか(公開鍵が`PUBLIC_KEY`で渡ってくるか、割り当てられたポートが22に向くか) | デプロイ後に`ssh root@<IP> -p <port>`を試す。Logsの`[runpod_ssh_start]`行も見る |
| `sleep infinity`で待っているコンテナで、Web Terminalも問題なく開けるか | デプロイ後に実際に押してみる |
| 80GBで本当にbatch_size=64、あるいはそれ以上が収まるか | 9節の実測 |
| `huggingface-cli upload`が、11GB級のディレクトリでも問題なく完走するか(タイムアウト・分割の有無) | 実際にアップロードして試す |
| フルパラメータ学習が、train_expert_onlyより実際にスコアで勝るか | 両方のチェックポイントをpolicy_server.py経由で評価して比べる以外に確かめる方法が無い |
