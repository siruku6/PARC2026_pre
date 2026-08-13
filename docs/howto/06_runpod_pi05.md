# 06. RunPodでπ0.5をフルパラメータ学習する

ローカル(H100 20GB)は`train_expert_only`で凍結した縮小版しか載らない。RunPodで
80GBクラスのGPUを借りれば、公式レシピどおりのフルパラメータ学習(VLMも含めて
全部学習)を試せる。ローカルの学習はそのまま保険として並行して回し、こちらは
上振れ狙いの追加の一手として位置付ける。

いきなり数時間かけて本番を回すのではなく、**少量データで数十歩だけ動かして
「フルパラメータ学習が本当に通るか」を確かめてから**、バッチサイズを探り、
最後に本番の歩数へ進む——という順で書く。

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

**Aは3〜4節、Bは5節でポッドを作る。どちらを選んでも、6節(ライセンス同意)以降は
共通の手順になる。**

## 3. イメージをビルドしてpushする(道A)

このマシン(ローカル)で、ローカル用に使ったのと同じイメージをそのまま使う。
**Dockerfile自体は変更不要**——ローカルの`--user`回避(`/hf_home`・`HOME=/tmp`)は
RunPod側では効果自体が要らなくなる(Podは既定でrootのまま動くため)が、
入っていても害はないのでそのまま流用できる。

```bash
docker build -t <Docker Hubのユーザー名>/parc2026-train-pi05:v1 \
    -f solution/train/Dockerfile.pi05 solution/train

docker login   # 初回だけ。Docker Hubのアカウントが要る(無料で作れる)
docker push <Docker Hubのユーザー名>/parc2026-train-pi05:v1
```

**RunPodはlinux/amd64のイメージを前提にしている。** このマシンがx86_64
(Ubuntu)であれば意識しなくてよいが、Apple Silicon等でビルドした場合は
`docker buildx build --platform linux/amd64 ...`が要る。

## 4. ポッドを作る(道A)

1. RunPodの「Templates」→「New Template」で、3節でpushしたイメージ名
   (`<ユーザー名>/parc2026-train-pi05:v1`)を指定する。
2. **Container Diskは60GB以上にする。** 内訳の目安:
   初期重み(実測で1チェックポイント約11GB)＋データセット(全部使うと
   [約16GB](../../solution/train/train.py)、`train.py`の docstring による)
   ＋間引いて残す直近2つのチェックポイント(11GB×2=22GB)＋lerobot本体や
   pipのキャッシュ(数GB)で、**合計おおよそ55GB前後になる。** 余裕を見て60GB以上。
3. **Container Start Commandを`sleep infinity`で上書きする。** 空のままだと
   イメージの既定CMD(`python /workspace/train_pi05.py --smoke-test`)が
   デプロイ直後に動き出してしまうが、**5節のライセンス同意より前に初期重みの
   取得を試みて確実に失敗する。** `sleep infinity`にしておけばコンテナは
   何もせず起動し続けるので、Web Terminalから入って手動で進められる。
4. 「Deploy」→「GPU Pod」で、作ったテンプレートを選び、GPUは
   **A100 80GBまたはH100 80GB(Secure Cloud)**を選ぶ。
5. デプロイ後、Pod管理画面から**「Enable Web Terminal」→「Open Web Terminal」**
   を押すと、SSH鍵を用意しなくてもブラウザから直接コマンドを打てる。
   最初はこれで十分——ファイル転送が必要になったら、
   [公式のSSH手順](https://docs.runpod.io/pods/configuration/use-ssh)に従って
   フルSSH(鍵認証・scp対応)を別途設定する(**このイメージにはsshdを入れていない**
   ため、フルSSHが要るならDockerfile側の対応が別途要る。11節で代わりの持ち帰り方を書く)。

## 5. ポッドを作る(道B)

1. 「Deploy」→「GPU Pod」で、テンプレートは公式の**「RunPod PyTorch」系**
   (SSHが最初から使える状態で入っている)を選ぶ。
2. GPUは**A100 80GBまたはH100 80GB(Secure Cloud)**。ディスクサイズの考え方は
   4節と同じ(60GB以上)。
3. デプロイ後、「Connect」タブに出るSSHコマンド
   (`ssh root@<IP> -p <port> -i ~/.ssh/id_ed25519`)で接続する。無ければ
   [公式のSSH手順](https://docs.runpod.io/pods/configuration/use-ssh)に従って
   鍵を作る。**公式テンプレートはフルSSH(scp対応)が最初から設定済み**なので、
   道Aと違って11節の持ち帰りで追加設定は要らない。
4. このリポジトリから4ファイルを転送する。

   ```bash
   scp -P <port> -i ~/.ssh/id_ed25519 \
       solution/train/train.py solution/train/train_pi05.py \
       solution/train/runpod_setup.sh solution/train/prune_checkpoints.sh \
       root@<IP>:/workspace/
   ```
5. ポッド側で依存関係を入れる。

   ```bash
   cd /workspace
   bash runpod_setup.sh
   ```

## 6. ライセンス同意とログイン

π0.5はgemmaライセンス配布。**道A・道Bどちらでも共通の手順。**

```bash
hf auth login
```

未同意なら https://huggingface.co/google/paligemma-3b-pt-224 を先に開いて
同意しておくこと。

## 7. 少量データで、まずフルパラメータ学習が通るか確かめる

**いきなり本番(14,347エピソード全部・30,000歩)を回さない。** `--episodes-per-task`
(`train.py`から再利用している、`train_pi05.py`が既に対応済みの引数)で
データを絞り、**VLMを含めた全パラメータの学習ループが、そもそもエラーなく
1周するか**を数十秒で確認する。

```bash
python /workspace/train_pi05.py --run-name pi05_small_check --steps 20 --batch-size 2 \
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

## 8. 適正なバッチサイズを探す

ローカルで使ったのと同じ考え方——**学習ログ自身が出す`mem_gb`を見比べる**。
公式ドキュメントは80GBでbatch_size=64を挙げているが、それが**本当にこの
環境で収まるか**、**もっと上げられる余地がないか**を実測する。

```bash
for bs in 32 64 96; do
    python /workspace/train_pi05.py --run-name "pi05_bs_check_${bs}" \
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

## 9. 本番のフルパラメータ学習を始める

バッチサイズが決まったら、本番へ進む。**Web Terminal/SSHの接続が切れても
プロセスが生き残るよう、`tmux`か`screen`の中で実行すること**
(ベースイメージに無ければ`apt-get install -y tmux`)。

```bash
tmux new -s train
cd /workspace
python train_pi05.py --run-name pi05_full --steps 30000 --batch-size 64 \
    --num-workers 8 --save-freq 5000 \
    --freeze-vision-encoder false --train-expert-only false
```

`--batch-size 64`は公式ドキュメント([lerobot/pi05](https://huggingface.co/docs/lerobot/en/pi05))
の値。8節の実測でもっと上げられると分かればそちらに変える。歩数(30000)は
公式ドキュメントがそのまま挙げている値——80GB GPU 1枚を前提にした数字。

`tmux`から抜けるには`Ctrl+b` → `d`(プロセスは動き続ける)。切断されても
再接続して`tmux attach -t train`で戻れる。

## 10. チェックポイントの間引き(ローカルと同じ問題がある)

チェックポイント1個は約11GBで、LeRobotは古いものを自動で消さない。
**別のWeb Terminal(またはSSHセッション)**を開いて、並行して回す。

```bash
tmux new -s prune
bash /workspace/prune_checkpoints.sh /out/pi05_full 2
```

(`/out`は`train_pi05.py`の既定の出力先。)

## 11. 終わったら——チェックポイントを持ち帰る

**持ち帰り方は、道Aと道Bで変わる。**

**道B(公式テンプレート)は、フルSSHが最初から使えるのでscpでそのまま持ち帰れる。**

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r \
    root@<IP>:/out/pi05_full/checkpoints/030000/pretrained_model \
    ./work/train/pi05_full_runpod/
```

**道A(カスタムイメージ)はsshdを入れていないため、上のscpはそのままでは使えない。**
Hugging Face Hubへ直接アップロードする方が確実(`6節`で`hf auth login`は
済んでいる前提)。

```bash
huggingface-cli upload <自分のHFユーザー名>/pi05_full_runpod \
    /out/pi05_full/checkpoints/030000/pretrained_model .
```

(`huggingface-cli`の代わりに新しい`hf upload`が使える版もある。どちらも
リポジトリが無ければ自動で作られる。)

いずれの方法でも、**持ち帰りが終わってからポッドをStop(またはTerminate)して
課金を止めること。** Volume Diskを使わずContainer Diskだけで進めた場合、
Stopしただけでも中身は消える(Terminateと同じ扱い)ので注意。

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| Docker Hubへpushしたカスタムイメージが、RunPodの「New Template」画面でそのまま認識されるか | 実際にテンプレートを作って試す |
| `sleep infinity`で起動したコンテナで、Web Terminalが問題なく開けるか | デプロイ後に実際に押してみる |
| 80GBで本当にbatch_size=64、あるいはそれ以上が収まるか | 8節の実測 |
| `huggingface-cli upload`が、11GB級のディレクトリでも問題なく完走するか(タイムアウト・分割の有無) | 実際にアップロードして試す |
| フルパラメータ学習が、train_expert_onlyより実際にスコアで勝るか | 両方のチェックポイントをpolicy_server.py経由で評価して比べる以外に確かめる方法が無い |
