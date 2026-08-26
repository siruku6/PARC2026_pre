# Blackwell サーバの最初の 1 セッション（下見と配布物の確認）

**このファイルだけで作業が終わるように書いた。**別のファイルを開く必要はない。

対象は運営が配布する GPU 環境（**Blackwell サーバ**・NVIDIA RTX PRO 6000 Blackwell 1 台）。
**本選 1 の期間全体で 60 時間しか使えず、起動してから自分で止めるまでずっと減る。**
この作業は合計 **約 40 分**を使う。

作成: 2026-08-26。出典は
[本選の配布資料 v1.1](../materials/PARC2026開発コンペティション_本選_v1.1.pdf)（2026-08-26 公開）。

| # | 手順 | 所要 | 何が分かるか |
|---|---|---|---|
| 0 | [入る](#手順-0-入る) | 5 分 | — |
| 1 | [環境を数える](#手順-1-環境を数える5-分) | 5 分 | VRAM・pip で足すもの・永続化の道具 |
| 2 | [配布データの中身を数える](#手順-2-配布データの中身を数える10-分) | 10 分 | タスク数・本数・fps・本選で増えたデータ |
| 3 | [配布ベースラインの中身を読む](#手順-3-配布ベースラインの中身を読む5-分) | 5 分 | 運営の既定の推論設定 |
| 4 | [提出前チェック用 Docker と採点環境の版](#手順-4-提出前チェック用-docker-と採点環境の版5-分) | 5 分 | 枠を使わずに zip を検査する道 |
| 5 | [止める](#手順-5-止める) | 1 分 | — |

読み終えたときに分かること: **Notebook の中で学習を回すために何を pip で足すのか、
配布データを学習に使うべきか、そして提出前に zip を検査できるか。**

## なぜ「下見」に 40 分も払うのか

**2026-08-26 の資料 v1.1 で、自分の Docker イメージを持ち込む道が閉じた。**

| | v1.0（8/24） | **v1.1（8/26）** |
|---|---|---|
| 自分のイメージ | 「Docker は使用可能」 | **サポート外。**「使用不可であることを想定しており、もし使用しても運営では対応しかねます」 |
| 使えるもの | — | **Omnicampus 提出の事前チェック用 Docker の `docker run` だけ** |

**イメージを持ち込めるなら、中に何が入っているかを自分で決められるので、素の環境が
どうなっていようと関係なかった。** 閉じた以上、Notebook の Python 環境へ直に pip で
足すしかなく、**そのためには最初から何が入っているかを正確に知る必要がある。**
torch の版が 1 つ違うだけで torchcodec や torchvision が読めなくなる。

## 時間の減り方（先に頭に入れる）

| 止まり方 | 条件 |
|---|---|
| **自分で止める** | `File > Hub Control Panel > Stop Server`。**押した瞬間に時間の消費が止まる** |
| 待機による自動停止 | セルの実行が何も無い状態が **1 時間**続いたとき |
| 起動時間による自動停止 | 起動から **12 時間** |

**本選 1 では「延長」ボタンが使えない設定になっている。**1 セッションで連続して回せるのは
最大 12 時間である。**ブラウザのタブを閉じても時間は減り続ける。**

画面の残り時間は 2 つある。**緑枠が今回のセッションの 12 時間に対する残り、青枠が
配布された 60 時間に対する残り**で、青枠は起動時点の値のまま更新されない。

## 手順 0 入る

1. `https://parc-playground.omnicamp.us/hub/login` を開く
2. `Sign in with Auth0` をクリック
3. Omnicampus のアカウントとパスワードでログイン
4. GPU の割り当てが終わるまで待つ（**割り当てられるリージョンは複数あり、`us-east-1`
   とは限らない**）
5. Notebook の画面に変わったら起動完了

**この時点から時間が減り始める。**

## 手順 1 環境を数える（5 分）

実行場所: **Notebook のセル**（先頭に `!` を付ける、または `%%bash` のセルにする）。

**出力は `~/recon.txt` にも落とす。`$HOME` は 10 分ごとと終了時に自動保存され、
次回起動時に復元されるので、貼り忘れても残る。**

```bash
{
echo "### GPU"; nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
echo "### CPU/RAM"; nproc; free -g | head -2
echo "### disk"; df -h $HOME ~/data /opt/dlami/nvme 2>&1
echo "### python"; python -V; which python pip
echo "### 主要ライブラリ"; pip list 2>/dev/null | grep -iE "^(torch|torchvision|torchaudio|torchcodec|transformers|lerobot|accelerate|safetensors|peft|numpy) "
echo "### docker"; docker images 2>&1 | head
echo "### 永続化の道具"; which parc-home-sync; ls ~/data 2>&1 | head
} 2>&1 | tee ~/recon.txt
```

**一番知りたいのは VRAM である。** RTX PRO 6000 Blackwell の製品仕様は 96 GB なので、
**そのとおりなら π0.5 のフルパラメータ学習が載り、RunPod を借りる必要が無くなる。**
手元の H100-20C（20 GB）では行動側だけの学習しか載らない。

**pip の一覧も同じくらい大事である。** torch が入っていれば、その版に合う
torchvision / torchcodec を選ぶことになる（版がずれると `undefined symbol` や
`Could not load libtorchcodec` で落ちる）。

## 手順 2 配布データの中身を数える（10 分）

**これがこのセッションの主目的である。** 配布ベースラインが学習に使った
`libero_combined_20hz`（統合 20Hz データセット）の中身を、**展開せずに**見る。
展開すると一時領域を大量に使い、時間も食う。

```bash
{
echo "### dataset の並び"; find ~/dataset -maxdepth 3 | head -60
echo "### 容量"; du -sh ~/dataset/* 2>&1
echo "### tar の中身（先頭だけ）"
for t in $(find ~/dataset -name "*.tar*" | head -5); do
  echo "--- $t"; tar -tf "$t" 2>/dev/null | head -20; echo "  総エントリ数: $(tar -tf "$t" 2>/dev/null | wc -l)"
done
} 2>&1 | tee ~/recon_dataset.txt
```

**分かること 4 つ。**

| 見るもの | なぜ知りたいか |
|---|---|
| タスク数（**40 か 113 か**） | 手元の `lerobot/libero_plus`（14,347 本）と同じ母集団かどうか |
| エピソードの本数 | 学習データを差し替える価値があるか |
| `fps` が本当に 20 か | 「20Hz で統合した」の意味の裏取り |
| **本選で新しく増えたデータがあるか** | あるなら Track2・Track3 の対策が変わる |

`~/dataset` は**読み出し専用**である。使うときは `~/data` へ展開する。

## 手順 3 配布ベースラインの中身を読む（5 分）

**運営自身の実装なので、こちらの `policy_server.py` と比べる価値がある。**
この重みのリーダーボード参考スコアは Track1 0.286 / Track2 0.165 / Track3 0.000 で、
**こちらの 4.2.0（Track1 0.370 / Track2 0.154）と同じ土俵の数字である。**

```bash
{
unzip -l ~/dataset/pi05_step005000_submission_py310.zip | head -30
unzip -p ~/dataset/pi05_step005000_submission_py310.zip requirements.txt
unzip -p ~/dataset/pi05_step005000_submission_py310.zip policy_server.py | head -120
} 2>&1 | tee ~/recon_baseline.txt
```

**分かること。** 運営が既定にしている chunk 長・ノイズ除去の回数・後処理の有無。
**0.286 を出した設定そのものである。**

## 手順 4 提出前チェック用 Docker と採点環境の版（5 分）

**これは唯一サポートされている Docker で、リーダーボードの枠を使わずに zip を検査できる。**
枠は 1 日 1 トラック 1 つしか無く、エラーでも 1 つ消えるので、**使えるなら提出事故を
そのまま防げる。**

```bash
{
echo "### README の場所"
find ~ -maxdepth 3 -name "README*" 2>/dev/null | head
echo "### 採点環境の節"
grep -rn -A 40 "採点環境" ~/README.md ~/*/README.md 2>/dev/null | head -60
echo "### 事前チェック用イメージ"
docker images 2>&1
} 2>&1 | tee ~/recon_kit.txt
```

**分かること。** 固定されている CUDA / PyTorch / ライブラリの版（v1.1 が「READMEの
『採点環境』を参照」と書いている先）と、事前チェックの叩き方。

## 手順 5 止める

`File > Hub Control Panel > Stop Server` を押す。**押した瞬間に配布時間の消費が止まる。**

**ブラウザを閉じるだけでは止まらない。**

## 終わったら

`~/recon.txt`・`~/recon_dataset.txt`・`~/recon_baseline.txt`・`~/recon_kit.txt` の
4 つを貼る。**これで「Notebook で学習を回す手順（pip で何を足すか、12 時間に何歩
入るか）」を書ける。**

## 確認したい仮定

| 仮定 | 根拠 | 外れると何が変わる |
|---|---|---|
| **VRAM は 96 GB** | NVIDIA の製品仕様。**配布資料には書かれていない** | 少ないなら、フルパラメータ学習は RunPod のままになる |
| `pi05_step005000_submission_py310.zip` に `policy_server.py` が入っている | v1.0 の記述（「重み・ポリシーサーバー・`requirements.txt` 入り」） | 手順 3 の 3 行目が空振りする。`unzip -l` の一覧を見て名前を直す |
| ネットワークに出られる（HF から重みを落とせる） | 未確認 | 重みを `$HOME` 経由で持ち込む必要がある（40 GiB しかないので 1 本ずつ） |

## あとで見る資料

| 知りたいこと | ファイル |
|---|---|
| どのマシンに何があるか・60 時間の減り方 | [../resources/infra_and_docker.md](../resources/infra_and_docker.md) |
| 会社の RTX6000 へ評価環境を持ち込む | [12_bring_eval_to_rtx6000.md](12_bring_eval_to_rtx6000.md) |
| 流した学習・重み・版・出した点 | [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) |
| 学習データ 14,347 本の中身 | [../reference/06_training_data_composition.md](../reference/06_training_data_composition.md) |
