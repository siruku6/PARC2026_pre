# 09. ディスクを空ける

**このマシンの `/` は 298 GB しかなく、提出 zip 1 本 9.4 GB・π0.5 のチェックポイント 1 本
8.8〜11 GB を置くので、数本で埋まる。** 2026-08-14 には空きが尽きて `policy_server.py` が
0 バイトになり、走っていた評価が黙って死んだ（[../knowledge/12_workbench_accidents.md](../knowledge/12_workbench_accidents.md)）。

**この文書を読むと、消してよいものを取り違えずに選べて、「消したのに空かない」3 つの罠を
避けられる。** 今どれだけ空いているかはここには書かない（`df -h /` で分かる）。

| # | 節 | 結論 |
|---|---|---|
| 1 | [まず 4 か所を測る](#1-まず-4-か所を測る) | 太るのは学習成果物・HF キャッシュ・docker の 4 か所だけ |
| 2 | [同じ中身かはパス名ではなく sha256 で判定する](#2-同じ中身かはパス名ではなく-sha256-で判定する) | 「HF に無い」と書いてあった 26 GB が、全部よそにあるコピーだった |
| 3 | [ハードリンクは両方消して初めて空く](#3-ハードリンクは両方消して初めて空く) | 片方だけ消しても 1 バイトも空かない |
| 4 | [docker はイメージとビルドキャッシュの両方を消す](#4-docker-はイメージとビルドキャッシュの両方を消す) | 片方だけだと、申告 9.54 GB に対して 1.1 GB しか空かなかった |
| 5 | [HF の非公開容量は画面で消しても空かない](#5-hf-の非公開容量は画面で消しても空かない) | git 履歴に実体が残る。潰すしかない |
| 6 | [消してよいものと残すもの](#6-消してよいものと残すもの) | 軸は「取り直せるか・作り直せるか」だけ |
| 7 | [消す作業は script にして流す](#7-消す作業は-script-にして流す) | 既定は表示だけ、`CONFIRM=yes` で実行 |

## 1. まず 4 か所を測る

```bash
cd /home/ubuntu/lab/PARC2026_pre
df -h /                                     # 全体の空き
du -sh work/* | sort -h | tail              # 学習の成果物
du -sh ~/.cache/huggingface/* | sort -h     # 上流の重みと学習データ
docker system df -v                         # イメージとビルドキャッシュ
```

| 場所 | 入っているもの | どの節で判定するか |
|---|---|---|
| `work/train/`・`work/artifacts/` | 学習チェックポイント。1 本 8.8〜11 GB | 2 節・3 節 |
| `~/.cache/huggingface/hub` | 上流の素の重み（`pi05_libero_base` が 14 GB） | 6 節 |
| `~/.cache/huggingface/lerobot` | 学習データ本体（15 GB） | 6 節（触らない） |
| docker | イメージとビルドキャッシュ | 4 節 |

## 2. 同じ中身かはパス名ではなく sha256 で判定する

**パス名や資料の記述では「ほかに控えが無い」を判定できない。** 2026-08-23 に消した
`work/train/pi05_compile_infer_eval`（26 GB）は、資料に長く「HF に無い」と書かれていたが、
重み 3 本の sha256 を全バイト計算すると、3 本とも既にある重みのコピーだった。

| 中身の sha256（先頭 8 桁） | 同じものがある場所 |
|---|---|
| `89fc796d` | ローカルの `work/train/pi05_quant_c0` |
| `0c5d542e` | HF `siruku6/pi05_full_runpod`（予選の最終提出の重み） |
| `d46bffcf` | HF `siruku6/pi05_stage1_24000` |

```bash
stat -c '%s %n' work/train/<消す候補>/model.safetensors   # まずバイト数で絞る
sha256sum work/train/<消す候補>/model.safetensors         # 一致した候補だけ計算する
```

**バイト数が違えば中身も違う**ので、先に `stat` で絞る。このマシンは 2 コアなので、
`sha256sum` は 8.8 GB 1 本で 2〜3 分かかる（2026-08-23 に `pi05_quant_c0` の
`016000/pretrained_model/model.safetensors` で実測）。既知の重みの
sha256 は [../reference/19_weights_catalog.md](../reference/19_weights_catalog.md) の一覧の
右端の列にある。HF 側は LFS の `sha256` を API から引ける（同じ文書の「引き直すコマンド」の節）。

## 3. ハードリンクは両方消して初めて空く

**同じ実体を 2 つの名前で指している重み（ハードリンク）がある。** 片方だけ消しても
1 バイトも空かない。

```bash
stat -c '%i %h %s %n' <パス>     # 実体につく通し番号（inode）・リンク数・バイト数
find work -samefile <パス>       # 同じ実体を指す別名を探す
```

**リンク数（2 番目の値）が 2 以上なら、どこかに別名がある。** 2026-08-23 に実測した 2 組:

| 同じ実体を指す 2 つの名前 | inode | リンク数 |
|---|---|---|
| `work/artifacts/pi05_rp3000/model.safetensors` ＝ `work/train/pi05_full_runpod/003000/model.safetensors` | 805401061 | 2 |
| `work/artifacts/model_weights_pi05_024000/model.safetensors` ＝ `work/train/pi05_full_local/checkpoints/024000/pretrained_model/model.safetensors` | 25474754 | 2 |

**逆の場合もある。**sha256 が一致するのにリンク数が 1 なら、同じ中身を別の実体として
2 つ持っている。こちらは片方消すだけでその分が空く。

## 4. docker はイメージとビルドキャッシュの両方を消す

**`docker system df` の `SIZE` は共有している分を二重に数えている。** イメージを消して
実際に空くのは、`-v` を付けたときの `UNIQUE SIZE` の分だけである。

| 何をしたか | `docker system df` の申告 | 実際に空いた分 |
|---|---|---|
| 2026-08-23 にイメージ 2 本を消した | 9.54 GB | **約 1.1 GB** |

**残りが空かなかったのは、ビルドキャッシュ側が同じレイヤを押さえていたためである。**
`Build Cache` の `RECLAIMABLE` が 1.135 GB から 9.541 GB へ動いただけだった。
**イメージを消したあとにビルドキャッシュも消さないと空かない。**

```bash
docker system df -v        # イメージごとの SHARED SIZE と UNIQUE SIZE を見る
docker builder prune -f    # そのあとでビルドキャッシュを捨てる
```

**逆の順番だと空かない。** 2026-08-18 に `docker builder prune --force` を先に流したときは、
現存イメージが同じレイヤを `SHARED` として押さえていて、`RECLAIMABLE` が 0 だったため
1 バイトも空かなかった。

## 5. HF の非公開容量は画面で消しても空かない

**「容量上限に達しました」と出たので画面からファイルを消しても、使用量は減らない。**
git 履歴に実体が残るためである。無料プランの非公開の上限は 100 GB
（`HfApi().whoami()` の `isPro` が `False`）。

```bash
# 履歴を潰す（取り消せないので、他に手が無いときだけ）
.venv/bin/python3 -c "
from huggingface_hub import HfApi
HfApi().super_squash_history(repo_id='<user>/<repo>')"
```

**潰すと「いつどの zip を上げたか」は残らない。** 先に、その情報が
[02_build_submission.md](02_build_submission.md) の「既知のハッシュ」の版の表と
`outputs/` の採点ログにあることを確かめる。script は `work/admin/cleanup_hf.py`
（`CONFIRM=yes` を付けないと何もせず、何をするかだけ表示する）。

**そもそも 9.4 GB を運び直さなくてよい場合がある**（候補どうしの違いが
`policy_server.py` の 72 KB だけのとき）。
[../knowledge/12_workbench_accidents.md](../knowledge/12_workbench_accidents.md) の
「9.4 GB のうち、変わるのは 72 KB だけだった」を先に読む。

## 6. 消してよいものと残すもの

**軸は「取り直せるか・作り直せるか」の 1 つだけである。**

| 対象 | 判定 |
|---|---|
| 本番スコアが返ってきた提出 zip | **消してよい。**控えは HF と Mac にある。古い候補を残す理由は無い |
| sha256 がよその重みと一致した重み | **消してよい**（2 節）。ハードリンクなら両方消す（3 節） |
| 上流から取り直せる素の重み（`lerobot/*`） | 消せるが、学習・評価のたびに要るので消す意味は薄い |
| `~/.cache/huggingface/lerobot`（学習データ） | **触らない。**取り直しが遅い |
| Xet のキャッシュ | **触らない。**消すと HF からの取り直しが全部やり直しになる |
| `training_state/`（optimizer の状態） | そのステップから学習を続ける道を捨てるかの判断。**ユーザーに聞く** |
| どこにも控えが無い重み | **ユーザーに聞く。**消すと作り直しに学習時間がかかる |

## 7. 消す作業は script にして流す

**`rm` を直接叩かず、消す一覧と根拠を script に書いてから流す。** 既定では何もせず一覧を出し、
`CONFIRM=yes` を付けたときだけ消す形にする（`work/admin/cleanup_*.sh` がこの形）。

```bash
bash work/admin/<script>.sh              # 何を消すか出すだけ
CONFIRM=yes bash work/admin/<script>.sh  # 実際に消す
```

**根拠は script のコメントに書く。**「sha256 がよそと一致した」「別の資料で廃止と決めた」
「変換したら要らない中間ファイル」のどれなのかを、パスごとに残す。

## 確認したい仮定

- `docker builder prune -f` が本当に空けるかは、押さえているイメージを先に消したかどうかで
  変わる、と 4 節で読んでいます。2026-08-18（1 バイトも空かなかった）と 2026-08-23
  （イメージを消したあとに `RECLAIMABLE` が 9.541 GB へ動いた）の 2 回の結果からの推測で、
  順番を入れ替えて切り分けた実験はしていません。外れていた場合、4 節の「順番」は根拠を
  失いますが、「両方消す」という結論は変わりません。
- 4 節の「申告 9.54 GB に対して実際に空いたのは約 1.1 GB」は 2026-08-23 の 1 回の測定です。
  そのあとビルドキャッシュ側も消えているため、同じ状態での再測定はできません。
