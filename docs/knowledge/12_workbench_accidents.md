# 12. 作業環境で起きた事故

**締切当日、1 時間のうちに 2 つ壊れた。**どちらも「気づきにくい壊れ方」だったので残す。

## ① ディスク満杯で、ソースが 0 バイトになった

**提出物の zip は 1 本 9.4 GB ある。**候補を 4 通り組んだ時点で 298 GB のディスクが
埋まり、そのときに走っていた書き換えでこうなった。

```python
path.write_text(text)   # ← 書く前にファイルを空にする。ENOSPC で中身が消える
```

**`solution/policy_server.py` が 1,315 行から 0 行になった。**
運営配布の `validate_submission.py` の検査を通した zip の中に同じファイルが入っていたので、そこから取り出して復旧できた。

**直し方**: 一時ファイルに書いてから差し替える。

```python
temporary = path.with_suffix(".py.new")
temporary.write_text(text)
shutil.move(str(temporary), str(path))
```

## ② 評価が、理由を出さずに死んだ

**同じディスク満杯で、走っていた 12 件の評価が 4 件目で止まった。**

| 見えたもの | |
|---|---|
| `docker ps` | **空**（方策サーバーが消えていた） |
| `eval.log` の末尾 | **行の途中で切れている**（`pipeline.ro`） |
| 直前の警告 | `VideoWriter が開けない` |
| GPU | 0% / 0 MiB |

**どこにも「disk full」とは出ない。**「途中で切れたログ」と「消えたコンテナ」から
推測するしかなかった。

## ③ 評価とビルドを同時に走らせない

この機械は **`nproc` が 2** である。9.4 GB の zip を圧縮しながら評価も回すと、
**load average が 5.5 まで上がり、どちらも遅くなり、ディスクを食い合う。**
①②はどちらもこの状態で起きた。

## 事故を防ぐ手順

```bash
df -h /home/ubuntu          # zip を組む前に必ず見る。1 本 9.4 GB
docker ps                   # 評価が生きているか（コンテナが消えていたら死んでいる）
tail -2 <出力先>/eval.log   # 行の途中で切れていないか
```

**スコアが返ってきた zip は、その場で消してよい。**
控えは HuggingFace と Mac にある。**古い候補を残す理由は無い。**

## HuggingFace の非公開容量は、画面で消しても空かない

**「容量上限に達しました」と出たので画面からファイルを削除したが、変わらなかった。**
**git 履歴に実体が残るため**である。解放するには履歴を潰すか、リポジトリごと消す。

```bash
# 履歴を潰す（取り消せないので、他に手が無いときだけ）
.venv/bin/python3 -c "
from huggingface_hub import HfApi
HfApi().super_squash_history(repo_id='<user>/<repo>')"
```

**ただし今回は、そもそも 9.4 GB を運ぶ必要が無かった。**

## 9.4 GB のうち、変わるのは 72 KB だけだった

**候補どうしの違いは `policy_server.py`（72 KB）だけ**で、重み 8.8 GB は同じである。
**Mac に既にある zip の中の 1 ファイルを差し替えれば、新しい版になる。**

```bash
cp submission_A.zip submission_B.zip
cp <repo>/solution/policy_server.py .
zip submission_B.zip policy_server.py      # 同名の既存エントリを置き換える

# 差し替わったことの確認（zip 全体のハッシュは一致しない。更新時刻が焼かれるため）
unzip -p submission_B.zip policy_server.py | shasum -a 256
unzip -l submission_B.zip | tail -1        # エントリ数が変わっていないこと
```

**結局この手は使わずに済んだ**（リポジトリを 1 つ消して容量が空いた）が、
**「大きいものを運び直さない」は次も使える。**

## Claude 側 / 人間側

**Claude 側**: **大きなファイルを作る前に、`df` で空き容量を見る。**
そして**追跡中のソースを、切り詰める書き方（`write_text`）で書き換えない。**
一時ファイル経由にする。**評価とビルドを同時に走らせない。**

**人間側**: **「評価が落ちている気がする」という勘は当たる。**
今回それで気づいた。**進捗が数分止まっていたら `docker ps` を見る** ——
生きていればコンテナが残っている。

**両方**: **`validate_submission.py` の検査を通した zip は、ソースの控えでもある。**
`policy_server.py` が消えたとき、git より zip の方が新しかった。
**「動くと確認した成果物」は、それ自体が復旧手段になる。**
