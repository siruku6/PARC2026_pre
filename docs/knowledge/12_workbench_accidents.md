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

## ③ 動画を書いたつもりで、8 バイトの空の TIFF が残っていた

**`imageio.mimwrite("....mp4", ...)` が MP4 ではなく TIFF を書こうとしていた。**
評価用イメージに **`imageio_ffmpeg` が入っていない**ため MP4 の実装が見つからず、
TIFF の実装に落ちる。TIFF は `fps` を知らないので
`TiffWriter.write() got an unexpected keyword argument 'fps'` で落ち、**8 バイトのファイルだけが残る。**

**イメージの中で動画に使えるものを実機で数えた**（`docker start parc2026_verify` で
既存のコンテナを起こし、`docker exec` で入った）。

| 実装 | ある／ない | 動画を書けるか |
|---|---|---|
| `imageio` 2.37.4 | ある | **MP4 は書けない**（`imageio_ffmpeg` が無い） |
| `imageio_ffmpeg` | **無い** | — |
| `cv2` 4.11.0（`opencv-python-headless`） | ある | **`mp4v` は書けた。**`avc1`（H.264）は encoder が無く `isOpened()` が False |
| `PIL` 12.3.0 | ある | GIF が書ける（外部のコーデックに依存しない） |

**黙って通った理由は、書いたあとに読み戻していなかったことである。**
`solution/tools/eval/kit_patches/rollout.py` の `_write_video` は、
**mp4 なら `cv2.VideoCapture` で 1 フレーム読み、GIF なら `n_frames` を見て、
確かめられたものだけを成功と見なす**（駄目なら GIF → PNG 連番へ落ちる）。
フレームは 128×256（front と wrist の並び）で、**素のままでは腕の行き先を目で追うには小さい**ので
`T3_VIDEO_SCALE`（既定 2）で拡大する。

## ④ 評価とビルドを同時に走らせない

この機械は **`nproc` が 2** である。9.4 GB の zip を圧縮しながら評価も回すと、
**load average が 5.5 まで上がり、どちらも遅くなり、ディスクを食い合う。**
①②はどちらもこの状態で起きた。

## 事故を防ぐ手順

```bash
df -h /home/ubuntu          # zip を組む前に必ず見る。1 本 9.4 GB
docker ps                   # 評価が生きているか（コンテナが消えていたら死んでいる）
tail -2 <出力先>/eval.log   # 行の途中で切れていないか
```

**空きが足りないときの空け方は [../howto/09_free_disk_space.md](../howto/09_free_disk_space.md)
にまとめた。** 消してよいものの判定（スコアが返ってきた zip は消してよい・sha256 がよその重みと
一致したものは消してよい）と、HuggingFace の非公開容量が画面で消しても空かないこともそちらにある。

**このときの本当の問題は、そもそも 9.4 GB を運ぼうとしていたことだった。**

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
