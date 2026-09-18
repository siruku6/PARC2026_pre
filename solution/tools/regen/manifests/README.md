# データセットの組み上げ台帳

**ここに入っているのは、データセットの「どの本を外し、どの本で埋め戻したか」という判断の現物である。**
**組み上げスクリプト（`../*.py`・`../*.sh`）はこの台帳を入力に取る。**

## 項目一覧

| 節 | 一行要約 |
|---|---|
| [1. なぜ git に入れたか](#1-なぜ-git-に入れたか) | 台帳が無いと、混合データを消したら二度と作れない |
| [2. 何が入っているか](#2-何が入っているか) | ファイルごとの役割と、どのスクリプトが読むか |
| [3. まだ足りないもの](#3-まだ足りないもの) | `l90clean` の段だけ、スクリプトが残っていない |

---

## 1. なぜ git に入れたか

**2026-09-18 に移した。**それまで `work/regen/` にあり、`.gitignore` の `work/*` で
git の外にあった。**つまり 1 台のマシンにしか無かった。**

**データセットは 3 種類に分かれる。**

| 種類 | 例 | 失ったら |
|---|---|---|
| **配布元から落とせるもの** | LIBERO / LIBERO-plus の元データ | 落とし直せばよい |
| **作り直せるもの**（＝台帳 ＋ 部品 ＋ スクリプトから決まる） | 混ぜたデータセット | **台帳があれば作り直せる。無ければ作り直せない** |
| **撮り直した実演そのもの** | 丼 5 課題・初期姿勢を振った手本 | **二度と同じものにならない**（台本の方策を回し直す時間が要る） |

**台帳が git に無いと、2 番目が 3 番目と同じ扱いになる**——
つまり「混ぜたデータセットは部品から作り直せるから消してよい」という判断が成り立たない。
**台帳をここに置くことで、混ぜたものを消して部品だけ残す選択ができるようになる。**

## 2. 何が入っているか

| ファイル | 何を決めているか | 読むもの |
|---|---|---|
| `keep_replace_v2.json` | **`libero_plus_bowl5` の番号で、残す 12,553 本と外す 1,532 本** | `contact_clean/rebuild_dataset.py --keep-file` |
| `keep_without_bowl5.json` | 配布データから丼 5 課題を抜いた番号の並び。**配布の番号 → 土台の番号を引く鍵**（「この中の位置」で引ける） | `make_replace_keep.py` |
| `keep_batch1_without_bowl5.json` | 同じものの `lerobot_batch1` 版（**この系列は使わなくなった**） | — |
| `contact_v2_contact_scores.json` | **各デモが対象外の物体に接触しているかの判定**（9/14 に作り直した版） | `make_replace_keep.py` |
| `contact_v2_demo_map.json` | 配布 LeRobot データの本 ↔ 本家 hdf5 のデモの対応 | 同上 |
| `contact_pairs_probe.json` | 接触判定を作るときの探り（どの物体対を見るか） | `generate_perturbed.py` |
| `initstate_screen_v2.json` | **振った初期姿勢のうち、実際に合格した組み合わせ** | `generate_scripted.py` |
| `initstate_offsets.json`・`initstates_48.json`・`initstate_directions.json` | 初期姿勢のずらし方（量・48 通りの選び方・向き） | 同上 |
| `init_pose_coverage.json` | 振った初期姿勢の覆い具合の測定結果 | `measure_init_pose_coverage.py` |
| `robot_initstates_used.json` | 実際に使った初期姿勢 | `generate_scripted.py` |
| `quota_4000.json` | **40 課題へ何本ずつ配るかの割り当て** | `build_initpose_datasets.sh` |
| `index.json` | `libero_states`（下ごしらえ用の生 state/action）の索引 | `contact_clean/*` |
| `pairs_1.json`・`pairs_10.json`・`pick_demos.json`・`prealign_video_pairs.json`・`*_smoke.json` | 動作確認・見比べ用の小さな組（**組み上げには要らない**） | `render_*.py` |

**組み上げの手順そのものは [../../../../docs/resources/datasets.md](../../../../docs/resources/datasets.md) の 2.1・2.2 節にある。**

## 3. まだ足りないもの

**`libero_combined_bowl5_initpose_l90clean` を組んだ段のスクリプトが残っていない。**

`build_initpose_datasets.sh` は `STAGE=p`（`libero_plus_bowl5_initpose`）と
`STAGE=c`（`libero_combined_bowl5_initpose`）を持つが、
**そこから `libero_90` の汚れた 250 本を綺麗な手本に置き換える段は、
スクリプトにも台帳にもなっていない**（2026-09-16 に手で実施した）。

**だから、いまこのデータセットを消すと作り直せない。**
消す前に、次のどちらかが要る。

1. **置き換えた 250 本の番号と、埋め戻しに使った手本を台帳として書き出す**
   （`l90clean_replace.json` のような形で、ここに置く）
2. あるいは**このデータセットだけは混合のまま残す**

**このファイルを読んでいる時点でまだ 1 が無ければ、2 を選ぶこと。**
