# 16. 丼 5 課題を作り直して、学習と対照評価を回す

> **この文書は打つコマンドだけを持つ。** なぜそうするのかは
> [plan/24](../plan/24_contact_avoidance_redesign.md) にある。
>
> **所要**: 生成と変換 3 時間（無人）→ 組み立て 30 分 → 学習 9〜10 時間（無人・2 台で並行）→ 評価 2 時間 × 3。
> **ディスク**: 作った npz を段ごとに捨てるので 20 GB あれば足りる。
>
> **この手順は 1 変数である。**配布データに足すのは**今回作り直した丼 5 課題の手本だけ**で、
> 摂動下再生 907 本は足さない（[9 節](#9-第-2-段摂動下再生-907-本を足す)）。
> **2 つ同時に足すと、点が動いたときにどちらが効いたか読めない。**

| 節 | 中身 |
|---|---|
| [1 何をするのか](#1-何をするのか) | 1 枚の図 |
| [2 準備を確かめる](#2-準備を確かめる) | ここが揃っていないと 3 節で落ちる |
| [3 ディスクを空ける](#3-ディスクを空ける) | 20 GB 空けば足りる |
| [4 手本を作る](#4-手本を作る) | 無人で 1 時間 |
| [5 データセットを組む](#5-データセットを組む) | 入れ替え・混ぜる |
| [6 学習する](#6-学習する--2-本立てにする) | **2 本立て。**JAIST が action expert だけ・Blackwell が全パラメータ |
| [7 対照評価する](#7-対照評価する) | **3 つの重みを同じ条件で** |
| [8 どう読むか](#8-どう読むか) | 効いたと言ってよい条件 |
| [9 第 2 段](#9-第-2-段摂動下再生-907-本を足す) | **8 節が出てから**足す |

---

## 1. 何をするのか

```
配布データ 14,347 本
   │  丼 5 課題の 1,880 本を外す        ← make_replace_keep.py + rebuild_dataset.py
   ▼
12,467 本
   │  作り直した手本 約 1,800 本を足す   ← run_scripted_bowls.sh
   ▼
約 14,300 本  →  24,000 歩の学習  →  Track2 で対照評価

（摂動下再生 907 本はここでは足さない。→ 9 節）
```

**対照は「配布データそのままで学習した重み」**である。手元にある
`work/artifacts/pi05_full_local` がそれに当たるが、**学習の設定が揃っていないなら
対照も同じ設定で回し直すこと**（[8 節](#8-どう読むか)）。

**変える所を 1 つに絞ってある。**対照との差は**丼 5 課題の手本を入れ替えたことだけ**で、
本数も 14,347 対 約 14,300 とほぼ同じである。**摂動下再生 907 本（`work/regen/lerobot_batch1`）は
入れない**——あれは配布データの行動列をそのまま撃ち直したもので、
**今回の掴み方は入っていない**（丼 5 課題ぶんの 102 本は古い掴み方のままである）。
狙う穴も違う（手先のずれ 2.4〜6.5 cm の被覆・[plan/23 の 2 節](../plan/23_data_regeneration_design.md)）。
**同時に足すと、点が動いたときにどちらが効いたか読めない。**足すなら [9 節](#9-第-2-段摂動下再生-907-本を足す)。

---

## 2. 準備を確かめる

```bash
cd ~/src/PARC2026_pre
# 配布データ（15 GB・14,347 本）
python3 -c "import json;d=json.load(open('work/artifacts/libero_plus/meta/info.json'));print(d['total_episodes'],'本')"
# 人の手本の初期状態（di を使うのに要る）
ls work/artifacts/libero_states/index.json
# イメージ 2 つ
docker images | grep -E "parc2026-(final|train-pi05)"
```

**scripted policy が通ることを先に確かめる**（1 課題 3 分・接触の門も同じ規則）。

```bash
bash solution/tools/regen/run_in_container.sh \
    solution/tools/scripted_demo/scripted_demo.py verify \
    --task pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate \
    --max-steps 300
# 期待: 成功 47/50 前後
```

---

## 3. ディスクを空ける

**`work/regen/npz_batch1`（60 GB）は摂動下再生の中継物である。**
この手順では使わないので、**[9 節](#9-第-2-段摂動下再生-907-本を足す)をやらないと決めたなら消してよい。**

**9 節をやる気があるなら消さないこと。**`lerobot_batch1` は指示文が配布データと
違うので npz から作り直す必要があり、その材料がこれである（[9.1 節](#91-指示文を直して作り直す)）。

```bash
du -sh work/regen/npz_batch1
rm -rf work/regen/npz_batch1        # 9 節をやらないと決めたときだけ。60 GB 空く
df -h /home/ubuntu | tail -1        # 20 GB 以上あれば 4 節は回る
```

---

## 4. 手本を作る

**無人で回る。**段（摂動 × 初期状態の出どころ）ごとに作って変換し、
npz をその場で捨てる。

```bash
nohup bash solution/tools/regen/run_scripted_bowls.sh \
    > work/regen/logs/bowls_all.log 2>&1 &
tail -f work/regen/logs/bowls_all.log
```

| 段 | 中身 | 期待する本数 |
|---|---|---|
| `base:pi` `282:pi` `242:pi` `372:pi` | `.pruned_init` の 50 通り（採点器と同じ） | 各 235 前後 |
| `base:di` `282:di` `242:di` `372:di` | 人の手本の 50 通り（`on_the_ramekin` を除く 4 課題） | 各 190 前後 |

**合わせて約 1,700〜1,800 本。**配布データの 5 課題は 1,880 本なので同規模である。

**失敗した本は書き出されない**（`record` が `success`＝ゴール到達 ∧ 対象外 1 mm 未満
で弾く）。**落ちた割合はそのまま歩留まりなので、ログを読むこと。**

```bash
grep -c "成功 True" work/regen/logs/bowls_gen_*.log | head
for d in work/regen/lerobot_bowl/*/; do
  python3 -c "import json,sys;print(sys.argv[1], json.load(open(sys.argv[1]+'/meta/info.json'))['total_episodes'])" "$d"
done
```

---

## 5. データセットを組む

**1 本で通せる。**下の 5.1〜5.2 と、4 つの門と、HF へ上げるコマンドの印字までを
[`build_bowl5_dataset.sh`](../../solution/tools/regen/build_bowl5_dataset.sh) が持っている。
**何度打ってもよい**（出来上がっている段は数えて飛ばす）。

```bash
bash solution/tools/regen/build_bowl5_dataset.sh
```

| 門 | 見るもの |
|---|---|
| 1 | 本数・コマ数・課題数。**外した残りと作り直しの足し算にちょうど一致すること** |
| 2 | 指示文が評価でモデルが見る文と合っているか（[knowledge/23](../knowledge/23_self_made_demos_need_the_instruction_the_model_sees.md)） |
| 3 | 指令の形が配布データと揃っているか |
| 4 | 正規化統計の倍率が桁として動いていないか（[6.3 節](#63-正規化統計は据え置かない)） |

**下の 5.1・5.2 は、その中で何が起きているかである。**手で追うときに読む。

### 5.1 配布データから丼 5 課題を外す

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/regen/make_replace_keep.py \
    --root work/artifacts/libero_plus \
    --out work/artifacts/libero_states/keep_without_bowl5.json
# 期待: 全 14347 本 → 残す 12467 本・外す 1880 本

IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/contact_clean/rebuild_dataset.py \
    --root work/artifacts/libero_plus \
    --output work/artifacts/libero_plus_nobowl5 \
    --keep-file work/artifacts/libero_states/keep_without_bowl5.json
```

**動画は作り直さない**（ハードリンク）。**ディスクは増えない。**

### 5.2 作り直した段を 1 つにまとめ、配布データへ足す

**`merge` は再エンコードしない**（1 本 1 秒）。段を順に足していく。

```bash
BASE=work/artifacts/libero_plus_nobowl5
for d in work/regen/lerobot_bowl/*/ ; do
  NEXT="work/regen/mix_$(basename "$d")"
  IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
      solution/tools/scripted_demo/npz_to_lerobot.py merge \
      --base "$BASE" --add "$d" --output "$NEXT" \
      --output-repo-id local/libero_plus_bowl5_replaced
  BASE="$NEXT"
done
mv "$BASE" work/artifacts/libero_plus_bowl5
python3 -c "
import json;d=json.load(open('work/artifacts/libero_plus_bowl5/meta/info.json'))
print(d['total_episodes'],'本',d['total_frames'],'コマ')"   # 14,200〜14,300 本
```

**足し終えたら、指令の形が配布データと揃っているかを見る**（[plan/24 の 4.1.1](../plan/24_contact_avoidance_redesign.md)）。

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/motion/report_smoothness.py \
    --dataset work/artifacts/libero_plus --episodes 200 \
    --npz "作り直し=work/regen/npz_bowl/*/*.npz"
```

---

## 6. 学習する —— **2 本立てにする**

**動かす範囲を変えて 2 本回し、どちらがどれだけ効くかを見る。**

| | JAIST（`gpu-422591`） | Blackwell |
|---|---|---|
| 動かす範囲 | **action expert だけ**（693M / 4.14B） | **全部**（4.14B） |
| batch | 16 | 64 |
| 歩数 | **24,000**（対照と同じ） | **残り時間から決める**（下の 6.2） |
| 所要 | **9 時間 15 分**（1.39 秒/step 実測） | 8.5〜10.5 時間（6.85 秒/step 実測） |
| 制限 | 無し | **本選 1 で 60 時間・1 セッション 12 時間** |

**20 GB には全パラメータが載らない**ので、この割り方しかない
（[resources/infra_and_docker.md](../resources/infra_and_docker.md)）。
**会社の RTX6000（Turing）は退避先にならない**——bf16 のテンソルコアが無く、
実測 **428 秒/step** で 2 歩目に OOM した（2026-09-12）。

### 6.1 JAIST —— action expert だけ（**対照と 1 変数**）

**`jaist_launch.sh` の既定は逆向きデモ向け**（`rp3000` から 8,500 歩・5e-6・
カメラ拡張あり）なので、**5 つ渡して対照の条件へ戻す。**
これは `pi05_clean24k` の例そのもので、違うのはデータのパスだけである。

```bash
hf download siruku6/libero_plus_bowl5 --repo-type dataset \
    --local-dir ~/.cache/huggingface/parc_work/libero_plus_bowl5

RUN=pi05_bowl5 WEIGHTS=base \
  DATASET=$HOME/.cache/huggingface/parc_work/libero_plus_bowl5 \
  DATASET_REPO_ID=local/libero_plus_bowl5 \
  STEPS=24000 BATCH=16 LR=2.5e-5 SAVE_FREQ=8000 CAMERA_AUGMENT=0 FREEZE_CHECK=0 \
  nohup bash solution/train/jaist_launch.sh > ~/launch_pi05_bowl5.log 2>&1 &
```

| 渡す値 | なぜ | 対照 `pi05_full_local` |
|---|---|---|
| `WEIGHTS=base` | 素の `pi05_libero_base` から | 同じ（rev `a217bfd3`） |
| `STEPS=24000` | 既定 8,500 を戻す | 24,000 |
| `LR=2.5e-5` | 既定 5e-6 を戻す | 2.5e-5 → 2.5e-6 |
| `CAMERA_AUGMENT=0` | 既定 あり を戻す | なし |
| `FREEZE_CHECK=0` | **据え置いていない**ので入口の検査を外す（[6.3 節](#63-正規化統計は据え置かない)） | — |
| `SAVE_FREQ=8000` | 学習に影響しない。既定 2,000 だと 12 個 108 GB になる | — |

**空きは 47 GB 要る**（チェックポイント 3 個 ＋ データ）。

**20 歩で 3 つ確かめる**（`~/launch_pi05_bowl5.log` に出る）。

| 見るもの | 期待 |
|---|---|
| `num_learnable_params` | **693,422,112** |
| `dataset.num_episodes` | データの本数と一致 |
| `updt_s` / `mem_gb` | **1.39 秒 / 16.27 GB**（過去 4 本と同じ。違えば設定が効いていない） |

### 6.2 Blackwell —— 全パラメータ（**歩数は打つ前に決めない**）

```bash
. ~/env_pi05.sh
hf download siruku6/pi05_code --local-dir ~/pi05_code
bash ~/pi05_code/solution/train/blackwell_bowl5.sh
```

**1 本で準備から離席まで行く。**降ろす 2 つ（データ 17 GB ＋ 出発点 8.8 GB）を
並列に流し、本数・コマ数・課題数を `bowl5_expected.json` と突き合わせ、
残り時間から歩数を決めて起こす。**30 分で離席できることを狙っている。**

**`blackwell_setup.sh` は打たない。**あちらは Blackwell の上でデータセットを
組むので 2 時間かかる。**この走りのデータは手元で組んで HF に置いてあるので、
降ろすだけで済む。**

**歩数は起動時刻からの残り時間で決まる**（後片付けに 60 分を残す）。

| 準備に要した時間 | 打つ歩数 | サンプル数（JAIST 比） |
|---|---|---|
| **40 秒**（2026-09-12 の実測） | **5,500** | 352,000（92%） |
| 30〜60 分 | 5,000 | 320,000（83%） |
| 90〜120 分 | 4,500 | 288,000（75%） |

**実測は 40 秒だった**（2026-09-12。データ 17 GB ＋ 出発点 8.8 GB を並列に降ろして）。
**`blackwell_setup.sh` を打っていた頃は 1〜2 時間かかっていた**——あちらは Blackwell の
上でデータセットを組む。**手元で組んで HF に置き、降ろすだけにしたのが効いた。**

**決めた歩数は `--policy.scheduler_decay_steps` にも入る。**LeRobot の既定は
**30,000 歩固定で `--steps` に連動しない**ので、入れないと 5,000 歩では
学習率がほぼ 2.5e-5 のまま終わる（`--steps 20` で回して確かめてある）。
**だから「時間が余ったら伸ばす」ができない。**打つ前に確定させて完走させる。

**20 歩で `num_learnable_params=4143404816` を確かめてから離席する**
（693,422,112 なら action expert だけになっている）。

### 6.3 正規化統計は据え置かない

**据え置くのは「統計を持っている重みから続ける」ときだけである**
（[reference/29 の 6.3](../reference/29_training_data_requirements.md)）。
ここは統計を持たない `pi05_libero_base` から通すので、最初の学習が尺度を決める。

**揃えるべきは数値ではなく手順である。**対照は「そのデータ自身の統計」で
学習している。据え置くと**こちらだけが「自分のデータに合わない統計を使う」**
ことになり、差が 1 つ増える。**出るずれはデータを入れ替えたことの帰結であって、
別の変数ではない。**

**それでも桁は見る**（[5 節](#5-データセットを組む)の門 4）。逆向きを 2 割混ぜた回は
配布側の回転チャンネルが 0.43 倍に縮み、Track2 が 0.9167 から 0.5833 へ落ちている。
**倍率が 0.5 を割るか 2 を超えたら、混ぜ方そのものを疑うこと。**

### 6.4 終わったら重みを取り込む

**Blackwell の出力は `/opt/dlami/nvme` にあり、セッション終了で消える。**
上げるループが `_all_uploaded` を置くまで止めないこと（[17](17_checkpoint_upload.md)）。
**JAIST の `work/train` は消えない**ので、終わってから手で上げれば足りる。

```bash
python3 solution/tools/weights/install_pi05_weights.py \
    work/train/pi05_bowl5/checkpoints/024000/pretrained_model --name pi05_bowl5
```

**チェックポイントの間引きを動かすこと**（1 個 11 GB。
[howto/07](07_train_pi05_local.md) の「チェックポイントを自動で間引く」）。

---

## 7. 対照評価する

**3 つの重みを、同じ条件で、同じ本数だけ回す。**

| 重み | 何を答えるか |
|---|---|
| `pi05_full_local`（既にある） | 対照。配布データそのまま・action expert だけ |
| `pi05_bowl5`（JAIST） | **入れ替えたデータが効いたか。**対照と 1 変数 |
| `pi05_bowl5_full`（Blackwell） | **全パラメータにするとどれだけ変わるか。**ただし全パラメータ × 元データの対照が無いので、**「全パラが効いた」と「データが効いた」は分けられない。**分けたければ Blackwell をもう 1 セッション使う |

```bash
# 作り直した側
TRACK_EPISODES=5 LABEL=bowl5 WEIGHTS="$PWD/work/artifacts/pi05_bowl5" \
    POLICY_IMAGE=parc2026-train-pi05 MUJOCO_GL=egl \
    bash solution/tools/eval/run_final_eval.sh track2
# 対照（配布データそのまま）
TRACK_EPISODES=5 LABEL=base WEIGHTS="$PWD/work/artifacts/pi05_full_local" \
    POLICY_IMAGE=parc2026-train-pi05 MUJOCO_GL=egl \
    bash solution/tools/eval/run_final_eval.sh track2
```

**`TRACK_EPISODES` を 5 未満にしないこと。**3 本では読めない
（[plan/23 の 7 節](../plan/23_data_regeneration_design.md)）。

読み比べる。

```bash
python3 solution/tools/eval/compare_object_disp.py \
    work/final_eval/<日時>_track2_base work/final_eval/<日時>_track2_bowl5
cat work/final_eval/<日時>_track2_bowl5/score_composite.md
```

---

## 8. どう読むか

**見る数は 3 つある。並べて見ること。**

| 見るもの | どこで見るか | 期待 |
|---|---|---|
| **丼 5 課題の衝突率** | `compare_object_disp.py` の課題別 | **60.6% から下がる**。ここが動かなければ手本は転移していない |
| 丼 5 課題の `done` | 同上 | **92.8% を保つ**。下がったら作り直した手本が別の失敗を入れている |
| 全体の合計点 | `score_composite.md` | 上がる |

**「効いた」と言ってよい条件。**
**丼 5 課題の衝突率が下がり、かつ `done` が落ちていないこと。**
合計点だけを見ると、他の 35 課題の揺らぎに埋もれる——
5 課題は全体の 20% しかない。

**効かなかったときに次を決めるための切り分け。**

| 観察 | 読み | 次の手 |
|---|---|---|
| 衝突率が下がらない | 手本が転移していない | **本数か学習の刻みの問題。**0.17 エポックしか回っていない（[plan/23 の 2.3](../plan/23_data_regeneration_design.md)）ので、5 課題を重み付けして厚くする |
| 衝突率は下がるが `done` が落ちた | 作り直した手本が別の失敗を教えた | **指令の形を疑う**（停止コマ 14.3%）。**ただし Ruckig は効かない**——測って外してある（[reference/35 の 3.4](../reference/35_command_profile_and_ruckig.md#34-scripted_demo丼-5-課題では効かなかった--直す欠点が無かった)）。停止はすべて `hold`（掴む・落ち着かせる・離す・整定）で、**疑うならそちらの作り** |
| どちらも良いが合計点が動かない | 5 課題では全体が動かない | **[plan/24 の 8 節](../plan/24_contact_avoidance_redesign.md#8-広げる順序転移が確認できてから)の第 1 段・第 2 段へ広げる** |

---

## 9. 第 2 段——摂動下再生 907 本を足す

> **8 節の結果が出てから読むこと。**ここを 1 段目と同時にやると 2 変数になる。

**これは別の穴を埋める手である。**配布データは手先のずれ 1.0 cm までしか
覆っていないのに、Track2 の採点では 2.4〜6.5 cm が出て、その帯で
**成功率 0.833 → 0.535・衝突率 0.139 → 0.319** に崩れる
（[plan/23 の 2 節](../plan/23_data_regeneration_design.md)）。
`work/regen/lerobot_batch1` の 1,009 本は、**配布データの人の手本の行動列を
そのまま撃ち直し**（初期姿勢だけ振る）、採点と同じ規則を通ったものである。

**だから指令の形は配布データそのものである**（実測・2026-09-12）。

| 見るもの | 配布 | 摂動下再生 1,009 | 作り直した手本 |
|---|---|---|---|
| 並進 ±0.9375 以上 [%] | 7.79 | 6.64 | 0.00 |
| 停止コマ [%] | 0.96 | 1.35 | 14.40 |
| 停止の塊 [個/本] | 0.8 | 1.0 | 3.4 |

**Ruckig はどちらにも入っていない。**作り直した手本のほうは 2026-09-12 に
通して測り、**効かないので外した**（[reference/35 の 3.4](../reference/35_command_profile_and_ruckig.md#34-scripted_demo丼-5-課題では効かなかった--直す欠点が無かった)。
停止コマ 14.33% → 12.74% とほぼ動かず、符号反転が 0.46% → 2.37% と悪化した。
**停止は継ぎ目ではなく `hold` から出ている**ので、通過点の作り方では届かない）。
**1,009 本に無いのは Ruckig ではなく今回の掴み方である**——
丼 5 課題ぶんの 102 本は古い掴み方のままなので、[9.2 節](#92-丼-5-課題を外す)で外す。

### 9.1 指示文を直して作り直す

**`work/regen/lerobot_batch1`（1,009 本）は、62%（626 本）が
評価で一度も渡らない指示文の下に入っている。**`record` が残すのは bddl の
書き方（`pick the akita black bowl next to the cookies box …`）で、配布データは
`pick up the black bowl next to the cookie box …` である。
**`policy_server.py` は採点で来た文を学習データの 40 種へ寄せてから**モデルへ渡すので、
**寄せた先の書き方で入っていない手本は誰も引かない。**

**npz から作り直す**（`work/regen/npz_batch1` がまだあるなら。2〜3 時間・無人）。

```bash
IMAGE=parc2026-train-pi05 nohup bash solution/tools/regen/run_in_container.sh \
    solution/tools/scripted_demo/npz_to_lerobot.py build \
    --npz-dir work/regen/npz_batch1 \
    --output work/regen/lerobot_batch1_fixed \
    --repo-id local/regen_robotinit_b1 --normalize-instructions \
    > work/regen/logs/rebuild_batch1.log 2>&1 &
# 寄せ方を目で確かめる（38 種すべてが lexical か exact であること）
sed -n "/指示文を学習データ/,/^Map/p" work/regen/logs/rebuild_batch1.log | head -80
```

**`none` が出た文があれば止めること。**寄せられていない＝評価で引かれない。

**なぜこうなるかは [knowledge/23](../knowledge/23_self_made_demos_need_the_instruction_the_model_sees.md) にある。**
**組み終えたデータセットは検査を通すこと**（ずれがあれば終了コード 1）。

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/scripted_demo/check_dataset_instructions.py \
    work/artifacts/libero_plus_bowl5 --grader-texts work/regen/grader_texts.txt
```

**これが終わってから `work/regen/npz_batch1` を消す**（[3 節](#3-ディスクを空ける)）。

### 9.2 丼 5 課題を外す

**1,009 本のうち丼 5 課題ぶんは 102 本である。**作り直した手本と同じ課題を
**元の掴み方のまま**足すことになるので外す。

```bash
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/regen/make_replace_keep.py \
    --root work/regen/lerobot_batch1_fixed \
    --out work/regen/keep_batch1_without_bowl5.json
IMAGE=parc2026-train-pi05 bash solution/tools/regen/run_in_container.sh \
    solution/tools/contact_clean/rebuild_dataset.py \
    --root work/regen/lerobot_batch1_fixed \
    --output work/regen/lerobot_batch1_nobowl5 \
    --keep-file work/regen/keep_batch1_without_bowl5.json
# 期待: 1009 本 → 907 本
```

### 9.3 混ぜて学習し直す

[5.2 節](#52-作り直した段を-1-つにまとめ配布データへ足す)の `for` に
`work/regen/lerobot_batch1_nobowl5` を足して回し直し、6〜7 節を
`--run-name pi05_bowl5_cov` で繰り返す。
**対照は 8 節で使った `pi05_bowl5` のほうである**（配布そのままではない）——
ここで見たいのは「被覆を足すと上乗せがあるか」だけだからである。

---

## 10. 第 3 段——初期姿勢を振ったデータで全パラメータ学習する（Blackwell・9/14）

**`pi05_bowl5_full`（6.2 節）と同じスクリプト・同じ設定で、データだけを
`libero_plus_bowl5_initpose` に替えて 1 本回す。**違いはデータ 1 つなので、
評価の差はそのままデータの効き目として読める。

| | `pi05_bowl5_full`（対照） | **`pi05_initpose_full`（これ）** |
|---|---|---|
| データ | `siruku6/libero_plus_bowl5`（14,085 本） | **`siruku6/libero_plus_bowl5_initpose`**（公開・16,820 本・2,712,308 コマ・40 課題・22.8 GB） |
| 出発点 | `lerobot/pi05_libero_base`（版は `train_pi05.py` の `_BASE_MODEL_REVISION`） | 同じ |
| 動かす範囲 | 全部（4,143,404,816） | 同じ |
| 学習率・warmup・減衰 | 2.5e-5・500 歩・5,500 歩で減衰 | 同じ |
| batch・種・歩数 | 64・42・5,500 | 同じ |
| チェックポイントの間隔 | 2,000（HF には `004000`・`005500`） | **5,500（最後の 1 個だけ）。学習には影響しない**（10.1 節） |

データの中身は [resources/datasets.md の 2.1 節](../resources/datasets.md)。
**RunPod で `pi05_bowl5_full` から続ける走り（`runpod_initpose_ft.sh`）は、
HF の空きが無いので一旦見送った**（ユーザ 9/14 夜）。

### 10.1 打つ前に —— HF の空き（**ここで詰まると重みが消える**）

**チェックポイントの置き場は HF だけである。**`/opt/dlami/nvme` はセッション終了で
消えるので、**HF に上がらなければ 10 時間の学習が丸ごと失われる。**

| 数字 | 値 |
|---|---|
| チェックポイント 1 個（HF 上） | **約 9.4 GB**（`training_state` は上げずに消す） |
| `SAVE_FREQ=5500` のときに要る空き | **約 9.4 GB**（1 個） |
| `SAVE_FREQ=2750` のときに要る空き | 約 18.8 GB（2 個） |
| 非公開リポジトリの使用量（**9/14 21:50 にこのマシンから数えた**） | **65.8 GB**（うち `libero_combined_bowl5_initpose` 28.1・`pi05_bowl5_full` 18.7・`libero_plus_bowl5` 17.6） |

**このマシンから数えると非公開は 65.8 GB で、枠が 100 GB なら約 34 GB 空いている計算になる。**
**「空きが無い」と見えているなら数え方が違うので、HF の設定画面（Settings → Storage）の数字を正にする。**

**上げるのは学習が終わってから（打ってから約 10.5 時間後）である。**
**それまでに 10 GB 空けば間に合う。**間に合わなくても、上げるループは失敗したら
**60 秒ごとにやり直す**ので、**T0 ＋ 12 時間までに空ければ自動で上がる**（10.5 節）。

**上げ先は先に非公開で作る。**作らずに打つと、上げるループの `hf upload` が
**公開リポジトリとして作る**（`pi05_bowl5_full` は非公開である）。

### 10.2 打つコマンド（**T0 から 60 分以内に打ち終える**）

**`T0`（Notebook の画面が出た時刻）を控える。**5,500 歩は 6.85 秒/歩で約 10.5 時間かかり、
12 時間の枠にぎりぎり収まる。**`STEPS` を渡すとスクリプトは残り時間を見ない**ので、
遅れて打つと最後の上げが枠に入らない。

```bash
# 1. 環境（毎セッション。2 回目以降は 1 分）
export HF_TOKEN="$(cat ~/.hf_token)"
. ~/env_pi05.sh
hf download siruku6/pi05_code --local-dir ~/pi05_code
bash ~/pi05_code/solution/train/omnicampus_setup.sh 2>&1 | tee ~/recon_setup.txt
. ~/env_pi05.sh

# 2. 上げ先を非公開で作る（作らないと公開で作られる）
python3 -c "from huggingface_hub import create_repo; print(create_repo('siruku6/pi05_initpose_full', private=True, exist_ok=True))"

# 3. 降ろす → 数える → 学習と上げるループを起こす → 20 歩で検算（1 本で離席まで行く）
RUN=pi05_initpose_full STEPS=5500 SAVE_FREQ=5500 \
  DATASET_REPO=siruku6/libero_plus_bowl5_initpose \
  DATASET_ID=local/libero_plus_bowl5_initpose \
  DS=/opt/dlami/nvme/data/libero_plus_bowl5_initpose \
  bash ~/pi05_code/solution/train/blackwell_bowl5.sh
```

**HF の空きが 19 GB 以上あると確かめられたら、`SAVE_FREQ=2750` にして途中の 1 個も残してよい**
（学習の中身は変わらない）。

**`blackwell_initpose.sh` は使わない。**あちらは 9/13 の T1′（`lerobot_init_pose_all` 1,250 本を
Blackwell の上で足す）用で、データが違う。

### 10.3 離席の前に、スクリプトの出力で 5 つ確かめる

| 段 | 見るもの | 期待 |
|---|---|---|
| 降りたものを数える | 本数・コマ数・課題数 | **16820・2712308・40**（`bowl5_expected.json` と一致しなければ止まる） |
| 歩数を決める | `STEPS=5500 を渡された` と `チェックポイント 1 個` | その通り |
| 動き出し | `num_learnable_params` | **4143404816**（693422112 なら action expert だけ。止まる） |
| 動き出し | `dataset.num_episodes` | **16820** |
| 最後 | `=== 走っている。離席してよい ===` と `終わる目安` | **10 時間 30 分前後**（1 歩 6.85 秒前後。目立って遅ければ枠に入らない） |

### 10.4 監視セルを貼って離席する

[howto/14 の B6](14_blackwell_session.md#b6-監視セルを実行状態のまま残して離席する5-分) のセルを
Notebook に貼り、**先頭の 2 行だけ**次に替えて実行する。**貼らないと 1 時間の待機で学習ごと止まる。**

```python
RUN  = "pi05_initpose_full"
ULOG = os.path.expanduser(f"~/upload_ckpt_{RUN}.log")   # blackwell_bowl5.sh はこの名前で書く
```

`LOG` と `DONE` は `RUN` から決まるので替えなくてよい。

### 10.5 戻ったら

| セルの最後の行 | 意味 | やること |
|---|---|---|
| **`全部上がった。UI から Stop Server を押してよい`** | `005500` とログが HF に上がり、確かめられた | HF の `siruku6/pi05_initpose_full` に `005500/` と `logs/` があるのを見てから **File > Hub Control Panel > Stop Server** |
| `上げ待ち: … hf upload が失敗` が続く | **HF の空きが足りない**（ほかの理由もありうるので `~/upload_ckpt_pi05_initpose_full.log` を読む） | **HF で 10 GB 空ける。**ループが 60 秒ごとにやり直すので、空けば自動で上がる。**T0 ＋ 12 時間が締切** |
| `人を待っている` | 上げるループが居ないのに印が無い | [howto/17](17_checkpoint_upload.md) の手順で手で上げる |

**評価の組み方はまだ決めていない。**取り込むときは 6.4 節と同じく
`install_pi05_weights.py` に `005500/pretrained_model` を渡し、`--name pi05_initpose_full` とする。
