# 16. 丼 5 課題を作り直して、学習と対照評価を回す

> **この文書は打つコマンドだけを持つ。** なぜそうするのかは
> [plan/24](../plan/24_contact_avoidance_redesign.md) にある。
>
> **所要**: 生成と変換 1〜2 時間（無人）→ 学習 10 時間（無人）→ 評価 2 時間 × 2。
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
| [6 学習する](#6-学習する) | 24,000 歩・10 時間 |
| [7 対照評価する](#7-対照評価する) | **2 つの重みを同じ条件で** |
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

## 6. 学習する

**24,000 歩・batch 16 で約 10 時間**（[howto/07](07_train_pi05_local.md)）。

```bash
docker run -d --gpus all --rm -it --user "$(id -u):$(id -g)" \
    -v "$PWD/work/train:/out" \
    -v "$PWD/work/artifacts:/artifacts" \
    -v "$HOME/.cache/huggingface:/hf_home" \
    parc2026-train-pi05 \
    python /opt/pi05/train_pi05.py \
        --run-name pi05_bowl5 \
        --dataset-root /artifacts/libero_plus_bowl5 \
        --steps 24000 --batch-size 16
tail -f work/train/pi05_bowl5.log
```

**チェックポイントの間引きを動かすこと**（1 個 11 GB）。cron 登録済みなら
出力先を足す（[howto/07](07_train_pi05_local.md) の「チェックポイントを自動で間引く」）。

学習が終わったら重みを取り込む。

```bash
python3 solution/tools/weights/install_pi05_weights.py \
    work/train/pi05_bowl5/checkpoints/024000/pretrained_model --name pi05_bowl5
```

---

## 7. 対照評価する

**2 つの重みを、同じ条件で、同じ本数だけ回す。**

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
| 衝突率は下がるが `done` が落ちた | 作り直した手本が別の失敗を教えた | **指令の形を疑う**（停止コマ 14.4%）。[reference/35](../reference/35_command_profile_and_ruckig.md) の Ruckig を入れて作り直す |
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

**Ruckig はどちらにも入っていない**（[reference/35](../reference/35_command_profile_and_ruckig.md) は
まだ実装していない）。**1,009 本に無いのは Ruckig ではなく今回の掴み方である**——
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
