# 15. 自作デモの指令を配布データと同じ形にする

**この文書はコマンドだけを持つ。**
**なぜそうするのか・何が直るのかは
[reference/35](../reference/35_command_profile_and_ruckig.md) が持っている。**
先にそちらを読むこと。ここを読んでも「なぜ」は分からない。

## どの節から始めるか

| やること | 入り口 |
|---|---|
| **初めて触る**（`ruckig` がまだ入っていない） | **[1 節](#1-ruckig-を挿す初回だけ)** |
| 逆向きデモを作り直す（028 の生成器） | [2 節](#2-逆向きデモを作り直す) |
| **作ったデータが配布データと同じ形か測る** | **[3 節](#3-できたものを測る)** |
| 摘みを振ってみる（シムを回さない） | [4 節](#4-シムを回さずに摘みを絞る) |
| 共有モジュールを直したあとの確認 | [5 節](#5-試験) |
| **自分の生成器に組み込む**（026・(A)・新規） | **[6 節](#6-自分の生成器に組み込む)** |

---

## 1. `ruckig` を挿す（初回だけ）

**`parc2026-final` は 69 GB あるので、イメージには焼かない。**
ホストへ `--target` で入れたものを、読み取り専用で挿す。

```bash
cd ~/src/PARC2026_pre
pip install --target work/vendor_py310 --python-version 310 \
    --only-binary=:all: --platform manylinux_2_28_x86_64 ruckig
```

**コンテナの python は 3.10 である。**ホストの python（3.12）で普通に
`pip install ruckig` しても、コンテナからは読めない。`--python-version 310` が要る。

確かめる（**通信が無いことも一緒に見る**）:

```bash
docker run --rm --network none \
    -v "$PWD/solution/tools:/tools:ro" -v "$PWD/work/vendor_py310:/vendor:ro" \
    -e PYTHONPATH=/tools:/vendor parc2026-final \
    python -c "from motion.otg import HAS_RUCKIG, RUCKIG_VERSION; print(HAS_RUCKIG, RUCKIG_VERSION)"
```

`True 0.19.4` と出れば入っている。**入っていなくても落ちない**——
`otg.py` が自前の S 字プロファイルに落ちる（時間最適ではないぶん数コマ長い）。

---

## 2. 逆向きデモを作り直す

**`run_gripper_shift_sweep.sh` の環境変数がそのまま効く。**
違うのは `MOTION_*` の 2 つと `OUT` だけである。

```bash
cd ~/src/PARC2026_pre
env MODE=phased INIT_SOURCE=t3 GRASP_FRAME=clamped STANDOFF=0.06 T3_INIT=0 \
    LIFT=0.08 RELEASE=contact SETTLE=80 MAX_TAIL=-1 MAX_STEPS=300 \
    ROT_LIMIT=0.375 MAX_APPROACH=200 TASK=book EP_NPZ=ep_data.npz \
    OUT=$PWD/work/probe_ruckig WRITE_NPZ=npz_ruckig \
    MOTION_PLANNER=ruckig MOTION_EXIT_SPEED=0.25 \
  bash solution/tools/motion/run_ruckig_replay.sh "0" "4061,4062,4063"
```

**`$OUT` に順方向デモの npz（`EP_NPZ`）を先に置いておくこと。**
`work/probe_028/L/ep_data.npz` からコピーすればよい。

**`MOTION_PLANNER=minjerk` に替えると 028 と同じ通過点に戻る。**
**1 変数だけの比較はこれで取る**——入れ物も環境変数も同じまま、作り方だけが変わる。

課題ごとの残りの環境変数は
[reference/30 の 9 節](../reference/30_reverse_grasp_pose.md)の `COMMON` と同じ。

### 2.1 摘みの一覧

| 環境変数 | 既定 | 何が変わるか |
|---|---|---|
| `MOTION_PLANNER` | `ruckig` | `minjerk` で 028 と同じ通過点に戻る |
| `MOTION_EXIT_SPEED` | `0.25` | 区間の出口に残す速さ（上限に対する割合）。**0 にすると区間ごとに止まる** |
| `MOTION_ACCEL_POS` / `_ROT` | `0.25` / `0.05` | 指令が 1 コマで変わってよい量。**立ち上がりの速さを決める** |
| `MOTION_JERK_POS` / `_ROT` | `0.18` / `0.046` | その 1 階差分 |
| `MOTION_INITIAL_ACCEL_RATIO` | `1.0` | 区間の頭でいきなり出してよい加速度の割合。**0 にすると 5 次多項式と同じ遅い立ち上がりになる** |
| `APPROACH_*` | 028 と同じ | 上限と、指令 1 単位あたりの実移動量 |

---

## 3. できたものを測る

**028 の 4 条件だけでは足りない**（[35 の 1 節](../reference/35_command_profile_and_ruckig.md#1-なぜ物差しを足したか--4-条件を満たしたデータに残っていた-2-つ)）。
この道具は 4 条件と、停止・立ち上がりを同じ表に並べる。

```bash
cd ~/src/PARC2026_pre
python3 solution/tools/motion/report_smoothness.py --episodes 200 \
    --npz "028版=work/probe_028/L/npz/*.npz" \
    --npz "Ruckig=work/probe_ruckig/npz_ruckig/*.npz"
```

`--npz` は**何組でも並べられる**（`名前=glob` の形）。
`--dataset ""` を渡すと配布データを読まない（parquet が無い機械向け）。

**見るところは 3 つである。**

| 行 | 通す目安 |
|---|---|
| ①②③（飽和） | 028 の 6 節のまま。①は 0.00%、②は 10% 未満、③は 3% 未満 |
| **停止の塊** | **配布は 0.8 個/本。**5 個を超えていたら区間ごとに止まっている |
| **立ち上がり** | **配布は 1 コマ。**5 コマを超えていたら頭が空の列になっている |

---

## 4. シムを回さずに摘みを絞る

OSC を 1 次サーボで近似して指令の形を出す。**1 秒で終わるので、
シムを回す前にここで摘みを絞る。合否はシムで測ること。**

```bash
cd ~/src/PARC2026_pre
python3 solution/tools/motion/compare_waypoint_profiles.py --episodes 200 --exit-speed 0.25
```

ホストの python には `ruckig` が入っていないので、そのままだと自前の S 字で走る。
**Ruckig で見たいときは venv を作って挿す。**

```bash
python3 -m venv /tmp/rkvenv && /tmp/rkvenv/bin/pip install -q ruckig numpy pyarrow
PYTHONPATH=$(ls -d /tmp/rkvenv/lib/python3.*/site-packages) \
    python3 solution/tools/motion/compare_waypoint_profiles.py --episodes 200
```

---

## 5. 試験

**共有モジュールを直したら必ず打つ。**
`TestParityWith028` が本丸で、**切り出した 5 次多項式が
`replay_video_probe.py` と同じ通過点を返すこと**を見ている。

```bash
cd ~/src/PARC2026_pre
bash solution/tools/motion/run_motion_tests.sh          # コンテナ（robosuite と ruckig が居る）
```

**ホストで `pytest tests/test_motion_waypoints.py` を打つと、
robosuite が無いので一致の試験が 2 件 skip になる。**そこが本丸なので、
共有化に手を入れたときはコンテナで打つこと。

---

## 6. 自分の生成器に組み込む

**`replay_video_probe.py` 以外から使うときは、差し替えの層は要らない。**
`PoseTrajectoryPlanner` を直に呼ぶ。

```python
from motion.envelope import envelope_from_environment
from motion.otg import PoseTrajectoryPlanner

planner = PoseTrajectoryPlanner(envelope_from_environment())

for episode in episodes:
    planner.reset()                                  # エピソードの頭で持ち越しを捨てる
    for target_pose, next_pose, is_last in segments(episode):
        # 止まる必要のない継ぎ目は速さを残す。掴む・置くは 0 にする。
        # next_pose を渡すと、残った速度が次の行き先を向く（継ぎ目で折れない）
        for waypoint in planner.waypoints(current_pose, target_pose,
                                          0.0 if is_last else 0.25,
                                          next_target=next_pose):
            action = your_controller(observation, waypoint)
            observation = env.step(action)
            planner.observe(current_pose_of(observation))   # 実測を返す
```

**`observe()` を呼ぶこと。**呼ばないと、区間の切れ目で「計画上の速度」を
持ち越してしまう。実測を返せば、止まっている継ぎ目では 0 から、
流れている継ぎ目では流れたまま組み直す。

**コンテナに `/tools` と `/vendor` を挿すこと。**
`solution/tools/motion/run_ruckig_replay.sh` の `docker run` がそのまま雛形になる。

```
-v "$REPO/solution/tools:/tools:ro" \
-v "$REPO/work/vendor_py310:/vendor:ro" \
-e PYTHONPATH=/tools:/vendor:<これまでの PYTHONPATH>
```
