# `motion/` — 指令のなめらかさを作る側の共有部品

**どこに何があるかだけを書いた索引である。**
**なぜこれが要るのか・実測は [docs/reference/35](../../../docs/reference/35_command_profile_and_ruckig.md)、
打つコマンドは [docs/howto/15](../../../docs/howto/15_smooth_command_profile.md) にある。**

| ファイル | 役割 |
|---|---|
| `envelope.py` | 配布データから測った**指令の封筒**（上限・1 階差分・2 階差分） |
| `rotation.py` | 姿勢の回転の約束（world 合成の軸角）。robosuite があればそれを使う |
| `waypoints.py` | 028 の 5 次多項式・記録の割り直しを**そのまま切り出したもの** |
| `otg.py` | **Ruckig による、区間をまたいで止まらない通過点**（無ければ自前の S 字） |
| `smoothness.py` | 配布データと形が揃っているかを測る**物差し** |
| `compare_waypoint_profiles.py` | シムを回さずに作り方を見比べる |
| `report_smoothness.py` | 作った npz と配布データを並べて表にする |
| `track3_ruckig_replay.py` | 028 の生成器を、**実行時に**差し替えて回す |
| `run_ruckig_replay.sh` | その入れ物（`ruckig` を挿したコンテナを起こす） |
| `run_motion_tests.sh` | `tests/test_motion_waypoints.py` をコンテナで打つ |

**`replay_video_probe.py`（028）と `contact_clean/`（026）には 1 行も書き込んでいない。**
差し替えは実行時に行うので、あちらは今までどおり動く。
