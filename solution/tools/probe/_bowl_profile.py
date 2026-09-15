"""丼の当たり判定を、高さごとの半径として出す。**縁をまたぐ高さと半径を決める値。**"""
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/repo/solution/tools/scripted_demo")
import numpy as np
import scripted_demo as sd
from libero.libero import get_libero_path

TASK = "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
root = Path(get_libero_path("bddl_files")).parent
plan = sd.TASK_PLANS[TASK]
env = sd._make_env(sd._bddl_path(root, plan.suite, TASK), False)
env.seed(0)
obs = sd.reset_to(env, sd._init_states(root, plan.suite, TASK)[0])
sim = env.env.sim
centre = np.asarray(obs["akita_black_bowl_1_pos"])
print(f"丼の原点 {np.round(centre,4)}")
rows = []
for i in range(sim.model.ngeom):
    name = sim.model.geom_id2name(i) or ""
    if not name.startswith("akita_black_bowl_1") or not sim.model.geom_contype[i]:
        continue
    p = sim.data.geom_xpos[i]; size = sim.model.geom_size[i]
    rows.append((float(p[2]-size[2]), float(p[2]+size[2]),
                 float(np.linalg.norm(p[:2]-centre[:2])),
                 float(max(size[0], size[1])), name))
print(f"{'z 下端':>8}{'z 上端':>8}{'中心からの距離':>14}{'geom の半径':>11}  name")
for low, high, dist, rad, name in sorted(rows):
    print(f"{low:>8.4f}{high:>8.4f}{100*dist:>13.2f}cm{100*rad:>10.2f}cm  {name}")
# 高さごとの外周
print("\n高さごとの外周（その高さに被る geom の 中心距離＋半径 の最大）")
for z in np.arange(0.888, 0.952, 0.006):
    hit = [100*(d+r) for low, high, d, r, _ in rows if low <= z <= high]
    inner = [100*(d-r) for low, high, d, r, _ in rows if low <= z <= high]
    if hit:
        print(f"  z {z:.3f}  外周 {max(hit):5.2f} cm  内壁 {min(inner):5.2f} cm")
env.close()
