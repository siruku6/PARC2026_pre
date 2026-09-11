"""置いたのにゴールが立たない理由を、最後の位置で見る。"""
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
states = sd._init_states(root, plan.suite, TASK)
for index in (0, 1, 8):
    env.seed(0)
    rec = sd.run_episode(env, states[index], plan, 300)
    obs = env.env._get_observations()
    bowl = np.asarray(obs["akita_black_bowl_1_pos"])
    plate = np.asarray(obs["plate_1_pos"])
    d = bowl - plate
    print(f"ep{index}: goal={rec.goal_reached} 衝突={rec.collided} 歩={rec.steps}")
    print(f"   丼 {np.round(bowl,4)}  皿 {np.round(plate,4)}")
    print(f"   差 水平 {100*np.linalg.norm(d[:2]):.2f} cm  高さ {100*d[2]:+.2f} cm")
    print(f"   対象外の変位 {rec.object_displacements}")
env.close()
