# 7. 遅さの原因を切り分ける（初回のみ・3 分）

> **ここだけは関数に畳んでいない。** 飛ばしてよい手順なので、必要になったときに
> 下のセルを貼る。**`ce.run()` の 1 本目の `[結果]` が 200 秒を超えたときだけ**
> ここへ来る（100〜130 秒なら想定どおり）。

1 エピソードの所要時間は **シミュレーション ＋ 推論**で決まる。

```
1 エピソード ≈ 600 × (env.step の秒数) + 12 × (重い /act の秒数)
```

| 構成 | 内訳 | 1 エピソード | 80 エピソード |
|---|---|---|---|
| **EGL ＋ GPU 推論**（狙い） | 600×0.04 + 12×0.85 | **34 秒** | **45 分** |
| osmesa ＋ GPU 推論 | 600×0.30 + 12×0.85 | 190 秒 | 4.2 時間 |
| EGL ＋ CPU 推論 | 600×0.04 + 12×15 | 204 秒 | 4.5 時間 |

**遅いときは、どちらの項が大きいかで原因が決まる。** 両方まとめて測る。

```python
# === 遅さの原因を切り分ける（描画 vs 推論）===
# 手順 6 を実行済みであること（eval_env / server_env / PORT を使う）。
import msgpack
import numpy as np

# --- 1. 描画: env.step の速度 ---
benchmark = '''
import contextlib, io, time
import numpy as np
from pipeline.config import EvalConfig
from pipeline.environment import EnvironmentManager

with contextlib.redirect_stdout(io.StringIO()):
    manager = EnvironmentManager(EvalConfig())
    tasks = manager.get_task_infos("libero_t1")
    env = manager.create_env(tasks[0])
    env.reset()
    env.set_init_state(tasks[0].init_states[0])
    for _ in range(10):
        env.step(np.zeros(7, dtype=np.float32))
    started = time.time()
    for _ in range(50):
        env.step(np.zeros(7, dtype=np.float32))
    elapsed = time.time() - started
print(f"env.step: {elapsed / 50 * 1000:.1f} ms/step")
'''
result = subprocess.run(
    [f"{KIT_DIR}/venv/bin/python", "-u", "-c", benchmark],
    cwd=str(KIT_DIR), env=eval_env, capture_output=True, text=True, timeout=900,
)
render_line = result.stdout.strip() or result.stderr.strip()[-800:]
print("[描画] " + render_line)

# --- 2. 推論: /act の速度 ---
# 配布キットと同じ形式で送る。obs は {key: {data, dtype, shape}} の msgpack、
# 応答は {"data": bytes} である（policy_server.py の serialize/deserialize に対応）。
def post(path: str, body: bytes, content_type: str) -> bytes:
    request = urllib.request.Request(
        f"http://localhost:{PORT}{path}", data=body,
        headers={"Content-Type": content_type},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()

def serialize_obs(observation: dict) -> bytes:
    return msgpack.packb({
        key: {"data": value.tobytes(), "dtype": str(value.dtype), "shape": list(value.shape)}
        for key, value in observation.items()
    }, use_bin_type=True)

log = open("/content/server_probe.log", "w")
server = subprocess.Popen([sys.executable, str(SERVER), "--port", str(PORT)],
                          cwd=str(SUBMISSION_DIR), env=server_env,
                          stdout=log, stderr=subprocess.STDOUT)
try:
    started = time.time()
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            break
        except Exception:
            if server.poll() is not None:
                raise RuntimeError(open("/content/server_probe.log").read()[-2000:])
            time.sleep(1)
    print(f"[推論] 起動 {time.time() - started:.0f} 秒")

    # /reset は JSON、/act は msgpack。配布キットの remote_policy.py と同じ。
    post("/reset", b'{"instruction": "Put the bowl on the stove"}', "application/json")

    observation = {
        "agentview_image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    payload = serialize_obs(observation)
    timings = []
    for index in range(3):
        begin = time.time()
        post("/act", payload, "application/x-msgpack")
        timings.append(time.time() - begin)
        print(f"[推論] act[{index}] {timings[-1]:6.3f} 秒")
finally:
    server.terminate()

# --- 3. 判定 ---
render_ms = float(render_line.split(":")[-1].replace("ms/step", "").strip()) if "ms/step" in render_line else None
if render_ms is not None:
    episode = 600 * render_ms / 1000 + 12 * timings[0]
    print()
    print(f"1 エピソードの見込み: {episode:.0f} 秒  → 80 エピソードで {episode * 80 / 3600:.1f} 時間")
    print(f"  うちシミュレーション {600 * render_ms / 1000:.0f} 秒 / 推論 {12 * timings[0]:.0f} 秒")
```

| 測定 | 正常 | 異常 | 異常時の意味 |
|---|---|---|---|
| **`env.step`** | 40〜60 ms | 250〜350 ms | **osmesa に落ちている。GPU 描画になっていない** |
| **`act[0]`** | 0.5〜1.5 秒 | 5 秒以上 | **CPU で推論している** |
| `act[1]` `act[2]` | ほぼ 0 秒 | 大きい | chunk のキャッシュが効いていない |

## `env.step` が遅いとき（400 ms 前後）

**Colab で最初に疑うのはここである。** EGL は「初期化に成功したが
ソフトウェア実装に落ちた」という壊れ方をする。**例外は出ない。**

原因は **NVIDIA の EGL ベンダ設定ファイルが無い**ことである。
`apt-get install libegl1` が入れるのは**ローダ**（libglvnd）だけで、
どのベンダ実装を使うかは `/usr/share/glvnd/egl_vendor.d/` の JSON が決める。
ここに NVIDIA の登録が無いと、ローダは Mesa のソフトウェア EGL を掴む。

**2026-08-09 の Colab で実際にこうなっていた。**

```
ベンダ設定: ['50_mesa.json']          ← NVIDIA の登録が無い
MUJOCO_GL = egl                        ← 環境変数は届いている
GLContext = robosuite.renderers.context.egl_context   ← robosuite は EGL を選んでいる
env.step: 431.8 ms/step                ← それでもソフトウェア描画
```

`10_nvidia.json` は `50_mesa.json` より優先される（数字が小さい方が先）。
**この 1 ファイルを置くだけで直った。**

```
env.step: 431.8 ms/step  →  38.6 ms/step   （11.2 倍）
1 エピソードのシミュレーション: 259 秒  →  23 秒
```

38.6 ms/step は、運営 Q&A No.3 で参加者が L4 上で計測した 39.1 ms/step と
ほぼ一致する。**本番と同じ描画性能が出ている。**

```python
# === EGL が NVIDIA の実装を掴んでいるか確認して、必要なら直す ===
import os, subprocess
from pathlib import Path

icd = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
print("ベンダ設定:", sorted(p.name for p in Path("/usr/share/glvnd/egl_vendor.d").glob("*.json"))
      if Path("/usr/share/glvnd/egl_vendor.d").is_dir() else "(ディレクトリが無い)")

if not icd.is_file():
    icd.parent.mkdir(parents=True, exist_ok=True)
    icd.write_text(
        '{\n'
        '    "file_format_version" : "1.0.0",\n'
        '    "ICD" : {\n'
        '        "library_path" : "libEGL_nvidia.so.0"\n'
        '    }\n'
        '}\n'
    )
    print(f"{icd} を作成した")
else:
    print(f"{icd} は既にある")

# robosuite がどの実装を選んだかを確かめる
probe = '''
import os
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
import robosuite.utils.binding_utils as binding
print("GLContext =", binding.GLContext.__module__)
'''
result = subprocess.run([f"{KIT_DIR}/venv/bin/python", "-c", probe],
                        cwd=str(KIT_DIR), env=eval_env, capture_output=True, text=True)
print(result.stdout or result.stderr[-1500:])
```

`GLContext = robosuite.renderers.context.egl_context` になっていれば、
robosuite としては EGL を選んでいる。**それでも遅いならベンダ設定の問題**である。
上のセルで JSON を作ったあと、`env.step` の計測をやり直す。

| 直したあとの `env.step` | 判断 |
|---|---|
| 40〜60 ms | **解決。** 80 エピソードが 45 分になる |
| 400 ms 前後のまま | ソフトウェア実装のまま。下の妥協案へ |

## どうしても EGL が効かないとき

`N_EPISODES` を下げて測る。**測れないよりはるかによい。**

| `N_EPISODES` | 総エピソード | 所要（431 ms/step のとき） | 成功率 10% の 95%CI |
|---|---|---|---|
| 20 | 80 | 5.8 時間 | [0.05, 0.19] |
| **5** | **20** | **1.4 時間** | [0.03, 0.30] |
| 3 | 12 | 52 分 | [0.02, 0.40] |

信頼区間は広がるが、**素の重みと追加学習後で大きな差があれば 20 エピソードでも見える**。
細かい比較（`n_action_steps` の 25 と 10 など）は EGL が直ってからにする。

## `act[0]` が遅いとき（CPU で推論している）

`/content/server_probe.log` の先頭に出る読み込みログと、`nvidia-smi` を見る。
`policy_server.py` は `torch.cuda.is_available()` で自動判定するため、
サーバーのプロセスから GPU が見えていない可能性が高い。

---

**次は [8. 評価を実行する](08_run.md) へ。**
