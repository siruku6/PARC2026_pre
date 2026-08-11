# 6. 評価の設定（毎回ここから）

**以降のセルはすべてここで定義した変数を使う。** 診断も本番も、まずこれを流す。

```python
# === 評価の設定 ===
# ここだけを書き換えて、下のセルは触らずに設定を変えて比べる。
# **1 タスクあたり**の本数である。総数は「タスク数 × この値」。
# **1 のまま触らない。** 手順 5-1 で検証スイートへ差し替えるとタスクが
# 4 → 84 に増えるので、**20 にすると 1,680 エピソード（50 時間）になって
# 終わらない。** 例題 4 タスクだけを測るとき以外は 1 である。
N_EPISODES = 1
# **本番の打ち切りは 300 歩である。** 採点ログの /reset と /act の並びから
# 1 本ずつのステップ数を数えたところ、8 本中 4〜5 本がちょうど 300 で
# 止まっていた。配布既定の 600 で測ると**倍の時間を与えてしまい、
# 成功率を実際より高く見積もる。**
MAX_STEPS = 300
PORT = 8000

# **疎通確認のときだけ 5 を入れる。** 84 件を 2.6 時間かけて回す前に、
# 5 件だけ流して最後まで行くか見る。**None なら全部回す。**
# **手順 8 のセルを書き換えないこと** —— 消し忘れると 84 件のはずが 5 件のまま
# 走り、前の回と比べられなくなる（51 分は戻らない）。
MAX_TASKS = None            # 疎通確認は 5、本番は None

# **84 件を 3 つに割って回す。** 1 本 2.6 時間は Colab のセッション切断に耐えない
# （8/11 に 56 件目・1 時間 43 分で落ちた）。**1 つ 28 件・51 分**なら、
# 落ちてもその 51 分だけで済む。
#   手順 5-2 で /content/chunks/VAL_1.csv 〜 VAL_3.csv を作ってから、
#   CHUNK を 1 → 2 → 3 と変えて 手順 6 → 8 → 9 を 3 回まわす。
#   **None なら 84 件を一度に回す**（GPU マシンなど、切れない環境向け）。
CHUNK = 1                   # 1〜3。None なら 84 件を一度に

# 推論時のつまみ。**空にするのが正しい既定である。**
# policy_server.py の既定は **n_action_steps=5 / ema_alpha=0.3** で、
# **これが 3 回目の提出（0.1409）と 4 回目、および 8/11 の 84 件の評価の構成**である。
# 書かなければその値になる。**重みを比べるときは空にすること** ——
# ここに値を書くと「重み」と「つまみ」を同時に変えることになり、
# 1 件ずつ対にした比較が成り立たない。
POLICY_ENV = {
    # "PARC_N_ACTION_STEPS": "2",    # **既定 5。** 1 回の推論で実行するステップ数
    # "PARC_EMA_ALPHA": "0.15",      # **既定 0.3。** 直前の行動を混ぜる割合
    # "PARC_BLEND_STEPS": "3",       # 既定 0（無効）。継ぎ目を均すステップ数
    # "PARC_SAMPLES": "3",           # 既定 1（無効）。chunk を引く本数
}

# ロールアウトを mp4 に撮るか。**None なら撮らない。**
# 撮るのは採点ハーネス側（キットの pipeline/rollout.py）なので、
# POLICY_ENV ではなく下の eval_env に入る。**提出物とは無関係である。**
#
# **既定では失敗した回だけ残る。** 84 件回しても成功した回は書き出さない。
# 上乗せは 1 歩 0.33 ms（env.step 39 ms の 0.8%）で、90 分の測定なら +3 分。
RECORD_DIR = "/content/videos"      # 撮らないなら None
RECORD_WHICH = "failures"           # "all" にすると成功も撮る

import os, subprocess, sys, time, urllib.request, json
from pathlib import Path

# --- 方策サーバー側の環境 ---
server_env = dict(os.environ)
server_env.update({k: v for k, v in POLICY_ENV.items() if v})
# policy_server.py は setdefault で offline を立てる。Colab は通信できるので
# 明示的に解除する。**採点環境は遮断されるため、オフラインの確認は Docker 側で行う。**
server_env["HF_HUB_OFFLINE"] = "0"
server_env["TRANSFORMERS_OFFLINE"] = "0"
server_env["CUDA_VISIBLE_DEVICES"] = "0"
# 別プロセスなので、ノートブック側の sys.path 操作は効かない。
server_env["PYTHONPATH"] = str(LEROBOT_SRC)

# --- 評価パイプライン側の環境 ---
eval_env = dict(os.environ)
eval_env.update({
    "MUJOCO_GL": "egl",          # GPU 描画。手順 1 のベンダ設定と対で効く
    "MPLBACKEND": "Agg",         # Colab の既定値だと libero の import が落ちる
    "PYTHONUNBUFFERED": "1",     # 無いと出力がバッファされ、止まったように見える
    "PYTHONPATH": f"{KIT_DIR}/LIBERO-plus:{KIT_DIR}:{KIT_DIR}/compe",
    "LIBERO_ROOT": f"{KIT_DIR}/LIBERO-plus",
})

# --- 録画（評価パイプライン側に渡す）---
if RECORD_DIR:
    eval_env["PARC_RECORD_DIR"] = RECORD_DIR
    eval_env["PARC_RECORD_WHICH"] = RECORD_WHICH
    # 上限。84 件で失敗が多いと本数が伸びるので、既定の 40 より広く取る。
    eval_env["PARC_RECORD_MAX"] = "60"

# --- 小分けの切り替え（CSV とキットと出力先を対で入れ替える）---
# **3 つが揃っていないと、別の 28 件の結果を同じ場所へ書いてしまう。**
from pathlib import Path as _Path
import shutil as _shutil

if CHUNK:
    chunk_csv = _Path(f"/content/chunks/VAL_{CHUNK}.csv")
    assert chunk_csv.is_file(), f"{chunk_csv} が無い。手順 5-2 を実行すること"
    _shutil.copy(chunk_csv, _Path(KIT_DIR) / "compe/t1/T1_TASKS.csv")
    RESULTS_DIR = f"/content/results/chunk{CHUNK}"
    rows = sum(1 for _ in chunk_csv.open()) - 1
    print(f"小分け {CHUNK}: {rows} 件（{chunk_csv}）→ {RESULTS_DIR}")
else:
    RESULTS_DIR = "/content/results"

for name in ("KIT_DIR", "LEROBOT_SRC", "SUBMISSION_DIR", "SERVER", "WEIGHTS_DIR"):
    assert name in globals(), f"{name} が未定義です。手順 1〜4 を順に実行してください"

print(f"設定: {N_EPISODES} エピソード/タスク, 後処理 {POLICY_ENV or '既定'}")
print(f"  タスク数: {MAX_TASKS if MAX_TASKS else '全部（84 件）'}"
      + ("  ← **疎通確認。終わったら None に戻す**" if MAX_TASKS else ""))
print(f"  サーバー: {SERVER}")
print(f"  キット  : {KIT_DIR}")
print(f"  録画    : {RECORD_DIR or '（撮らない）'}"
      + (f"  which={RECORD_WHICH}" if RECORD_DIR else ""))
```

> **録画はキットにパッチが当たっていないと動かない。** 手順 2 で
> `patch_kit_for_variants.py` を流していれば当たっている（`rollout_recorder.py`
> の複製もそこで行われる）。当たっていなければ**何も起きないだけ**で、
> 評価は普通に走る。

---

**次は [7. 遅さの原因を切り分ける](07_speed_check.md) へ。**
