# 9. 結果を読む・集計する・退避する

**出力に関わる処理はすべてこのセルにある。** 手順 8 は評価を回すだけで、
読み取りも集計も Drive への退避もここでやる。

**単体で何度でも実行してよい。** 直近の評価ログを自動で探すため、
手順 8 のあとならいつでも走る。**評価が途中で落ちて JSON が無くても、
ログだけで集計できる。**

```python
# === 9. 結果を読む・集計する・退避する ===
# 版 3。**この行が出ないなら古いセルを実行している。**
# 貼り替えたつもりで古いままだと、直したはずのエラーがそのまま再現し、
# 原因を別の場所に探しに行くことになる（実際に起きた）。
print("[F] 版 3")

import json, math, re, shutil, time
from collections import Counter
from pathlib import Path

RESULT_JSON = Path(f"{RESULTS_DIR}/server_8000.json")
SERVER_LOG = Path("/content/server.log")
logs = sorted(Path(LOG_DIR).glob("eval_*.log"), key=lambda p: p.stat().st_mtime)
EVAL_LOG = logs[-1] if logs else None

print(f"評価ログ  {EVAL_LOG or '（無い。手順 8 を実行していない）'}")
print(f"JSON      {RESULT_JSON if RESULT_JSON.exists() else '（無い。評価が完走していない）'}")
# JSON は完走時にしか書かれない。落ちた回では**前回の結果が残っている**ため、
# 気づかずに古い数字を読むことがある。時刻で検出する。
if RESULT_JSON.exists() and EVAL_LOG and RESULT_JSON.stat().st_mtime < EVAL_LOG.stat().st_mtime - 60:
    print("  **注意: JSON が評価ログより古い。前の回の結果を読んでいる。**")


def wilson(rate: float, n: int) -> tuple[float, float]:
    """二項分布の 95% 信頼区間。20 本と 80 本では判定できる差が違う。"""
    z = 1.96
    d = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / d
    margin = z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / d
    return center - margin, center + margin


# --- 1. 集計値（JSON があるときだけ） ---------------------------------
# **JSON があっても中身が結果とは限らない。** トラックが例外で落ちると
# pipeline.py の _errored_track_score が {"error": 1.0, "n_tasks": 0.0} だけを
# 書き出す。ここで鍵の有無を見ずに読むと KeyError になり、**本当の原因である
# 例外の行を探しに行けない。**
if RESULT_JSON.exists():
    track = json.loads(RESULT_JSON.read_text())["tracks"][0]
    overall = track["overall_metrics"]

    if "n_total_episodes" not in overall:
        print("\n**評価が例外で落ちている。** JSON はエラー記録だけを持つ。")
        print(f"  overall_metrics = {overall}")
        if EVAL_LOG:
            log_text = EVAL_LOG.read_text(encoding="utf-8")
            marks = [
                line for line in log_text.splitlines()
                if "Traceback" in line or "Error" in line or "失敗" in line
                or "例外" in line
            ]
            print(f"\n  ログ中のエラー行（末尾 15 行）: 全 {len(marks)} 件")
            for line in marks[-15:]:
                print(f"    {line[:160]}")
            print(f"\n  全文: {EVAL_LOG}")
    else:
        n, rate = overall["n_total_episodes"], overall["mean_success_rate"]
        low, high = wilson(rate, n)

        print(f"\n成功率   {rate:.3f}   95%CI [{low:.3f}, {high:.3f}]  n={n}")
        print(f"衝突率   {overall['mean_collision_rate']:.3f}\n")
        for task in track["tasks"]:
            print(f"  成功 {task['success_rate']:.2f}  衝突 "
                  f"{task['metrics']['collision_rate']:.2f}  {task['task_name'][:50]}")
        print()
        for key in ("mean_avg_cartesian_jerk", "mean_rms_cartesian_jerk",
                    "mean_sparc", "mean_cartesian_path_length"):
            print(f"  {key:32s} {overall[key]:9.4f}")

# --- 2. エピソード単位の集計（ログから） ------------------------------
# JSON には「衝突したか」しか入らない。**何にぶつかったのか**は
# B のパッチが出すログ行にしかない。
RESULTS_TXT = None
if EVAL_LOG:
    lines = [l for l in EVAL_LOG.read_text(encoding="utf-8").splitlines() if "[結果]" in l]
    # 全文は数 MB あって持ち帰るのが億劫になる。結果行だけなら数十 KB。
    RESULTS_TXT = EVAL_LOG.with_suffix(".results.txt")
    RESULTS_TXT.write_text("\n".join(lines), encoding="utf-8")

    if not lines:
        print("\n**[結果] 行が 0 件。B のパッチを当てずに回している。**")
    else:
        outcomes, objects, by_task = Counter(), Counter(), {}
        for line in lines:
            found = re.search(r"\[結果\] (\S+) ep(\d+): (\S+)", line)
            if not found:
                continue
            name, _, outcome = found.groups()
            outcomes[outcome] += 1
            by_task.setdefault(name, Counter())[outcome] += 1
            for obj, _mm in re.findall(r"(\w+)=([\d.]+)mm", line):
                objects[obj] += 1

        total = sum(outcomes.values())
        print(f"\n=== ログから {total} エピソード分 ===")
        for outcome, count in outcomes.most_common():
            print(f"  {outcome:6s} {count:3d}  ({count / total:.1%})")

        if objects:
            print("\nぶつけた物体（衝突したエピソード数）")
            for obj, count in objects.most_common():
                print(f"  {count:3d}  {obj}")

        # --- 摂動カテゴリ別 ---
        # **H の検証用スイートで評価したときに効く。** どのカテゴリが弱いかが
        # 分かれば、学習で何を厚くすべきかが決まる。
        # 使っている CSV に category 列があるときだけ集計する。
        import csv as _csv
        definition = KIT_DIR / "compe/t1/T1_TASKS.csv"
        category_of = {}
        # **CSV が無くてもここで止まらない。** 集計とログの退避のほうが大事で、
        # カテゴリ別は「あれば嬉しい」情報である。
        if definition.is_file():
            with definition.open(newline="") as stream:
                for row in _csv.DictReader(stream):
                    if row.get("category"):
                        category_of[row["task_id"]] = row["category"]
        else:
            print(f"\n（{definition} が無いのでカテゴリ別は省略）")

        if category_of:
            per_category: dict[str, Counter] = {}
            for name, slot in by_task.items():
                key = category_of.get(name, "（不明）")
                per_category.setdefault(key, Counter()).update(slot)
            print("\n摂動カテゴリ別")
            for key in sorted(per_category):
                slot = per_category[key]
                total_here = sum(slot.values())
                rate = slot.get("成功", 0) / total_here
                print(f"  {key:24s} 成功 {rate:.2f}  ({total_here} 本)  "
                      + " ".join(f"{k} {v}" for k, v in slot.most_common()))

        print("\nタスク別")
        for name, slot in sorted(by_task.items()):
            print(f"  {name[:46]:46s} "
                  + " ".join(f"{k} {v}" for k, v in slot.most_common()))

# --- 2.5 撮れた動画を並べる -------------------------------------------
# **見るべき順に並べる。** 打ち切りに張り付いた回が先頭に来る。
# コマ並べ画像も作るので、動画を再生する前に当たりを付けられる。
VIDEO_DIR = Path(RECORD_DIR) if RECORD_DIR else None
# 集計スクリプトは collect_colab_files.py が /content/ へ置いている。
SUMMARIZE = Path("/content/summarize_rollout_videos.py")
if VIDEO_DIR and VIDEO_DIR.is_dir() and any(VIDEO_DIR.glob("*.mp4")):
    print()
    if SUMMARIZE.is_file():
        # キットの venv で回す。cv2 はそちらに入っている。
        subprocess.run([f"{KIT_DIR}/venv/bin/python", str(SUMMARIZE), str(VIDEO_DIR)],
                       check=False)
    else:
        print(f"**{SUMMARIZE} が無い。** 一覧だけ出す:")
        for path in sorted(VIDEO_DIR.glob("*.mp4")):
            print(f"  {path.stat().st_size / 1e3:8.0f} KB  {path.name}")
elif RECORD_DIR:
    print(f"\n**動画が 1 本も無い（{RECORD_DIR}）。**")
    print("  失敗した回が無かったか、パッチが当たっていない。"
          "eval ログに [録画] の行があるか見ること:")
    if EVAL_LOG:
        for line in EVAL_LOG.read_text(errors="replace").splitlines():
            if "[録画]" in line:
                print(f"    {line.strip()}")

# --- 3. Drive へ退避 ---------------------------------------------------
# **/content はランタイムが切れると消える。** 実際に一度失っている。
# 退避に失敗しても集計はもう出ているので、例外で止めない。
def run_label() -> str:
    """設定から、取り違えようのないディレクトリ名を作る。"""
    parts = [f"n{POLICY_ENV.get('PARC_N_ACTION_STEPS', '10')}", f"ep{N_EPISODES}"]
    for key, value in sorted(POLICY_ENV.items()):
        if key != "PARC_N_ACTION_STEPS":
            parts.append(f"{key.replace('PARC_', '').lower()}{value}".replace(".", ""))
    return "_".join(parts)


try:
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")          # 初回は認証を求められる

    destination = (Path("/content/drive/MyDrive/parc2026/eval")
                   / f"{time.strftime('%m%d_%H%M%S')}_{run_label()}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in (RESULT_JSON, RESULTS_TXT, EVAL_LOG, SERVER_LOG):
        if path and path.exists():
            shutil.copy(path, destination / path.name)

    # **動画も一緒に退避する。** これを忘れると 90 分の測定の映像が
    # ランタイム切断で消える。1 本 1 MB 程度なので 60 本でも 60 MB。
    if VIDEO_DIR and VIDEO_DIR.is_dir():
        videos = sorted(VIDEO_DIR.glob("*.mp4")) + sorted(VIDEO_DIR.glob("sheets/*.png"))
        if videos:
            video_destination = destination / "videos"
            video_destination.mkdir(exist_ok=True)
            for path in videos:
                shutil.copy(path, video_destination / path.name)
            print(f"  動画 {len(videos)} 件を {video_destination} へ")

    print(f"\n=== Drive へ退避した ===\n  {destination}")
    for path in sorted(destination.iterdir()):
        if path.is_dir():
            print(f"  {'':8s}    {path.name}/  ({len(list(path.iterdir()))} 件)")
        else:
            print(f"  {path.stat().st_size / 1e3:8.0f} KB  {path.name}")
except Exception as error:
    print(f"\n**Drive への退避に失敗した: {error}**")
    print("左のファイルペインから手動でダウンロードすること:")
    for path in (RESULT_JSON, RESULTS_TXT, EVAL_LOG):
        if path and path.exists():
            print(f"  {path}")
```

## 退避先は設定から自動で決まる

```
/content/drive/MyDrive/parc2026/eval/
├── 0810_091203_n10_ep20/                    … つまみ無し
├── 0810_101500_n10_ep20_ema_alpha03/        … EMA 0.3
└── 0810_113000_n10_ep20_blend_steps3/       … 境界ブレンド 3
```

**どの重みを評価した回なのかが後から分かる。** 手で名前を付けていると、
付け忘れと取り違えが必ず起きる。

## 読む順序

**信頼区間を必ず見ること。** 20 エピソードで成功率 0.10 なら 95%CI は
おおよそ [0.03, 0.30] で、0.20 との差は判定できない。
**条件を比べるなら 1 条件あたり 80 エピソード以上**が要る。

| 優先 | 見るもの | なぜ |
|---|---|---|
| 1 | 成功率と衝突率 | **ゲート**。ここが下がったら他が良くても採用しない |
| 2 | ぶつけた物体 | 同じ物体が繰り返し出るなら、そこが本丸 |
| 3 | jerk / SPARC / 軌道長 | ゲートを通ったエピソードの中での配点 |

`wooden_cabinet` ばかりなら引き出しへ手を入れる動作の問題で、平滑化では
直らない。`basket` ばかりなら置く瞬間に縁へ触れており、`PARC_EMA_ALPHA` が
効く見込みがある。

---

---

**次は [10. 参考（過去の測定と注意点）](10_notes.md) へ。**
