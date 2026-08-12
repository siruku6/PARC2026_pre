# 8. 評価を実行する

> **いまは [`ce.run(chunk=N)`](../../../solution/tools/colab_eval.py) が、
> ここと手順 9 を 1 回の呼び出しで通す。** このページは
> **なぜ画面へ出さずログへ書くのか**、**なぜ `--verbose` を付けないのか**、
> **落ちたときにどこを見るか**を残す。

**手順 6 を実行済みであること。** サーバーを起動し、キットのパイプラインを当てる。

> **画面には出さない。ログファイルへ書く。**
> 以前は評価の出力をそのままセルへ流していたが、**ブラウザが固まる**。
> Colab の出力はブラウザ側にすべて保持されるため、数万行を出すと Chrome が
> 応答しなくなり、貼り付けることすらできなくなる。ノートブックのカーネル側にも
> 出力が積み上がるので、メモリ不足で評価ごと落ちうる。
>
> **`--verbose` も外した。** これは root logger を DEBUG にする指定で、
> `urllib3` が `/act` 1 回ごとに 1 行出す原因だった（数万行の正体）。
> 手順 2 のパッチがエピソードごとの結果を `logger.info` で出すようになったため、
> **DEBUG に落とす理由がもう無い。**
>
> 画面に出るのは**エピソードの結果行と、60 秒ごとの心拍だけ**である。
> 全文は `LOG_DIR/eval_*.log` に残るので、あとから grep できる。

```python
# === 評価の実行 ===
# 設定は 手順 6 で決めている。条件を変えて比べるときは 手順 6 の POLICY_ENV だけを変えて、
# 手順 6 → 8 の順に流し直す。
# 版 2。**この行が出ないなら古いセルを実行している**（逐次退避は 8/11 22:50 に入れた）。
print(f"[8] 版 2   CHUNK={CHUNK or 'なし'}  →  {RESULTS_DIR}")

# **手順 6 を流さずに CHUNK を書き換えただけなら、ここで止まる。**
# それに気づかないと、同じ 28 件をもう一度 51 分かけて回すことになる
# （画面上は何も変わらないので、集計を見比べるまで分からない）。
kit_csv = Path(KIT_DIR) / "compe/t1/T1_TASKS.csv"
kit_text = kit_csv.read_text()
if CHUNK:
    want = Path(f"/content/chunks/VAL_{CHUNK}.csv")
    assert want.read_text() == kit_text, (
        f"キットが指しているのは VAL_{CHUNK}.csv ではない。**手順 6 を実行すること**"
    )
print(f"    タスク {len(kit_text.splitlines()) - 1} 件  録画 {RECORD_DIR or '（撮らない）'}")

log = open("/content/server.log", "w")
server = subprocess.Popen(
    [sys.executable, str(SERVER), "--port", str(PORT)],
    cwd=str(SUBMISSION_DIR), env=server_env, stdout=log, stderr=subprocess.STDOUT,
)
print(f"サーバー起動 (pid={server.pid})  設定: {POLICY_ENV or '既定'}")
# **サーバーの出力はファイルへ流している。** 版と実効設定はそこにしか出ない。
# 起動待ちのあとに読み出して、意図した構成で立っているかを画面で確かめる。

for elapsed in range(300):
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
        print(f"/health 応答（{elapsed} 秒）")
        break
    except Exception:
        if server.poll() is not None:
            print(open("/content/server.log").read()[-3000:])
            raise RuntimeError("サーバーが起動に失敗しました")
        time.sleep(1)

# 版と実効設定を画面へ出す。**古い policy_server.py のままだとこの行が出ない**ので、
# 差し替え忘れにその場で気づける。
for line in open("/content/server.log"):
    if line.startswith("[parc]"):
        print(line.rstrip())

EVAL_LOG = Path(LOG_DIR) / f"eval_{time.strftime('%H%M%S')}.log"
HEARTBEAT_SEC = 60          # 画面に心拍を出す間隔

# **[結果] 行は 1 本ごとに Drive へ書く。** /content はランタイムが切れると
# 消える。手順 9 の退避を待つと、落ちた回の 40 分ぶんが丸ごと失われる（8/11 に失った）。
# 1 行ずつ flush するので、**落ちても直前のエピソードまで残る。**
LIVE_LOG = None
try:
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    live_dir = Path("/content/drive/MyDrive/parc2026/eval/live")
    live_dir.mkdir(parents=True, exist_ok=True)
    LIVE_LOG = (live_dir / f"{time.strftime('%m%d_%H%M%S')}"
                f"_chunk{CHUNK or 'all'}.results.txt").open("w", encoding="utf-8")
    print(f"逐次退避: {LIVE_LOG.name}")
except Exception as error:
    print(f"**Drive へ逐次退避できない: {error}** 落ちたらこの回は失われる")

started = time.time()
done = collided = succeeded = 0
last_beat = started

try:
    process = subprocess.Popen(
        [f"{KIT_DIR}/venv/bin/python", "-m", "pipeline",
         "--server-url", f"http://localhost:{PORT}",
         "--track", "track1",
         "--n-episodes", str(N_EPISODES),
         "--max-steps", str(MAX_STEPS),
         # **疎通確認は 手順 6 の MAX_TASKS で切り替える。** ここを手で
         # 書き換えると消し忘れる（84 件のはずが 5 件のまま走る）。
         *(["--max-tasks", str(MAX_TASKS)] if MAX_TASKS else []),
         "--output-dir", RESULTS_DIR],
        # --verbose は付けない。root logger が DEBUG になり、urllib3 が
        # /act 1 回ごとに 1 行出すため数万行になる。エピソードごとの結果は
        # 手順 2 のパッチが logger.info で出すので、DEBUG に落とす必要が無い。
        cwd=str(KIT_DIR), env=eval_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    # **全文はファイルへ、画面へは要点だけ。**
    # Colab の出力はブラウザ側に全部保持される。数万行流すと Chrome が固まり、
    # カーネル側にも積み上がってメモリ不足で落ちる。
    with EVAL_LOG.open("w", encoding="utf-8") as sink:
        for line in process.stdout:
            sink.write(line)

            if "[結果]" in line:
                done += 1
                collided += "衝突" in line
                succeeded += "成功" in line
                if LIVE_LOG is not None:
                    LIVE_LOG.write(line)
                    LIVE_LOG.flush()          # **落ちても直前まで残す**
                print(f"[{time.time() - started:5.0f}s] {line.rstrip()[-110:]}")
                last_beat = time.time()
            elif time.time() - last_beat >= HEARTBEAT_SEC:
                print(f"[{time.time() - started:5.0f}s] 実行中… "
                      f"{done} エピソード完了（成功 {succeeded} / 衝突 {collided}）")
                last_beat = time.time()

    process.wait()
finally:
    server.terminate()
    if LIVE_LOG is not None:
        LIVE_LOG.close()

print(f"\n所要 {time.time() - started:.0f} 秒  終了コード {process.returncode}")

# **[結果] 行だけを小さいファイルへ抜き出す。**
# 全文ログは数 MB あり、手元へ持ち帰るのが億劫になる。エピソードごとの
# 結果と、ぶつけた物体だけなら数十 KB で済む。
RESULTS_TXT = EVAL_LOG.with_suffix(".results.txt")
RESULTS_TXT.write_text(
    "\n".join(l for l in EVAL_LOG.read_text(encoding="utf-8").splitlines()
              if "[結果]" in l),
    encoding="utf-8",
)

if process.returncode != 0:
    print("--- ログの末尾 ---")
    print(EVAL_LOG.read_text(encoding="utf-8")[-3000:])

print(f"全文: {EVAL_LOG}（{EVAL_LOG.stat().st_size / 1e6:.1f} MB）")
print("**結果の読み取り・集計・退避は 手順 9 で行う。続けて手順 9 を実行すること。**")
```

**このセルは評価を回すだけである。** 読み取りも集計も Drive への退避も
[手順 9](09_read.md) に集約してある。役割が混ざると、
どのセルを実行したのか分からなくなる。

> **セルを消してしまったら、このページから貼り直す。**
> ノートブックは使い捨てだが、セルの内容はこのリポジトリが正である。

> **手順 2 のパッチを当てていないと `[結果]` が 1 行も出ない。**
> その場合、画面には 60 秒ごとの心拍だけが出て `0 エピソード完了` のままになる。
> 進んではいるので、`EVAL_LOG` を見れば分かる。
>
> `--verbose` を外しても `logger.info` は出る。`pipeline/cli.py` は
> `log_level = logging.DEBUG if args.verbose else logging.INFO` なので、
> **既定が INFO** である。`--verbose` は DEBUG に下げるための指定でしかない。

**設定を変えて何度も回す前に、メニューの「編集 → すべての出力を消去」を実行すること。**
Colab の出力は**セッションを通じて積み上がる**。1 回分が収まっていても、
3 回 4 回と回すうちにブラウザが重くなる。

## 途中で落ちたとき

**終了コードとログの末尾が自動で出る。** それを見て切り分ける。

| 症状 | 見当 |
|---|---|
| 終了コードが `-9`（SIGKILL） | **メモリ不足。** Colab がプロセスを殺した。`N_EPISODES` を下げる |
| 終了コードが `1` で末尾に traceback | その例外が原因。ログをそのまま読む |
| セル自体が消える・ブラウザが固まる | 出力の量。**このセルが古い版のままでないか確認する** |

途中まででも `[結果]` 行は残っているので、**そこまでの成功率と衝突は集計できる。**
次の「結果を読む」の集計セルは、`eval_result` の JSON が無くてもログだけで動く。

## 進んでいるか確かめる

ノートブックのカーネルは評価セルで占有されているため、**別のセルを実行できない**。
判断材料は 3 つある。

**① 心拍が出ているか。** 60 秒ごとに `実行中… N エピソード完了` が出る。
これが増えていれば進んでいる。

**② `PYTHONUNBUFFERED=1` を入れ忘れていないか。**
入れないと子プロセスの出力がブロックバッファリングされ、数 KB 溜まるまで
一行も表示されない。**動いているのに止まって見える。**

**③ 中断してサーバーログを見る。**
評価セルを止めても、`finally` でサーバーが終了し、ログは残る。

```python
log = open("/content/server.log").read()
print("POST /act の回数:", log.count("POST /act"))
print("POST /reset の回数:", log.count("POST /reset"))
print(log[-1500:])
```

`POST /act` が数百〜数千回あれば**確実に進んでいた**。回数からどこまで進んだかも分かる。

| `/act` の回数 | 意味 |
|---|---|
| 0 | 評価が始まっていない。環境構築か EGL の問題 |
| 数十 | 1 エピソードの途中。遅いだけの可能性が高い → EGL を測る |
| 数百以上 | 正常に進んでいた。`PYTHONUNBUFFERED` を入れて回し直せばよい |

**中断せずに待てるなら待つ方がよい。** 結果の JSON は最後にまとめて
書かれるため、途中で止めるとその回は捨てることになる。
80 エピソードで 45〜60 分が目安である。

> **`--timeout` は指定しない。** 既定の 10 秒が本番と同じであり、GPU なら
> 推論 0.85 秒で余裕がある。ここで落ちるなら本番でも落ちる。

---

---

**次は [9. 結果を読む](09_read.md) へ。**
