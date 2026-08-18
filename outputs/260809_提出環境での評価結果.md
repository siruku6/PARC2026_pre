image check: ok (sandbox あり), TRACK=track1
[evaluate] 提出物を静的検査
[evaluate] 提出物を安全に展開: /home/app/assets/submission..zip
[evaluate] 参加者依存を専用 venv にインストール
[evaluate] サンドボックス: 参加者サーバーを 'nobody' で起動し FS を遮断
[evaluate] ポリシーサーバーを起動 (port=8000, python=/tmp/submission_69ciumet_venv/bin/python)
[evaluate] サーバーの起動を待機... (上限 1200秒)
Reducing the number of VLM layers to 16 ...
Loading weights from local directory
Policy server starting on 0.0.0.0:8000
INFO:     Started server process [28]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:57148 - "GET /health HTTP/1.1" 200 OK
[evaluate] サーバー起動確認 (38秒)
[evaluate] 評価対象トラック: ['track1']
[evaluate] 評価パイプラインを実行
[evaluate] タスク用途: 3_Omni
[evaluate] レンダリング: egl（GPU 描画 / EGL_VENDOR=NVIDIA）
[robosuite WARNING] No private macro file found! (__init__.py:7)
[robosuite WARNING] It is recommended to use a private macro file (__init__.py:8)
[robosuite WARNING] To setup, run: python /usr/local/lib/python3.10/dist-packages/robosuite/scripts/setup_macros.py (__init__.py:9)
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
INFO:     127.0.0.1:51532 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:59402 - "POST /reset HTTP/1.1" 200 OK
INFO:     127.0.0.1:59402 - "POST /act HTTP/1.1" 200 OK

... 途中省略

INFO:     127.0.0.1:59402 - "POST /act HTTP/1.1" 200 OK
INFO:     127.0.0.1:59402 - "POST /act HTTP/1.1" 200 OK
INFO:     127.0.0.1:59402 - "POST /act HTTP/1.1" 200 OK

============================================================
提出ID: server_8000
合計時間: 147.6秒

  track1: 総合スコア 0.058
============================================================
[evaluate] 完了
[evaluate] ポリシーサーバーを停止 (PID=28)
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [28]