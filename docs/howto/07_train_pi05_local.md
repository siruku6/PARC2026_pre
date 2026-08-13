# 07. π0.5をこのマシン(H100 20GB)で学習する

SmolVLAの提出は保険としてそのまま残し、**並行してπ0.5が間に合うか試す**。
この文書は、そのローカル側(H100 20GB 1枚)の本番学習を実際に回す手順である。
出発点の調査は[docs/reference/14_pi05_switch_feasibility.md](../reference/14_pi05_switch_feasibility.md)、
学習スクリプト自体は[solution/train/train_pi05.py](../../solution/train/train_pi05.py)を参照。

## batch sizeは16に決めた

学習スクリプト自身が出す`mem_gb`(GPUメモリのピーク使用量)を、batchを変えながら
20〜50歩だけ回して測った。

| batch | mem_gb | smp/s(1秒あたりのサンプル数) |
|---|---|---|
| 2 | 12.90 | 2〜4 |
| 8 | 13.99 | 9〜10 |
| 16 | 16.16〜16.27 | 10〜12 |

19.8GB中16GB強で、まだ3.5GBほど余裕がある。ただしbatch 8→16の増分(+2.3GB)は
batch 2→8の増分(+1.1GB)より大きく、このペースで24まで上げると18GB台に達し
OOMのリスクが増える一方、`smp/s`の伸びは頭打ちに近い(8→16で2割増程度)。
**伸びしろよりOOMのリスクの方が大きくなる境目**と判断し、**batch=16で確定**した。

## 実行前に: ディスクの空き容量を確認する

**チェックポイント1個が約11GBあり、LeRobotは古いチェックポイントを自動で
消さない。** 10時間規模の学習を`save_freq`の既定(500歩ごと)で回すと、
数十個×11GB=数百GBに達して確実に途中で止まる(GitHub上の`train_utils.py`を
確認済み。削除する仕組みが無い)。

学習を始める前に、次の2つを確認すること。

1. `df -h /`で空き容量が十分か(目安: 最低30GB以上)
2. `/home/ubuntu/.cache/huggingface`と`work/train`が自分のユーザー所有になっているか
   (`ls -la`で確認。過去に`docker run`をroot権限で実行した形跡が残っていると
   `root:root`のままになっており、書き込みに失敗する——下の「詰まりやすい点」参照)

## 本番の学習を始める

```bash
docker run --gpus all --rm -it \
    --user "$(id -u):$(id -g)" \
    -v "$PWD/work/train:/out" -v "$HOME/.cache/huggingface:/hf_home" \
    parc2026-train-pi05 \
    python /workspace/train_pi05.py --run-name pi05_full_local --steps 24000 --batch-size 16
```

24,000歩を選んだ理由は、`smp/s≈10.5`で見積もると`24000×16÷10.5÷3600≈10.2時間`と、
確保できる時間(10時間以上)に収まるためである。**歩数(`--steps`)は学習率の
下げ方(スケジュール)そのものを決める値なので、大きくしすぎて時間内に完走
できないと、学習率が下がりきらないまま終わってしまう**(SmolVLA側で同じ理由の
教訓がある: [docs/reference/11_training_schedule.md](../reference/11_training_schedule.md))。
確保できる時間が伸びるようなら、この`--steps`を上げてから最初から回し直すこと
(`--resume`では歩数を後から変えられない——`train.py`の`_build_resume_command`の
コメント参照)。

**別のターミナルで、チェックポイントの間引きを並行して回す。**

```bash
bash solution/train/prune_checkpoints.sh work/train/pi05_full_local 2
```

最新2個だけを残し、古いものを30秒おきに削除し続ける。学習が終わったら
`Ctrl+C`で止めてよい(その時点の最新チェックポイントは残る)。

## 詰まりやすい点(実機で実際に踏んだもの)

ローカルでこのDockerイメージを組み立てる過程で、3つの権限まわりの問題に
当たった。再発したときに同じ調査をしなくて済むよう記録しておく。

| 症状 | 原因 | 直し方 |
|---|---|---|
| `hf auth login`が`PermissionError`で落ちる | `~/.cache/huggingface`がroot所有(過去のroot実行の名残) | `sudo chown -R $(whoami):$(whoami) ~/.cache/huggingface` |
| `--user`を付けたら`/root/.cache/huggingface/token`で`Permission denied` | `/root`自体が`drwx------`で、非rootは中身の権限に関係なく通過できない | `HF_HOME`を`/root`配下ではない場所(`/hf_home`)にする。[Dockerfile.pi05](../../solution/train/Dockerfile.pi05)で対応済み |
| `getpass.getuser()`が`KeyError: getpwuid(): uid not found`で落ちる | `--user`のUIDが`/etc/passwd`に無い(torchの内部が呼んでいる) | `USER`/`LOGNAME`環境変数を設定しておく。同上のDockerfileで対応済み |
| チェックポイント保存の直前で`/out/<run-name>`に`Permission denied` | マウント先(`work/train`)がroot所有 | `sudo chown -R $(whoami):$(whoami) work/train` |

いずれも「過去にrootとして`docker run`した結果、ホスト側のディレクトリが
root所有のまま残る」という同じ根に行き着く。**`--user "$(id -u):$(id -g)"`を
最初から付けて回していれば起きなかった。**

## 学習が終わったら

`work/train/pi05_full_local/checkpoints/<最終ステップ>/pretrained_model/`に
提出可能な形の重みが残る(LoRAではなくフルパラメータ学習なので、`merge.py`に
相当するマージ手順は不要——そのまま`PI05Policy.from_pretrained()`で読める)。

次はこの重みを使ってpolicy_server.py側にπ0.5用の`MyPolicy`実装を用意し、
L4 24GB・10秒/推論の制約に収まるかを実測する段階に進む。

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| 24,000歩まで完走したときの実際の所要時間(`smp/s`は50歩の実測値で、長時間走らせると変わる可能性がある) | 学習ログの経過時間を見る |
| batch=16のまま学習データに手作業の画像を混ぜたときにメモリ・速度が変わるか | 混ぜたデータセットで同じsmoke-testを再実行する |
