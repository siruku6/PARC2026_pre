# 023 明日の手元評価用データセット——作れる組を全部作る

> **状態: 検討中（規模の決定待ち）** ／ 更新: 2026-09-07
>
> **これは明日の手元評価専用である。**`compe/t3` の 4 タスクだけを見る学習に使う。
> **本番学習用は [020](020_reverse_dataset_regeneration.md)** で、そちらは
> 2 因子を振って 1 タスク 50 本に絞る。**方針が逆なので混同しないこと。**

## 1. 方針——**絞らない**

| | 020（本番） | **023（明日の手元評価）** |
|---|---|---|
| 目的 | 学習全体に混ぜる | **4 タスクだけを評価する** |
| 1 指示文あたり | 配布と同程度（50 本） | **作れるだけ** |
| 組の選び方 | 試走 → 割り当て | **全部書き出す** |
| two mugs | 含める | **含める**（うまくいかないこと自体が示唆になる） |
| 実装の追加 | `--pairs` が必要 | **不要。いまのコードで回る** |

**絞らない理由。**タスクが 4 つしかないので、1 指示文あたりの本数の偏りを気に
する段ではない。**いま知りたいのは「作ったデータで 4 タスクが動くか」だけ**である。

## 2. 手持ちのデモと、作れる組の数

抜き出し済みの順方向デモ（`work/probe_020/all_*.npz` と `work/probe_019/ep_twomugs20.npz`）:

| タスク | 抜いたデモ | 配布にある総数 | **順再生の通過率** | 使えるデモ（見込み） |
|---|---|---|---|---|
| 本 | **44** | 44 | 未測定 | 〜30 |
| 缶 | **44** | 44 | 未測定 | 〜30 |
| 赤マグ | **34** | 34 | **70%**（実測 20 本中 14） | 〜24 |
| two mugs | **20** | 313 | **50%**（実測 20 本中 10） | 〜10 |

**組の総数は「使えるデモ × init 50 行」である。**全部だと **約 4,700 本**になる。

## 3. **「全部」は明日には終わらない**——実測に基づく試算

書き出し（`--write-npz`）の実測は **31 秒/本・npz 60 MB/本**、env 構築が 215 秒
（[020 の 2 節](020_reverse_dataset_regeneration.md)）。two mugs は 1 本が 280 コマ
あるので **50 秒/本・95 MB/本**とみておく。

| 1 タスクで使うデモ D | エピソード（4 タスク計） | 生成時間 | npz |
|---|---|---|---|
| 3（旧方式と同じ） | 600 | **6 時間** | 39 GB |
| **6** | **1,200** | **12 時間** | **78 GB** |
| 10 | 2,000 | 20 時間 | 130 GB |
| **全部**（30/30/24/10） | **4,700** | **44 時間** | **290 GB** |

**空きディスクは 241 GB なので、「全部」は時間だけでなく容量でも入らない。**

**ただし逃げ道が 2 つある。**

- **変換してから消す。**npz は LeRobot 形式に変換すると 67 分の 1 に縮む。
  タスクごとに「生成 → 変換 → npz 削除」を回せば、容量の上限は外れる（時間は変わらない）
- **タスクを並列に回す。**4 タスクは独立なので、コンテナを 4 つ同時に立てれば
  理屈の上では 4 分の 1 になる。**EGL の描画が競合するかは未検証。**
  試すなら 2 タスクで測ってから

## 4. 効率の良い点——**使えないデモは順再生だけで捨てられる**

`gripper_shift_sweep.py` は **goal が求める物を 1 つでも動かせていないデモを
その場で飛ばす**（[reference/30 の 7 節](../../reference/30_reverse_grasp_pose.md)）。
飛ばす判定は**順再生 1 回（約 20 秒）**で済み、init 50 行ぶんの逆再生は走らない。

**だから D を大きくしても、無駄になるのは 1 デモあたり 20 秒だけである。**
デモを 44 本渡して 30 本が生き残るなら、捨てた 14 本のコストは 5 分に過ぎない。

## 5. 手順（実装の追加は不要）

### 5.0 置き場と順方向デモ

```bash
cd /home/ubuntu/src/PARC2026_pre
mkdir -p work/probe_021
cp work/probe_020/all_book.npz work/probe_020/all_soup.npz \
   work/probe_020/all_mug.npz work/probe_021/
cp work/probe_019/ep_twomugs20.npz work/probe_021/
```

**`EP_NPZ` は `$OUT` からの相対でしか指せない。**

### 5.1 生成（`parc2026-final`。タスクごとに 1 回）

```bash
COMMON="MODE=phased INIT_SOURCE=t3 GRASP_FRAME=clamped STANDOFF=0.06 \
        T3_INIT=all MAX_APPROACH=120 LIFT=0.08 RELEASE=contact SETTLE=20 \
        MAX_TAIL=5 MAX_STEPS=300"
R=solution/tools/track3_reversal/run_gripper_shift_sweep.sh
O=$PWD/work/probe_021

env OUT=$O TASK=book    EP_NPZ=all_book.npz  SYMMETRY=2 \
    WRITE_NPZ=reverse_npz_book    $COMMON bash $R "0" "<デモ番号をカンマ区切り>"
env OUT=$O TASK=soup    EP_NPZ=all_soup.npz  GRASP_STYLE=top-down \
    WRITE_NPZ=reverse_npz_soup    $COMMON bash $R "0" "<…>"
env OUT=$O TASK=mug     EP_NPZ=all_mug.npz   LEVEL=1 \
    WRITE_NPZ=reverse_npz_mug     $COMMON bash $R "0" "<…>"
env OUT=$O TASK=twomugs EP_NPZ=ep_twomugs20.npz LEVEL=1 \
    WRITE_NPZ=reverse_npz_twomugs $COMMON bash $R "0" ""
```

**`--episodes` を空にすると npz の全デモを回す。**D を絞るなら番号を並べる。
**タスクごとの掴み方・置き方は [020 の 5 節](020_reverse_dataset_regeneration.md)
と同じで、変えてはいけない。**

**json はタスクごとに上書きされる。**1 回ごとに
`cp work/probe_021/gripper_shift_sweep.json work/probe_021/sweep_<タスク>.json`。

### 5.2 変換 → merge → npz を消す

```bash
N=solution/tools/track3_reversal/run_npz_to_lerobot.sh
for t in book soup mug twomugs; do
  env OUT=$O bash $N reverse_npz_$t reverse_lerobot_$t local/t3_reverse_$t
done
env OUT=$PWD/work bash solution/tools/track3_reversal/run_lerobot_merge.sh \
  probe_021/reverse_lerobot_all \
  probe_021/reverse_lerobot_book probe_021/reverse_lerobot_soup \
  probe_021/reverse_lerobot_mug probe_021/reverse_lerobot_twomugs
```

**容量が厳しいなら、タスク 1 つぶんを変換して `inspect` が通った時点で
その npz を消す。**変換前に消してはいけない。

## 6. 出したら報告すること

| 何 | なぜ |
|---|---|
| タスクごとの **使えたデモ数 / 渡したデモ数** | 通過率が本・缶で未測定なので、ここで初めて分かる |
| タスクごとの **本数と、埋まった init の数** | 落ちた init があるならどれか |
| `disturb_mm` / `drop_mm` / `n_steps` の最大 | 置き方が崩れていないか（[020 の 7 節](020_reverse_dataset_regeneration.md)） |
| **two mugs の歩留まり** | ここが低いこと自体が示唆になる |

## 7. 決めてほしいこと

| # | 決めること | 選択肢 |
|---|---|---|
| 1 | **1 タスクで使うデモ数 D** | 6（12 時間・78 GB）／ 10（20 時間・130 GB）／ 全部（44 時間・要変換削除） |
| 2 | タスクを並列に回すか | 直列（確実）／ 並列 4（未検証。EGL の競合を確かめてから） |
| 3 | 変換を挟んで npz を消しながら回すか | D が大きいなら必須 |

## 8. 関連

| 何 | どこ |
|---|---|
| **本番学習用の生成**（2 因子を振って 50 本に絞る） | **[020](020_reverse_dataset_regeneration.md)** |
| 掴む姿勢の決め方 | [reference/30](../../reference/30_reverse_grasp_pose.md) |
| 運び方と置き方 | [reference/31](../../reference/31_reverse_place_without_contact.md) |
| 数の関係（デモ × init の 2 因子） | [reference/06](../../reference/06_training_data_composition.md) の末尾 |
