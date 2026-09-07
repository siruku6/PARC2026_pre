# 020 逆向きデータ 389 本を、擦らない・落とさない作り方で作り直す

> **状態: 依頼中** ／ 受け側: 未定 ／ 更新: 2026-09-07
>
> **判断は要らない。設定は確定していて、長いだけの作業である。**
> 詰まったら「4. 落とし穴」を先に見ること。

## 1. なぜ作り直すか

いまある 389 本（`work/probe_017/reverse_lerobot_all`）は、**掴んだ物を横へ
引きずって下の皿やトレイを動かし、空中で放して落としている。**どちらも直し方が
決まり、実測で消えることを確かめた（[reference/31](../../reference/31_reverse_place_without_contact.md)）。
**足すのは環境変数 2 つ（`LIFT=0.08 RELEASE=contact`）だけ**で、通る本数は
変わらない。

| | 直す前 | 直した後 |
|---|---|---|
| 他の物が動いた量（皿・トレイ） | 最大 21.0 mm | **0.9 mm** |
| 放してから落ちた高さ（缶） | 36.9 mm | **2.6 mm** |
| 同（赤マグ。`LEVEL=1` も足す） | 12.3 mm | **1.5 mm** |
| 逆向き goal | 10/10 | 10/10（変わらない） |

**`LEVEL=1` は赤マグにだけ付ける。**缶に付けると最後の一手で 90° 回そうとして
隣のケチャップを 116 mm 薙ぎ倒す（[reference/31 の 5 節](../../reference/31_reverse_place_without_contact.md)）。

**`MAX_TAIL=5` と `MAX_STEPS=300` は全タスクに付ける。**置いたあとに腕が初期姿勢
へ戻る区間は goal にも学習にも要らず、**本番は 300 コマで打ち切られる**
（[reference/31 の 6 節](../../reference/31_reverse_place_without_contact.md)）。
物理は 1 本ごとに完全一致で変わらない。

## 2. 作り直す手順

**3 タスクを別々に生成 → それぞれ LeRobot 形式へ変換 → 1 つに merge。**
シムの画像（`parc2026-final`）に lerobot は入っておらず、lerobot の画像
（`parc2026-train-pi05`）に LIBERO は入っていないので、**生成と変換は必ず
別のコンテナに分かれる。**

### 2.0 置き場を作って、順方向デモの npz を持ってくる

```bash
cd /home/ubuntu/src/PARC2026_pre
mkdir -p work/probe_019
cp work/probe_017/ep_data.npz work/probe_017/ep_soup.npz work/probe_019/
cp work/probe_017_mug/ep_mug20.npz work/probe_019/
```

**`EP_NPZ` は `$OUT` からの相対で解決される**（コンテナに `$OUT` を `/out` として
mount するため）。よそのディレクトリのファイルは指せない。

### 2.1 生成（`parc2026-final`。3 回まわす）

**共通して渡すもの。**`T3_INIT=all` が配布 T3 の init 50 行すべてを使う指定で、
これがデータ量を決める。

```bash
COMMON="MODE=phased INIT_SOURCE=t3 GRASP_FRAME=clamped STANDOFF=0.06 \
        T3_INIT=all MAX_APPROACH=120 LIFT=0.08 RELEASE=contact SETTLE=20 \
        MAX_TAIL=5 MAX_STEPS=300"
R=solution/tools/track3_reversal/run_gripper_shift_sweep.sh
```

| タスク | 順方向デモ | 掴み方 | 置き方 | 期待 |
|---|---|---|---|---|
| 本 | `4061,4062,4063` | `transfer`（既定）＋ `SYMMETRY=2` | — | 150/150 |
| 缶 | `3696,3697` | `GRASP_STYLE=top-down` | — | 100/100 |
| 赤マグ | `2206,2199,2202` | `transfer`（既定） | **`LEVEL=1`** | 139/150 |

```bash
env OUT=$PWD/work/probe_019 TASK=book EP_NPZ=ep_data.npz SYMMETRY=2 \
  WRITE_NPZ=reverse_npz_book $COMMON bash $R "0" "4061,4062,4063"

env OUT=$PWD/work/probe_019 TASK=soup EP_NPZ=ep_soup.npz GRASP_STYLE=top-down \
  WRITE_NPZ=reverse_npz_soup $COMMON bash $R "0" "3696,3697"

env OUT=$PWD/work/probe_019 TASK=mug EP_NPZ=ep_mug20.npz LEVEL=1 \
  WRITE_NPZ=reverse_npz_mug $COMMON bash $R "0" "2206,2199,2202"
```

**`--write-npz` は逆向き goal が立った回だけ書く。**落ちた回はディスクに残らない。

**`docker run` を直に叩くと拒否される。**必ずこのランナー（スクリプト）を通す。

### 2.2 確かめる — **本数だけ見て次へ進まない**

`work/probe_019/gripper_shift_sweep.json` の各行に、今回から次の 2 つが入る。
**この 2 つが直したかったものそのものなので、ここで見る。**

| 列 | 意味 | 通す目安 |
|---|---|---|
| `disturb_mm` | 掴む物**以外**が動いた量（物ごと） | 成功した回で 1 mm 未満 |
| `drop_mm` | 放してから物が落ちた高さ | 数 mm |

```bash
python3 - <<'PY'
import json
d = json.load(open("work/probe_019/gripper_shift_sweep.json"))
for r in d:
    ok = [s for s in r["shifts"] if s["reverse_goal"]]
    print(f"ep {r['episode']}: {len(ok)}/{len(r['shifts'])}  "
          f"他の物 最大 {max((s['disturb_max_mm'] for s in ok), default=0):.1f} mm  "
          f"落下 最大 {max((s['drop_mm'] for s in ok), default=0):.1f} mm")
PY
```

**json はタスクごとに上書きされる。**3 タスクぶん残したいなら、生成 1 回ごとに
`cp work/probe_019/gripper_shift_sweep.json work/probe_019/sweep_<タスク>.json`
しておく。

### 2.3 LeRobot 形式へ変換（`parc2026-train-pi05`。3 回）

```bash
N=solution/tools/track3_reversal/run_npz_to_lerobot.sh
for t in book soup mug; do
  env OUT=$PWD/work/probe_019 bash $N reverse_npz_$t reverse_lerobot_$t local/t3_reverse_$t
done
```

`build` のあと `inspect` まで走る。**揃っていることと、学習が読めることは別である**
——動画の符号化が壊れていても `build` は通り、`inspect` で初めて落ちる。

変換で 67 分の 1 に縮む（npz は画素のまま、変換後は動画）。npz は本 150 本で
9.7 GB になるので、**変換が通ったら消してよい。**

### 2.4 1 つに merge

```bash
env OUT=$PWD/work bash solution/tools/track3_reversal/run_lerobot_merge.sh \
  probe_019/reverse_lerobot_all \
  probe_019/reverse_lerobot_book probe_019/reverse_lerobot_soup \
  probe_019/reverse_lerobot_mug
```

**`merge` は 2 つずつしか混ぜられない**ので、ランナーが順に足す。中間物は
`_tmpN` に置かれて最後に消える。**入力は書き換えない。**

### 2.5 出来上がりの目安

| | 本数 | コマ数 |
|---|---|---|
| 本 | 150 | 26,443 |
| 缶 | 100 | 17,141 |
| 赤マグ | 139 | 24,191 |
| **合計** | **389** | **67,775** |

**コマ数は今回のほうが増える。**真上へ逃げる段と、下ろして触れるまで待つ段が
足されるため（本タスクで 177 → 213 コマの例がある）。**本数が 389 から動いたら、
そこは追うこと。**

## 3. 済んだら

- [reference/31 の 5 節](../../reference/31_reverse_place_without_contact.md) の
  実測表に、**全 init での値**を足す（いまは init 5 行だけの値である）
- [018](018_reverse_dataset_writeout_plan.md) の 4 節の表と、
  [reference/30 の 7 節](../../reference/30_reverse_grasp_pose.md) の実績表を更新する
- **古い 389 本（`work/probe_017/reverse_lerobot_all`）は消さずに残す。**
  学習で比べるまで、どちらが良いかは分からない

## 4. 落とし穴

| 症状 | 原因と対処 |
|---|---|
| `docker run` が拒否される | 直書きが拒否される。**ランナー経由なら通る** |
| `EP_NPZ` が見つからない | `$OUT` からの相対でしか指せない。`$OUT` の直下へコピーする |
| merge の中間物が消えずに残る | コンテナが root で書いたものを、所有を戻す前に消そうとしている。ランナーは chown を先にやる |
| 接近が 120 コマに張り付く | `MAX_APPROACH` の既定 60 は本タスク（約 200 mm）の値。缶・マグは 300 mm 以上ある |
| 缶で腕が横を向いてトレイの壁に当たる | `GRASP_STYLE=top-down` を渡し忘れている（[reference/30 の 4 節](../../reference/30_reverse_grasp_pose.md)） |
| 落下や接触の原因が読めない | `TRACE=20` を足すと、放す前後を 1 コマずつ `grip` / `z` / `tilt` / `touch` で出す |
| コマ数が 300 を超えて npz が書かれない | 表示に `goal=○/長` と出る。`MAX_TAIL` を短くするか、短い順方向デモを選ぶ |

## 5. 関連

| 何 | どこ |
|---|---|
| **運び方と置き方**（今回足す 2 つの設定の理由と実測） | **[reference/31](../../reference/31_reverse_place_without_contact.md)** |
| 掴む姿勢の決め方（`transfer` と `top-down` の使い分け） | [reference/30](../../reference/30_reverse_grasp_pose.md) |
| 1 本に何が要るか・npz の鍵 | [reference/29](../../reference/29_training_data_requirements.md) |
| ここまでの段取り全体 | [018](018_reverse_dataset_writeout_plan.md) |
