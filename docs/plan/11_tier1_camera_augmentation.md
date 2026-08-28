# 11. Tier 1（Camera Viewpoints 対策の画像拡張）着手記録

**この文書の位置づけ**: [09_demo_perturbation_pipeline.md](09_demo_perturbation_pipeline.md) 2.2 節が
「未着手」としていた Tier 1 に着手した記録。対象は Camera Viewpoints に絞る
（Sensor Noise は[09 §3.2](09_demo_perturbation_pipeline.md#32-高強度sensor-noiseの追加実験実行済み2026-08-22)の
高強度 probe で崩れず投資優先度を下げた）。**GPU/CPU を使う実行はユーザーが行う**——
この session では実行せず、コードの実装・CPU だけでの動作確認・実行コマンドの用意までを行った。

| 節 | 内容 |
|---|---|
| 1 | 着手前に確認したこと（他セッションとの競合・GPU/CPU の空き状況） |
| 2 | 作り直し（本格対応）——`scripted_demo.py` の変更なしで動くことを確認した |
| 3 | 2D 近似（安価な並行策）——`train_pi05.py` に実装し、CPU で動作確認した |
| 4 | 比較の手順（学習後に実行する） |
| 5 | 確認したい仮定 |

---

## 1. 着手前に確認したこと

**Tier 0 のデータ量産が別セッションによって実行中で、`scripted_demo.py` は変更していない。**
`docker ps -a` で確認したところ、`awesome_lehmann`（`parc2026` イメージ）が
`scripted_demo.py record --task KITCHEN_SCENE3_... --action-noise 0.05` を実行中
（CPU 約112%、開始から1時間超）だった。**Tier 1 の実装はこのファイルへの変更を
要らない**（2 節）ため、衝突は起きていない。

`nvidia-smi` では GPU（H100・20GB）は使用中 0MiB・0% で空いていた——**Tier0 の
描画は CPU（`osmesa`）で行われる**（[09 §4](09_demo_perturbation_pipeline.md#4-うまくいったかどうかの判断)
に既述の理由）ため、GPU が空いていても CPU は Tier 0 の record ジョブに
占有されている。**Tier 1 の作り直し（4 節のコマンド）を今すぐ並行実行すると、
同じ CPU 描画ジョブとして競合する。** Tier 0 の完了を待つか、ユーザーが実行する
直前に `docker ps -a` を取り直して空きを確認すること。

---

## 2. 作り直し（本格対応）——コード変更ゼロで動く

**variant** = 元のタスク 1 つに摂動を 1 つ当てたもの（`.bddl` 1 個）。初期配置は元タスクのものを共有するので、**1 つの variant から最大 50 エピソード作れる。**

**`scripted_demo.py --variant` に LIBERO-plus の公式 Camera Viewpoints variant 名を渡すだけで、
scripted policy を1行も変えずに作り直せる。** 変更が要らない理由と、実際に確認した結果を記す。

### 2.1 なぜコード変更が要らないか

`env_wrapper.py` の `ControlEnv.__init__` は、渡された `bddl_file_name` 文字列そのものに
`_view_` と `_initstate_` の両方が含まれていれば、自分でそれを解析してカメラ設定
（方位角・仰角相当・`scale_factor`・カメラ自体の向き）を取り出し、**元の `.bddl` パスへ
戻してから読む**（`bddl_file_name.split("_view_")` した前半に `.bddl` を足す）。
`scripted_demo.py` の `_bddl_path()` は `variant` をそのままファイル名に使うので、
`--variant <base>_view_<h>_<v>_<scale*100>_<rot>_<vert>_initstate_0` を渡せば、
**この解析がそのまま働く。** `--variant` 自体は既存の CLI 引数で、
Light Conditions の variant（`..._light_2`）で既に使われている仕組みと同じ経路である。

**scripted policy の物理・成功判定には影響しない。** カメラ設定は描画（`_setup_camera`）にしか
関わらず、scripted policy は実測の手先・物体位置から行動を計算する閉ループ制御なので、
**同じ初期状態番号なら Base で作ったものと全く同じ軌道・同じ成功／失敗になり、画素だけが変わる。**
つまり作り直しには σ スイープのような成功率確認は要らない。

### 2.2 実際に確認した結果（CPU のみ、`parc2026_verify` コンテナで実行）

`KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` の Base と、
Camera Viewpoints variant の1つ（`_view_38_15_100_0_0_initstate_0`、方位角相当38・
仰角相当15・scale_factor=1.00）を、同じ初期状態（`pruned_init` の0番目）から
`ControlEnv` で構築・reset・1ステップ実行し、`agentview_image`（256×256×3）を比較した。

| | 平均輝度 | 標準偏差 |
|---|---|---|
| Base | 126.5 | 69.8 |
| Camera Viewpoints variant | 150.9 | 54.2 |

**画素が明確に変わっており、variant が実際に効いていることを確認した。** `docker run` は
使わず、既に起動していた検証用コンテナへ `docker exec` した（この環境で拒否されるのは
`docker run` だけで、既存コンテナへの `docker exec` は拒否されない。
[09 §4](09_demo_perturbation_pipeline.md#4-うまくいったかどうかの判断)で確認済みの制約と同じ）。

### 2.3 作り直す variant の一覧

**評価スイート（`compe/val/TRACK2_PROBE.csv`）と重ならない variant を選ぶ**必要がある
（[08_training_data_creation_cautions.md](08_training_data_creation_cautions.md) 2.8 節の
分離原則）。新規に書いた `solution/tools/probe/make_tier1_camera_variants.py` が、
LIBERO-plus の分類表（`task_classification.json`）から scripted policy のある2タスクの
Camera Viewpoints variant を集め、`TRACK2_PROBE.csv` に載っている variant（`KITCHEN_SCENE3` の
1件のみ。`KITCHEN_SCENE8` はこの疑似検証スイートに1件も入っていない）を除外し、
難易度・`id` の昇順で等間隔に選ぶ。

```bash
python3 solution/tools/probe/make_tier1_camera_variants.py \
    --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --task KITCHEN_SCENE8_put_both_moka_pots_on_the_stove \
    --exclude compe/val/TRACK2_PROBE.csv \
    --count 6 --output work/tier1_camera/CAMERA_VARIANTS.csv
```

実行済み。`KITCHEN_SCENE3`（分類表47件・除外後46件から6件）・`KITCHEN_SCENE8`
（分類表43件・除外0件から6件）を選び、`work/tier1_camera/CAMERA_VARIANTS.csv`
（12行）に書き出した。**`KITCHEN_SCENE8` の Camera Viewpoints variant は難易度4〜5しか
分類表に無い**（1〜3が存在しない）——弱い摂動を選べないという制約であり、
「わざと強い variant だけを選んだ」わけではない。

### 2.4 作り直し本体（ユーザーが実行）

12件それぞれについて、既存の Tier 0 の record と同じ手順でエピソードを作る（1本100秒・CPU描画）。
1タスク variant 6 件×1本のミニマム構成なら12本・20分ほど。本数を増やす場合は
`--episodes` で調整する（`scripted_demo.py` の既存引数）。

```bash
MUJOCO_GL=osmesa python3 solution/tools/scripted_demo/scripted_demo.py record \
    --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --variant KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_38_15_100_0_0_initstate_0 \
    --output outputs/scripted/tier1_camera
# ...（work/tier1_camera/CAMERA_VARIANTS.csv の残り11行も同様。
#     スクリプト実行時に全12行分のコマンドを標準出力に書き出す）
```

**実行前に `docker ps -a`・`nvidia-smi` を取り直すこと**（1節）。Tier 0 の record
ジョブ（`awesome_lehmann`）がまだ動いていれば、同じ CPU 描画を奪い合う。

---

## 3. 2D 近似（安価な並行策）——`train_pi05.py` に実装済み

**LeRobot の `image_transforms` を有効化し、既定6種に crop+zoom を1種足し、
affine の回転・並行移動・拡大縮小を広げる形で実装した。** `--camera-viewpoint-augment`
フラグを付けるだけで有効になる。3次元の視点回転（方位角・仰角）は2D変換では
再現できないので、あくまで安価な近似である。

### 3.1 draccus の制約——鍵ごとの上書きは効かない（訂正）

[02_appearance_and_data.md](02_appearance_and_data.md) は「`--dataset.image_transforms.tfs.*`
で追加する」という書き方をしていたが、**実際には draccus 0.8.0 はこの形の CLI 引数を
受け付けない。** `quant_train` コンテナ（`docker exec`、GPU不要）で
`draccus.parse(ImageTransformsConfig, args=["--tfs.affine.kwargs", "..."])` を試すと
`unrecognized arguments` で落ちる。`tfs` は `dict[str, ImageTransformConfig]` という
1つのフィールドで、**辞書全体を1つの JSON 文字列として渡すしかない**（鍵を1つだけ
追加・上書きすることはできず、既定6種を含めた全体を書き直す必要がある）。

### 3.2 実装した変換

`train_pi05.py` の `_CAMERA_VIEWPOINT_AUGMENT_TFS` に、既定6種（brightness・contrast・
saturation・hue・sharpness・affine）をそのまま含めた上で、次の2点を変えた辞書全体を
持たせ、`--camera-viewpoint-augment` を付けたときだけ
`--dataset.image_transforms.enable=true` と併せて渡す。

| 変換 | 既定 | 変更後 | 根拠 |
|---|---|---|---|
| `affine`（`RandomAffine`） | `degrees=(-5,5)`・`translate=(0.05,0.05)`・`scale` 無し | `degrees=(-15,15)`・`translate=(0.12,0.12)`・`scale=(0.82,1.22)` | 回転・並行移動を約3倍に広げ、`scale` を新規に足してズームを近似する。LIBERO-plus 公式の Camera Viewpoints variant は `scale_factor` が 1.00〜2.00 の範囲で使われている（`task_classification.json` を実際に数えて確認）が、**2D近似はあくまで近似**なので、その低め側（±22%）に留めた |
| `camera_crop`（新規、`RandomResizedCrop`） | （無し） | `size=[256,256]`・`scale=(0.75,1.0)`・`ratio=(0.9,1.111)` | ランダムクロップ＋ズーム。`lerobot/libero_plus` の画像 shape（256×256×3。`meta/info.json` で確認）に合わせ、出力サイズは変えない |
| `max_num_transforms` | 3 | 4 | 6種→8種に増やしたぶん、1種あたりの出現率が下がる（[02_appearance_and_data.md](02_appearance_and_data.md)が既に指摘していた注意点）。3のままだと新設の2種が選ばれる機会が薄まる |

### 3.3 CPU だけで確認した動作

`quant_train` コンテナ（`docker exec`、`sleep infinity` で既に起動済み。GPU不要）で、
`train_pi05.py` が組み立てる JSON をそのまま `draccus.parse(ImageTransformsConfig, ...)`
に通し、`ImageTransforms(cfg)` を実際にダミー画像 `torch.rand(3,256,256)` へ適用した。

| 確認項目 | 結果 |
|---|---|
| draccus が JSON を解析できるか | できた（8種すべてが `ImageTransformConfig` として復元された） |
| `ImageTransforms(cfg)` の forward が例外なく通るか | 通った（`torch.Size([3, 256, 256])`、値は `[0, 1]` の範囲内） |
| `train_pi05.py` の `py_compile` | 通った |

**この確認は GPU も `docker run` も使っていない。** `quant_train` は量子化スイープ用に
既に起動していたコンテナへ `docker exec` した。

### 3.4 使い方・既知の限界

```bash
python3 solution/train/train_pi05.py --run-name pi05_tier1_camera_aug \
    --steps <ステップ数> --batch-size <サイズ> --camera-viewpoint-augment
```

**限界が2つある。**

1. **3次元の視点回転（方位角・仰角）は再現できない。** 2D変換は光軸沿いの並進・回転・
   拡大縮小しか表現できない（[07_track2_preparation.md §2.1.1](07_track2_preparation.md#211-tier-0新設-ノイズを乗せてずれから戻るデータを作る)
   で既に指摘済みの限界）。
2. **手首カメラ（`observation.images.wrist`）にも同じ変換が独立にかかる。**
   `dataset_reader.py` の `_image_transforms` はカメラ列ごとに `for cam in camera_keys:`
   で回すだけで、列を選ぶ仕組みが無い。実際のカメラ視点摂動は俯瞰カメラ
   （`agentview`/`front`）だけに効き、手先に固定された手首カメラには効かない
   （[10_eval_perturbation_suite.md §5.2](10_eval_perturbation_suite.md#52-結果)で
   実測済み：ロボット初期姿勢を変えても俯瞰図の画素差分は0.0だった、という関係と対称）。
   **手首カメラだけ対象から外すには `dataset_reader.py` 側の変更が要り、
   「ほぼゼロコスト」の範囲を超えるため、今回は見送った。**

---

## 4. 比較の手順（学習後に実行する）

**成功率だけでは判断しない。** [09 §3.1](09_demo_perturbation_pipeline.md#31-track2疑似検証スイートの結果2026-08-22-実行08-22-に読み方を訂正)の
訂正どおり、本番と繋がっている量は「基準で完走していた件だけのステップ数」である。

1. `--camera-viewpoint-augment`（3節）と、2節で作り直したデータを混ぜた学習を
   それぞれ別の run として行い、**両方を同じ `compe/val/TRACK2_PROBE.csv` で評価する**
   （`run_docker_eval.sh`、`EVAL_CSV=compe/val/TRACK2_PROBE.csv`）。
2. 各 run の出力ディレクトリに対して
   `python3 solution/tools/probe/summarize_probe_by_category.py <run_dir>` を実行し、
   Camera Viewpoints 行の「基準完走件のステップ比」を、追加学習前
   （[09 §3.1](09_demo_perturbation_pipeline.md#31-track2疑似検証スイートの結果2026-08-22-実行08-22-に読み方を訂正)の
   +34.6%）と比較する。
3. 2D近似（3節）・作り直し（2節）・両方混合、の3通りを切り分けたい場合は、
   `train_pi05.py --dataset-root` で作り直しデータだけを混ぜた run と、
   `--camera-viewpoint-augment` だけを付けた run を別々に用意し、同じ手順で比較する。

---

## 5. 確認したい仮定

- **`--dataset.image_transforms` の値を `--resume` した2段目に持ち越せるか**は未確認。
  `train_pi05.py` の既存コードは `--resume` 時に `image_transforms` 系の引数を
  明示的に渡していない（前回の `train_config.json` を読み直す設計。optimizer/scheduler と
  同じ扱い）。前提が外れる場合、2段目の学習で `--camera-viewpoint-augment` が
  黙って無効に戻っていないかを確認する必要がある。
- **crop+zoom（`RandomResizedCrop`）とアフィンの `scale` を両方入れたことで、
  1フレームに掛かる幾何変換の強さが想定より強くなっていないか**は、実際に
  学習してから比較評価（4節）で確認する以外に確かめる方法が無い。
