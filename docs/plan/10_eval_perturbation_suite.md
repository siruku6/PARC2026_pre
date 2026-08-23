# 10. 評価データに摂動を加える計画（単一摂動は現状維持、複合摂動は技術検証と初期状態解決の修正まで実施）

**この文書の位置づけ**: [09_demo_perturbation_pipeline.md](09_demo_perturbation_pipeline.md)は学習データ側の計画である。
本戦（8/24〜）に向けて、評価データ側でも「各摂動」「複数摂動を組み合わせたもの」を用意したいという
依頼を受け、実際に何が既にあり何が無いかを一次資料で確認した上で、ユーザーに選んでもらった範囲を記録する。

---

## 1. 評価データは「動画像を作る」作業ではない

09の学習データ計画は、摂動を加えた動画像を撮影・生成し `lerobot` データセットへ混ぜる作業だった。
評価側はこれと種類が違う。**カメラ視点・ロボット初期姿勢・物体配置・センサノイズの4軸とも、
シミュレータが評価の実行時にその場でレンダリングする**（`work/LIBERO-plus/libero/libero/envs/env_wrapper.py`・
`libero_tabletop_manipulation.py`で確認）。評価側で用意すべきものは動画像ファイルではなく、
**「どの task_id・どの`.bddl`・どの`.pruned_init`を使うか」という組み合わせの指定**である。
これが評価データを検討する上での前提になる。

---

## 2. 今すでにあるものと、無いもの

| 観点 | 現状 |
|---|---|
| **単一摂動の評価データ（a）** | ある。`compe/val/TRACK2_PROBE.csv`（8タスク×[Base+4摂動]=40行）。生成スクリプト`solution/tools/probe/make_track2_probe.py`は各カテゴリで難易度1〜5の中央値を1件だけ選ぶ。スクリプト自身のdocstringが「1セル1件しか測らない順位付けツールで、本評価ではない」と明記している |
| **複数摂動を組み合わせた評価データ（b）** | 無い。`work/LIBERO-plus`の公式タスク一覧（`task_classification.json`、7カテゴリ計10,030件）を数えたところ、**カテゴリを跨いで摂動を組み合わせたタスクは1件も無い**。LIBERO-plus論文自体は「組み合わせ摂動には単独の積を下回る相互作用効果がある（Negative Compositional Generalization）」と報告しているが、そのための生成コード・`.bddl`・分類表はこのリポジトリに同梱されていない |
| **技術的な可能性（検証済み・4節参照）** | `env_wrapper.py`の`ControlEnv.__init__`は、`<base>_view_..._initstate_..._noise_...`という1つのtask_id名から、カメラ視点・ロボット初期姿勢・センサノイズの3要素を独立にパースできる構造になっている。**この3軸の組み合わせ、およびそこにObjects Layoutを加えた4軸の組み合わせは、実際に動作することを確認した**（4節）。Objects Layoutを混ぜた場合に評価ハーネス側の初期状態ファイル解決が見つけていなかった箇所は、原因を特定して修正済み（4.2節） |
| **評価データの分離原則** | [08_training_data_creation_cautions.md](08_training_data_creation_cautions.md)の「学習で見せたカメラ角度・初期姿勢・物体配置の集合と、検証で使う集合を分離しないと汎化を測れない」（2.8節）は評価データにも直接効く。09で決まるTier 0/1/2の学習側variant IDと、評価側で使うvariant IDが重ならないようにする仕組み（`solution/tools/make_validation_suite.py --exclude`）は既にある |

---

## 3. 今回決めた範囲

| 論点 | 決定 |
|---|---|
| **単一摂動側（TRACK2_PROBE.csv）** | **現状維持**。40行・約80分のまま拡張しない |
| **複合摂動側** | **技術検証のみ並行して行う**。本格的な評価データ生成は行わない |

複合摂動側を「検証のみ」に留めた理由は、07で「Track2の位置・視点・ノイズがLIBERO-plusの
4カテゴリに対応する」という対応関係自体がまだ**推測**であり（07 §1.1）、本戦の実際の内容は
8/24のQ&Aで初めて確認できるためである。本戦が単軸の摂動しか課さない場合、複合摂動の評価
データを今から本格的に作る投資は空振りになる。技術検証だけを先に済ませておけば、
Q&A後に「作るべき」と判断が変わったときに前提の確認からやり直さずに着手できる。

**技術検証の結果、Objects Layoutを含む複合が動かない原因が具体的に分かり、
直す箇所が1関数に限られると判明したため、その修正まではこの場で実施した**
（4.2節）。「本格的な評価データ生成は行わない」という決定自体は変わっていない
——直したのは複合摂動の行を作れるようにする下地であり、複合摂動の行を
実際に量産するCSV生成スクリプトはまだ書いていない。

---

## 4. 技術検証の内容と結果（実行済み）

**確かめたこと**は、`env_wrapper.py`のファイル名パーサが本当に複数軸を同時に受理し、
かつ**それぞれの軸が実際に効いているか**の2段である（09の4節がTier 0の実装確認に
使った「a. 実装が正しいか」の考え方と同じ）。GPUを使う量子化スイープの評価回収
（`work/quant_sweep/`、朝までかかる見込み）と資源を取り合わないよう、方策モデルの
推論を挟まない・シミュレータのみのCPU確認として設計し、ユーザーに軽量コンテナ
（`parc2026_verify`、GPUなし・`--network none`）を新たに起動してもらって実行した。

### 4.1 結果1: Camera + Robot Initial States + Sensor Noise の3軸同時指定は動く

`KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`をベースに、
`_view_0_0_121_0_0_initstate_3_noise_1`（カメラscale=1.21・ロボット初期姿勢クラス3・
センサノイズ1）という1つのtask_idで`ControlEnv`を構築し、Baseと直接比較した。

| 確認項目 | 結果 |
|---|---|
| env構築がエラーなく通るか | **通った**（`noise=1`が正しく設定された） |
| ロボットの初期姿勢クラス（`Panda3`）が指定どおり効いているか | **効いている**。reset直後のqposがBaseと異なる |
| カメラ・ノイズが指定どおり画像に効いているか | **効いている**。`agentview_image`がBaseと異なる（画素の平均・標準偏差もBaseから変化） |

**Camera・Robot Initial States・Sensor Noiseの3軸は、公式タスク一覧に無い組み合わせでも
1つのtask_idに同時に載せられ、実際に3軸とも効くことを確認した。**

### 4.2 結果2: Objects Layoutを混ぜると、評価ハーネスが使う初期状態ファイル解決に見落としがあった（特定・修正済み）

同じtask_idの先頭に`_add_14`（Objects Layoutの既存variant）を挟んだ
`..._add_14_view_0_0_121_0_0_initstate_3_noise_1`も試した。

**まず切り分けが要った。** 「初期状態ファイルを解決する処理」は、このリポジトリに
2か所ある。

| 場所 | 何をするか | このtask_idでの結果 |
|---|---|---|
| `work/LIBERO-plus/libero/libero/benchmark/__init__.py`の`get_task_init_states`（LIBERO-plus本体） | 公式10,030タスク一覧を単体で読むときに使う | `_view_`より前を切り出した`..._add_14.pruned_init`を**標準の`init_files/libero_10/`配下**で探すため見つからない（バグ）。**しかしこの関数は本戦の評価パイプラインからは呼ばれていない**（`benchmark_scripts/`の単体スクリプト専用。`grep`で使用箇所を確認済み） |
| `compe/t1/register.py`の`get_task_init_states`（このリポジトリの評価パイプラインが実際に使うもの） | `solution/tools/patch_kit_for_variants.py`が配布キットに当てるパッチにより、CSVの`init_states_file`列の値をそのまま使う設計に**すでに直っている**（`/`を含む値なら`libero_newobj`配下も読み、reshapeする） | **こちらは元から正しく動く。** 問題はCSVの`init_states_file`列の値を計算する側にあった |

**実際のバグは評価パイプライン自体ではなく、検証スイートCSVを作る
`solution/tools/make_validation_suite.py`の`_resolve_newobj_init`にあった。**
この関数は`_add_14`のような単独のObjects Layout変種名なら`task_id`とファイル名が
完全一致するので正しく動くが、`..._add_14_view_..._noise_1`のような複合名では
「`task_id`と完全一致するファイル」を探すため、`_add_14`用のファイルを見つけられない
（bddlの探し方（`_resolve_bddl`）が`_view_`より前でも探すのに対し、初期状態の探し方は
完全一致しか試していなかった、という非対称がそのまま見落としの原因だった）。

**修正**: `_resolve_newobj_init`が`task_id`自身に加えて`_view_`より前の名前でも探すように
した（`_resolve_bddl`と同じ考え方）。既存の単独Objects Layout名は`_view_`を含まないため
挙動は変わらない（後方互換）。`solution/tools/make_validation_suite.py`を編集し、
`py_compile`で構文を確認した。

**修正後、`parc2026_verify`コンテナで実際のパイプラインと同じ手順を最初から最後まで
再現して確認した。**

| 確認項目 | 結果 |
|---|---|
| `_resolve_newobj_init`が単独`_add_14`名を正しく解決するか（後方互換） | **一致**。修正前と同じ結果 |
| `_resolve_newobj_init`が複合4軸名を正しく解決するか | **解決した**。`libero_newobj/libero_10/..._add_14.pruned_init`を返す |
| `_place_variant`（CSV行を組み立てる本体）が複合4軸名でbddlと初期状態を両方そろえられるか | **そろった** |
| `register.py`が使う形式（`<init_states_root>/<task_id>.bddl`）でbddlパスを組み、`ControlEnv`が構築できるか | **できた**（`noise=1`） |
| 解決した初期状態ファイルを`torch.load`→`reshape(1,-1)`→`env.sim.set_state_from_flattened()`まで流せるか | **流せた**。状態ベクトルの長さ（86）が`1+nq+nv`と一致し、エラーなく適用できた |
| 適用後に`env.step()`を実行できるか（ノイズ込みの観測画像が返るか） | **できた**。`agentview_image`が(128,128,3)で返った |

**Objects Layoutを含む4軸複合の評価データは、この修正により技術的な障害なく作れることを
end-to-endで確認した。**

### 4.3 この結果が複合摂動の評価データ計画に持つ意味

- **Camera×Robot×Noiseの複合摂動データは、追加の実装なしにすぐ作れる。** task_id文字列を
  組むだけで、既存の`.bddl`・`.pruned_init`を流用できる。
- **Objects Layoutを含む複合摂動データも、`make_validation_suite.py`の修正により
  追加実装なしで作れるようになった。** 修正はこの1関数（`_resolve_newobj_init`）に
  限られ、既存の使い方（単独のObjects Layout変種）には影響しない。
- **ここで確認したのは「技術的に作れる」ことだけである。** 実際に複合摂動の行を
  量産する新しいCSV生成スクリプトは、まだ書いていない（5節）。

---

## 5. 実機ロールアウトでの確認（Robot Initial States単軸、実行済み）

4節は「シミュレータの環境構築が通るか」までの確認だった。**実際に学習済み方策
（`work/artifacts/pi05_rp3000`）を走らせても、本当にロボットの初期姿勢だけが変わった
問題設定になっているか**を、動画と画素差分で確認した。

### 5.1 作ったもの・実行したコマンド

- `work/robot_init_sanity/ROBOT_INIT_SANITY.csv`: `TRACK2_PROBE.csv`の実在する2行
  （`KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`のBase行と、
  `..._view_0_0_100_0_0_initstate_3`行）をそのまま抜いた2行だけのCSV。
- `solution/tools/run_docker_eval.sh`に`PARC_RECORD_WHICH`の転送を追加。**既定では
  失敗した回しか録画しない**ため、2件とも成功すると動画が1本も残らない問題を防ぐ
  （`PARC_RECORD_WHICH=all`を渡せば両方録画される）。既存の呼び出しは既定値
  `failures`が保たれるため後方互換。

```bash
WEIGHTS="$PWD/work/artifacts/pi05_rp3000" \
  EVAL_CSV="$PWD/work/robot_init_sanity/ROBOT_INIT_SANITY.csv" \
  LABEL=robot_init_sanity N_EPISODES=1 MAX_STEPS=300 \
  RECORD=1 PARC_RECORD_WHICH=all \
  POLICY_IMAGE=parc2026-train-pi05 \
  bash solution/tools/run_docker_eval.sh
```

出力先: `work/gpu_eval/0822_070934_robot_init_sanity_n5_ep1_max300/`。

### 5.2 結果

両エピソードとも成功した（`eval.log`より）。

| | Base | Robot Initial States（Panda3・L2） |
|---|---|---|
| 結果 | 成功 235 steps | 成功 241 steps |
| 経路長 | 1.44m | 1.45m |
| 変位 | 0.27m | 0.27m |
| 握り回数 | ×3 | ×4 |

録画した2本のmp4の**1フレーム目**を、カメラ区画ごとに画素差分（絶対値、0〜255階調）で
比較した（`parc2026_verify`コンテナ内でcv2を使用）。

| 区画 | 平均絶対差分 | 差分>10の画素比率 | 読み方 |
|---|---|---|---|
| 俯瞰図（物体配置） | **0.0** | **0.0%** | **完全一致。物体配置は文字通り変わっていない** |
| 前方固定カメラ（front） | 5.8 | 9.3% | 変化はロボット本体が写る領域だけ。カメラの視点・スケールは同一 |
| 手首カメラ（wrist） | 38.8 | 52.6% | ロボットの関節姿勢が変わったので、手先に固定されたカメラの視点自体が動く（想定どおりの副作用） |

**「ロボットの初期姿勢だけが変わった問題設定」が実際に作れていることを、数値・静止画・
動画の3段で確認した。** 俯瞰図の差分が完全に0であることが、物体配置に手を付けていない
という主張の直接的な証拠になっている。

### 5.3 比較ページ（Artifact）と既知の問題

両動画・拡大静止画・上表を1ページにまとめ、Artifactとして公開した
（<https://claude.ai/code/artifact/868a88c4-92c1-44dd-b522-de05d2e02b57>）。
**ユーザー環境でページ内の動画が再生できないという報告があり、未解決。** 現在は
mp4を`data:`URI（base64、2本で約7.8MB）としてページに直接埋め込む方式にしている。
Artifactの実行環境が推奨する動画の扱い方は`data:`埋め込みではなく専用のアセット機能
（`action: "upload_asset"`）であるため、次に同種の比較ページを作るときはそちらを
優先し、再生できるかを検証してから展開する。

### 5.4 途中で起きたこと（GPU競合の記録）

このロールアウトの実行中、2回のOOM（メモリ不足）に遭遇した。

1. 1回目: 量子化スイープの評価回収ジョブ（`work/quant_sweep/`）が実行中と誤って
   並行実行し、GPUメモリを取り合って失敗。回収完了を確認してから再試行した。
2. 2回目: 回収完了の確認とユーザーの実行の間に、別の検証作業
   （`pi05_c3_compile_smoke`、C3のtorch.compileスモークテスト）が新たにGPUを
   使い始めており、再度OOMになった。ユーザー側でそのプロセスを停止してから
   再実行し、成功した。

**教訓**: このマシンのGPUは複数の作業（量子化スイープ・その回収・compileの
スモークテスト・推論速度ベンチマーク等）が入り乱れて使われている。**「空いていた」
という直前の確認は、実行の直前にもう一度取り直さないと保証にならない。**

---

## 6. まだ決まっていないこと

- **4軸（Camera×Robot×Noise×Objects Layout）の複合摂動評価データを本格的に作るかどうか。**
  技術的な障害は無く（4節）、`make_validation_suite.py`の修正で下地も直っているが、
  本戦が実際に複合摂動を課すかは8/24のQ&A待ちのまま。作るとなった場合、複合の行を
  組み立てて`compe/val/`向けのCSVに書き出す生成スクリプト自体はまだ無いので、
  新規に書く作業が残る。5節の実機確認と同じやり方（少数タスク×実ロールアウト×
  画素差分）を、4軸複合でも1件試すと、着手前の最後の裏取りになる。
- 単一摂動側（TRACK2_PROBE.csv）を厚くするかどうかは、今回「現状維持」と決めたが、
  Q&Aで本戦の摂動軸が確定した後に見直す余地は残す。
- 検証用に立てた`parc2026_verify`コンテナ（GPUなし）は、この検証以外に使わないなら
  停止・削除してよい。
- ~~**`compe/val/TRACK2_PROBE.csv`（40行）自体のGPU評価が、まだ実行されていない。**~~
  **【8/22 に実行済み】** 結果と、そこから続いた5本の追加実測は7節にまとめた。

---

## 7. 強度依存の実測と、読み方の訂正（2026-08-23 に確定）

**5本の評価（合計 184 ロールアウト）を回して分かったことと、決めたことをここに集める。**
このうち最も影響が大きいのは**「今までの読み方が間違っていた」**という項目なので、
他のセッションが同じ数字を引くときは 7.1 の (1) を先に読むこと。

| 回 | 行数 | 目的 |
|---|---|---|
| `0822_092556_track2_probe` | 40 | 4カテゴリを1枚で測る |
| `0822_120617_sensor_noise_high` | 32 | Sensor Noise の severity 25〜50 |
| `0823_092324_robot_init_high` | 16 | Robot Initial States の難易度5 |
| `0823_141340_camera_azimuth` | 48 | Camera 方位角 15〜75° |
| `0823_190621_camera_azimuth_blend4` | 48 | 同じ48行を `blend_steps=4` で |
| （未実行） | 32 | 乱数の床を測る対照 → 7.3 の項目1 |

### 7.1 判明事項

**(1) ステップ比は 300 ステップの打ち切りに支配されていて、強度の指標として読めない。**
摂動側が打ち切られた組では、比が `300 ÷ Base の歩数 − 1` になり、**方策の悪さではなく
「Base がどれだけ短かったか」で決まる。** 方位角45°の3組は Base 223→300（+35%）・
125→300（+140%）・82→300（+266%）で、この3つが +51.1% の主因だった。打ち切られた組を
外すと **+51.1% → −11.8%** と符号が反転する。**同じ理由と対処は
`solution/tools/compare_steps.py:95-118` に既にあった**（全体と「両方成功した件だけ」の
両方を出す）。probe 側の道具にだけ入っていなかった。

**(2) 打ち切られない量「300 に達した割合」で読み直すと、4カテゴリのうち差が出たのは
Camera 方位角の45°以上だけである。**

| 測ったもの | 300 に達した割合 | Base との差（Fisher 両側） |
|---|---|---|
| TRACK2_PROBE の Camera Viewpoints（拡大率・レンズ向き） | 0/8 | p = 1.000 |
| 同 Objects Layout | 1/8 | p = 1.000 |
| 同 Robot Initial States | 1/8 | p = 1.000 |
| 同 Sensor Noise | 2/8 | p = 1.000 |
| （同 Base） | 1/8 | — |
| Sensor Noise の severity 25 / 35 / 45 / 50 | 0/8・1/8・1/8・1/8 | 差なし |
| Robot Initial States の難易度5 | 1/8（Base 1/8） | 差なし |
| **Camera 方位角 45〜75° をまとめて** | **13/24** | **0〜30° の 3/24 に対し p = 0.005** |

**元の probe が出したカテゴリ順位（+34.6% / +24.4% / +17.5% / +15.6%）は、打ち切られない
量で読むと1つも差が無い。** Camera の +34.6% は 0/8 で、Base の 1/8 より良い。

**(3) 方位角は1点ずつでは有意にならない。まとめて初めて言える。**
Base 2/8 対 45° 5/8 は p = 0.294。1点ずつで有意にするには**1点あたり18タスク**必要
（今は8）。**「45°以上で崩れる」は言えるが、「30°から崩れ始める」は言えない**——30° の
時点では 300 到達が 1/8（Base の 2/8 より良い）、タスク別ステップ比の中央値も +5.1%。

**(4) 推論は乱数を使い、seed を固定していない。**`policy_server.py` は
`_get_action_chunk` に `noise=None` を渡し、モデル側が毎回引く。**そして手元の評価 49 回
すべてが1タスク1エピソードで、同じ条件を回し直した差を一度も測っていない。**

**(5) `blend_steps=4` は効かなかった。** [knowledge/06](../knowledge/06_step_count_drives_score.md)
が「打ち切り率が高かったときの再検討材料として置く」と条件を明示していた項目で、方位角
45〜75° の 300 到達率が 54% になったことで条件が満たされたため、同じ48行を `blend0` と
`blend4` で回して比べた。

| | blend0 | blend4 |
|---|---|---|
| 45〜75° の 300 到達 | 13/24 | **10/24** |
| 0〜30° の 300 到達 | 3/24 | 4/24 |
| 成功した回 | 31/48 | **31/48（同数）** |
| 衝突した回 | 7/48 | 6/48 |
| 対応のある反転（45〜75°） | — | **救われた 6 件 / 悪化した 3 件、McNemar p = 0.508** |
| 両方が完走した行のステップ数（0〜30°） | 2,664 歩 | 2,686 歩（**+0.8%**） |

**(6) 1エピソードでの設定比較は、境界での転びに埋もれる。** 45〜75° の24行のうち
**9行が向きを問わず反転した**（救われた6・悪化した3）。300歩の境界にいるタスクは、行動列が
わずかに変わればどちらへも転ぶ。**設定の効果を見るには、まず反転の床を測る必要がある**
（7.3 の項目1）。

**(7) Track1 で見えた blend4 の「減速の代償」は、今の重みでは小さい。**
knowledge/06 は手元36件で「基準で300未満だった30件が +13.1% 遅くなる」と記録しているが、
今回の 0〜30° の19行では **+0.8%** だった。重みが `pi05_rp3000` に変わっているため。
**代償が小さいかわりに、救う効果も測れなかった。**

**(8) 交絡していた読みが1つあった。** 「邪魔物体が 1〜2 個で衝突 0.00、5個以上で 0.42」を
物体数の効果と読みかけたが、バケットの中身は 1〜2個が2タスク・5個以上が2タスク（どちらも
「かごに入れる」課題）で、**物体数とタスクが完全に重なっていた。** 物体数の効果は、同じ
タスクの中で物体数を変えないと分からない。

**(9) 摂動の強さを決める整数の幅・方位角が二山であること・一覧に無い組み合わせでも評価が
動くこと**は [reference/20 の 2.2・2.3 節](../reference/20_track2_perturbation_categories.md)
に書いた。**`difficulty_level` は強度の段ではない**（Camera の方位角の中央値が難易度1〜5 で
9.0 / 12.5 / 23.0 / **11.5** / 41.0）。

### 7.2 決定事項

| 決めたこと | 理由 |
|---|---|
| **主に読む量は「300 に達した割合」にする。** ステップ比は「打ち切りを外した値」と併記し、単独では使わない | 7.1 (1)。本番の上限も 300（`compare_steps.py:21`） |
| **`blend_steps` は 0 のまま。この件は閉じる** | 7.1 (5)。事前に置いた再検討条件は満たされたが、効果が測れなかった。過去に本番で 0.264 → 0.169 に落とした実績もある |
| **Tier1（Camera 対策）に投資する根拠は残す。撮り直しは方位角45°以上を狙い、15°は入れない** | 7.1 (2)(3)。4カテゴリで唯一、強度を上げて悪化したのが Camera |
| **Camera の拡大率は測らない** | TRACK2_PROBE の拡大率 121〜193% の8行が 0/8 で崩れていない。上限は 200% |
| **Camera の方位角の負側・仰角は測らない** | 撮り直しは物理を変えないので、測らずに左右両方・仰角両方を撮れば済む |
| **`PARC_SEED` を既定オフで実装した**（`policy_server.py`・`run_docker_eval.sh`） | 複数回で設定を比べるときに、対応が取れてばらつきが小さくなる。**ただし seed を揃えても blend のように行動を書き換える設定は軌道が変わるので、これだけでは 7.1 (6) は解けない** |

### 7.3 今後の方針

| # | やること | 何が決まるか | 妨げ |
|---|---|---|---|
| **1** | **`CAMERA_HIGH_CONTROL.csv`（8 Base + 45/60/75° の24行）を blend0 で2エピソード回す** | **反転の床。**これが無いと設定値の比較が一切できない。7.1 の (5)(6) の判定もここで確定する | なし。CSV は作成済み |
| **2** | Objects Layout の物体数スイープ（`LAYOUT_ADD_PROBE.csv`、48行） | **4カテゴリで唯一測っていない軸。**同じタスクの中で物体数を変えるので 7.1 (8) の交絡が無い | なし。CSV は作成済み |
| 3 | `summarize_probe_by_category.py` に「300 に達した割合」と「打ち切りを外した比」を足す | 今後の読み方。過去49回に遡って効く | なし（GPU 不要） |
| 4 | カテゴリあたりのサンプル数を増やす | 1点ずつの判定（7.1 (3) の18タスク） | **G の量子化評価高速化待ち** |
| 5 | 複合摂動の probe 生成スクリプト | 単軸の和で近似できるか | **8/24 の Q&A 待ち** |

---

## 出典

| 事実 | 出典 |
|---|---|
| 評価が動画像ファイルを不要としレンダリングが実行時に行われること | `work/LIBERO-plus/libero/libero/envs/env_wrapper.py`（`ControlEnv.step`/`reset`）・`libero_tabletop_manipulation.py`（`_setup_camera`） |
| TRACK2_PROBE.csv の行数・生成方法・「順位付けツール」という位置づけ | `compe/val/TRACK2_PROBE.csv`（実際に確認）・`solution/tools/probe/make_track2_probe.py`のdocstring |
| LIBERO-plus公式タスク一覧に複合摂動タスクが無いこと | `work/LIBERO-plus/libero/libero/benchmark/task_classification.json`（7カテゴリ計10,030件を実際に数えて確認） |
| ファイル名パーサが複数軸を同時に受理できる構造であること、および3軸（Camera・Robot・Noise）が実際に動くこと | `env_wrapper.py`の`ControlEnv.__init__`・`step`/`reset`。`parc2026_verify`コンテナで`ControlEnv`を直接構築し、Base条件とqpos・`agentview_image`を比較して確認 |
| Objects Layout（`_add_14`）を混ぜた場合、env構築自体は通ること | 同上コンテナで`..._add_14_view_0_0_121_0_0_initstate_3_noise_1`を構築し確認 |
| LIBERO-plus本体の`get_task_init_states`（`benchmark/__init__.py`）が複合task_idに対して実在しないパスを組み立てること、かつこの関数が本戦の評価パイプラインからは呼ばれていないこと | 同関数のコードを読み、標準の`init_files/libero_10/..._add_14.pruned_init`が存在せず`init_files/libero_newobj/libero_10/..._add_14.pruned_init`にのみ実在することを`os.path.exists`で確認。呼び出し元は`grep -rn`で`benchmark_scripts/check_task_suites.py`・`render_single_task.py`のみと確認 |
| 本戦の評価パイプラインが実際に使う`compe/t1/register.py`の`get_task_init_states`は、`solution/tools/patch_kit_for_variants.py`のパッチによりCSVの`init_states_file`列をそのまま使う設計に直っていること | `compe/t1/register.py`・`solution/tools/patch_kit_for_variants.py`を読んで確認 |
| 実際のバグが`solution/tools/make_validation_suite.py`の`_resolve_newobj_init`（完全一致でしか探さない）にあったこと、および修正後に単独名・複合4軸名の両方を正しく解決すること | 同ファイルを読み、修正前後の挙動を`_resolve_newobj_init`・`_place_variant`を直接呼んで比較。修正後は`py_compile`で構文確認済み |
| 修正後、bddlパス構築→`ControlEnv`構築→初期状態の`torch.load`/`reshape`/`set_state_from_flattened`→`env.step()`まで実際のパイプラインと同じ手順で通ること | `parc2026_verify`コンテナで一連の手順を再現し、状態ベクトル長86（`1+nq+nv`）が一致・エラー無しで確認 |
| LIBERO-plus論文の Negative Compositional Generalization の主張 | [07_track2_preparation.md](07_track2_preparation.md) §1.3.1（arXiv HTML再取得による2次情報） |
| 検証データの分離原則・`make_validation_suite.py --exclude` | [08_training_data_creation_cautions.md](08_training_data_creation_cautions.md) 2.8節・4.3節、`solution/tools/make_validation_suite.py` |
| `quant_train`にrobosuite/liberoが入っておらず、`parc2026`イメージには入っていること | 各コンテナへ`docker exec`で直接importを試して確認 |
| 検証当時、`parc2026`イメージのコンテナ（`suspicious_hopper`）は別のGPU評価ジョブを実行中で、その後量子化スイープの評価回収（朝までかかる見込み）に引き継がれたこと | `docker exec suspicious_hopper ps aux`・`work/gpu_eval/`のログ・`work/quant_sweep/PROGRESS.md`（2026-08-21 21:47の記載） |
| Track2の4カテゴリ対応が推測であり8/24のQ&Aで確認する計画であること | [07_track2_preparation.md](07_track2_preparation.md) §1.1 |
| Robot Initial States単軸の実機ロールアウト結果・画素差分の数値 | `work/gpu_eval/0822_070934_robot_init_sanity_n5_ep1_max300/eval.log`、動画2本を`parc2026_verify`コンテナでcv2により1フレーム目を区画ごとに`np.abs`差分（実際に計算） |
| `PARC_RECORD_WHICH`が既定で`run_docker_eval.sh`からkitコンテナへ転送されていなかったこと・修正内容 | `solution/tools/run_docker_eval.sh`のkit_environment配列を読んで確認し、実際に1行追加、`bash -n`で構文確認 |
| 実行中にOOMが2回発生した経緯（量子化スイープ回収との重複、`pi05_c3_compile_smoke`との重複） | 各時点で`nvidia-smi`・`docker ps -a`・`ps aux`・該当ログを実際に確認 |
| `compe/val/TRACK2_PROBE.csv`（40行）のGPU評価がまだ実行されていないこと | `work/gpu_eval/*/VAL_TASKS.csv`を全件確認（41行のものが無い）・`grep -rl TRACK2_PROBE work/gpu_eval/*/*.log`が0件 |
| 09の3節が想定していた並行実行が、実際には量子化スイープ関連の複数ジョブ（回収・compileスモークテスト・`work/train/_bench/bench_driver.py`による推論速度ベンチマーク）に占有され続けたこと | `work/quant_sweep/PROGRESS.md`・`docker exec quant_train ps aux`・`work/train/_bench/bench_driver.py`のdocstringを実際に確認 |
