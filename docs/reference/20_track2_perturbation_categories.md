# 20. Track2 で発生すると見込まれる摂動の種類

**Track2（本選、8/24〜）の「位置・視点・ノイズ」という文言が何を指すかの前提を、
1 か所にまとめた記録である。** これまで [07_track2_preparation.md](../plan/07_track2_preparation.md)
の §1.1・§1.2 に埋め込まれていた事実に、[08_training_data_creation_cautions.md](../plan/08_training_data_creation_cautions.md)・
[09_demo_perturbation_pipeline.md](../plan/09_demo_perturbation_pipeline.md)・自チームの
Track2 疑似検証スイート実測（2026-08-22）を合わせて、**単一の参照先**にした。
Track1 の 7 カテゴリ全体の事実は [05_perturbation_categories.md](05_perturbation_categories.md)
（本ファイルは Track2 に絞った続き）。

| 節 | 中身 |
|---|---|
| 1 | 対応関係は推測であり、確定は本選開始後の Q&A 待ち |
| 2 | 4 カテゴリそれぞれの中身・母集団・難易度分布 |
| 3 | 外部文献（LIBERO-plus 論文）が他モデルで報告する脆さ |
| 4 | 自チームの実測（Track2 疑似検証スイート、2026-08-22） |
| 5 | 実測と文献の食い違い（確認したい仮定） |

---

## 1. 対応関係は推測であり、確定は本選開始後の Q&A 待ち

**予選 PDF（p8）の Track2 は「同一タスク・未知ドメイン」＋「位置・視点・ノイズ変更」と
書くだけで、LIBERO-plus のカテゴリ名を 1 つも挙げていない。** Track1 の教訓（運営は
Track1 についてもカテゴリ名を挙げず、こちらが Q&A No.21 の原文から逆算する必要が
あった。[05_perturbation_categories.md](05_perturbation_categories.md)）を踏まえ、
以下の対応づけは**字面を LIBERO-plus の 7 カテゴリ名と突き合わせた強い推測**であり、
運営が確認した事実ではない。

| Track2 の文言 | 対応すると見立てた LIBERO-plus のカテゴリ |
|---|---|
| 視点 | Camera Viewpoints |
| ロボットの位置 | Robot Initial States |
| 物体の位置 | Objects Layout |
| ノイズ | Sensor Noise |

**本選開始後に配布される Track2 の説明・Q&A を最優先で確認し、この対応が正しいかを
裏取りする**ことが、本選 1 日目にやるべき最初のタスクとして残っている
（[07 §4](../plan/07_track2_preparation.md#4-わからないこと確認が必要なこと) の項目 1）。

---

## 2. 4 カテゴリそれぞれの中身・母集団・難易度分布

**LIBERO-plus 全 10,030 件のうち、この 4 カテゴリで 6,275 件（62.6%）を占める。**
`task_classification.json`（revision `4976dc3`）を実際に数えた値である。

| カテゴリ | 件数 | 割合 | 名前の接尾辞 |
|---|---|---|---|
| Sensor Noise | 1,601 | 16.0% | `_noise_N` |
| Camera Viewpoints | 1,599 | 15.9% | `_initstate_N`（他 2 カテゴリと同じ接尾辞。`category` 列でしか見分けられない） |
| Robot Initial States | 1,550 | 15.5% | `_initstate_N` |
| Objects Layout | 1,525 | 15.2% | `_add_N`（829）／`_levelN_sampleM`（696） |
| **4 カテゴリ合計** | **6,275** | **62.6%** | |

難易度 1〜5 の分布は 972 / 1,454 / 1,328 / 1,153 / 1,368 で、大きな偏りは無い。
suite 別も libero_goal 1,621・libero_object 1,619・libero_10 1,573・libero_spatial 1,462
とほぼ均等（[07 §1.1](../plan/07_track2_preparation.md#11-track2-の位置視点ノイズとは何か)）。

**各カテゴリの実体**は、`LIBERO-plus/libero/libero/envs/` 配下の実装コードを直接読んで
確認した一次情報である。

| カテゴリ | 何を変えるか | 具体的な数値（実装コードで確認） |
|---|---|---|
| **Camera Viewpoints** | `agentview` カメラをシーン中心を軸に回転・ズーム | 水平回転（方位角）・垂直回転（仰角）をかけ、距離（`scale_factor`）を変え、カメラ自体の向きも独立に回す。サンプル値は方位角 `[30,60,...,330]`、仰角 `[15,345]`、`scale_factor=1.25` 等（`libero_tabletop_manipulation.py:305-376`） |
| **Robot Initial States** | アーム初期関節角 `qpos` に固定長の摂動を加える | 基準姿勢 `[0, -0.161, 0, -2.445, 0, 2.227, π/4]` に、ランダムな単位ベクトル方向へ大きさ 0.1〜0.5（5 段階・各 100 個体）を加える（`robots/new_init.py`） |
| **Objects Layout** | 邪魔物体の追加（`_add_N`）／対象物体の位置ずらし（`_levelN_sampleM`） | 邪魔物体は既定の物体プールから追加。ずらし幅は難易度レベル 1〜5 に対応（cm 単位の正確な値は未確認） |
| **Sensor Noise** | `agentview_image` に事後的な画像劣化（ImageNet-C 系）をかける | 5 種×severity 1〜10 の合成：`motion_blur`(1-10)・`gaussian_blur`(11-20)・`zoom_blur`(21-30)・`fog`(31-40)・`glass_blur`(41-50)。Hendrycks & Dietterich の ImageNet-C 実装（`env_wrapper.py:26-174`） |

**Camera・Noise は「物理を変えない」（レンダリング設定・画像後処理）が、Robot・Layout は
物理そのものを変える。** 同じロールアウトを別のカメラ設定・画質で撮り直しても行動ラベル
（action）は変わらないが、初期姿勢や物体配置が変わると既存の行動列をそのまま再生しても
正しい観測・行動対にはならない。この違いが学習データの作り方（Tier 1 と Tier 2 の区別、
[09 §2](../plan/09_demo_perturbation_pipeline.md#2-シナリオ--摂動--程度)）に直結する。

### 2.1 Sensor Noise は「いつ・どこで」画像にかかるか

**MuJoCo が描いたきれいな画像を、Python 側で毎フレーム後処理して汚す。** シミュレータの
物理・レンダリング設定は一切変えず、`ControlEnv.step()` と `ControlEnv.reset()` が返す
観測の中の `agentview_image` だけを、返す直前に配列ごと差し替える
（`env_wrapper.py:285-309`・`311-348`）。

| 観点 | 中身 |
|---|---|
| **いつ** | 毎ステップ（`step()` 内）と、エピソード先頭の 1 枚（`reset()` 内）の両方。1 エピソードにつき 1 回だけかかる摂動ではなく、**全フレームにかかり続ける** |
| **どこに** | `agentview`（据え付け俯瞰カメラ）の画像だけ。**手首カメラ（`robot0_eye_in_hand_image`）は汚されない**（コード中に該当カメラへの処理が無い） |
| **何に対して** | シミュレータの出力（レンダリング後の RGB 配列）に対する後処理。ロボットの行動・物理状態・カメラの位置や向きは変えない |
| **強度の決め方** | ファイル名の末尾 `_noise_N`（N=1〜50）を `ControlEnv.__init__` が解析し、`self.noise = N` として保持する（`env_wrapper.py:207-233`） |

**N=1〜50 は、5 種類の劣化×severity 1〜10 を連番にしただけである**（`env_wrapper.py:294-307`）。

| N の範囲 | 劣化の種類 | 中身 |
|---|---|---|
| 1〜10 | `motion_blur` | 被写体ブレ。radius・sigma が severity で増加、**角度は呼び出す度に `np.random.uniform(-45, 45)` で毎回引き直す**（`env_wrapper.py:30-58`） |
| 11〜20 | `gaussian_blur` | ガウシアンぼかし。sigma=1〜10（severity と同値） |
| 21〜30 | `zoom_blur` | ズームブレ（中心から外へ複数倍率でぼかしを重ねる） |
| 31〜40 | `fog` | 霧。diamond-square 法で生成したフラクタル地形を明るさに加算 |
| 41〜50 | `glass_blur` | すりガラス。ぼかした後、近傍画素をランダムに入れ替える処理を複数回反復 |

**severity（1〜10）は劣化の強さの段階を表す整数で、確率的な「効き方」ではない。** ただし
劣化の関数自体が乱数を使うため、**同じ N（＝同じ severity）でも 1 フレームごとに見た目の
実現値は変わる**（motion_blur の角度、glass_blur の入れ替え先など）。「決まった 1 枚のノイズ
画像を貼り付ける」のではなく、「決まった強さの劣化関数を、乱数を引き直しながら毎フレーム
再適用する」という動き方をする。

**5 種の実装（`motion_blur`〜`glass_blur`・`plasma_fractal`）は Hendrycks & Dietterich が
ImageNet-C ベンチマークのために公開したコードそのものである。** コメントアウトされた
severity 5 段階の定数列（例: `motion_blur` の `# c = [(10, 3), (15, 5), ...]`）が元の
ImageNet-C の値で、LIBERO-plus は同じ関数を 10 段階に細分化して使っている。

- 論文: Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations*, ICLR 2019. <https://arxiv.org/abs/1903.12261>
- 公式実装: <https://github.com/hendrycks/robustness>
  （`ImageNet-C/create_c/make_imagenet_c.py` に同名の `motion_blur`・`gaussian_blur`・
  `zoom_blur`・`fog`・`glass_blur`・`plasma_fractal` がある）

---

## 3. 外部文献（LIBERO-plus 論文）が他モデルで報告する脆さ

**arXiv:2510.13626 は、Camera・Robot が他カテゴリより一段深く成功率を落とすと報告する。**
自チームが使っていない OpenVLA・OpenVLA-OFT・π0 での実測である（自チームは π0.5 系）。

| モデル | 素の成功率 | Camera | Robot | Noise | Layout |
|---|---|---|---|---|---|
| OpenVLA | 76.5% | 1.1% | 4.1% | 19.3% | 31.6% |
| OpenVLA-OFT | 97.1% | 59.7% | 37.2% | 76.7% | 77.1% |
| π0 | 94.2% | 15.8% | 6.6% | 79.4% | 70.4% |

**LIBERO-plus データを混ぜた追加学習（mix-SFT）の効果**（OpenVLA-OFT+）:

| | 追加学習前 | 追加学習後 | 差分 |
|---|---|---|---|
| Camera | 約 56〜58% | 92.8% | **+35〜37pt** |
| Robot | 21.7% | 30.3% | +8.6pt（絶対値は依然低い） |
| Noise | 78.6% | 89.3% | +10.7pt |
| Layout | 68.7% | 77.6% | +8.9pt |

**この表からは「Camera・Robot がいちばん深刻で、Camera は追加学習でよく伸びるが
Robot は伸びにくい」と読める。** 4 節の自チームの実測は、この読みと**一致しない**。

---

## 4. 自チームの実測（Track2 疑似検証スイート、2026-08-22。**成功率だけの読みは誤りで、
ステップ数の読みに訂正済み**）

**`compe/val/TRACK2_PROBE.csv`（8 タスク×[Base+4 摂動]=40 行）を、提出物
`work/artifacts/pi05_rp3000`（π0.5 系）で実際に評価した結果である**
（`work/gpu_eval/0822_092556_track2_probe_n16_ep1_max300_ema03_blend0_s1/summary.txt`）。
推論設定は予選最終提出 4.2.0 相当に固定（本番 0.485 と並べて読むため）。

**撤回した判断**: 最初は成功率だけを見て「Sensor Noise だけが Base より明確に低く、
Camera・Robot・Layout は Base と同率 88% で崩れていない」と判断した。この判断は誤りだった。
**手元の成功率は本番の順位を付けられない**という、このチームが予選ですでに確立していた
教訓（[knowledge/07_local_eval_cannot_rank.md](../knowledge/07_local_eval_cannot_rank.md)。
手元 +3 件が本番 0.264→0.169 にも 0.264→0.485 にもなった 2 例）を、この 40 行の
読み方には適用していなかった。本番と繋がっている量は**「基準（Base）で完走していた件
だけのステップ数」**（[knowledge/06_step_count_drives_score.md](../knowledge/06_step_count_drives_score.md)、
r=−0.868）であり、成功率は「壊れていないことの確認」にしか使えない。

**訂正後の数字**（`solution/tools/summarize_probe_by_category.py`で算出）:

| カテゴリ | 成功率 | 打ち切り | ステップ中央値 | ステップ最大 | **基準完走件のステップ比** |
|---|---|---|---|---|---|
| **Camera Viewpoints** | 88%（7/8） | 0/8 | 177 | 290 | **+34.6%** |
| **Sensor Noise** | 75%（6/8） | 2/8 | 136 | 300 | **+24.4%** |
| Robot Initial States | 88%（7/8） | 1/8 | 132 | 300 | +17.5% |
| Objects Layout | 88%（7/8） | 1/8 | 136 | 300 | +15.6% |
| Base（無摂動） | 88%（7/8） | 1/8 | — | — | 基準 |

**成功率の順位とステップ数の順位が逆転している。** 成功率だけでは崩れていないように
見えた Camera Viewpoints が、ステップ数では最も大きく崩れている（+34.6%）。Robot・Layout
も +15〜18% と無視できない。**「成功はしたが遅い」「300 ステップの打ち切りに近づいている」
という、成功率には出ない副作用がカテゴリ間で異なる大きさで存在する。**

**標本数は 1 カテゴリあたり 8 件（比較は n=7）しかない。** Camera Viewpoints の +34.6% は
8 タスク中 2 タスク（衝突 1 件・290/300 ステップ 1 件）が主因であり、統計的に強い結論
ではない。**今は事前準備段階なので n=8 のまま進めるが、量子化推論の高速化で GPU 評価の
コストが下がり次第、標本数を増やして数字を確かなものにする**（2026-08-22 決定）。

---

## 5. 実測と文献の食い違い（確認したい仮定）

**3 節（他モデルの文献値）と 4 節（自チームの実測、ステップ数ベース）を比べると、
食い違いは当初思っていたほど大きくない。** 文献では Camera・Robot が他カテゴリより
一段深く崩れると報告されており、4 節のステップ数ベースの読みでも **Camera Viewpoints が
最も大きく崩れている（+34.6%）**点は文献と方向が一致する。一方 **Robot Initial States は
文献（mix-SFT でも伸びが最も小さい弱点カテゴリ）ほど突出して悪いわけではなく、
Objects Layout と近い水準（+17.5% 対 +15.6%）**である。

**Sensor Noise は依然として食い違いが残る。** CRT 論文は π0.5・SmolVLA がセンサ劣化で
90%→2% まで崩れると報告するが、4 節の実測は +24.4%（打ち切り 2/8）に留まる。原因の
候補として、**このスイートの Sensor Noise 8 行が `_noise_3`〜`_noise_14`——LIBERO-plus の
5 種の画質劣化のうち最も軽い motion_blur・gaussian_blur の低〜中強度しか使っておらず、
より崩れの大きい zoom_blur・fog・glass_blur を 1 件も試していないこと**が判明している
（[09 §3.2](../plan/09_demo_perturbation_pipeline.md#32-高強度sensor-noiseの追加実験準備済み実行待ち)）。
**この確認のための追加実験（`SENSOR_NOISE_HIGH_PROBE.csv`、32 行）は実行済みである
（2026-08-22、[09 §3.2](../plan/09_demo_perturbation_pipeline.md#32-高強度sensor-noiseの追加実験実行済み2026-08-22)）。**
結果は総合成功率 90.6%（29/32）——severity を上げても崩れなかった。つまり文献値との
食い違いは「強度の違い」では説明できず、**LIBERO-plus の合成画質劣化は π0.5 系モデルに
対して CRT 論文が報告するほどの崩れを起こしていない**、という結論になった
（Sensor Noise 対策の投資優先度を下げる判断に反映済み）。

**Robot Initial States には、同じ構造の未解決の食い違いが残っている。** 文献では
mix-SFT 後も伸びが最小（+8.6pt 止まり）の弱点カテゴリだが、自チームの実測は +17.5%で
Objects Layout（+15.6%）とほぼ同水準——さらに `make_track2_probe.py` の
`_pick_variant` は各カテゴリで「難易度が中央値の 1 件」しか選んでいないため、
Robot Initial States は難易度 1〜5・`robots/new_init.py` の摂動の大きさ 0.1〜0.5 が
どちらも「5 段階×各 100 個体」という同じ構成であり、両者が対応すると見ているが、
**分類表（`task_classification.json`）の`difficulty_level`をどの処理が割り振っているかを
示すコードは見つかっておらず、対応は未確認**（コードは静的な JSON を読むだけで、
生成処理は同梱されていない）。この対応が正しい前提で、中央値（難易度 3、0.3 相当と
見立てている段階）でしか見ていない可能性がある。Sensor Noise と同じ「severity を
上げたら見え方が変わるか」を確認する高強度 probe が次の実験として準備中である。

---

## 出典

| 事実 | 出典 |
|---|---|
| Track2 の文言・4 カテゴリへの対応づけが推測であること | `docs/PARC2026開発コンペティション_予選.pdf` p8、[07 §1.1](../plan/07_track2_preparation.md#11-track2-の位置視点ノイズとは何か) |
| 4 カテゴリの件数・接尾辞・難易度分布 | `work/LIBERO-plus/libero/libero/benchmark/task_classification.json`（revision `4976dc3`）を実際に数えた |
| 各カテゴリの実装詳細 | `LIBERO-plus/libero/libero/envs/env_wrapper.py`・`libero_tabletop_manipulation.py`・`robots/new_init.py` |
| Sensor Noise が「毎フレーム・agentview のみ・後処理」であること | `env_wrapper.py:207-233`（`_noise_N` の解析）・`285-309`（`step()`）・`311-348`（`reset()`） |
| Sensor Noise 5 種の劣化関数の実装 | `env_wrapper.py:26-174`（ImageNet-C 系）。原典は Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR 2019（<https://arxiv.org/abs/1903.12261>）、公式実装 <https://github.com/hendrycks/robustness> |
| 他モデルでの成功率・mix-SFT の効果 | arXiv:2510.13626（researcher エージェントが HTML 版を複数回取得、[07 §1.3.1](../plan/07_track2_preparation.md) に確認度を明記） |
| 自チームの Track2 疑似検証スイート実測（生ログ） | `work/gpu_eval/0822_092556_track2_probe_n16_ep1_max300_ema03_blend0_s1/summary.txt`・`eval.log` |
| 手元の成功率は本番の順位を付けられないこと | [knowledge/07_local_eval_cannot_rank.md](../knowledge/07_local_eval_cannot_rank.md) |
| ステップ数（基準完走件）が本番と繋がる量であること（r=−0.868） | [knowledge/06_step_count_drives_score.md](../knowledge/06_step_count_drives_score.md) |
| カテゴリ別のステップ数の読み直し（訂正後の数字） | `solution/tools/summarize_probe_by_category.py` を実行して算出 |
| Sensor Noise 8 行が低〜中強度の劣化しか含まないこと・高強度追加実験の準備 | [09 §3.2](../plan/09_demo_perturbation_pipeline.md#32-高強度sensor-noiseの追加実験準備済み実行待ち)、`work/robot_init_sanity/SENSOR_NOISE_HIGH_PROBE.csv` |
| TRACK2_PROBE.csv が「順位付けツールで本評価ではない」こと | `solution/tools/make_track2_probe.py` の docstring |
| 学習データに摂動カテゴリ別の内訳が特定できないこと | [06_training_data_composition.md](06_training_data_composition.md) |
| π0.5 系提出物が手首カメラを使っていること | `solution/policy_server.py`（`robot0_eye_in_hand_image` を観測に使用）、[07 §4](../plan/07_track2_preparation.md#4-わからないこと確認が必要なこと) |
