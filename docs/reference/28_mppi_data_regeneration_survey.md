# 28. MPPI で学習データを再生成できるか——調査

**問い**：Track3（逆向きタスク）向けの学習データを、現在予定している「時間反転」や
「手書きの scripted policy」ではなく、**MPPI（モデル予測型パスインテグレーション）で
自動生成**できないか。

**結論**：このコンペの条件下では**不向き**。理由の 9 割は「計算コスト」であり、
残りは「把持・接触が MPPI の弱い領域であること」と「動く scripted/時間反転の経路が
既に存在すること」である。MPPI が効くのは**学習済みダイナミクスモデル（= シミュレータ
を回さない代用品）が既にあって、それを安価に rollout できる場合**だが、この作業では
rollout の主役が MuJoCo シミュレータそのもので、それこそ一番高価なものなので、
MPPI の利点（モデル不要・非線形を扱える）が活きず、欠点（1 決定ごとの
サンプル数 × ホライズんの sim 計算）だけが残る。

---

## 1. MPPI が何 doing か（アルゴリズムの要点）

MPPI は、各ステップで「候補行動系列を多数サンプルし、将来のコストで重み付けし、
重み付き平均を取った行動を実行する」手続きである。1 回の制御決定（control decision）に
要るのは、**サンプル数 N 本 × ホライズン H ステップ分のシミュレーション rollout** である。

- サンプルはガウスの混合や前回の分布から引かれ、コストレインを
  `exp(−cost/温度)` で重み付けし、上位のelite から分布を再推定して次のサンプル源にする。
- **model-free な MPPI**（シミュレータを直接回す方式）は、シミュレータを N×H 回
  呼ぶ。**学習済みダイナミクスモデル**（TD-MPC2 など）に置き換えると rollout が安くなり、
  そこが MPPI/MPC 系の本領である。

> 出典（web・1 次情報）：arXiv 検索「MPPI」の直近応用（45 件を確認）。
> 対象はナビゲーション・自律運転・クアッドコプター・ヒューマノイド運動制御が中心で、
> **把持・接触を含む操作タスクへの適用は少数**。代表的なもの：
> - Object-informed MPPI for non-prehensile manipulation (push) — arXiv:2605.30778
> - KernelSOS + 局所 MPPI で接触が多い操作 — arXiv:2604.27175
> - Dynamics-based torque-sampling MPPI（7 自由度・166 Hz 以上）— arXiv:2609.02020
> 正典は Walter, Kuwata et al. "Sampling-Based Model Predictive Control with
> Nonparametric Cost Models"（arXiv、四つ星機向け）。

---

## 2. このコンペで MPPI を回すのに掛かる計算コスト（推定）

MPPI の決定ごとの計算量は `N × H` の sim ステップ。実測のシミュレータ速度を掛けて
1 エピソード（300 歩、action chunk 10 なので replan は 30 回）の所要時間を推定した。

**シミュレータ速度の実測**（一次情報・ログ／howto）：

| 機械 | env.step | 根拠 |
|---|---|---|
| JAIST H100-20C（vGPU・osmesa） | 384 ms/step | [docs/howto/03_gpu_eval.md](../howto/03_gpu_eval.md) の表 |
| RTX6000（会社・Pass-Through・EGL） | 描画 1 枚 0.6 ms（osmesa は 9.7 ms）。osmesa 60.6 s → EGL 30.6 s | [docs/howto/03_gpu_eval.md](../howto/03_gpu_eval.md) |
| Colab（EGL） | env.step 38.6 ms／本あたり 110 s 実測 | [docs/howto/11_colab_eval.md](../howto/11_colab_eval.md) |
| 採点環境 | NVIDIA L4（運営が計測し明示） | Q&A CSV No.64 付近 |

**推定**（EGL 環境・env.step ≈ 16〜38 ms とし、N=2,000、H=20）：

```
1 制御決定 = N×H = 2,000 × 20 = 40,000 sim ステップ
           ≈ 40,000 × 16〜38 ms = 640〜1,520 s = 11〜25 分
1 エピソード（30 決定）      ≈ 5.5〜12.5 時間
100 タスク × 10 デモ = 1,000 エピソード  →  数千〜万時間
```

対して**現在の実行経路（scripted / 時間反転）はシミュレータを 1 回だけリアルタイムで
回す**ため、1 エピソードは数十秒〜数分（描画込み）。**MPPI はこれより 2〜3 桁高い**。

- 無料 GPU（Blackwell 残り約 53 時間・9/17 に消える）は**学習用**であり、
  シミュレータ rollout に充てると学習の梯子と取り合う（[../plan/21_initiative_priority_second_opinion.md](../plan/21_initiative_priority_second_opinion.md) 7 節）。
- 有料 RunPod（1,500 歩 = 4 時間 16 分・$8〜9）で 1,000 エピソードの MPPI を回すと、
  上記推定では**何十〜何百ドルを大きく超え、日数的にも 9/17 に届かない**。

**要点**：MPPI を本物の MuJoCo を使って回すと「1 決定に 11〜25 分」になる。
これは learned-model 型の MPC（rollout が安価な NN を回すだけ）とは本質が違い、
**このコンペの制約（低速 sim・無料 GPU 期限・毎日の提出枠）ではスケーリングできない**。

---

## 3. 把持・接触は MPPI の弱い領域

MPPI は連続的な軌道最適化が得意で、**グリップの離散的な開閉・接触・把持のような
非光滑な遷移は苦手**。Track3 は把持・置物が中心なので、コストの問題がなくても
成功率で不利になる。この点は scripted（物をどこに置くか分かるので手順を直接書ける）
や時間反転（順方向デモが既に接触遷移を含んでいる）の方が自然に扱える。

---

## 4. ルール上は禁止事項に抵触しない（ただし効かない）

ご自身の環境で LIBERO を動かしてロールアウト（観測画像・状態・行動）を保存し
学習データに使うことは**明示的に可**。禁止されるのは運営採点環境から得られた
情報を利用することだけ。

> 出典（一次情報）：Q&A CSV
> [docs/materials/PARC2026  テクニカル関連質問 - テクニカル質問.csv](../materials/PARC2026%20%20%E3%83%86%E3%82%AF%E3%83%8B%E3%82%AB%E3%83%AB%E9%96%8B%E8%AA%AC%E5%95%8F%E3%81%84%20-%20%E3%83%86%E3%82%AF%E3%83%8B%E3%82%AB%E3%83%AB%E8%AB%97%E5%95%8F.csv)
> No.33（motofukao）「配布 kit をご自身で実行し、自分で選んだ LIBERO-plus タスクの
> ロールアウトを保存して学習に使うことは可能。kit を経由せず LIBERO を直接動かして
> 収集したデータも同様。自己評価の結果で"何を学習するか"を選定することも問題ない」。

つまり **MPPI でデモを作ってもルール違反にはならない**。問題は「作れるか」でなく
「採算・日数で成立するか」であり、§2 で負けている。

---

## 5. リポジトリ側の現状（別作業で確認した事実）

- **既存の MPC/MPPI コードは無い**。`work/lerobot-src/.../tdmpc/` は **TD-MPC**（学習
  モデルベース）で、計画は MPPI+CEM（`modeling_tdmpc.py:199-258`）だが、**学習済み
  潜在ダイナミクスモデルが必要**。古典的な model-free MPPI（シミュレータを直接
  rollout する実装）はリポジトリに存在しない → **新実装が必要**。
- **scripted デモ生成は既に動く**。`solution/tools/scripted_demo/scripted_demo.py`
  が LIBERO-plus フォークの `ControlEnv` を回し、`npz_to_lerobot.py` が **LeRobot v3.0
  （fps 20・画像 256×256・state 8・action 7・panda）**へ変換するので、出力がそのまま
  学習データ形式になる。既定は osmesa（CPU）で、並列は `--episodes` を分けて
  複数プロセスを回す方式。
- **π0.5 が読むデータは raw states を含む**。`observation.state` が 8 次元
  （eef pos 3 + 軸角 3 + gripper 2）で画像と並ぶ（`solution/train/train_pi05.py` の
  state [8] 宣言／`policy_server.py:583`）。**時間反転に使える raw states は
  学習データセットに含まれる**。
- **1 回の `/act` = 1 歩の action（7 次元）**。action は
  `[dx, dy, dz, droll, dpitch, dyaw, gripper]`（`policy_server.py:214`）。
  replan は action chunk ごとに（`n_action_steps` 既定 16）。
- **推論と sim driver の結合は緩い**。`pipeline/rollout.py` が `PolicyInterface`
  （`get_action`／`reset`）で policy を抽象化しており、**planner を policy server に
  差し替えるのは `PolicyInterface` を満たすクライアントを作るだけ**。ただし §2 の
  コストで、差し替え自体が安易に有利にはならない。
- **タスク資産はローカルに揃っている**。LIBERO-plus フォークが全 100 SCENE 名タスクの
  BDDL（`bddl_files/libero_90/` 90 + 派生、`bddl_files/libero_10/`）と初期状態
  （`init_files/`）をローカルに持ち、`ControlEnv(bddl_file_name=...)` が任意の BDDL を
  開ける。→ 100 タスクぶんを回すための**環境側の手間は無い**。
- **Track3 の 4 タスクの BDDL・初期状態は本リポジトリに無い**（`b_PARC2026_final`
  キット側の `compe/t3/assets/`）。逆向き BDDL は順方向 BDDL から自分で逆にするか、
  キットの t3 assets を取ってくる必要がある。

---

## 6. 「それでもやるなら」——手順・準備・やり方の案

結論は不向きだが、限定的に試す価値がある場合（例：4 タスクのうちの 1 つだけ、
MPPI の実装を兼ねて go/no-go を測る）の手順を置いておく。

1. **実装**（新コード）。model-free MPPI を `PolicyInterface` に実装し、
   `pipeline` から呼べる形にする。シミュレータの clone を N 本並列で回し、
   各サンプルを H 歩 rollout してコスト（goal 未達の距離・衝突・smooth 系の項）を
   付けて重み付け → elite から分布更新 → 重み付き平均の先頭 action を返す。
   既存 TD-MPC の CEM 部分（`modeling_tdmpc.py:199-258`）をモデル rollout →
   実シミュレータ rollout に書き換えるのが近道。
2. **コストを実測で潰す**。1 タスクで N∈{256, 1024, 4096}・H∈{10, 20, 50} を
   スイープ（少しずつ変えながら測り直すこと）し、「1 決定の所要時間」を RTX6000 EGL
   で実測する。§2 の推定が ±何倍か確定させる。**ここが go/no-go の分岐点**。
3. **並列と機械**。EGL が効く RTX6000（会社）か Colab で回す。osmesa（JAIST）では
   env.step が 384 ms になり §2 の数字がさらに 2〜4 倍になるため使わない。
   無料 Blackwell は学習に回すので使わない。
4. **成功と衝突の選別**。MPPI のコストには衝突項を含める。生成したデモは必ず
   ギャップがない（操作対象以外の物が 1 mm 動いていない）ものだけ残す。
   再生して goal 条件を満たしたものだけ学習に入れる。
5. **時間反転と 1 タスクで同時比較**。§2 のコストが「1 決定 = 数分」以内で
   止まらなければ、MPPI は時間反転/脚本ed に負けて打ち切る。止まっていれば
   100 タスクへの拡張性を §2 の見積もりで再判定する。

**必要な準備**：EGL 環境の 1 台（RTX6000 か Colab）、LIBERO-plus フォーク、
MPPI 実装（約数百行・既存 TD-MPC の CEM を流用）、4 タスクの逆 BDDL
（順方向から生成 or キット取得）、cost 関数の設計（goal 距離・衝突・smooth）。

---

## 7. 判断

| 案 | コスト | 把持・接触 | 100 タスクへの拡張 | 判定 |
|---|---|---|---|---|
| **MPPI（model-free・実 sim）** | §2 で 2〜3 桁高い | 苦手 | 厳しい | **不向き** |
| **時間反転** | 1 回リアルタイム replay | 順デモが接触を含む | 自動で 100 ぶん | **本命**（raw states の確認が前提） |
| **scripted policy** | リアルタイム replay | 手順を直接書ける | 4 タスク程度で頭打ち | **補完** |

**推奨**：MPPI は**本物の MuJoCo を使って 100 タスクのデモを量産する手段としては
採らない**。時間反転（raw states が入っていることを実データで確認してから）を本命に、
scripted を補完にする現行計画を維持する。MPPI は「1 タスクで実装兼 go/no-go」に
留める場合のみ、§6 の手順でコストを実測してから判断する。

---

## Gap（確認できていないこと）

1. **Track3 の 4 タスクの BDDL・目標述語・初期状態の中身**。`compe/t3` は本リポジトリに
   無く、`b_PARC2026_final` キットが作業ディレクトリに無い。
2. **`lerobot/libero_plus` の raw states が実データで本当にエピソード単位の raw robot
   状態か**。HF／配布 tar 側のため、実ファイルでの確認はしていない（学習コードの
   state 8 次元宣言と `policy_server.py:583` の記述から推定）。時間反転の go/no-go に
   は**この確認がそのまま前提**になる。
3. **本選 pipeline/scorer の実コード**。本リポジトリの `pipeline/` は予選 track1 用で、
   本選 track3 を評価する pipeline はキット側。
4. **MPPI のコスト推定は推算**（N・H の取り方で変わる）。確定には §6-2 の実測が要る。