# 自作データセット一覧（今使えるもの）

TOC:
- [0. このファイルの役割](#0-このファイルの役割)
- [1. 一覧表](#1-一覧表)
- [2. 各データセットの補足](#2-各データセットの補足)
- [3. 確認したい仮定](#3-確認したい仮定)

## 0. このファイルの役割

学習データを新しく組むかどうかを検討するとき、「すでにどんな切り口を試したか・
どこに置いてあるか」を毎回過去のセッションログから掘り起こすのは無駄が大きい。
このファイルは**自分たちで作った・組み合わせたデータセットのうち、今も参照する価値が
あるもの**を一覧にし、置き場所と工夫の要点だけを引けるようにする。

**書かないもの**: 各データセットで何点取れたかという提出履歴は
[reference/19_weights_catalog.md](../reference/19_weights_catalog.md) に、
配布データそのものの内訳（`libero_plus` や `libero_combined_20hz` の構成）は
[reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md) にある。
このファイルは概要だけを載せ、詳しい経緯はそちらを参照する形にする。

**載せていないもの**: 試したが何の手がかりも得られず、その方向を続ける予定が無い
もの（例: 汚れデモを除いた `libero_plus_clean` は Track1/Track2 とも動かず、
ディレクトリ自体も削除済み）は表に含めていない。

## 1. 一覧表

| データセット | 置き場所 | 作り方の概要 | 工夫 | 今の位置づけ |
|---|---|---|---|---|
| `libero_states` | ローカルのみ（`work/artifacts/libero_states`） | 配布元の hdf5（`yifengzhu-hf/LIBERO-datasets`）から MuJoCo の生 state/action を抜き出した下ごしらえ用データ（40 課題・2,000 デモ） | 各デモに接触判定（`contact_scores.json`）を付け、配布 LeRobot データの「どのデモが対象外に接触しているか」を機械的に判定できるようにした | `libero_plus_bowl5` の丼課題入れ替え判定や、Track3 の順方向デモ対応付けの支援ツールとして使用中 |
| `reverse_lerobot_v2` | ローカルのみ（`work/artifacts/reverse_lerobot_v2`、397 episodes） | 順方向の手本デモを時間反転し、行動（相対差分）の符号を反転して「逆再生」の教師データを作る（Track3: 物を置いた状態から取り出す方向） | 初代（`reverse_lerobot_389`、HF `siruku6/reverse_lerobot_389`）で出た事故を修正——回転量のクリップ方式変更・不要な水平合わせ動作の除去・持ち上げ後の静止時間追加で、手本の落下事故をほぼ解消 | `jaist_mix_v2_x8` の材料として使用中。単独では HF に上げていない |
| `jaist_mix_v2_x8` | HF `siruku6/jaist_mix_v2_x8`（非公開）／ローカル `work/artifacts/jaist_mix_v2_x8`（6,176 episodes） | `libero_plus` を 75 episodes/課題に間引いたもの（3,000 本）と `reverse_lerobot_v2` を 8 倍複製したもの（3,176 本）を混合し、反転デモの比率を約 5 割まで引き上げた | 混合比を意図的に反転側へ寄せることで、順方向データに埋もれず反転動作を学習させる設計。正規化統計は既存の学習済みモデル（`rp3000`）に固定して比較を単純化 | 初代の `jaist_mix_75_x7`（HF `siruku6/jaist_mix`）の後継。この系列で初めて Track3 のローカル評価が非ゼロになった（提出は 0.000 のまま） |
| `libero_plus_bowl5` | HF `siruku6/libero_plus_bowl5`（非公開）／ローカル `work/artifacts/libero_plus_bowl5`（14,085 episodes） | 配布データのうち衝突率が高かった「丼」5 課題ぶん（1,880 本）を外し、作り直したスクリプト操作のデモに入れ替えた | 掴む半径を縁の壁の中心に合わせ、降りる勢いを絞ることで、対象 5 課題の成功率を摂動込みで 130/150 まで改善してから収録 | データセット自体は完成・公開設定込みでアップロード済み。学習・評価はまだ結果が出ていない（進行中） |
| `lerobot_batch1` | ローカルのみ（`work/regen/lerobot_batch1`、1,009 episodes） | 配布デモの行動列を、初期姿勢をずらした状態で実際の採点ルール通りに再生し直した「摂動下再生」データ | 配布データが薄い摂動幅（手のオフセット 2.4〜6.5cm）を狙って埋め、その帯域で成功率が落ちる問題に対応 | 作成済みだが、`libero_plus_bowl5` の結果を先に見てから 1 変数ずつ足す方針のため、学習にはまだ投入していない |

## 2. 各データセットの補足

- **`jaist_mix_v2_x8` の HF リポジトリ名は資料によって表記が割れている。** 大半は
  `siruku6/jaist_mix_v2_x8` だが、一部の資料では `siruku6/jaist_mix_v2` とも書かれている。
  同じ中身を指しているとみて一覧には代表として `_x8` 付きの名前を採用した。
- 上記はいずれも土台となる配布データ `lerobot/libero_plus`（14,347 episodes・40 課題）の
  上に成り立っている。配布データ自体の内訳・素性は
  [reference/25_dataset_inventory.md](../reference/25_dataset_inventory.md) を見る。
- `libero_plus_sub75`・`libero_plus_nobowl5`・`lerobot_bowl_all` などは、上表のデータセットを
  組む過程で生まれる中間生成物であり、単独では使わないため表に含めていない。

## 3. 確認したい仮定

- `jaist_mix_v2_x8` と `jaist_mix_v2` が同一リポジトリの表記ゆれなのか、実際に別リポジトリ
  として存在するのかは、`HfApi` で数え直して確認していません。
- `libero_plus_bowl5` を使った学習（`pi05_bowl5` / `pi05_bowl5_full`）の結果はまだ
  `work/final_eval/` に出ていません。結果が出た時点でこの表の「今の位置づけ」列を更新してください。
