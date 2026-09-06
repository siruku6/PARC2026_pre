# reference/ — 調べて分かった事実

**「いま判断に使う事実」だけを置く。**役目を終えたものは
[archive/](archive/) へ退避する（消さない）。

**書くのは事実である。**手順は [../howto/](../howto/)、踏んだ事故と回避策は
[../knowledge/](../knowledge/)、1 つずつの手の設計は [../plan/](../plan/) にある。

| 引きたいもの | ファイル |
|---|---|
| **流した学習・重み・版・出した点** | **[19_weights_catalog.md](19_weights_catalog.md)（実験台帳）** |
| モデルの正体（素の重み・中身の説明） | [01_base_model.md](01_base_model.md)・[09_model_explained.md](09_model_explained.md) |
| 採点環境の構成と制限 | [02_evaluation_data.md](02_evaluation_data.md) |
| 採点式 | [17_scoring_formula.md](17_scoring_formula.md) |
| **本番は 8 エピソードで決まる** | [22_production_scale.md](22_production_scale.md) |
| 採点のタスクの母集団（Track1 の母集団） | [04_track1_population.md](04_track1_population.md) |
| 学習データ 14,347 本の中身 | [06_training_data_composition.md](06_training_data_composition.md) |
| **自分でエピソードを作るとき、1 本に何が要るか**（形式・中継物の npz・既にある変換器） | **[29_training_data_requirements.md](29_training_data_requirements.md)** |
| **逆向きエピソードの掴む姿勢の決め方**（順方向から移せる条件・「立つ軸」の測り方・真上からの握り・順方向デモのふるい） | **[30_reverse_grasp_pose.md](30_reverse_grasp_pose.md)** |
| **逆向きエピソードの運び方と置き方**（下の皿やトレイを擦らない・空中で放して落とさない） | **[31_reverse_place_without_contact.md](31_reverse_place_without_contact.md)** |
| 学習の条件と、次にどこを直すか | [11_training_schedule.md](11_training_schedule.md) |
| LoRA をどこに当てるか | [12_lora_targets.md](12_lora_targets.md) |
| 検証用に取り分けたエピソード | [13_holdout_episodes.md](13_holdout_episodes.md) |
| 学習後に変えてよい設定 | [07_chunk_params.md](07_chunk_params.md) |
| 観測がどう渡るか | [10_observation_interface.md](10_observation_interface.md) |
| 摂動のカテゴリ（3 本の分担は [20](20_track2_perturbation_categories.md) の冒頭） | [05](05_perturbation_categories.md)・[20](20_track2_perturbation_categories.md)・[21](21_weakness_map.md) |
| 見た目の振れ幅 | [08_appearance_spread.md](08_appearance_spread.md) |
| 失敗の内訳 | [18_failure_breakdown.md](18_failure_breakdown.md) |
| **Track3 の中身（配布 4 件・目標条件・母集団の作り）・配布 init と把持の幾何** | **[28_track3_contents.md](28_track3_contents.md)**（5 節に実測） |
| **LIBERO シムを回すときの規約と落とし穴**（`action` の意味・gripper の符号化・状態復元で復らないもの・時刻反転特有の落とし穴） | **[25](25_dataset_inventory.md) 8.7 節**・**[28_mppi_data_regeneration_survey.md](28_mppi_data_regeneration_survey.md) 5.1〜5.2 節** |
| weak3x を π0.5 に使うか | [16_weak3x_for_pi05.md](16_weak3x_for_pi05.md) |
| レポートに書く値 | 重み・版・点は [19_weights_catalog.md](19_weights_catalog.md)、ライセンスは [../knowledge/04_licenses.md](../knowledge/04_licenses.md)、必須欄は [../knowledge/01_report_requirements.md](../knowledge/01_report_requirements.md) |
