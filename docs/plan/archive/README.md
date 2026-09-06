# plan/archive/ — 役目を終えた設計

**消さずに退避した設計である。**[plan/](../) には、いま動いている手の設計だけを置く。

**消さないのは、2026-08-26 に `13_finals_exploration_menu.md` を統合後に削除して
取り戻せなくした（git の追跡外だった）ためである。**退避なら戻せる。

| ファイル | 何だったか | なぜ退避したか |
|---|---|---|
| [03_split_learning_rate.md](03_split_learning_rate.md) | 画像側を薄い学習率で動かす設計 | SmolVLA 期のもの。いまの土台は π0.5 |
| [05_postqualifier_cleanup.md](05_postqualifier_cleanup.md) | 予選直後の片づけの段取り | 片づけが済んだ |
| [06_eval_gap_measurement.md](06_eval_gap_measurement.md) | 手元評価と本番のずれを測る依頼 | **2026-08-22 に本文で取り下げ済み** |
| [07_track2_preparation.md](07_track2_preparation.md) | 本選 Track2（位置・視点・ノイズ）への準備。摂動 4 カテゴリの中身・対策・推奨アクション | **2026-09-08 に退避。**内容は [../reference/20_track2_perturbation_categories.md](../../reference/20_track2_perturbation_categories.md)・[../reference/21_weakness_map.md](../../reference/21_weakness_map.md) へ引き継いだ |
| [09_demo_perturbation_pipeline.md](09_demo_perturbation_pipeline.md) | 学習デモに摂動を加える作業の整理（Tier 0〜2） | **2026-09-08 に退避。**Tier 1・2 は plan/11・12 で実施済み、実測は [../reference/20](../../reference/20_track2_perturbation_categories.md)・[21](../../reference/21_weakness_map.md) へ |
| [10_eval_perturbation_suite.md](10_eval_perturbation_suite.md) | 評価データに摂動を加える計画と実測（7・8 節） | **2026-09-08 に退避。**結論と 4 カテゴリの地図は [../reference/21_weakness_map.md](../../reference/21_weakness_map.md) へ |
| [11_tier1_camera_augmentation.md](11_tier1_camera_augmentation.md) | Tier 1（Camera 対策の画像拡張）の実装記録 | **2026-09-08 に退避。**成果物は `--camera-viewpoint-augment` の学習（[../reference/19_weights_catalog.md](../../reference/19_weights_catalog.md) の trial1500）で採点済み |
