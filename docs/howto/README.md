# howto —— 手順書

**頻繁に使う手順書を置く。**上から順にやれば終わる形にしてある。
1 回だけの作業は [../plan/](../plan/) へ、踏んだ事故と回避策は [../knowledge/](../knowledge/) へ書く。

| 手順書 | 何をする手順か |
|---|---|
| [01_retrain.md](01_retrain.md) | 再学習する |
| [02_build_submission.md](02_build_submission.md) | 提出物の zip を作る |
| [03_gpu_eval.md](03_gpu_eval.md) | GPU マシンで評価を回す |
| [04_proxy_eval.md](04_proxy_eval.md) | 本番と同じ形で評価する（提出前の判断に使える検証を作る） |
| [05_scripted_demos.md](05_scripted_demos.md) | 手本を scripted policy で作る |
| [06_runpod_pi05.md](06_runpod_pi05.md) | RunPod で π0.5 をフルパラメータ学習する |
| [07_train_pi05_local.md](07_train_pi05_local.md) | π0.5 をこのマシン（H100 20GB）で学習する |
| [08_hf_transfer_troubleshooting.md](08_hf_transfer_troubleshooting.md) | Hugging Face 経由で重みを運ぶときに詰まったら |
| [09_free_disk_space.md](09_free_disk_space.md) | ディスクを空ける |
| [10_rebuild_machine.md](10_rebuild_machine.md) | このマシンを作り替えたあと、元通りにする |
| [11_colab_eval.md](11_colab_eval.md) | Colab GPU で評価を回す |
| [13_colab_vggt_latency.md](13_colab_vggt_latency.md) | Colab の L4 で VGGT の推論時間だけを測る |
| [14_blackwell_session.md](14_blackwell_session.md) | Blackwell サーバで 1 本回す（毎セッションの手順書） |
| [15_smooth_command_profile.md](15_smooth_command_profile.md) | 自作デモの指令を配布データと同じ形にする |
| [16_bowl_regen_train_eval.md](16_bowl_regen_train_eval.md) | 丼 5 課題を作り直して、学習と対照評価を回す |
| [17_checkpoint_upload.md](17_checkpoint_upload.md) | 学習したチェックポイントを HF へ運ぶ |
| [18_inspect_weights.md](18_inspect_weights.md) | 重みの素性（sha256・公開/非公開・使用量・学習の種類・推論の設定）をコマンドで引く |
| [20_rebuild_l90clean_elsewhere.md](20_rebuild_l90clean_elsewhere.md) | `libero_combined_bowl5_initpose_l90clean` を部品と台帳から別のマシンで組み直す（消せるようにするための確認） |
| [19_train_l90clean_runpod.md](19_train_l90clean_runpod.md) | `libero_combined_bowl5_initpose_l90clean` で 1 本学習する（RunPod・π0.5 と MolmoAct2 の選び方と手順） |

**12 は欠番である**（削除済み）。
