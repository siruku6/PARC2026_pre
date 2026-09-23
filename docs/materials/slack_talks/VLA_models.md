# open 10 checkpoints

公開されているチェックポイント10モデルをLIBERO-Plus(Spatial)タスク比較しました。

LeRobot 系5つ — LeRobot 上でそのまま推論できる LIBERO/LIBERO-Plusファインチューニング済みモデルです。

| モデル | weight |
| smolvla | lerobot/smolvla_libero_plus |
| pi05 | lerobot/pi05_libero_finetuned_v044 |
| vla_jepa | lerobot/VLA-JEPA-LIBERO |
| xvla | lerobot/xvla-libero |
| fastwam | lerobot/fastwam_libero_uncond_2cam224 |

リーダーボード上位5本 — LIBERO-plus benchmark
および VLA Evaluation Harness leaderboard
で上位、かつweightとコードの両方が公開されているモデルです。

| モデル | weight | コード |
| anchor_align | Dwipz/Anchor-Align | dwipddalal/Anchor-Align |
| rldx1 | RLWRLD/RLDX-1-FT-LIBERO | RLWRLD/RLDX-1 |
| multiview_vla | junjin0/Multi-view-VLA | junjxiao/Multi-view-VLA |
| gam | SeonghuJeon/3da-libero-gam | cvlab-kaist/Geometric-Action-Model |
| internvla_a15 | InternRobotics/InternVLA-A1.5-Libero | InternRobotics/InternVLA-A-series |

学習・ファインチューニングは一切行わず、公開weightをそのまま推論に用いました。

評価

1モデルあたり90 rolloutを実行しました。

成功率（70 rollout）

LIBERO-Spatialの10ベースタスクそれぞれについて、7 つの摂動軸に対応するタスクを実施させています。

# MolmoAct2-LIBERO

単純に基盤モデルをSmolVLA→MolmoAct2-LIBEROへ替えただけで、学習前評価が50%→91.7%相当まで上がった。
SmolVLA Base          50.0%
SmolVLA + LoRA        33.3%
MolmoAct2 Base        91.7%