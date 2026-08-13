# 14. π0.5 への切り替えは可能か（並行トライの前提事実）

**結論：技術的には可能。ただし残り時間・GPU メモリ・推論速度の3点が未検証。**
[train_pi05.py](../../solution/train/train_pi05.py) を用意したので、まず
`--smoke-test` で 20GB GPU に載るかを確かめてから本番の歩数へ進むこと。

> 出典はすべて GitHub の v0.6.0 タグそのもの、Hugging Face の API/モデルカード、
> `nvidia-smi` の実測。リポジトリ内の他の `.md` は出典にしていない。

## 確認した事実

| 事実 | 値 | 出典 |
|---|---|---|
| lerobot v0.6.0 に pi0/pi0.5 の実装があるか | **ある**（`policies/pi0`, `policies/pi05`） | https://github.com/huggingface/lerobot/tree/v0.6.0/src/lerobot/policies |
| pi05 の extras 名 | `pi`（`training` と併用） | https://raw.githubusercontent.com/huggingface/lerobot/v0.6.0/pyproject.toml |
| LIBERO 向け出発点チェックポイント | `lerobot/pi05_libero_base` | https://huggingface.co/lerobot/pi05_libero_base |
| そのコミット | `a217bfd3b14673cf2ce597e69997ab21866438dd`（2026-08-13 確認） | `curl -s https://huggingface.co/api/models/lerobot/pi05_libero_base` |
| ライセンス | gemma（`google/paligemma-3b-pt-224` の利用規約への同意と `hf auth login` が必要） | 同モデルカード |
| 入力の宣言 | `observation.images.image` / `.image2` [3,256,256]、`observation.state` [8]、`action` [7]、`empty_cameras=1`、`chunk_size=50`／`n_action_steps=10` | `curl -s https://huggingface.co/lerobot/pi05_libero_base/raw/main/config.json` |
| パラメータ構成 | `paligemma_variant=gemma_2b` + `action_expert_variant=gemma_300m` | 同 config.json |
| 公式の学習コマンドの前提 | 「80GB GPU 1枚」（`train_expert_only=true` の縮小版でも batch_size=64） | https://huggingface.co/docs/lerobot/en/pi05 の Training セクション |
| この環境の GPU | 1枚・20GB（`nvidia-smi --query-gpu=name,memory.total`） | 本ホストで実測 |

## state=8 / action=7 は今の観測変換とそのまま一致する

`lerobot/pi05_libero_base` の入力宣言（state 8・action 7）は、このコンペの
`LiberoProcessorStep`（[10_observation_interface.md](10_observation_interface.md)）が
作る形とそのまま一致する。カメラ名は `--policy.pretrained_path` 使用時に
学習データ側（`lerobot/libero_plus` の `front`/`wrist`）へ自動的に差し替わるため、
SmolVLA と同様 rename_map は要らない見込み——ただし **pi05 側で実機テストはしていない。**

## SmolVLA と比べて不利な点

1. **パラメータ数がおよそ 9 倍**（pi05 ≈ 2.3B、SmolVLA ≈ 450M）。
   80GB 前提のレシピを 20GB へ切り詰める調整（VLM 凍結・batch を絞る・
   gradient checkpointing）が要り、載るかは実測するまで分からない。
2. **libero_plus 向けの公開チェックポイントが無い。** SmolVLA の出発点
   `lerobot/smolvla_libero_plus` は既にこのコンペのデータで一度学習済みだが
   （[01_base_model.md](01_base_model.md)）、pi05 は素の `lerobot/pi05_libero_base`
   （通常の LIBERO）からしか始められない。
3. **推論速度が未測定。** 採点環境は L4 24GB・10秒/推論（PDF p.12）。
   パラメータが 9 倍になった分、同じ chunk_size でも推論が遅くなる可能性が高い。

## まだ確認していないこと

| 事柄 | 確かめ方 |
|---|---|
| 20GB GPU で `train_expert_only=true` が実際に載るか（batch size をいくつまで上げられるか） | `train_pi05.py --smoke-test` を実機で実行 |
| L4 24GB での 1 推論の時間 | 学習後、policy_server.py の pi05 版を実測 |
| pi05 の rename_map・empty_cameras の扱いが SmolVLA と同じ理屈で成立するか | 学習済みチェックポイントで軽く推論して確認 |
| PEFT/LoRA が pi05 でも使えるか | 未調査。`train_pi05.py` は使わず、公式レシピどおりフルパラメータ学習にした |
