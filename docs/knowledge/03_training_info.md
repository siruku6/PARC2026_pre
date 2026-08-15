# 03. 学習情報（レポート必須欄 3）

**出典は `train_config.json` 2 つと、データセットの `meta/info.json` である。**
どれも学習が実際に書き出したもので、このリポジトリの `.md` は経由していない。

## 学習データ

| 欄 | 値 |
|---|---|
| データセット | **Hugging Face `lerobot/libero_plus`**（LIBERO-plus を LeRobot 形式にしたもの） |
| revision | **`f3f49f426d75030177b18778374005bc12ccd588`** |
| **エピソード数** | **14,347** |
| **フレーム数** | **2,238,036** |
| **タスク数** | **40** |
| ロボット | panda ／ 20 fps |
| 観測 | 前方カメラ・手首カメラの 2 視点 ＋ 関節状態 |
| 独自に収集・生成したデータ | **無し。**公開データセットのみを使用 |

## 学習方法 —— 2 段構成

**同じデータで 2 回学習した。1 段目は行動側だけ、2 段目は画像を見る側まで解いた。**

| | **段 1** | **段 2（提出したもの）** |
|---|---|---|
| 出発点 | `lerobot/pi05_libero_base`（rev `a217bfd3…`） | **段 1 の 24,000 歩** |
| 区分 | **Full fine-tuning（LoRA・adapter は不使用）** | **Full fine-tuning** |
| **学習対象** | **行動側（action expert）のみ**。`freeze_vision_encoder=true` / `train_expert_only=true` | **全パラメータ**。`freeze_vision_encoder=false` / `train_expert_only=false` |
| **更新したパラメータ数** | **693,422,112（693M）** | **4,143,404,816（4.14B）＝全部** |
| モデル全体のパラメータ数 | 4,143,404,816（両段とも同じ） | 同左 |
| **学習ステップ数** | **24,000** | **3,000** |
| batch size | 16 | **64** |
| 引いたフレーム数 | 384,000（**0.172 epoch**） | 192,000（**0.086 epoch**） |
| 最適化 | AdamW（betas 0.9/0.95・weight_decay 0.01・grad_clip 1.0・eps 1e-8） | 同じ |
| 学習率 | **2.5e-5 → 2.5e-6**（cosine decay ＋ warmup 1,000 歩） | **5e-6 → 2.5e-6**（同じ形） |
| seed | 42 | 42 |
| 計算資源 | ローカル H100-20C（20 GB）・**9 時間 31 分** | **RunPod A100 80GB PCIe・8 時間 33 分** |

**2 段合わせて 576,000 フレーム＝データ全体の 0.257 epoch しか引いていない。**

**LoRA・adapter は使っていない。** どちらの段も重みを直接更新する Full fine-tuning
である（初期に SmolVLA + LoRA も試したが、本番スコアが悪化したため採らなかった）。

## 段 2 で loss が動いた

**段 1 は 18,000 歩以降で頭打ちだったが、画像を見る側を解いた段 2 で再び下がった。**

| 段 1 の歩数 | loss |
|---|---|
| 18,001〜20,000 | 0.2375 |
| 20,001〜22,000 | 0.2352 |
| **22,001〜24,000** | **0.2360（上がっている＝頭打ち）** |

| 段 2 の歩数 | loss | grad norm |
|---|---|---|
| 1〜500 | 0.2294 | 0.692 |
| **2,501〜3,000** | **0.2100** | **0.640** |

**0.236 → 0.210（−11%）で、3,000 歩の時点でまだ下がり続けていた。**
**残っていた誤差は、凍らせていた側にあったと読める。**

> **loss は本番スコアの予想には使えない。**手元の成功率でさえ本番の順位を
> 付けられないことが実測で分かっている（[07](07_local_eval_cannot_rank.md)）。
> loss はそれよりさらに遠い。**「手本に近づいた」ことしか言っていない。**

## 確かめ直すコマンド

```bash
# 更新したパラメータ数（学習ログが起動時に出している）
grep -E "num_learnable_params|num_total_params" \
    work/train/pi05_full_local.log work/train/pi05_full_runpod.log

# 段 1 / 段 2 の設定
python3 -c "
import json
for name, path in (('段1','work/artifacts/model_weights_pi05_024000/train_config.json'),
                   ('段2','work/train/pi05_full_runpod/003000/train_config.json')):
    c = json.load(open(path)); p = c['policy']
    print(name, c['steps'], c['batch_size'], c['dataset']['repo_id'],
          c['dataset']['revision'][:8],
          p['freeze_vision_encoder'], p['train_expert_only'], c['scheduler'])"

# データセットの実数
.venv/bin/python3 -c "
from huggingface_hub import hf_hub_download
import json, pathlib
p = hf_hub_download('lerobot/libero_plus','meta/info.json', repo_type='dataset',
                    revision='f3f49f426d75030177b18778374005bc12ccd588')
i = json.loads(pathlib.Path(p).read_text())
print({k: i[k] for k in ('total_episodes','total_frames','total_tasks','fps','robot_type')})"
```

## 未取得

| 欄 | 状態 |
|---|---|
| — | **無し。必須欄はすべて 1 次資料から埋まった** |
