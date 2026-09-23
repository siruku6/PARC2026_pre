# 33. 設定に焼かれた Hub の名前は、通信の無い採点環境で 0 点になる

**2026-09-17、本選 1 の締切当日に、MolmoAct2 の提出物（版 23.0.0）を Track3 の枠へ出して 0 点になった。**
**方策サーバーが起動できず、8 本とも走らなかった。**
**手元では同じ zip の中身だけでリハーサルを通していたのに、である。**

この文書は、その 1 件から取れる一般則を残す。**要点は「手元のリハーサルが本番の経路を通っていなかった」ことで、
落ちた箇所そのものではない。**

## 項目一覧

| 節 | 一行要約 |
|---|---|
| [1. 何が落ちたか](#1-何が落ちたか) | 設定に書かれた Hub の名前を起動時に取りに行き、`OfflineModeIsEnabled` |
| [2. なぜ手元で気づけなかったか](#2-なぜ手元で気づけなかったか) | 評価スクリプトが HF のキャッシュをコンテナへ渡していた |
| [3. 一般則](#3-一般則) | **「ローカルに無ければ Hub」の分岐は、採点環境では必ず後者へ落ちる** |
| [4. どう直すか](#4-どう直すか) | 同梱してローカルのパスを指させる |
| [5. どう確かめるか](#5-どう確かめるか) | 空のキャッシュ ＋ `HF_HUB_OFFLINE=1` でリハーサルする |

---

## 1. 何が落ちたか

**採点環境は `HF_HUB_OFFLINE=1` を立てている。**そこで起動時にこうなった。

```
huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach
https://huggingface.co/api/models/allenai/MolmoAct2-FAST-Tokenizer/revision/main:
offline mode is enabled.
```

**取りに行った先は、重みに同梱の `policy_preprocessor.json` に焼かれていた文字列である。**

```json
{"discrete_action_tokenizer": "allenai/MolmoAct2-FAST-Tokenizer"}
```

**この 1 行のために 11.5 GB の提出物が 1 歩も動かず、リーダーボードの枠を 1 つ空振りさせた。**

解決の経路は、同梱 lerobot の次の関数にある。

```python
def _resolve_tokenizer_location(tokenizer_path, ...):
    local_path = Path(str(tokenizer_path)).expanduser()
    if local_path.exists():
        return str(local_path)          # ← ローカルにあればこちら
    return snapshot_download(repo_id=str(tokenizer_path), ...)   # ← 無ければ Hub
```

**同じ文字列が「ローカルのパス」としても「Hub のリポジトリ名」としても解釈される。**
手元には HF のキャッシュがあるので後者が成功し、採点環境では失敗する。

## 2. なぜ手元で気づけなかったか

**リハーサルは通っていた。**版 20.0.0 で CUDA OOM を捕まえたのと同じ仕組みで、
**zip の中身だけを取り出して起動し、2 行 × 1 本を走らせる**手順である。

**しかしその手順は、HF のキャッシュをコンテナへ渡していた。**

```bash
docker run ... \
    -v "$HOME/.cache/huggingface:/hf_home" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
```

**この 2 行があるかぎり、通信の有無を問う経路は一度も通らない。**
リハーサルは「zip の中身だけで動くか」を確かめているつもりで、
**実際には「zip の中身 ＋ この台のキャッシュで動くか」を確かめていた。**

**同じ穴は他にもありうる。**採点環境と手元で違うのは通信だけではない——
`$HOME` の中身、環境変数、書き込める場所、GPU の VRAM。
**リハーサルで再現していない差は、すべて同じ形の落とし穴になる。**

## 3. 一般則

**「ローカルに無ければ Hub から落とす」という分岐は、採点環境では必ず後者へ落ちる。**
**だから、その分岐が 1 つでも残っていれば起動に失敗する。**

**そして、その分岐はコードだけでなく設定ファイルの中にもある。**
上の件では `policy_preprocessor.json` に文字列として書かれていて、
**`policy_server.py` を読んでも見つからない。**

| 探す場所 | 例 |
|---|---|
| コード | `snapshot_download(...)`・`hf_hub_download(...)`・`AutoTokenizer.from_pretrained("org/name")` |
| **重みに同梱の設定** | `policy_preprocessor.json`・`config.json`・`processor_config.json` に書かれた `org/name` の形の文字列 |

**他人の重みを使うときほど危ない**——設定を書いたのは自分ではないからである。

## 4. どう直すか

**現物を同梱して、設定がローカルのパスを指すようにする。**

1. 取りに行っているものを落として、重みのディレクトリに入れる
   （この件では 4 ファイル・338 KB。`model_weights/fast_action_tokenizer/`）。
2. `policy_server.py` の processor override で、そのパスを渡す。

```python
_fast_tokenizer = weights / "fast_action_tokenizer"
if _fast_tokenizer.is_dir():
    _preprocessor_overrides["molmoact2_pack_inputs"]["discrete_action_tokenizer"] = str(_fast_tokenizer)
```

**`is_dir()` で挟むのは、同梱していない重みでも壊れないようにするためである。**

## 5. どう確かめるか

**空のディレクトリを HF のキャッシュとして渡し、`HF_HUB_OFFLINE=1` を立ててリハーサルする。**
これで採点環境と同じ経路を通る。

```bash
export PARC_FAKE_HF=$PWD/work/final_eval/_fake_hf_home
rm -rf "$PARC_FAKE_HF"; mkdir -p "$PARC_FAKE_HF"
# solution/tools/eval/_run_molmoact2_offline.sh は、この変数を
# $HOME/.cache/huggingface の代わりにマウントし、HF_HUB_OFFLINE=1 を立てる
MUJOCO_GL=egl TRACK_EPISODES=1 MAX_STEPS=150 TASKS_CSV=work/final_eval/_smoke2.csv \
    WEIGHTS=$PWD/work/artifacts/<重み> SERVER_PY=$R/policy_server.py \
    bash solution/tools/eval/_run_molmoact2_offline.sh track2
```

**版 24.0.0 はこれを通した**（2 行 × 1 本・成功率 100%・衝突 0・
`work/final_eval/0917_233533_track2_molmoact2_stage0_zip24_offline`）。

**提出物のリハーサルは、通信を切った側だけを正とする。**
キャッシュを渡すリハーサルは、通ったことの意味が無い。
