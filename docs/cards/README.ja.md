# PARC2026 で公開した模型とデータの見取り図

*English: [README.md](README.md)*

**ロボット操作のコンペ PARC2026（LIBERO 課題）のために作った模型とデータである。**
**何を使えばよいか・何を信じてよいかを、ここだけで決められるようにする。**

## 項目一覧

| 節 | 一行要約 |
|---|---|
| [1. どれを使えばよいか](#1-どれを使えばよいか) | 目的別に 1 つだけ選ぶ |
| [2. 模型の系統](#2-模型の系統) | どれがどれの子か・採点された点 |
| [3. データの系統](#3-データの系統) | 撮り直した実演がどこにあるか |
| [4. 保管の方針](#4-保管の方針) | **何を残し、何を消してよいか。その判断の規則** |
| [5. 踏んだ落とし穴](#5-踏んだ落とし穴) | 拾った人が同じ穴に落ちないための一覧 |

---

## 1. どれを使えばよいか

| やりたいこと | 使うもの |
|---|---|
| **LIBERO で動く π0.5 が 1 つ欲しい** | [`pi05_bowl5_full`](https://huggingface.co/siruku6/pi05_bowl5_full) の `005500`。**採点で確かめた中でいちばん良い**（Track1 0.498） |
| **MolmoAct2 の系統を試したい** | [`molmoact2_lora_v1`](https://huggingface.co/siruku6/molmoact2_lora_v1) の `024000`。**測定があるのはこれだけ** |
| **`libero_90` も含めて広く解かせたい** | [`pi05_l90clean_from_bowl5`](https://huggingface.co/siruku6/pi05_l90clean_from_bowl5) の `008000`。ただし 4 節の注意を読むこと |
| **自分でデータから学習したい** | [`libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose)（40 課題）か [`libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean)（111 課題） |

**素の `allenai/MolmoAct2-LIBERO-LeRobot` が、私たちのどの模型よりも採点では高かった**
（Track1 0.530・Track2 0.424）。**土台として出発するなら、まずそこから始めるのが妥当である。**

## 2. 模型の系統

```
lerobot/pi05_libero_base（Gemma Terms of Use）
  ├─ pi05_full_runpod ─ pi05_trial1500 ─ (ladder2500)     Track1 0.492 / 0.475
  │                       └─ pi05_t3rev_v2                 逆向き。Track3 0.000
  └─ pi05_bowl5_full                                       Track1 0.498 ← π0.5 の最良
        └─ pi05_l90clean_from_bowl5                        有効な採点が無い（5 節）

allenai/MolmoAct2-LIBERO-LeRobot（Apache 2.0）
  │     素のまま                                            Track1 0.530 / Track2 0.424 ← 全体の最良
  └─ molmoact2_lora_v1 (024000) ─ molmoact2_lora_v1_40k (040000)
                                   採点なし・評価なし
```

**採点は 8 本のエピソードで行われる。**1 本の出入りが 0.06 前後動くので、
**0.05 未満の差は読めない。**この表の差の多くはその内側にある。

個別のカード（日英）はこの下の各ディレクトリにある。

## 3. データの系統

**撮り直した実演が、作り直せない唯一のものである。**どのデータセットがそれを持っているかで並べる。

| データセット | 撮り直した実演を持っているか | 中身 |
|---|---|---|
| [`libero_plus_bowl5`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5) | **持っている**——丼 5 課題の入れ替え | 14,085 本・40 課題 |
| [`libero_plus_bowl5_initpose`](https://huggingface.co/datasets/siruku6/libero_plus_bowl5_initpose) | **持っている**——初期姿勢を振った 4,275 本 | 16,820 本・40 課題 |
| [`libero_combined_bowl5_initpose_l90clean`](https://huggingface.co/datasets/siruku6/libero_combined_bowl5_initpose_l90clean) | **持っている**——`libero_90` の綺麗な 250 本 | 21,642 本・111 課題 |
| [`jaist_mix_v2`](https://huggingface.co/datasets/siruku6/jaist_mix_v2) | **混ぜたものである**——逆向き 397 本を 8 倍複製して配布データと混合 | 6,176 本 |

**組み上げの台帳とスクリプトは
[`solution/tools/regen/`](../../solution/tools/regen/) にある**
（残す本・外す本の番号、接触の判定、初期姿勢の振り方）。
**混ぜたデータセットは、部品とこの台帳から作り直せる。**

## 4. 保管の方針

**データセットは 3 つに分かれる。消してよいかは、どれに当たるかで決まる。**

| 種類 | 失ったら | 方針 |
|---|---|---|
| **配布元から落とせるもの**（LIBERO・LIBERO-plus の元データ） | 落とし直せばよい | **置かない** |
| **運営が配った統合データ**（`libero_combined_20hz`・21 GB） | **落とし直せる。**HF には無いが Google Drive で配られている（[通知](../materials/notifications/260914_1000_location_of_libero_combined_20hz.txt)） | **置かなくてよい。**ただし取るのに時間がかかるので、使う台には残す |
| **混ぜたもの**（部品 ＋ 台帳 ＋ スクリプトから決まる） | 台帳があれば作り直せる | **公開した模型が指しているものだけ残す** |
| **撮り直した実演そのもの** | **二度と同じものにならない**（台本の方策を回し直す時間が要る） | **必ず残す。部品のまま、別々に** |

**「別々に持って、混ぜ方をドキュメントに書く」が正しい。**理由は 2 つある。

1. **混ぜると、どの部品が効いたかを測れなくなる。**
   私たちが「課題を 40 から 111 に広げたら下がった」（0.4055 → 0.3515）と知れたのは、
   **両方が別々に存在したからである。**混合しか残っていなければ、この比較は永久にできない。
2. **混ぜたものは容量を重複して食う。**`jaist_mix_v2` の 6,176 本のうち 3,176 本は、
   **同じ 397 本の 8 つの複製である。**部品で持てば 8 分の 1 で済み、情報は増える。

**ただし混合を 1 つだけ残す条件がある**——**公開した模型の学習データは、引けなければならない。**
だから `libero_combined_bowl5_initpose_l90clean` は残す。
`molmoact2_lora_v1` と `molmoact2_lora_v1_40k` がそれを指している。

**2026-09-18 に、この条件は外れた。**`libero_90` の 250 本を置き換えた段は
2026-09-16 に手で実施したもので、スクリプトにも台帳にも無かったが、
**台帳に書き出し、別の台（Mac）で部品から組み直して、21,642 本すべての行動列が
一致することを確かめた**（[howto/20 の 5 節](../howto/20_rebuild_l90clean_elsewhere.md#5-実測)）。

```
行動列が一致した本: 21642 / 21642
```

**したがって `libero_combined_bowl5_initpose_l90clean` は消してよい。**
ただし**公開した模型が指している間は残す**——上の条件のほうは生きている。

### いま残っているものへの当てはめ

| データセット | 判断 | なぜ |
|---|---|---|
| `libero_plus_bowl5` | **残す** | 丼の撮り直しを持つ唯一の現物 |
| `libero_plus_bowl5_initpose` | **残す** | 初期姿勢の撮り直しを持つ唯一の現物 |
| `libero_combined_bowl5_initpose_l90clean` | **残す**（公開にする） | 公開した MolmoAct2 が指しているため。**再現できないという理由は 2026-09-18 に消えた**（部品から組み直して 21,642/21,642 一致） |
| `jaist_mix_v2`（6.3 GB） | **部品に置き換えることを勧める** | **部品の `reverse_lerobot_v2`（397 本）は HF に上がっていない。**先にそれを上げ、混ぜ方（`libero_plus` を 75 本/課題に間引く ＋ 逆向きを 8 倍）を書けば、6.3 GB の混合は要らなくなる |
| `lerobot_init_pose_all`（1.4 GB） | **消してよい** | 9/12 の初期姿勢 1,250 本。**素の見た目だけで、一度も学習に使っていない。**`libero_plus_bowl5_initpose` の 4,275 本に置き換わっている |

## 5. 踏んだ落とし穴

**どれも実際に点を失ったものである。**

| 落とし穴 | 何が起きたか |
|---|---|
| **π0.5 の `vlm/` に別モデルのトークナイザを入れた** | SmolVLM2（`GPT2Tokenizer`）を Gemma の代わりに入れた提出が **0.498 → 0.075**。**方策は動くが、指示に従わない。**`vlm/` は 4 ファイル・`tokenizer_class: GemmaTokenizer` であること。`merges.txt` があったら間違っている |
| **設定に書かれた Hub の名前を、通信の無い環境で取りに行く** | MolmoAct2 の `discrete_action_tokenizer` が `allenai/MolmoAct2-FAST-Tokenizer` を指し、**採点環境で起動に失敗して 0 点。**同梱してローカルのパスを指させること |
| **手元のリハーサルが本番の経路を通っていない** | 上の 2 つとも、手元では通っていた——**HF のキャッシュをコンテナに渡していたから。**通信を切った側だけを正とすること |
| **続きの学習は足し算にならない** | 0.498 の重みに 8,000 歩足したら、`libero_90` で上がって狙っている分布では同等になった。**足したぶんだけ良くなるわけではない** |
| **手元で差が無いことは、入れ替えてよいという意味ではない** | 差が無い候補と入れ替えると、**上振れが無く下振れだけが開く** |

詳しくは [`docs/knowledge/`](../knowledge/) にある。
