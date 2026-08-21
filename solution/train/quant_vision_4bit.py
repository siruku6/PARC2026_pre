#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""量子化実験・候補 C2: pi05 の vision encoder（+ その射影層）だけを bitsandbytes の
4bit（`bnb.nn.Linear4bit`）に量子化する。

lerobot.scripts.lerobot_train.main() を呼ぶ前に、副作用目当てでこのモジュールを import する
こと（lerobot_train_with_quant.py 参照）。LeRobot 本体（`work/lerobot-src` を v0.6.0 として
2026-08-20 に確認）には量子化を差し込むプラグイン点が無いため、ここでは
`lerobot.policies.pi05.modeling_pi05.PaliGemmaWithExpertModel.__init__` を直接
モンキーパッチする —— 元の `__init__` を呼んだ直後（重みロード前）に、
vision_tower / multi_modal_projector 配下の `nn.Linear` を `bnb.nn.Linear4bit` へ
差し替える。

## 範囲を vision_tower + multi_modal_projector だけに絞っている理由

`compute_layer_complete`（modeling_pi05.py 236-305 行、2026-08-20 確認）は
`self.paligemma`（gemma_2b 言語バックボーン）と `self.gemma_expert` の transformer 層の中で、
`layer.self_attn.o_proj.weight.dtype` や `layer.mlp.up_proj.weight.dtype` を直接見て分岐する
（例: `if att_output.dtype != layer.self_attn.o_proj.weight.dtype:`）。`Linear4bit` の重み
（`bnb.nn.Params4bit`）の `.dtype` はパック後のストレージ dtype を返すため、これらの分岐が
壊れる。**そのため両方の言語バックボーン（gemma_2b・gemma_expert）には手を出さない** ——
それをやるなら本物のソース修正が要り、モンキーパッチの範囲外。`embed_image`
（446-455 行）が呼ぶ `self.paligemma.model.get_image_features(image)` はこの手の
dtype 分岐を持たないため、vision 側だけを量子化する分にはこの衝突は起きない
（2026-08-20 に該当行を確認済み）。

## 既知のハザード: 重みロードより先に `.to(device)` が呼ばれる

`PI05Policy.from_pretrained`（943-1059 行）は
`model = cls(config, **kwargs)`（ここで上記パッチが効く）→
`model.load_state_dict(remapped_state_dict, strict=strict)` の順で動く。ところが
`PI05Policy.__init__`（917-941 行）は、その `cls(config, **kwargs)` の中で
`self.model.to(config.device)`（939 行）を**重みロードより前に**呼んでいる。

`bitsandbytes==0.49.2`（Dockerfile.pi05 で固定、2026-08-20 に wheel を展開して確認）の
`Params4bit.to()`（bitsandbytes/nn/modules.py 341-361 行）は
「meta 以外のデバイスへの移動で、かつまだ量子化していなければその場で量子化する」
という設計（`if device is not None and device.type != "meta" and not self.bnb_quantized:
return self._quantize(device)`）。素朴に `bnb.nn.Linear4bit` へ差し替えるだけだと、
上記の `self.model.to(config.device)` がランダム初期化のゴミ重みを量子化してしまい、
その後の `load_state_dict` はパック済み 4bit blob と平のチェックポイント重みの
形状が合わず `size mismatch` で落ちる（`strict=True` が既定のため）。

これは「ロード前に量子化してしまう」という意味では大きな声で落ちるので実害は無いが、
それでは C2 は `--policy.pretrained_path` を使う本番実行（train_pi05.py は常にこれを渡す。
2026-08-20 確認）で毎回確実に落ちるだけになり、A/B の比較材料にならない。そこで
`_LazyQuantLinear4bit` で「本当の重みが `load_state_dict` で入るまで量子化を遅らせる」
仕掛けを足してある（下記クラス docstring 参照）。この仕掛けは
`PaliGemmaWithExpertModel.__init__` を hook する中だけで完結しており、
`PI05Policy.from_pretrained`/`__init__` 自体には一切手を出していない
（元の調査メモが「`from_pretrained` を後から hook するな、`__init__` を hook しろ」と
言っていた制約はそのまま守っている）。

## スモークテストの通し方

    python train_pi05.py --vision-quant-mode bnb4bit --smoke-test

- **成功時**: いつも通り 20 ステップ回り、ログに
  `[quant_vision_4bit] replaced N nn.Linear layers with bnb.nn.Linear4bit`（N>0）が出て、
  OOM や dtype エラーなしで完走する。
- **失敗時（想定される代表例）**:
  - `bitsandbytes` が import できない環境 → import 時点で
    `RuntimeError("quant_vision_4bit: bitsandbytes が import できません...")`。
  - `vision_tower`/`multi_modal_projector` の属性パスが v0.6.0 から変わっていた →
    `_quantize_vision_tower` 内で `RuntimeError("quant_vision_4bit: ... 属性パス ... が
    見つかりません")`。
  - 何らかの理由で 1 層も差し替わらなかった →
    `RuntimeError("quant_vision_4bit: ... 1つも見つかりませんでした")`。
  - `_LazyQuantLinear4bit.forward()` が量子化確定前に呼ばれた（チェックポイントを使わない
    フルスクラッチ学習など、この候補が想定していない使い方）→
    `RuntimeError("quant_vision_4bit: ... 量子化が確定していません")`。
  いずれも「なぜ落ちたか」がその場のメッセージだけで分かる設計にしてある。

## 範囲外（この候補ではやらないこと）

- gemma_2b（`self.paligemma` の言語バックボーン）・`self.gemma_expert` の量子化は上記の
  dtype 分岐ハザードのため対象外。
- 4bit の重みへの勾配学習（QLoRA 的な仕組み）は用意していない —— `train_pi05.py` の既定
  （`--freeze-vision-encoder=true --train-expert-only=true`）どおり vision encoder ごと
  凍結して使う前提（work/quant_sweep/CANDIDATES.md の C2 の説明のとおり）。
"""

import logging

import torch
from torch import nn

try:
    import bitsandbytes as bnb
except ImportError as e:
    raise RuntimeError(
        "quant_vision_4bit: bitsandbytes が import できません。"
        "Dockerfile.pi05 は bitsandbytes==0.49.2 をインストールしている前提です。"
        f"（元の ImportError: {e}）"
    ) from e

from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel

log = logging.getLogger(__name__)


class _LazyQuantLinear4bit(bnb.nn.Linear4bit):
    """`bnb.nn.Linear4bit` だが、`load_state_dict` で本物の重みが入るまで量子化を遅らせる。

    理由はモジュール docstring の「既知のハザード」参照。要点だけ書くと:
    `Params4bit.to(device)` は「まだ量子化していない状態で meta 以外のデバイスへ移動する」
    ことをトリガーに実際の量子化（`_quantize()`）を行う設計になっている
    （bitsandbytes==0.49.2、nn/modules.py 341-361 行）。ところが pi05 側は
    `PI05Policy.__init__` が重みロードより先に `self.model.to(config.device)` を呼ぶため、
    素の `Linear4bit` だとランダム初期化のゴミ重みをこの時点で量子化してしまう。

    対策: 生成直後に `weight.bnb_quantized = True` という嘘をついて、この最初の
    `.to(device)` では「量子化はもう済んでいる」体で「ただの device 移動」だけをさせる
    （同ファイルの `to()` 実装の else 分岐 —— データはそのまま正しく device 移動されるだけで
    壊れない）。その後 `load_state_dict` で本物のチェックポイント値が `weight.data` に
    コピーされたら（形状はまだ平の float のままなので通常の `copy_` で問題なく入る）、
    `register_load_state_dict_post_hook` で登録した `_finalize_quant` が嘘を取り消して
    改めて `.to(device)` を呼び、今度は本物の重みを実際に量子化する。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.bnb_quantized = True  # 嘘。本当に量子化が終わったら _finalize_quant が戻す
        self._quant_finalized = False
        self.register_load_state_dict_post_hook(self._finalize_quant)

    def _finalize_quant(self, module, incompatible_keys):
        del module, incompatible_keys  # hook のシグネチャ上必要なだけで未使用
        # ここに来る時点で self.weight は既に正しい device 上にある（構築直後の
        # self.model.to(config.device) で「嘘」のおかげで device 移動だけは済んでいるため）。
        # 同じ device へもう一度 .to() することで、今度こそ本物の重みを量子化させる。
        target_device = self.weight.device
        self.weight.bnb_quantized = False  # 嘘を撤回
        self.weight.to(target_device)  # Params4bit._quantize() が in-place で発火する
        self._quant_finalized = True

    def forward(self, x: torch.Tensor):
        if not self._quant_finalized:
            # load_state_dict を一度も通さずに forward が呼ばれた == この候補が前提にしている
            # 「事前学習済みチェックポイントからのロード」を経ていない使い方。フルスクラッチ
            # 学習はこの monkeypatch では対応していない（train_pi05.py は常に
            # --policy.pretrained_path を渡すため、この候補の実運用では起こらない想定）。
            raise RuntimeError(
                "quant_vision_4bit: vision_tower/multi_modal_projector の Linear4bit 層で、"
                "重みが一度も load_state_dict でロードされないまま forward() が呼ばれました "
                "（量子化が確定していません）。この monkeypatch は事前学習済みチェックポイントを"
                "ロードする学習だけを想定しています（train_pi05.py は常に "
                "--policy.pretrained_path を渡す）。フルスクラッチ学習は対応していません。"
            )
        return super().forward(x)


def _replace_linear_with_4bit(module: nn.Module, prefix: str) -> int:
    """`module` 配下の `nn.Linear` を再帰的に `_LazyQuantLinear4bit` へ in-place で差し替える。

    戻り値は差し替えた層数 —— 呼び出し側で「0 件なら何もしていない」を検出するのに使う
    （黙って no-op になるのを避けるため）。
    """
    replaced = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}"
        if isinstance(child, nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
            new_layer = _LazyQuantLinear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
            )
            # child の重みはここでコピーしない: PI05Policy.from_pretrained がこの直後に
            # load_state_dict でチェックポイントの値を流し込む。ここでコピーしても
            # どうせ上書きされるだけなので意味が無い（replace-before-load がこの実装の要）。
            setattr(module, name, new_layer)
            replaced += 1
        else:
            replaced += _replace_linear_with_4bit(child, full_name)
    return replaced


def _quantize_vision_tower(model: PaliGemmaWithExpertModel) -> None:
    """vision_tower + multi_modal_projector 配下だけを 4bit 化する。範囲を絞る理由は
    モジュール docstring の「範囲を vision_tower + multi_modal_projector だけに絞っている
    理由」を参照。gemma_2b（言語バックボーン）・gemma_expert には触れない。
    """
    try:
        vision_tower = model.paligemma.model.vision_tower
        multi_modal_projector = model.paligemma.model.multi_modal_projector
    except AttributeError as e:
        raise RuntimeError(
            "quant_vision_4bit: PaliGemmaWithExpertModel に期待する属性パス "
            "self.paligemma.model.vision_tower / self.paligemma.model.multi_modal_projector が"
            "見つかりません。lerobot の pi05 実装がこの monkeypatch を書いた時点（v0.6.0、"
            "2026-08-20 確認）から変わっている可能性があります。黙って量子化をスキップせず、"
            "ここで属性パスを見直してください。"
        ) from e

    n_replaced = _replace_linear_with_4bit(vision_tower, "vision_tower")
    n_replaced += _replace_linear_with_4bit(multi_modal_projector, "multi_modal_projector")
    if n_replaced == 0:
        raise RuntimeError(
            "quant_vision_4bit: vision_tower + multi_modal_projector を走査しましたが、"
            "置き換え対象の nn.Linear が1つも見つかりませんでした。黙って何もしないままにせず、"
            "ここで落としています。"
        )
    log.info("[quant_vision_4bit] replaced %d nn.Linear layers with bnb.nn.Linear4bit", n_replaced)

    # _set_requires_grad()（元の __init__ の最後で既に一度呼ばれている）は
    # freeze_vision_encoder / train_expert_only の凍結を、上で捨てた元の nn.Linear に対して
    # 適用済み。新しく差し替えた Linear4bit はそれぞれ requires_grad の既定
    # （weight=False 固定、bias=True）を bnb 側の都合で持っているだけなので、
    # このポリシーが指定した凍結方針を改めて掛け直す。
    model._set_requires_grad()


_original_init = PaliGemmaWithExpertModel.__init__


def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    _quantize_vision_tower(self)


PaliGemmaWithExpertModel.__init__ = _patched_init
