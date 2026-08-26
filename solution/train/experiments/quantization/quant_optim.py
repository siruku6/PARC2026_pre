"""量子化実験・候補 C1: LeRobot の OptimizerConfig レジストリに "adamw_bnb8bit" を登録する。

lerobot.scripts.lerobot_train.main() を呼ぶ前に、副作用目当てでこのモジュールを import する
こと（lerobot_train_with_quant.py 参照）。draccus の CLI パーサーがこの選択肢名を知るには、
import されて @OptimizerConfig.register_subclass が実行されている必要がある。

pi05 は既定で use_policy_training_preset=true になっており、PI05Config.get_optimizer_preset()
（lerobot.policies.pi05.configuration_pi05）が AdamWConfig を決め打ちで返すため、この選択肢を
登録するだけでは効かない。呼び出し側（train_pi05.py）で以下を必ず一緒に渡すこと:

    --use_policy_training_preset=false
    --optimizer.type=adamw_bnb8bit
    --optimizer.lr=<学習率>
    --scheduler.type=cosine_decay_with_warmup
    --scheduler.peak_lr=<学習率>
    --scheduler.decay_lr=2.5e-6
    --scheduler.num_warmup_steps=1000
    --scheduler.num_decay_steps=30000

decay_lr/num_warmup_steps/num_decay_steps の値は PI05Config の実際のデフォルト
（configuration_pi05.py, 2026-08-20 確認）をそのまま複製したもの — baseline と条件を
オプティマイザの実装以外すべて揃えるための値であり、変える場合はそちらの値も合わせて変えること。

bitsandbytes の import は build() まで遅延させている。そのため、未インストールの環境でも
このモジュール自体の import（＝レジストリへの登録）は失敗しない —— 実際に
"adamw_bnb8bit" が選ばれない限り（--optimizer-mode default のまま）は無害。

## チェックポイント保存時のバグと修正（2026-08-20、実機の --smoke-test で発見）

bitsandbytes の 8bit optimizer（`Optimizer2State` 系）は `state[p]["step"]` を素の
Python `int` で持つ。一方 torch 標準の Adam/AdamW は 2.x 以降これを 0 次元 tensor で
持つ（capturable/compile 対応のため）。LeRobot の `save_optimizer_state`
（`lerobot/optim/optimizers.py`）は `optimizer.state_dict()` をそのまま
`safetensors.torch.save_file` に渡すため、int のままだと
`ValueError: Key 'state/.../step' is invalid, expected torch.Tensor but received int`
で 20 ステップ目のチェックポイント保存が毎回落ちる（学習自体は正常に進み、
loss も下がる — 保存でだけ落ちる）。`_patch_state_dict_for_safetensors` で
`state_dict()` をラップし、`step` を int から 0 次元 tensor に変換してから返す。

**2つ目のバグ（同じく実機で発見、上の修正の直後に踏んだ）**: bitsandbytes の
8bit optimizer は量子化テーブル `__bnb_optimizer_quant_state__/qmap1` /
`qmap2`（8bit への丸め込み方を決める 256 要素の対応表）を、**全パラメータで
同じ tensor を使い回す**（パラメータごとに複製しない、意図的な省メモリ設計）。
`safetensors.torch.save_file` は「複数のキーが同じメモリを指している」状態を
`RuntimeError`（"Some tensors share memory..."）で拒否する——保存対象を
key ごとに独立した実体だと仮定しているため。`_patch_state_dict_for_safetensors`
で、2 回目以降に同じ `data_ptr()` が出てきた tensor をその場で `.clone()` し、
実体を独立させてから返す（qmap は 256 要素の小さな表なので複製コストは無視できる）。
"""
from dataclasses import asdict, dataclass

import torch

from lerobot.optim.optimizers import OptimizerConfig, OptimizerParams


def _declone_shared_tensors(obj, seen_ptrs: set):
    """obj の中の tensor のうち、同じメモリ（data_ptr）が2回目以降に出てきたものを
    .clone() して独立させる。dict/list は再帰的に辿る。他の値はそのまま返す。

    safetensors.torch.save_file は共有メモリの tensor を拒否するため
    （モジュール docstring の「2つ目のバグ」参照）、保存直前にここで解消する。
    """
    if isinstance(obj, torch.Tensor):
        ptr = obj.data_ptr()
        if ptr in seen_ptrs:
            return obj.clone()
        seen_ptrs.add(ptr)
        return obj
    if isinstance(obj, dict):
        return {key: _declone_shared_tensors(value, seen_ptrs) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_declone_shared_tensors(value, seen_ptrs) for value in obj]
    return obj


def _patch_state_dict_for_safetensors(optimizer: torch.optim.Optimizer) -> None:
    """optimizer.state_dict() を、safetensors 保存が通る形に変換してから返すよう、
    インスタンス単位でラップする（モジュール docstring の2つのバグ参照）。

    1. 各パラメータの "step" を int から 0 次元 tensor に変換する。
    2. 量子化テーブル（qmap1/qmap2 等）で複数キーが同じメモリを共有している状態を
       クローンで解消する。

    load_state_dict() 側は変換後の値をそのまま state に書き戻すだけで、以降
    bnb 内部が `state["step"] += 1` を呼んでも 0 次元 tensor のまま演算できるし、
    クローンされた qmap も値としては元と同じなので、resume 後の挙動に支障は無い。
    """
    original_state_dict = optimizer.state_dict

    def _patched_state_dict():
        state_dict = original_state_dict()
        for param_state in state_dict.get("state", {}).values():
            step = param_state.get("step")
            if isinstance(step, int):
                param_state["step"] = torch.tensor(step)
        state_dict["state"] = _declone_shared_tensors(state_dict.get("state", {}), set())
        return state_dict

    optimizer.state_dict = _patched_state_dict


@OptimizerConfig.register_subclass("adamw_bnb8bit")
@dataclass
class AdamWBnb8bitConfig(OptimizerConfig):
    # 既定値は lerobot.policies.pi05.configuration_pi05.PI05Config.get_optimizer_preset()
    # と完全に一致させてある（同ファイル 91-95 行、2026-08-20 確認）。
    # --optimizer-mode bnb8bit と baseline との違いが「オプティマイザの実装だけ」に
    # なるようにするため。
    lr: float = 2.5e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        import bitsandbytes as bnb  # 遅延 import。理由はモジュール docstring 参照

        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        optimizer = bnb.optim.AdamW8bit(params, **kwargs)
        _patch_state_dict_for_safetensors(optimizer)
        return optimizer
