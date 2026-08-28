"""幾何の接ぎ木を学習の経路へ差し込む（板 F25）。**接ぎ木の中身はここに無い。**

**形は `solution/geometry_graft.py` が持っている。** 学習と推論で同じ計算をしないと
学習した重みが意味を失うので、**接ぎ木の定義は 1 か所にしか置かない。**
このファイルが持つのは「lerobot の学習経路にどう割り込むか」だけである。

副作用目当てで import する（`lerobot_train_with_geometry.py` を参照）。

## 割り込む先が 2 つある理由

| 差し替えるもの | なぜ |
|---|---|
| `PI05Policy.load_state_dict` | 学習済みのチェックポイントを読み直すとき、`geometry_fuser.*` を取り分けてから残りを読む |
| `PI05Policy.from_pretrained` | **重みを読み終えたあとに**接ぎ木を生やし、取り分けた分を当てる |

**接ぐのは重みを読んだあとでなければならない。** `from_pretrained` は
`load_state_dict(..., strict=True)` を呼ぶ（modeling_pi05.py:1032）。**先に接ぎ木の層が
生えていると、土台の重みにその名前が無いので例外になる。しかもその例外は握りつぶされて
`Warning: Could not load state dict` と出るだけで、土台の重みが 1 つも載らないまま
学習が始まる**（同 1057-1060 行）。**`All keys loaded successfully!` がログに出ているかを
毎回確かめること。**

最適化器はこのあとに作られる（`lerobot_train.py` が `policy.get_optim_params()` を
呼ぶのは `make_policy` の後）ので、**接ぎ木の層はちゃんと最適化器に入る。**
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# `geometry_graft.py` は、リポジトリでは `solution/` に、学習コンテナでは
# `train_pi05.py` と同じ階層に置かれる。**どちらでも見つかるように両方を探す。**
for _candidate in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[3]):
    if (_candidate / "geometry_graft.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
else:  # pragma: no cover - 置き忘れたときにだけ通る
    raise ImportError(
        "geometry_graft.py が見つからない。リポジトリでは solution/ に、"
        "学習コンテナでは train_pi05.py と同じ階層に置くこと"
    )

import geometry_graft  # noqa: E402


def install(model_id: str | None = None, image_size: int | None = None) -> None:
    """学習の経路に接ぎ木を割り込ませる。"""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    resolved_id = model_id or os.environ.get("GEO_VGGT_MODEL", "facebook/VGGT-1B")
    resolved_size = image_size or int(os.environ.get("GEO_VGGT_IMAGE_SIZE", "252"))

    original_from_pretrained = PI05Policy.from_pretrained.__func__
    original_load_state_dict = PI05Policy.load_state_dict

    # 取り分けた接ぎ木の重み。`load_state_dict` が詰め、`from_pretrained` が使う。
    stash: dict = {}

    def patched_load_state_dict(self, state_dict, strict: bool = True, **kwargs):  # noqa: ANN001, ANN003, ANN201, FBT001, FBT002
        stash.clear()
        remainder = {}
        for key, value in state_dict.items():
            if ".geometry_fuser." in key or key.startswith("geometry_fuser."):
                # 保存側は `model.geometry_fuser.<層>`。接ぎ木から見た名前へ戻す。
                stash[key.split("geometry_fuser.", 1)[1]] = value
            else:
                remainder[key] = value
        if stash:
            print(f"[geometry] チェックポイントから接ぎ木の層 {len(stash)} 個を取り分けた",
                  flush=True)
        return original_load_state_dict(self, remainder, strict=strict, **kwargs)

    def patched_from_pretrained(cls, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        policy = original_from_pretrained(cls, *args, **kwargs)
        fuser = geometry_graft.attach(
            policy.model, image_size=resolved_size, model_id=resolved_id, pretrained=True
        )
        trainable = sum(p.numel() for p in fuser.parameters() if p.requires_grad)

        if stash:
            # **当て損なうと、学習した 2,099 万個が捨てられて乱数のまま推論することになる。**
            reference = next(policy.model.parameters())
            moved = {
                name: value.to(device=reference.device, dtype=reference.dtype)
                for name, value in stash.items()
            }
            missing, unexpected = fuser.load_state_dict(moved, strict=False)
            print(
                f"[geometry] 学習済みの接ぎ木を当てた: {len(moved)} 個 / "
                f"当たらなかった {len(missing)} 個 / 余った {len(unexpected)} 個 / "
                f"fusion_gate={float(fuser.fusion_gate):.6f}",
                flush=True,
            )
            stash.clear()
        else:
            print("[geometry] チェックポイントに接ぎ木は無かった（新規に生やす）", flush=True)

        print(
            f"[geometry] 重みを読んだあとに接いだ: {resolved_id} / "
            f"{resolved_size}x{resolved_size} / "
            f"学習するパラメータ {trainable:,} 個 / "
            f"層 {geometry_graft.trainable_parameter_names(fuser)}",
            flush=True,
        )
        return policy

    PI05Policy.load_state_dict = patched_load_state_dict
    PI05Policy.from_pretrained = classmethod(patched_from_pretrained)


install()
