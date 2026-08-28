"""幾何の特徴を π0.5 に接ぐ（板 F25）。Evo-0（arXiv 2507.00416）の形をなぞる。

**何をするか。** 凍結した VGGT に `front` と `wrist` の 2 枚を通し、出てきた
幾何のトークンを、π0.5 の画像トークンへ**交差注意 1 層**で混ぜる。
学習するのはこのファイルが作る層だけで、VGGT も π0.5 の視覚エンコーダも動かさない。

**なぜ効くと考えるか。** カメラの角度が変わると見た目のパターンは崩れるが、
物が立体としてどこにあるかは変わらない。VGGT はカメラ姿勢・奥行き・点群を当てる
課題で学習されているので、その特徴は視点が変わっても同じ物理を指す。

## 2 つの論文の違い（写すのは Evo-0 のほう）

| | GeoAware-VLA（arXiv 2509.14117） | **Evo-0（arXiv 2507.00416）** |
|---|---|---|
| 土台 | BAKU | **π0** |
| 視覚エンコーダ | VGGT で置き換える | **元のまま残し、VGGT を横に足す** |
| つなぎ | 中間 4 層を 1 次元畳み込み | **交差注意 1 層。2D が質問側・VGGT の最終層が答え側** |

**π0.5 の値打ちは事前学習された VLM そのものなので、置き換える形は採れない。**

## 差し込む場所

`PI05Pytorch.embed_prefix`（vendor/lerobot/policies/pi05/modeling_pi05.py:654、
2026-08-28 に確認）が、カメラごとに `embed_image(img)` を呼んで画像トークンを作り、
言語トークンと連結して prefix にしている。**その `img_emb` を混ぜたものに差し替える。**
LeRobot v0.6.0 には差し込み口が無いので、`quant_vision_4bit.py` と同じくモンキーパッチする。

## 画像の受け渡し

`embed_prefix` に届く時点で、画像は `[B, 3, 224, 224]` の `[-1, 1]` である
（modeling_pi05.py:1192-1195 が `resize_with_pad_torch` のあと `img * 2.0 - 1.0` する）。
**VGGT は `[0, 1]` を受け取って中で ImageNet の平均と分散で正規化する**
（`vggt/models/aggregator.py:205`）ので、`(img + 1) / 2` に戻してから渡す。
**大きさは 14 の倍数でなければ落ちる**（`patch_embed.py:69` の assert）。

## 既定を 252×252 にした理由（H100-20C・20 GB での実測）

| 大きさ | batch 16 の 1 回 | VGGT のピーク | π0.5 の学習 11.86 GB と足すと |
|---|---|---|---|
| **252×252** | **190 ms** | **5.56 GB** | **17.4 GB。収まる** |
| 518×518（VGGT の既定） | 801 ms | 8.20 GB | 20.1 GB。**20 GB の上限に触る** |

**推論側の差は小さい**（1 問い合わせ 63 ms 対 72 ms）。**学習と推論で大きさを
そろえる必要がある**ので、学習側の都合で 252 に決めた。

## 学習の始まりで base を壊さないための細工

**混ぜる量に係数を 1 つ置き、0 で初期化してある**（`fusion_gate`）。
**学習ステップ 0 の出力は素の π0.5 と 1 ビットも違わない。**
壊れるとすればそれは学習が進んだ結果であって、接いだこと自体ではない。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# VGGT の aggregator が返すトークンの幅。frame 注意と global 注意の出力を連結するので
# `embed_dim`（1024）の 2 倍になる。2026-08-28 に実測して確認した。
_VGGT_TOKEN_DIM = 2048

# aggregator が返すトークンの先頭には camera / register の特別なトークンが並ぶ。
# その本数は `patch_start_idx` として一緒に返る（実測 5）ので、そこから後ろだけを使う。
_DEFAULT_IMAGE_SIZE = 252
_DEFAULT_NUM_HEADS = 8


class VGGTFuser(nn.Module):
    """凍結した VGGT の幾何トークンを、π0.5 の画像トークンへ交差注意で混ぜる。

    Parameters
    ----------
    embed_dim : int
        π0.5 の画像トークンの幅（PaliGemma の width）。
    model_id : str
        VGGT の重み。既定は門の無い `facebook/VGGT-1B`。提出物に入れる段で
        商用可の `facebook/VGGT-1B-Commercial` に差し替える（板 F26）。
    image_size : int
        VGGT へ渡す 1 辺。**14 の倍数でなければ VGGT が落ちる。**
    num_heads : int
        交差注意の頭の数。
    """

    def __init__(
        self,
        embed_dim: int,
        model_id: str = "facebook/VGGT-1B",
        image_size: int = _DEFAULT_IMAGE_SIZE,
        num_heads: int = _DEFAULT_NUM_HEADS,
    ) -> None:
        super().__init__()
        if image_size % 14 != 0:
            raise ValueError(f"image_size は 14 の倍数でなければならない: {image_size}")

        from vggt.models.vggt import VGGT

        self.image_size = image_size
        # 凍結する。`requires_grad_(False)` と `eval()` の両方が要る——前者は勾配を、
        # 後者は dropout や BatchNorm の学習時の振る舞いを止める。
        self.vggt = VGGT.from_pretrained(model_id).eval()
        self.vggt.requires_grad_(False)

        self.to_policy_dim = nn.Linear(_VGGT_TOKEN_DIM, embed_dim)
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_key = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        # 0 で初期化する。**学習ステップ 0 では素の π0.5 と出力が一致する。**
        self.fusion_gate = nn.Parameter(torch.zeros(1))

    def train(self, mode: bool = True) -> VGGTFuser:  # noqa: FBT001, FBT002
        """VGGT だけは常に eval に留める（凍結しているので学習時の振る舞いは要らない）。"""
        super().train(mode)
        self.vggt.eval()
        return self

    def state_dict(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        """VGGT の重みを保存しない。

        **これを外すと、チェックポイント 1 つが 4.69 GB 太る。**VGGT は公開されている
        重みをそのまま使うので、読み込み時に `from_pretrained` で取り直せばよい。
        """
        full = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        return {k: v for k, v in full.items() if not k.startswith(f"{prefix}vggt.")}

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):  # noqa: ANN001, ANN003, ANN201, FBT001, FBT002
        """VGGT の重みは保存していないので、欠けていることを許す。"""
        return super().load_state_dict(state_dict, strict=False, **kwargs)

    @torch.no_grad()
    def encode(self, images: list[Tensor]) -> Tensor:
        """カメラ画像から幾何のトークンを作る。1 問い合わせにつき 1 回だけ呼ぶ。

        Parameters
        ----------
        images : list of Tensor
            カメラごとの `[B, 3, H, W]`。値は `[-1, 1]`（π0.5 の内部形式）。

        Returns
        -------
        tokens : Tensor
            `[B, カメラ数 × パッチ数, 2048]`。
        """
        prepared = []
        for image in images:
            # [-1, 1] → [0, 1]。VGGT は中で ImageNet の平均と分散を掛ける。
            unit_range = (image.to(torch.float32) + 1.0) / 2.0
            if unit_range.shape[-2:] != (self.image_size, self.image_size):
                unit_range = F.interpolate(
                    unit_range,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            prepared.append(unit_range)

        # VGGT は [B, S, 3, H, W] を取る。S はカメラ（VGGT にとっては視点）の数。
        stacked = torch.stack(prepared, dim=1)
        with torch.autocast(stacked.device.type, dtype=torch.bfloat16):
            layer_tokens, patch_start_idx = self.vggt.aggregator(stacked)

        # Evo-0 が使うのは最終層である（論文 3.2 節「3D tokens from VGGT's final layer」）。
        # 先頭の camera / register トークンは幾何そのものではないので落とす。
        tokens = layer_tokens[-1][:, :, patch_start_idx:, :]
        batch, views, patches, dim = tokens.shape
        return tokens.reshape(batch, views * patches, dim)

    def forward(self, image_embeddings: Tensor, geometry_tokens: Tensor) -> Tensor:
        """π0.5 の画像トークンに幾何のトークンを混ぜて返す。

        Parameters
        ----------
        image_embeddings : Tensor
            `[B, トークン数, embed_dim]`。カメラ 1 つぶん。
        geometry_tokens : Tensor
            `encode` が返したもの。**全カメラぶんをまとめて答え側に置く**
            ——手前のカメラが写していない面を、もう一方が写していることがある。

        Returns
        -------
        fused : Tensor
            `image_embeddings` と同じ形。
        """
        keys = self.norm_key(self.to_policy_dim(geometry_tokens.to(image_embeddings.dtype)))
        queries = self.norm_query(image_embeddings)
        attended, _ = self.cross_attention(queries, keys, keys, need_weights=False)
        return image_embeddings + self.fusion_gate * attended


def trainable_parameter_names(fuser: VGGTFuser) -> list[str]:
    """学習する層の名前。**VGGT が混ざっていないことを確かめるために使う。**"""
    return [name for name, param in fuser.named_parameters() if param.requires_grad]


def install(model_id: str | None = None, image_size: int | None = None) -> None:
    """`PI05Policy.from_pretrained` と `PI05Pytorch.embed_prefix` を差し替える。

    副作用目当てで import する（`lerobot_train_with_geometry.py` を参照）。

    ## なぜ `PI05Pytorch.__init__` で層を足さないのか（2026-08-28 に踏んだ穴）

    `PI05Policy.from_pretrained`（modeling_pi05.py:1032）は
    `model.load_state_dict(remapped_state_dict, strict=True)` を呼ぶ。**`__init__` の
    時点で接ぎ木の層が生えていると、土台の重みにその名前が無いので strict で例外になる。**
    しかもその例外は外側で握りつぶされて `Warning: Could not load state dict` と
    出るだけで（同 1057-1060 行）、**土台の重みが 1 つも載らないまま学習が始まる。**

    **だから、接ぐのは重みを読み終えたあとである。** 最適化器はこのあとに作られる
    （`lerobot_train.py` が `policy.get_optim_params()` を呼ぶのは `make_policy` の後）
    ので、**接ぎ木の層はちゃんと最適化器に入る。**

    ## なぜ勾配チェックポイントの内側に置かないのか（同じ日に踏んだ穴）

    最初は `embed_image` を包む形にしたが、`embed_prefix` はそれを
    `self._apply_checkpoint(...)` の中で呼ぶ（同 668 行）。**逆伝播のときに
    再計算した値が元と一致せず、`check_recomputed_tensors_match` で落ちた。**
    **接ぎ木はチェックポイントの外側に置く。**
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch

    resolved_id = model_id or os.environ.get("GEO_VGGT_MODEL", "facebook/VGGT-1B")
    resolved_size = image_size or int(
        os.environ.get("GEO_VGGT_IMAGE_SIZE", _DEFAULT_IMAGE_SIZE)
    )

    original_from_pretrained = PI05Policy.from_pretrained.__func__
    original_load_state_dict = PI05Policy.load_state_dict

    # 学習後のチェックポイントから読み直すとき、`geometry_fuser.*` は
    # **土台を読む時点ではまだ生えていない**ので strict の照合に引っかかる。
    # 取り分けておき、接いだあとで当てる（下の `_STASH`）。
    stash: dict[str, Tensor] = {}

    def patched_load_state_dict(self, state_dict, strict: bool = True, **kwargs):  # noqa: ANN001, ANN003, ANN201, FBT001, FBT002
        """`geometry_fuser.*` を取り分けてから、残りを元のやり方で読む。

        **取り分けないと strict の照合で例外になり、それが握りつぶされて
        土台の重みが 1 つも載らない**（modeling_pi05.py:1057-1060）。
        """
        stash.clear()
        remainder = {}
        for key, value in state_dict.items():
            if ".geometry_fuser." in key or key.startswith("geometry_fuser."):
                # 保存側では `model.geometry_fuser.<層>` になっているので、
                # 接ぎ木モジュールから見た名前へ戻す。
                stash[key.split("geometry_fuser.", 1)[1]] = value
            else:
                remainder[key] = value
        if stash:
            print(f"[geometry] チェックポイントから接ぎ木の層 {len(stash)} 個を取り分けた",
                  flush=True)
        return original_load_state_dict(self, remainder, strict=strict, **kwargs)

    def patched_from_pretrained(cls, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        policy = original_from_pretrained(cls, *args, **kwargs)
        inner = policy.model
        reference = next(inner.parameters())
        embed_dim = inner.paligemma_with_expert.paligemma.config.text_config.hidden_size
        fuser = VGGTFuser(embed_dim, model_id=resolved_id, image_size=resolved_size)
        inner.geometry_fuser = fuser.to(device=reference.device, dtype=reference.dtype)
        trainable = sum(
            param.numel() for param in fuser.parameters() if param.requires_grad
        )

        if stash:
            # 学習済みの接ぎ木を当てる。**当て損なうと、学習した 2,099 万個が
            # 捨てられて乱数のまま推論することになる。**
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
            f"{resolved_size}x{resolved_size} / embed_dim={embed_dim} / "
            f"学習するパラメータ {trainable:,} 個 / 層 {trainable_parameter_names(fuser)}",
            flush=True,
        )
        return policy

    PI05Policy.load_state_dict = patched_load_state_dict
    PI05Policy.from_pretrained = classmethod(patched_from_pretrained)

    def patched_embed_prefix(self, images, img_masks, tokens, masks):  # noqa: ANN001, ANN202
        """元の embed_prefix と同じ組み立てだが、画像トークンに幾何を混ぜてから並べる。

        元の実装（modeling_pi05.py:654-693）を写してある。**変えたのは
        `img_emb` を `fuser(...)` に通す 1 行だけである。**
        """
        fuser = getattr(self, "geometry_fuser", None)
        if fuser is None:
            # 土台の重みを読まずに作られた場合（from_pretrained を通っていない）。
            # 接がずに元の振る舞いに戻す。黙って落とすより素の π0.5 で動くほうがよい。
            return original_embed_prefix(self, images, img_masks, tokens, masks)

        geometry_tokens = fuser.encode(images)

        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):  # noqa: ANN001, ANN202
                return self.paligemma_with_expert.embed_image(img)

            # 勾配チェックポイントは元の画像埋め込みにだけ掛ける。**接ぎ木はこの外側**
            # ——内側に入れると逆伝播の再計算が元と一致せずに落ちる。
            img_emb = self._apply_checkpoint(image_embed_func, img)
            img_emb = fuser(img_emb, geometry_tokens)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(tokens):  # noqa: ANN001, ANN202
            return self.paligemma_with_expert.embed_language_tokens(tokens)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))

        return embs, pad_masks, att_masks

    original_embed_prefix = PI05Pytorch.embed_prefix
    PI05Pytorch.embed_prefix = patched_embed_prefix


install()
