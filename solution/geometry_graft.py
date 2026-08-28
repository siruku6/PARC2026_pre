"""幾何の特徴を π0.5 に接ぐ（板 F25）。Evo-0（arXiv 2507.00416）の形をなぞる。

**このファイルは学習と推論の両方から使う。** 学習側は
`solution/train/experiments/geometry/vggt_fuser.py` が、推論側は
`solution/policy_server.py` が import する。**片方だけ直すと、学習した重みと
推論の計算が食い違う**——接ぎ木の形は 1 か所にしか置かない。

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

# `facebook/VGGT-1B` の config.json の値（2026-08-28 に取得して確認）。
# `pretrained=False` で骨だけ組むときに要る。
_VGGT_IMG_SIZE = 518
_VGGT_PATCH = 14
_VGGT_EMBED_DIM = 1024


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
    pretrained : bool
        True なら VGGT の重みを Hugging Face から読む（学習側）。False なら
        骨だけ組む（推論側。重みはチェックポイントから流し込まれる）。
    """

    def __init__(
        self,
        embed_dim: int,
        model_id: str = "facebook/VGGT-1B",
        image_size: int = _DEFAULT_IMAGE_SIZE,
        num_heads: int = _DEFAULT_NUM_HEADS,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if image_size % 14 != 0:
            raise ValueError(f"image_size は 14 の倍数でなければならない: {image_size}")

        from vggt.models.vggt import VGGT

        self.image_size = image_size
        # 凍結する。`requires_grad_(False)` と `eval()` の両方が要る——前者は勾配を、
        # 後者は dropout や BatchNorm の学習時の振る舞いを止める。
        #
        # **`pretrained=False` は推論側で使う。** 採点環境では Hugging Face へ
        # 取りに行けないので骨だけ組み、重みはチェックポイントから流し込む
        # （VGGT の重みは学習のチェックポイントに入っている。理由は下の「保存」節）。
        if pretrained:
            self.vggt = VGGT.from_pretrained(model_id).eval()
        else:
            self.vggt = VGGT(
                img_size=_VGGT_IMG_SIZE,
                patch_size=_VGGT_PATCH,
                embed_dim=_VGGT_EMBED_DIM,
            ).eval()
        self.vggt.requires_grad_(False)

        self.to_policy_dim = nn.Linear(_VGGT_TOKEN_DIM, embed_dim)
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_key = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        # 0 で初期化する。**学習ステップ 0 では素の π0.5 と出力が一致する。**
        self.fusion_gate = nn.Parameter(torch.zeros(1))

        # **学習時の画像の大きさをチェックポイントに残す。** 推論側がこれを取り違えると、
        # VGGT に入る画像の大きさが変わって特徴がずれる。学習した重みが意味を失うので、
        # 値そのものを重みと一緒に持ち運ぶ（`persistent=True` の buffer）。
        self.register_buffer(
            "image_size_marker", torch.tensor([float(image_size)]), persistent=True
        )

    def train(self, mode: bool = True) -> VGGTFuser:  # noqa: FBT001, FBT002
        """VGGT だけは常に eval に留める（凍結しているので学習時の振る舞いは要らない）。"""
        super().train(mode)
        self.vggt.eval()
        return self

    # **VGGT の重みはチェックポイントに入る。外そうとしたが外せなかった。**
    # `state_dict` を上書きして `vggt.` を落とす細工を書いたが、**PyTorch は親から
    # 呼ぶとき子の `state_dict` の戻り値を使わない**（共有の辞書へ書き込む作りなので、
    # 除外した新しい辞書は捨てられる）。2026-08-28 に実測して確認した。
    #
    # **結果としてこちらの都合には合っている。**チェックポイント 1 個が 4.69 GB
    # 太る代わりに、**採点環境で VGGT の重みを別に用意する必要が無くなる**
    # ——`policy_server.py` の「1 テンソルずつ流し込む」読み込みが、
    # 接ぎ木も VGGT もまとめて埋める。提出 zip は約 14 GB で上限 20 GB に収まる。

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
        # **π0.5 は dtype が混在している**（vision_tower や layernorm は fp32 のまま、
        # 本体は bf16）。だから「模型の先頭パラメータの dtype」で決め打ちにすると、
        # 交差注意の入口で `mat1 and mat2 must have the same dtype` で落ちる
        # （2026-08-28 に推論側で踏んだ）。**自分の重みの dtype に合わせて計算し、
        # 最後に呼び出し側の dtype へ戻す。**
        #
        # **学習時と推論時で dtype がそろうことは、チェックポイントが保証する**
        # ——`policy_server.py` の流し込みは `target.data = tensor` なので、
        # 保存されている dtype がそのまま入る。
        work_dtype = self.to_policy_dim.weight.dtype
        keys = self.norm_key(self.to_policy_dim(geometry_tokens.to(work_dtype)))
        queries = self.norm_query(image_embeddings.to(work_dtype))
        attended, _ = self.cross_attention(queries, keys, keys, need_weights=False)
        return image_embeddings + (self.fusion_gate * attended).to(image_embeddings.dtype)


def trainable_parameter_names(fuser: VGGTFuser) -> list[str]:
    """学習する層の名前。**VGGT が混ざっていないことを確かめるために使う。**"""
    return [name for name, param in fuser.named_parameters() if param.requires_grad]




_PATCHED = False


def patch_embed_prefix() -> None:
    """`PI05Pytorch.embed_prefix` を、画像トークンに幾何を混ぜる版へ差し替える。

    **学習側と推論側の両方から呼ぶ。**2 回呼んでも二重には当たらない。

    ## 差し込む場所

    `PI05Pytorch.embed_prefix`（vendor/lerobot/policies/pi05/modeling_pi05.py:654、
    2026-08-28 に確認）が、カメラごとに `embed_image(img)` を呼んで画像トークンを作り、
    言語トークンと連結して prefix にしている。**元の実装を写し、`img_emb` を
    `fuser(...)` に通す 1 行だけを足してある。**

    ## 勾配チェックポイントの外側に置く理由

    最初は `embed_image` を包む形にしたが、`embed_prefix` はそれを
    `self._apply_checkpoint(...)` の中で呼ぶ。**逆伝播のときに再計算した値が元と
    一致せず `CheckpointError: A different number of tensors was saved during the
    original forward and recomputation` で落ちた**（2026-08-28）。
    **接ぎ木はチェックポイントの外側に出す。**

    ## 接ぎ木が無いときは素の振る舞いに戻す

    `geometry_fuser` を持たないモデル（今までの重み）でも、そのまま動く。
    **黙って落とすより、素の π0.5 として動くほうがよい。**
    """
    global _PATCHED  # noqa: PLW0603
    if _PATCHED:
        return

    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    original_embed_prefix = PI05Pytorch.embed_prefix

    def patched_embed_prefix(self, images, img_masks, tokens, masks):  # noqa: ANN001, ANN202
        fuser = getattr(self, "geometry_fuser", None)
        if fuser is None:
            return original_embed_prefix(self, images, img_masks, tokens, masks)

        geometry_tokens = fuser.encode(images)

        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):  # noqa: ANN001, ANN202
                return self.paligemma_with_expert.embed_image(img)

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

    PI05Pytorch.embed_prefix = patched_embed_prefix
    _PATCHED = True


def attach(inner, image_size: int, model_id: str = "facebook/VGGT-1B",
           pretrained: bool = True) -> VGGTFuser:
    """`PI05Pytorch` に接ぎ木を生やし、`embed_prefix` を差し替える。

    Parameters
    ----------
    inner : PI05Pytorch
        `PI05Policy.model`。
    image_size : int
        VGGT へ渡す 1 辺。**学習時と推論時でそろえること**——違えば特徴がずれる。
    model_id : str
        `pretrained=True` のときに読む VGGT の重み。
    pretrained : bool
        推論側は False。骨だけ組み、重みはチェックポイントから流し込む。

    Returns
    -------
    fuser : VGGTFuser
        生やした接ぎ木（`inner.geometry_fuser` にも入る）。
    """
    reference = next(inner.parameters())
    embed_dim = inner.paligemma_with_expert.paligemma.config.text_config.hidden_size
    fuser = VGGTFuser(
        embed_dim, model_id=model_id, image_size=image_size, pretrained=pretrained
    )
    inner.geometry_fuser = fuser.to(device=reference.device, dtype=reference.dtype)
    patch_embed_prefix()
    return fuser
