"""量子化実験・候補 C3: torchao の float8 訓練で action expert (gemma_expert) だけを量子化する。

lerobot.scripts.lerobot_train.main() を呼ぶ前に、副作用目当てでこのモジュールを import する
こと（lerobot_train_with_quant.py 参照。`QUANT_EXPERT_MODE=float8` のときだけ import される）。
quant_optim.py（C1, OptimizerConfig への登録）や quant_vision_4bit.py（C2, vision encoder の
4bit 化）とは独立に効く別候補 —— このファイル単体で完結し、他の2つの状態を一切前提にしない。

## 何をパッチするか

`lerobot.policies.pi05.modeling_pi05.PI05Policy.__init__` をラップし、元の __init__
（モデル構築 → 精度キャスト → `_set_requires_grad()` によるフリーズ → `.to(device)`）が
完了した直後に、`torchao.float8.convert_to_float8_training` で action expert 配下の
`torch.nn.Linear` だけを `Float8Linear` に差し替える。

### フック位置がここで正しい理由（v0.6.0 の実際のコードを読んで確認済み、2026-08-20）

- `PI05Policy.__init__`（modeling_pi05.py ~917-941行）は `self.model = PI05Pytorch(config, ...)`
  を作った直後に `self.model.to(config.device)` まで終える。`PI05Pytorch.__init__` の内部で
  `PaliGemmaWithExpertModel.__init__`（同ファイル ~344-404行）が呼ばれ、その中で
  `self.to_bfloat16_for_selected_params(precision)` → `self._set_requires_grad()` の順に実行
  される。つまり `PI05Policy.__init__` が戻る時点で、精度キャストも requires_grad の凍結も
  デバイス配置もすべて確定済み —— float8 への差し替え自体は requires_grad を見ないが
  （`module_filter_fn` は fqn だけで判定する）、凍結済みの状態のツリーに対して変換をかける
  ことにしている。
- `lerobot.policies.factory.make_policy()` は `cfg.pretrained_path` が無ければ
  `policy_cls(config=cfg, ...)` を直接呼び、あれば `policy_cls.from_pretrained(...)` を呼ぶ。
  後者も内部で `model = cls(config, **kwargs)`（modeling_pi05.py ~984行）としており、つまり
  どちらの経路でも最終的に `PI05Policy.__init__` を通る —— このパッチ1箇所で両経路を
  カバーできる。
- `lerobot.scripts.lerobot_train.train()` は `policy = make_policy(...)`（~293行）の後、
  かなり後（~366行、`optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)`）に
  なってようやくオプティマイザを作る。この間に policy の Linear ツリーを書き換えても
  オプティマイザ構築より前なので、`policy.parameters()`（`get_optim_params()` がそのまま
  返す）はパッチ後のツリーを見る。
- `from_pretrained` 経路での重み読み込み順序も安全と確認済み: `torchao.float8.float8_linear.
  Float8Linear.from_float()`（torchao 0.17.0, float8_linear.py 293-335行）は
  `new_mod.weight = mod.weight` / `new_mod.bias = mod.bias` として **同じ Parameter オブジェクト
  を使い回す**（コピーしない）。`Float8Linear` は `torch.nn.Linear` のサブクラスで
  `weight`/`bias` という名前でパラメータ登録する点も変わらない。したがって
  `PI05Policy.from_pretrained()` が変換より後で呼ぶ `model.load_state_dict(...)`
  （modeling_pi05.py ~1031行）は、変換前後にかかわらず同じキー名 (`...action_in_proj.weight` 等)
  に対して `copy_()` するだけで、キー不一致や strict=True での失敗は起きない。

## 何を float8 化するか（scope の根拠、v0.6.0 のコードを読んで確認済み）

`--freeze_vision_encoder=true --train_expert_only=true`（このリポジトリの既定レシピ）では
`PaliGemmaWithExpertModel._set_requires_grad()`（modeling_pi05.py 429-437行）が
`self.paligemma`（vision tower + multi_modal_projector + gemma_2b 言語バックボーン）を
丸ごと凍結し、学習対象として残るのは `self.gemma_expert`（action expert, ~300M）と、
`PI05Pytorch.__init__`（同 589-593行）が作る `action_in_proj` / `action_out_proj` /
`time_mlp_in` / `time_mlp_out` の4つの Linear。

`convert_to_float8_training` の実体（torchao 0.17.0, float8_linear_utils.py の
`swap_linear_layers`）は、渡した root モジュールを起点にした **相対** fqn で
`module_filter_fn(module, fqn)` を呼ぶ（`nn.Linear` のインスタンスに対してのみ）。
このパッチでは root として `policy.model`（`PI05Pytorch` インスタンス）を渡すので、fqn は
以下のようになる（実際のツリーを読んで確認済み）:

- 凍結されている VLM 側: `paligemma_with_expert.paligemma.*` → 変換しない
- 学習対象の action expert 本体: `paligemma_with_expert.gemma_expert.model.*`
  （`self_attn.{q,k,v,o}_proj` と `mlp.{gate,up,down}_proj` が18層分。
  `pi_gemma.py` の `GemmaAttention`/`GemmaMLP` は標準 HF 実装のエイリアスで、
  この4つ・3つの Linear 名は transformers 側の命名。）→ 変換する
- `paligemma_with_expert.gemma_expert.lm_head`: `PiGemmaForCausalLM`
  （`pi_gemma.py` 326-335行）が継承元の `GemmaForCausalLM.__init__` からそのまま
  引き継ぐ Linear だが、`PaliGemmaWithExpertModel.forward()` は
  `self.gemma_expert.model.forward(...)` しか呼ばず、`lm_head` は pi05 の
  forward 計算グラフに一度も登場しない（embed_tokens も None にされ、実質使われない
  頭が残っているだけ）。呼ばれない Linear を float8 化しても速度上のメリットはゼロなので、
  対象から外す（`gemma_expert.model.` 配下だけを対象にし、`gemma_expert.lm_head` は
  対象にしない）。
- 学習対象の4つの投影/MLP: `action_in_proj` / `action_out_proj` / `time_mlp_in` /
  `time_mlp_out`（`PI05Pytorch` 直下、fqn はドットなしの単純な名前）→ 変換する

## ハードウェア要件

torchao の float8 訓練（`torch._scaled_mm` 経由）は compute capability 8.9 以上
（Hopper/Ada 以降）が必要（`torchao.utils.is_sm_at_least_89`, torchao 0.17.0で確認）。
この箱の GPU は H100-20C（cc 9.0）なので満たすが、変換の失敗が torchao 内部の
わかりにくいエラーではなく、ここで明示的な `RuntimeError` になるよう起動時にチェックする。

## 依存バージョン

`torchao==0.17.0`（`Dockerfile.pi05` が `torch==2.11.0+cu128` に合わせてピン留め、
2026-08-20 確認 —— 最新の 0.18.0 は torch 2.12 nightly 向けで対象外）。
`torchao.float8.convert_to_float8_training` は 0.16.0〜0.18.0 で公開 API 名として安定。

## スモークテストのやり方

**`--smoke-test` は使わないこと。** `--smoke-test` が強制する `batch_size=2` は
下記の「バッチサイズの制約」に反するため、この候補では確実に落ちる——モデルや
このモジュールの実装の問題ではなく、smoke test 自体がこの候補と両立しない
組み合わせになっている。代わりに `--batch-size` を 16 の倍数にして短い
ステップ数で手動検証すること:

    python train_pi05.py --expert-quant-mode float8 \
        --run-name pi05_c3_calib --steps 20 --batch-size 16 --save-freq 20

成功時: ログに `[quant_float8_expert] converted N nn.Linear layers ...`（N はゼロでない）
が出て、通常どおり数ステップ学習が進む。

失敗時に分かること:
- `torchao` が import できない / compute capability 不足 / 期待した属性パス
  （`model.paligemma_with_expert.gemma_expert` 等）が無い → このモジュールが送出する
  明確な `RuntimeError`（メッセージに原因を書く）。
- **バッチサイズの制約（2026-08-20、実機で発見。当初の見立ては誤りで、
  2回の実機確認を経て確定した）**: FP8 の Tensor Core 演算は、実行時の
  活性化テンソルの trailing dimension が 16 の倍数であることを要求するが、
  **これは Linear ごとに違う形で現れる** ——action 系の投影（`action_out_proj` 等）
  では `batch_size × chunk_size(50)` が、time_mlp 系の投影では `batch_size` 単体が
  その trailing dimension になる。実機で2種類の失敗を確認済み:
  `batch_size=2` では `RuntimeError: ...mat1 shape: (32x100)`（100 = 2×50、
  こちらが 16 の倍数でない）、`batch_size=8` では
  `RuntimeError: ...mat1 shape: (3072x8)`（8 = batch_size 単体、こちらは
  50 との積とは無関係に、それ自体が 16 の倍数でない）。**`batch_size` 自体が
  16 の倍数であれば両方の形を同時に満たす**（`batch_size×50` も自動的に
  16 の倍数になるため）。`train_pi05.py` の `main()` が `--expert-quant-mode
  float8` かつ `batch_size` が 16 の倍数でないときに早期 `SystemExit` する
  ガードを持つので、学習を始める前に気づける。
  `_make_module_filter_fn` の `in_features`/`out_features` チェック（下記）は
  静的な形の不整合に対する別の安全策であり、この動的なバッチ次元の制約は
  静的チェックでは検出できない。
- `_make_module_filter_fn` は `in_features`/`out_features` の両方が 16 の倍数か
  どうかも確認し、満たさない層は自動的に bf16 のまま除外する（ログの
  `skip {fqn}` 行と、最終行の「bf16 のまま残した層」件数で分かる）。今回の
  実機確認では該当する層は無かった（0件）——上記のバッチ次元の制約とは別の、
  将来モデル構造が変わった場合の安全策として残す。
- **勾配チェックポイントとの組み合わせは未検証** ——
  `PaliGemmaWithExpertModel.forward()` の `torch.utils.checkpoint.checkpoint(...,
  use_reentrant=False, preserve_rng_state=False)`（modeling_pi05.py 510-520行）は
  gemma_expert のレイヤーも含めて checkpoint 対象にする。float8 の `Float8Linear.forward`
  は forward 内で動的にキャストして `torch.autograd.Function`
  （`matmul_with_hp_or_float8_args`）を挟むため、re-entrant でない checkpoint との相性は
  torchao 公式にも一次情報が無く未検証。スモークテストが `torch.utils.checkpoint` の中で
  例外を出した場合、それはこのモジュールの実装ミスの兆候ではなく、**既知の未検証事項**が
  実際に踏まれたということ。
"""

import logging

import torch

try:
    import torchao  # noqa: F401  未インストール環境で即座にわかりやすく落とすため import 自体を試す
    from torchao.float8 import convert_to_float8_training
    from torchao.utils import is_sm_at_least_89
except ImportError as e:
    raise RuntimeError(
        "quant_float8_expert: torchao が import できません。"
        "Dockerfile.pi05 は torchao==0.17.0 をインストールしている前提です。"
        f"（元の ImportError: {e}）"
    ) from e

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

log = logging.getLogger(__name__)


def _assert_hardware_supported() -> None:
    """torchao float8 訓練が要求する compute capability >= 8.9 を起動時に確認する。

    torchao 自身は `Float8Linear.from_float()`/`convert_to_float8_training()` の時点では
    ハードウェアを検査しない（0.17.0 の float8_linear.py を読んで確認済み）ため、放置すると
    実際に `torch._scaled_mm` を呼ぶ最初の forward まで失敗が遅延し、わかりにくいエラーになる。
    ここで先に明示チェックして落とす。
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "quant_float8_expert: CUDA GPU が見えません。torchao の float8 訓練には "
            "compute capability 8.9 以上の NVIDIA GPU が必要です。"
        )
    if not is_sm_at_least_89():
        cc_major, cc_minor = torch.cuda.get_device_capability()
        raise RuntimeError(
            f"quant_float8_expert: この GPU（{torch.cuda.get_device_name()}, "
            f"compute capability {cc_major}.{cc_minor}）は torchao の float8 訓練が要求する "
            "compute capability 8.9 以上（Hopper/Ada 以降）を満たしません。"
        )


# fqn は convert_to_float8_training(root_module=policy.model, ...) を起点にした相対パス。
# PI05Pytorch 直下の4つの投影/MLP（ドットなしの単純な名前になる）。
_EXPERT_TOP_LEVEL_LINEARS = frozenset(
    {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}
)
# gemma_expert 本体（self_attn の q/k/v/o_proj, mlp の gate/up/down_proj、全18層）。
# "paligemma_with_expert.gemma_expert.lm_head" は意図的に含めない（forward で一度も
# 呼ばれない死んだ Linear のため、docstring の "何を float8 化するか" 参照）。
_EXPERT_MODEL_PREFIX = "paligemma_with_expert.gemma_expert.model."


# **FP8 の Tensor Core 演算（torch._scaled_mm）はハードウェア側の制約として、
# 行列の次元が 16 の倍数であることを要求する。** 2026-08-20、実機の --smoke-test で
# 発見: `time_mlp_in`/`time_mlp_out`（時刻埋め込み用の投影。幅が 16 の倍数になる
# 保証が無い）の逆伝播で `torch._scaled_mm` が
# `RuntimeError: Expected trailing dimension of mat1 to be divisible by 16 but
# got mat1 shape: (32x100)` を出した。これは実装の誤りではなく、対象にする
# Linear の形が FP8 のハードウェア要件を満たすかどうかという、対象選定の問題。
# fqn ベースの一覧に加えて、in_features/out_features の両方が 16 の倍数かどうかを
# ここで直接確認し、満たさない層は変換対象から静かに除外する（除外した件数は
# ログに出す。gemma_expert 本体の attention/MLP は hidden_size が 16 の倍数で
# 揃っているはずなので、実際に除外されるのは time_mlp 系だけになる見込み）。
_FP8_DIM_MULTIPLE = 16


def _make_module_filter_fn():
    """`module_filter_fn(mod, fqn) -> bool` と、変換した/除外した件数を数える
    カウンタを返す。

    torchao の `swap_linear_layers`（0.17.0, float8_linear_utils.py）はこの関数を
    `nn.Linear` インスタンスに対してのみ呼ぶので、ここで isinstance チェックはしていない。
    """
    state = {"n_converted": 0, "n_skipped_dim": 0}

    def module_filter_fn(mod, fqn) -> bool:  # noqa: ANN001
        in_scope = fqn in _EXPERT_TOP_LEVEL_LINEARS or fqn.startswith(_EXPERT_MODEL_PREFIX)
        if not in_scope:
            return False

        dim_ok = (
            mod.in_features % _FP8_DIM_MULTIPLE == 0
            and mod.out_features % _FP8_DIM_MULTIPLE == 0
        )
        if not dim_ok:
            state["n_skipped_dim"] += 1
            log.info(
                f"[quant_float8_expert] skip {fqn}: shape "
                f"({mod.in_features}, {mod.out_features}) は {_FP8_DIM_MULTIPLE} の"
                " 倍数でない（FP8 の Tensor Core 演算のハードウェア要件を満たさない"
                "ため、bf16 のまま残す）"
            )
            return False

        state["n_converted"] += 1
        return True

    return module_filter_fn, state


def _convert_expert_to_float8(policy: PI05Policy) -> None:
    """`policy.model` 配下の action expert 由来の Linear だけを Float8Linear に差し替える。"""
    model = getattr(policy, "model", None)
    if model is None:
        raise RuntimeError(
            "quant_float8_expert: policy.model が存在しません。lerobot のバージョンが "
            "想定（v0.6.0）と違い、PI05Policy の構造が変わっている可能性があります。"
        )

    required_attrs = (
        "paligemma_with_expert",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    )
    missing = [attr for attr in required_attrs if not hasattr(model, attr)]
    if missing:
        raise RuntimeError(
            f"quant_float8_expert: PI05Pytorch に期待する属性が見つかりません: {missing}。"
            "lerobot v0.6.0 の modeling_pi05.py を前提にした対象指定なので、バージョン違いの"
            "可能性を確認してください。"
        )
    if not hasattr(model.paligemma_with_expert, "gemma_expert"):
        raise RuntimeError(
            "quant_float8_expert: model.paligemma_with_expert.gemma_expert が見つかりません。"
        )

    module_filter_fn, state = _make_module_filter_fn()
    convert_to_float8_training(model, module_filter_fn=module_filter_fn)

    if state["n_converted"] == 0:
        # 対象ゼロ件は「間違って何も変換されなかった」を無言で見逃さないための安全策。
        raise RuntimeError(
            "quant_float8_expert: nn.Linear を1つも float8 に変換できませんでした。"
            "module_filter_fn の対象 fqn（'paligemma_with_expert.gemma_expert.model.*' / "
            f"{sorted(_EXPERT_TOP_LEVEL_LINEARS)}）がモデルの実際の構造と噛み合っていません。"
        )

    recipe_matches_scope = getattr(policy.config, "freeze_vision_encoder", False) and getattr(
        policy.config, "train_expert_only", False
    )
    if not recipe_matches_scope:
        log.warning(
            "[quant_float8_expert] freeze_vision_encoder/train_expert_only が既定の "
            "true/true ではありません。この候補は「paligemma は凍結、gemma_expert のみ学習」"
            "という前提で対象を選んでいるので、他の学習レシピで使う場合は対象範囲を見直すこと。"
        )

    log.info(
        f"[quant_float8_expert] converted {state['n_converted']} nn.Linear layers under "
        "gemma_expert (+ action/time projections) to Float8Linear "
        f"（次元が{_FP8_DIM_MULTIPLE}の倍数でないため bf16 のまま残した層: "
        f"{state['n_skipped_dim']}）。"
    )


_assert_hardware_supported()

_ORIGINAL_PI05POLICY_INIT = PI05Policy.__init__


def _patched_init(self, config, **kwargs):
    _ORIGINAL_PI05POLICY_INIT(self, config, **kwargs)
    _convert_expert_to_float8(self)


PI05Policy.__init__ = _patched_init
