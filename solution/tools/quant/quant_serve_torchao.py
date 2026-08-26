"""量子化スイープ・推論時候補 C5: torchao の重みのみ int8 量子化を、読み込み済みの
π0.5 に後掛けする（policy_server.py 専用。work/quant_sweep/CANDIDATES.md の C5）。

**学習側の候補 C3（float8, action expert 限定。実装中）とは使う API が違う。**
torchao の `quantize_` は「すでに実体化・読み込み済みのモデルへ後から掛ける」ことを
公式に想定した推論専用の API である —— torchao の README・ドキュメントは
`quantize_(model, Int8WeightOnlyConfig())` の形を "Inference" の節でのみ例示しており
（2026-08-20 確認）、学習中の重みを段階的に float8 化する
`torchao.float8.convert_to_float8_training`（C3 が使う側）とは別物。ここでは
`_build_pi05` がフル精度で読み終えたあとの、推論専用の凍結重みに対して掛ける。

`Int8WeightOnlyConfig` の実体は `torchao.quantization.quant_api.Int8WeightOnlyConfig`
で、`torchao.quantization` パッケージのトップレベルからも import できる
（`torchao==0.17.0` の wheel を展開して確認済み。この版は Dockerfile.pi05 が固定
しているのと同じもの——torchao 0.18.0 は torch 2.12(nightly) 向けで、この
torch==2.11.0+cu128 には 0.17.0 が対応版であるため）。

対象範囲・危険性は quant_serve_common.py のモジュール docstring と共通なので
そちらを参照。既定 (`QUANT_SERVE_SCOPE=vision_only`) は vision_tower +
multi_modal_projector だけで、modeling_pi05.py の `compute_layer_complete`
（236-305行）が触らない安全な範囲。

**`QUANT_SERVE_SCOPE=full` にしたときの壊れ方（torchao 版。bnb とは非対称）:**
torchao の量子化テンソル（`AffineQuantizedTensor` 系、Int8WeightOnlyConfig の
version による）は「既存コードに対する drop-in 互換」を狙って `__torch_function__`
を実装しており、`.dtype` は量子化前の論理的な dtype（bf16/fp32）をそのまま返す
実装が多い —— bnb の `Params4bit`（4bit にパックした `uint8` の実体をそのまま
`.dtype` に反映する）と違い、`compute_layer_complete` の `.weight.dtype` 分岐
自体はすり抜ける可能性がある。**ただしこれは torchao のテンソル実装の一般的な
設計方針からの推測であり、この版・この形（quantize_ を Linear の集合へ再帰適用）
で実際にすり抜けるかは GPU での smoke test でしか確認できない。** 分岐をすり抜けた
としても、`bnb.matmul_4bit` のような専用カーネル呼び出しは経由しない
（`quantize_` は `nn.Linear.forward` をそのまま使う）ため、bnb ほど明確な
クラッシュ経路は無いと見ているが、精度の劣化（int8 化した attention/mlp の
重みで生成する行動の質が落ちる）は別途あり得る。**いずれにせよ壊れても
compute_layer_complete 自体は直さない**（要 lerobot 本体の変更で対象外）。

policy_server.py からは `QUANT_SERVE_MODE=torchao_int8` のときだけ import され、
`apply(policy, device)` を呼ぶだけの副作用フリーなモジュールである。
"""
import quant_serve_common as _common


def apply(policy, device: str) -> None:
    """読み込み済みの π0.5 方策へ、torchao の int8 weight-only 量子化を後掛けする。

    呼び出し側（policy_server.py の `MyPolicy.__init__`）が
    `self._policy.to(self._device)` を呼ぶ**前**に呼ぶ想定。`quantize_` の
    `device=` 引数に最終 device をそのまま渡すので、対象部分木はこの 1 呼び出しで
    「device へ移動 → その場で量子化」まで完了する（`torchao/quantization/quant_api.py`
    の `_replace_with_custom_fn_if_matches_filter` が `device is not None` のとき
    `model.to(device=device)` を先に呼んでから置き換えることを、0.17.0 の実 wheel を
    展開して確認済み）。その後の全体 `.to(device)` は対象外の部分
    （`QUANT_SERVE_SCOPE=vision_only` なら gemma_expert 等）だけを動かす。

    Parameters
    ----------
    policy : PI05Policy
        `_build_pi05` が読み終えた（フル精度の）方策。
    device : str
        最終的な推論 device（"cuda" か "cpu"）。**torchao の int8
        weight-only 量子化自体は CPU でも動く**（bnb の 4bit と違い CUDA 専用
        カーネルに縛られない）が、この採点環境は GPU 前提
        （`_build_pi05` の docstring・run_docker_eval.sh の GPU チェック参照）
        なので、実質つねに "cuda" が渡ってくる。

    Raises
    ------
    ImportError
        `torchao` が import できない場合
        （Dockerfile.pi05 が `torchao==0.17.0` を入れている前提）。
    RuntimeError
        対象部分木に nn.Linear が 1 つも無かった場合
        （属性パスは通っても中身が期待と違う可能性がある）。
    """
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError as error:
        raise ImportError(
            "QUANT_SERVE_MODE=torchao_int8 が指定されたが torchao が"
            " import できない。Dockerfile.pi05 が torchao==0.17.0 を"
            " 入れているはずのイメージ（parc2026-train-pi05 由来）で"
            " 動かしているか確認すること。"
        ) from error

    targets = _common.target_modules(policy)
    total = 0
    for label, module in targets.items():
        expected = _common.count_linears(module)
        if expected == 0:
            raise RuntimeError(
                f"{label} の中に nn.Linear が 1 つも無い。属性パスは通っているが"
                " 中身が期待と違う可能性がある（別モデルのチェックポイントを"
                " 読んでいる等）。"
            )
        # config は Linear ごとの重み変換を持つだけの値オブジェクトなので、
        # target ごとに使い回す（quantize_ が呼ぶたびに複製する必要は無い）。
        quantize_(module, Int8WeightOnlyConfig(), device=device)
        print(
            f"[parc][quant_serve_torchao] {label}: Linear {expected} 個を"
            " int8 weight-only 化した", flush=True,
        )
        total += expected

    print(
        f"[parc][quant_serve_torchao] scope={_common.scope()}  合計 {total} 個の"
        " Linear を int8 化した", flush=True,
    )
