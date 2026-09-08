#!/usr/bin/env python3
"""重みの系譜を並べて、**各学習で何を得て何を失ったか**を推論だけで測る。

**ロールアウトは回さない。**同じ画・同じ姿勢を固定して、指示文だけを変えて入れる。
1 点あたり 10 分ほどで、7 点並べても 1 時間強で終わる。

## 測るもの 2 つ

### A. 到達先 —— **どの物へ手を伸ばすか**

正しい指示文で行動列（50×7）を出し、**並進の 50 歩を積んで向きを作る。**
手先から各物へ向かう単位ベクトルと突き合わせ、**いちばん向きの合う物**を挙げる。

**これが「対象物を選べているか」の直接の測りである。**
`success_rate` は「選べて・掴めて・運べて・衝突しなかった」の積なので、
どこで落ちたのか分けられない。**選ぶところだけを切り出す。**

### B. 指示文の効き —— **何に反応しているか**

指示文を変えて、出力の変化量を測る。**変え方を 4 通りに分ける。**

| 種別 | 変えたもの | 何が分かるか |
|---|---|---|
| `same` | **何も変えない** | ゆらぎの下限。**0 でなければ測定が壊れている**（noise を固定してある） |
| `color` | **色の語だけ** | 色を見ているか |
| `place` | 位置を表す句だけ | 位置を見ているか |
| `noun` | 物体名 | 反応の上限。ここが 0 なら指示文自体を読んでいない |

**2 か所を別々に測る。**

| 場所 | 何が分かるか |
|---|---|
| **① 言語層を通った後の prefix**（画像トークンの位置だけ） | **VLM が語を画に接地しているか。**画像トークンの本数は指示文によらず一定で、PaliGemma の prefix は双方向注意なので**画像トークンは言語トークンを見ている** |
| **② 行動列 50×7** | ①が動いていても**行動に届いているか** |

**①は `.forward()` を直接呼ぶ経路なので、forward hook では捕れない**
（`sample_actions` が `paligemma.model.language_model.forward(...)` と書いている）。
**メソッドを差し替えて捕る。**

## 単位について

行動列は正規化された空間で出てくる。**物理量へ戻すには、その重み自身の
unnormalizer を使う**——重みごとに尺度が違うので、共通の尺度で戻すと
「そのモデルが出したかった行動」ではなくなる。
**`pi05_libero_base` は統計を持っていないので、`--fallback-norm` で補う。**

## 使い方

    bash solution/tools/probe/run_lineage_probe.sh <重み> <観測npz> [出力json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "solution/tools/viz"))

from instruction_variants import VARIANTS  # noqa: E402


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """`a` を基準にした相対差。**0 なら完全一致。**"""
    denom = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom) if denom else float("nan")


def action_std_mean(weights: Path, fallback: Path | None):
    """行動を物理量へ戻す mean/std を返す。無ければ `fallback` から借りる。"""
    from safetensors.numpy import load_file

    for root in (weights, fallback):
        if root is None:
            continue
        for name in ("policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                     "policy_preprocessor_step_2_normalizer_processor.safetensors"):
            path = root / name
            if not path.is_file():
                continue
            blob = load_file(path)
            mean = [v for k, v in blob.items() if "action" in k and "mean" in k]
            std = [v for k, v in blob.items() if "action" in k and "std" in k]
            if mean and std and np.isfinite(std[0]).all() and float(np.abs(std[0]).max()) > 0:
                return (np.asarray(mean[0]).ravel(), np.asarray(std[0]).ravel(),
                        "自身" if root is weights else f"借り物（{root.name}）")
    return None, None, "無し"


def unnormalize(chunk: np.ndarray, mean, std) -> np.ndarray:
    """行動列を物理量（OSC の ±1 指令）へ戻す。**pi05 は mean/std 正規化である。**"""
    act = chunk[0, :, :7].astype(np.float64)
    return act * std[:7] + mean[:7] if mean is not None else act


def head_direction(act: np.ndarray) -> np.ndarray:
    """並進 3 次元を 50 歩ぶん積んで、手先が向かう向き（単位ベクトル）を返す。"""
    total = act[:, :3].sum(axis=0)
    norm = float(np.linalg.norm(total))
    return total / norm if norm else total


def motion_profile(act: np.ndarray) -> dict[str, float]:
    """**「漂って掴まない」を「選べていない」と分けるための数。**

    動画では、正しい向きへ行っても掴まずにホバリングする回があった。
    **向きが合っているかだけでは、そこが分けられない。**次の 3 つを見る。

    | 出す数 | 何が分かるか |
    |---|---|
    | `reach_mm` | 50 歩の並進を積んだ長さ。**小さければそもそも動く気が無い** |
    | `grip_close_steps` | グリッパ指令が正（閉じる）の歩数。**0 なら掴む気が無い** |
    | `grip_first_close` | 最初に閉じる歩。**早すぎれば着く前に閉じている** |

    グリッパは ±1 のバンバン制御で std がほぼ 1 である（`policy_server.py` の
    `_smooth` の表）。だから符号だけを見ればよい。
    """
    # OSC の並進の上限は 1 コマ 0.05 m（`replay_video_probe.OUT_MAX_POS`）。
    reach = float(np.linalg.norm(act[:, :3].sum(axis=0))) * 0.05 * 1000
    grip = act[:, 6]
    close = np.flatnonzero(grip > 0)
    return {"reach_mm": reach,
            "grip_close_steps": int(close.size),
            "grip_first_close": int(close[0]) if close.size else -1,
            "grip_min": float(grip.min()), "grip_max": float(grip.max())}


def candidates(blob, task: str) -> dict[str, np.ndarray]:
    """物の名前 → 位置。**`*_to_robot0_eef` は相対ベクトルなので外す。**"""
    out = {}
    prefix = f"{task}/obj/"
    for key in blob.files:
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if name.endswith("_to_robot0_eef"):
            continue
        out[name] = np.asarray(blob[key], dtype=np.float64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("obs", type=Path, nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-mode", default="lowmem", choices=("lowmem", "builtin"))
    ap.add_argument("--tokenizer", default="auto")
    ap.add_argument("--fallback-norm", type=Path,
                    default=REPO / "work/artifacts/pi05_rp3000",
                    help="統計を持たない重み（base）のために借りる先")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from infer_pi05_once import _build_batch, load_policy, resolve_device

    device = resolve_device(args.device)
    tokenizer = args.tokenizer
    if tokenizer == "auto":
        for root in (args.weights, args.fallback_norm):
            if root and (root / "vlm/tokenizer.json").is_file():
                tokenizer = str(root / "vlm")
                break
    # **前後処理を持たない重みは借りる。**`pi05_libero_base` は正規化統計を
    # 同梱していないので、そのままでは組み立てられない。
    # **`policy_preprocessor.json` の有無では判定できない。**`pi05_libero_base` は
    # json は持っているが**統計の safetensors が無い**（stats が空のまま組まれる）。
    proc_root = None
    features_from = None
    load_mode = args.load_mode
    if not list(args.weights.glob("policy_preprocessor_step_*normalizer*.safetensors")):
        proc_root = args.fallback_norm
        features_from = args.fallback_norm
        print(f"  **前後処理と列の名前を借りる: {proc_root}**"
              f"（この重みは正規化統計を同梱していない）")
    # **`model.` 接頭辞が無い重みは lowmem の streamer で写せない**（9/8 に踏んだ。
    # `copied: 0 / unknown: 812` になる）。**本体の loader は 812 個を remap して
    # 通す**ので、そちらへ切り替える。ピークは 18 GB 級だが 1 点だけである。
    from safetensors import safe_open
    with safe_open(args.weights / "model.safetensors", framework="np") as f:
        has_prefix = any(k.startswith("model.") for k in f.keys())
    if not has_prefix and load_mode == "lowmem":
        load_mode = "builtin"
        print("  **本体の loader に切り替える**（`model.` 接頭辞が無い重みである）")
    print(f"=== {args.weights} を {device} で読む（tokenizer {tokenizer}）===", flush=True)
    policy, pre, post, tally = load_policy(args.weights, device,
                                           load_mode=load_mode,
                                           tokenizer=tokenizer,
                                           processor_root=proc_root,
                                           features_from=features_from)
    if tally:
        print(f"  重みの突き合わせ: {tally}")
    mean, std, origin = action_std_mean(args.weights, args.fallback_norm)
    print(f"  行動の逆正規化: {origin}")

    # ① prefix を捕まえる。**`.forward()` の直接呼び出しなので、メソッドを差し替える。**
    captured: dict[str, np.ndarray] = {}
    lang = policy.model.paligemma_with_expert.paligemma.model.language_model
    original_forward = lang.forward

    def recording_forward(*a, **kw):
        out = original_forward(*a, **kw)
        state = getattr(out, "last_hidden_state", None)
        if state is None and isinstance(out, (tuple, list)):
            state = out[0]
        if state is not None and "prefix" not in captured:
            captured["prefix"] = state.detach().float().cpu().numpy()
        return out

    lang.forward = recording_forward

    n_img: dict[str, int] = {}
    embed_image = policy.model.paligemma_with_expert.embed_image

    def counting_embed_image(img):
        out = embed_image(img)
        n_img["per_image"] = int(out.shape[1])
        return out

    policy.model.paligemma_with_expert.embed_image = counting_embed_image

    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.env_processor import LiberoProcessorStep
    env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    blobs = [np.load(p, allow_pickle=True) for p in args.obs]
    have = {t: b for b in blobs for t in VARIANTS if f"{t}/agentview_image" in b}
    tasks = args.tasks or list(have)
    results: dict[str, dict] = {}

    for task in tasks:
        blob = have[task]
        obs = {k.split("/", 1)[1]: blob[k] for k in blob.files
               if k.startswith(f"{task}/") and "/obj/" not in k
               and not k.endswith(("instruction", "objects"))}
        print(f"\n=== {task} ===")
        rows = []
        for kind, text in VARIANTS[task]:
            captured.pop("prefix", None)
            batch = _build_batch(obs, text, env_pre, pre, torch)
            # **noise を固定する。**flow matching なので、固定しないと
            # 同じ文でも出力が変わって「差」が測れない。
            gen = torch.Generator(device="cpu").manual_seed(0)
            shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
            noise = torch.randn(shape, generator=gen).to(device=device,
                                                         dtype=torch.float32)
            with torch.no_grad():
                chunk = policy.predict_action_chunk(batch, noise=noise)
            rows.append({"kind": kind, "text": text,
                         "prefix": captured.get("prefix"),
                         "chunk": np.asarray(chunk.detach().float().cpu().numpy())})

        base = rows[0]

        # ---- A. 到達先
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        act = unnormalize(base["chunk"], mean, std)
        direction = head_direction(act)
        prof = motion_profile(act)
        objs = candidates(blob, task)
        scored = []
        for name, pos in objs.items():
            to_obj = pos[:3] - eef
            dist = float(np.linalg.norm(to_obj))
            cos = float(direction @ (to_obj / dist)) if dist else float("nan")
            scored.append((name, cos, dist * 1000))
        scored.sort(key=lambda r: -r[1])
        print(f"  【A】50 歩で動く気の量: 並進 {prof['reach_mm']:.0f} mm 相当"
              f" ／ グリッパを閉じる歩 {prof['grip_close_steps']}/50"
              f"（最初 {prof['grip_first_close']}・範囲 {prof['grip_min']:+.2f}"
              f"〜{prof['grip_max']:+.2f}）")
        print("  【A】手先が向かう先（行動列 50 歩を積んだ向きとの一致）")
        for name, cos, dist in scored:
            print(f"       {name:24} cos {cos:+.3f}  距離 {dist:6.1f} mm")
        results.setdefault(task, {})["reach"] = [
            {"object": n, "cos": c, "dist_mm": d} for n, c, d in scored]
        results[task]["motion"] = prof

        # ---- B. 指示文の効き
        n_i = n_img.get("per_image", 0) * 2
        if base["prefix"] is None:
            print("  【B】prefix を捕れなかった（経路が変わった可能性）")
        else:
            print(f"  【B】指示文の効き（画像トークン {n_i} / prefix "
                  f"{base['prefix'].shape[1]}）")
            print(f"       {'種別':6}{'prefix':>10}{'行動列':>10}  指示文")
        out_rows = []
        for r in rows:
            d_prefix = (relative_l2(base["prefix"][:, :n_i], r["prefix"][:, :n_i])
                        if base["prefix"] is not None and r["prefix"] is not None
                        else float("nan"))
            d_chunk = relative_l2(base["chunk"], r["chunk"])
            print(f"       {r['kind']:6}{d_prefix:>10.4f}{d_chunk:>10.4f}  {r['text'][:48]}")
            out_rows.append({"kind": r["kind"], "text": r["text"],
                             "d_prefix_img": d_prefix, "d_chunk": d_chunk})
        results[task]["sensitivity"] = out_rows

    lang.forward = original_forward
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"weights": str(args.weights), "unnormalizer": origin, "tasks": results},
            ensure_ascii=False, indent=1))
        print(f"\n{args.json_out} へ書いた")


if __name__ == "__main__":
    main()
