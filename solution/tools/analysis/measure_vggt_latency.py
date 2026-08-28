"""F18: VGGT を 1 回通すのに何秒かかるかを測る。

**測るのは「1 問い合わせあたりの増分」である。** 採点の上限は 1 問い合わせ 10 秒で、
いまの π0.5 の推論は本番設定で 1.5〜2.4 秒（docs/knowledge/14）。残りは 7.6〜8.5 秒ある。

**2 つに分けて測る。** Evo-0（arXiv 2507.00416）が使うのは aggregator の最終層の
トークンだけで、深度・点群・カメラ姿勢の頭は要らない。

- `aggregator` のみ ── **これが実際に足す分**
- `forward` 全体 ── 上限として
"""
import argparse
import time

import torch
from vggt.models.vggt import VGGT


def bench(fn, images: torch.Tensor, warmup: int, iters: int) -> tuple[float, float]:
    """warmup のあと iters 回まわし、1 回あたりの秒数の中央値と最大値を返す。"""
    for _ in range(warmup):
        fn(images)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(images)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2], times[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    # 重みは float32 のまま置き、計算だけ bf16 にする（VGGT 公式のデモと同じ形）。
    # モデルごと bf16 に落とすと、深度の頭の中の転置畳み込みが float32 を作って型が食い違う。
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    load_sec = time.perf_counter() - load_start
    params = sum(p.numel() for p in model.parameters())
    weights_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"読み込み {load_sec:.1f} 秒 / パラメータ {params/1e8:.1f} 億 / 重み {weights_gb:.2f} GB")

    for size in (252, 518):  # VGGT の patch は 14 なので 14 の倍数にする（256 は割り切れない）
        # 1 問い合わせで front と wrist の 2 枚。VGGT は (B, S, 3, H, W) を取る。
        images = torch.rand(1, 2, 3, size, size, device="cuda")
        for label, fn in (
            ("aggregator のみ（Evo-0 が使う分）", lambda x: model.aggregator(x)),
            ("forward 全体（上限）", lambda x: model(x)),
        ):
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    median, worst = bench(fn, images, args.warmup, args.iters)
            except Exception as exc:  # noqa: BLE001
                print(f"{size}x{size} 2 枚 / {label}: 失敗 {type(exc).__name__}: {exc}")
                continue
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"{size}x{size} 2 枚 / {label}: "
                  f"中央値 {median*1000:.0f} ms  最大 {worst*1000:.0f} ms  ピーク {peak:.2f} GB")


if __name__ == "__main__":
    main()
