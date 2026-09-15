"""LIBERO-plus が場面カメラに掛ける劣化 5 種を、学習イメージで動く形に写したもの。

**正本は採点イメージの `/workspace/LIBERO-plus/libero/libero/envs/env_wrapper.py` である。**
学習イメージ（`parc2026-train-pi05`）には `skimage` と `wand` が無いので、
同じ計算を `scipy`・numpy で書き直した。

| 種類 | 正本の実装 | ここ | 正本と一致するか |
|---|---|---|---|
| gaussian | `skimage.filters.gaussian` | skimage が中で呼ぶ `scipy.ndimage.gaussian_filter` を同じ引数で | 一致 |
| zoom | `scipy.ndimage.zoom` | 同じ関数 | 一致 |
| fog | numpy | 同じ式（乱数の出どころだけ違う） | 分布は一致 |
| motion | ImageMagick の `MotionBlurImage` | 同じ核（片側・ガウスの重み）を numpy で | 近似 |
| glass | skimage ＋ 画素ごとの Python ループ | 近傍から拾う処理を配列演算で一度に | 近似 |

**乱数は必ず引数の `rng` から引く。**DataLoader の worker は numpy の大域乱数を同じ状態で
引き継ぐので、`np.random.*` を使うと worker 間で同じ劣化が並ぶ。

**入力も出力も (H, W, 3) の uint8 である。**採点は 128×128 に描いた画像に掛けるので、
強度の数字を採点と揃えるには 128×128 で渡すこと（`sensor_noise_patch.py` が縮めてから渡す）。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom as scipy_zoom

FAMILIES = ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur")

# 以下の 5 表は env_wrapper.py の値をそのまま写した。添字は severity - 1。
_MOTION = [(5, 2), (8, 3), (10, 4), (12, 5), (15, 6),
           (18, 8), (20, 10), (25, 12), (30, 15), (35, 20)]
_GAUSSIAN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_ZOOM = [np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01), np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02), np.arange(1, 1.31, 0.03), np.arange(1, 1.36, 0.01),
         np.arange(1, 1.41, 0.01), np.arange(1, 1.46, 0.02), np.arange(1, 1.51, 0.02),
         np.arange(1, 1.56, 0.03)]
_FOG = [(0.5, 3), (1.0, 2.8), (1.5, 2.5), (2.0, 2.2), (2.5, 2.0),
        (3.0, 1.8), (3.5, 1.6), (4.0, 1.5), (4.5, 1.4), (5.0, 1.3)]
# (sigma, max_delta, iterations)
_GLASS = [(0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2), (1.1, 3, 2),
          (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1)]


def corrupt(image: np.ndarray, family: str, severity: int,
            rng: np.random.Generator) -> np.ndarray:
    """1 枚の画像に、指定した種類・強度の劣化を掛ける。

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) の uint8。
    family : str
        `FAMILIES` のどれか。
    severity : int
        1〜10。採点のタスク名の `_noise_N` から、種類ごとの先頭を引いた値。
    rng : np.random.Generator
        劣化の中で使う乱数の出どころ。

    Returns
    -------
    corrupted : np.ndarray
        入力と同じ形の uint8。
    """
    if not 1 <= severity <= 10:
        raise ValueError(f"severity は 1〜10 である: {severity}")
    functions = {
        "motion_blur": motion_blur,
        "gaussian_blur": gaussian_blur,
        "zoom_blur": zoom_blur,
        "fog": fog,
        "glass_blur": glass_blur,
    }
    if family not in functions:
        raise ValueError(f"知らない種類: {family!r}（{FAMILIES} のどれか）")
    return functions[family](image, severity, rng)


def _skimage_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """`skimage.filters.gaussian(image, sigma, channel_axis=-1)` と同じ計算。

    skimage は色の軸の sigma を 0 にし、`mode="nearest"`・`truncate=4.0` で
    scipy を呼んでいるだけなので、同じ引数で直接呼べば一致する。
    """
    return gaussian_filter(image, sigma=(sigma, sigma, 0), mode="nearest",
                           cval=0, truncate=4.0)


def gaussian_blur(image: np.ndarray, severity: int,
                  rng: np.random.Generator) -> np.ndarray:
    """ガウスぼかし。`sigma` は強度そのもの（画素単位）。"""
    sigma = _GAUSSIAN[severity - 1]
    blurred = _skimage_gaussian(image / 255.0, sigma)
    # 正本は `np.clip(x, 0, 1) * 255` を `astype(np.uint8)` で切り捨てている
    return (np.clip(blurred, 0, 1) * 255).astype(np.uint8)


def motion_blur(image: np.ndarray, severity: int,
                rng: np.random.Generator) -> np.ndarray:
    """動きのぶれ。**向きは 1 枚ごとに ±45° から引き直す**（正本も毎歩引き直す）。"""
    radius, sigma = _MOTION[severity - 1]
    angle = rng.uniform(-45, 45)
    return motion_blur_at(image, radius, sigma, angle)


def motion_blur_at(image: np.ndarray, radius: float, sigma: float,
                   angle: float) -> np.ndarray:
    """ImageMagick の `MotionBlurImage` と同じ核で、向きを決めてぶらす。

    核は**片側**である——画素 i 個ぶん先の画素に `exp(-i²/2σ²)` の重みを付けて
    足し合わせる（i = 0〜2·ceil(radius)）。はみ出した先は端の画素を使う
    （ImageMagick の既定の `Edge`）。向きと比べる検証は
    `work/blur_ladder/verify_ports.py` にある。
    """
    width = 2 * int(np.ceil(radius)) + 1
    steps = np.arange(width, dtype=np.float64)
    kernel = np.exp(-(steps * steps) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    theta = np.deg2rad(angle)
    offset_x = np.ceil(steps * np.cos(theta) - 0.5).astype(int)
    offset_y = np.ceil(steps * np.sin(theta) - 0.5).astype(int)

    height, width_px = image.shape[:2]
    source = image.astype(np.float64)
    rows = np.arange(height)[:, None]
    cols = np.arange(width_px)[None, :]
    blurred = np.zeros_like(source)
    for weight, dx, dy in zip(kernel, offset_x, offset_y):
        blurred += weight * source[np.clip(rows + dy, 0, height - 1),
                                   np.clip(cols + dx, 0, width_px - 1)]
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def _clipped_zoom(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    """中央を切り出して拡大し、元の大きさに切りそろえる（正本の `clipped_zoom`）。"""
    size = image.shape[0]
    crop = int(np.ceil(size / float(zoom_factor)))
    top = (size - crop) // 2
    zoomed = scipy_zoom(image[top:top + crop, top:top + crop],
                        (zoom_factor, zoom_factor, 1), order=1)
    trim = (zoomed.shape[0] - size) // 2
    return zoomed[trim:trim + size, trim:trim + size]


def zoom_blur(image: np.ndarray, severity: int,
              rng: np.random.Generator) -> np.ndarray:
    """放射状のぶれ。倍率で書かれているので解像度に依らない。"""
    factors = _ZOOM[severity - 1]
    source = (image / 255.0).astype(np.float32)
    accumulated = np.zeros_like(source)
    for factor in factors:
        accumulated += _clipped_zoom(source, factor)
    blended = (source + accumulated) / (len(factors) + 1)
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def _plasma_fractal(rng: np.random.Generator, mapsize: int = 256,
                    wibbledecay: float = 3) -> np.ndarray:
    """正本の `plasma_fractal`（ダイヤモンド・スクエア法）。乱数だけ `rng` から引く。"""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array: np.ndarray) -> np.ndarray:
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares() -> None:
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds() -> None:
        size = maparray.shape[0]
        drgrid = maparray[stepsize // 2:size:stepsize, stepsize // 2:size:stepsize]
        ulgrid = maparray[0:size:stepsize, 0:size:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:size:stepsize, stepsize // 2:size:stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2:size:stepsize, 0:size:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """霧。雲模様を足して全体を明るく薄める。"""
    strength, decay = _FOG[severity - 1]
    source = image / 255.0
    max_val = source.max()
    height, width = source.shape[:2]
    fogged = source + strength * _plasma_fractal(rng, wibbledecay=decay)[:height, :width][..., None]
    return (np.clip(fogged * max_val / (max_val + strength), 0, 1) * 255).astype(np.uint8)


def glass_blur(image: np.ndarray, severity: int,
               rng: np.random.Generator) -> np.ndarray:
    """すりガラス。ぼかして、近傍の画素を写し込み、もう一度ぼかす。

    **正本のループは「入れ替え」ではなく「写し」になっている。**
    `x[h, w], x[h2, w2] = x[h2, w2], x[h, w]` の右辺は numpy の参照なので、
    1 つ目の代入のあとで 2 つ目が自分自身を書き戻すだけになる。
    正本は右下から左上へ 1 画素ずつ写すので、先に写した画素をまた拾うことがあるが、
    **ここでは全画素を同時に、写す前の画像から拾う**（1 枚 200〜700 ms を数 ms にするため）。
    """
    sigma, max_delta, iterations = _GLASS[severity - 1]
    blurred = np.uint8(_skimage_gaussian(image / 255.0, sigma) * 255)
    height, width = blurred.shape[:2]
    # 正本の `range(height - max_delta, max_delta, -1)` と同じ画素を動かす
    rows = np.arange(max_delta + 1, height - max_delta + 1)[:, None]
    cols = np.arange(max_delta + 1, width - max_delta + 1)[None, :]
    for _ in range(iterations):
        # 正本の `np.random.randint(-d, d)` は d を含まない
        dy = rng.integers(-max_delta, max_delta, size=(rows.shape[0], cols.shape[1]))
        dx = rng.integers(-max_delta, max_delta, size=(rows.shape[0], cols.shape[1]))
        source = blurred.copy()
        blurred[rows, cols] = source[rows + dy, cols + dx]
    return (np.clip(_skimage_gaussian(blurred / 255.0, sigma), 0, 1) * 255).astype(np.uint8)
