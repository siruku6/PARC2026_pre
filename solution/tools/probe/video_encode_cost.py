"""Measure video-encoding throughput, the cost that dominates dataset writing.

026 12.4 found LeRobot's own AV1 re-encode path took tens of minutes for 90
episodes. The dataset in use is AV1, crf 30, preset 12. If encoding is the
bottleneck, a regeneration plan has to budget for it or change the codec.
"""

from __future__ import annotations

import io
import time

import numpy as np

FRAMES = 169
SIZE = 256


def synth() -> np.ndarray:
    """Frames with realistic structure: a moving block on a textured background."""
    rng = np.random.default_rng(0)
    background = rng.integers(60, 200, size=(SIZE, SIZE, 3), dtype=np.uint8)
    out = np.repeat(background[None], FRAMES, axis=0).copy()
    for i in range(FRAMES):
        y = 20 + (i % 100)
        out[i, y:y + 40, 60:110] = 255
    return out


def encode(frames: np.ndarray, codec: str, options: dict[str, str]) -> tuple[float, int]:
    import av

    buffer = io.BytesIO()
    start = time.perf_counter()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream(codec, rate=20, options=options)
    stream.width = SIZE
    stream.height = SIZE
    stream.pix_fmt = "yuv420p"
    for frame in frames:
        container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
    container.mux(stream.encode())
    container.close()
    return time.perf_counter() - start, buffer.getbuffer().nbytes


def main() -> None:
    frames = synth()
    trials = [
        ("libsvtav1", {"crf": "30", "preset": "12", "g": "2"}),
        ("libsvtav1", {"crf": "30", "preset": "12"}),
        ("libx264", {"crf": "23", "preset": "veryfast"}),
    ]
    for codec, options in trials:
        try:
            elapsed, size = encode(frames, codec, options)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{codec:<12} {str(options):<46} FAILED: {exc}")
            continue
        print(f"{codec:<12} {str(options):<46} {elapsed:6.2f} s / {FRAMES} frames"
              f"  ({1000 * elapsed / FRAMES:5.1f} ms/frame, {size / 1024:6.0f} KiB)")

    print()
    for codec, options in trials[:1]:
        elapsed, _ = encode(frames, codec, options)
        per_episode = 2 * elapsed  # front + wrist
        print(f"one episode (2 cameras, {codec} preset 12): {per_episode:5.2f} s")
        for count in (1681, 12396, 16810):
            hours = per_episode * count / 3600
            print(f"  {count:6d} episodes: {hours:7.1f} h single process, "
                  f"{hours / 14:6.1f} h on 14 workers")


if __name__ == "__main__":
    main()
