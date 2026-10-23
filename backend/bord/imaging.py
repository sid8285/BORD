"""Cheap image fingerprints for the anti-cheating stage (Checkpoint 3).

Everything here uses Pillow only, so the 0.5 s anti-cheating stage of
docs/realtime_budget.md does not depend on OpenCV or numpy. A photo is decoded
once at a reduced JPEG scale (``Image.draft``), shrunk to two tiny thumbnails
and cached, so the user's past photos are hashed a single time.

Two fingerprints are computed from the thumbnails:

- ``dhash``: a 64-bit difference hash. Two files of the same photo, even
  re-encoded or resized, differ in a few bits at most. Used for the reuse check.
- ``scene_similarity``: a 0..1 score of how alike two photos are as a whole
  (blurred luminance correlation plus colour histogram overlap). Used to check
  that the AFTER photo shows the same scene as the BEFORE photo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

GRAY_SIDE = 64   # luminance thumbnail, GRAY_SIDE x GRAY_SIDE
RGB_SIDE = 32    # colour thumbnail, RGB_SIDE x RGB_SIDE
CORR_SIDE = 16   # luminance is blurred down to this before correlating
HASH_SIDE = 8    # dhash is HASH_SIDE x HASH_SIDE = 64 bits
HIST_LEVELS = 4  # colour histogram has HIST_LEVELS ** 3 bins


@dataclass(frozen=True)
class Thumb:
    gray: bytes  # GRAY_SIDE x GRAY_SIDE, mode L, row-major
    rgb: bytes   # RGB_SIDE x RGB_SIDE, mode RGB, row-major


@lru_cache(maxsize=512)
def _load_thumb(path: str, mtime_ns: int) -> Thumb:
    # mtime_ns is part of the cache key so an overwritten file is not served stale.
    with Image.open(path) as img:
        # JPEG only: decode at a reduced DCT scale that is still >= 256 px.
        img.draft("RGB", (256, 256))
        img = ImageOps.exif_transpose(img).convert("RGB")
        gray = img.convert("L").resize((GRAY_SIDE, GRAY_SIDE), Image.Resampling.BOX)
        rgb = img.resize((RGB_SIDE, RGB_SIDE), Image.Resampling.BOX)
        return Thumb(gray.tobytes(), rgb.tobytes())


def load_thumb(path: str | Path) -> Thumb:
    p = Path(path)
    return _load_thumb(str(p), p.stat().st_mtime_ns)


def dhash(thumb: Thumb) -> int:
    """64-bit difference hash: is each pixel brighter than its right neighbour?"""
    small = Image.frombytes("L", (GRAY_SIDE, GRAY_SIDE), thumb.gray).resize(
        (HASH_SIDE + 1, HASH_SIDE), Image.Resampling.BOX
    )
    data = small.tobytes()
    width = HASH_SIDE + 1
    bits = 0
    for row in range(HASH_SIDE):
        for col in range(HASH_SIDE):
            bits = (bits << 1) | (1 if data[row * width + col] > data[row * width + col + 1] else 0)
    return bits


@lru_cache(maxsize=2048)
def _file_dhash(path: str, mtime_ns: int) -> int:
    return dhash(_load_thumb(path, mtime_ns))


def file_dhash(path: str | Path) -> int:
    """dhash of a photo on disk; cached, so a user's past photos are hashed once."""
    p = Path(path)
    return _file_dhash(str(p), p.stat().st_mtime_ns)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _pearson(a: bytes, b: bytes) -> float:
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = var_a = var_b = 0.0
    for x, y in zip(a, b):
        dx, dy = x - mean_a, y - mean_b
        cov += dx * dy
        var_a += dx * dx
        var_b += dy * dy
    if var_a == 0.0 or var_b == 0.0:
        return 0.0  # a flat image carries no scene information
    return cov / math.sqrt(var_a * var_b)


def _blurred(thumb: Thumb) -> bytes:
    return Image.frombytes("L", (GRAY_SIDE, GRAY_SIDE), thumb.gray).resize(
        (CORR_SIDE, CORR_SIDE), Image.Resampling.BOX
    ).tobytes()


def _histogram(thumb: Thumb) -> list[float]:
    step = 256 // HIST_LEVELS
    counts = [0] * HIST_LEVELS ** 3
    data = thumb.rgb
    for i in range(0, len(data), 3):
        r, g, b = data[i] // step, data[i + 1] // step, data[i + 2] // step
        counts[(r * HIST_LEVELS + g) * HIST_LEVELS + b] += 1
    total = len(data) // 3
    return [c / total for c in counts]


def scene_similarity(a: Thumb, b: Thumb) -> dict:
    """How alike two photos are as a whole, 0 (unrelated) to 1 (same view).

    ``score`` mixes the correlation of the blurred luminance (layout survives a
    small hand-held shift once blurred to 16x16) with the overlap of the colour
    histograms (same room, same palette). Both parts are returned so the
    threshold can be calibrated from data (bord.anticheat_eval).
    """
    correlation = max(0.0, _pearson(_blurred(a), _blurred(b)))
    ha, hb = _histogram(a), _histogram(b)
    overlap = sum(min(x, y) for x, y in zip(ha, hb))
    return {"score": 0.6 * correlation + 0.4 * overlap, "correlation": correlation, "histogram": overlap}
