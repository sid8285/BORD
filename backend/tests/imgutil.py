"""Synthetic photos for the anti-cheating tests (no image files are committed for them).

``smooth_scene(seed)`` is a smooth random colour field: neighbouring cells always differ,
so its dhash is stable under re-encoding, and two different seeds are unrelated scenes.
``changed(img)`` is the same scene after a small change (a white patch, about 1.5% of
the area), like a task that moved one object.
"""

import io
import random

from PIL import Image, ImageDraw


def smooth_scene(seed: int, size: tuple[int, int] = (320, 240)) -> Image.Image:
    rng = random.Random(seed)
    small = Image.new("RGB", (6, 5))
    small.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(30)])
    return small.resize(size, Image.Resampling.BICUBIC)


def changed(img: Image.Image) -> Image.Image:
    out = img.copy()
    ImageDraw.Draw(out).rectangle([140, 100, 175, 130], fill=(250, 250, 250))
    return out


def jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
