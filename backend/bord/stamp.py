"""The capture stamp: the challenge code, drawn into the frame so the server can read it back.

The browser (frontend/src/camera.js) draws the challenge code onto every captured
frame in two forms: as text for the user and the verdict model, and as a strip of
black and white cells that encodes the same code for the server. On upload the
server reads the strip back (``check_stamp``). A photo taken in the app always
carries the code of the challenge it was taken for, so old bytes from an earlier
capture, uploaded again with a new challenge, carry the wrong code.

What it does not stop: a modified client can draw any code on any image, and an
old photo shown on a screen and captured through the app gets a fresh, valid stamp.
Those are left to the reuse and scene checks and the verdict model (docs/anticheat.md).

Layout, in pixels of the captured frame of width W (``jround`` is JavaScript's
Math.round, so both sides compute the same integers)::

    pad = jround(0.02 W), size = jround(0.06 W), box = size + 2 pad   (the text box)
    c = max(4, jround(0.012 W))                                        (one cell)
    strip background: x = pad, y = 2 pad + box, 36c wide, 4c tall, white
    cell i (0..33): x = pad + c + i c, y = 2 pad + box + c, c wide, 2c tall, black for a 1

Cells: 1 0 | 30 code bits (6 characters x 5 bits, each the index in CODE_ALPHABET,
most significant bit first) | 0 1. The fixed cells at both ends check the reading.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageStat

from .challenge import CODE_ALPHABET

BITS_PER_CHAR = 5  # 2**5 = 32 >= len(CODE_ALPHABET)
GUARD_START, GUARD_END = (1, 0), (0, 1)


def jround(x: float) -> int:
    """JavaScript's Math.round for the non-negative values used here."""
    return math.floor(x + 0.5)


@dataclass(frozen=True)
class Layout:
    x0: int
    y0: int
    cell: int

    @classmethod
    def for_width(cls, width: int) -> "Layout":
        pad, size = jround(0.02 * width), jround(0.06 * width)
        return cls(x0=pad, y0=2 * pad + size + 2 * pad, cell=max(4, jround(0.012 * width)))


def encode(code: str) -> list[int]:
    bits = []
    for ch in code.upper():
        index = CODE_ALPHABET.index(ch)  # ValueError for a character outside the alphabet
        bits += [(index >> shift) & 1 for shift in range(BITS_PER_CHAR - 1, -1, -1)]
    return [*GUARD_START, *bits, *GUARD_END]


def decode(cells: list[int]) -> str | None:
    if tuple(cells[:2]) != GUARD_START or tuple(cells[-2:]) != GUARD_END:
        return None
    bits = cells[2:-2]
    if len(bits) % BITS_PER_CHAR:
        return None
    chars = []
    for i in range(0, len(bits), BITS_PER_CHAR):
        index = int("".join(map(str, bits[i:i + BITS_PER_CHAR])), 2)
        if index >= len(CODE_ALPHABET):
            return None
        chars.append(CODE_ALPHABET[index])
    return "".join(chars)


def draw(img: Image.Image, code: str) -> Image.Image:
    """Python twin of the browser's drawing (tests and tooling): the strip only."""
    out = img.copy()
    lay = Layout.for_width(out.width)
    cells = encode(code)
    d = ImageDraw.Draw(out)
    d.rectangle((lay.x0, lay.y0, lay.x0 + 36 * lay.cell - 1, lay.y0 + 4 * lay.cell - 1), fill=(255, 255, 255))
    for i, bit in enumerate(cells):
        if bit:
            x = lay.x0 + lay.cell + i * lay.cell
            d.rectangle((x, lay.y0 + lay.cell, x + lay.cell - 1, lay.y0 + 3 * lay.cell - 1), fill=(0, 0, 0))
    return out


def read(path: str | Path, code_length: int = 6) -> str | None:
    """The code in a photo's strip, or None when there is no readable strip."""
    with Image.open(path) as img:
        width = img.width  # the layout is defined on the full-size frame
        lay = Layout.for_width(width)
        if lay.cell >= 12:  # JPEG: decode at half size when a cell still spans 6+ pixels (faster)
            img.draft("L", (img.width // 2, img.height // 2))
        gray = ImageOps.grayscale(img)
    scale = gray.width / width
    n_cells = len(GUARD_START) + code_length * BITS_PER_CHAR + len(GUARD_END)

    def mean(x0: float, y0: float, x1: float, y1: float) -> float:
        # The middle half of the box, in decoded pixels, away from the JPEG-blurred edges.
        mx, my = (x1 - x0) / 4, (y1 - y0) / 4
        box = tuple(round(v * scale) for v in (x0 + mx, y0 + my, x1 - mx, y1 - my))
        if box[2] <= box[0] or box[3] <= box[1] or box[2] > gray.width or box[3] > gray.height:
            raise ValueError("stamp is outside the frame")
        return ImageStat.Stat(gray.crop(box)).mean[0]

    c, top = lay.cell, lay.y0 + lay.cell
    try:
        white = mean(lay.x0, lay.y0, lay.x0 + c, lay.y0 + 4 * c)  # the quiet column left of the cells
        levels = [mean(lay.x0 + c + i * c, top, lay.x0 + 2 * c + i * c, top + 2 * c) for i in range(n_cells)]
    except ValueError:
        return None
    black = levels[0]  # the first guard cell is always black
    if white - black < 64:  # no strip here: the area is not a white field with a black cell
        return None
    threshold = (white + black) / 2
    return decode([1 if v < threshold else 0 for v in levels])


@dataclass(frozen=True)
class StampCheck:
    expected: str
    found: str | None

    @property
    def ok(self) -> bool:
        return self.found is not None and self.found == self.expected.strip().upper()

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def check_stamp(path: str | Path, expected: str) -> StampCheck:
    try:
        found = read(path, code_length=len(expected.strip()))
    except (OSError, ValueError):
        found = None
    return StampCheck(expected, found)
