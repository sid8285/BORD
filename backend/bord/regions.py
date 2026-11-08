"""Per-region before/after difference check for staged-partial attacks (Checkpoint 4).

A staged-partial submission does part of the task and frames the AFTER photo so the
rest is out of view or rearranged (Checkpoint 3: 2 of 3 fooled Claude Opus 5.5). The
whole-image ``scene_consistency`` score cannot see it: the scene really is the same.
This check looks region by region instead:

1. **Align.** Both photos are reduced to ``ALIGN_SIDE`` px grayscale, histogram-equalized,
   and the AFTER photo is searched over a small grid of scales and shifts for the
   transform that best matches the BEFORE photo (lowest mean absolute difference). A
   hand-held re-shot of the same scene moves a few percent; a re-framed shot moves more,
   and the search records how far (``scale``, ``shift``).
2. **Grid.** The overlapping area is split into ``GRID`` x ``GRID`` cells. For each cell
   the check measures how much it changed (mean absolute luminance difference) and how
   busy it is before and after (edge density, a cheap proxy for clutter).
3. **Features.** ``changed_frac``: share of cells that changed. ``busy_unchanged_frac``:
   of the cells that were busy BEFORE, the share that did not change (clutter left in
   place). ``clutter_ratio``: total edge density AFTER / BEFORE (a tidying task should
   lower it). ``reframe``: how far the best alignment had to move the AFTER photo.

The signal flags when the clutter is still there (``clutter_ratio`` at or above
``max_clutter_ratio``) **and** a large share of the busy cells did not change
(``busy_unchanged_frac`` at or above ``max_busy_unchanged``), or when the AFTER photo
had to be re-framed beyond ``max_reframe`` to match. Like every signal it can only turn
a release into a hold. The thresholds are starting points, measured with
``bord.region_eval`` on the honest set and the staged-partial attacks; Assumption 7 in
docs/assumptions.md states the target and the number of attacks needed.

Pillow only (C-level ``ImageChops``/``ImageStat``), so it fits inside the 0.5 s
anti-cheating stage of docs/realtime_budget.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

ALIGN_SIDE = 96           # px, both photos are compared at this size
GRID = 4                  # GRID x GRID cells
SCALES = (0.85, 0.92, 1.0, 1.08, 1.17)
SHIFTS = (-12, -8, -4, 0, 4, 8, 12)   # px at ALIGN_SIDE, i.e. up to 12.5% of the frame
MARGIN = 12               # px cropped on every side after alignment (max shift)


@dataclass(frozen=True)
class RegionConfig:
    change_min: float = 22.0        # mean |diff| (0..255) above which a cell counts as changed
    max_clutter_ratio: float = 0.9  # AFTER edge density / BEFORE at or above this: clutter still there
    max_busy_unchanged: float = 0.5 # share of busy BEFORE cells left unchanged at or above this
    max_reframe: float = 0.2        # |scale - 1| + |shift| / side beyond this: the AFTER photo was re-framed


@dataclass(frozen=True)
class RegionReport:
    flagged: bool
    changed_frac: float
    busy_unchanged_frac: float
    clutter_ratio: float
    reframe: float
    scale: float
    shift: tuple[int, int]
    align_mad: float
    cells: list[dict]
    reasons: tuple[str, ...]

    def summary(self) -> str:
        why = "; ".join(self.reasons) if self.reasons else "no staged-partial pattern"
        return (f"changed {self.changed_frac:.2f} of cells, busy-unchanged {self.busy_unchanged_frac:.2f}, "
                f"clutter ratio {self.clutter_ratio:.2f}, re-frame {self.reframe:.2f} ({why})")

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=256)
def _prepared(path: str, mtime_ns: int) -> Image.Image:
    with Image.open(path) as img:
        img.draft("RGB", (ALIGN_SIDE * 4, ALIGN_SIDE * 4))
        img = ImageOps.exif_transpose(img).convert("L")
        small = img.resize((ALIGN_SIDE, ALIGN_SIDE), Image.Resampling.BOX)
        return ImageOps.equalize(small)


def _load(path: str | Path) -> Image.Image:
    p = Path(path)
    return _prepared(str(p), p.stat().st_mtime_ns)


def _transform(img: Image.Image, scale: float, dx: int, dy: int) -> Image.Image:
    """Scale about the centre, then shift; the uncovered border is cropped later."""
    side = ALIGN_SIDE
    c = side / 2
    # PIL's affine maps output -> input coordinates.
    a = 1 / scale
    return img.transform(
        (side, side), Image.Transform.AFFINE,
        (a, 0, c - a * (c + dx), 0, a, c - a * (c + dy)),
        resample=Image.Resampling.BILINEAR,
    )


def _crop(img: Image.Image) -> Image.Image:
    return img.crop((MARGIN, MARGIN, ALIGN_SIDE - MARGIN, ALIGN_SIDE - MARGIN))


def align(before: Image.Image, after: Image.Image) -> tuple[float, int, int, float]:
    """Best (scale, dx, dy, mean |diff|) of AFTER onto BEFORE over the search grid."""
    ref = _crop(before)
    best = (1.0, 0, 0, float("inf"))
    for s in SCALES:
        for dx in SHIFTS:
            for dy in SHIFTS:
                cand = _crop(_transform(after, s, dx, dy))
                mad = ImageStat.Stat(ImageChops.difference(ref, cand)).mean[0]
                if mad < best[3]:
                    best = (s, dx, dy, mad)
    return best


def _edges(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.FIND_EDGES)


def region_report(before_path: str | Path, after_path: str | Path,
                  config: RegionConfig = RegionConfig()) -> RegionReport:
    before, after = _load(before_path), _load(after_path)
    scale, dx, dy, mad = align(before, after)
    b = _crop(before)
    a = _crop(_transform(after, scale, dx, dy))
    diff = ImageChops.difference(b, a)
    eb, ea = _edges(b), _edges(a)

    side = b.width
    step = side // GRID
    cells = []
    for r in range(GRID):
        for c in range(GRID):
            box = (c * step, r * step, (c + 1) * step, (r + 1) * step)
            cells.append({
                "row": r, "col": c,
                "diff": ImageStat.Stat(diff.crop(box)).mean[0],
                "busy_before": ImageStat.Stat(eb.crop(box)).mean[0],
                "busy_after": ImageStat.Stat(ea.crop(box)).mean[0],
            })

    busy_cut = sorted(x["busy_before"] for x in cells)[len(cells) // 2]  # median
    busy = [x for x in cells if x["busy_before"] >= busy_cut]
    changed = [x for x in cells if x["diff"] >= config.change_min]
    busy_unchanged = [x for x in busy if x["diff"] < config.change_min]
    total_before = sum(x["busy_before"] for x in cells) or 1e-9
    clutter_ratio = sum(x["busy_after"] for x in cells) / total_before
    reframe = abs(scale - 1.0) + (abs(dx) + abs(dy)) / ALIGN_SIDE

    changed_frac = len(changed) / len(cells)
    busy_unchanged_frac = len(busy_unchanged) / len(busy) if busy else 0.0
    reasons = []
    if clutter_ratio >= config.max_clutter_ratio and busy_unchanged_frac >= config.max_busy_unchanged:
        reasons.append(f"clutter kept (ratio {clutter_ratio:.2f}) and {busy_unchanged_frac:.0%} "
                       f"of busy regions unchanged")
    if reframe > config.max_reframe:
        reasons.append(f"AFTER re-framed by {reframe:.2f} (flag above {config.max_reframe})")
    return RegionReport(
        flagged=bool(reasons), changed_frac=changed_frac, busy_unchanged_frac=busy_unchanged_frac,
        clutter_ratio=clutter_ratio, reframe=reframe, scale=scale, shift=(dx, dy), align_mad=mad,
        cells=cells, reasons=tuple(reasons),
    )
