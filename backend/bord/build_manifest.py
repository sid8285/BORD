"""Build data/manifest.csv from the photo folders (Checkpoint 2).

Expected layout under the data directory (see docs/collecting_pairs.md)::

    pairs.csv                         one row per honest pair you photographed
    pairs/<pair_id>/before.jpg        honest photos (.jpg/.jpeg/.png; convert iPhone HEIC first)
    pairs/<pair_id>/after.jpg
    web/*.jpg                         optional: stock or web images (web_sourced)
    adversarial/<pair_id>-staged_partial.jpg   optional, made by hand
    adversarial/<pair_id>-ai_edited.jpg        optional, made by hand

``pairs.csv`` columns: pair_id, task_type, task_description, label, scene_id.
``scene_id`` names the physical scene (e.g. ``desk``); leave it empty to use the
task description. Repeating a task in the same scene on different days is what
lets the script build "reused" attacks.

Adversarial rows generated automatically from the honest pairs:

- ``reused``: a later attempt in the same scene submitted with an earlier
  attempt's "after" photo (the user's own old photo).
- ``wrong_scene``: a pair's "before" with the "after" photo of a different scene.
- ``web_sourced``: a pair's "before" with an image from ``web/``.

Hand-made files in ``adversarial/`` are picked up by name. Every adversarial row
is labeled not_completed and points at its honest base pair.

Usage (from backend/)::

    python -m bord.build_manifest --data ../data
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from .dataset import LABELS, MANIFEST_COLUMNS, TASK_TYPES, load_manifest

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
DATETIME_ORIGINAL = 36867
DATETIME = 306
HAND_MADE = {"staged_partial": "phone", "ai_edited": "generated"}


def capture_time(path: Path) -> str:
    """EXIF capture time as ISO 8601 (no timezone; EXIF has none), or ''."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            raw = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
    except OSError:
        return ""
    if not raw:
        return ""
    try:
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return ""


def has_gps(path: Path) -> bool:
    """True if the photo carries a GPS location. Phone photos usually do, and the repo is public."""
    try:
        with Image.open(path) as img:
            return bool(img.getexif().get_ifd(GPS_IFD))
    except OSError:
        return False


def find_image(folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        for candidate in (folder / f"{stem}{ext}", folder / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                return candidate
    return None


def honest_rows(data: Path) -> tuple[list[dict], list[str]]:
    rows, warnings = [], []
    with (data / "pairs.csv").open(newline="", encoding="utf-8") as fh:
        for n, raw in enumerate(csv.DictReader(fh), start=2):
            r = {k: (v or "").strip() for k, v in raw.items() if k}
            pid = r.get("pair_id", "")
            if not pid:
                continue
            if r.get("label") not in LABELS or r.get("task_type") not in TASK_TYPES:
                raise SystemExit(f"pairs.csv row {n} ({pid}): label must be one of {LABELS}, "
                                 f"task_type one of {TASK_TYPES}")
            folder = data / "pairs" / pid
            before, after = find_image(folder, "before"), find_image(folder, "after")
            if not before or not after:
                raise SystemExit(f"pairs.csv row {n}: need {folder}/before.jpg and after.jpg")
            b_t, a_t = capture_time(before), capture_time(after)
            if b_t and a_t and a_t <= b_t:
                warnings.append(f"{pid}: after photo is not later than before photo ({b_t} vs {a_t})")
            if not (b_t and a_t):
                warnings.append(f"{pid}: no EXIF capture time on one or both photos")
            rows.append({
                "pair_id": pid, "set": "honest", "task_type": r["task_type"],
                "task_description": r["task_description"],
                "before_path": before.relative_to(data).as_posix(),
                "after_path": after.relative_to(data).as_posix(),
                "label": r["label"], "adversarial_category": "", "base_pair_id": "",
                "capture_source": "phone", "before_captured_at": b_t, "after_captured_at": a_t,
                "_scene": r.get("scene_id") or r["task_description"].lower(),
            })
    return rows, warnings


def _attack(base: dict, after_path: str, category: str, suffix: str, source: str) -> dict:
    return {
        "pair_id": f"{base['pair_id']}-{suffix}", "set": "adversarial",
        "task_type": base["task_type"], "task_description": base["task_description"],
        "before_path": base["before_path"], "after_path": after_path,
        "label": "not_completed", "adversarial_category": category,
        "base_pair_id": base["pair_id"], "capture_source": source,
        "before_captured_at": "", "after_captured_at": "",
    }


def adversarial_rows(data: Path, honest: list[dict]) -> list[dict]:
    rows: list[dict] = []
    ordered = sorted(honest, key=lambda r: (r["before_captured_at"] or "", r["pair_id"]))

    # reused: the most recent earlier completed "after" photo from the same scene.
    for i, later in enumerate(ordered):
        earlier = [e for e in ordered[:i] if e["_scene"] == later["_scene"] and e["label"] == "completed"]
        if earlier:
            src = earlier[-1]
            rows.append(_attack(later, src["after_path"], "reused", f"reused-{src['pair_id']}", "phone"))

    # wrong_scene: the next completed pair (in id order) from a different scene.
    by_id = sorted(honest, key=lambda r: r["pair_id"])
    for i, base in enumerate(by_id):
        for step in range(1, len(by_id)):
            other = by_id[(i + step) % len(by_id)]
            if other["_scene"] != base["_scene"] and other["label"] == "completed":
                rows.append(_attack(base, other["after_path"], "wrong_scene", "wrong_scene", "phone"))
                break

    # web_sourced: web images assigned round-robin.
    web_dir = data / "web"
    web = sorted(p for p in web_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS) if web_dir.is_dir() else []
    for i, base in enumerate(by_id if web else []):
        img = web[i % len(web)]
        rows.append(_attack(base, img.relative_to(data).as_posix(), "web_sourced", f"web-{img.stem}", "web"))

    # Hand-made attacks, picked up by file name.
    adv_dir = data / "adversarial"
    for base in by_id:
        for category, source in HAND_MADE.items():
            img = find_image(adv_dir, f"{base['pair_id']}-{category}") if adv_dir.is_dir() else None
            if img:
                rows.append(_attack(base, img.relative_to(data).as_posix(), category, category, source))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the evaluation manifest from data/.")
    parser.add_argument("--data", default=Path("../data"), type=Path)
    parser.add_argument("--out", type=Path, help="default: <data>/manifest.csv")
    args = parser.parse_args(argv)
    out = args.out or args.data / "manifest.csv"

    honest, warnings = honest_rows(args.data)
    attacks = adversarial_rows(args.data, honest)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(honest + attacks)

    load_manifest(out)  # validate against the dataset spec
    counts: dict[str, int] = {}
    for r in attacks:
        counts[r["adversarial_category"]] = counts.get(r["adversarial_category"], 0) + 1
    print(f"wrote {out}: {len(honest)} honest, {len(attacks)} adversarial {counts}")
    image_paths = sorted({args.data / r[k] for r in honest + attacks for k in ("before_path", "after_path")})
    located = [p for p in image_paths if has_gps(p)]
    if located:
        warnings.append(
            f"{len(located)} photo(s) contain a GPS location, e.g. {located[0].relative_to(args.data)}. "
            f"Strip it before committing (the repo is public): exiftool -r -gps:all= -overwrite_original {args.data}"
        )
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
