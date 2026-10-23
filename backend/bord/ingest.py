"""Put new photos into data/ (Checkpoint 3): convert, strip GPS, place, and list them.

Photos come off a phone as JPEG or HEIC, often with a GPS location, and with
names that say nothing. This takes an intake list, one line per photo::

    file,kind,pair_id,task_description,label,scene_id,task_id,source_url
    IMG_0412.HEIC,before,p016,Put the dishes in the sink,completed,kitchen,,
    IMG_0413.HEIC,after,p016,,,,,
    IMG_0420.jpg,recaptured,p031,,,,,
    clean1.jpg,web,w01,,,,,https://example.com/photo

``kind`` is ``before`` or ``after`` (an honest pair, written to
data/pairs/<pair_id>/ and to pairs.csv), ``staged_partial``, ``recaptured`` or
``ai_edited`` (data/adversarial/<pair_id>-<kind>.jpg), or ``web``
(data/web/<pair_id>.jpg, with its source in data/web/SOURCES.csv). The pair's
text columns only need to be on one of its lines.

Every photo ends up as a JPEG without a GPS location. A JPEG that has one is
fixed without re-encoding (only its EXIF block is rewritten); a HEIC or PNG is
converted to JPEG (quality 95, upright, EXIF kept minus GPS). Capture times stay,
since bord.build_manifest orders pairs by them.

Usage (from ``backend/``)::

    python -m bord.ingest --data ../data --intake intake.csv --source /path/to/photos
    python -m bord.ingest --data ../data --strip-only        # GPS out of everything in data/
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

from PIL import Image, ImageOps

try:  # optional: iPhone photos are HEIC unless the camera is set to Most Compatible
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - depends on the optional install
    pass

GPS_IFD = 0x8825
EXIF_IFD = 0x8769
ORIENTATION = 0x0112
PAIR_COLUMNS = ("pair_id", "task_type", "task_description", "label", "scene_id", "task_id")
HONEST_KINDS = ("before", "after")
ATTACK_KINDS = ("staged_partial", "recaptured", "ai_edited")
KINDS = (*HONEST_KINDS, *ATTACK_KINDS, "web")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")


def _exif_without_gps(exif: Image.Exif) -> Image.Exif:
    exif.get_ifd(EXIF_IFD)  # load the sub-IFD so it is written back out
    exif.pop(GPS_IFD, None)
    return exif


def strip_gps_jpeg(data: bytes) -> bytes:
    """The same JPEG without its GPS IFD. Only the EXIF segment changes; the image data is
    copied byte for byte. A file without GPS comes back unchanged."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    out, i = [data[:2]], 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is image data
            break
        length = int.from_bytes(data[i + 2:i + 4], "big")
        segment = data[i:i + 2 + length]
        payload = segment[4:]
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            exif = Image.Exif()
            exif.load(payload)
            if GPS_IFD in exif:
                body = _exif_without_gps(exif).tobytes()
                if not body.startswith(b"Exif\x00\x00"):
                    body = b"Exif\x00\x00" + body
                segment = b"\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body
        out.append(segment)
        i += 2 + length
    out.append(data[i:])
    return b"".join(out)


def to_clean_jpeg(src: Path) -> bytes:
    """JPEG bytes for any supported photo, without a GPS location."""
    raw = src.read_bytes()
    with Image.open(io.BytesIO(raw)) as img:
        fmt = img.format
        if fmt == "JPEG":
            return strip_gps_jpeg(raw)
        exif = _exif_without_gps(img.getexif())
        upright = ImageOps.exif_transpose(img).convert("RGB")
    exif[ORIENTATION] = 1  # the pixels are upright now
    buf = io.BytesIO()
    upright.save(buf, format="JPEG", quality=95, exif=exif.tobytes())
    return buf.getvalue()


def has_gps(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            return bool(img.getexif().get_ifd(GPS_IFD))
    except OSError:
        return False


def destination(data: Path, kind: str, pair_id: str) -> Path:
    if kind in HONEST_KINDS:
        return data / "pairs" / pair_id / f"{kind}.jpg"
    if kind == "web":
        return data / "web" / f"{pair_id}.jpg"
    return data / "adversarial" / f"{pair_id}-{kind}.jpg"


def _read_pairs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["pair_id"]: {c: (r.get(c) or "").strip() for c in PAIR_COLUMNS} for r in csv.DictReader(fh)}


def _write_pairs(path: Path, pairs: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(pairs[pid] for pid in sorted(pairs))


def ingest(data: Path, intake: Path, source: Path) -> list[str]:
    """Place every photo in the intake list. Returns one line per photo for the log."""
    with intake.open(newline="", encoding="utf-8") as fh:
        rows = [{k: (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(fh)]
    for n, r in enumerate(rows, start=2):
        if r.get("kind") not in KINDS or not r.get("pair_id") or not r.get("file"):
            raise SystemExit(f"intake row {n}: need file, pair_id and kind (one of {', '.join(KINDS)})")
        if not (source / r["file"]).is_file():
            raise SystemExit(f"intake row {n}: {source / r['file']} not found")

    pairs = _read_pairs(data / "pairs.csv")
    web_sources: list[tuple[str, str]] = []
    log = []
    for r in rows:
        dest = destination(data, r["kind"], r["pair_id"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(to_clean_jpeg(source / r["file"]))
        log.append(f"{r['file']} -> {dest.relative_to(data)}")
        if r["kind"] in HONEST_KINDS:
            entry = pairs.setdefault(r["pair_id"], {c: "" for c in PAIR_COLUMNS} | {
                "pair_id": r["pair_id"], "task_type": "single_scene"})
            for col in ("task_description", "label", "scene_id", "task_id"):
                if r.get(col):
                    entry[col] = r[col]
        elif r["kind"] == "web":
            web_sources.append((dest.name, r.get("source_url", "")))

    incomplete = [p for p, e in pairs.items() if not (e["task_description"] and e["label"])]
    if incomplete:
        raise SystemExit(f"pairs without a task_description or label: {', '.join(sorted(incomplete))}")
    _write_pairs(data / "pairs.csv", pairs)
    if web_sources:
        sources = data / "web" / "SOURCES.csv"
        known = {}
        if sources.exists():
            with sources.open(newline="", encoding="utf-8") as fh:
                known = {r["file"]: r["source_url"] for r in csv.DictReader(fh)}
        known.update(web_sources)
        with sources.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "source_url"])
            writer.writerows(sorted(known.items()))
    return log


def strip_all(data: Path) -> list[Path]:
    """Remove GPS from every JPEG under data/ in place. Returns the files changed."""
    changed = []
    for path in sorted(p for p in data.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")):
        if has_gps(path):
            path.write_bytes(strip_gps_jpeg(path.read_bytes()))
            changed.append(path)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert, strip GPS from, and place new photos in data/.")
    parser.add_argument("--data", default=Path("../data"), type=Path)
    parser.add_argument("--intake", type=Path, help="CSV: file,kind,pair_id,task_description,label,scene_id,task_id,source_url")
    parser.add_argument("--source", type=Path, help="folder the intake's file names are relative to")
    parser.add_argument("--strip-only", action="store_true", help="only remove GPS from the JPEGs already in data/")
    args = parser.parse_args(argv)

    if args.strip_only:
        for path in strip_all(args.data):
            print(f"removed GPS: {path.relative_to(args.data)}")
    else:
        if not args.intake:
            parser.error("--intake is required unless --strip-only")
        for line in ingest(args.data, args.intake, args.source or args.intake.parent):
            print(line)
    left = [p for p in args.data.rglob("*") if p.suffix.lower() in IMAGE_EXTS and has_gps(p)]
    for p in left:
        print(f"warning: still has GPS: {p.relative_to(args.data)}", file=sys.stderr)
    return 1 if left else 0


if __name__ == "__main__":
    raise SystemExit(main())
