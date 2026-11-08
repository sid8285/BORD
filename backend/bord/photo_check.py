"""A checkable record that the committed photos carry no GPS location (Checkpoint 4).

The Checkpoint 3 grader noted that photo-based claims cannot be checked from the report's
text. This walks every image under ``data/`` and writes one line per file to
``results/photo_check_<time>/photos.jsonl`` and a ``summary.json``:

- ``sha256`` and size, so the line can be matched to the exact committed file;
- ``gps``: whether the file has a GPS IFD, and how many GPS tags it holds (must be 0);
- the camera EXIF fields (make, model, capture time) and C2PA marker from
  bord.detectors.file_metadata, which show the photos are otherwise left as shot;
- the pixel size.

It exits with status 1 if any photo still has a GPS location, so it can run in CI.
Usage (from ``backend/``)::

    python -m bord.photo_check --data ../data --out ../results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .detectors import file_metadata
from .ingest import GPS_IFD

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def gps_tags(path: Path) -> int:
    with Image.open(path) as img:
        return len(img.getexif().get_ifd(GPS_IFD))


def check(path: Path, root: Path) -> dict:
    data = path.read_bytes()
    with Image.open(path) as img:
        size = list(img.size)
    n_gps = gps_tags(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "pixels": size,
        "gps": n_gps > 0,
        "gps_tags": n_gps,
        **file_metadata(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that no committed photo has a GPS location.")
    parser.add_argument("--data", default=Path("data"), type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    args = parser.parse_args(argv)

    files = sorted(p for p in args.data.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    started = datetime.now(timezone.utc)
    out_dir = args.out / ("photo_check_" + started.strftime("%Y%m%dT%H%M%SZ"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [check(p, args.data) for p in files]
    with (out_dir / "photos.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with_gps = [r["path"] for r in rows if r["gps"]]
    by_folder: dict[str, int] = {}
    for r in rows:
        folder = r["path"].split("/")[0]
        by_folder[folder] = by_folder.get(folder, 0) + 1
    summary = {
        "checked_at": started.isoformat(),
        "files": len(rows),
        "by_folder": by_folder,
        "with_gps": with_gps,
        "camera_exif_kept": sum(1 for r in rows if r["camera_make"] and r["capture_time"]),
        "all_gps_free": not with_gps,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0 if not with_gps else 1


if __name__ == "__main__":
    raise SystemExit(main())
