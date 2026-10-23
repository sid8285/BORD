"""Offline tests for the photo intake (bord.ingest)."""

import csv
import io

import pytest
from PIL import Image

from bord.build_manifest import main as build_manifest
from bord.dataset import load_manifest
from bord.ingest import EXIF_IFD, GPS_IFD, has_gps, ingest, strip_all, strip_gps_jpeg, to_clean_jpeg

DATETIME_ORIGINAL = 36867


def phone_jpeg(path, when="2026:10:24 09:00:00", gps=True, orientation=None, size=(64, 48), colour="red"):
    img = Image.new("RGB", size, colour)
    exif = Image.Exif()
    exif[271], exif[272] = "Apple", "iPhone 16"
    exif.get_ifd(EXIF_IFD)[DATETIME_ORIGINAL] = when
    if gps:
        exif.get_ifd(GPS_IFD)[2] = (33.0, 46.0, 30.0)  # GPSLatitude
    if orientation:
        exif[0x0112] = orientation
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", exif=exif, quality=90)
    return path


def test_strip_keeps_the_image_data_and_capture_time(tmp_path):
    src = phone_jpeg(tmp_path / "a.jpg")
    assert has_gps(src)
    raw = src.read_bytes()
    clean = strip_gps_jpeg(raw)
    # Byte-identical from the start of the scan: the pixels were not re-encoded.
    assert clean[clean.index(b"\xff\xda"):] == raw[raw.index(b"\xff\xda"):]
    with Image.open(io.BytesIO(clean)) as img:
        exif = img.getexif()
        assert not exif.get_ifd(GPS_IFD)
        assert exif.get_ifd(EXIF_IFD)[DATETIME_ORIGINAL] == "2026:10:24 09:00:00"
        assert exif[271] == "Apple"
    no_gps = phone_jpeg(tmp_path / "b.jpg", gps=False)
    assert strip_gps_jpeg(no_gps.read_bytes()) == no_gps.read_bytes()


def test_heic_becomes_an_upright_jpeg_without_gps(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    jpg = phone_jpeg(tmp_path / "src.jpg", size=(60, 40))
    with Image.open(jpg) as img:
        exif = img.getexif()
        exif.get_ifd(EXIF_IFD)
        img.save(tmp_path / "IMG_1.HEIC", format="HEIF", exif=exif.tobytes())
    out = to_clean_jpeg(tmp_path / "IMG_1.HEIC")
    with Image.open(io.BytesIO(out)) as img:
        assert img.format == "JPEG" and img.size == (60, 40)
        assert not img.getexif().get_ifd(GPS_IFD)
        assert img.getexif().get_ifd(EXIF_IFD)[DATETIME_ORIGINAL] == "2026:10:24 09:00:00"


def test_ingest_places_photos_and_builds_a_manifest(tmp_path):
    data, inbox = tmp_path / "data", tmp_path / "inbox"
    (data / "pairs").mkdir(parents=True)
    (data / "pairs.csv").write_text("pair_id,task_type,task_description,label,scene_id\n"
                                    "p001,single_scene,Close the laptop,completed,desk\n")
    phone_jpeg(data / "pairs" / "p001" / "before.jpg", "2026:10:01 09:00:00", gps=False)
    phone_jpeg(data / "pairs" / "p001" / "after.jpg", "2026:10:01 10:00:00", gps=False, colour="blue")
    phone_jpeg(inbox / "1", "2026:10:24 09:00:00")
    phone_jpeg(inbox / "2", "2026:10:24 09:05:00", colour="green")
    phone_jpeg(inbox / "3", "2026:10:24 09:06:00", colour="white")
    phone_jpeg(inbox / "4", gps=False, colour="yellow")
    (inbox / "intake.csv").write_text(
        "file,kind,pair_id,task_description,label,scene_id,task_id,source_url\n"
        "1,before,p016,Close the laptop,completed,desk,laptop,\n"
        "2,after,p016,,,,,\n"
        "3,recaptured,p016,,,,,\n"
        "4,web,w01,,,,,https://example.com/clean-room\n")
    # p001 is the same task as p016: give it the same task_id so p016 can resubmit its photo.
    ingest(data, inbox / "intake.csv", inbox)
    rows = {r["pair_id"]: r for r in csv.DictReader((data / "pairs.csv").open())}
    assert rows["p016"]["task_id"] == "laptop" and rows["p016"]["scene_id"] == "desk"
    rows["p001"]["task_id"] = "laptop"
    with (data / "pairs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows["p016"]))
        w.writeheader()
        w.writerows(rows.values())
    assert not strip_all(data)  # nothing left to strip
    assert (data / "web" / "SOURCES.csv").read_text().splitlines()[1] == "w01.jpg,https://example.com/clean-room"

    assert build_manifest(["--data", str(data)]) == 0
    cats = {s.pair_id: s.adversarial_category for s in load_manifest(data / "manifest.csv")}
    assert cats["p016-resubmitted-p001"] == "resubmitted"
    assert cats["p016-recaptured"] == "recaptured"
    assert "web_sourced" in cats.values()


def test_ingest_refuses_a_pair_without_its_label(tmp_path):
    data, inbox = tmp_path / "data", tmp_path / "inbox"
    data.mkdir()
    phone_jpeg(inbox / "1")
    (inbox / "intake.csv").write_text("file,kind,pair_id\n1,before,p020\n")
    with pytest.raises(SystemExit):
        ingest(data, inbox / "intake.csv", inbox)
