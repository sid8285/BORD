"""Offline tests for the manifest builder."""

from PIL import Image

from bord.build_manifest import main
from bord.dataset import load_manifest


def _photo(path, when=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 24), "white")
    exif = Image.Exif()
    if when:
        exif.get_ifd(0x8769)[36867] = when
    img.save(path, exif=exif)


def test_builds_all_attack_types(tmp_path):
    data = tmp_path
    (data / "pairs.csv").write_text(
        "pair_id,task_type,task_description,label,scene_id\n"
        "p01,single_scene,Clean my desk,completed,desk\n"
        "p02,single_scene,Clean my desk,completed,desk\n"
        "p03,single_scene,Make my bed,completed,bed\n"
        "p04,single_scene,Make my bed,not_completed,bed\n"
    )
    days = {"p01": "01", "p02": "02", "p03": "01", "p04": "03"}
    for pid, d in days.items():
        _photo(data / "pairs" / pid / "before.jpg", f"2026:10:{d} 09:00:00")
        _photo(data / "pairs" / pid / "after.jpg", f"2026:10:{d} 10:00:00")
    _photo(data / "web" / "stock1.jpg")
    _photo(data / "adversarial" / "p01-staged_partial.jpg")
    _photo(data / "adversarial" / "p03-ai_edited.png")

    assert main(["--data", str(data)]) == 0
    subs = load_manifest(data / "manifest.csv")
    by_id = {s.pair_id: s for s in subs}
    cats = sorted(s.adversarial_category for s in subs if s.is_adversarial)
    assert cats.count("reused") == 2  # p02 reuses p01, p04 reuses p03
    assert by_id["p02-reused-p01"].after_path == data / "pairs/p01/after.jpg"
    assert by_id["p02-reused-p01"].before_path == data / "pairs/p02/before.jpg"
    assert cats.count("wrong_scene") == 4 and cats.count("web_sourced") == 4
    assert by_id["p01-wrong_scene"].after_path.parent.name == "p03"  # other scene
    assert by_id["p01-staged_partial"].capture_source == "phone"
    assert by_id["p03-ai_edited"].capture_source == "generated"
    assert by_id["p01"].before_captured_at == "2026-10-01T09:00:00"
    assert all(s.expected_verdict == "not_completed" for s in subs if s.is_adversarial)


def test_warns_about_gps_location(tmp_path, capsys):
    (tmp_path / "pairs.csv").write_text("pair_id,task_type,task_description,label,scene_id\n"
                                        "p01,single_scene,Clean my desk,completed,desk\n")
    _photo(tmp_path / "pairs" / "p01" / "before.jpg", "2026:10:01 09:00:00")
    img = Image.new("RGB", (32, 24), "white")
    exif = Image.Exif()
    exif.get_ifd(0x8825)[2] = (33.0, 46.0, 30.0)  # GPSLatitude
    img.save(tmp_path / "pairs" / "p01" / "after.jpg", exif=exif)
    assert main(["--data", str(tmp_path)]) == 0
    assert "GPS location" in capsys.readouterr().err
