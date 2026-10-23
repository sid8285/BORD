"""The capture stamp: the browser draws it, the server reads it back (bord.stamp)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from imgutil import jpeg_bytes, smooth_scene

from bord import stamp
from bord.challenge import CODE_ALPHABET

CAMERA_JS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "camera.js"


def test_encode_decode_round_trip():
    for code in ("ABCDEF", "Z23456", "K7QX2M", CODE_ALPHABET[-6:]):
        cells = stamp.encode(code)
        assert len(cells) == 34 and stamp.decode(cells) == code
    assert stamp.decode([0, 0] + stamp.encode("ABCDEF")[2:]) is None  # broken guard
    with pytest.raises(ValueError):
        stamp.encode("ABCDE0")  # 0 is not in the alphabet


@pytest.mark.parametrize("size", [(320, 240), (1280, 960), (1920, 1440), (1440, 1920)])
def test_reads_the_code_back_from_a_compressed_photo(tmp_path, size):
    path = tmp_path / "capture.jpg"
    path.write_bytes(jpeg_bytes(stamp.draw(smooth_scene(3, size), "K7QX2M"), quality=60))
    assert stamp.read(path) == "K7QX2M"
    assert stamp.check_stamp(path, "k7qx2m").ok


def test_no_stamp_reads_as_none(tmp_path):
    path = tmp_path / "plain.jpg"
    path.write_bytes(jpeg_bytes(smooth_scene(4, (1280, 960))))
    assert stamp.read(path) is None
    check = stamp.check_stamp(path, "K7QX2M")
    assert not check.ok and check.to_dict() == {"expected": "K7QX2M", "found": None, "ok": False}


def test_wrong_code_is_not_ok(tmp_path):
    path = tmp_path / "old.jpg"
    path.write_bytes(jpeg_bytes(stamp.draw(smooth_scene(5, (1280, 960)), "ABCDEF")))
    check = stamp.check_stamp(path, "K7QX2M")
    assert check.found == "ABCDEF" and not check.ok


def test_frontend_uses_the_same_alphabet():
    assert f'export const CODE_ALPHABET = "{CODE_ALPHABET}";' in CAMERA_JS.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_frontend_encodes_the_same_cells():
    script = f"import {{stampCells}} from {json.dumps(CAMERA_JS.as_uri())}; " \
             "console.log(JSON.stringify(['ABCDEF', 'K7QX2M', 'Z23456'].map(stampCells)));"
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == [stamp.encode(c) for c in ("ABCDEF", "K7QX2M", "Z23456")]
