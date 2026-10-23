"""Offline tests for the fingerprints and the anti-cheating signals."""

from pathlib import Path

import pytest
from imgutil import changed, jpeg_bytes, smooth_scene
from PIL import Image

from bord.anticheat import AntiCheat, AntiCheatConfig, ChallengeEvidence, ManifestHistory
from bord.challenge import ChallengeStore
from bord.dataset import Submission
from bord.detectors import HFImageDetector, file_metadata, looks_camera_made
from bord.imaging import file_dhash, hamming, load_thumb, scene_similarity


def save(tmp_path: Path, name: str, img: Image.Image, quality: int = 95) -> Path:
    p = tmp_path / name
    p.write_bytes(jpeg_bytes(img, quality))
    return p


def sub(before: Path, after: Path, pair_id="p01", source="phone", base=None, adversarial=False) -> Submission:
    return Submission(pair_id, "adversarial" if adversarial else "honest", "single_scene", "Clear the desk",
                      before, after, "not_completed" if adversarial else "completed",
                      "reused" if adversarial else None, base, source, "", "")


# --- fingerprints -----------------------------------------------------------------

def test_dhash_survives_reencoding_and_resizing(tmp_path):
    scene = smooth_scene(1)
    a = save(tmp_path, "a.jpg", scene, 95)
    b = save(tmp_path, "b.jpg", scene.resize((160, 120)), 70)
    c = save(tmp_path, "c.jpg", smooth_scene(2))
    assert hamming(file_dhash(a), file_dhash(a)) == 0
    assert hamming(file_dhash(a), file_dhash(b)) <= 8   # same photo, smaller and recompressed
    assert hamming(file_dhash(a), file_dhash(c)) >= 10  # a different scene


def test_thumbnail_cache_follows_the_file(tmp_path):
    p = save(tmp_path, "a.jpg", smooth_scene(1))
    first = file_dhash(p)
    save(tmp_path, "a.jpg", smooth_scene(2))
    import os
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 5_000_000_000))  # a later mtime
    assert file_dhash(p) != first


def test_scene_similarity_same_scene_vs_other(tmp_path):
    scene = smooth_scene(3)
    before = load_thumb(save(tmp_path, "b.jpg", scene))
    after = load_thumb(save(tmp_path, "a.jpg", changed(scene)))
    other = load_thumb(save(tmp_path, "o.jpg", smooth_scene(4)))
    same = scene_similarity(before, after)
    different = scene_similarity(before, other)
    assert same["score"] > 0.8 and same["correlation"] > 0.9
    assert different["score"] < same["score"] - 0.2


def test_flat_image_has_no_scene_information(tmp_path):
    flat = load_thumb(save(tmp_path, "f.jpg", Image.new("RGB", (320, 240), "white")))
    scene = load_thumb(save(tmp_path, "s.jpg", smooth_scene(5)))
    assert scene_similarity(flat, scene)["correlation"] == 0.0


# --- signals ------------------------------------------------------------------------

class FakeDetector:
    name = "fake"

    def __init__(self, p):
        self.p = p

    def score(self, path):
        return self.p


def test_reuse_flags_a_photo_already_submitted(tmp_path):
    scene = smooth_scene(1)
    old_after = save(tmp_path, "old_after.jpg", changed(scene))
    before = save(tmp_path, "before.jpg", smooth_scene(2))
    reused = save(tmp_path, "reused.jpg", changed(scene), 70)  # re-encoded copy of the old photo
    fresh = save(tmp_path, "fresh.jpg", changed(smooth_scene(2)))

    # Wider than the default 3 so the test does not hinge on exact JPEG noise; fresh photos are far away.
    ac = AntiCheat(AntiCheatConfig(use_metadata=False, reuse_max_distance=8))
    assert ac.check(sub(before, reused), [old_after]).signal("reuse").flagged is True
    s = ac.check(sub(before, fresh), [old_after]).signal("reuse")
    assert s.flagged is False and s.score >= 10
    assert ac.check(sub(before, fresh), []).signal("reuse").flagged is False


def test_scene_consistency_flags_a_different_scene(tmp_path):
    scene = smooth_scene(6)
    before = save(tmp_path, "before.jpg", scene)
    ok_after = save(tmp_path, "ok.jpg", changed(scene))
    wrong_after = save(tmp_path, "wrong.jpg", smooth_scene(7))
    ac = AntiCheat(AntiCheatConfig(use_metadata=False))
    assert ac.check(sub(before, ok_after)).signal("scene_consistency").flagged is False
    # Threshold sits between the two scores, so the test does not depend on the default value.
    wrong = ac.check(sub(before, wrong_after)).signal("scene_consistency")
    ok = ac.check(sub(before, ok_after)).signal("scene_consistency")
    assert wrong.score < ok.score
    strict = AntiCheat(AntiCheatConfig(use_metadata=False, scene_min_score=(wrong.score + ok.score) / 2))
    assert strict.check(sub(before, wrong_after)).signal("scene_consistency").flagged is True
    assert strict.check(sub(before, ok_after)).signal("scene_consistency").flagged is False


def test_ai_image_signal_uses_the_detector(tmp_path):
    scene = smooth_scene(8)
    before, after = save(tmp_path, "b.jpg", scene), save(tmp_path, "a.jpg", changed(scene))
    none = AntiCheat(AntiCheatConfig(use_metadata=False)).check(sub(before, after)).signal("ai_image")
    assert none.flagged is None  # not applicable without a detector
    hot = AntiCheat(AntiCheatConfig(use_metadata=False), FakeDetector(0.9)).check(sub(before, after))
    assert hot.signal("ai_image").flagged is True and hot.flagged and hot.flagged_by == ["ai_image"]
    cold = AntiCheat(AntiCheatConfig(use_metadata=False), FakeDetector(0.1)).check(sub(before, after))
    assert cold.signal("ai_image").flagged is False and not cold.flagged


def test_challenge_and_timestamp_signals(tmp_path):
    scene = smooth_scene(9)
    before, after = save(tmp_path, "b.jpg", scene), save(tmp_path, "a.jpg", changed(scene))

    t = [100.0]
    store = ChallengeStore(clock=lambda: t[0])
    cb = store.issue("u1", "before")
    t[0] += 1
    rb = store.redeem(cb.id, cb.code, "u1", "before")
    t[0] += 1
    ca = store.issue("u1", "after")
    t[0] += 10
    ra = store.redeem(ca.id, ca.code, "u1", "after")

    ac = AntiCheat(AntiCheatConfig(use_metadata=False))
    good = ac.check(sub(before, after, source="browser"), (), ChallengeEvidence(rb, ra))
    assert good.signal("challenge").flagged is False
    assert good.signal("server_timestamps").flagged is False and not good.flagged

    bad_after = store.redeem(ca.id, ca.code, "u1", "after")  # second use of the same challenge
    report = ac.check(sub(before, after, source="browser"), (), ChallengeEvidence(rb, bad_after))
    assert report.signal("challenge").flagged is True
    assert "already_used" in report.signal("challenge").detail and report.flagged

    offline = ac.check(sub(before, after))
    assert offline.signal("challenge").flagged is None and offline.signal("server_timestamps").flagged is None
    # No stamp evidence (offline rows, or the Checkpoint 3 branch's two-field evidence): not applicable.
    assert offline.signal("stamp").flagged is None and good.signal("stamp").flagged is None


def test_stamp_signal():
    from bord.stamp import StampCheck

    t = [100.0]
    store = ChallengeStore(clock=lambda: t[0])
    cb, ca = store.issue("u1", "before"), store.issue("u1", "after")
    rb = store.redeem(cb.id, cb.code, "u1", "before")
    t[0] += 20
    ra = store.redeem(ca.id, ca.code, "u1", "after")
    ac = AntiCheat(AntiCheatConfig(use_metadata=False))

    def stamp_signal(after_found):
        evidence = ChallengeEvidence(rb, ra, StampCheck(cb.code, cb.code), StampCheck(ca.code, after_found))
        # Nothing to hash here, so pass no photos: only the evidence-based signals matter.
        return ac._stamp(evidence)

    assert stamp_signal(ca.code).flagged is False
    old = stamp_signal("ABCDEF" if ca.code != "ABCDEF" else "BCDEFG")
    assert old.flagged is True and "after: read" in old.detail
    assert stamp_signal(None).flagged is True


def test_a_check_that_cannot_run_fails_closed(tmp_path):
    before = save(tmp_path, "b.jpg", smooth_scene(1))
    report = AntiCheat(AntiCheatConfig(use_metadata=False)).check(sub(before, tmp_path / "missing.jpg"), [before])
    assert report.signal("scene_consistency").flagged is True
    assert report.signal("reuse").flagged is True
    assert "check failed" in report.signal("reuse").detail and report.flagged


def test_metadata_is_diagnostic_only(tmp_path):
    scene = smooth_scene(10)
    before, after = save(tmp_path, "b.jpg", scene), save(tmp_path, "a.jpg", changed(scene))
    meta = file_metadata(after)
    assert meta == {"camera_make": False, "camera_model": False, "capture_time": False, "c2pa_marker": False}
    assert not looks_camera_made(meta)
    report = AntiCheat().check(sub(before, after))  # saved by Pillow: no camera EXIF
    signal = report.signal("file_metadata")
    assert signal.flagged is True and signal.diagnostic
    assert not report.flagged  # a diagnostic never flags a submission
    browser = AntiCheat().check(sub(before, after, source="browser")).signal("file_metadata")
    assert browser.flagged is None

    exif = Image.Exif()
    exif[271], exif[272], exif[306] = "Apple", "iPhone 16 Pro", "2026:10:11 20:11:24"
    camera = tmp_path / "camera.jpg"
    changed(scene).save(camera, exif=exif)
    assert looks_camera_made(file_metadata(camera))


def test_hf_detector_sums_the_ai_labels():
    def fake_pipeline(image, top_k=None):
        return [{"label": "artificial", "score": 0.7}, {"label": "human", "score": 0.3}]

    detector = HFImageDetector(pipeline=fake_pipeline)
    assert detector.name == "hf:umm-maybe/AI-image-detector"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jpg"
        p.write_bytes(jpeg_bytes(smooth_scene(1)))
        assert detector.score(p) == pytest.approx(0.7)


# --- offline benchmark history -----------------------------------------------------

def test_manifest_history_is_earlier_honest_after_photos(tmp_path):
    def row(pid, t, base=None, adv=False):
        s = sub(Path(f"{pid}-b.jpg"), Path(f"{pid}-a.jpg"), pid, base=base, adversarial=adv)
        return Submission(**{**s.__dict__, "before_captured_at": t})

    subs = [row("p01", "2026-10-11T10:00:00"), row("p02", "2026-10-11T11:00:00"),
            row("p03", "2026-10-11T12:00:00"), row("p03-reused-p01", "", base="p03", adv=True)]
    history = ManifestHistory(subs)
    assert history(subs[0]) == []
    assert history(subs[2]) == [Path("p01-a.jpg"), Path("p02-a.jpg")]
    # An attack is judged with the history of the pair it was built on.
    assert history(subs[3]) == history(subs[2])
