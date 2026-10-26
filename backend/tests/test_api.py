"""The capture and verification API, with a fake verdict model and a fake server clock."""

import json

import pytest
from imgutil import changed, jpeg_bytes, smooth_scene

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from bord.api import create_app  # noqa: E402
from bord import stamp  # noqa: E402
from bord.challenge import ChallengeStore  # noqa: E402
from bord.verifier import VerifierResult  # noqa: E402


class FakeModel:
    model = "fake"

    def __init__(self, verdict="completed"):
        self.verdict = verdict

    def prepare(self, sub):
        return []

    def call(self, sub, content, timeout_s):
        return VerifierResult(sub.pair_id, self.verdict, 0.9, "the desk is clear", 0.1, 100, 10, 0.0, "fake")


class Session:
    """Drives the API like the frontend does."""

    def __init__(self, tmp_path, verdict="completed"):
        self.t = [1000.0]
        store = ChallengeStore(clock=lambda: self.t[0])
        self.app = create_app(FakeModel(verdict), tmp_path / "uploads", store, device_log=tmp_path / "devices.jsonl")
        self.client = TestClient(self.app)

    def capture(self, role, photo, user="u1", code=None):
        """``photo`` is a PIL image, stamped with the challenge code the way the browser does
        it, or raw bytes uploaded as they are (an old capture, a tampered client)."""
        c = self.client.post("/api/challenges", json={"user_id": user, "role": role}).json()
        if not isinstance(photo, bytes):
            photo = jpeg_bytes(stamp.draw(photo, c["code"]))
        r = self.client.post(
            "/api/captures",
            data={"user_id": user, "role": role, "challenge_id": c["id"], "code": code or c["code"]},
            files={"photo": ("capture.jpg", photo, "image/jpeg")},
        )
        assert r.status_code == 200, r.text
        return r.json()

    def submit(self, before_id, after_id, user="u1", task="Clear the desk"):
        return self.client.post("/api/submissions", json={
            "user_id": user, "task_description": task,
            "before_capture_id": before_id, "after_capture_id": after_id})

    def task(self, scene_seed, user="u1", after_photo=None):
        scene = smooth_scene(scene_seed)
        before = self.capture("before", scene, user)
        self.t[0] += 20  # the user does the task
        after = self.capture("after", after_photo or changed(scene), user)
        return before, after


def test_happy_path_releases(tmp_path):
    s = Session(tmp_path)
    before, after = s.task(1)
    assert before["challenge_ok"] and after["challenge_ok"]
    assert after["received_at"] - before["received_at"] == pytest.approx(20)
    body = s.submit(before["capture_id"], after["capture_id"]).json()
    assert body["decision"] == "release" and body["verdict"] == "completed"
    assert body["flagged_by"] == [] and not body["deadline_missed"]
    names = {sig["name"]: sig for sig in body["signals"]}
    assert names["challenge"]["flagged"] is False and names["server_timestamps"]["flagged"] is False
    assert names["stamp"]["flagged"] is False and before["stamp_ok"] and after["stamp_ok"]
    assert {"preprocess_s", "anticheat_s", "llm_s"} <= set(body["stages"])


def test_wrong_code_holds_the_release(tmp_path):
    s = Session(tmp_path)
    scene = smooth_scene(2)
    before = s.capture("before", scene, code="NOPE22")
    assert before["challenge_ok"] is False and before["reason"] == "wrong_code"
    s.t[0] += 20
    after = s.capture("after", changed(scene))
    body = s.submit(before["capture_id"], after["capture_id"]).json()
    assert body["verdict"] == "completed" and body["decision"] == "hold"
    assert "challenge" in body["flagged_by"]


def test_instant_pair_fails_the_server_timestamps(tmp_path):
    s = Session(tmp_path)
    scene = smooth_scene(3)
    before = s.capture("before", scene)
    s.t[0] += 0.5  # an after photo half a second later: no time to do a task
    after = s.capture("after", changed(scene))
    body = s.submit(before["capture_id"], after["capture_id"]).json()
    assert body["decision"] == "hold" and "server_timestamps" in body["flagged_by"]


def test_resubmitting_an_old_after_photo_is_flagged_as_reused(tmp_path):
    s = Session(tmp_path)
    scene = smooth_scene(4)
    old_after = changed(scene)
    b1, a1 = s.task(4, after_photo=old_after)
    assert s.submit(b1["capture_id"], a1["capture_id"]).json()["decision"] == "release"

    s.t[0] += 3600
    # The same finished-desk photo again, a new task, freshly stamped (a tampered client
    # or a photo of a screen): the stamp passes and the reuse check has to catch it.
    b2, a2 = s.task(4, after_photo=old_after)
    body = s.submit(b2["capture_id"], a2["capture_id"]).json()
    assert body["decision"] == "hold" and body["flagged_by"] == ["reuse"]


def test_an_old_capture_uploaded_again_carries_the_wrong_stamp(tmp_path):
    s = Session(tmp_path)
    scene = smooth_scene(9)
    b1, a1 = s.task(9)
    old_bytes = (tmp_path / "uploads" / f"{a1['capture_id']}.jpg").read_bytes()  # stamped with the old code
    assert s.submit(b1["capture_id"], a1["capture_id"]).json()["decision"] == "release"

    s.t[0] += 3600
    before = s.capture("before", scene)
    s.t[0] += 20
    after = s.capture("after", old_bytes)  # the right new code typed in, the old photo uploaded
    assert after["challenge_ok"] is True and after["stamp_ok"] is False
    body = s.submit(before["capture_id"], after["capture_id"]).json()
    assert body["decision"] == "hold" and "stamp" in body["flagged_by"]


def test_an_unstamped_photo_is_flagged(tmp_path):
    s = Session(tmp_path)
    scene = smooth_scene(10)
    before = s.capture("before", scene)
    s.t[0] += 20
    after = s.capture("after", jpeg_bytes(changed(scene)))  # uploaded without the browser's stamp
    assert after["stamp_ok"] is False
    body = s.submit(before["capture_id"], after["capture_id"]).json()
    assert body["decision"] == "hold" and body["flagged_by"] == ["stamp"]


def test_a_capture_backs_one_submission(tmp_path):
    s = Session(tmp_path)
    before, after = s.task(5)
    assert s.submit(before["capture_id"], after["capture_id"]).status_code == 200
    again = s.submit(before["capture_id"], after["capture_id"])
    assert again.status_code == 404


def test_captures_belong_to_their_user(tmp_path):
    s = Session(tmp_path)
    before, after = s.task(6, user="u1")
    assert s.submit(before["capture_id"], after["capture_id"], user="u2").status_code == 403


def test_roles_must_be_before_then_after(tmp_path):
    s = Session(tmp_path)
    before, after = s.task(7)
    assert s.submit(after["capture_id"], before["capture_id"]).status_code == 422


def test_rejects_things_that_are_not_photos(tmp_path):
    s = Session(tmp_path)
    c = s.client.post("/api/challenges", json={"user_id": "u1", "role": "before"}).json()
    r = s.client.post("/api/captures",
                      data={"user_id": "u1", "role": "before", "challenge_id": c["id"], "code": c["code"]},
                      files={"photo": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422
    assert s.client.post("/api/challenges", json={"user_id": "u1", "role": "middle"}).status_code == 422
    assert s.client.post("/api/submissions", json={
        "user_id": "u1", "task_description": "x", "before_capture_id": "a", "after_capture_id": "b"}).status_code == 404


def test_device_report_inspects_the_bytes(tmp_path):
    s = Session(tmp_path)
    r = s.client.post("/api/device-reports", data={"user_agent": "TestPhone/1.0", "label": "test phone"},
                      files={"photo": ("frame.jpg", jpeg_bytes(smooth_scene(8)), "image/jpeg")})
    assert r.status_code == 200
    report = r.json()
    assert report["carries_exif"] is False and report["exif_tag_count"] == 0 and report["format"] == "JPEG"
    logged = [json.loads(line) for line in (tmp_path / "devices.jsonl").read_text().splitlines()]
    assert logged[0]["user_agent"] == "TestPhone/1.0" and logged[0]["label"] == "test phone"
    assert list((tmp_path / "uploads").glob("probe-*")) == []  # the probe photo is deleted
    assert logged[0]["capture_ms"] is None


def test_device_report_logs_the_capture_time(tmp_path):
    s = Session(tmp_path)
    r = s.client.post("/api/device-reports", data={"user_agent": "TestPhone/1.0", "capture_ms": "42"},
                      files={"photo": ("frame.jpg", jpeg_bytes(smooth_scene(8)), "image/jpeg")})
    assert r.status_code == 200 and r.json()["capture_ms"] == 42.0
