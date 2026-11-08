"""Tasks with wagers through the API: fake verdict model, fake server clock, simulated holds."""

import random

import pytest
from imgutil import changed, jpeg_bytes, smooth_scene

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from bord import stamp, task_checks  # noqa: E402
from bord.api import create_app  # noqa: E402
from bord.challenge import ChallengeStore  # noqa: E402
from bord.payments import SimulatedHolds  # noqa: E402
from bord.tasks import TaskStore  # noqa: E402
from bord.verifier import VerifierResult  # noqa: E402


class FakeModel:
    model = "fake"

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)  # one per call, in order; the last one repeats

    def prepare(self, sub):
        return []

    def call(self, sub, content, timeout_s):
        v = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        return VerifierResult(sub.pair_id, v, 0.9, "looks " + v, 0.1, 100, 10, 0.0, "fake")


class Api:
    def __init__(self, tmp_path, verdicts=("completed",)):
        self.t = [1000.0]
        clock = lambda: self.t[0]  # noqa: E731
        self.holds = SimulatedHolds(clock=clock)
        self.tasks = TaskStore(clock=clock, rng=random.Random(3))
        self.app = create_app(FakeModel(verdicts), tmp_path / "uploads", ChallengeStore(clock=clock),
                              device_log=tmp_path / "d.jsonl", tasks=self.tasks, holds=self.holds)
        self.c = TestClient(self.app)

    def create(self, **kw):
        body = dict(user_id="u1", description="Clear the desk", task_type="single_scene",
                    wager_cents=500, deadline_s=3600)
        body.update(kw)
        r = self.c.post("/api/tasks", json=body)
        assert r.status_code == 200, r.text
        return r.json()["task"]

    def capture(self, role, img, task_id=None, view_index=None, user="u1"):
        req = {"user_id": user, "role": role}
        if task_id is not None:
            req.update(task_id=task_id, view_index=view_index)
        ch = self.c.post("/api/challenges", json=req)
        assert ch.status_code == 200, ch.text
        ch = ch.json()
        r = self.c.post("/api/captures", data={"user_id": user, "role": role, "challenge_id": ch["id"],
                                               "code": ch["code"]},
                        files={"photo": ("c.jpg", jpeg_bytes(stamp.draw(img, ch["code"])), "image/jpeg")})
        assert r.status_code == 200, r.text
        return r.json()["capture_id"], ch

    def submit(self, task_id, befores, afters, user="u1"):
        return self.c.post(f"/api/tasks/{task_id}/submit", json={
            "user_id": user, "before_capture_ids": befores, "after_capture_ids": afters})


def single_pair(api, task, seed=1):
    base = smooth_scene(seed)
    b, _ = api.capture("before", base)
    api.t[0] += 10
    a, _ = api.capture("after", changed(base))
    return b, a


def events(api):
    return [r["event"] for r in api.holds.ledger.rows]


def test_create_task_places_a_hold(tmp_path):
    api = Api(tmp_path)
    task = api.create()
    assert task["status"] == "active" and task["wager_cents"] == 500
    assert events(api) == ["authorized"]


def test_bad_task_is_refused_before_any_hold(tmp_path):
    api = Api(tmp_path)
    r = api.c.post("/api/tasks", json={"user_id": "u1", "description": "x", "wager_cents": 5, "deadline_s": 3600})
    assert r.status_code == 422 and events(api) == []


def test_verified_task_releases_the_hold(tmp_path):
    api = Api(tmp_path)
    task = api.create()
    b, a = single_pair(api, task)
    r = api.submit(task["id"], [b], [a]).json()
    assert r["decision"] == "release" and r["task"]["status"] == "released"
    assert r["money"]["event"] == "hold_cancelled"
    assert events(api) == ["authorized", "released"]


def test_unverified_task_forfeits_to_the_charity(tmp_path):
    api = Api(tmp_path, verdicts=("not_completed",))
    task = api.create()
    b, a = single_pair(api, task)
    r = api.submit(task["id"], [b], [a]).json()
    assert r["decision"] == "forfeit" and r["task"]["status"] == "forfeited"
    assert events(api) == ["authorized", "captured", "charity_payout"]


def test_flagged_release_is_held_and_money_stays_put(tmp_path):
    api = Api(tmp_path)
    task = api.create()
    base = smooth_scene(1)
    b, _ = api.capture("before", base)
    api.t[0] += 10
    a, _ = api.capture("after", smooth_scene(77))  # a different scene: the scene check flags it
    r = api.submit(task["id"], [b], [a]).json()
    assert r["decision"] == "hold" and r["money"]["event"] == "hold_kept"
    assert events(api) == ["authorized"]


def test_submission_after_the_deadline_is_refused_and_forfeits(tmp_path):
    api = Api(tmp_path)
    task = api.create(deadline_s=600)
    b, a = single_pair(api, task)
    api.t[0] += 601  # the capture challenges are 60 s old by now, but the deadline is what is tested
    r = api.submit(task["id"], [b], [a])
    assert r.status_code == 409
    assert events(api) == ["authorized", "captured", "charity_payout"]
    assert api.c.get(f"/api/tasks/{task['id']}", params={"user_id": "u1"}).json()["status"] == "expired"


def test_sweep_forfeits_tasks_nobody_submitted(tmp_path):
    api = Api(tmp_path)
    task = api.create(deadline_s=600)
    api.t[0] += 700
    out = api.c.post("/api/tasks/sweep").json()["expired"]
    assert [e["task_id"] for e in out] == [task["id"]]
    assert api.c.post("/api/tasks/sweep").json()["expired"] == []


def test_a_task_cannot_be_submitted_twice(tmp_path):
    api = Api(tmp_path)
    task = api.create()
    b, a = single_pair(api, task)
    assert api.submit(task["id"], [b], [a]).status_code == 200
    assert api.submit(task["id"], [b], [a]).status_code == 409


def test_another_user_cannot_submit(tmp_path):
    api = Api(tmp_path)
    task = api.create()
    b, a = single_pair(api, task)
    assert api.submit(task["id"], [b], [a], user="intruder").status_code == 403


# --- multi_view -------------------------------------------------------------------------

def multi_view_captures(api, task, same_after=False):
    befores, afters = [], []
    scenes = [smooth_scene(10 + i) for i in range(len(task["view_prompts"]))]
    for i, scene in enumerate(scenes):
        b, ch = api.capture("before", scene, task["id"], i)
        assert ch["angle"] == task["view_prompts"][i]  # the server fixes the angle per view
        befores.append(b)
    api.t[0] += 10
    for i, scene in enumerate(scenes):
        after = changed(scenes[0]) if same_after else changed(scene)
        a, _ = api.capture("after", after, task["id"], i)
        afters.append(a)
    return befores, afters


def test_multi_view_releases_when_every_angle_passes(tmp_path):
    api = Api(tmp_path)
    task = api.create(task_type="multi_view", n_views=3)
    assert len(set(task["view_prompts"])) == 3
    befores, afters = multi_view_captures(api, task)
    r = api.submit(task["id"], befores, afters).json()
    assert r["decision"] == "release" and len(r["pairs"]) == 3
    assert r["task_check"]["ok"]


def test_multi_view_with_disagreeing_angles_is_held(tmp_path):
    api = Api(tmp_path, verdicts=("not_completed", "completed", "not_completed"))
    task = api.create(task_type="multi_view", n_views=3)
    befores, afters = multi_view_captures(api, task)
    r = api.submit(task["id"], befores, afters).json()
    assert r["decision"] == "hold"  # the angles disagree: manual review, no money moves
    assert events(api) == ["authorized"]


def test_multi_view_needs_one_capture_per_angle(tmp_path):
    api = Api(tmp_path)
    task = api.create(task_type="multi_view", n_views=2)
    b0, _ = api.capture("before", smooth_scene(10), task["id"], 0)
    b1, _ = api.capture("before", smooth_scene(11), task["id"], 0)  # angle 0 twice
    api.t[0] += 10
    a0, _ = api.capture("after", changed(smooth_scene(10)), task["id"], 0)
    a1, _ = api.capture("after", changed(smooth_scene(11)), task["id"], 1)
    assert api.submit(task["id"], [b0, b1], [a0, a1]).status_code == 422
    # the refusal reopens the task, so a corrected submission can still go through
    assert api.c.get(f"/api/tasks/{task['id']}", params={"user_id": "u1"}).json()["status"] == "active"


def test_multi_view_with_one_photo_for_every_angle_is_held(tmp_path):
    api = Api(tmp_path)
    task = api.create(task_type="multi_view", n_views=2)
    scene = smooth_scene(10)
    b0, _ = api.capture("before", scene, task["id"], 0)
    b1, _ = api.capture("before", scene, task["id"], 1)
    api.t[0] += 10
    a0, _ = api.capture("after", changed(scene), task["id"], 0)
    a1, _ = api.capture("after", changed(scene), task["id"], 1)  # the same after photo for both angles
    r = api.submit(task["id"], [b0, b1], [a0, a1]).json()
    assert r["decision"] == "hold" and not r["task_check"]["ok"]


def test_view_challenge_requires_a_multi_view_task(tmp_path):
    api = Api(tmp_path)
    task = api.create()
    r = api.c.post("/api/challenges", json={"user_id": "u1", "role": "before", "task_id": task["id"], "view_index": 0})
    assert r.status_code == 422


# --- time_spread ------------------------------------------------------------------------

def test_time_spread_flow(tmp_path, monkeypatch):
    # Identical synthetic check-ins would trip the exact-copy rule; give each file its own hash.
    monkeypatch.setattr(task_checks, "file_dhash", lambda p: hash(str(p)) & ((1 << 64) - 1) | (1 << 40))
    api = Api(tmp_path)
    task = api.create(task_type="time_spread", session_s=1200, n_checkins=3, deadline_s=7200)
    base = smooth_scene(5)
    b, _ = api.capture("before", base)
    assert api.c.post(f"/api/tasks/{task['id']}/session", json={"user_id": "u1"}).status_code == 200
    server_task = api.tasks.get(task["id"])
    assert "checkin_offsets" not in api.c.get(f"/api/tasks/{task['id']}", params={"user_id": "u1"}).json()

    for i in range(3):
        api.t[0] = server_task.session_started_at + server_task.checkin_offsets[i] + 5
        view = api.c.get(f"/api/tasks/{task['id']}", params={"user_id": "u1"}).json()
        assert view["checkin_due"] == i
        cap, _ = api.capture("checkin", changed(base))
        r = api.c.post(f"/api/tasks/{task['id']}/checkins", json={"user_id": "u1", "capture_id": cap}).json()
        assert r["ok"] and r["index"] == i

    api.t[0] = server_task.session_started_at + 1200 + 1
    a, _ = api.capture("after", changed(base))
    r = api.submit(task["id"], [b], [a]).json()
    assert r["decision"] == "release" and r["task_check"]["ok"], r


def test_time_spread_with_a_missed_checkin_is_held(tmp_path):
    api = Api(tmp_path)
    task = api.create(task_type="time_spread", session_s=1200, n_checkins=3, deadline_s=7200)
    base = smooth_scene(5)
    b, _ = api.capture("before", base)
    api.c.post(f"/api/tasks/{task['id']}/session", json={"user_id": "u1"})
    server_task = api.tasks.get(task["id"])
    api.t[0] = server_task.session_started_at + 1201  # nobody photographed any check-in
    a, _ = api.capture("after", changed(base))
    r = api.submit(task["id"], [b], [a]).json()
    assert r["decision"] == "hold" and not r["task_check"]["ok"]
    assert events(api) == ["authorized"]


def test_cannot_submit_while_the_session_is_running(tmp_path):
    api = Api(tmp_path)
    task = api.create(task_type="time_spread", session_s=1200, n_checkins=3, deadline_s=7200)
    base = smooth_scene(5)
    b, _ = api.capture("before", base)
    api.c.post(f"/api/tasks/{task['id']}/session", json={"user_id": "u1"})
    api.t[0] += 10
    a, _ = api.capture("after", changed(base))
    assert api.submit(task["id"], [b], [a]).status_code == 409


def test_checkin_outside_a_window_is_not_counted(tmp_path):
    api = Api(tmp_path)
    task = api.create(task_type="time_spread", session_s=1200, n_checkins=3, deadline_s=7200)
    api.c.post(f"/api/tasks/{task['id']}/session", json={"user_id": "u1"})
    server_task = api.tasks.get(task["id"])
    api.t[0] = server_task.session_started_at + server_task.checkin_offsets[0] - 20
    cap, _ = api.capture("checkin", smooth_scene(5))
    r = api.c.post(f"/api/tasks/{task['id']}/checkins", json={"user_id": "u1", "capture_id": cap}).json()
    assert not r["ok"] and r["reason"] == "no_checkin_due"
