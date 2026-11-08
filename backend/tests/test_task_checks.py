"""Cross-photo checks for multi_view and time_spread tasks."""

from pathlib import Path

from imgutil import changed, jpeg_bytes, smooth_scene

from bord import task_checks
from bord.realtime import FORFEIT, HOLD, RELEASE
from bord.task_checks import check_multi_view, check_time_spread, combine_decisions, photos_distinct
from bord.tasks import Task


def save(tmp_path, name, img):
    p = tmp_path / name
    p.write_bytes(jpeg_bytes(img))
    return p


def test_combine_decisions_is_conservative():
    assert combine_decisions([RELEASE, RELEASE]) == RELEASE
    assert combine_decisions([FORFEIT, FORFEIT]) == FORFEIT
    assert combine_decisions([RELEASE, FORFEIT]) == HOLD
    assert combine_decisions([RELEASE, HOLD]) == HOLD
    assert combine_decisions([]) == HOLD


def test_one_photo_cannot_serve_two_views(tmp_path):
    a = save(tmp_path, "a.jpg", smooth_scene(1))
    b = save(tmp_path, "b.jpg", smooth_scene(1))      # the same photo again
    c = save(tmp_path, "c.jpg", smooth_scene(2))
    assert not check_multi_view([a, b]).ok
    assert check_multi_view([a, c]).ok
    assert not check_multi_view([a]).ok


def test_photos_distinct_reports_the_pair(tmp_path):
    a = save(tmp_path, "a.jpg", smooth_scene(1))
    b = save(tmp_path, "b.jpg", smooth_scene(3))
    c = save(tmp_path, "c.jpg", smooth_scene(1))
    r = photos_distinct([a, b, c])
    assert not r.ok and "photos 1 and 3" in r.detail


def session_task(checkins):
    t = Task(id="t", user_id="u", description="study", task_type="time_spread", wager_cents=500,
             created_at=0, deadline_at=9999, session_s=1200, n_checkins=len(checkins))
    t.checkin_offsets = tuple(float(100 * (i + 1)) for i in range(3))
    t.checkins = {i: {"capture_id": f"c{i}", "received_at": 0.0, "path": str(p)} for i, p in checkins.items()}
    return t


def test_time_spread_needs_every_checkin(tmp_path, monkeypatch):
    base = smooth_scene(5)
    before = save(tmp_path, "before.jpg", base)
    c0 = save(tmp_path, "c0.jpg", changed(base))
    r = check_time_spread(session_task({0: c0}), before)
    assert not r.ok and "[2, 3]" in r.detail


def test_time_spread_accepts_same_scene_checkins(tmp_path, monkeypatch):
    base = smooth_scene(5)
    before = save(tmp_path, "before.jpg", base)
    paths = {i: save(tmp_path, f"c{i}.jpg", changed(base)) for i in range(3)}
    # Real check-ins of one desk differ by hand shake; the synthetic ones are identical, so
    # give each path its own hash (the exact-copy rule is tested separately below).
    monkeypatch.setattr(task_checks, "file_dhash", lambda p: hash(Path(p).name) & ((1 << 64) - 1) | (1 << 40))
    # hash() of different short strings differs in many bits with overwhelming probability
    r = check_time_spread(session_task(paths), before)
    assert r.ok, r.detail


def test_time_spread_rejects_a_checkin_of_another_scene(tmp_path, monkeypatch):
    base = smooth_scene(5)
    before = save(tmp_path, "before.jpg", base)
    paths = {0: save(tmp_path, "c0.jpg", changed(base)), 1: save(tmp_path, "c1.jpg", smooth_scene(99)),
             2: save(tmp_path, "c2.jpg", changed(base))}
    r = check_time_spread(session_task(paths), before)
    assert not r.ok and "check-in 2" in r.detail


def test_time_spread_rejects_a_repeated_checkin_photo(tmp_path):
    base = smooth_scene(5)
    before = save(tmp_path, "before.jpg", base)
    same = changed(base)
    paths = {i: save(tmp_path, f"c{i}.jpg", same) for i in range(3)}
    r = check_time_spread(session_task(paths), before)
    assert not r.ok and "check-ins" in r.detail
