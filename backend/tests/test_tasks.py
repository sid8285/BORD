"""Task store: validation, deadlines on the server clock, view prompts, check-in schedule."""

import random

import pytest

from bord.tasks import (CHECKIN_MIN_SPACING_S, CHECKIN_WINDOW_S, MAX_DEADLINE_S, TaskError, TaskStore,
                        schedule_checkins)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def store(clock=None):
    return TaskStore(clock=clock or Clock(), rng=random.Random(7))


def single(s, **kw):
    args = dict(user_id="u1", description="Clear the desk", task_type="single_scene", wager_cents=500, deadline_s=3600)
    args.update(kw)
    return s.create(**args)


def test_create_sets_server_deadline():
    clock = Clock()
    task = single(store(clock))
    assert task.deadline_at == 1000.0 + 3600
    assert task.status == "created"


@pytest.mark.parametrize("kw", [
    dict(wager_cents=10), dict(wager_cents=10**6), dict(deadline_s=5), dict(deadline_s=MAX_DEADLINE_S + 1),
    dict(task_type="nope"), dict(description="  "), dict(n_views=2), dict(session_s=600),
])
def test_create_rejects_bad_input(kw):
    with pytest.raises(TaskError):
        single(store(), **kw)


def test_multi_view_assigns_distinct_angles():
    task = single(store(), task_type="multi_view", n_views=3)
    assert task.n_views == 3 and len(set(task.view_prompts)) == 3
    with pytest.raises(TaskError):
        single(store(), task_type="multi_view", n_views=1)


def test_checkin_schedule_keeps_spacing_and_windows_inside_session():
    for seed in range(50):
        offsets = schedule_checkins(1800, 5, random.Random(seed))
        assert len(offsets) == 5 and list(offsets) == sorted(offsets)
        assert all(b - a >= CHECKIN_MIN_SPACING_S - 0.2 for a, b in zip(offsets, offsets[1:]))  # rounding to 0.1
        assert offsets[0] >= 0 and offsets[-1] + CHECKIN_WINDOW_S <= 1800 + 0.2


def test_checkin_schedule_too_short_session():
    with pytest.raises(TaskError):
        schedule_checkins(150, 4, random.Random(1))


def test_submit_after_deadline_is_refused_and_task_expires():
    clock = Clock()
    s = store(clock)
    task = single(s)
    s.activate(task.id, "hold1")
    clock.t += 3601
    with pytest.raises(TaskError) as e:
        s.begin_submit(task.id, "u1")
    assert e.value.code == "deadline_passed"
    expired = s.expire_overdue()
    assert [t.id for t in expired] == [task.id] and task.status == "expired" and task.decision == "forfeit"
    assert s.expire_overdue() == []  # only once


def test_submit_on_time_then_settle_once():
    clock = Clock()
    s = store(clock)
    task = single(s)
    s.activate(task.id, "hold1")
    clock.t += 3599
    s.begin_submit(task.id, "u1")
    with pytest.raises(TaskError):
        s.begin_submit(task.id, "u1")  # already submitting
    s.settle(task.id, "release", "ok")
    assert task.status == "released"
    with pytest.raises(TaskError):
        s.settle(task.id, "forfeit", "again")


def test_abort_submit_reopens_the_task():
    s = store()
    task = single(s)
    s.activate(task.id, "h")
    s.begin_submit(task.id, "u1")
    s.abort_submit(task.id)
    assert task.status == "active"


def test_other_user_cannot_touch_a_task():
    s = store()
    task = single(s)
    s.activate(task.id, "h")
    with pytest.raises(TaskError) as e:
        s.begin_submit(task.id, "intruder")
    assert e.value.status == 403


def make_session(clock, s, session_s=1200, n=3):
    task = single(s, task_type="time_spread", session_s=session_s, n_checkins=n, deadline_s=7200)
    s.activate(task.id, "h")
    s.start_session(task.id, "u1")
    return task


def test_time_spread_session_and_checkin_windows():
    clock = Clock()
    s = store(clock)
    task = make_session(clock, s)
    assert task.due_checkin(clock.t) is None or task.checkin_offsets[0] == 0
    first_open, first_close = task.checkin_window(0)
    # before the window opens: refused
    r = s.record_checkin(task.id, "u1", "c0", first_open - 1, "p0")
    assert not r["ok"] and r["reason"] == "no_checkin_due"
    # inside the window: accepted once
    clock.t = first_open + 5
    assert task.due_checkin(clock.t) == 0
    assert s.record_checkin(task.id, "u1", "c0", clock.t, "p0")["index"] == 0
    assert not s.record_checkin(task.id, "u1", "c0b", clock.t, "p0b")["ok"]
    # after the window closed with check-in 1 unfilled: missed
    clock.t = task.checkin_window(1)[1] + 1
    assert 1 in task.missed_checkins(clock.t)


def test_public_view_never_reveals_the_schedule():
    clock = Clock()
    s = store(clock)
    task = make_session(clock, s)
    assert "checkin_offsets" not in task.public(clock.t)
    assert all(str(off) not in str(task.public(clock.t)) for off in task.checkin_offsets)


def test_cannot_submit_a_time_spread_task_while_the_session_runs():
    clock = Clock()
    s = store(clock)
    task = make_session(clock, s)
    with pytest.raises(TaskError) as e:
        s.begin_submit(task.id, "u1")
    assert e.value.code == "session_running"
    clock.t += 1200
    s.begin_submit(task.id, "u1")


def test_session_must_fit_before_the_deadline():
    clock = Clock()
    s = store(clock)
    task = single(s, task_type="time_spread", session_s=1200, n_checkins=3, deadline_s=1300)
    s.activate(task.id, "h")
    clock.t += 200
    with pytest.raises(TaskError) as e:
        s.start_session(task.id, "u1")
    assert e.value.code == "deadline_too_close"
