"""Tasks, wagers and deadlines (Checkpoint 4).

A task is the unit the user bets on: a description, a wager held through Stripe
(bord.payments), a deadline and a task type (docs/dataset_spec.md):

- ``single_scene``: one before photo and one after photo of the same scene.
- ``multi_view``: the server assigns ``n_views`` distinct camera angles; the user
  photographs each angle before and after, and every angle is verified as its own pair.
- ``time_spread``: a study or workout session of ``session_s`` seconds. The server
  schedules ``n_checkins`` random check-ins inside it; each one must be photographed
  inside its window. The check-in times are never sent to the client ahead of time.

Every time that matters is read from the server clock (``clock`` is injectable for
tests). The deadline is enforced here: a submission that starts after it is refused,
and a task nobody submitted is expired by ``expire_overdue`` and its wager forfeited.

Status flow::

    created -> active -> submitting -> released | forfeited | held
                  |            \\-> active (the pipeline crashed; the user may retry)
                  \\-> expired (deadline passed with no submission; wager forfeited)
    created -> payment_failed (the hold could not be placed)
"""

from __future__ import annotations

import random
import secrets
import threading
import time
from dataclasses import dataclass, field

from .challenge import ANGLES

TASK_TYPES = ("single_scene", "multi_view", "time_spread")

MIN_WAGER_CENTS = 50            # Stripe's minimum charge in USD
MAX_WAGER_CENTS = 10_000        # test mode: $100 keeps a demo from looking like real money
MIN_DEADLINE_S = 60.0
MAX_DEADLINE_S = 7 * 24 * 3600  # a card authorization lasts 7 days (Assumption 5)
MIN_VIEWS, MAX_VIEWS = 2, 4
MIN_CHECKINS, MAX_CHECKINS = 2, 6
MIN_SESSION_S, MAX_SESSION_S = 60.0, 4 * 3600.0
CHECKIN_WINDOW_S = 120.0        # a check-in photo must arrive within this long after it is due
CHECKIN_MIN_SPACING_S = 30.0    # two check-ins are never scheduled closer together than this


class TaskError(Exception):
    """A refused task operation. ``status`` is the HTTP status the API answers with."""

    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class Task:
    id: str
    user_id: str
    description: str
    task_type: str
    wager_cents: int
    created_at: float
    deadline_at: float
    status: str = "created"
    hold_id: str | None = None
    view_prompts: tuple[str, ...] = ()
    session_s: float = 0.0
    n_checkins: int = 0
    session_started_at: float | None = None
    checkin_offsets: tuple[float, ...] = ()         # seconds after session start; server-side only
    checkins: dict[int, dict] = field(default_factory=dict)   # index -> {received_at, capture_id, path}
    decision: str | None = None
    reason: str | None = None
    settled_at: float | None = None

    @property
    def n_views(self) -> int:
        return len(self.view_prompts)

    def public(self, now: float) -> dict:
        """What the browser may see. The check-in schedule is deliberately absent."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "description": self.description,
            "task_type": self.task_type,
            "wager_cents": self.wager_cents,
            "created_at": self.created_at,
            "deadline_at": self.deadline_at,
            "seconds_left": max(0.0, self.deadline_at - now),
            "status": self.status,
            "view_prompts": list(self.view_prompts),
            "session_s": self.session_s,
            "n_checkins": self.n_checkins,
            "session_started_at": self.session_started_at,
            "checkins_done": sorted(self.checkins),
            "checkin_due": self.due_checkin(now),
            "decision": self.decision,
            "reason": self.reason,
        }

    def checkin_window(self, index: int) -> tuple[float, float] | None:
        """[opens, closes] of check-in ``index`` on the server clock, once the session started."""
        if self.session_started_at is None or not 0 <= index < len(self.checkin_offsets):
            return None
        opens = self.session_started_at + self.checkin_offsets[index]
        return opens, opens + CHECKIN_WINDOW_S

    def due_checkin(self, now: float) -> int | None:
        """The check-in the user should photograph right now, or None."""
        for i in range(len(self.checkin_offsets)):
            if i in self.checkins:
                continue
            window = self.checkin_window(i)
            if window and window[0] <= now <= window[1]:
                return i
        return None

    def session_end(self) -> float | None:
        return None if self.session_started_at is None else self.session_started_at + self.session_s

    def missed_checkins(self, now: float) -> list[int]:
        """Check-ins whose window has closed with no accepted photo."""
        missed = []
        for i in range(len(self.checkin_offsets)):
            window = self.checkin_window(i)
            if i not in self.checkins and window and now > window[1]:
                missed.append(i)
        return missed


def schedule_checkins(session_s: float, n: int, rng) -> tuple[float, ...]:
    """``n`` random offsets in the session, at least CHECKIN_MIN_SPACING_S apart, each with a
    full window before the session ends. Rejection sampling; raises if the session is too short."""
    latest = session_s - CHECKIN_WINDOW_S
    needed = CHECKIN_MIN_SPACING_S * (n - 1)
    if latest < needed:
        raise TaskError("session_too_short",
                        f"a {session_s:.0f}s session cannot fit {n} check-ins "
                        f"{CHECKIN_MIN_SPACING_S:.0f}s apart with a {CHECKIN_WINDOW_S:.0f}s window")
    # Spread the slack randomly over the gaps: offsets are the sorted points of a compositions
    # problem, which guarantees the spacing without retries.
    slack = latest - needed
    points = sorted(rng.uniform(0, slack) for _ in range(n))
    return tuple(round(p + CHECKIN_MIN_SPACING_S * i, 1) for i, p in enumerate(points))


class TaskStore:
    """In-memory task registry (one process; a database would replace it)."""

    def __init__(self, clock=time.time, rng=None):
        self.clock = clock
        self.rng = rng or random.SystemRandom()
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    # --- creation ---------------------------------------------------------------------

    def create(self, user_id: str, description: str, task_type: str, wager_cents: int,
               deadline_s: float, n_views: int = 0, session_s: float = 0.0, n_checkins: int = 0) -> Task:
        description = description.strip()
        if not user_id.strip() or not description:
            raise TaskError("missing_field", "user_id and description are required")
        if task_type not in TASK_TYPES:
            raise TaskError("bad_task_type", f"task_type must be one of {TASK_TYPES}")
        if not MIN_WAGER_CENTS <= wager_cents <= MAX_WAGER_CENTS:
            raise TaskError("bad_wager", f"wager must be {MIN_WAGER_CENTS} to {MAX_WAGER_CENTS} cents")
        if not MIN_DEADLINE_S <= deadline_s <= MAX_DEADLINE_S:
            raise TaskError("bad_deadline", f"deadline must be {MIN_DEADLINE_S:.0f} s to 7 days from now")

        views: tuple[str, ...] = ()
        if task_type == "multi_view":
            if not MIN_VIEWS <= n_views <= min(MAX_VIEWS, len(ANGLES)):
                raise TaskError("bad_views", f"n_views must be {MIN_VIEWS} to {min(MAX_VIEWS, len(ANGLES))}")
            views = tuple(self.rng.sample(ANGLES, n_views))
        elif n_views:
            raise TaskError("bad_views", "n_views only applies to multi_view tasks")

        if task_type == "time_spread":
            if not MIN_SESSION_S <= session_s <= MAX_SESSION_S:
                raise TaskError("bad_session", f"session_s must be {MIN_SESSION_S:.0f} to {MAX_SESSION_S:.0f}")
            if not MIN_CHECKINS <= n_checkins <= MAX_CHECKINS:
                raise TaskError("bad_checkins", f"n_checkins must be {MIN_CHECKINS} to {MAX_CHECKINS}")
            if session_s >= deadline_s:
                raise TaskError("bad_session", "the session must end before the deadline")
            schedule_checkins(session_s, n_checkins, self.rng)  # fail now if it cannot fit
        elif session_s or n_checkins:
            raise TaskError("bad_session", "session_s and n_checkins only apply to time_spread tasks")

        now = self.clock()
        task = Task(
            id=secrets.token_urlsafe(9), user_id=user_id, description=description, task_type=task_type,
            wager_cents=wager_cents, created_at=now, deadline_at=now + deadline_s,
            view_prompts=views, session_s=session_s, n_checkins=n_checkins,
        )
        with self._lock:
            self._tasks[task.id] = task
        return task

    def get(self, task_id: str, user_id: str | None = None) -> Task:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise TaskError("unknown_task", "unknown task", 404)
        if user_id is not None and task.user_id != user_id:
            raise TaskError("wrong_user", "task belongs to another user", 403)
        return task

    # --- state changes ----------------------------------------------------------------

    def activate(self, task_id: str, hold_id: str) -> Task:
        with self._lock:
            task = self._must(task_id, {"created"})
            task.hold_id, task.status = hold_id, "active"
            return task

    def payment_failed(self, task_id: str, reason: str) -> Task:
        with self._lock:
            task = self._must(task_id, {"created"})
            task.status, task.reason, task.settled_at = "payment_failed", reason, self.clock()
            return task

    def start_session(self, task_id: str, user_id: str) -> Task:
        """Start the clock of a time-spread task and draw its check-in schedule."""
        with self._lock:
            task = self._must(task_id, {"active"}, user_id)
            if task.task_type != "time_spread":
                raise TaskError("not_time_spread", "only time_spread tasks have a session")
            if task.session_started_at is not None:
                raise TaskError("session_started", "the session already started", 409)
            now = self.clock()
            if now + task.session_s > task.deadline_at:
                raise TaskError("deadline_too_close", "the session would run past the deadline")
            task.session_started_at = now
            task.checkin_offsets = schedule_checkins(task.session_s, task.n_checkins, self.rng)
            return task

    def record_checkin(self, task_id: str, user_id: str, capture_id: str, received_at: float, path: str) -> dict:
        """Accept a check-in photo that arrived at ``received_at`` (the capture's server time)."""
        with self._lock:
            task = self._must(task_id, {"active"}, user_id)
            if task.task_type != "time_spread" or task.session_started_at is None:
                raise TaskError("no_session", "start the session first")
            for i in range(len(task.checkin_offsets)):
                window = task.checkin_window(i)
                if i not in task.checkins and window and window[0] <= received_at <= window[1]:
                    task.checkins[i] = {"capture_id": capture_id, "received_at": received_at, "path": path}
                    return {"index": i, "ok": True, "reason": "ok"}
            return {"index": None, "ok": False, "reason": "no_checkin_due"}

    def begin_submit(self, task_id: str, user_id: str) -> Task:
        """Move an active task to ``submitting``. Refuses after the deadline (server clock)."""
        with self._lock:
            task = self._must(task_id, {"active"}, user_id)
            now = self.clock()
            if now > task.deadline_at:
                raise TaskError("deadline_passed", "the deadline has passed", 409)
            if task.task_type == "time_spread":
                end = task.session_end()
                if end is None:
                    raise TaskError("no_session", "start the session first")
                if now < end:
                    raise TaskError("session_running", f"the session ends in {end - now:.0f}s", 409)
            task.status = "submitting"
            return task

    def abort_submit(self, task_id: str) -> None:
        """The pipeline crashed before a decision: let the user try again."""
        with self._lock:
            task = self._tasks[task_id]
            if task.status == "submitting":
                task.status = "active"

    def settle(self, task_id: str, decision: str, reason: str) -> Task:
        status = {"release": "released", "forfeit": "forfeited", "hold": "held"}[decision]
        with self._lock:
            task = self._must(task_id, {"submitting"})
            task.status, task.decision, task.reason, task.settled_at = status, decision, reason, self.clock()
            return task

    def expire_overdue(self) -> list[Task]:
        """Expire every active task past its deadline; the caller forfeits their holds."""
        now = self.clock()
        expired = []
        with self._lock:
            for task in self._tasks.values():
                if task.status == "active" and now > task.deadline_at:
                    task.status, task.decision = "expired", "forfeit"
                    task.reason, task.settled_at = "deadline passed with no submission", now
                    expired.append(task)
        return expired

    def _must(self, task_id: str, statuses: set[str], user_id: str | None = None) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskError("unknown_task", "unknown task", 404)
        if user_id is not None and task.user_id != user_id:
            raise TaskError("wrong_user", "task belongs to another user", 403)
        if task.status not in statuses:
            raise TaskError("bad_status", f"task is {task.status}", 409)
        return task
