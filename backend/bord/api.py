"""FastAPI service for camera-only capture and verification (Checkpoint 3).

The flow, all on the server's clock::

    POST /api/challenges   {user_id, role}                -> code + angle prompt
    POST /api/captures     form: user_id, role, challenge_id, code, photo
                                                          -> capture_id (challenge redeemed now)
    POST /api/submissions  {user_id, task_description, before_capture_id, after_capture_id}
                                                          -> verdict and decision

The browser (frontend/src/CameraCapture.jsx) draws the challenge code onto each
captured frame and uploads it right away, so ``received_at`` is when the photo
was taken as far as the server can tell. There is no file-upload path for photos:
the only way a photo reaches ``/api/captures`` is the capture page, and the server
never trusts a client filename or timestamp. The decision is the Checkpoint 2
real-time pipeline plus the anti-cheating stage (bord.realtime, bord.anticheat).
``/api/submissions`` stops at the decision (Checkpoint 3). Checkpoint 4 adds tasks that
carry a wager held through Stripe test mode (bord.tasks, bord.payments)::

    POST /api/tasks                    {user_id, description, task_type, wager_cents, deadline_s, ...}
                                                          -> task (the hold is placed first)
    GET  /api/tasks/{id}?user_id=      -> task, with the check-in due right now (time_spread)
    POST /api/tasks/{id}/session       {user_id}          -> starts a time-spread session
    POST /api/tasks/{id}/checkins      {user_id, capture_id}
    POST /api/tasks/{id}/submit        {user_id, before_capture_ids, after_capture_ids}
                                                          -> decision, and the hold is cancelled
                                                             (release), captured (forfeit) or kept (hold)
    POST /api/tasks/sweep              -> forfeits the wagers of tasks past their deadline

Run from ``backend/``::

    uvicorn bord.api:app --port 8000
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .anticheat import AntiCheat, ChallengeEvidence
from .challenge import ROLES, ChallengeStore, Redemption
from .dataset import Submission
from .detectors import file_metadata
from .payments import HoldError, default_holds
from .qwen_verifier import QwenVerifier
from .realtime import FORFEIT, HOLD, RELEASE, Budget, RealtimeVerifier
from .stamp import StampCheck, check_stamp
from .task_checks import check_multi_view, check_time_spread, combine_decisions
from .tasks import TaskError, TaskStore

MAX_PHOTO_BYTES = 15 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class ChallengeRequest(BaseModel):
    user_id: str
    role: str
    task_id: str | None = None      # multi_view: the task whose angle prompt this capture follows
    view_index: int | None = None   # multi_view: which of the task's angles


class TaskRequest(BaseModel):
    user_id: str
    description: str
    task_type: str = "single_scene"
    wager_cents: int
    deadline_s: float               # seconds from now, on the server clock
    n_views: int = 0                # multi_view
    session_s: float = 0.0          # time_spread
    n_checkins: int = 0             # time_spread


class UserRequest(BaseModel):
    user_id: str


class CheckinRequest(BaseModel):
    user_id: str
    capture_id: str


class TaskSubmitRequest(BaseModel):
    user_id: str
    before_capture_ids: list[str]   # one per view for multi_view (in any order), else one
    after_capture_ids: list[str]


class SubmissionRequest(BaseModel):
    user_id: str
    task_description: str
    before_capture_id: str
    after_capture_id: str


@dataclass(frozen=True)
class Capture:
    id: str
    user_id: str
    role: str
    path: Path
    redemption: Redemption
    stamp: StampCheck
    task_id: str | None = None      # set when the challenge was issued for a multi_view angle
    view_index: int | None = None


class AppState:
    """Everything the endpoints share. Replaced by a database in Checkpoint 4."""

    def __init__(self, verifier, upload_dir: Path, store: ChallengeStore, anticheat: AntiCheat, budget: Budget,
                 device_log: Path, tasks: TaskStore, holds):
        self.upload_dir = upload_dir
        self.device_log = device_log
        self.store = store
        self.tasks = tasks
        self.holds = holds
        self.challenge_views: dict[str, tuple[str, int]] = {}  # challenge id -> (task id, view index)
        self.anticheat = anticheat
        self.captures: dict[str, Capture] = {}
        self.history: dict[str, list[Path]] = {}  # user -> AFTER photos already submitted, oldest first
        self._pending_history: dict[str, list[Path]] = {}  # pair_id -> history for the call in flight
        self.lock = threading.Lock()
        self.pipeline = RealtimeVerifier(
            verifier, budget=budget, anticheat=anticheat,
            history_for=lambda sub: self._pending_history.get(sub.pair_id, []),
        )

    def run_pipeline(self, sub: Submission, history: list[Path], evidence: ChallengeEvidence):
        self._pending_history[sub.pair_id] = history
        try:
            return self.pipeline.verify(sub, evidence)
        finally:
            self._pending_history.pop(sub.pair_id, None)


def _default_verifier():
    return QwenVerifier(
        provider=os.environ.get("BORD_PROVIDER", "ollama"),
        model=os.environ.get("BORD_MODEL") or None,
        max_image_side=int(os.environ.get("BORD_IMAGE_SIDE", "1568")),
        timeout_s=Budget().llm_s,
    )


def create_app(verifier=None, upload_dir: Path | None = None, store: ChallengeStore | None = None,
               anticheat: AntiCheat | None = None, device_log: Path | None = None,
               tasks: TaskStore | None = None, holds=None) -> FastAPI:
    """Build the app. Tests pass a fake verifier, a store with a fake clock and simulated holds."""
    state = AppState(
        verifier or _default_verifier(),
        upload_dir or Path(os.environ.get("BORD_UPLOADS", "uploads")),
        store or ChallengeStore(),
        anticheat or AntiCheat(),
        Budget(),
        device_log or Path(os.environ.get("BORD_DEVICE_LOG", "../data/device_tests.jsonl")),
        tasks or TaskStore(),
        holds or default_holds(ledger_path=Path(os.environ.get("BORD_LEDGER", "../data/ledger.jsonl"))),
    )
    state.upload_dir.mkdir(parents=True, exist_ok=True)
    state.device_log.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="BORD")
    app.state.bord = state

    def _task(task_id: str, user_id: str | None = None):
        try:
            return state.tasks.get(task_id, user_id)
        except TaskError as e:
            raise HTTPException(e.status, str(e)) from e

    def _settle_hold(task, decision: str) -> str | None:
        """Move the money for a decision; returns an error text if Stripe failed.
        A hold decision moves nothing: the authorization stays for manual review."""
        try:
            if decision == RELEASE:
                state.holds.release(task.hold_id, task.id)
            elif decision == FORFEIT:
                state.holds.forfeit(task.hold_id, task.id)
        except HoldError as e:
            return str(e)
        return None

    @app.post("/api/challenges")
    def issue_challenge(req: ChallengeRequest) -> dict:
        if req.role not in ROLES:
            raise HTTPException(422, f"role must be one of {ROLES}")
        angle = None
        if req.task_id is not None:
            task = _task(req.task_id, req.user_id)
            if task.task_type != "multi_view" or req.role not in ("before", "after"):
                raise HTTPException(422, "task_id only applies to before/after captures of a multi_view task")
            if req.view_index is None or not 0 <= req.view_index < task.n_views:
                raise HTTPException(422, f"view_index must be 0 to {task.n_views - 1}")
            angle = task.view_prompts[req.view_index]
        challenge = state.store.issue(req.user_id, req.role, angle)
        if req.task_id is not None:
            with state.lock:
                state.challenge_views[challenge.id] = (req.task_id, req.view_index)
        return challenge.public()

    @app.post("/api/captures")
    def upload_capture(
        user_id: str = Form(...),
        role: str = Form(...),
        challenge_id: str = Form(...),
        code: str = Form(...),
        photo: UploadFile = File(...),
    ) -> dict:
        if role not in ROLES:
            raise HTTPException(422, f"role must be one of {ROLES}")
        # Redeem first: the server timestamp is when the capture arrived, not when it was decoded.
        redemption = state.store.redeem(challenge_id, code, user_id, role)
        data = photo.file.read(MAX_PHOTO_BYTES + 1)
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(413, "photo too large")
        capture_id = secrets.token_urlsafe(10)
        path = state.upload_dir / f"{capture_id}.img"
        path.write_bytes(data)
        try:
            with Image.open(path) as img:
                if img.format not in ALLOWED_FORMATS:
                    raise ValueError(f"unsupported format {img.format}")
                img.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            path.unlink(missing_ok=True)
            raise HTTPException(422, f"not a usable photo: {e}") from e
        # Pillow chooses the decoder from the bytes, not the name; keep a matching extension.
        with Image.open(path) as img:
            suffix = ".jpg" if img.format == "JPEG" else ".png"
        final = path.with_suffix(suffix)
        path.rename(final)
        # Read the code back out of the pixels. Done on upload, so it is off the verdict's
        # critical path; the anti-cheating stage only compares the stored result.
        stamp = check_stamp(final, code)
        with state.lock:
            task_id, view_index = state.challenge_views.pop(challenge_id, (None, None))
            state.captures[capture_id] = Capture(capture_id, user_id, role, final, redemption, stamp,
                                                 task_id, view_index)
        return {"capture_id": capture_id, "challenge_ok": redemption.ok, "reason": redemption.reason,
                "received_at": redemption.received_at, "stamp_ok": stamp.ok}

    @app.post("/api/submissions")
    def submit(req: SubmissionRequest) -> dict:
        with state.lock:
            before = state.captures.get(req.before_capture_id)
            after = state.captures.get(req.after_capture_id)
            if before is None or after is None:
                raise HTTPException(404, "unknown or already used capture")
            if before.user_id != req.user_id or after.user_id != req.user_id:
                raise HTTPException(403, "capture belongs to another user")
            if before.role != "before" or after.role != "after":
                raise HTTPException(422, "need a before capture and an after capture")
            # A capture can back one submission only.
            del state.captures[before.id], state.captures[after.id]
            history = list(state.history.get(req.user_id, []))

        def iso(ts: float) -> str:
            return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")

        sub = Submission(
            pair_id=f"{req.user_id}-{secrets.token_hex(4)}",
            set="honest",  # ground truth is unknown in production; the label below is not used
            task_type="single_scene",
            task_description=req.task_description,
            before_path=before.path,
            after_path=after.path,
            label="completed",
            adversarial_category=None,
            base_pair_id=None,
            capture_source="browser",
            before_captured_at=iso(before.redemption.received_at),
            after_captured_at=iso(after.redemption.received_at),
        )
        evidence = ChallengeEvidence(before.redemption, after.redemption, before.stamp, after.stamp)
        result = state.run_pipeline(sub, history, evidence)
        with state.lock:
            state.history.setdefault(req.user_id, []).append(after.path)
        return {
            "submission_id": sub.pair_id,
            "decision": result.decision,        # release | forfeit | hold
            "verdict": result.verdict,
            "confidence": result.confidence,
            "reason": result.reason,
            "error": result.error,
            "deadline_missed": result.deadline_missed,
            "flagged_by": result.anticheat.get("flagged_by", []),
            "signals": result.anticheat.get("signals", []),
            "stages": result.stages,
            "server_s": result.latency_s,
        }

    @app.post("/api/device-reports")
    def device_report(
        user_agent: str = Form(...),
        label: str = Form(""),
        capture_ms: float | None = Form(None),
        photo: UploadFile = File(...),
    ) -> dict:
        """Assumption 3 device test: does a frame captured with getUserMedia carry EXIF?

        The frontend's /?device-test page posts one captured frame from each phone. The server
        inspects the bytes itself rather than trusting a client-side claim, appends one line to
        the device log (commit it as evidence) and deletes the photo.
        """
        data = photo.file.read(MAX_PHOTO_BYTES + 1)
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(413, "photo too large")
        probe = state.upload_dir / f"probe-{secrets.token_hex(6)}.img"
        probe.write_bytes(data)
        try:
            with Image.open(probe) as img:
                fmt, size, exif_tags = img.format, list(img.size), len(img.getexif())
            meta = file_metadata(probe)
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            raise HTTPException(422, f"not a usable photo: {e}") from e
        finally:
            probe.unlink(missing_ok=True)
        record = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "user_agent": user_agent[:300], "label": label[:100], "format": fmt, "size": size,
            "bytes": len(data), "exif_tag_count": exif_tags, **meta,
            "carries_exif": exif_tags > 0,
            "capture_ms": capture_ms,
        }
        with state.lock, state.device_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    # --- tasks, wagers and deadlines (Checkpoint 4) ---------------------------------------

    def _sweep() -> list[dict]:
        """Forfeit every wager whose deadline passed with no submission."""
        out = []
        for task in state.tasks.expire_overdue():
            error = _settle_hold(task, FORFEIT)
            out.append({"task_id": task.id, "decision": FORFEIT, "error": error})
        return out

    @app.post("/api/tasks")
    def create_task(req: TaskRequest) -> dict:
        try:
            task = state.tasks.create(req.user_id, req.description, req.task_type, req.wager_cents,
                                      req.deadline_s, req.n_views, req.session_s, req.n_checkins)
        except TaskError as e:
            raise HTTPException(e.status, str(e)) from e
        try:
            hold = state.holds.authorize(task.id, task.wager_cents)
        except HoldError as e:
            state.tasks.payment_failed(task.id, str(e))
            raise HTTPException(402, f"the wager could not be held: {e}") from e
        task = state.tasks.activate(task.id, hold.id)
        return {"task": task.public(state.tasks.clock()), "hold": hold.to_dict(), "payments": state.holds.mode}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str, user_id: str) -> dict:
        _sweep()
        return _task(task_id, user_id).public(state.tasks.clock())

    @app.post("/api/tasks/sweep")
    def sweep_tasks() -> dict:
        return {"expired": _sweep()}

    @app.post("/api/tasks/{task_id}/session")
    def start_session(task_id: str, req: UserRequest) -> dict:
        try:
            task = state.tasks.start_session(task_id, req.user_id)
        except TaskError as e:
            raise HTTPException(e.status, str(e)) from e
        return task.public(state.tasks.clock())

    @app.post("/api/tasks/{task_id}/checkins")
    def add_checkin(task_id: str, req: CheckinRequest) -> dict:
        _task(task_id, req.user_id)
        with state.lock:
            cap = state.captures.get(req.capture_id)
            if cap is None:
                raise HTTPException(404, "unknown or already used capture")
            if cap.user_id != req.user_id:
                raise HTTPException(403, "capture belongs to another user")
            if cap.role != "checkin":
                raise HTTPException(422, "need a checkin capture")
            del state.captures[cap.id]
        if not cap.redemption.ok or not cap.stamp.ok:
            # Not counted: the check-in stays missing and the final check holds the task.
            return {"ok": False, "index": None,
                    "reason": f"challenge {cap.redemption.reason}, stamp {'ok' if cap.stamp.ok else 'not read'}"}
        try:
            return state.tasks.record_checkin(task_id, req.user_id, cap.id, cap.redemption.received_at, str(cap.path))
        except TaskError as e:
            raise HTTPException(e.status, str(e)) from e

    @app.post("/api/tasks/{task_id}/submit")
    def submit_task(task_id: str, req: TaskSubmitRequest) -> dict:
        task = _task(task_id, req.user_id)
        n = task.n_views if task.task_type == "multi_view" else 1
        if len(req.before_capture_ids) != n or len(req.after_capture_ids) != n:
            raise HTTPException(422, f"this task needs {n} before and {n} after capture(s)")
        try:
            task = state.tasks.begin_submit(task_id, req.user_id)
        except TaskError as e:
            if e.code == "deadline_passed":
                _sweep()  # forfeits this wager now rather than at the next sweep
            raise HTTPException(e.status, str(e)) from e

        try:
            with state.lock:
                befores = [state.captures.get(i) for i in req.before_capture_ids]
                afters = [state.captures.get(i) for i in req.after_capture_ids]
                if any(c is None for c in befores + afters):
                    raise HTTPException(404, "unknown or already used capture")
                if any(c.user_id != req.user_id for c in befores + afters):
                    raise HTTPException(403, "capture belongs to another user")
                if any(c.role != "before" for c in befores) or any(c.role != "after" for c in afters):
                    raise HTTPException(422, "need before captures and after captures")
                if task.task_type == "multi_view":
                    for group in (befores, afters):
                        if any(c.task_id != task.id for c in group) or {c.view_index for c in group} != set(range(n)):
                            raise HTTPException(422, "each angle needs exactly one capture taken for this task")
                        group.sort(key=lambda c: c.view_index)
                for c in befores + afters:
                    del state.captures[c.id]
                history = list(state.history.get(req.user_id, []))
        except HTTPException:
            state.tasks.abort_submit(task_id)
            raise

        def iso(ts: float) -> str:
            return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")

        def run(i: int):
            b, a = befores[i], afters[i]
            sub = Submission(
                pair_id=f"{task.id}-v{i}", set="honest", task_type=task.task_type,
                task_description=task.description if n == 1 else f"{task.description} (view {i + 1}: {task.view_prompts[i]})",
                before_path=b.path, after_path=a.path, label="completed", adversarial_category=None,
                base_pair_id=None, capture_source="browser",
                before_captured_at=iso(b.redemption.received_at), after_captured_at=iso(a.redemption.received_at),
            )
            evidence = ChallengeEvidence(b.redemption, a.redemption, b.stamp, a.stamp)
            return state.run_pipeline(sub, history, evidence)

        try:
            # The pairs run side by side, so n views cost one LLM call of wall-clock time,
            # not n, and each pair keeps its own 8 s server deadline.
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(run, range(n)))
        except Exception:
            state.tasks.abort_submit(task_id)
            raise

        decision = combine_decisions([r.decision for r in results])
        reasons = [r.reason for r in results if r.reason]
        extra = None
        if task.task_type == "multi_view":
            extra = check_multi_view([a.path for a in afters])
        elif task.task_type == "time_spread":
            extra = check_time_spread(task, befores[0].path, state.anticheat.config)
        if extra is not None:
            reasons.append(extra.detail)
            if not extra.ok and decision == RELEASE:
                decision = HOLD  # like every signal, a failure here can stop a release but not forfeit one

        t_money = state.tasks.clock()
        money_error = _settle_hold(task, decision)
        if money_error:
            decision = HOLD
            reasons.append(f"payment error: {money_error}")
        reason = "; ".join(reasons)
        with state.lock:
            state.history.setdefault(req.user_id, []).extend(a.path for a in afters)
        task = state.tasks.settle(task_id, decision, reason)
        return {
            "task": task.public(state.tasks.clock()),
            "decision": decision,
            "reason": reason,
            "pairs": [{"view": i, "decision": r.decision, "verdict": r.verdict, "confidence": r.confidence,
                       "flagged_by": r.anticheat.get("flagged_by", []), "deadline_missed": r.deadline_missed,
                       "server_s": r.latency_s} for i, r in enumerate(results)],
            "task_check": None if extra is None else {"ok": extra.ok, "detail": extra.detail},
            "money": {"event": {RELEASE: "hold_cancelled", FORFEIT: "captured_to_charity", HOLD: "hold_kept"}[decision],
                      "error": money_error},
            "server_s": max(r.latency_s for r in results) + (state.tasks.clock() - t_money),
        }

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "model": state.pipeline.model}

    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app


_APP: FastAPI | None = None


# `uvicorn bord.api:app` builds the default app on first access, so importing this module
# (tests, tooling) neither creates an uploads folder nor builds a model client.
def __getattr__(name: str):
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)
