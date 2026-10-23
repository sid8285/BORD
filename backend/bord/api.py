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
real-time pipeline plus the anti-cheating stage (bord.realtime, bord.anticheat);
Stripe holds arrive in Checkpoint 4, so ``release`` / ``forfeit`` are not yet
connected to money.

Run from ``backend/``::

    uvicorn bord.api:app --port 8000
"""

from __future__ import annotations

import json
import os
import secrets
import threading
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
from .qwen_verifier import QwenVerifier
from .realtime import Budget, RealtimeVerifier

MAX_PHOTO_BYTES = 15 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class ChallengeRequest(BaseModel):
    user_id: str
    role: str


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


class AppState:
    """Everything the endpoints share. Replaced by a database in Checkpoint 4."""

    def __init__(self, verifier, upload_dir: Path, store: ChallengeStore, anticheat: AntiCheat, budget: Budget,
                 device_log: Path):
        self.upload_dir = upload_dir
        self.device_log = device_log
        self.store = store
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
               anticheat: AntiCheat | None = None, device_log: Path | None = None) -> FastAPI:
    """Build the app. Tests pass a fake verifier and a store with a fake clock."""
    state = AppState(
        verifier or _default_verifier(),
        upload_dir or Path(os.environ.get("BORD_UPLOADS", "uploads")),
        store or ChallengeStore(),
        anticheat or AntiCheat(),
        Budget(),
        device_log or Path(os.environ.get("BORD_DEVICE_LOG", "../data/device_tests.jsonl")),
    )
    state.upload_dir.mkdir(parents=True, exist_ok=True)
    state.device_log.parent.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="BORD")
    app.state.bord = state

    @app.post("/api/challenges")
    def issue_challenge(req: ChallengeRequest) -> dict:
        if req.role not in ROLES:
            raise HTTPException(422, f"role must be one of {ROLES}")
        return state.store.issue(req.user_id, req.role).public()

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
        with state.lock:
            state.captures[capture_id] = Capture(capture_id, user_id, role, final, redemption)
        return {"capture_id": capture_id, "challenge_ok": redemption.ok, "reason": redemption.reason,
                "received_at": redemption.received_at}

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
        result = state.run_pipeline(sub, history, ChallengeEvidence(before.redemption, after.redemption))
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
        }
        with state.lock, state.device_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

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
