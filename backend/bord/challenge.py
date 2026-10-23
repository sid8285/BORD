"""Server-issued capture challenges and server-side timestamps (Checkpoint 3).

Provenance in BORD cannot come from EXIF: frames captured in the browser carry
none (Assumption 3). It comes from the server instead. Before each capture the
server issues a challenge, a short random code the browser draws onto the
frame it captures (frontend/src/CameraCapture.jsx) plus an angle prompt. When
the photo arrives the server checks that:

- the challenge exists, was issued to this user for this role (before / after),
  and has not been used (single use);
- the code echoed back matches (constant-time comparison);
- the photo arrived within ``ttl_s`` of the challenge being issued, by the
  server's own clock (so a photo prepared in advance cannot satisfy it);
- the AFTER challenge was issued, and the AFTER photo received, later than the
  BEFORE ones, with a plausible gap between the two.

Nothing here trusts a client clock. ``clock`` is injectable for tests.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import asdict, dataclass

# No 0/O, 1/I/L: the code is read off a screen and drawn on a photo.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
ANGLES = ("straight on", "from the left", "from the right", "from slightly above", "from a low angle")
ROLES = ("before", "after")


@dataclass(frozen=True)
class Challenge:
    id: str
    user_id: str
    role: str           # before | after
    code: str           # drawn onto the frame by the browser
    angle: str          # prompt shown to the user
    issued_at: float    # server clock, seconds since the epoch
    expires_at: float

    def public(self) -> dict:
        """What the browser receives."""
        return {"id": self.id, "role": self.role, "code": self.code, "angle": self.angle,
                "issued_at": self.issued_at, "expires_at": self.expires_at}


@dataclass(frozen=True)
class Redemption:
    ok: bool
    reason: str          # "ok" or why it failed
    challenge_id: str
    received_at: float   # server clock when the photo arrived
    issued_at: float | None
    age_s: float | None  # received_at - issued_at

    def to_dict(self) -> dict:
        return asdict(self)


class ChallengeStore:
    """In-memory challenge registry. One process; swap for Redis if the API scales out."""

    def __init__(self, ttl_s: float = 60.0, code_length: int = 6, clock=time.time, max_used: int = 10_000):
        self.ttl_s = ttl_s
        self.code_length = code_length
        self.clock = clock
        self._open: dict[str, Challenge] = {}
        self._used: dict[str, None] = {}  # insertion-ordered set, bounded by max_used
        self._max_used = max_used
        self._lock = threading.Lock()

    def issue(self, user_id: str, role: str) -> Challenge:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        now = self.clock()
        challenge = Challenge(
            id=secrets.token_urlsafe(12),
            user_id=user_id,
            role=role,
            code="".join(secrets.choice(CODE_ALPHABET) for _ in range(self.code_length)),
            angle=secrets.choice(ANGLES),
            issued_at=now,
            expires_at=now + self.ttl_s,
        )
        with self._lock:
            self._purge(now)
            self._open[challenge.id] = challenge
        return challenge

    def redeem(self, challenge_id: str, code: str, user_id: str, role: str) -> Redemption:
        """Check a returned challenge. A challenge is consumed by the first attempt that
        reaches the code comparison, right or wrong, so a code cannot be guessed repeatedly."""
        now = self.clock()

        def result(ok: bool, reason: str, challenge: Challenge | None = None) -> Redemption:
            issued = challenge.issued_at if challenge else None
            return Redemption(ok, reason, challenge_id, now, issued, None if issued is None else now - issued)

        with self._lock:
            if challenge_id in self._used:
                return result(False, "already_used")
            challenge = self._open.get(challenge_id)
            if challenge is None:
                return result(False, "unknown_challenge")
            if challenge.user_id != user_id:
                return result(False, "wrong_user", challenge)
            if challenge.role != role:
                return result(False, "wrong_role", challenge)
            self._open.pop(challenge_id)
            self._used[challenge_id] = None
            while len(self._used) > self._max_used:
                self._used.pop(next(iter(self._used)))

        if now > challenge.expires_at:
            return result(False, "expired", challenge)
        if not hmac.compare_digest(challenge.code.encode(), code.strip().upper().encode()):
            return result(False, "wrong_code", challenge)
        return result(True, "ok", challenge)

    def _purge(self, now: float) -> None:
        for cid in [c.id for c in self._open.values() if now > c.expires_at + self.ttl_s]:
            del self._open[cid]


def check_order(before: Redemption, after: Redemption, min_gap_s: float = 2.0, max_gap_s: float = 3600.0) -> tuple[bool, str]:
    """Server-clock sanity of a before/after pair.

    The AFTER photo must be received later than the BEFORE photo, and its challenge
    issued later than the BEFORE photo was received, by at least ``min_gap_s`` (a
    task takes time) and at most ``max_gap_s``.
    """
    if before.issued_at is None or after.issued_at is None:
        return False, "missing_challenge"
    if after.issued_at < before.received_at:
        return False, "after_challenge_issued_before_before_photo_received"
    gap = after.received_at - before.received_at
    if gap < min_gap_s:
        return False, f"gap_too_short ({gap:.1f}s)"
    if gap > max_gap_s:
        return False, f"gap_too_long ({gap:.0f}s)"
    return True, "ok"
