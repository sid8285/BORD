"""Wager holds through Stripe test mode (Checkpoint 4, Assumption 5).

Assumption 5: Stripe test mode supports manual-capture payment holds (authorize
now, capture or cancel later) for up to 7 days. BORD uses exactly that:

    task created   -> authorize   PaymentIntent(capture_method="manual", confirm=True)
    verdict release-> release     cancel the PaymentIntent: the user is never charged
    verdict forfeit-> forfeit     capture the PaymentIntent, then record a simulated
                                  payout to the charity (no real transfer exists)
    verdict hold   -> nothing     the authorization stays in place for manual review

``StripeHolds`` talks to Stripe and refuses any key that is not a test key.
``SimulatedHolds`` has the same interface and no network, for tests and for running
the app without a key. Both record a ledger line per money event.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HOLD_WINDOW_S = 7 * 24 * 3600  # card authorizations are guaranteed for 7 days
CHARITY = "Simulated charity (test mode, no transfer)"
TEST_KEY_PREFIXES = ("sk_test_", "rk_test_")
# Stripe's documented test payment method; it always authorizes.
TEST_PAYMENT_METHOD = "pm_card_visa"


class HoldError(Exception):
    """A hold could not be placed, released or captured."""


@dataclass(frozen=True)
class Hold:
    id: str
    amount_cents: int
    authorized_at: float
    expires_at: float
    status: str  # requires_capture | canceled | succeeded

    def to_dict(self) -> dict:
        return asdict(self)


class _Ledger:
    def __init__(self, path: Path | None, clock):
        self.path, self.clock = path, clock
        self.rows: list[dict] = []
        self._lock = threading.Lock()

    def add(self, event: str, hold_id: str, task_id: str | None, amount_cents: int, **extra) -> dict:
        row = {"at": self.clock(), "event": event, "hold_id": hold_id, "task_id": task_id,
               "amount_cents": amount_cents, **extra}
        with self._lock:
            self.rows.append(row)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
        return row


class SimulatedHolds:
    """In-memory stand-in with Stripe's hold semantics: a hold settles exactly once."""

    mode = "simulated"

    def __init__(self, clock=time.time, ledger_path: Path | None = None):
        self.clock = clock
        self.ledger = _Ledger(ledger_path, clock)
        self._holds: dict[str, Hold] = {}
        self._n = 0
        self._lock = threading.Lock()

    def authorize(self, task_id: str, amount_cents: int) -> Hold:
        if amount_cents <= 0:
            raise HoldError("amount must be positive")
        now = self.clock()
        with self._lock:
            self._n += 1
            hold = Hold(f"sim_pi_{self._n}", amount_cents, now, now + HOLD_WINDOW_S, "requires_capture")
            self._holds[hold.id] = hold
        self.ledger.add("authorized", hold.id, task_id, amount_cents)
        return hold

    def release(self, hold_id: str, task_id: str | None = None) -> Hold:
        hold = self._settle(hold_id, "canceled")
        self.ledger.add("released", hold.id, task_id, hold.amount_cents)
        return hold

    def forfeit(self, hold_id: str, task_id: str | None = None) -> Hold:
        hold = self._settle(hold_id, "succeeded")
        self.ledger.add("captured", hold.id, task_id, hold.amount_cents)
        self.ledger.add("charity_payout", hold.id, task_id, hold.amount_cents, charity=CHARITY, simulated=True)
        return hold

    def get(self, hold_id: str) -> Hold:
        try:
            return self._holds[hold_id]
        except KeyError:
            raise HoldError(f"unknown hold {hold_id}") from None

    def _settle(self, hold_id: str, status: str) -> Hold:
        with self._lock:
            hold = self.get(hold_id)
            if hold.status != "requires_capture":
                raise HoldError(f"hold {hold_id} is already {hold.status}")
            if self.clock() > hold.expires_at:
                raise HoldError(f"hold {hold_id} expired")
            settled = Hold(hold.id, hold.amount_cents, hold.authorized_at, hold.expires_at, status)
            self._holds[hold_id] = settled
            return settled


class StripeHolds:
    """Stripe test-mode manual-capture holds. ``stripe_module`` is injectable for tests."""

    mode = "stripe_test"

    def __init__(self, secret_key: str | None = None, payment_method: str = TEST_PAYMENT_METHOD,
                 stripe_module=None, clock=time.time, ledger_path: Path | None = None):
        key = secret_key or os.environ.get("STRIPE_SECRET_KEY", "")
        if not key.startswith(TEST_KEY_PREFIXES):
            raise HoldError("STRIPE_SECRET_KEY must be a Stripe test key (sk_test_...): BORD never uses live mode")
        if stripe_module is None:
            import stripe as stripe_module  # lazy: the offline tests and the benchmark do not need it
        self.stripe = stripe_module
        self.key = key
        self.payment_method = payment_method
        self.clock = clock
        self.ledger = _Ledger(ledger_path, clock)
        self._error = getattr(stripe_module, "StripeError", Exception)

    def authorize(self, task_id: str, amount_cents: int) -> Hold:
        if amount_cents <= 0:
            raise HoldError("amount must be positive")
        try:
            intent = self.stripe.PaymentIntent.create(
                api_key=self.key,
                amount=amount_cents,
                currency="usd",
                capture_method="manual",
                confirm=True,
                payment_method=self.payment_method,
                automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
                metadata={"bord_task_id": task_id},
                description="BORD wager hold (test mode)",
                idempotency_key=f"bord-hold-{task_id}",  # a retried request cannot place two holds
            )
        except self._error as e:
            raise HoldError(f"Stripe refused the hold: {e}") from e
        if intent["status"] != "requires_capture":
            # Not authorized (3-D Secure, declined, ...): do not leave a half-open intent behind.
            try:
                self.stripe.PaymentIntent.cancel(intent["id"], api_key=self.key)
            except self._error:
                pass
            raise HoldError(f"hold not authorized (status {intent['status']})")
        now = self.clock()
        hold = Hold(intent["id"], amount_cents, now, now + HOLD_WINDOW_S, "requires_capture")
        self.ledger.add("authorized", hold.id, task_id, amount_cents)
        return hold

    def release(self, hold_id: str, task_id: str | None = None) -> Hold:
        try:
            intent = self.stripe.PaymentIntent.cancel(hold_id, api_key=self.key)
        except self._error as e:
            raise HoldError(f"could not cancel {hold_id}: {e}") from e
        self.ledger.add("released", hold_id, task_id, intent["amount"])
        return Hold(hold_id, intent["amount"], 0.0, 0.0, "canceled")

    def forfeit(self, hold_id: str, task_id: str | None = None) -> Hold:
        try:
            intent = self.stripe.PaymentIntent.capture(hold_id, api_key=self.key)
        except self._error as e:
            raise HoldError(f"could not capture {hold_id}: {e}") from e
        amount = intent["amount_received"] or intent["amount"]
        self.ledger.add("captured", hold_id, task_id, amount)
        # Stripe has no charity transfer in test mode: record the payout instead of making one.
        self.ledger.add("charity_payout", hold_id, task_id, amount, charity=CHARITY, simulated=True)
        return Hold(hold_id, amount, 0.0, 0.0, "succeeded")


def default_holds(clock=time.time, ledger_path: Path | None = None):
    """Stripe test mode when STRIPE_SECRET_KEY is set, the simulation otherwise."""
    if os.environ.get("STRIPE_SECRET_KEY"):
        return StripeHolds(clock=clock, ledger_path=ledger_path)
    return SimulatedHolds(clock=clock, ledger_path=ledger_path)
