"""Wager holds: the simulation, and the Stripe wrapper against a fake stripe module."""

import json

import pytest

from bord.payments import HOLD_WINDOW_S, HoldError, SimulatedHolds, StripeHolds, default_holds


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def test_simulated_release_cancels_and_forfeit_pays_the_charity():
    h = SimulatedHolds(clock=Clock())
    a = h.authorize("t1", 500)
    assert a.status == "requires_capture" and a.expires_at - a.authorized_at == HOLD_WINDOW_S
    assert h.release(a.id, "t1").status == "canceled"
    b = h.authorize("t2", 700)
    assert h.forfeit(b.id, "t2").status == "succeeded"
    events = [(r["event"], r["task_id"]) for r in h.ledger.rows]
    assert events == [("authorized", "t1"), ("released", "t1"), ("authorized", "t2"),
                      ("captured", "t2"), ("charity_payout", "t2")]
    assert h.ledger.rows[-1]["simulated"] is True


def test_a_hold_settles_only_once():
    h = SimulatedHolds(clock=Clock())
    a = h.authorize("t1", 500)
    h.release(a.id)
    with pytest.raises(HoldError):
        h.forfeit(a.id)
    with pytest.raises(HoldError):
        h.release("sim_pi_nope")


def test_an_expired_hold_cannot_be_captured():
    clock = Clock()
    h = SimulatedHolds(clock=clock)
    a = h.authorize("t1", 500)
    clock.t += HOLD_WINDOW_S + 1
    with pytest.raises(HoldError):
        h.forfeit(a.id)


def test_ledger_is_appended_to_disk(tmp_path):
    path = tmp_path / "ledger.jsonl"
    h = SimulatedHolds(clock=Clock(), ledger_path=path)
    h.authorize("t1", 500)
    assert json.loads(path.read_text().splitlines()[0])["event"] == "authorized"


class FakeStripeError(Exception):
    pass


class FakeStripe:
    """The three PaymentIntent calls BORD makes, with Stripe's statuses."""

    StripeError = FakeStripeError

    def __init__(self, status_after_create="requires_capture"):
        self.calls = []
        self.status_after_create = status_after_create
        self.intents = {}
        outer = self

        class PaymentIntent:
            @staticmethod
            def create(**kw):
                outer.calls.append(("create", kw))
                intent = {"id": f"pi_{len(outer.intents) + 1}", "status": outer.status_after_create,
                          "amount": kw["amount"], "amount_received": 0}
                outer.intents[intent["id"]] = intent
                return intent

            @staticmethod
            def capture(pid, **kw):
                outer.calls.append(("capture", pid))
                intent = outer.intents[pid]
                if intent["status"] != "requires_capture":
                    raise FakeStripeError("not capturable")
                intent.update(status="succeeded", amount_received=intent["amount"])
                return intent

            @staticmethod
            def cancel(pid, **kw):
                outer.calls.append(("cancel", pid))
                intent = outer.intents[pid]
                if intent["status"] not in ("requires_capture", "requires_action"):
                    raise FakeStripeError("not cancelable")
                intent["status"] = "canceled"
                return intent

        self.PaymentIntent = PaymentIntent


def test_stripe_holds_use_manual_capture():
    fake = FakeStripe()
    h = StripeHolds("sk_test_abc", stripe_module=fake, clock=Clock())
    hold = h.authorize("task9", 1250)
    _, kw = fake.calls[0]
    assert kw["capture_method"] == "manual" and kw["confirm"] is True and kw["amount"] == 1250
    assert kw["currency"] == "usd" and kw["idempotency_key"] == "bord-hold-task9"
    assert kw["metadata"] == {"bord_task_id": "task9"}
    assert hold.status == "requires_capture"


def test_stripe_release_cancels_and_forfeit_captures():
    fake = FakeStripe()
    h = StripeHolds("sk_test_abc", stripe_module=fake, clock=Clock())
    a, b = h.authorize("t1", 500), h.authorize("t2", 800)
    assert h.release(a.id, "t1").status == "canceled"
    assert h.forfeit(b.id, "t2").amount_cents == 800
    assert [c[0] for c in fake.calls] == ["create", "create", "cancel", "capture"]
    assert [r["event"] for r in h.ledger.rows][-2:] == ["captured", "charity_payout"]
    with pytest.raises(HoldError):
        h.forfeit(a.id)  # already cancelled


def test_stripe_hold_that_needs_action_is_cancelled_and_refused():
    fake = FakeStripe(status_after_create="requires_action")
    h = StripeHolds("sk_test_abc", stripe_module=fake, clock=Clock())
    with pytest.raises(HoldError):
        h.authorize("t1", 500)
    assert fake.calls[-1][0] == "cancel"


@pytest.mark.parametrize("key", ["sk_live_abc", "", "pk_test_abc"])
def test_live_or_missing_keys_are_refused(key, monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    with pytest.raises(HoldError):
        StripeHolds(key, stripe_module=FakeStripe())


def test_default_holds_falls_back_to_the_simulation(monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    assert default_holds().mode == "simulated"
