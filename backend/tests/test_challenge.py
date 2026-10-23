"""Offline tests for server-issued challenges and server-clock ordering."""

import pytest

from bord.challenge import CODE_ALPHABET, ChallengeStore, check_order


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(ttl=60.0):
    clock = FakeClock()
    return ChallengeStore(ttl_s=ttl, clock=clock), clock


def test_issue_and_redeem():
    store, clock = make()
    c = store.issue("u1", "before")
    assert len(c.code) == 6 and set(c.code) <= set(CODE_ALPHABET)
    assert c.expires_at == pytest.approx(c.issued_at + 60.0)
    assert c.public()["code"] == c.code

    clock.t += 12.5
    r = store.redeem(c.id, c.code.lower(), "u1", "before")  # the code is case-insensitive
    assert r.ok and r.reason == "ok"
    assert r.age_s == pytest.approx(12.5) and r.received_at == pytest.approx(1012.5)


def test_challenge_is_single_use():
    store, _ = make()
    c = store.issue("u1", "before")
    assert store.redeem(c.id, c.code, "u1", "before").ok
    assert store.redeem(c.id, c.code, "u1", "before").reason == "already_used"


def test_wrong_code_consumes_the_challenge():
    store, _ = make()
    c = store.issue("u1", "after")
    assert store.redeem(c.id, "ZZZZZZ", "u1", "after").reason == "wrong_code"
    # No second guess at the same challenge.
    assert store.redeem(c.id, c.code, "u1", "after").reason == "already_used"


def test_expiry_user_role_and_unknown_id():
    store, clock = make(ttl=30.0)
    c = store.issue("u1", "before")
    clock.t += 31
    r = store.redeem(c.id, c.code, "u1", "before")
    assert not r.ok and r.reason == "expired" and r.age_s == pytest.approx(31)

    c2 = store.issue("u1", "before")
    assert store.redeem(c2.id, c2.code, "u2", "before").reason == "wrong_user"
    assert store.redeem(c2.id, c2.code, "u1", "after").reason == "wrong_role"
    assert store.redeem("nope", "ABC234", "u1", "before").reason == "unknown_challenge"
    # wrong user / role do not consume it: the right user can still redeem.
    assert store.redeem(c2.id, c2.code, "u1", "before").ok


def test_codes_are_not_repeated():
    store, _ = make()
    codes = {store.issue("u1", "before").code for _ in range(200)}
    assert len(codes) > 190  # 31**6 codes; collisions in 200 draws are vanishingly rare


def test_bad_role():
    store, _ = make()
    with pytest.raises(ValueError):
        store.issue("u1", "middle")


def _pair(gap_s, after_issued_delay=1.0):
    store, clock = make()
    cb = store.issue("u1", "before")
    clock.t += 3
    before = store.redeem(cb.id, cb.code, "u1", "before")
    clock.t += after_issued_delay
    ca = store.issue("u1", "after")
    clock.t += gap_s
    after = store.redeem(ca.id, ca.code, "u1", "after")
    return before, after


def test_check_order():
    before, after = _pair(gap_s=20)
    assert check_order(before, after) == (True, "ok")
    ok, why = check_order(*_pair(gap_s=0.5))
    assert not ok and why.startswith("gap_too_short")
    ok, why = check_order(*_pair(gap_s=7200))
    assert not ok and why.startswith("gap_too_long")


def test_check_order_needs_real_challenges():
    store, _ = make()
    bad = store.redeem("nope", "X", "u1", "before")
    good_before, good_after = _pair(gap_s=20)
    assert check_order(bad, good_after) == (False, "missing_challenge")
    # An AFTER challenge issued before the BEFORE photo was received was prepared in advance.
    ok, why = check_order(good_after, good_before)
    assert not ok
