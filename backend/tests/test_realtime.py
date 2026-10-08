"""Offline tests for the real-time pipeline and the confidence intervals."""

import pytest

from bord.dataset import Submission
from bord.evaluate import summarize, wilson_interval
from bord.realtime import FORFEIT, HOLD, RELEASE, Budget, RealtimeVerifier, Thresholds, decide
from bord.verifier import VerifierResult

SUB = Submission("p01", "honest", "single_scene", "Clean my desk", None, None,
                 "completed", None, None, "phone", "", "")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeLLM:
    """Scripted LLM stage: each entry is (seconds taken, verdict or error, confidence)."""

    model = "fake"

    def __init__(self, clock, script):
        self.clock, self.script, self.timeouts = clock, list(script), []

    def prepare(self, sub):
        self.clock.t += 0.1
        return []

    def call(self, sub, content, timeout_s):
        self.timeouts.append(timeout_s)
        took, outcome, conf = self.script.pop(0)
        took = min(took, timeout_s)
        self.clock.t += took
        if outcome in ("completed", "not_completed"):
            return VerifierResult(sub.pair_id, outcome, conf, "", took, 2000, 100, 0.01, "fake")
        return VerifierResult(sub.pair_id, None, None, "", took, model="fake", error=outcome)


def run(script, thresholds=Thresholds(), budget=Budget()):
    clock = FakeClock()
    llm = FakeLLM(clock, script)
    return RealtimeVerifier(llm, budget, thresholds, clock=clock).verify(SUB), llm


def test_success_releases():
    r, llm = run([(3.0, "completed", 0.9)])
    assert r.decision == RELEASE and r.attempts == 1 and not r.deadline_missed
    assert llm.timeouts == [6.5]
    assert r.stages["preprocess_s"] == pytest.approx(0.1)
    assert r.latency_s == pytest.approx(3.1)


def test_fast_failure_is_retried_with_remaining_time():
    r, llm = run([(0.4, "connection: reset", None), (2.0, "not_completed", 0.8)])
    assert r.attempts == 2 and r.decision == FORFEIT
    # 8.0 deadline - 0.5 elapsed - 0.5 decision reserve = 7.0 s left for the retry
    assert llm.timeouts[1] == pytest.approx(7.0)
    assert r.cost_usd == pytest.approx(0.01)


def test_timeout_leaves_no_room_for_retry_and_holds():
    r, llm = run([(10.0, "timeout", None)])
    assert r.attempts == 1 and r.decision == HOLD
    assert r.stages["llm_s"] == pytest.approx(6.5)


def test_refusal_is_not_retried():
    r, _ = run([(1.0, "refusal", None)])
    assert r.attempts == 1 and r.decision == HOLD


def test_late_verdict_is_discarded():
    # A tight deadline: the verdict arrives after it, so it must not move money.
    r, _ = run([(3.0, "completed", 0.99)], budget=Budget(server_deadline_s=2.0, llm_s=6.5))
    assert r.deadline_missed and r.decision == HOLD and r.verdict == "completed"


def test_low_confidence_holds():
    t = Thresholds(release_min_conf=0.8, forfeit_min_conf=0.6)
    assert decide(VerifierResult("x", "completed", 0.7, "", 1.0), t) == HOLD
    assert decide(VerifierResult("x", "completed", 0.85, "", 1.0), t) == RELEASE
    assert decide(VerifierResult("x", "not_completed", 0.65, "", 1.0), t) == FORFEIT
    assert decide(VerifierResult("x", None, None, "", 1.0, error="timeout"), t) == HOLD


def test_wilson_interval():
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(14, 15)
    assert lo == pytest.approx(0.702, abs=1e-3) and hi == pytest.approx(0.988, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.278, abs=1e-3)


def _rec(set_, label, verdict, decision, latency=1.0, cat=None, missed=False):
    return {"set": set_, "label": label, "verdict": verdict, "decision": decision,
            "latency_s": latency, "stages": {"preprocess_s": 0.1, "llm_s": latency - 0.1},
            "adversarial_category": cat, "error": None if verdict else "timeout",
            "input_tokens": 2000 if verdict else 0, "output_tokens": 100 if verdict else 0,
            "cost_usd": 0.01 if verdict else 0.0, "deadline_missed": missed, "attempts": 1,
            "correct": None if verdict is None else verdict == ("not_completed" if set_ == "adversarial" else label)}


def test_summary_realtime_and_decisions():
    records = [_rec("honest", "completed", "completed", RELEASE, 2.0) for _ in range(30)]
    records += [_rec("adversarial", "not_completed", "completed", RELEASE, 3.0, "reused") for _ in range(12)]
    records += [_rec("adversarial", "not_completed", "not_completed", FORFEIT, 3.0, "reused") for _ in range(17)]
    records += [_rec("adversarial", "not_completed", None, HOLD, 8.5, "web_sourced", missed=True)]
    s = summarize(records)
    # Even 30/30 correct cannot show >= 90% at 95% confidence (Wilson low 0.886).
    assert s["assumptions"]["1"]["honest_accuracy"]["status"] == "holding (not significant)"
    assert s["fooled_rate"] == pytest.approx(12 / 29)
    assert s["realtime"]["deadline_misses"] == 1 and s["realtime"]["holds"] == 1
    assert s["adversarial_released"]["k"] == 12 and s["adversarial_released"]["n"] == 30
    assert s["decisions"]["adversarial"] == {RELEASE: 12, FORFEIT: 17, HOLD: 1}
    assert set(s["stage_latency_s"]) == {"preprocess_s", "llm_s"}
    assert s["assumptions"]["2"]["p95_llm_latency_s"] == pytest.approx(2.9)
