"""The anti-cheating stage inside the real-time pipeline."""

import pytest

from bord.anticheat import AntiCheatReport, Signal
from bord.dataset import Submission
from bord.realtime import FORFEIT, HOLD, RELEASE, Budget, RealtimeVerifier
from bord.verifier import VerifierResult

SUB = Submission("p01", "honest", "single_scene", "Clean my desk", None, None,
                 "completed", None, None, "phone", "", "")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeLLM:
    model = "fake"

    def __init__(self, clock, verdict, took=3.0):
        self.clock, self.verdict, self.took, self.calls = clock, verdict, took, 0

    def prepare(self, sub):
        self.clock.t += 0.1
        return []

    def call(self, sub, content, timeout_s):
        self.calls += 1
        self.clock.t += self.took
        return VerifierResult(sub.pair_id, self.verdict, 0.9, "", self.took, 2000, 100, 0.01, "fake")


class FakeAntiCheat:
    """Advances the clock like a real stage would and returns a scripted report."""

    def __init__(self, clock, flagged, took=0.2):
        self.clock, self.flagged, self.took, self.seen = clock, flagged, took, []

    def check(self, sub, history, evidence):
        self.seen.append((sub.pair_id, list(history), evidence))
        self.clock.t += self.took
        return AntiCheatReport([Signal("reuse", self.flagged, 0.0, "scripted")], self.took)


def run(verdict, flagged, history=None, evidence=None, **kw):
    clock = FakeClock()
    llm, ac = FakeLLM(clock, verdict, **kw), FakeAntiCheat(clock, flagged)
    pipeline = RealtimeVerifier(llm, Budget(), clock=clock, anticheat=ac,
                                history_for=(lambda sub: history) if history is not None else None)
    return pipeline.verify(SUB, evidence), llm, ac


def test_clean_submission_is_released_and_the_stage_is_timed():
    r, llm, ac = run("completed", flagged=False)
    assert r.decision == RELEASE and llm.calls == 1
    assert r.stages["anticheat_s"] == pytest.approx(0.2)
    assert r.latency_s == pytest.approx(0.1 + 0.2 + 3.0)  # the stage counts against the deadline
    assert r.anticheat["flagged"] is False and r.anticheat["signals"][0]["name"] == "reuse"


def test_a_flag_turns_a_release_into_a_hold():
    r, _, _ = run("completed", flagged=True)
    assert r.verdict == "completed" and r.decision == HOLD
    assert r.anticheat["flagged_by"] == ["reuse"]


def test_a_flag_does_not_stop_a_forfeit():
    r, _, _ = run("not_completed", flagged=True)
    assert r.decision == FORFEIT


def test_history_and_evidence_reach_the_stage():
    r, _, ac = run("completed", flagged=False, history=["old.jpg"], evidence="evidence")
    assert ac.seen == [("p01", ["old.jpg"], "evidence")]


def test_without_the_stage_nothing_changes():
    clock = FakeClock()
    r = RealtimeVerifier(FakeLLM(clock, "completed"), Budget(), clock=clock).verify(SUB)
    assert r.decision == RELEASE and "anticheat_s" not in r.stages and r.anticheat == {}


def test_a_slow_stage_can_still_miss_the_deadline():
    clock = FakeClock()
    llm = FakeLLM(clock, "completed", took=6.0)
    ac = FakeAntiCheat(clock, flagged=False, took=2.0)  # 0.1 + 2.0 + 6.0 > 8.0
    r = RealtimeVerifier(llm, Budget(), clock=clock, anticheat=ac).verify(SUB)
    assert r.deadline_missed and r.decision == HOLD
