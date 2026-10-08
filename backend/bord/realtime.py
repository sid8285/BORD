"""Server-side real-time pipeline around a verifier (Checkpoint 2).

The verdict has a firm 10 s deadline at p95, measured from the moment the user
taps "submit" (docs/realtime_budget.md). The browser owns the first 2 s
(capture checks and upload); this module owns the remaining server budget and
splits it into stages:

    preprocess  ->  [anti-cheat checks, Checkpoint 3]  ->  LLM call  ->  decision

Each stage has a budget. The LLM call gets its budget as the request timeout.
If it fails fast (connection error, overload) and enough of the server budget
is left, it is retried once with whatever time remains. Anything that misses
the deadline, fails, or is not confident enough becomes a ``hold``: the Stripe
authorization stays in place, no money moves, and the submission goes to
manual review. A late verdict is discarded rather than acted on (firm deadline).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .dataset import Submission
from .verifier import VerifierResult

# Errors worth one retry. Refusals and unparseable output are deterministic
# enough that a retry would only burn the budget.
RETRYABLE_PREFIXES = ("timeout", "connection", "rate_limited", "api_status_5")

RELEASE = "release"  # task verified: return the wager
FORFEIT = "forfeit"  # task not done: donate the wager
HOLD = "hold"  # no trustworthy verdict in time: keep the hold, manual review


@dataclass(frozen=True)
class Budget:
    """Server-side stage budgets in seconds (see docs/realtime_budget.md)."""

    server_deadline_s: float = 8.0  # 10 s total minus 2 s for browser checks + upload
    preprocess_s: float = 0.5
    anticheat_s: float = 0.5  # reserved for Checkpoint 3; unused now
    llm_s: float = 6.5  # first-attempt request timeout
    decision_s: float = 0.5  # reserved for the decision and the Stripe call (Checkpoint 4)
    min_retry_s: float = 2.0  # do not start a retry with less time than this


@dataclass(frozen=True)
class Thresholds:
    """Confidence needed before money moves. Calibrated in Checkpoint 5.

    The Checkpoint 2 defaults of 0.0 act on the raw verdict, so the baseline is
    measured as is; holds then come only from failures and deadline misses.
    """

    release_min_conf: float = 0.0
    forfeit_min_conf: float = 0.0


def decide(result: VerifierResult, thresholds: Thresholds) -> str:
    if result.verdict == "completed" and (result.confidence or 0.0) >= thresholds.release_min_conf:
        return RELEASE
    if result.verdict == "not_completed" and (result.confidence or 0.0) >= thresholds.forfeit_min_conf:
        return FORFEIT
    return HOLD


def _retryable(error: str | None) -> bool:
    return bool(error) and error.startswith(RETRYABLE_PREFIXES)


class RealtimeVerifier:
    """Runs a verifier stage by stage against the server deadline."""

    def __init__(
        self,
        verifier,  # LLMOnlyVerifier or QwenVerifier: anything with prepare() and call()
        budget: Budget = Budget(),
        thresholds: Thresholds = Thresholds(),
        clock=time.perf_counter,
    ):
        self.verifier = verifier
        self.budget = budget
        self.thresholds = thresholds
        self.clock = clock

    @property
    def model(self) -> str:
        return self.verifier.model

    def verify(self, sub: Submission) -> VerifierResult:
        b = self.budget
        t0 = self.clock()
        stages: dict[str, float] = {}

        content = self.verifier.prepare(sub)
        stages["preprocess_s"] = self.clock() - t0

        attempts = 0
        llm_total = 0.0
        cost = 0.0
        in_tok = out_tok = 0
        timeout = b.llm_s
        while True:
            attempts += 1
            result = self.verifier.call(sub, content, timeout)
            llm_total += result.latency_s
            cost += result.cost_usd
            in_tok += result.input_tokens
            out_tok += result.output_tokens
            remaining = b.server_deadline_s - (self.clock() - t0) - b.decision_s
            if attempts == 1 and _retryable(result.error) and remaining >= b.min_retry_s:
                timeout = remaining
                continue
            break
        stages["llm_s"] = llm_total

        t_decide = self.clock()
        decision = decide(result, self.thresholds)
        elapsed = self.clock() - t0
        missed = elapsed > b.server_deadline_s
        if missed:
            decision = HOLD  # firm deadline: a late verdict is not acted on
        stages["decision_s"] = self.clock() - t_decide

        result.latency_s = elapsed
        result.attempts = attempts
        result.cost_usd = cost
        result.input_tokens = in_tok
        result.output_tokens = out_tok
        result.stages = stages
        result.deadline_missed = missed
        result.decision = decision
        return result
