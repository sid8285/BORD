"""Evaluation harness (Checkpoint 2).

Runs a verifier over every submission in a manifest and writes:

- ``predictions.jsonl``: one line per submission (verdict, confidence,
  latency, tokens, cost, error) joined with its ground truth.
- ``summary.json``: honest-set accuracy, false-accept / false-reject rates,
  fooled rate overall and per adversarial category, latency p50/p95, cost,
  and a pass/fail check for Assumptions 1 and 2.

Usage (from ``backend/``)::

    python -m bord.evaluate --manifest ../data/manifest.csv --out ../results

Commit the run folder it prints so the Checkpoint 2 evidence table can point
at it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from .dataset import ADVERSARIAL_CATEGORIES, Submission, load_manifest
from .verifier import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_S,
    LLMOnlyVerifier,
    VerifierResult,
)

# Thresholds stated in the Checkpoint 1 Technology Assumptions register.
A1_MIN_HONEST_ACCURACY = 0.90
A1_MIN_FOOLED_RATE = 0.30
A2_MAX_COST_USD = 0.02
A2_MAX_P95_LATENCY_S = 8.0


class Verifier(Protocol):
    def verify(self, sub: Submission) -> VerifierResult: ...


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in 0..100). None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_record(sub: Submission, result: VerifierResult) -> dict:
    record = result.to_dict()
    record.update(
        set=sub.set,
        task_type=sub.task_type,
        label=sub.label,
        adversarial_category=sub.adversarial_category,
        base_pair_id=sub.base_pair_id,
        expected_verdict=sub.expected_verdict,
        correct=None if result.verdict is None else result.verdict == sub.expected_verdict,
    )
    return record


def summarize(records: list[dict]) -> dict:
    """Aggregate metrics from build_record() outputs.

    Failed calls (verdict None) are excluded from accuracy and fooled-rate
    denominators and counted separately; a failed call never releases a wager,
    so counting it as a correct rejection would flatter the baseline.
    """
    answered = [r for r in records if r["verdict"] is not None]
    honest = [r for r in answered if r["set"] == "honest"]
    adversarial = [r for r in answered if r["set"] == "adversarial"]

    honest_completed = [r for r in honest if r["label"] == "completed"]
    honest_not_completed = [r for r in honest if r["label"] == "not_completed"]

    fooled = [r for r in adversarial if r["verdict"] == "completed"]
    per_category = {}
    for cat in ADVERSARIAL_CATEGORIES:
        rows = [r for r in adversarial if r["adversarial_category"] == cat]
        hits = sum(1 for r in rows if r["verdict"] == "completed")
        per_category[cat] = {"n": len(rows), "fooled": hits, "fooled_rate": _rate(hits, len(rows))}

    errors: dict[str, int] = defaultdict(int)
    for r in records:
        if r["error"]:
            errors[r["error"].split(":")[0]] += 1

    latencies = [r["latency_s"] for r in records]
    # Billed calls only: timeouts and connection errors cost nothing.
    costs = [r["cost_usd"] for r in records if r["input_tokens"] or r["output_tokens"]]

    honest_accuracy = _rate(sum(1 for r in honest if r["correct"]), len(honest))
    fooled_rate = _rate(len(fooled), len(adversarial))
    mean_cost = sum(costs) / len(costs) if costs else None
    p95 = percentile(latencies, 95)

    def check(ok: bool | None) -> str:
        return "untested" if ok is None else ("holding" if ok else "broken")

    a1 = None
    if honest_accuracy is not None and fooled_rate is not None:
        a1 = honest_accuracy >= A1_MIN_HONEST_ACCURACY and fooled_rate >= A1_MIN_FOOLED_RATE
    a2 = None
    if mean_cost is not None and p95 is not None:
        a2 = mean_cost < A2_MAX_COST_USD and p95 < A2_MAX_P95_LATENCY_S

    return {
        "n_submissions": len(records),
        "n_answered": len(answered),
        "n_honest": len(honest),
        "n_adversarial": len(adversarial),
        "errors": dict(errors),
        "honest_accuracy": honest_accuracy,
        "false_reject_rate": _rate(
            sum(1 for r in honest_completed if r["verdict"] == "not_completed"),
            len(honest_completed),
        ),
        "false_accept_rate_honest": _rate(
            sum(1 for r in honest_not_completed if r["verdict"] == "completed"),
            len(honest_not_completed),
        ),
        "fooled_rate": fooled_rate,
        "fooled_by_category": per_category,
        "latency_s": {
            "p50": percentile(latencies, 50),
            "p95": p95,
            "max": max(latencies) if latencies else None,
        },
        "cost_usd": {"mean_per_call": mean_cost, "total": sum(costs) if costs else 0.0},
        "assumptions": {
            "1": {
                "status": check(a1),
                "honest_accuracy": honest_accuracy,
                "fooled_rate": fooled_rate,
                "thresholds": {"min_honest_accuracy": A1_MIN_HONEST_ACCURACY,
                               "min_fooled_rate": A1_MIN_FOOLED_RATE},
            },
            "2": {
                "status": check(a2),
                "mean_cost_usd": mean_cost,
                "p95_latency_s": p95,
                "thresholds": {"max_cost_usd": A2_MAX_COST_USD,
                               "max_p95_latency_s": A2_MAX_P95_LATENCY_S},
            },
        },
    }


def run(
    submissions: Iterable[Submission],
    verifier: Verifier,
    out_dir: Path,
    run_meta: dict,
) -> dict:
    """Verify every submission, stream predictions to disk, write the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for sub in submissions:
            record = build_record(sub, verifier.verify(sub))
            records.append(record)
            fh.write(json.dumps(record) + "\n")
            fh.flush()  # keep partial results if the run is interrupted
            status = record["verdict"] or record["error"]
            print(f"{sub.pair_id}: {status} ({record['latency_s']:.2f}s)", file=sys.stderr)

    summary = {"run": run_meta, **summarize(records)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the BORD LLM-only baseline verifier.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT_S, type=float)
    parser.add_argument("--limit", type=int, help="only the first N rows (smoke test)")
    args = parser.parse_args(argv)

    submissions = load_manifest(args.manifest)
    if args.limit:
        submissions = submissions[: args.limit]

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + f"_{args.model}"
    out_dir = args.out / run_id
    meta = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "manifest": str(args.manifest),
        "verifier": "llm_only_baseline",
        "model": args.model,
        "effort": args.effort,
        "timeout_s": args.timeout,
    }
    verifier = LLMOnlyVerifier(model=args.model, effort=args.effort, timeout_s=args.timeout)
    summary = run(submissions, verifier, out_dir, meta)

    print(json.dumps({k: summary[k] for k in ("honest_accuracy", "fooled_rate",
                                               "latency_s", "cost_usd", "assumptions")}, indent=2))
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
