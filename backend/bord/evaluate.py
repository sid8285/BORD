"""Evaluation harness (Checkpoint 2).

Runs a verifier over every submission in a manifest and writes:

- ``predictions.jsonl``: one line per submission (verdict, confidence,
  latency, tokens, cost, error) joined with its ground truth.
- ``summary.json``: honest-set accuracy, false-accept / false-reject rates,
  fooled rate overall and per adversarial category (each with a 95% Wilson
  confidence interval), end-to-end and per-stage latency, deadline misses,
  decisions (release / forfeit / hold), cost, and a check for Assumptions 1
  and 2 plus the real-time budget.

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
from .realtime import HOLD, RELEASE, Budget, RealtimeVerifier, Thresholds
from .verifier import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    LLMOnlyVerifier,
    VerifierResult,
)

# Thresholds stated in the Checkpoint 1 Technology Assumptions register.
A1_MIN_HONEST_ACCURACY = 0.90
A1_MIN_FOOLED_RATE = 0.30
A2_MAX_COST_USD = 0.02
A2_MAX_P95_LATENCY_S = 8.0
# Real-time budget (docs/realtime_budget.md): server share of the 10 s verdict deadline.
RT_SERVER_DEADLINE_S = Budget().server_deadline_s
WILSON_Z = 1.96  # 95% interval


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


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> list[float] | None:
    """Wilson score interval for k successes out of n. None when n is 0.

    Preferred over the normal approximation for small n and rates near 0 or 1,
    which is exactly the regime of a 15-30 pair dataset.
    """
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _proportion(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": _rate(k, n), "ci95": wilson_interval(k, n)}


def _check(point_ok: bool | None, ci_ok: bool | None = None) -> str:
    """holding / broken / untested on the point estimate; 'holding (not significant)'
    when the point estimate passes but the 95% interval still crosses the threshold."""
    if point_ok is None:
        return "untested"
    if not point_ok:
        return "broken"
    return "holding" if ci_ok in (None, True) else "holding (not significant)"


def _latency_stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


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


def summarize(records: list[dict], server_deadline_s: float = RT_SERVER_DEADLINE_S) -> dict:
    """Aggregate metrics from build_record() outputs.

    Failed calls (verdict None) are excluded from accuracy and fooled-rate
    denominators and counted separately; a failed call never releases a wager,
    so counting it as a correct rejection would flatter the baseline. The
    decision metrics at the end include every row, because there a failure is
    a real outcome (a hold).
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
        per_category[cat] = {"n": len(rows), "fooled": hits, "fooled_rate": _rate(hits, len(rows)),
                             "ci95": wilson_interval(hits, len(rows))}

    errors: dict[str, int] = defaultdict(int)
    for r in records:
        if r["error"]:
            errors[r["error"].split(":")[0]] += 1

    latencies = [r["latency_s"] for r in records]
    llm_latencies = [r.get("stages", {}).get("llm_s", r["latency_s"]) for r in records]
    stage_names = sorted({k for r in records for k in r.get("stages", {})})
    # Billed calls only: timeouts and connection errors cost nothing.
    costs = [r["cost_usd"] for r in records if r["input_tokens"] or r["output_tokens"]]

    n_honest_correct = sum(1 for r in honest if r["correct"])
    honest_acc = _proportion(n_honest_correct, len(honest))
    fooled_p = _proportion(len(fooled), len(adversarial))
    mean_cost = sum(costs) / len(costs) if costs else None
    llm_p95 = percentile(llm_latencies, 95)
    e2e_p95 = percentile(latencies, 95)

    def ge(prop: dict, threshold: float) -> tuple[bool | None, bool | None]:
        if prop["rate"] is None:
            return None, None
        return prop["rate"] >= threshold, prop["ci95"][0] >= threshold

    acc_ok, acc_ci_ok = ge(honest_acc, A1_MIN_HONEST_ACCURACY)
    fool_ok, fool_ci_ok = ge(fooled_p, A1_MIN_FOOLED_RATE)
    a1 = None if acc_ok is None or fool_ok is None else (acc_ok and fool_ok)
    a1_ci = None if a1 is None else (acc_ci_ok and fool_ci_ok)

    cost_ok = None if mean_cost is None else mean_cost < A2_MAX_COST_USD
    lat_ok = None if llm_p95 is None else llm_p95 < A2_MAX_P95_LATENCY_S
    a2 = None if cost_ok is None or lat_ok is None else (cost_ok and lat_ok)

    # Decisions: what would have happened to the money, over every row.
    def decisions(rows: list[dict]) -> dict:
        out: dict[str, int] = defaultdict(int)
        for r in rows:
            out[r.get("decision") or "none"] += 1
        return dict(out)

    honest_all = [r for r in records if r["set"] == "honest"]
    adv_all = [r for r in records if r["set"] == "adversarial"]
    released_adv = sum(1 for r in adv_all if r.get("decision") == RELEASE)
    misses = sum(1 for r in records if r.get("deadline_missed"))
    holds = sum(1 for r in records if r.get("decision") == HOLD)
    retried = sum(1 for r in records if r.get("attempts", 1) > 1)

    return {
        "n_submissions": len(records),
        "n_answered": len(answered),
        "n_honest": len(honest),
        "n_adversarial": len(adversarial),
        "errors": dict(errors),
        "honest_accuracy": honest_acc["rate"],
        "honest_accuracy_ci95": honest_acc["ci95"],
        "false_reject_rate": _rate(
            sum(1 for r in honest_completed if r["verdict"] == "not_completed"),
            len(honest_completed),
        ),
        "false_accept_rate_honest": _rate(
            sum(1 for r in honest_not_completed if r["verdict"] == "completed"),
            len(honest_not_completed),
        ),
        "fooled_rate": fooled_p["rate"],
        "fooled_rate_ci95": fooled_p["ci95"],
        "fooled_by_category": per_category,
        "latency_s": _latency_stats(latencies),
        "stage_latency_s": {
            name: _latency_stats([r["stages"][name] for r in records if name in r.get("stages", {})])
            for name in stage_names
        },
        "realtime": {
            "server_deadline_s": server_deadline_s,
            "deadline_misses": misses,
            "deadline_miss_rate": _rate(misses, len(records)),
            "retried": retried,
            "holds": holds,
            "hold_rate": _rate(holds, len(records)),
            "e2e_p95_s": e2e_p95,
            "status": _check(None if e2e_p95 is None else e2e_p95 <= server_deadline_s),
        },
        "decisions": {"honest": decisions(honest_all), "adversarial": decisions(adv_all)},
        "adversarial_released": _proportion(released_adv, len(adv_all)),
        "cost_usd": {"mean_per_call": mean_cost, "total": sum(costs) if costs else 0.0},
        "assumptions": {
            "1": {
                "status": _check(a1, a1_ci),
                "honest_accuracy": {**honest_acc, "threshold": A1_MIN_HONEST_ACCURACY,
                                    "status": _check(acc_ok, acc_ci_ok)},
                "fooled_rate": {**fooled_p, "threshold": A1_MIN_FOOLED_RATE,
                                "status": _check(fool_ok, fool_ci_ok)},
            },
            "2": {
                "status": _check(a2),
                "mean_cost_usd": mean_cost,
                "cost_status": _check(cost_ok),
                "p95_llm_latency_s": llm_p95,
                "latency_status": _check(lat_ok),
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
            decision = record.get("decision") or "-"
            print(f"{sub.pair_id}: {status} -> {decision} ({record['latency_s']:.2f}s)", file=sys.stderr)

    deadline = run_meta.get("server_deadline_s", RT_SERVER_DEADLINE_S)
    summary = {"run": run_meta, **summarize(records, server_deadline_s=deadline)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the BORD LLM-only baseline verifier.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--llm-timeout", default=Budget().llm_s, type=float,
                        help="first-attempt LLM request timeout (stage budget), seconds")
    parser.add_argument("--deadline", default=Budget().server_deadline_s, type=float,
                        help="server-side firm deadline, seconds")
    parser.add_argument("--release-min-conf", default=0.0, type=float)
    parser.add_argument("--forfeit-min-conf", default=0.0, type=float)
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
        "llm_timeout_s": args.llm_timeout,
        "server_deadline_s": args.deadline,
        "release_min_conf": args.release_min_conf,
        "forfeit_min_conf": args.forfeit_min_conf,
    }
    budget = Budget(server_deadline_s=args.deadline, llm_s=args.llm_timeout)
    verifier = RealtimeVerifier(
        LLMOnlyVerifier(model=args.model, effort=args.effort, timeout_s=args.llm_timeout),
        budget=budget,
        thresholds=Thresholds(args.release_min_conf, args.forfeit_min_conf),
    )
    summary = run(submissions, verifier, out_dir, meta)

    print(json.dumps({k: summary[k] for k in ("honest_accuracy", "fooled_rate",
                                               "latency_s", "realtime", "cost_usd",
                                               "assumptions")}, indent=2))
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
