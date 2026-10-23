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
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from .anticheat import AntiCheat, AntiCheatConfig, ManifestHistory
from .dataset import ADVERSARIAL_CATEGORIES, Submission, load_manifest
from .detectors import HFImageDetector
from .qwen_verifier import PROVIDERS, QwenVerifier
from .realtime import HOLD, RELEASE, Budget, RealtimeVerifier, Thresholds
from .verifier import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    MAX_IMAGE_SIDE,
    LLMOnlyVerifier,
    VerifierResult,
)

# Thresholds stated in the Checkpoint 1 Technology Assumptions register.
A1_MIN_HONEST_ACCURACY = 0.90
A1_MIN_FOOLED_RATE = 0.30
A2_MAX_COST_USD = 0.02
A2_MAX_P95_LATENCY_S = 8.0
A4_MAX_DETECTION_RATE = 0.70  # Assumption 4: detectors catch fewer than 70% of edited images
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


def _signal_flag(record: dict, name: str | None) -> bool | None:
    """Did this record's anti-cheating report flag it (overall when name is None, else
    by one signal)? None when the signal did not apply to the row."""
    report = record["anticheat"]
    if name is None:
        return report["flagged"]
    signal = next((s for s in report["signals"] if s["name"] == name), None)
    return None if signal is None else signal["flagged"]


def summarize_anticheat(records: list[dict]) -> dict | None:
    """Flag rates of the Checkpoint 3 anti-cheating stage, over the rows it ran on.

    For the overall flag and for each signal: the share of honest submissions flagged
    (false alarms; each should be near 0), the share of adversarial submissions flagged
    (detections) and the same per attack category, all with 95% Wilson intervals.
    Rows a signal does not apply to (e.g. no detector installed) are left out of its n.
    """
    rows = [r for r in records if r.get("anticheat")]
    if not rows:
        return None
    honest = [r for r in rows if r["set"] == "honest"]
    adversarial = [r for r in rows if r["set"] == "adversarial"]

    def flagged_share(subset: list[dict], name: str | None) -> dict:
        applicable = [r for r in subset if _signal_flag(r, name) is not None]
        return _proportion(sum(1 for r in applicable if _signal_flag(r, name)), len(applicable))

    def block(name: str | None) -> dict:
        return {
            "honest": flagged_share(honest, name),
            "adversarial": flagged_share(adversarial, name),
            "by_category": {
                cat: flagged_share([r for r in adversarial if r["adversarial_category"] == cat], name)
                for cat in ADVERSARIAL_CATEGORIES
            },
        }

    names = sorted({s["name"] for r in rows for s in r["anticheat"]["signals"]})
    elapsed = [r["stages"]["anticheat_s"] for r in rows if "anticheat_s" in r.get("stages", {})]
    return {
        "overall": block(None),
        "signals": {name: block(name) for name in names},
        "stage_s": _latency_stats(elapsed),
        "over_budget": sum(1 for s in elapsed if s > Budget().anticheat_s),
        "n": len(rows),
    }


def _assumption_4(anticheat: dict) -> dict:
    """Assumption 4: open-source detectors catch fewer than 70% of edited images.

    Measured as the share of the AI-edited attacks the pixel detector flagged. The
    assumption holds when that share is below the threshold.
    """
    caught = anticheat["signals"].get("ai_image", {}).get("by_category", {}).get("ai_edited", {})
    k, n, rate, ci = caught.get("k", 0), caught.get("n", 0), caught.get("rate"), caught.get("ci95")
    point_ok = None if rate is None else rate < A4_MAX_DETECTION_RATE
    ci_ok = None if ci is None else ci[1] < A4_MAX_DETECTION_RATE
    return {"status": _check(point_ok, ci_ok), "detected": {"k": k, "n": n, "rate": rate, "ci95": ci},
            "threshold": A4_MAX_DETECTION_RATE}


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
    stage_names = sorted({k for r in records for k in r.get("stages", {})})
    # Billed calls only: timeouts and connection errors cost nothing.
    costs = [r["cost_usd"] for r in records if r["input_tokens"] or r["output_tokens"]]

    n_honest_correct = sum(1 for r in honest if r["correct"])
    honest_acc = _proportion(n_honest_correct, len(honest))
    fooled_p = _proportion(len(fooled), len(adversarial))
    mean_cost = sum(costs) / len(costs) if costs else None
    e2e_p95 = percentile(latencies, 95)
    # Assumption 2 latency: a timed-out call's real latency is unknown but at least its
    # timeout, so it counts as infinitely slow. Using the capped time instead would let a
    # run in which every call timed out pass the latency check.
    def llm_s(r: dict) -> float:
        return r.get("stages", {}).get("llm_s", r["latency_s"])

    timed_out = [llm_s(r) for r in records if (r["error"] or "").startswith("timeout")]
    a2_latencies = [math.inf if (r["error"] or "").startswith("timeout") else llm_s(r) for r in records]
    llm_p95 = percentile(a2_latencies, 95)
    timeout_floor = min(timed_out, default=None)

    def ge(prop: dict, threshold: float) -> tuple[bool | None, bool | None]:
        if prop["rate"] is None:
            return None, None
        return prop["rate"] >= threshold, prop["ci95"][0] >= threshold

    acc_ok, acc_ci_ok = ge(honest_acc, A1_MIN_HONEST_ACCURACY)
    fool_ok, fool_ci_ok = ge(fooled_p, A1_MIN_FOOLED_RATE)
    a1 = None if acc_ok is None or fool_ok is None else (acc_ok and fool_ok)
    a1_ci = None if a1 is None else (acc_ci_ok and fool_ci_ok)

    cost_ok = None if mean_cost is None else mean_cost < A2_MAX_COST_USD
    if llm_p95 is None:
        lat_ok = None
    elif math.isinf(llm_p95):
        # p95 is only known to be at least the timeout: broken if that is already over
        # the threshold, otherwise this run cannot tell (rerun with a longer timeout).
        lat_ok = False if timeout_floor >= A2_MAX_P95_LATENCY_S else None
    else:
        lat_ok = llm_p95 < A2_MAX_P95_LATENCY_S
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
    released_by_category = {}
    for cat in ADVERSARIAL_CATEGORIES:
        rows = [r for r in adv_all if r["adversarial_category"] == cat]
        released_by_category[cat] = _proportion(sum(1 for r in rows if r.get("decision") == RELEASE), len(rows))
    honest_done = [r for r in honest_all if r["label"] == "completed"]
    honest_release_rate = _proportion(sum(1 for r in honest_done if r.get("decision") == RELEASE), len(honest_done))
    anticheat = summarize_anticheat(records)
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
        # Checkpoint 3: the same rates per attack category, and how often an honest,
        # completed task still gets its wager released (the anti-cheating layer's cost).
        "released_by_category": released_by_category,
        "honest_completed_released": honest_release_rate,
        "anticheat": anticheat,
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
                "p95_llm_latency_s": None if llm_p95 is None or math.isinf(llm_p95) else llm_p95,
                # Set when timeouts push the p95 past what was measured: the p95 is at least this.
                "p95_llm_latency_at_least_s": timeout_floor if llm_p95 is not None and math.isinf(llm_p95) else None,
                "llm_timeouts": len(timed_out),
                "latency_status": _check(lat_ok),
                "thresholds": {"max_cost_usd": A2_MAX_COST_USD,
                               "max_p95_latency_s": A2_MAX_P95_LATENCY_S},
            },
            **({"4": _assumption_4(anticheat)} if anticheat else {}),
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


def host_info() -> dict:
    """The machine the run used. Local-model latency is a property of this hardware."""
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version()}
    if sys.platform == "darwin":
        for key, name in (("machdep.cpu.brand_string", "cpu"), ("hw.memsize", "memory_bytes")):
            try:
                info[name] = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                            text=True, timeout=2).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
    return info


def build_verifier(args) -> tuple[object, dict]:
    """The verdict model for --provider, plus what the run metadata should record."""
    if args.provider == "anthropic":
        model = args.model or DEFAULT_MODEL
        v = LLMOnlyVerifier(model=model, effort=args.effort, timeout_s=args.llm_timeout)
        return v, {"provider": "anthropic", "model": model, "effort": args.effort}
    v = QwenVerifier(provider=args.provider, model=args.model, base_url=args.base_url,
                     timeout_s=args.llm_timeout, json_mode=args.json_mode,
                     max_image_side=args.max_image_side)
    return v, {"provider": args.provider, "model": v.model, "json_mode": v.json_mode,
               "base_url": args.base_url or PROVIDERS[args.provider].base_url,
               "price_per_mtok": list(v.prices), "max_image_side": v.max_image_side}


def add_anticheat_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by bord.evaluate and bord.anticheat_eval."""
    cfg = AntiCheatConfig()
    parser.add_argument("--detector", choices=["none", "hf"], default="none",
                        help="pixel AI-image detector for the ai_image signal; 'hf' needs "
                             "requirements-detector.txt (default: none, the signal is skipped)")
    parser.add_argument("--detector-model", default="umm-maybe/AI-image-detector",
                        help="Hugging Face model id for --detector hf")
    parser.add_argument("--reuse-max-distance", default=cfg.reuse_max_distance, type=int,
                        help="flag a photo whose dhash is within this many of 64 bits of a past one")
    parser.add_argument("--scene-min", default=cfg.scene_min_score, type=float,
                        help="flag a pair whose scene similarity is below this")
    parser.add_argument("--ai-min", default=cfg.ai_min_score, type=float,
                        help="flag a photo whose detector P(AI) is at least this")


def build_anticheat(args) -> tuple[AntiCheat, dict]:
    detector = HFImageDetector(args.detector_model) if args.detector == "hf" else None
    config = AntiCheatConfig(reuse_max_distance=args.reuse_max_distance,
                             scene_min_score=args.scene_min, ai_min_score=args.ai_min)
    meta = {"reuse_max_distance": config.reuse_max_distance, "scene_min_score": config.scene_min_score,
            "ai_min_score": config.ai_min_score, "detector": detector.name if detector else None}
    return AntiCheat(config, detector), meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the BORD LLM-only baseline verifier.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    parser.add_argument("--provider", default="ollama", choices=["anthropic", *PROVIDERS],
                        help="where the verdict model runs (default: Qwen VL on local Ollama)")
    parser.add_argument("--model", help="model id; default depends on --provider")
    parser.add_argument("--base-url", help="override the provider's API base URL")
    parser.add_argument("--json-mode", choices=["schema", "object", "none"],
                        help="Qwen providers: how to request JSON output")
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="anthropic provider only")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=None,
                        help="one untimed call before the run so model loading is not measured "
                             "(default: on for local providers)")
    parser.add_argument("--llm-timeout", default=Budget().llm_s, type=float,
                        help="first-attempt LLM request timeout (stage budget), seconds")
    parser.add_argument("--deadline", default=Budget().server_deadline_s, type=float,
                        help="server-side firm deadline, seconds")
    parser.add_argument("--release-min-conf", default=0.0, type=float)
    parser.add_argument("--forfeit-min-conf", default=0.0, type=float)
    parser.add_argument("--max-image-side", default=MAX_IMAGE_SIDE, type=int,
                        help="long side in pixels of the photos sent to a Qwen model (fewer pixels, "
                             "fewer image tokens, faster calls); 768 is about 4x fewer tokens than 1568")
    parser.add_argument("--anticheat", action=argparse.BooleanOptionalAction, default=False,
                        help="run the Checkpoint 3 anti-cheating stage before the LLM call; without "
                             "it the run reproduces the Checkpoint 2 baseline on the same manifest")
    add_anticheat_args(parser)
    parser.add_argument("--limit", type=int, help="only the first N rows (smoke test)")
    args = parser.parse_args(argv)

    submissions = load_manifest(args.manifest)
    history_for = ManifestHistory(submissions)  # built before --limit so history is complete
    if args.limit:
        submissions = submissions[: args.limit]

    model_verifier, model_meta = build_verifier(args)
    warmup = args.warmup if args.warmup is not None else args.provider in ("ollama", "openai-compatible")
    warmup_meta = None
    if warmup and submissions:
        # Untimed: the first request to a local server includes loading the weights,
        # which is a cold-start cost, not steady-state verdict latency.
        w = model_verifier.verify(submissions[0])
        warmup_meta = {"latency_s": w.latency_s, "error": w.error}
        print(f"warm-up: {w.latency_s:.2f}s ({w.error or 'ok'})", file=sys.stderr)

    started = datetime.now(timezone.utc)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_meta["model"])
    anticheat, anticheat_meta = build_anticheat(args) if args.anticheat else (None, None)
    run_id = (started.strftime("%Y%m%dT%H%M%SZ") + f"_{safe_model}"
              + (f"_{args.max_image_side}px" if model_meta.get("max_image_side", MAX_IMAGE_SIDE) != MAX_IMAGE_SIDE else "")
              + ("_anticheat" if anticheat else ""))
    out_dir = args.out / run_id
    meta = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "manifest": str(args.manifest),
        "verifier": "llm_plus_anticheat" if anticheat else "llm_only_baseline",
        "anticheat": anticheat_meta,
        **model_meta,
        "warmup": warmup_meta,
        "host": host_info(),
        "llm_timeout_s": args.llm_timeout,
        "server_deadline_s": args.deadline,
        "release_min_conf": args.release_min_conf,
        "forfeit_min_conf": args.forfeit_min_conf,
    }
    budget = Budget(server_deadline_s=args.deadline, llm_s=args.llm_timeout)
    verifier = RealtimeVerifier(
        model_verifier,
        budget=budget,
        thresholds=Thresholds(args.release_min_conf, args.forfeit_min_conf),
        anticheat=anticheat,
        history_for=history_for if anticheat else None,
    )
    summary = run(submissions, verifier, out_dir, meta)

    print(json.dumps({k: summary[k] for k in ("honest_accuracy", "fooled_rate",
                                               "adversarial_released", "released_by_category",
                                               "latency_s", "realtime", "cost_usd",
                                               "assumptions")}, indent=2))
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
