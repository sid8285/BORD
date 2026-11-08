"""Measure the per-region staged-partial check on the photos, without an LLM (Checkpoint 4).

Runs bord.regions on every honest pair and every staged-partial attack in the given
manifests and writes ``results/regions_<time>/``:

- ``regions.jsonl``: one line per pair with every feature (changed share, busy-unchanged
  share, clutter ratio, re-frame, alignment) and whether the default rule flags it.
- ``summary.json``: how many honest completed pairs the rule flags (false holds) and how
  many staged-partial attacks it catches, with 95% Wilson intervals, the stage time, and
  for each feature the honest and staged ranges and the best single threshold on it
  (``separation``), which is what a Checkpoint 5 rule should be chosen from.

Assumption 7 (docs/assumptions.md) is judged on these numbers together with the
``--prompt regions`` LLM run. Usage (from ``backend/``)::

    python -m bord.region_eval --manifest ../data/manifest.csv --manifest ../data/manifest_staged.csv --out ../results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .dataset import Submission, load_manifest
from .evaluate import _proportion, host_info
from .regions import RegionConfig, region_report

FEATURES = ("changed_frac", "busy_unchanged_frac", "clutter_ratio", "reframe")
# Assumption 7 targets: catch at least half of the staged-partial attacks while holding
# at most 10% of honest completed pairs, on at least 20 attacks.
A7_MIN_CATCH = 0.5
A7_MAX_FALSE_HOLD = 0.10
A7_MIN_ATTACKS = 20


def _rows(manifests: list[Path]) -> list[Submission]:
    seen, out = set(), []
    for m in manifests:
        for s in load_manifest(m):
            if s.pair_id in seen:
                continue
            seen.add(s.pair_id)
            if (s.set == "honest" and s.label == "completed") or s.adversarial_category == "staged_partial":
                out.append(s)
    return out


def separation(honest: list[float], staged: list[float]) -> dict:
    """Best single threshold on one feature: flag at >= t (or <= t), maximising
    catches minus false holds. Reported, not used: with a handful of attacks it overfits."""
    best = None
    for t in sorted(set(honest + staged)):
        for direction in (">=", "<="):
            hit = (lambda v: v >= t) if direction == ">=" else (lambda v: v <= t)
            caught = sum(hit(v) for v in staged)
            false = sum(hit(v) for v in honest)
            score = caught / max(1, len(staged)) - false / max(1, len(honest))
            if best is None or score > best["score"]:
                best = {"threshold": t, "direction": direction, "caught": caught, "n_staged": len(staged),
                        "false_holds": false, "n_honest": len(honest), "score": score}
    return best or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the per-region staged-partial check.")
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    args = parser.parse_args(argv)

    config = RegionConfig()
    started = datetime.now(timezone.utc)
    out_dir = args.out / ("regions_" + started.strftime("%Y%m%dT%H%M%SZ"))
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with (out_dir / "regions.jsonl").open("w", encoding="utf-8") as fh:
        for sub in _rows(args.manifest):
            t0 = time.perf_counter()
            rep = region_report(sub.before_path, sub.after_path, config)
            elapsed = time.perf_counter() - t0
            rec = {"pair_id": sub.pair_id, "set": sub.set, "adversarial_category": sub.adversarial_category,
                   "flagged": rep.flagged, "reasons": list(rep.reasons), "elapsed_s": elapsed,
                   **{f: getattr(rep, f) for f in FEATURES},
                   "scale": rep.scale, "shift": list(rep.shift), "align_mad": rep.align_mad}
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")

    honest = [r for r in records if r["set"] == "honest"]
    staged = [r for r in records if r["adversarial_category"] == "staged_partial"]
    false_hold = _proportion(sum(r["flagged"] for r in honest), len(honest))
    caught = _proportion(sum(r["flagged"] for r in staged), len(staged))
    times = sorted(r["elapsed_s"] for r in records)
    summary = {
        "run": {"started_at": started.isoformat(), "manifests": [m.as_posix() for m in args.manifest],
                "config": config.__dict__, "host": host_info()},
        "honest_completed_flagged": false_hold,
        "staged_partial_caught": caught,
        "stage_s": {"p50": times[len(times) // 2] if times else None,
                    "p95": times[min(len(times) - 1, int(0.95 * len(times)))] if times else None,
                    "max": times[-1] if times else None},
        "features": {f: {"honest": [min((r[f] for r in honest), default=None), max((r[f] for r in honest), default=None)],
                         "staged_partial": [min((r[f] for r in staged), default=None),
                                            max((r[f] for r in staged), default=None)],
                         "separation": separation([r[f] for r in honest], [r[f] for r in staged])}
                     for f in FEATURES},
        "assumption_7_pixel_rule": {
            "targets": {"min_catch": A7_MIN_CATCH, "max_false_hold": A7_MAX_FALSE_HOLD,
                        "min_attacks": A7_MIN_ATTACKS},
            "enough_attacks": len(staged) >= A7_MIN_ATTACKS,
            "meets_catch": None if caught["rate"] is None else caught["rate"] >= A7_MIN_CATCH,
            "meets_false_hold": None if false_hold["rate"] is None else false_hold["rate"] <= A7_MAX_FALSE_HOLD,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("honest_completed_flagged", "staged_partial_caught", "stage_s",
                                              "assumption_7_pixel_rule")}, indent=2))
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
