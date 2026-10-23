"""Measure the anti-cheating signals on the manifest without calling an LLM (Checkpoint 3).

The LLM runs are slow (about 10 s per submission on the Checkpoint 2 hardware);
the deterministic signals take a fraction of a second, so they can be measured,
tuned and re-measured on their own. This writes ``results/anticheat_<time>/``:

- ``signals.jsonl``: one line per manifest row with every signal's flag and score.
- ``summary.json``: per signal and per attack category, how many honest submissions
  it flagged (false alarms) and how many attacks (detections), with 95% Wilson
  intervals; Assumption 4 if a pixel detector was used; and the score distributions
  the thresholds should be chosen from (``calibration``).

Usage (from ``backend/``)::

    python -m bord.anticheat_eval --manifest ../data/manifest.csv --out ../results
    python -m bord.anticheat_eval --manifest ../data/manifest.csv --out ../results --detector hf

Run ``bord.evaluate --anticheat`` afterwards to see the combined effect on the
fooled rate next to the Checkpoint 2 baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .anticheat import ManifestHistory
from .dataset import Submission, load_manifest
from .evaluate import _assumption_4, add_anticheat_args, build_anticheat, host_info, summarize_anticheat


def _scores(records: list[dict], signal: str, category: str | None, honest: bool) -> list[float]:
    out = []
    for r in records:
        if (r["set"] == "honest") != honest:
            continue
        if category is not None and r["adversarial_category"] != category:
            continue
        s = next((s for s in r["anticheat"]["signals"] if s["name"] == signal), None)
        if s is not None and s["score"] is not None:
            out.append(s["score"])
    return out


def _range(values: list[float]) -> dict | None:
    return {"n": len(values), "min": min(values), "max": max(values)} if values else None


def calibration(records: list[dict]) -> dict:
    """Score ranges per group and, where the groups do not overlap, a threshold between them.

    scene_consistency flags scores *below* the threshold: it should sit above the highest
    wrong_scene score and below the lowest honest one. reuse flags distances *at or below*
    the threshold: it should be at least the largest reused distance and below the smallest
    honest one. Small samples: treat a suggestion as a starting point, not a result.
    """
    honest_scene = _scores(records, "scene_consistency", None, honest=True)
    wrong_scene = _scores(records, "scene_consistency", "wrong_scene", honest=False)
    honest_dist = _scores(records, "reuse", None, honest=True)
    reused_dist = _scores(records, "reuse", "reused", honest=False)

    scene_hint = None
    if honest_scene and wrong_scene:
        if max(wrong_scene) < min(honest_scene):
            scene_hint = {"separable": True, "suggested_scene_min": round((max(wrong_scene) + min(honest_scene)) / 2, 3)}
        else:
            scene_hint = {"separable": False, "note": "the highest wrong_scene score is not below the lowest honest score"}
    reuse_hint = None
    if honest_dist and reused_dist:
        if max(reused_dist) < min(honest_dist):
            reuse_hint = {"separable": True, "suggested_reuse_max_distance": int(max(reused_dist))}
        else:
            reuse_hint = {"separable": False, "note": "a reused photo is no closer than the closest honest one"}
    return {
        "scene_consistency": {"honest": _range(honest_scene), "wrong_scene": _range(wrong_scene), **(scene_hint or {})},
        "reuse_distance": {"honest": _range(honest_dist), "reused": _range(reused_dist), **(reuse_hint or {})},
        "ai_image": {"honest": _range(_scores(records, "ai_image", None, honest=True)),
                     "ai_edited": _range(_scores(records, "ai_image", "ai_edited", honest=False))},
    }


def record_for(sub: Submission, report) -> dict:
    return {
        "pair_id": sub.pair_id, "set": sub.set, "label": sub.label,
        "adversarial_category": sub.adversarial_category, "base_pair_id": sub.base_pair_id,
        "anticheat": report.to_dict(), "stages": {"anticheat_s": report.elapsed_s},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the BORD anti-cheating signals on a manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    add_anticheat_args(parser)
    args = parser.parse_args(argv)

    submissions = load_manifest(args.manifest)
    history_for = ManifestHistory(submissions)
    anticheat, anticheat_meta = build_anticheat(args)

    started = datetime.now(timezone.utc)
    run_id = "anticheat_" + started.strftime("%Y%m%dT%H%M%SZ") + ("_detector" if args.detector != "none" else "")
    out_dir = args.out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with (out_dir / "signals.jsonl").open("w", encoding="utf-8") as fh:
        for sub in submissions:
            record = record_for(sub, anticheat.check(sub, history_for(sub)))
            records.append(record)
            fh.write(json.dumps(record) + "\n")
            flagged = ",".join(record["anticheat"]["flagged_by"]) or "-"
            print(f"{sub.pair_id}: flagged by {flagged}", file=sys.stderr)

    stats = summarize_anticheat(records)
    summary = {
        "run": {"run_id": run_id, "started_at": started.isoformat(), "manifest": str(args.manifest),
                "anticheat": anticheat_meta, "host": host_info()},
        "n_submissions": len(records),
        "anticheat": stats,
        "assumption_4": _assumption_4(stats) if stats else None,
        "calibration": calibration(records),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"overall": stats["overall"]["by_category"], "honest_flagged": stats["overall"]["honest"],
                      "calibration": summary["calibration"]}, indent=2))
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
