"""A real Stripe test-mode authorize / capture / cancel round trip (Checkpoint 4, Assumption 5).

The course branch tested the hold logic against a fake ``stripe`` module only. This runs
the same ``StripeHolds`` class against Stripe's test API and commits what came back, so the
evidence is Stripe's own ids and statuses rather than a mock:

1. **release path**: authorize a hold, then cancel it (the user did the task).
2. **forfeit path**: authorize a hold, then capture it (the user did not).
3. **double settle**: try to capture the cancelled hold, which Stripe must refuse.

After each step the PaymentIntent is read back from Stripe (``retrieve``), and for each
authorization the card's ``capture_before`` time is recorded: that is Stripe's own statement
of how long the hold stays capturable, which is what Assumption 5 is about (7 days). Every
call is timed against the 0.5 s decision-and-Stripe stage of docs/realtime_budget.md.

Writes ``results/stripe_<time>/roundtrip.jsonl`` (one line per call), ``ledger.jsonl`` (the
app's money ledger for these holds) and ``summary.json``. Refuses anything but a test key.
Usage (from ``backend/``, with a key from https://dashboard.stripe.com/test/apikeys)::

    STRIPE_SECRET_KEY=sk_test_... python -m bord.stripe_roundtrip --out ../results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .payments import HOLD_WINDOW_S, HoldError, StripeHolds
from .realtime import Budget

DAY_S = 24 * 3600


def _intent_state(stripe, key: str, pid: str) -> dict:
    """What Stripe says about a PaymentIntent now, including the card's capture deadline."""
    intent = stripe.PaymentIntent.retrieve(pid, api_key=key, expand=["latest_charge"])
    charge = intent.get("latest_charge") or {}
    card = ((charge.get("payment_method_details") or {}).get("card") or {}) if charge else {}
    return {
        "status": intent["status"],
        "amount": intent["amount"],
        "amount_received": intent.get("amount_received"),
        "capture_method": intent.get("capture_method"),
        "livemode": intent.get("livemode"),
        "created": intent.get("created"),
        "charge_id": charge.get("id") if charge else None,
        "captured": charge.get("captured") if charge else None,
        "capture_before": card.get("capture_before"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stripe test-mode hold round trip.")
    parser.add_argument("--out", default=Path("results"), type=Path)
    parser.add_argument("--amount", default=500, type=int, help="cents per hold (default $5.00)")
    args = parser.parse_args(argv)

    key = os.environ.get("STRIPE_SECRET_KEY", "")
    started = datetime.now(timezone.utc)
    out_dir = args.out / ("stripe_" + started.strftime("%Y%m%dT%H%M%SZ"))
    try:
        holds = StripeHolds(key, ledger_path=out_dir / "ledger.jsonl")
    except HoldError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    stripe = holds.stripe
    out_dir.mkdir(parents=True, exist_ok=True)
    budget_s = Budget().decision_s
    lines: list[dict] = []

    def step(name: str, task_id: str, fn, pid: str | None = None):
        t0 = time.perf_counter()
        error, hold = None, None
        try:
            hold = fn()
        except HoldError as e:
            error = str(e)
        elapsed = time.perf_counter() - t0
        hid = hold.id if hold else pid
        state = _intent_state(stripe, key, hid) if hid else None
        line = {"step": name, "task_id": task_id, "hold_id": hid, "call_s": round(elapsed, 4),
                "within_stage_budget": elapsed <= budget_s, "error": error, "stripe": state}
        lines.append(line)
        print(json.dumps(line))
        return hold

    run = started.strftime("%Y%m%dT%H%M%SZ")
    a = step("authorize", f"rt-{run}-release", lambda: holds.authorize(f"rt-{run}-release", args.amount))
    if a is None:
        print("the first authorization failed; nothing else to try", file=sys.stderr)
        return 1
    step("release (cancel)", f"rt-{run}-release", lambda: holds.release(a.id, f"rt-{run}-release"), a.id)
    b = step("authorize", f"rt-{run}-forfeit", lambda: holds.authorize(f"rt-{run}-forfeit", args.amount))
    if b is None:
        print("the second authorization failed", file=sys.stderr)
        return 1
    step("forfeit (capture)", f"rt-{run}-forfeit", lambda: holds.forfeit(b.id, f"rt-{run}-forfeit"), b.id)
    step("capture after cancel (must fail)", f"rt-{run}-release", lambda: holds.forfeit(a.id, f"rt-{run}-release"),
         a.id)

    with (out_dir / "roundtrip.jsonl").open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")

    auths = [line for line in lines if line["step"] == "authorize" and line["stripe"]]
    windows = []
    for line in auths:
        st = line["stripe"]
        if st.get("capture_before") and st.get("created"):
            windows.append((st["capture_before"] - st["created"]) / DAY_S)
    calls = [line["call_s"] for line in lines]
    final = {line["step"]: (line["stripe"] or {}).get("status") for line in lines}
    summary = {
        "run_at": started.isoformat(),
        "mode": holds.mode,
        "livemode_seen": any((line["stripe"] or {}).get("livemode") for line in lines),
        "hold_ids": sorted({line["hold_id"] for line in lines if line["hold_id"]}),
        "statuses": final,
        "release_path_ok": lines[1]["error"] is None and (lines[1]["stripe"] or {}).get("status") == "canceled",
        "forfeit_path_ok": lines[3]["error"] is None and (lines[3]["stripe"] or {}).get("status") == "succeeded",
        "double_settle_refused": lines[4]["error"] is not None,
        "capture_window_days": windows,
        "assumption_5": {
            "threshold_days": HOLD_WINDOW_S / DAY_S,
            "status": ("Untested" if not windows else
                       "Holding" if min(windows) >= HOLD_WINDOW_S / DAY_S - 0.01 else "Broken"),
        },
        "stripe_call_s": {"max": max(calls), "mean": sum(calls) / len(calls), "budget_s": budget_s,
                          "over_budget": sum(1 for c in calls if c > budget_s)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", file=sys.stderr)
    ok = summary["release_path_ok"] and summary["forfeit_path_ok"] and summary["double_settle_refused"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
