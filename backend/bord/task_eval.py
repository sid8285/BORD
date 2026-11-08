"""False-hold rate on honest multi-view and time-spread submissions (Checkpoint 4, Assumption 6).

The course branch added the two task types and their cross-photo checks
(bord.task_checks) but never measured them on real photos. A check that holds honest
users' wagers for manual review is a cost too, so Assumption 6 sets a falsifiable target:
**at most 10% of honest, completed multi-view and time-spread submissions end as a hold,
measured on at least 20 of each type.**

Two commands (from ``backend/``):

``plan``: draws a time-spread task's check-in schedule the same way the server does
(bord.tasks.schedule_checkins), adds the task to ``data/tasks.jsonl`` and prints when to
take each photo. The BEFORE photo starts the session; check-ins are matched to their
windows by the photos' EXIF capture times, exactly as the server matches them by
arrival time::

    python -m bord.task_eval plan --data ../data --task-id ts01 --description "Study at the desk" --session-s 1200 --checkins 3

``run``: verifies every task in ``data/tasks.jsonl`` and writes ``results/tasks_<time>/``
(``tasks.jsonl`` per task, ``summary.json`` with the false-hold rate per type and the
Assumption 6 status). ``--provider none`` (the default) measures what the deterministic
checks alone hold: the anti-cheating signals on each pair plus the cross-photo checks,
with the verdict taken as "completed". ``--provider anthropic`` adds the real LLM call per
pair through the real-time pipeline, which is the number the assumption is judged on::

    python -m bord.task_eval run --data ../data --out ../results
    python -m bord.task_eval run --data ../data --out ../results --provider anthropic

Row format (paths relative to ``data/``)::

    {"task_id": "mv01", "task_type": "multi_view", "description": "Make the bed",
     "views": [{"angle": "from the left", "before": "tasks/mv01/v1_before.jpg", "after": "tasks/mv01/v1_after.jpg"}, ...]}
    {"task_id": "ts01", "task_type": "time_spread", "description": "Study at the desk", "session_s": 1200,
     "checkin_offsets": [312.4, 655.0, 901.7], "before": "tasks/ts01/before.jpg", "after": "tasks/ts01/after.jpg",
     "checkins": ["tasks/ts01/c1.jpg", "tasks/ts01/c2.jpg", "tasks/ts01/c3.jpg"]}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .anticheat import AntiCheat
from .dataset import Submission
from .evaluate import _check, _proportion, _significant, host_info
from .realtime import FORFEIT, HOLD, RELEASE, Budget, RealtimeVerifier
from .task_checks import check_multi_view, check_time_spread, combine_decisions
from .tasks import CHECKIN_WINDOW_S, MAX_CHECKINS, MIN_CHECKINS, Task, schedule_checkins

A6_MAX_FALSE_HOLD = 0.10
A6_MIN_PER_TYPE = 20
TYPES = ("multi_view", "time_spread")
EXIF_IFD, DATETIME_ORIGINAL, DATETIME = 0x8769, 36867, 306


def capture_time(path: Path) -> float:
    """EXIF capture time in seconds (local time is fine: only differences are used)."""
    with Image.open(path) as img:
        exif = img.getexif()
        raw = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
    if not raw:
        raise ValueError(f"{path} has no EXIF capture time; time-spread photos need one")
    return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


class _AlwaysCompleted:
    """--provider none: the verdict is 'completed', so only the checks can hold a task."""

    model = "none"

    def prepare(self, sub):
        return []

    def call(self, sub, content, timeout_s):
        from .verifier import VerifierResult
        return VerifierResult(sub.pair_id, "completed", 1.0, "checks only (no LLM)", 0.0, model="none")


@dataclass
class Outcome:
    decision: str
    pair_decisions: list[str]
    task_check: dict
    reasons: list[str]
    cost_usd: float


def _sub(task_id: str, i: int, task_type: str, description: str, before: Path, after: Path) -> Submission:
    return Submission(pair_id=f"{task_id}-v{i}", set="honest", task_type=task_type, task_description=description,
                      before_path=before, after_path=after, label="completed", adversarial_category=None,
                      base_pair_id=None, capture_source="phone", before_captured_at="", after_captured_at="")


def evaluate_task(row: dict, data: Path, pipeline: RealtimeVerifier, anticheat: AntiCheat) -> Outcome:
    task_type = row["task_type"]
    if task_type == "multi_view":
        views = row["views"]
        subs = [_sub(row["task_id"], i, task_type, f"{row['description']} (view {i + 1}: {v['angle']})",
                     data / v["before"], data / v["after"]) for i, v in enumerate(views)]
        results = [pipeline.verify(s) for s in subs]
        extra = check_multi_view([s.after_path for s in subs])
    elif task_type == "time_spread":
        before, after = data / row["before"], data / row["after"]
        start = capture_time(before)
        task = Task(id=row["task_id"], user_id="eval", description=row["description"], task_type=task_type,
                    wager_cents=500, created_at=start, deadline_at=start + 7 * 24 * 3600,
                    session_s=float(row["session_s"]), n_checkins=len(row["checkin_offsets"]),
                    session_started_at=start, checkin_offsets=tuple(row["checkin_offsets"]))
        # The server's rule (TaskStore.record_checkin): a photo fills the first open window it
        # arrived in; one outside every window is not counted.
        for path in row["checkins"]:
            t = capture_time(data / path)
            for i in range(len(task.checkin_offsets)):
                opens, closes = task.checkin_window(i)
                if i not in task.checkins and opens <= t <= closes:
                    task.checkins[i] = {"capture_id": path, "received_at": t, "path": str(data / path)}
                    break
        subs = [_sub(row["task_id"], 0, task_type, row["description"], before, after)]
        results = [pipeline.verify(subs[0])]
        extra = check_time_spread(task, before, anticheat.config)
    else:
        raise ValueError(f"task_eval measures multi_view and time_spread tasks, got {task_type!r}")

    decision = combine_decisions([r.decision for r in results])
    if not extra.ok and decision == RELEASE:
        decision = HOLD
    reasons = [r.reason for r in results if r.decision != RELEASE] + ([] if extra.ok else [extra.detail])
    for r in results:
        if r.anticheat.get("flagged_by"):
            reasons.append(f"{r.pair_id}: flagged by {', '.join(r.anticheat['flagged_by'])}")
    return Outcome(decision, [r.decision for r in results], {"ok": extra.ok, "detail": extra.detail}, reasons,
                   sum(r.cost_usd for r in results))


def summarize(records: list[dict]) -> dict:
    out = {}
    for t in TYPES:
        rows = [r for r in records if r["task_type"] == t]
        held = _proportion(sum(r["decision"] != RELEASE for r in rows), len(rows))
        rate, ci = held["rate"], held["ci95"]
        point_ok = None if rate is None else rate <= A6_MAX_FALSE_HOLD
        ci_all = None if ci is None else ci[1] <= A6_MAX_FALSE_HOLD
        ci_any = None if ci is None else ci[0] <= A6_MAX_FALSE_HOLD
        out[t] = {"false_hold": held, "status": _check(point_ok, ci_all, ci_any),
                  "significant": _significant(point_ok, ci_all, ci_any),
                  "enough_tasks": len(rows) >= A6_MIN_PER_TYPE,
                  "forfeited": sum(r["decision"] == FORFEIT for r in rows),
                  "held_by_task_check": sum(not r["task_check"]["ok"] for r in rows)}
    return out


def cmd_plan(args) -> int:
    if not MIN_CHECKINS <= args.checkins <= MAX_CHECKINS:
        raise SystemExit(f"--checkins must be {MIN_CHECKINS} to {MAX_CHECKINS}")
    offsets = list(schedule_checkins(args.session_s, args.checkins, random.SystemRandom()))
    folder = f"tasks/{args.task_id}"
    row = {"task_id": args.task_id, "task_type": "time_spread", "description": args.description,
           "session_s": args.session_s, "checkin_offsets": offsets,
           "before": f"{folder}/before.jpg", "after": f"{folder}/after.jpg",
           "checkins": [f"{folder}/c{i + 1}.jpg" for i in range(args.checkins)]}
    manifest = args.data / "tasks.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"Added {args.task_id} to {manifest}. Photos go in {args.data / folder}.")
    print("0:00  BEFORE photo (this starts the session)")
    for i, off in enumerate(offsets):
        m, s = divmod(int(off), 60)
        print(f"{m}:{s:02d}  check-in {i + 1} (window closes {int(CHECKIN_WINDOW_S)} s later)")
    m, s = divmod(int(args.session_s), 60)
    print(f"{m}:{s:02d}  AFTER photo (any time after the session ends)")
    return 0


def cmd_run(args) -> int:
    rows = [json.loads(line) for line in (args.data / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    anticheat = AntiCheat()
    if args.provider == "anthropic":
        from .verifier import LLMOnlyVerifier
        model = LLMOnlyVerifier(timeout_s=Budget().llm_s)
    else:
        model = _AlwaysCompleted()
    pipeline = RealtimeVerifier(model, anticheat=anticheat)
    started = datetime.now(timezone.utc)
    out_dir = args.out / ("tasks_" + started.strftime("%Y%m%dT%H%M%SZ") + f"_{model.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (out_dir / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            o = evaluate_task(row, args.data, pipeline, anticheat)
            rec = {"task_id": row["task_id"], "task_type": row["task_type"], "decision": o.decision,
                   "pair_decisions": o.pair_decisions, "task_check": o.task_check, "reasons": o.reasons,
                   "cost_usd": o.cost_usd}
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            print(f"{row['task_id']}: {o.decision} ({o.task_check['detail']})", file=sys.stderr)
    summary = {"run": {"started_at": started.isoformat(), "provider": args.provider, "model": model.model,
                       "host": host_info(), "tasks": len(records)},
               "assumption_6": {"threshold": A6_MAX_FALSE_HOLD, "min_per_type": A6_MIN_PER_TYPE,
                                **summarize(records)},
               "cost_usd": sum(r["cost_usd"] for r in records)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["assumption_6"], indent=2))
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Honest multi-view / time-spread false-hold rate (Assumption 6).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="draw a time-spread check-in schedule and add the task to tasks.jsonl")
    p.add_argument("--data", default=Path("data"), type=Path)
    p.add_argument("--task-id", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--session-s", default=1200.0, type=float)
    p.add_argument("--checkins", default=3, type=int)
    r = sub.add_parser("run", help="verify every task in tasks.jsonl")
    r.add_argument("--data", default=Path("data"), type=Path)
    r.add_argument("--out", default=Path("results"), type=Path)
    r.add_argument("--provider", choices=["none", "anthropic"], default="none")
    args = parser.parse_args(argv)
    return cmd_plan(args) if args.cmd == "plan" else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
