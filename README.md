# BORD

A to-do list backed by money, where completion is checked by computer vision and AI.

A user creates a task with a deadline and a wager, captures before and after evidence in the browser, and an automated verifier decides whether the wager is returned or donated to charity. The project focuses on making that verification **trustworthy**, not just possible:

1. **Anti-cheating:** stopping old, reused, web-sourced, or AI-generated photos.
2. **Hard-to-capture tasks:** tasks that one photo cannot prove (multi-view, time-spread).
3. **Confidence before money moves:** calibrated thresholds, measured false-accept and false-reject rates.
4. **Real-time budget:** under 1 s of in-browser checks and a verdict within 10 s at p95.

CS 4220/6235 Real Time Embedded Systems, Fall 2026, Group 34 (solo).

## Status

Checkpoint 1: planning only (submitted at `5f4de2bf`).
Checkpoint 2: LLM-only baseline verifier, evaluation harness, real-time deadline handling and the first evaluation runs.

## Docs

- [Dataset spec](docs/dataset_spec.md)
- [Related work](docs/related_work.md)
- [Technology assumptions and platforms](docs/assumptions.md)
- [Real-time budget](docs/realtime_budget.md)
- [Collecting pairs](docs/collecting_pairs.md)

## Planned stack

Python FastAPI backend, React frontend, a frontier multimodal LLM API, Stripe (test mode only; no real money moves).

## Checkpoint 2: baseline verifier and evaluation harness

Starter code from the course's `ai-suggestions/cp2` branch was imported unmodified in its own commit and then extended. See `git log`.

- `backend/bord/dataset.py`: manifest loader and validator for the honest and adversarial sets (schema in `data/manifest.example.csv`, following `docs/dataset_spec.md`).
- `backend/bord/build_manifest.py`: builds `data/manifest.csv` from the photo folders. It generates `reused`, `wrong_scene` and `web_sourced` attacks and picks up hand-made `staged_partial` and `ai_edited` ones ([how to collect](docs/collecting_pairs.md)).
- `backend/bord/verifier.py`: the LLM-only baseline. It makes one call with the task text and both photos and returns a verdict and confidence. Images are downscaled and re-encoded, which drops EXIF.
- `backend/bord/realtime.py`: the server side of the [real-time budget](docs/realtime_budget.md). It enforces stage budgets and a firm 8 s server deadline, retries once on a fast failure, and falls back to a hold so no money moves.
- `backend/bord/evaluate.py`: writes `results/<run_id>/predictions.jsonl` and `summary.json`. The summary covers accuracy, fooled rate overall and per category (with 95% Wilson intervals), per-stage latency, deadline misses, decisions, cost, and the status of Assumptions 1 and 2.

```bash
cd backend
pip install -r requirements.txt      # tested with anthropic 1.12.1, pillow 12.3.0, pytest 9.1.1
pytest                               # offline tests, no API key needed
python -m bord.build_manifest --data ../data
export ANTHROPIC_API_KEY=...
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --limit 2      # smoke test
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results                # Opus 5.5 baseline
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --model claude-sonnet-5-5
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --model claude-haiku-5-5
```

Commit each `results/<run_id>/` folder together with `data/manifest.csv`. They are the evidence for Assumptions 1 and 2 and the real-time budget.
