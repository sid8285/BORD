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

- `backend/bord/dataset.py`: manifest loader for the self-collected and adversarial sets (schema in `data/manifest.example.csv`, following `docs/dataset_spec.md`). Adversarial rows reference the honest pair they were built from; their correct verdict is always `not_completed`.
- `backend/bord/verifier.py`: LLM-only baseline. One call with the task text, the before photo and the after photo; returns verdict, confidence, latency, tokens and cost. Images are downscaled and re-encoded (EXIF dropped). Model via `BORD_MODEL`, effort via `BORD_EFFORT`, timeout via `BORD_TIMEOUT_S` (default 10 s).
- `backend/bord/evaluate.py`: runs the verifier over a manifest and writes `results/<run_id>/predictions.jsonl` and `summary.json` (honest accuracy, false-reject rate, fooled rate per adversarial category, latency p50/p95, cost per call, and a holding/broken check for Assumptions 1 and 2).

```bash
cd backend
pip install -r requirements.txt
pytest                       # offline tests, no API key needed
export ANTHROPIC_API_KEY=...
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --limit 2   # smoke test
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results
```

Commit the `results/<run_id>/` folder; it is the evidence for Assumptions 1 and 2.
