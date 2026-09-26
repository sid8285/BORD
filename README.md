# BORD

A to-do list backed by money, where completion is checked by computer vision and AI.

A user creates a task with a deadline and a wager, captures before and after evidence in the browser, and an automated verifier decides whether the wager is returned or donated to charity. The project focuses on making that verification **trustworthy**, not just possible:

1. **Anti-cheating:** stopping old, reused, web-sourced, or AI-generated photos.
2. **Hard-to-capture tasks:** tasks that one photo cannot prove (multi-view, time-spread).
3. **Confidence before money moves:** calibrated thresholds, measured false-accept and false-reject rates.
4. **Real-time budget:** under 1 s of in-browser checks and a verdict within 10 s at p95.

CS 4220/6235 Real Time Embedded Systems, Fall 2026, Group 34 (solo).

## Status

Checkpoint 1: planning only. There is no code yet.

## Docs

- [Dataset spec](docs/dataset_spec.md)
- [Related work](docs/related_work.md)
- [Technology assumptions and platforms](docs/assumptions.md)

## Planned stack

Python FastAPI backend, React frontend, a frontier multimodal LLM API, Stripe (test mode only; no real money moves).
