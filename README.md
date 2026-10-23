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
Checkpoint 3: anti-cheating layer, camera-only capture flow and the latency re-measurement tooling (see below).

## Docs

- [Dataset spec](docs/dataset_spec.md)
- [Related work](docs/related_work.md)
- [Technology assumptions and platforms](docs/assumptions.md)
- [Real-time budget](docs/realtime_budget.md)
- [Collecting pairs](docs/collecting_pairs.md)
- [Anti-cheating layer](docs/anticheat.md)

## Planned stack

Python FastAPI backend, React frontend, a frontier multimodal LLM API, Stripe (test mode only; no real money moves).

## Checkpoint 2: baseline verifier and evaluation harness

Starter code from the course's `ai-suggestions/cp2` branch was imported unmodified in its own commit and then extended. See `git log`.

- `backend/bord/dataset.py`: manifest loader and validator for the honest and adversarial sets (schema in `data/manifest.example.csv`, following `docs/dataset_spec.md`).
- `backend/bord/build_manifest.py`: builds `data/manifest.csv` from the photo folders. It generates `reused`, `wrong_scene` and `web_sourced` attacks and picks up hand-made `staged_partial` and `ai_edited` ones ([how to collect](docs/collecting_pairs.md)).
- `backend/bord/qwen_verifier.py`: the verdict model, **Qwen3-VL 8B Instruct**. It makes one call with the task text and both photos over an OpenAI-compatible API, so the same code runs against Ollama on this machine (the default), Alibaba Cloud Model Studio, or a vLLM server. Thinking is off and temperature is 0.
- `backend/bord/verifier.py`: the starter's Claude verifier, kept for an optional frontier-model comparison, plus the shared image preprocessing. Images are downscaled to 1568 px and re-encoded, which drops EXIF.
- `backend/bord/realtime.py`: the server side of the [real-time budget](docs/realtime_budget.md). It enforces stage budgets and a firm 8 s server deadline, retries once on a fast failure, and falls back to a hold so no money moves.
- `backend/bord/evaluate.py`: writes `results/<run_id>/predictions.jsonl` and `summary.json`. The summary covers accuracy, fooled rate overall and per category (with 95% Wilson intervals), per-stage latency, deadline misses, decisions, cost, and the status of Assumptions 1 and 2.

One-time setup for local Qwen (macOS): install [Ollama](https://ollama.com/download) 0.12.7 or later, start it, then pull the **instruct** model. The untagged `qwen3-vl:8b` is the thinking variant, which is too slow for the 6.5 s LLM budget.

```bash
ollama pull qwen3-vl:8b-instruct
```

```bash
cd backend
pip install -r requirements.txt      # tested with anthropic 1.12.1, openai 3.26.1, pillow 12.3.0, pytest 9.1.1
pytest                               # offline tests, no model server or API key needed
python -m bord.build_manifest --data ../data
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --limit 2      # smoke test
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results                # Qwen3-VL 8B on Ollama
python -m bord.compare_runs ../results/*/ > ../docs/results_cp2.md                     # comparison table for the report
```

Optional runs, each adding a row to the comparison:

```bash
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --model qwen3-vl:4b-instruct    # smaller local model
DASHSCOPE_API_KEY=... python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --provider dashscope   # hosted qwen3-vl-plus
ANTHROPIC_API_KEY=... python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --provider anthropic  # frontier comparison (Opus 5.5)
```

For local providers the harness makes one untimed warm-up call first, on a generated photo pair that is not in the manifest, so neither model loading nor a prompt-cache hit on the first row is counted as verdict latency, and it records the machine (`run.host`) in `summary.json`.

Commit each `results/<run_id>/` folder, `docs/results_cp2.md` and `data/manifest.csv`. They are the evidence for Assumptions 1 and 2 and the real-time budget.

## Checkpoint 3: anti-cheating layer and capture flow

Details and the signal table are in [docs/anticheat.md](docs/anticheat.md).

- `backend/bord/challenge.py`: server-issued, single-use capture challenges (code plus angle prompt) and server-clock ordering of the before and after captures.
- `backend/bord/stamp.py`: the challenge code as a cell strip the browser draws into each frame and the server reads back on upload.
- `backend/bord/imaging.py`: Pillow-only fingerprints, a 64-bit dhash for reuse and a scene-similarity score for the before/after consistency check.
- `backend/bord/detectors.py`: the optional pixel AI-image detector (Hugging Face model, `requirements-detector.txt`) and the file-metadata diagnostic.
- `backend/bord/anticheat.py`: runs the signals as stage 4 of the real-time pipeline; `ManifestHistory` stands in for a user's submission log in the offline benchmark.
- `backend/bord/anticheat_eval.py`: measures the signals on the manifest without an LLM and prints threshold calibration data.
- `backend/bord/api.py`: FastAPI service (`/api/challenges`, `/api/captures`, `/api/submissions`, `/api/device-reports`).
- `frontend/`: React capture flow with no file upload (`getUserMedia`, challenge code drawn into the frame) and the `?device-test` page for Assumption 3.
- `backend/bord/latency_sweep.py`: re-runs the manifest across model size, image size and host for Assumption 2.

```bash
cd backend
pip install -r requirements.txt
pytest                                                    # offline tests, no model server needed
python -m bord.anticheat_eval --manifest ../data/manifest.csv --out ../results
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --llm-timeout 60 --deadline 70 --anticheat
python -m bord.latency_sweep --manifest ../data/manifest.csv --out ../results
uvicorn bord.api:app --port 8000                          # API on :8000

cd ../frontend && npm install && npm run dev              # app on :5173, proxies /api to :8000
```

`getUserMedia` needs https or localhost; use a tunnel or a local certificate to try it on a phone.
