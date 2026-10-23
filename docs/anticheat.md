# Anti-cheating layer (Checkpoint 3)

The LLM alone is the baseline that Assumption 1 says is easy to fool. These checks run next to it, as separate signals, so each can be measured on its own. They are stage 4 of the [real-time budget](realtime_budget.md) (0.5 s).

| Signal | Question | Catches | Where |
|---|---|---|---|
| `challenge` | Did the server issue this capture, and is the returned code right, single-use and fresh? | Photos made in advance, replayed uploads | `bord/challenge.py` |
| `stamp` | Does the code read back out of each photo's pixels match that capture's challenge? | An earlier in-app capture uploaded again (it carries its old code), an upload that skipped the capture page | `bord/stamp.py`, read on upload in `bord/api.py` |
| `server_timestamps` | Was the AFTER photo received after the BEFORE one, with a plausible gap (2 s to 1 h), on the server clock? | Out-of-order or instant pairs | `bord/challenge.py` (`check_order`) |
| `reuse` | Is the AFTER photo within 3 of 64 dhash bits of a photo the user already submitted? | Reused photos (also re-encoded or resized copies) | `bord/imaging.py`, `bord/anticheat.py` |
| `scene_consistency` | Do BEFORE and AFTER show the same scene (blurred luminance correlation plus colour histogram overlap)? | Wrong scene | `bord/imaging.py`, `bord/anticheat.py` |
| `ai_image` | Does a pixel classifier think the AFTER photo is AI-made? Weak, see Assumption 4. | AI-edited photos, partly | `bord/detectors.py` |
| `file_metadata` | Does the file carry camera make, model and capture time? **Diagnostic only**: browser frames carry no EXIF (Assumption 3), so it cannot be used in the app. | AI exports in the benchmark | `bord/detectors.py` |

A signal is flagged `true` or `false`, or `null` when it does not apply (no challenge on an offline benchmark row, no detector installed). A check that raises is flagged: any failure ends as a hold. A diagnostic signal is reported but never flags a submission.

## What a flag does

In Checkpoint 3 a flag can only turn a release into a hold (`bord/realtime.py`). The wager stays authorized and the submission goes to manual review. A forfeit still stands. Checkpoint 5 decides what each signal is worth when it calibrates the two thresholds.

## Why the browser flow looks the way it does

- There is no file upload in the app. `frontend/src/CameraCapture.jsx` opens `getUserMedia`, draws the challenge code into the pixels of the captured frame (as text, and as a strip of black and white cells the server can read), and posts it at once to `/api/captures`.
- The server redeems the challenge when the photo arrives, so `received_at` is the capture time as far as the server can tell. It never reads a client clock or filename.
- The code is checked against the challenge store, and on upload the server also reads it back out of the photo's cell strip (`bord/stamp.py`, about 10 ms for a 1920 px frame, off the verdict's critical path). An earlier capture uploaded again carries its old code, and a photo that skipped the capture page carries none; either flags the `stamp` signal.
- What the stamp cannot stop: a modified client can draw any code on any image, and an old photo shown on a screen and captured through the app gets a fresh, valid stamp. Those attacks are the `recaptured` category in the benchmark, and they are left to the reuse and scene checks and the verdict model.

## Measuring the signals

Without an LLM (seconds, not minutes):

```bash
cd backend
python -m bord.anticheat_eval --manifest ../data/manifest.csv --out ../results
python -m bord.anticheat_eval --manifest ../data/manifest.csv --out ../results --detector hf   # needs requirements-detector.txt
```

This writes `results/anticheat_<time>/summary.json` with, per signal and per attack category, how many honest submissions were flagged (false alarms) and how many attacks (detections), each with a 95% Wilson interval, plus Assumption 4 and the score distributions the thresholds should come from (`calibration`). The scene threshold (0.45) and the reuse distance (3) are starting points, not validated values.

With the LLM, on the same manifest as Checkpoint 2:

```bash
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --llm-timeout 60 --deadline 70               # baseline again
python -m bord.evaluate --manifest ../data/manifest.csv --out ../results --llm-timeout 60 --deadline 70 --anticheat   # with the layer
python -m bord.compare_runs ../results/*/                                                                             # fooled / released per category
```

## Still to do by hand

- **Web-sourced attacks.** Put 5 to 15 clean-room images in `data/web/` (for example from the Kaggle Messy vs Clean Room set; check the licence first) and run `python -m bord.build_manifest --data ../data`. The builder already pairs them with the honest `before` photos.
- **Staged-partial attacks.** Clean only the photographed corner and save it as `data/adversarial/<pair_id>-staged_partial.jpeg`; the builder picks it up by name.
- **Device test (Assumption 3).** Serve the app over https or localhost, open `/?device-test` in mobile Chrome and Safari, capture one frame on each and commit `data/device_tests.jsonl`.
- **Latency (Assumption 2).** `python -m bord.latency_sweep --manifest ../data/manifest.csv --out ../results` (8B and 4B, 1568 px and 768 px); add `qwen3-vl-plus@768@dashscope` for the hosted model.
