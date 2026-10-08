# Real-time budget (Checkpoint 2)

BORD's verdict is a **firm real-time deadline**: within **10 s at p95** of the user tapping "submit", the system must either act on a verdict (return or forfeit the wager) or put the submission on hold. A verdict that arrives after the deadline is discarded rather than acted on. Money never moves on a late, failed, or low-confidence verdict.

## Where the 10 s goes

| # | Stage | Runs on | Budget | How it is enforced | On a miss |
|---|---|---|---|---|---|
| 1 | Capture checks (liveness, frame hash) | Browser | 1.0 s | Browser timer (CP3/CP4) | Re-prompt the capture |
| 2 | Upload two photos | Browser to server | 1.0 s | `fetch` timeout (CP4) | Retry upload once, then hold |
| 3 | Preprocess (decode, orient, downscale to 1568 px, re-encode) | Server | 0.5 s | Measured per call (`stages.preprocess_s`) | Counted against the deadline |
| 4 | Anti-cheating checks | Server | 0.5 s | Reserved for CP3, unused now | Hold |
| 5 | LLM verdict call (Qwen3-VL 8B, local) | Server to model server | 6.5 s first attempt | Request timeout (`--llm-timeout`) | One retry if time remains, else hold |
| 6 | Decision and Stripe capture/cancel | Server | 0.5 s | Reserved (`Budget.decision_s`) | Hold; the authorization stays valid for 7 days (Assumption 5) |
| | **Total** | | **10.0 s** | | |

Stages 3 to 6 are the **server budget of 8.0 s**. `backend/bord/realtime.py` enforces it and the evaluation harness measures it.

## Deadline handling (`RealtimeVerifier`)

1. Start the server clock when the submission arrives.
2. Preprocess both photos and record the time.
3. Call the LLM with a 6.5 s request timeout.
4. If the call fails fast with a retryable error (connection error, rate limit, 5xx or overload) and at least 2.0 s of the server budget remain after reserving 0.5 s for the decision, retry once with the remaining time as the timeout. A timed-out first call leaves too little time, so it goes straight to a hold. Refusals and unparseable output are not retried.
5. Decide. Return the wager if the verdict is `completed` with confidence at or above `release_min_conf`. Forfeit it if the verdict is `not_completed` with confidence at or above `forfeit_min_conf`. Otherwise hold. In CP2 both thresholds are 0, so the raw baseline is measured. CP5 calibrates them.
6. If the total server time exceeds 8.0 s, the decision becomes a hold whatever the verdict was. This makes the deadline firm.

The SDK timeout bounds each network phase, not total wall-clock time, so a slow response can overrun it slightly. Step 6 is the guarantee that a late verdict never moves money.

## Local inference and cold starts

The verdict model runs on the same machine, through Ollama. That removes network round-trips and provider queueing from stage 5. What's left is compute time on known hardware, which the run records in `summary.json` under `run.host`.

The first request after Ollama starts also loads the weights, which takes seconds and isn't representative of steady-state latency. The harness therefore makes one **untimed warm-up call** before a run and records its latency separately (`run.warmup`). In deployment, the server keeps the model loaded. A cold model counts as a failure mode, and a request that hits one ends as a hold.

## What the harness logs

`results/<run_id>/predictions.jsonl` has one line per submission with `stages` (preprocess, LLM and decision seconds), `attempts`, `deadline_missed` and `decision`. `summary.json` reports:

- `stage_latency_s`: p50, p95 and max per stage
- `latency_s`: end-to-end server time
- `realtime`: deadline misses and their rate, retries, holds and their rate, e2e p95 against the 8.0 s server deadline
- `decisions`: release, forfeit and hold counts for the honest and adversarial sets
- `adversarial_released`: how often a cheating attempt would have actually returned the wager, with a 95% interval

## Measured so far

- Preprocessing a 12 MP worst-case JPEG (random noise, 4032×3024) took about 0.41 s per photo after a JPEG reduced-scale decode was added, down from about 1.07 s. That is roughly 0.8 s for two photos, over the 0.5 s stage budget. Real photos compress better and decode faster, and the first evaluation run will show whether the budget holds. If it doesn't, the fix is to downscale in the browser before upload (CP4), which also shortens stage 2.
- The LLM stage numbers come from the first evaluation run.
