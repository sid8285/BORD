# Technology assumptions

Registered at Checkpoint 1. From Checkpoint 2 on, every status cites an evaluation run in `results/`. Statuses use the harness's rule: **holding** means the point estimate and the 95% Wilson interval both clear the threshold. **Holding (not significant)** means the point estimate passes but the interval still crosses the threshold. **Broken** means the point estimate fails.

| # | Assumption | Basis | Status | Evidence |
|---|---|---|---|---|
| 1 | A frontier multimodal LLM judges honest single-scene pairs with at least 90% accuracy, but used alone it accepts at least 30% of adversarial submissions. | Measured directly: the harness reports both halves separately, with 95% intervals. With 15 honest pairs, even 15/15 correct cannot show >= 90% at 95% confidence (that takes 35/35), so CP2 can at most show "not broken". The adversarial half can be confirmed with 15 attempts if 8 or more fool the baseline. | Untested (CP2 run pending) | `results/<run_id>/summary.json` → `assumptions.1` |
| 2 | One two-image verification call costs under $0.02 and returns in under 8 s at p95. | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing): Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 per million input/output tokens (checked 2026-10-08). At 1568 px each photo is about 2,500 input tokens, about 5,000 per call (estimate). That predicts roughly $0.02 to $0.03 for Opus 5.5, about $0.01 for Sonnet 5.5 and under $0.001 for Haiku 5.5. Latency is measured per call (`stages.llm_s`). | Untested (CP2 run pending) | `results/<run_id>/summary.json` → `assumptions.2`, one run per model |
| 3 | In-browser capture via getUserMedia works on mobile Chrome and Safari, but captured frames carry no EXIF metadata. | [MDN: getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia); device test in Checkpoint 3. | Holding (untested) | CP3 |
| 4 | Open-source AI-generated-image detectors catch fewer than 70% of images edited by current models. | [arXiv 2602.00192](https://arxiv.org/abs/2602.00192), [arXiv 2602.07814](https://arxiv.org/abs/2602.07814); retest in Checkpoint 3. | Holding (untested) | CP3 |
| 5 | Stripe test mode supports manual-capture holds for up to 7 days. | [Stripe: place a hold on a payment method](https://docs.stripe.com/payments/place-a-hold-on-a-payment-method); test in Checkpoint 4. | Holding (untested) | CP4 |

The real-time budget these timings feed into is in [realtime_budget.md](realtime_budget.md).

## Platforms
- Backend: [FastAPI](https://fastapi.tiangolo.com/)
- Frontend: [React](https://react.dev/)
- Payments: [Stripe test mode](https://docs.stripe.com/testing)
- LLM: [Anthropic Claude API](https://platform.claude.com/docs/en/about-claude/models/overview) (chosen in Checkpoint 2): Opus 5.5 baseline, Sonnet 5.5 and Haiku 5.5 for comparison
- Image libraries: [imagehash](https://github.com/JohannesBuchner/imagehash), [OpenCV](https://opencv.org/)
