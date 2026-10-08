# Technology assumptions

Registered at Checkpoint 1. From Checkpoint 2 on, every status cites an evaluation run in `results/`. Statuses use the harness's rule: **holding** means the point estimate and the 95% Wilson interval both clear the threshold. **Holding (not significant)** means the point estimate passes but the interval still crosses the threshold. **Broken** means the point estimate fails.

| # | Assumption | Basis | Status | Evidence |
|---|---|---|---|---|
| 1 | A multimodal vision-language model judges honest single-scene pairs with at least 90% accuracy, but used alone it accepts at least 30% of adversarial submissions. *(CP1 said "frontier multimodal LLM"; changed at CP2 when the verifier moved to open-weight Qwen3-VL 8B. See the report's Changes table.)* | Measured directly on Qwen3-VL 8B Instruct: the harness reports both halves separately, with 95% intervals. With 15 honest pairs, even 15/15 correct cannot show >= 90% at 95% confidence (that takes 35/35), so CP2 can at most show "not broken". The adversarial half can be confirmed with 15 attempts if 8 or more fool the baseline. | Untested (CP2 run pending) | `results/<run_id>/summary.json` → `assumptions.1` |
| 2 | One two-image verification call costs under $0.02 and returns in under 8 s at p95. | Cost: run locally through Ollama, a call has no per-call price, so this half holds by construction. The hosted option, Alibaba's `qwen3-vl-plus`, is listed by a third party at $0.20/$1.60 per million input/output tokens, about $0.001 per call (to confirm on Alibaba's pricing page). Latency: measured per call (`stages.llm_s`) on the machine recorded in `run.host`, after an untimed warm-up call. This is now the half that can fail. | Untested (CP2 run pending) | `results/<run_id>/summary.json` → `assumptions.2` and `run.host` |
| 3 | In-browser capture via getUserMedia works on mobile Chrome and Safari, but captured frames carry no EXIF metadata. | [MDN: getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia); device test in Checkpoint 3. | Holding (untested) | CP3 |
| 4 | Open-source AI-generated-image detectors catch fewer than 70% of images edited by current models. | [arXiv 2602.00192](https://arxiv.org/abs/2602.00192), [arXiv 2602.07814](https://arxiv.org/abs/2602.07814); retest in Checkpoint 3. | Holding (untested) | CP3 |
| 5 | Stripe test mode supports manual-capture holds for up to 7 days. | [Stripe: place a hold on a payment method](https://docs.stripe.com/payments/place-a-hold-on-a-payment-method); test in Checkpoint 4. | Holding (untested) | CP4 |

The real-time budget these timings feed into is in [realtime_budget.md](realtime_budget.md).

## Platforms
- Backend: [FastAPI](https://fastapi.tiangolo.com/)
- Frontend: [React](https://react.dev/)
- Payments: [Stripe test mode](https://docs.stripe.com/testing)
- Verdict model (chosen in Checkpoint 2): [Qwen3-VL](https://ollama.com/library/qwen3-vl) 8B Instruct, open weights, run locally with [Ollama](https://ollama.com) through its OpenAI-compatible API. The same code can target [Alibaba Cloud Model Studio](https://help.aliyun.com/en/model-studio/qwen-vl-compatible-with-openai) or a vLLM server. The [Anthropic Claude API](https://platform.claude.com/docs/en/about-claude/pricing) is kept as an optional frontier comparison.
- Image libraries: [imagehash](https://github.com/JohannesBuchner/imagehash), [OpenCV](https://opencv.org/)
