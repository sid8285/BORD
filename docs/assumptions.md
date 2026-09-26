# Technology assumptions (registered at Checkpoint 1)

| # | Assumption | Basis | Status |
|---|---|---|---|
| 1 | A frontier multimodal LLM judges honest single-scene pairs with at least 90% accuracy, but used alone it accepts at least 30% of adversarial submissions. | TA feedback; to be measured in Checkpoint 2. | Holding |
| 2 | One two-image verification call costs under $0.02 and returns in under 8 s at p95. | Provider pricing page (linked when the provider is chosen in Checkpoint 2); latency log in Checkpoint 2. | Holding |
| 3 | In-browser capture via getUserMedia works on mobile Chrome and Safari, but captured frames carry no EXIF metadata. | [MDN: getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia); device test in Checkpoint 3. | Holding |
| 4 | Open-source AI-generated-image detectors catch fewer than 70% of images edited by current models. | [arXiv 2602.00192](https://arxiv.org/abs/2602.00192), [arXiv 2602.07814](https://arxiv.org/abs/2602.07814); retest in Checkpoint 3. | Holding |
| 5 | Stripe test mode supports manual-capture holds for up to 7 days. | [Stripe: place a hold on a payment method](https://docs.stripe.com/payments/place-a-hold-on-a-payment-method); test in Checkpoint 4. | Holding |

## Platforms
- Backend: [FastAPI](https://fastapi.tiangolo.com/)
- Frontend: [React](https://react.dev/)
- Payments: [Stripe test mode](https://docs.stripe.com/testing)
- LLM: frontier multimodal API, provider chosen in Checkpoint 2
- Image libraries: [imagehash](https://github.com/JohannesBuchner/imagehash), [OpenCV](https://opencv.org/)
