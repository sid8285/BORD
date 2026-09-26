# Related work (Checkpoint 1)

| System | How completion is verified | Gap BORD targets |
|---|---|---|
| [StickK](https://www.stickk.com/) | Self-report or a human referee | No automated verification |
| [Beeminder](https://www.beeminder.com/) | Self-report or data integrations | No photo verification |
| [Focusmate](https://www.focusmate.com/) | Live co-working with a partner | Accountability, not verification |
| [Forfeit](https://www.forfeit.app/) | Live in-app photo or video capture; AI review with human escalation for edge cases; flags edited timestamps | Closed verifier; no published accuracy, thresholds, or adversarial evaluation |

Forfeit is the closest prior work. Its site states: "AI reviews it, and edge cases go to a human reviewer," "The photo has to be live (not from your camera roll)," and "Screenshots from the photo library or with edited timestamps get flagged." (Read 2026-09-25.) An AI photo check alone is therefore not novel; BORD's contribution is measuring and hardening that check.

## Techniques
- **Perceptual hashing** to catch reused images: [imagehash](https://github.com/JohannesBuchner/imagehash).
- **Content provenance:** [C2PA Content Credentials](https://c2pa.org/).
- **AI-generated image detection** and its limits on edited images:
  - [AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange](https://arxiv.org/abs/2602.00192)
  - [How well are open sourced AI-generated image detection models out-of-the-box](https://arxiv.org/abs/2602.07814)
- **Multimodal LLMs** as before/after judges (the baseline the TA noted is easy).
