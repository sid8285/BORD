"""LLM-only baseline verifier (Checkpoint 2).

Sends the task description plus the before and after photos to a multimodal
LLM in one call and asks for a verdict and a confidence. There are no
anti-cheating signals here on purpose: this is the baseline Assumption 1 says
is easy to fool, and the Checkpoint 3 deterministic checks are measured
against it.

Each call records wall-clock latency and token usage so the harness can test
Assumption 2 (under $0.02 and under 8 s at p95 per two-image call).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic
from PIL import Image, ImageOps

from .dataset import Submission

DEFAULT_MODEL = os.environ.get("BORD_MODEL", "claude-opus-5-5")
DEFAULT_EFFORT = os.environ.get("BORD_EFFORT", "low")
# Assumption 2 budget: the verdict must return within 10 s; 8 s is the p95 target.
DEFAULT_TIMEOUT_S = float(os.environ.get("BORD_TIMEOUT_S", "10"))
# Long side in pixels. Downscaling bounds image tokens (and therefore cost) and
# re-encoding drops EXIF, so the LLM judges pixels only.
MAX_IMAGE_SIDE = 1568

# USD per million tokens (input, output), from
# https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-08).
# Only models that accept output_config.effort are listed; Haiku 4.5 rejects it.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens
}

SYSTEM_PROMPT = (
    "You verify whether a user completed a task, for an app where money is "
    "returned only if the task was really done. You receive the task, a BEFORE "
    "photo and an AFTER photo. Decide whether the AFTER photo shows the task "
    "completed in the same scene as the BEFORE photo. Answer 'completed' only "
    "if the photos support it; otherwise answer 'not_completed'. Give a "
    "confidence between 0 and 1 that your verdict is correct."
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["completed", "not_completed"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}


@dataclass
class VerifierResult:
    pair_id: str
    verdict: str | None  # None when the call failed or timed out
    confidence: float | None
    reason: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    error: str | None = None
    extra: dict = field(default_factory=dict)
    # Filled by the real-time pipeline (bord.realtime); defaults describe a bare call.
    decision: str | None = None  # release | forfeit | hold
    attempts: int = 1
    deadline_missed: bool = False
    stages: dict = field(default_factory=dict)  # stage name -> seconds

    def to_dict(self) -> dict:
        return asdict(self)


def call_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call from its token usage. Unknown models raise KeyError."""
    in_price, out_price = PRICES_PER_MTOK[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def encode_image(path: Path, max_side: int = MAX_IMAGE_SIDE) -> dict:
    """Load, orient, downscale and JPEG-encode an image as an API content block."""
    with Image.open(path) as img:
        # JPEG only: decode at a reduced DCT scale that is still >= max_side.
        # A 12 MP phone photo otherwise takes about 1 s to decode, twice the
        # 0.5 s preprocess budget in docs/realtime_budget.md.
        img.draft("RGB", (max_side, max_side))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


class LLMOnlyVerifier:
    """Baseline: one multimodal LLM call per submission, no other signals."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: anthropic.Anthropic | None = None,
    ):
        if model not in PRICES_PER_MTOK:
            raise ValueError(f"no price registered for model {model!r}; add it to PRICES_PER_MTOK")
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        # max_retries=0: a retry would hide latency the harness is meant to measure.
        self.client = client or anthropic.Anthropic(timeout=timeout_s, max_retries=0)

    def prepare(self, sub: Submission) -> list[dict]:
        """Preprocess stage: load, downscale and encode both photos."""
        return self._content(sub)

    def _content(self, sub: Submission) -> list[dict]:
        return [
            {"type": "text", "text": f"Task: {sub.task_description}\nBEFORE photo:"},
            encode_image(sub.before_path),
            {"type": "text", "text": "AFTER photo:"},
            encode_image(sub.after_path),
            {"type": "text", "text": "Was the task completed?"},
        ]

    def verify(self, sub: Submission) -> VerifierResult:
        t0 = time.perf_counter()
        content = self.prepare(sub)  # encode before starting the LLM clock
        preprocess_s = time.perf_counter() - t0
        result = self.call(sub, content, self.timeout_s)
        result.stages = {"preprocess_s": preprocess_s, "llm_s": result.latency_s}
        return result

    def call(self, sub: Submission, content: list[dict], timeout_s: float) -> VerifierResult:
        """LLM stage: one request with its own timeout. latency_s is this call only."""
        client = self.client if timeout_s == self.timeout_s else self.client.with_options(timeout=timeout_s)
        start = time.perf_counter()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
                },
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APITimeoutError:
            return VerifierResult(
                pair_id=sub.pair_id, verdict=None, confidence=None, reason="",
                latency_s=time.perf_counter() - start, model=self.model, error="timeout",
            )
        except anthropic.RateLimitError as e:
            return VerifierResult(
                pair_id=sub.pair_id, verdict=None, confidence=None, reason="",
                latency_s=time.perf_counter() - start, model=self.model,
                error=f"rate_limited: {e.message}",
            )
        except anthropic.APIStatusError as e:
            return VerifierResult(
                pair_id=sub.pair_id, verdict=None, confidence=None, reason="",
                latency_s=time.perf_counter() - start, model=self.model,
                error=f"api_status_{e.status_code}: {e.message}",
            )
        except anthropic.APIConnectionError as e:
            return VerifierResult(
                pair_id=sub.pair_id, verdict=None, confidence=None, reason="",
                latency_s=time.perf_counter() - start, model=self.model,
                error=f"connection: {e}",
            )
        latency = time.perf_counter() - start

        usage = response.usage
        cost = call_cost_usd(self.model, usage.input_tokens, usage.output_tokens)
        base = dict(
            pair_id=sub.pair_id, latency_s=latency, input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens, cost_usd=cost, model=self.model,
            extra={"request_id": response._request_id, "stop_reason": response.stop_reason},
        )

        if response.stop_reason == "refusal":
            return VerifierResult(verdict=None, confidence=None, reason="", error="refusal", **base)

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            parsed = json.loads(text)
            verdict = parsed["verdict"]
            confidence = min(1.0, max(0.0, float(parsed["confidence"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return VerifierResult(
                verdict=None, confidence=None, reason=text[:500],
                error=f"unparseable_output (stop_reason={response.stop_reason})", **base,
            )
        return VerifierResult(
            verdict=verdict, confidence=confidence, reason=str(parsed.get("reason", "")), **base
        )
