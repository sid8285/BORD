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
from PIL import Image, ImageDraw, ImageOps

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


# Checkpoint 4, staged-partial attacks: the per-region prompt. Both photos are sent with
# a labelled 3x3 grid drawn on them, and the model must account for every region of the
# BEFORE photo that the task concerns before it may answer "completed". A staged-partial
# AFTER photo either leaves a region undone or frames it out, and both now have to be
# stated region by region instead of being glossed over in one overall judgement.
REGION_GRID = 3
REGION_LABELS = [f"{'ABC'[r]}{c + 1}" for r in range(REGION_GRID) for c in range(REGION_GRID)]

REGION_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + " Both photos have a 3x3 grid drawn on them, with cells labelled A1 to C3 "
    "(rows A-C top to bottom, columns 1-3 left to right). First, list every cell of the BEFORE "
    "photo that contains something the task concerns. For each such cell, find the same part of "
    "the scene in the AFTER photo (the framing may differ) and say whether it is visible there and "
    "whether the task is done in it. Answer 'completed' only if every listed region is visible in "
    "the AFTER photo and done. A region that is out of frame, hidden or only partly done means "
    "'not_completed'."
)

REGION_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cell": {"type": "string", "enum": REGION_LABELS},
                    "visible_in_after": {"type": "boolean"},
                    "done": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["cell", "visible_in_after", "done", "note"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": ["completed", "not_completed"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["regions", "verdict", "confidence", "reason"],
    "additionalProperties": False,
}

PROMPTS = {
    "default": (SYSTEM_PROMPT, VERDICT_SCHEMA),
    "regions": (REGION_SYSTEM_PROMPT, REGION_VERDICT_SCHEMA),
}


def draw_region_grid(img: Image.Image, grid: int = REGION_GRID) -> Image.Image:
    """The photo with a labelled grid, so the model can name regions (prompt ``regions``)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    line = max(2, min(w, h) // 300)
    for i in range(1, grid):
        draw.line([(w * i // grid, 0), (w * i // grid, h)], fill=(255, 255, 0), width=line)
        draw.line([(0, h * i // grid), (w, h * i // grid)], fill=(255, 255, 0), width=line)
    size = max(12, min(w, h) // 25)
    for r in range(grid):
        for c in range(grid):
            x, y = w * c // grid + line * 2, h * r // grid + line * 2
            draw.rectangle([x, y, x + size * 2, y + size + line * 2], fill=(0, 0, 0))
            draw.text((x + line, y + line), f"{'ABC'[r]}{c + 1}", fill=(255, 255, 0), font_size=size)
    return img


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
    # Filled when the anti-cheating stage ran (bord.anticheat.AntiCheatReport.to_dict()).
    anticheat: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def call_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call from its token usage. Unknown models raise KeyError."""
    in_price, out_price = PRICES_PER_MTOK[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def encode_jpeg_b64(path: Path, max_side: int = MAX_IMAGE_SIDE, grid: bool = False) -> str:
    """Load, orient, downscale and JPEG-encode an image; returns base64 text.

    Re-encoding drops EXIF, so every provider judges pixels only.
    """
    with Image.open(path) as img:
        # JPEG only: decode at a reduced DCT scale that is still >= max_side.
        # A 12 MP phone photo otherwise takes about 1 s to decode, twice the
        # 0.5 s preprocess budget in docs/realtime_budget.md.
        img.draft("RGB", (max_side, max_side))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side))
        if grid:
            img = draw_region_grid(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def encode_image(path: Path, max_side: int = MAX_IMAGE_SIDE, grid: bool = False) -> dict:
    """An image as an Anthropic API content block."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg",
                   "data": encode_jpeg_b64(path, max_side, grid)},
    }


class LLMOnlyVerifier:
    """Baseline: one multimodal LLM call per submission, no other signals."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: anthropic.Anthropic | None = None,
        prompt: str = "default",
    ):
        if prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {sorted(PROMPTS)}")
        self.prompt = prompt
        self.system_prompt, self.schema = PROMPTS[prompt]
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
        grid = self.prompt == "regions"
        return [
            {"type": "text", "text": f"Task: {sub.task_description}\nBEFORE photo:"},
            encode_image(sub.before_path, grid=grid),
            {"type": "text", "text": "AFTER photo:"},
            encode_image(sub.after_path, grid=grid),
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
                system=self.system_prompt,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": self.schema},
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
        if isinstance(parsed, dict) and "regions" in parsed:
            base["extra"]["regions"] = parsed["regions"]
        return VerifierResult(
            verdict=verdict, confidence=confidence, reason=str(parsed.get("reason", "")), **base
        )
