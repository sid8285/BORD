"""Qwen VL verifier over any OpenAI-compatible chat API.

The same request works against three places Qwen3-VL can run, chosen by a
provider preset:

- ``ollama``: on this machine via Ollama (``ollama pull qwen3-vl:8b-instruct``).
  No per-call cost; latency depends on the hardware.
- ``dashscope``: Alibaba Cloud Model Studio's hosted API (``DASHSCOPE_API_KEY``).
- ``openai-compatible``: any other server (vLLM, SGLang, LM Studio) given
  ``--base-url``.

It exposes the same ``prepare`` / ``call`` / ``verify`` interface and
``VerifierResult`` as the Claude baseline, so the real-time pipeline, the
harness and the summary work unchanged. Thinking stays off: the instruct
variant is used, and a thinking trace would not fit the 6.5 s LLM budget.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai

from .dataset import Submission
from .verifier import (
    DEFAULT_TIMEOUT_S,
    MAX_IMAGE_SIDE,
    SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    VerifierResult,
    encode_jpeg_b64,
)

JSON_INSTRUCTION = (
    " Respond with only a JSON object of the form "
    '{"verdict": "completed" or "not_completed", "confidence": a number from 0 to 1, '
    '"reason": "one short sentence"}.'
)


@dataclass(frozen=True)
class Provider:
    base_url: str
    default_model: str
    api_key_env: str | None  # None: the server needs no key
    # USD per million tokens (input, output). Local inference has no per-call price.
    prices_per_mtok: dict[str, tuple[float, float]] = field(default_factory=dict)
    json_mode: str = "schema"  # schema | object | none
    extra_body: dict = field(default_factory=dict)


PROVIDERS: dict[str, Provider] = {
    "ollama": Provider(
        base_url="http://localhost:11434/v1",
        # The untagged qwen3-vl:8b is the thinking variant; instruct answers directly.
        default_model="qwen3-vl:8b-instruct",
        api_key_env=None,
        json_mode="schema",
    ),
    "dashscope": Provider(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3-vl-plus",
        api_key_env="DASHSCOPE_API_KEY",
        # Third-party listing (opencode.ai, 2026-10-08); confirm on Alibaba's pricing page.
        prices_per_mtok={"qwen3-vl-plus": (0.20, 1.60)},
        json_mode="object",
        extra_body={"enable_thinking": False},
    ),
    "openai-compatible": Provider(
        base_url="http://localhost:8000/v1",
        default_model="Qwen/Qwen3-VL-8B-Instruct",
        api_key_env=None,
        json_mode="schema",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> tuple[str, float, str]:
    """Pull verdict, confidence and reason out of a model reply.

    Tolerates a stray thinking block, prose around the JSON, "not completed"
    with a space, and confidence given as a percentage. Raises ValueError when
    no usable verdict is present.
    """
    text = _THINK.sub("", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _OBJECT.search(text)
        if not match:
            raise ValueError("no JSON object in reply")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("reply is not a JSON object")
    verdict = str(parsed.get("verdict", "")).strip().lower().replace(" ", "_").replace("-", "_")
    if verdict not in ("completed", "not_completed"):
        raise ValueError(f"unknown verdict {verdict!r}")
    confidence = float(parsed["confidence"])
    if confidence >= 2.0:  # "85" means 85%; 1.0-2.0 is just over-confident and clamps to 1
        confidence /= 100.0
    return verdict, min(1.0, max(0.0, confidence)), str(parsed.get("reason", ""))


class QwenVerifier:
    """Qwen VL as the verdict model: one chat call with both photos."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        json_mode: str | None = None,
        prices_per_mtok: tuple[float, float] | None = None,
        max_image_side: int = MAX_IMAGE_SIDE,
        client: openai.OpenAI | None = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")
        self.provider_name = provider
        self.provider = PROVIDERS[provider]
        self.model = model or self.provider.default_model
        self.timeout_s = timeout_s
        self.json_mode = json_mode or self.provider.json_mode
        self.max_image_side = max_image_side
        if prices_per_mtok is not None:
            self.prices = prices_per_mtok
        else:
            self.prices = self.provider.prices_per_mtok.get(self.model, (0.0, 0.0))
        if client is None:
            key = os.environ.get(self.provider.api_key_env, "") if self.provider.api_key_env else "none"
            if not key:
                raise ValueError(f"set {self.provider.api_key_env} for the {provider} provider")
            # max_retries=0: retries belong to the real-time layer, where they are counted.
            client = openai.OpenAI(base_url=base_url or self.provider.base_url, api_key=key,
                                   timeout=timeout_s, max_retries=0)
        self.client = client

    def prepare(self, sub: Submission) -> list[dict]:
        def image(path: Path) -> dict:
            data = encode_jpeg_b64(path, self.max_image_side)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}

        return [
            {"type": "text", "text": f"Task: {sub.task_description}\nBEFORE photo:"},
            image(sub.before_path),
            {"type": "text", "text": "AFTER photo:"},
            image(sub.after_path),
            {"type": "text", "text": "Was the task completed?"},
        ]

    def _response_format(self) -> dict | None:
        if self.json_mode == "schema":
            return {"type": "json_schema",
                    "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA, "strict": True}}
        if self.json_mode == "object":
            return {"type": "json_object"}
        return None

    def verify(self, sub: Submission) -> VerifierResult:
        t0 = time.perf_counter()
        content = self.prepare(sub)
        preprocess_s = time.perf_counter() - t0
        result = self.call(sub, content, self.timeout_s)
        result.stages = {"preprocess_s": preprocess_s, "llm_s": result.latency_s}
        return result

    def call(self, sub: Submission, content: list[dict], timeout_s: float) -> VerifierResult:
        client = self.client if timeout_s == self.timeout_s else self.client.with_options(timeout=timeout_s)
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + JSON_INSTRUCTION},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=512,
        )
        fmt = self._response_format()
        if fmt:
            kwargs["response_format"] = fmt
        if self.provider.extra_body:
            kwargs["extra_body"] = self.provider.extra_body

        def failed(error: str) -> VerifierResult:
            return VerifierResult(pair_id=sub.pair_id, verdict=None, confidence=None, reason="",
                                  latency_s=time.perf_counter() - start, model=self.model, error=error)

        start = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except openai.APITimeoutError:
            return failed("timeout")
        except openai.RateLimitError as e:
            return failed(f"rate_limited: {e.message}")
        except openai.APIStatusError as e:
            return failed(f"api_status_{e.status_code}: {e.message}")
        except openai.APIConnectionError as e:
            return failed(f"connection: {e}")
        latency = time.perf_counter() - start

        usage = response.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        cost = (in_tok * self.prices[0] + out_tok * self.prices[1]) / 1_000_000
        choice = response.choices[0] if response.choices else None
        finish = getattr(choice, "finish_reason", None)
        base = dict(
            pair_id=sub.pair_id, latency_s=latency, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=cost, model=self.model,
            extra={"provider": self.provider_name, "finish_reason": finish,
                   "request_id": getattr(response, "id", None)},
        )
        text = (choice.message.content or "") if choice else ""
        try:
            verdict, confidence, reason = parse_verdict(text)
        except (ValueError, KeyError, TypeError) as e:
            return VerifierResult(verdict=None, confidence=None, reason=text[:500],
                                  error=f"unparseable_output (finish_reason={finish}): {e}", **base)
        return VerifierResult(verdict=verdict, confidence=confidence, reason=reason, **base)
