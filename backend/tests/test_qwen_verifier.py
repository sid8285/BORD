"""Offline tests for the Qwen VL verifier (no model server needed)."""

import json
from types import SimpleNamespace

import httpx2
import openai
import pytest
from PIL import Image

from bord.dataset import Submission
from bord.evaluate import main as evaluate_main
from bord.qwen_verifier import QwenVerifier, parse_verdict
from bord.realtime import HOLD, RELEASE, RealtimeVerifier


def _sub(tmp_path):
    for name in ("b.jpg", "a.jpg"):
        Image.new("RGB", (64, 48), "white").save(tmp_path / name)
    return Submission("p01", "honest", "single_scene", "Clean my desk", tmp_path / "b.jpg",
                      tmp_path / "a.jpg", "completed", None, None, "phone", "", "")


class FakeClient:
    """Stands in for openai.OpenAI: returns scripted replies or raises scripted errors."""

    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, **kwargs):
        return self

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(
            id="req-1",
            choices=[SimpleNamespace(message=SimpleNamespace(content=out), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=3000, completion_tokens=40),
        )


REQ = httpx2.Request("POST", "http://localhost:11434/v1/chat/completions")


@pytest.mark.parametrize("text, expected", [
    ('{"verdict": "completed", "confidence": 0.9, "reason": "clean"}', ("completed", 0.9, "clean")),
    ('<think>hmm</think>\n{"verdict": "not completed", "confidence": 85}', ("not_completed", 0.85, "")),
    ('Sure! ```json\n{"verdict": "Completed", "confidence": 1.4}\n```', ("completed", 1.0, "")),
])
def test_parse_verdict_tolerates_common_qwen_output(text, expected):
    assert parse_verdict(text) == expected


@pytest.mark.parametrize("text", ["no json here", '{"verdict": "maybe", "confidence": 0.5}', '{"verdict": "completed"}'])
def test_parse_verdict_rejects_unusable_output(text):
    with pytest.raises((ValueError, KeyError)):
        parse_verdict(text)


def test_request_shape_and_result(tmp_path):
    client = FakeClient(['{"verdict": "completed", "confidence": 0.8, "reason": "desk is clear"}'])
    v = QwenVerifier(provider="ollama", client=client)
    r = v.verify(_sub(tmp_path))
    req = client.requests[0]
    assert req["model"] == "qwen3-vl:8b-instruct" and req["temperature"] == 0
    assert req["response_format"]["type"] == "json_schema"
    images = [c for c in req["messages"][1]["content"] if c["type"] == "image_url"]
    assert len(images) == 2 and images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert r.verdict == "completed" and r.confidence == 0.8
    assert r.cost_usd == 0.0 and r.input_tokens == 3000  # local inference has no per-call price
    assert set(r.stages) == {"preprocess_s", "llm_s"}


def test_dashscope_preset_prices_and_disables_thinking(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    client = FakeClient(['{"verdict": "not_completed", "confidence": 0.7, "reason": ""}'])
    v = QwenVerifier(provider="dashscope", client=client)
    r = v.verify(_sub(tmp_path))
    assert client.requests[0]["extra_body"] == {"enable_thinking": False}
    assert client.requests[0]["response_format"] == {"type": "json_object"}
    assert r.cost_usd == pytest.approx((3000 * 0.20 + 40 * 1.60) / 1e6)


def test_dashscope_needs_a_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        QwenVerifier(provider="dashscope")


def test_errors_map_to_the_retry_rules(tmp_path):
    client = FakeClient([openai.APIConnectionError(request=REQ),
                         '{"verdict": "completed", "confidence": 0.9, "reason": ""}'])
    r = RealtimeVerifier(QwenVerifier(provider="ollama", client=client)).verify(_sub(tmp_path))
    assert r.attempts == 2 and r.decision == RELEASE

    # The fake times out instantly, so time is left for the one retry; both fail -> hold.
    client = FakeClient([openai.APITimeoutError(request=REQ), openai.APITimeoutError(request=REQ)])
    r = RealtimeVerifier(QwenVerifier(provider="ollama", client=client)).verify(_sub(tmp_path))
    assert r.error == "timeout" and r.attempts == 2 and r.decision == HOLD

    client = FakeClient(["I think the desk looks cleaner."])
    r = RealtimeVerifier(QwenVerifier(provider="ollama", client=client)).verify(_sub(tmp_path))
    assert r.error.startswith("unparseable_output") and r.attempts == 1 and r.decision == HOLD


def test_evaluate_cli_with_unreachable_local_server(tmp_path):
    for name in ("b.jpg", "a.jpg"):
        Image.new("RGB", (64, 48), "white").save(tmp_path / name)
    (tmp_path / "manifest.csv").write_text(
        "pair_id,set,task_type,task_description,before_path,after_path,label,adversarial_category,"
        "base_pair_id,capture_source,before_captured_at,after_captured_at\n"
        "p01,honest,single_scene,Clean my desk,b.jpg,a.jpg,completed,,,phone,,\n")
    out = tmp_path / "results"
    assert evaluate_main(["--manifest", str(tmp_path / "manifest.csv"), "--out", str(out),
                          "--base-url", "http://127.0.0.1:9/v1"]) == 0
    (run,) = out.iterdir()
    assert run.name.endswith("_qwen3-vl-8b-instruct")  # ':' removed from the folder name
    summary = json.loads((run / "summary.json").read_text())
    assert summary["run"]["provider"] == "ollama" and summary["run"]["warmup"]["error"]
    assert summary["realtime"]["holds"] == 1 and "platform" in summary["run"]["host"]
