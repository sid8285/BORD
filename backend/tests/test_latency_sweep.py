"""The latency sweep's argument handling and the per-category comparison table."""

import json
from pathlib import Path

import pytest
from PIL import Image

from bord import latency_sweep
from bord.compare_runs import CATEGORY_HEADER, main as compare_main
from bord.evaluate import summarize
from bord.qwen_verifier import QwenVerifier
from bord.realtime import RELEASE


def test_parse_config():
    assert latency_sweep.parse_config("qwen3-vl:4b-instruct@768") == {
        "model": "qwen3-vl:4b-instruct", "side": 768, "provider": "ollama"}
    assert latency_sweep.parse_config("qwen3-vl-plus@768@dashscope")["provider"] == "dashscope"
    for bad in ("qwen3-vl", "qwen3-vl@big", "a@1@b@c"):
        with pytest.raises(ValueError):
            latency_sweep.parse_config(bad)


def test_evaluate_args_relax_the_timeout():
    cfg = latency_sweep.parse_config("qwen3-vl:8b-instruct@768")
    argv = latency_sweep.evaluate_args(cfg, Path("m.csv"), Path("out"), 60.0, anticheat=True, limit=5)
    assert argv[argv.index("--max-image-side") + 1] == "768"
    assert argv[argv.index("--llm-timeout") + 1] == "60.0" and argv[argv.index("--deadline") + 1] == "70.0"
    assert "--anticheat" in argv and argv[argv.index("--limit") + 1] == "5"
    assert "--anticheat" not in latency_sweep.evaluate_args(cfg, Path("m.csv"), Path("out"), 60.0, False, None)


def test_default_sweep_covers_model_size_and_image_size():
    sides = {latency_sweep.parse_config(c)["side"] for c in latency_sweep.DEFAULT_CONFIGS}
    models = {latency_sweep.parse_config(c)["model"] for c in latency_sweep.DEFAULT_CONFIGS}
    assert sides == {1568, 768} and models == {"qwen3-vl:8b-instruct", "qwen3-vl:4b-instruct"}


def test_smaller_photos_are_what_the_qwen_verifier_sends(tmp_path):
    import base64
    import io
    from types import SimpleNamespace
    from bord.dataset import Submission

    for name in ("b.jpg", "a.jpg"):
        Image.new("RGB", (3000, 2000), "gray").save(tmp_path / name)
    sub = Submission("p01", "honest", "single_scene", "Clean my desk", tmp_path / "b.jpg", tmp_path / "a.jpg",
                     "completed", None, None, "phone", "", "")
    sides = {}
    for side in (1568, 768):
        v = QwenVerifier(provider="ollama", client=SimpleNamespace(), max_image_side=side)
        url = next(c["image_url"]["url"] for c in v.prepare(sub) if c["type"] == "image_url")
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        sides[side] = max(img.size)
    assert sides == {1568: 1568, 768: 768}


def test_category_table(tmp_path, capsys):
    def rec(category, decision):
        return {"set": "adversarial", "label": "not_completed", "verdict": "completed", "decision": decision,
                "latency_s": 1.0, "stages": {"preprocess_s": 0.1, "llm_s": 0.9}, "adversarial_category": category,
                "error": None, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0,
                "deadline_missed": False, "attempts": 1, "correct": False}

    summary = summarize([rec("reused", RELEASE), rec("reused", "hold"), rec("ai_edited", RELEASE)])
    summary["run"] = {"model": "m", "max_image_side": 768, "anticheat": {"detector": None}}
    run = tmp_path / "r"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps(summary))
    compare_main([str(run)])
    out = capsys.readouterr().out
    assert "| 768 | on |" in out
    assert CATEGORY_HEADER in out
    assert "| `r` | 1/2 | n/a | 1/1 | n/a | n/a | n/a |" in out
