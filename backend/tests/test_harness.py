"""Offline tests for the manifest loader and the evaluation metrics (no API calls)."""

import json
from pathlib import Path

import pytest
from PIL import Image

from bord.dataset import ManifestError, MANIFEST_COLUMNS, load_manifest
from bord.evaluate import percentile, run, summarize
from bord.verifier import VerifierResult, call_cost_usd, encode_image

HEADER = ",".join(MANIFEST_COLUMNS)


def _write_manifest(tmp_path: Path, rows: list[str]) -> Path:
    for name in ("b.jpg", "a.jpg"):
        Image.new("RGB", (64, 48), "white").save(tmp_path / name)
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


HONEST = "p01,honest,single_scene,Clean my desk,b.jpg,a.jpg,completed,,,phone,2026-10-01T10:00,2026-10-01T11:00"
HONEST_NO = "p02,honest,single_scene,Clean my desk,b.jpg,b.jpg,not_completed,,,phone,2026-10-01T10:00,2026-10-01T11:00"
ADV_REUSED = "p01-r,adversarial,single_scene,Clean my desk,b.jpg,a.jpg,not_completed,reused,p01,phone,,"
ADV_WEB = "p01-w,adversarial,single_scene,Clean my desk,b.jpg,a.jpg,not_completed,web_sourced,p01,web,,"


def test_load_manifest_valid(tmp_path):
    subs = load_manifest(_write_manifest(tmp_path, [HONEST, HONEST_NO, ADV_REUSED]))
    assert [s.pair_id for s in subs] == ["p01", "p02", "p01-r"]
    assert subs[0].expected_verdict == "completed"
    assert subs[1].expected_verdict == "not_completed"
    assert subs[2].is_adversarial and subs[2].expected_verdict == "not_completed"
    assert subs[0].before_path == tmp_path / "b.jpg"


@pytest.mark.parametrize(
    "bad_row, message",
    [
        (HONEST.replace("single_scene", "panorama"), "task_type"),
        (ADV_REUSED.replace("reused", "deepfake"), "adversarial_category"),
        (ADV_REUSED.replace(",p01,", ",,"), "base_pair_id"),
        (ADV_REUSED.replace(",p01,", ",p99,"), "not an honest pair"),
        (HONEST.replace("a.jpg", "missing.jpg"), "image not found"),
    ],
)
def test_load_manifest_rejects_bad_rows(tmp_path, bad_row, message):
    rows = [bad_row] if bad_row.startswith("p01,") else [HONEST, bad_row]
    with pytest.raises(ManifestError, match=message):
        load_manifest(_write_manifest(tmp_path, rows))


def test_duplicate_pair_id(tmp_path):
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write_manifest(tmp_path, [HONEST, HONEST]))


def test_percentile_nearest_rank():
    assert percentile([], 95) is None
    assert percentile([5.0], 95) == 5.0
    values = [float(i) for i in range(1, 21)]  # 1..20
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0


def test_cost():
    assert call_cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert call_cost_usd("claude-opus-5-5", 2000, 300) == pytest.approx(0.008 + 0.006)


def test_encode_image_downscales_and_strips_exif(tmp_path):
    p = tmp_path / "big.png"
    Image.new("RGB", (4000, 3000), "gray").save(p)
    block = encode_image(p, max_side=1000)
    assert block["source"]["media_type"] == "image/jpeg"
    import base64, io
    img = Image.open(io.BytesIO(base64.b64decode(block["source"]["data"])))
    assert max(img.size) == 1000
    assert not img.getexif()


class ScriptedVerifier:
    """Returns fixed verdicts by pair_id, mimicking the LLM baseline."""

    def __init__(self, verdicts: dict):
        self.verdicts = verdicts

    def verify(self, sub):
        verdict = self.verdicts[sub.pair_id]
        if verdict == "timeout":
            return VerifierResult(sub.pair_id, None, None, "", 10.0, model="m", error="timeout")
        return VerifierResult(
            sub.pair_id, verdict, 0.9, "", 2.0, input_tokens=2000, output_tokens=200,
            cost_usd=0.01, model="m",
        )


def test_run_and_summary(tmp_path):
    subs = load_manifest(_write_manifest(tmp_path, [HONEST, HONEST_NO, ADV_REUSED, ADV_WEB]))
    verifier = ScriptedVerifier(
        {"p01": "completed", "p02": "not_completed", "p01-r": "completed", "p01-w": "timeout"}
    )
    out = tmp_path / "run"
    summary = run(subs, verifier, out, {"run_id": "test"})

    lines = (out / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[2])["correct"] is False  # reused photo was accepted

    assert summary["honest_accuracy"] == 1.0
    assert summary["n_adversarial"] == 1  # the timed-out row is not answered
    assert summary["fooled_rate"] == 1.0
    assert summary["fooled_by_category"]["reused"]["fooled_rate"] == 1.0
    assert summary["fooled_by_category"]["web_sourced"]["n"] == 0
    assert summary["errors"] == {"timeout": 1}
    assert summary["cost_usd"]["mean_per_call"] == pytest.approx(0.01)
    assert summary["latency_s"]["p95"] == 10.0
    # n=2 is far too small for the 95% interval to clear the thresholds.
    assert summary["assumptions"]["1"]["status"] == "Holding"
    assert summary["assumptions"]["1"]["honest_accuracy"]["significant"] is False
    assert summary["honest_accuracy_ci95"][0] < 0.9
    assert summary["assumptions"]["2"]["status"] == "Broken"  # p95 10 s > 8 s
    assert json.loads((out / "summary.json").read_text())["run"]["run_id"] == "test"


def test_summary_untested_without_data():
    s = summarize([])
    assert s["assumptions"]["1"]["status"] == "Untested"
    assert s["assumptions"]["2"]["status"] == "Untested"


def test_warmup_input_is_not_a_manifest_row(tmp_path):
    from PIL import Image
    from bord.evaluate import warmup_submission

    sub = warmup_submission(tmp_path)
    assert sub.pair_id == "warmup" and sub.before_path != sub.after_path
    for p in (sub.before_path, sub.after_path):
        assert p.parent == tmp_path
        with Image.open(p) as img:
            assert img.size == (1568, 1176)  # same image-token count as a preprocessed photo


def test_cli_warms_up_on_its_own_input(tmp_path, monkeypatch):
    import bord.evaluate as evaluate

    calls = []

    class Recording:
        model = "m"

        def verify(self, sub):  # the untimed warm-up
            calls.append(("warmup", sub.pair_id))
            return VerifierResult(sub.pair_id, "completed", 0.9, "", 1.0, model="m")

        def prepare(self, sub):
            return []

        def call(self, sub, content, timeout_s):  # timed calls, through RealtimeVerifier
            calls.append(("timed", sub.pair_id))
            return VerifierResult(sub.pair_id, "completed", 0.9, "", 1.0, model="m")

    monkeypatch.setattr(evaluate, "build_verifier", lambda args: (Recording(), {"model": "m"}))
    manifest = _write_manifest(tmp_path, [HONEST])
    assert evaluate.main(["--manifest", str(manifest), "--out", str(tmp_path / "out"), "--warmup"]) == 0
    assert calls == [("warmup", "warmup"), ("timed", "p01")]



def test_resummarize_keeps_the_run_block(tmp_path):
    from bord.resummarize import resummarize

    subs = load_manifest(_write_manifest(tmp_path, [HONEST, HONEST_NO, ADV_REUSED]))
    out = tmp_path / "run"
    verdicts = {"p01": "completed", "p02": "not_completed", "p01-r": "completed"}
    run(subs, ScriptedVerifier(verdicts), out, {"run_id": "old", "host": {"cpu": "x"}})
    stale = json.loads((out / "summary.json").read_text())
    stale["assumptions"]["1"]["status"] = "holding (not significant)"  # a Checkpoint 2 word
    (out / "summary.json").write_text(json.dumps(stale))

    summary = resummarize(out)
    assert summary["run"] == {"run_id": "old", "host": {"cpu": "x"}}
    assert summary["assumptions"]["1"]["status"] == "Holding"
    assert json.loads((out / "summary.json").read_text()) == summary
