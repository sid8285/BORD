"""Summaries of the anti-cheating signals, and the offline benchmark run end to end."""

import json

import pytest
from imgutil import changed, jpeg_bytes, smooth_scene

from bord.anticheat_eval import calibration, main as anticheat_eval_main
from bord.dataset import MANIFEST_COLUMNS
from bord.evaluate import _assumption_4, summarize, summarize_anticheat
from bord.realtime import HOLD, RELEASE


def rec(set_, category, flags, scores=None, decision=RELEASE, verdict="completed", label="completed"):
    """A record with scripted signal flags, e.g. flags={"reuse": True, "ai_image": None}."""
    scores = scores or {}
    signals = [{"name": n, "flagged": f, "score": scores.get(n), "detail": "", "diagnostic": False}
               for n, f in flags.items()]
    flagged = any(f for f in flags.values())
    return {"set": set_, "label": label, "adversarial_category": category, "verdict": verdict,
            "decision": HOLD if flagged and decision == RELEASE else decision,
            "latency_s": 3.0, "stages": {"preprocess_s": 0.1, "anticheat_s": 0.2, "llm_s": 2.7},
            "error": None, "input_tokens": 100, "output_tokens": 10, "cost_usd": 0.0,
            "deadline_missed": False, "attempts": 1,
            "correct": verdict == ("not_completed" if set_ == "adversarial" else label),
            "anticheat": {"flagged": flagged, "flagged_by": [n for n, f in flags.items() if f],
                          "elapsed_s": 0.2, "signals": signals}}


def test_summarize_anticheat_counts_flags_per_category():
    records = (
        [rec("honest", None, {"reuse": False, "ai_image": None}) for _ in range(10)]
        + [rec("honest", None, {"reuse": True, "ai_image": None})]
        + [rec("adversarial", "reused", {"reuse": True, "ai_image": None}) for _ in range(4)]
        + [rec("adversarial", "ai_edited", {"reuse": False, "ai_image": True}) for _ in range(2)]
        + [rec("adversarial", "ai_edited", {"reuse": False, "ai_image": False}) for _ in range(3)]
    )
    a = summarize_anticheat(records)
    assert a["n"] == 20
    assert a["overall"]["honest"]["k"] == 1 and a["overall"]["honest"]["n"] == 11
    assert a["signals"]["reuse"]["by_category"]["reused"]["k"] == 4
    assert a["signals"]["reuse"]["by_category"]["ai_edited"]["k"] == 0
    # No detector: every ai_image flag is None, so the signal has no applicable honest rows.
    assert a["signals"]["ai_image"]["honest"]["n"] == 0 and a["signals"]["ai_image"]["honest"]["rate"] is None
    assert a["signals"]["ai_image"]["by_category"]["ai_edited"] == {
        "k": 2, "n": 5, "rate": pytest.approx(0.4), "ci95": a["signals"]["ai_image"]["by_category"]["ai_edited"]["ci95"]}
    assert a["stage_s"]["p95"] == pytest.approx(0.2) and a["over_budget"] == 0
    assert summarize_anticheat([{"set": "honest"}]) is None


def test_assumption_4_is_about_detection_below_70_percent():
    def records(caught, total):
        return ([rec("adversarial", "ai_edited", {"ai_image": True}) for _ in range(caught)]
                + [rec("adversarial", "ai_edited", {"ai_image": False}) for _ in range(total - caught)])

    # 2 of 5 caught: under 70%, but 5 attacks cannot rule out 70% at 95% confidence
    # (Wilson upper bound 0.77).
    few = _assumption_4(summarize_anticheat(records(2, 5)))
    assert few["status"] == "Holding" and few["significant"] is False
    many = _assumption_4(summarize_anticheat(records(2, 40)))
    assert many["status"] == "Holding" and many["significant"] is True
    assert _assumption_4(summarize_anticheat(records(38, 40)))["status"] == "Broken"
    # 4 of 5 caught (80%) misses the threshold, but the interval (0.38-0.96) still
    # includes rates under 70%: Weakened, not Broken.
    assert _assumption_4(summarize_anticheat(records(4, 5)))["status"] == "Weakened"
    no_detector = [rec("adversarial", "ai_edited", {"ai_image": None})]
    assert _assumption_4(summarize_anticheat(no_detector))["status"] == "Untested"


def test_summary_carries_the_new_blocks():
    records = (
        [rec("honest", None, {"reuse": False}, decision=RELEASE) for _ in range(5)]
        + [rec("adversarial", "reused", {"reuse": True}, verdict="completed") for _ in range(3)]
        + [rec("adversarial", "wrong_scene", {"reuse": False}, verdict="not_completed", decision="forfeit")]
    )
    s = summarize(records)
    # The LLM said "completed" for the three reused attacks (fooled), the flag turned each into a hold.
    assert s["fooled_by_category"]["reused"]["fooled"] == 3
    assert s["released_by_category"]["reused"]["k"] == 0 and s["released_by_category"]["reused"]["n"] == 3
    assert s["adversarial_released"]["k"] == 0
    assert s["honest_completed_released"]["k"] == 5 and s["honest_completed_released"]["n"] == 5
    assert s["anticheat"]["overall"]["adversarial"]["k"] == 3
    assert "4" in s["assumptions"]
    assert "anticheat_s" in s["stage_latency_s"]


def test_summary_without_anticheat_is_unchanged():
    bare = dict(rec("honest", None, {"reuse": False}))
    del bare["anticheat"]  # a Checkpoint 2 style record
    s = summarize([bare])
    assert s["anticheat"] is None and "4" not in s["assumptions"]
    assert s["honest_completed_released"]["n"] == 1


def test_calibration_suggests_thresholds_between_the_groups():
    records = (
        [rec("honest", None, {"scene_consistency": False, "reuse": False}, scores={"scene_consistency": 0.9, "reuse": 20.0}),
         rec("honest", None, {"scene_consistency": False, "reuse": False}, scores={"scene_consistency": 0.8, "reuse": 14.0})]
        + [rec("adversarial", "wrong_scene", {"scene_consistency": True}, scores={"scene_consistency": 0.3})]
        + [rec("adversarial", "reused", {"reuse": True}, scores={"reuse": 1.0})]
    )
    c = calibration(records)
    assert c["scene_consistency"]["separable"] is True
    assert c["scene_consistency"]["suggested_scene_min"] == pytest.approx(0.55)
    assert c["reuse_distance"]["separable"] is True and c["reuse_distance"]["suggested_reuse_max_distance"] == 1
    overlap = calibration(records + [rec("adversarial", "wrong_scene", {"scene_consistency": True},
                                         scores={"scene_consistency": 0.85})])
    assert overlap["scene_consistency"]["separable"] is False


def test_offline_benchmark_end_to_end(tmp_path):
    data = tmp_path / "data"
    (data / "pairs").mkdir(parents=True)
    rows = []
    header = ",".join(MANIFEST_COLUMNS)
    for i, hour in ((1, 10), (2, 11), (3, 12)):
        scene = smooth_scene(i)
        (data / "pairs" / f"p0{i}").mkdir()
        (data / "pairs" / f"p0{i}" / "before.jpg").write_bytes(jpeg_bytes(scene))
        (data / "pairs" / f"p0{i}" / "after.jpg").write_bytes(jpeg_bytes(changed(scene)))
        rows.append(f"p0{i},honest,single_scene,Move the cup,pairs/p0{i}/before.jpg,pairs/p0{i}/after.jpg,"
                    f"completed,,,phone,2026-10-11T{hour}:00:00,2026-10-11T{hour}:00:10")
    # p03 resubmits p01's after photo; p01's before is paired with p02's after (another scene).
    rows.append("p03-reused-p01,adversarial,single_scene,Move the cup,pairs/p03/before.jpg,pairs/p01/after.jpg,"
                "not_completed,reused,p03,phone,,")
    rows.append("p01-wrong_scene,adversarial,single_scene,Move the cup,pairs/p01/before.jpg,pairs/p02/after.jpg,"
                "not_completed,wrong_scene,p01,phone,,")
    (data / "manifest.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    out = tmp_path / "results"
    assert anticheat_eval_main(["--manifest", str(data / "manifest.csv"), "--out", str(out)]) == 0
    (run,) = out.iterdir()
    lines = [json.loads(line) for line in (run / "signals.jsonl").read_text().splitlines()]
    assert [r["pair_id"] for r in lines] == ["p01", "p02", "p03", "p03-reused-p01", "p01-wrong_scene"]

    def signal(pair_id, name):
        record = next(r for r in lines if r["pair_id"] == pair_id)
        return next(s for s in record["anticheat"]["signals"] if s["name"] == name)

    assert signal("p03-reused-p01", "reuse")["flagged"] is True and signal("p03-reused-p01", "reuse")["score"] == 0
    assert signal("p03", "reuse")["flagged"] is False and signal("p01", "reuse")["score"] is None
    assert signal("p01-wrong_scene", "scene_consistency")["score"] < signal("p01", "scene_consistency")["score"]

    summary = json.loads((run / "summary.json").read_text())
    assert summary["anticheat"]["signals"]["reuse"]["by_category"]["reused"]["k"] == 1
    assert summary["anticheat"]["signals"]["ai_image"]["honest"]["n"] == 0  # no detector configured
    assert summary["calibration"]["reuse_distance"]["reused"]["max"] == 0
