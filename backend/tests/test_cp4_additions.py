"""Tests for the Checkpoint 4 additions beyond the course branch: the per-region staged-partial
check and prompt, the GPS check output, the Stripe round trip (against a fake stripe module)
and the multi-view / time-spread false-hold harness."""

import base64
import io
import json
from pathlib import Path

from imgutil import changed, jpeg_bytes, smooth_scene
from PIL import Image, ImageDraw

from bord import photo_check, region_eval, stripe_roundtrip, task_eval
from bord.anticheat import AntiCheat, AntiCheatConfig
from bord.dataset import Submission
from bord.regions import RegionConfig, align, region_report
from bord.verifier import PROMPTS, REGION_LABELS, encode_jpeg_b64


def save(tmp_path: Path, name: str, img: Image.Image) -> Path:
    p = tmp_path / name
    p.write_bytes(jpeg_bytes(img))
    return p


def cluttered(seed: int) -> Image.Image:
    """A smooth scene with a busy patch of small rectangles (the 'clutter') in the lower left."""
    img = smooth_scene(seed, (320, 320))
    draw = ImageDraw.Draw(img)
    for i in range(12):
        for j in range(8):
            if (i + j) % 2:
                draw.rectangle([20 + i * 10, 200 + j * 10, 27 + i * 10, 207 + j * 10], fill=(10, 10, 10))
    return img


def tidied(img: Image.Image) -> Image.Image:
    out = img.copy()
    ImageDraw.Draw(out).rectangle([15, 195, 145, 285], fill=(180, 170, 160))
    return out


# --- regions ------------------------------------------------------------------------

def test_alignment_finds_no_shift_for_the_same_photo(tmp_path):
    from bord.regions import _load
    p = save(tmp_path, "a.jpg", cluttered(1))
    scale, dx, dy, mad = align(_load(p), _load(p))
    assert (scale, dx, dy) == (1.0, 0, 0) and mad < 1.0


def test_untouched_clutter_is_flagged_and_cleared_clutter_is_not(tmp_path):
    before = save(tmp_path, "before.jpg", cluttered(2))
    same = save(tmp_path, "same.jpg", cluttered(2))           # nothing done
    done = save(tmp_path, "done.jpg", tidied(cluttered(2)))   # clutter cleared
    left = region_report(before, same)
    cleared = region_report(before, done)
    assert left.flagged and left.busy_unchanged_frac == 1.0 and left.clutter_ratio > 0.9
    assert not cleared.flagged and cleared.clutter_ratio < left.clutter_ratio
    assert len(left.cells) == 16 and left.summary()


def test_reframed_after_photo_is_reported(tmp_path):
    img = cluttered(3)
    before = save(tmp_path, "before.jpg", img)
    zoomed = save(tmp_path, "zoomed.jpg", img.crop((0, 0, 220, 220)).resize((320, 320)))
    rep = region_report(before, zoomed, RegionConfig(max_reframe=0.2))
    assert rep.reframe > 0.2 and any("re-framed" in r for r in rep.reasons)


def test_regions_signal_is_diagnostic_by_default(tmp_path):
    before = save(tmp_path, "b.jpg", cluttered(4))
    after = save(tmp_path, "a.jpg", cluttered(4))
    sub = Submission("p1", "honest", "single_scene", "Tidy", before, after, "completed", None, None, "phone", "", "")
    report = AntiCheat(AntiCheatConfig(reuse_max_distance=-1, scene_min_score=0.0)).check(sub)
    sig = report.signal("regions")
    assert sig is not None and sig.flagged and sig.diagnostic
    assert "regions" not in report.flagged_by          # reported, never holds a release yet
    flagging = AntiCheat(AntiCheatConfig(scene_min_score=0.0, regions_flag=True)).check(sub)
    assert "regions" in flagging.flagged_by


def test_region_eval_writes_a_summary(tmp_path):
    data = tmp_path / "data"
    (data / "pairs").mkdir(parents=True)
    save(data / "pairs", "b.jpg", cluttered(5))
    save(data / "pairs", "a.jpg", tidied(cluttered(5)))
    save(data / "pairs", "s.jpg", cluttered(5))
    header = ("pair_id,set,task_type,task_description,before_path,after_path,label,adversarial_category,"
              "base_pair_id,capture_source,before_captured_at,after_captured_at\n")
    (data / "m.csv").write_text(
        header
        + "p1,honest,single_scene,Tidy,pairs/b.jpg,pairs/a.jpg,completed,,,phone,2026-10-01T10:00:00,2026-10-01T10:05:00\n"
        + "p1-staged_partial,adversarial,single_scene,Tidy,pairs/b.jpg,pairs/s.jpg,not_completed,staged_partial,p1,phone,,\n"
    )
    assert region_eval.main(["--manifest", str(data / "m.csv"), "--out", str(tmp_path / "out")]) == 0
    out = next((tmp_path / "out").iterdir())
    summary = json.loads((out / "summary.json").read_text())
    assert summary["staged_partial_caught"]["k"] == 1 and summary["honest_completed_flagged"]["k"] == 0
    assert summary["assumption_7_pixel_rule"]["enough_attacks"] is False


# --- per-region prompt ----------------------------------------------------------------

def test_region_prompt_draws_a_labelled_grid(tmp_path):
    p = save(tmp_path, "a.jpg", Image.new("RGB", (600, 600), (40, 40, 40)))
    plain = Image.open(io.BytesIO(base64.b64decode(encode_jpeg_b64(p))))
    grid = Image.open(io.BytesIO(base64.b64decode(encode_jpeg_b64(p, grid=True))))
    assert plain.getpixel((200, 100))[0] < 80           # no grid line
    assert grid.getpixel((200, 100))[0] > 200           # yellow line at one third of the width
    system, schema = PROMPTS["regions"]
    assert "A1 to C3" in system and schema["properties"]["regions"]["items"]["properties"]["cell"]["enum"] == REGION_LABELS


def test_region_prompt_is_sent_and_regions_are_kept():
    from bord.verifier import LLMOnlyVerifier

    class Block:
        type = "text"
        text = json.dumps({"regions": [{"cell": "C1", "visible_in_after": False, "done": False, "note": "out of frame"}],
                           "verdict": "not_completed", "confidence": 0.8, "reason": "C1 not shown"})

    class Usage:
        input_tokens, output_tokens = 1000, 100

    class Response:
        content, usage, stop_reason, _request_id = [Block()], Usage(), "end_turn", "req_1"

    class Messages:
        kw = None

        def create(self, **kw):
            Messages.kw = kw
            return Response()

    class Client:
        messages = Messages()

    v = LLMOnlyVerifier(client=Client(), prompt="regions")
    sub = Submission("p1", "adversarial", "single_scene", "Tidy", Path("b"), Path("a"), "not_completed",
                     "staged_partial", "p0", "phone", "", "")
    r = v.call(sub, [], v.timeout_s)
    assert Messages.kw["system"] == PROMPTS["regions"][0]
    assert r.verdict == "not_completed" and r.extra["regions"][0]["cell"] == "C1"


# --- GPS check ------------------------------------------------------------------------

def test_photo_check_reports_gps(tmp_path):
    data = tmp_path / "data"
    (data / "pairs").mkdir(parents=True)
    clean = data / "pairs" / "clean.jpg"
    clean.write_bytes(jpeg_bytes(smooth_scene(1)))
    assert photo_check.main(["--data", str(data), "--out", str(tmp_path / "out")]) == 0

    img = smooth_scene(2)
    exif = Image.Exif()
    exif.get_ifd(0x8825)[2] = (33.0, 46.0, 0.0)  # GPSLatitude
    tagged = data / "pairs" / "tagged.jpg"
    img.save(tagged, format="JPEG", exif=exif)
    if photo_check.gps_tags(tagged) == 0:
        return  # this Pillow build did not write the GPS IFD; nothing to detect
    assert photo_check.main(["--data", str(data), "--out", str(tmp_path / "out2")]) == 1
    summary = json.loads(next((tmp_path / "out2").iterdir()).joinpath("summary.json").read_text())
    assert summary["with_gps"] == ["pairs/tagged.jpg"]


# --- Stripe round trip ------------------------------------------------------------------

class FakeStripeError(Exception):
    pass


class FakeStripe:
    StripeError = FakeStripeError

    def __init__(self):
        intents = {}

        class PaymentIntent:
            @staticmethod
            def create(**kw):
                pid = f"pi_{len(intents) + 1}"
                intents[pid] = {"id": pid, "status": "requires_capture", "amount": kw["amount"], "amount_received": 0,
                                "capture_method": "manual", "livemode": False, "created": 1_000_000}
                return intents[pid]

            @staticmethod
            def cancel(pid, **kw):
                if intents[pid]["status"] != "requires_capture":
                    raise FakeStripeError("not cancelable")
                intents[pid]["status"] = "canceled"
                return intents[pid]

            @staticmethod
            def capture(pid, **kw):
                if intents[pid]["status"] != "requires_capture":
                    raise FakeStripeError("not capturable")
                intents[pid].update(status="succeeded", amount_received=intents[pid]["amount"])
                return intents[pid]

            @staticmethod
            def retrieve(pid, **kw):
                i = dict(intents[pid])
                i["latest_charge"] = {"id": "ch_" + pid, "captured": i["status"] == "succeeded",
                                      "payment_method_details": {"card": {"capture_before": 1_000_000 + 7 * 86400}}}
                return i

        self.PaymentIntent = PaymentIntent


def test_stripe_roundtrip_logs_ids_and_the_capture_window(tmp_path, monkeypatch):
    fake = FakeStripe()
    real_init = stripe_roundtrip.StripeHolds.__init__

    def init(self, key, **kw):
        real_init(self, key, stripe_module=fake, **kw)

    monkeypatch.setattr(stripe_roundtrip.StripeHolds, "__init__", init)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
    assert stripe_roundtrip.main(["--out", str(tmp_path)]) == 0
    out = next(tmp_path.iterdir())
    summary = json.loads((out / "summary.json").read_text())
    assert summary["release_path_ok"] and summary["forfeit_path_ok"] and summary["double_settle_refused"]
    assert summary["hold_ids"] == ["pi_1", "pi_2"] and summary["capture_window_days"] == [7.0, 7.0]
    assert summary["assumption_5"]["status"] == "Holding"
    events = [json.loads(line)["event"] for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert events == ["authorized", "released", "authorized", "captured", "charity_payout"]


def test_stripe_roundtrip_refuses_a_live_key(tmp_path, monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_nope")
    assert stripe_roundtrip.main(["--out", str(tmp_path)]) == 2
    assert not any(tmp_path.iterdir())


# --- multi-view / time-spread harness ------------------------------------------------------

def photo_at(path: Path, img: Image.Image, when: str) -> None:
    exif = Image.Exif()
    exif[306] = when
    img.save(path, format="JPEG", quality=95, exif=exif)


def test_task_eval_checks_only(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "tasks").mkdir(parents=True)
    rows = []
    # multi_view: two distinct angles, each cleaned
    views = []
    for i in range(2):
        b, a = data / "tasks" / f"mv_b{i}.jpg", data / "tasks" / f"mv_a{i}.jpg"
        photo_at(b, smooth_scene(20 + i), "2026:11:01 10:00:00")
        photo_at(a, changed(smooth_scene(20 + i)), "2026:11:01 10:05:00")
        views.append({"angle": f"angle {i}", "before": f"tasks/mv_b{i}.jpg", "after": f"tasks/mv_a{i}.jpg"})
    rows.append({"task_id": "mv01", "task_type": "multi_view", "description": "Tidy", "views": views})
    # time_spread: check-ins at 100 s and 400 s; the second one is taken late (outside its window)
    base = smooth_scene(30)
    photo_at(data / "tasks" / "ts_b.jpg", base, "2026:11:01 12:00:00")
    photo_at(data / "tasks" / "ts_c1.jpg", changed(base), "2026:11:01 12:01:50")
    photo_at(data / "tasks" / "ts_c2.jpg", changed(base), "2026:11:01 12:12:00")
    photo_at(data / "tasks" / "ts_a.jpg", changed(base), "2026:11:01 12:20:30")
    rows.append({"task_id": "ts01", "task_type": "time_spread", "description": "Study", "session_s": 1200,
                 "checkin_offsets": [100.0, 400.0], "before": "tasks/ts_b.jpg", "after": "tasks/ts_a.jpg",
                 "checkins": ["tasks/ts_c1.jpg", "tasks/ts_c2.jpg"]})
    (data / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert task_eval.main(["run", "--data", str(data), "--out", str(tmp_path / "out")]) == 0
    out = next((tmp_path / "out").iterdir())
    recs = {json.loads(line)["task_id"]: json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()}
    assert recs["mv01"]["decision"] == "release" and recs["mv01"]["task_check"]["ok"]
    assert recs["ts01"]["decision"] == "hold" and "[2]" in recs["ts01"]["task_check"]["detail"]
    a6 = json.loads((out / "summary.json").read_text())["assumption_6"]
    assert a6["multi_view"]["false_hold"]["k"] == 0 and a6["time_spread"]["false_hold"]["k"] == 1
    assert a6["multi_view"]["enough_tasks"] is False


def test_task_eval_plan_adds_a_time_spread_row(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    assert task_eval.main(["plan", "--data", str(data), "--task-id", "ts09", "--description", "Study",
                           "--session-s", "900", "--checkins", "3"]) == 0
    row = json.loads((data / "tasks.jsonl").read_text())
    assert row["task_type"] == "time_spread" and len(row["checkin_offsets"]) == 3
    assert "check-in 3" in capsys.readouterr().out
