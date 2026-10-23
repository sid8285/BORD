"""Anti-cheating stage (Checkpoint 3): deterministic signals next to the LLM.

Assumption 1 says an LLM used alone accepts a large share of cheating attempts,
so the checks that do not need a language model run as separate signals and are
measured separately (docs/realtime_budget.md stage 4, budget 0.5 s):

=================  ==================================================  ============
signal             question                                            catches
=================  ==================================================  ============
challenge          did the server issue this capture, and is the       pre-made photos,
                   returned code right, single use, fresh?             replayed uploads
server_timestamps  did the AFTER photo arrive later than the BEFORE    out-of-order or
                   one, by a plausible gap, on the server clock?       instant pairs
reuse              is the AFTER photo (near-)identical to one the      reused photos
                   user submitted before? (64-bit dhash distance)
scene_consistency  do BEFORE and AFTER show the same scene?            wrong scene
ai_image           does a pixel classifier think the AFTER photo is    AI-edited photos
                   AI-made? (weak, Assumption 4; needs a detector)
file_metadata      does the file carry camera EXIF? (diagnostic only:  AI exports in
                   browser frames never do)                            the benchmark
=================  ==================================================  ============

A signal is ``flagged`` True / False, or None when it does not apply (no
challenge in an offline benchmark row, no detector installed). A check that
raises is flagged: any failure ends as a hold. Diagnostic signals are reported
but never flag a submission. In the pipeline a flag can only turn a release
into a hold; CP5 decides what each signal is worth (bord.realtime).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .challenge import Redemption, check_order
from .dataset import Submission
from .detectors import ImageDetector, file_metadata, looks_camera_made
from .imaging import file_dhash, hamming, load_thumb, scene_similarity


@dataclass(frozen=True)
class AntiCheatConfig:
    # A re-encoded or resized copy of one photo differs in 0-2 of 64 bits; two different
    # photos of the same desk differ in more. Calibrate with bord.anticheat_eval.
    reuse_max_distance: int = 3
    # Unvalidated starting point; bord.anticheat_eval prints the honest and wrong-scene
    # score distributions the threshold should be chosen from.
    scene_min_score: float = 0.45
    ai_min_score: float = 0.5
    min_gap_s: float = 2.0
    max_gap_s: float = 3600.0
    use_metadata: bool = True  # report the diagnostic file_metadata signal


@dataclass(frozen=True)
class ChallengeEvidence:
    """What the API collected for one submission: the redemption of each capture's challenge."""

    before: Redemption
    after: Redemption


@dataclass
class Signal:
    name: str
    flagged: bool | None
    score: float | None
    detail: str
    diagnostic: bool = False


@dataclass
class AntiCheatReport:
    signals: list[Signal] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def flagged(self) -> bool:
        return any(s.flagged for s in self.signals if not s.diagnostic)

    @property
    def flagged_by(self) -> list[str]:
        return [s.name for s in self.signals if s.flagged and not s.diagnostic]

    def signal(self, name: str) -> Signal | None:
        return next((s for s in self.signals if s.name == name), None)

    def to_dict(self) -> dict:
        return {"flagged": self.flagged, "flagged_by": self.flagged_by,
                "elapsed_s": self.elapsed_s, "signals": [asdict(s) for s in self.signals]}


class AntiCheat:
    def __init__(self, config: AntiCheatConfig = AntiCheatConfig(), detector: ImageDetector | None = None):
        self.config = config
        self.detector = detector

    def check(
        self,
        sub: Submission,
        history: Sequence[Path] = (),
        evidence: ChallengeEvidence | None = None,
    ) -> AntiCheatReport:
        """Run every signal. ``history`` are the user's past AFTER photos, oldest first."""
        t0 = time.perf_counter()
        checks = [
            lambda: self._challenge(evidence),
            lambda: self._timestamps(evidence),
            lambda: self._reuse(sub, history),
            lambda: self._scene(sub),
            lambda: self._ai_image(sub),
        ]
        names = ["challenge", "server_timestamps", "reuse", "scene_consistency", "ai_image"]
        if self.config.use_metadata:
            checks.append(lambda: self._metadata(sub))
            names.append("file_metadata")

        signals = []
        for name, check in zip(names, checks):
            try:
                signals.append(check())
            except (OSError, ValueError, RuntimeError) as e:
                # Fail closed: a check that cannot run must not let a release through.
                signals.append(Signal(name, True, None, f"check failed: {type(e).__name__}: {e}",
                                      diagnostic=(name == "file_metadata")))
        return AntiCheatReport(signals, time.perf_counter() - t0)

    # --- signals -----------------------------------------------------------------

    def _challenge(self, evidence: ChallengeEvidence | None) -> Signal:
        if evidence is None:
            return Signal("challenge", None, None, "no challenge (offline benchmark row)")
        bad = [f"{r.reason} ({role})" for role, r in (("before", evidence.before), ("after", evidence.after)) if not r.ok]
        return Signal("challenge", bool(bad), None, "; ".join(bad) or "both captures matched their challenges")

    def _timestamps(self, evidence: ChallengeEvidence | None) -> Signal:
        if evidence is None:
            return Signal("server_timestamps", None, None, "no server timestamps (offline benchmark row)")
        ok, why = check_order(evidence.before, evidence.after, self.config.min_gap_s, self.config.max_gap_s)
        gap = evidence.after.received_at - evidence.before.received_at
        return Signal("server_timestamps", not ok, gap, why)

    def _reuse(self, sub: Submission, history: Sequence[Path]) -> Signal:
        if not history:
            return Signal("reuse", False, None, "no past submissions to compare with")
        after = file_dhash(sub.after_path)
        best = min(hamming(after, file_dhash(p)) for p in history)
        return Signal("reuse", best <= self.config.reuse_max_distance, float(best),
                      f"closest of {len(history)} past photos differs by {best}/64 bits "
                      f"(flag at <= {self.config.reuse_max_distance})")

    def _scene(self, sub: Submission) -> Signal:
        sim = scene_similarity(load_thumb(sub.before_path), load_thumb(sub.after_path))
        return Signal("scene_consistency", sim["score"] < self.config.scene_min_score, sim["score"],
                      f"similarity {sim['score']:.2f} (luminance correlation {sim['correlation']:.2f}, "
                      f"colour overlap {sim['histogram']:.2f}; flag below {self.config.scene_min_score})")

    def _ai_image(self, sub: Submission) -> Signal:
        if self.detector is None:
            return Signal("ai_image", None, None, "no pixel detector configured")
        p = self.detector.score(sub.after_path)
        return Signal("ai_image", p >= self.config.ai_min_score, p,
                      f"{self.detector.name}: P(AI) = {p:.2f} (flag at >= {self.config.ai_min_score})")

    def _metadata(self, sub: Submission) -> Signal:
        if sub.capture_source == "browser":
            return Signal("file_metadata", None, None, "browser frames carry no EXIF", diagnostic=True)
        meta = file_metadata(sub.after_path)
        made = looks_camera_made(meta)
        fields = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in meta.items())
        return Signal("file_metadata", not made, None, fields, diagnostic=True)


class ManifestHistory:
    """Past AFTER photos for the offline benchmark: a stand-in for a user's submission log.

    For a submission built on honest pair B (an honest row is its own base), the history is
    the AFTER photo of every honest pair captured before B, oldest first. That is what the
    user "had already submitted" when B happened, and it is where a ``reused`` attack gets
    its photo from. Pairs are ordered the way bord.build_manifest orders them: by
    ``before_captured_at`` (ISO text sorts correctly), then pair id.
    """

    def __init__(self, submissions: Sequence[Submission]):
        honest = [s for s in submissions if not s.is_adversarial]
        self._key = {s.pair_id: (s.before_captured_at, s.pair_id) for s in honest}
        self._honest = sorted(honest, key=lambda s: self._key[s.pair_id])

    def __call__(self, sub: Submission) -> list[Path]:
        key = self._key.get(sub.base_pair_id or sub.pair_id)
        if key is None:
            return []
        return [s.after_path for s in self._honest if self._key[s.pair_id] < key]
