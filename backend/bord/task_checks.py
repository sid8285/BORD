"""Checks that belong to the multi-view and time-spread task types (Checkpoint 4).

The per-photo anti-cheating signals (bord.anticheat) run on each before/after pair as
before. These checks look across the photos of one task:

- multi_view: the user cannot satisfy several angles with one photo, so the AFTER
  photos must differ from each other; and the verdicts of the per-angle pairs are
  combined conservatively (``combine_decisions``).
- time_spread: every check-in must be a stamped, in-window photo of the same scene as
  the session's BEFORE photo, and no two check-ins may be the same photo.

Like every other signal, a failure here can only turn a release into a hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .anticheat import AntiCheatConfig
from .imaging import file_dhash, hamming, load_thumb, scene_similarity
from .realtime import FORFEIT, HOLD, RELEASE
from .tasks import Task


CHECKIN_MIN_BITS = 2


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    detail: str


def combine_decisions(decisions: Sequence[str]) -> str:
    """One decision for a task verified as several pairs.

    Release only if every pair is released and forfeit only if every pair is forfeited.
    Any disagreement (some angles look done, others not) or any hold goes to manual
    review: forfeiting a wager on a partial disagreement would be a false reject we
    cannot yet calibrate (Checkpoint 5).
    """
    kinds = set(decisions)
    if not decisions:
        return HOLD
    if kinds == {RELEASE}:
        return RELEASE
    if kinds == {FORFEIT}:
        return FORFEIT
    return HOLD


def photos_distinct(paths: Sequence[Path], min_bits: int = AntiCheatConfig.reuse_max_distance + 1) -> CheckResult:
    """No two photos in ``paths`` may be near-identical (dhash distance below ``min_bits``)."""
    hashes = [file_dhash(p) for p in paths]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = hamming(hashes[i], hashes[j])
            if d < min_bits:
                return CheckResult(False, f"photos {i + 1} and {j + 1} differ by only {d}/64 bits")
    return CheckResult(True, f"{len(paths)} photos, all different")


def check_multi_view(after_paths: Sequence[Path]) -> CheckResult:
    if len(after_paths) < 2:
        return CheckResult(False, "a multi-view task needs at least two views")
    result = photos_distinct(after_paths)
    return CheckResult(result.ok, "views: " + result.detail)


def check_time_spread(task: Task, before_path: Path, config: AntiCheatConfig = AntiCheatConfig()) -> CheckResult:
    """All scheduled check-ins were photographed in their windows and show the session's scene.

    Called once the session is over: a missing check-in is a miss, not a pending one.
    """
    missing = [i + 1 for i in range(len(task.checkin_offsets)) if i not in task.checkins]
    if missing:
        return CheckResult(False, f"check-in(s) {missing} were never photographed in their window")
    before = load_thumb(before_path)
    paths = [Path(task.checkins[i]["path"]) for i in sorted(task.checkins)]
    weak = []
    for i, p in zip(sorted(task.checkins), paths):
        score = scene_similarity(before, load_thumb(p))["score"]
        if score < config.scene_min_score:
            weak.append(f"check-in {i + 1} (similarity {score:.2f})")
    if weak:
        return CheckResult(False, "check-in photos do not show the session's scene: " + ", ".join(weak))
    # Check-ins of one desk legitimately look alike, so only a (near-)exact copy counts as the
    # same photo here (<= 1 bit); a replayed in-app capture is already caught by its stamp.
    same = photos_distinct(paths, min_bits=CHECKIN_MIN_BITS)
    if not same.ok:
        return CheckResult(False, "check-ins: " + same.detail)
    return CheckResult(True, f"{len(paths)} check-ins in their windows, same scene, all different")
