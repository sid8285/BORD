"""Evaluation manifest for the BORD verifier (Checkpoint 2).

One row per submission the verifier is asked to judge. The schema follows
docs/dataset_spec.md:

- Honest rows (``set=honest``) are the self-collected before/after pairs, each
  labeled ``completed`` or ``not_completed``.
- Adversarial rows (``set=adversarial``) are cheating attempts built on an
  honest pair (``base_pair_id``), labeled with one of the spec's categories.
  The correct verdict for every adversarial row is ``not_completed``: the wager
  must not be released for a reused, web-sourced, AI-edited, wrong-scene or
  staged-partial submission.

Image paths are relative to the manifest file's directory.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

LABELS = ("completed", "not_completed")
SETS = ("honest", "adversarial")
TASK_TYPES = ("single_scene", "multi_view", "time_spread")
ADVERSARIAL_CATEGORIES = (
    "reused",
    "web_sourced",
    "ai_edited",
    "wrong_scene",
    "staged_partial",
)
CAPTURE_SOURCES = ("phone", "browser", "web", "generated")

MANIFEST_COLUMNS = (
    "pair_id",
    "set",
    "task_type",
    "task_description",
    "before_path",
    "after_path",
    "label",
    "adversarial_category",
    "base_pair_id",
    "capture_source",
    "before_captured_at",
    "after_captured_at",
)


class ManifestError(ValueError):
    """Raised when a manifest row does not follow the dataset spec."""


@dataclass(frozen=True)
class Submission:
    pair_id: str
    set: str
    task_type: str
    task_description: str
    before_path: Path
    after_path: Path
    label: str
    adversarial_category: str | None
    base_pair_id: str | None
    capture_source: str
    before_captured_at: str
    after_captured_at: str

    @property
    def is_adversarial(self) -> bool:
        return self.set == "adversarial"

    @property
    def expected_verdict(self) -> str:
        """Verdict a trustworthy verifier must return for this submission."""
        return "not_completed" if self.is_adversarial else self.label


def _require(value: str, allowed: tuple[str, ...], field: str, row_no: int) -> str:
    if value not in allowed:
        raise ManifestError(
            f"row {row_no}: {field}={value!r} is not one of {', '.join(allowed)}"
        )
    return value


def load_manifest(manifest_path: str | Path, check_files: bool = True) -> list[Submission]:
    """Read and validate a manifest CSV. Raises ManifestError on the first bad row."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    submissions: list[Submission] = []
    seen_ids: set[str] = set()

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"manifest is missing columns: {', '.join(missing)}")

        for row_no, row in enumerate(reader, start=2):  # row 1 is the header
            row = {k: (v or "").strip() for k, v in row.items() if k is not None}
            pair_id = row["pair_id"]
            if not pair_id:
                raise ManifestError(f"row {row_no}: pair_id is empty")
            if pair_id in seen_ids:
                raise ManifestError(f"row {row_no}: duplicate pair_id {pair_id!r}")
            seen_ids.add(pair_id)

            set_name = _require(row["set"], SETS, "set", row_no)
            task_type = _require(row["task_type"], TASK_TYPES, "task_type", row_no)
            label = _require(row["label"], LABELS, "label", row_no)
            capture_source = _require(
                row["capture_source"], CAPTURE_SOURCES, "capture_source", row_no
            )

            category = row["adversarial_category"] or None
            base_pair_id = row["base_pair_id"] or None
            if set_name == "adversarial":
                _require(category or "", ADVERSARIAL_CATEGORIES, "adversarial_category", row_no)
                if not base_pair_id:
                    raise ManifestError(
                        f"row {row_no}: adversarial rows need the base_pair_id they were built from"
                    )
                if label != "not_completed":
                    raise ManifestError(
                        f"row {row_no}: adversarial rows must be labeled not_completed"
                    )
            elif category:
                raise ManifestError(
                    f"row {row_no}: honest rows must leave adversarial_category empty"
                )

            before = root / row["before_path"]
            after = root / row["after_path"]
            if check_files:
                for p in (before, after):
                    if not p.is_file():
                        raise ManifestError(f"row {row_no}: image not found: {p}")

            submissions.append(
                Submission(
                    pair_id=pair_id,
                    set=set_name,
                    task_type=task_type,
                    task_description=row["task_description"],
                    before_path=before,
                    after_path=after,
                    label=label,
                    adversarial_category=category,
                    base_pair_id=base_pair_id,
                    capture_source=capture_source,
                    before_captured_at=row["before_captured_at"],
                    after_captured_at=row["after_captured_at"],
                )
            )

    # Every adversarial row must point at an honest pair that is in the manifest.
    honest_ids = {s.pair_id for s in submissions if s.set == "honest"}
    for s in submissions:
        if s.is_adversarial and s.base_pair_id not in honest_ids:
            raise ManifestError(
                f"{s.pair_id}: base_pair_id {s.base_pair_id!r} is not an honest pair in this manifest"
            )
    return submissions
