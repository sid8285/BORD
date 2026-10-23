"""AI-generated-image signals (Checkpoint 3, Assumption 4).

Two kinds of signal, kept apart on purpose:

``HFImageDetector`` looks at pixels only. It wraps an open-source image
classifier from Hugging Face (Assumption 4 says such detectors catch fewer than
70% of edited images, so it is a weak signal, not a verdict). The dependency is
optional and heavy (``pip install -r requirements-detector.txt``); the model is
imported lazily so the rest of the backend, and the offline tests, never need it.

``file_metadata`` reads what is *in the file* (EXIF camera fields, C2PA / JUMBF
markers). A phone photo has a camera make and model and a capture time; an
image-model export usually has neither. That separates BORD's benchmark photos
cleanly, but only because they arrive as files. Frames captured in the browser
carry no EXIF (Assumption 3), so in the app this check says nothing. The harness
therefore reports it as a *diagnostic*: it does not feed the release decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

EXIF_IFD = 0x8769
MAKE, MODEL, DATETIME_ORIGINAL = 271, 272, 36867
# JUMBF / C2PA boxes are stored as APP11 segments labelled "c2pa" in the file bytes.
C2PA_MARKERS = (b"c2pa", b"jumb")
SCAN_BYTES = 2_000_000  # the manifest sits near the start of the file


class ImageDetector(Protocol):
    name: str

    def score(self, path: Path) -> float:
        """Probability in [0, 1] that the image is AI-generated or AI-edited."""
        ...


class HFImageDetector:
    """Hugging Face ``image-classification`` model as an AI-image detector.

    ``ai_labels`` are the class names that mean "AI-made" for the chosen model.
    The default model labels its classes ``artificial`` and ``human``.
    """

    def __init__(
        self,
        model_id: str = "umm-maybe/AI-image-detector",
        ai_labels: tuple[str, ...] = ("artificial", "ai", "fake", "generated", "ai-generated"),
        pipeline=None,
    ):
        self.name = f"hf:{model_id}"
        self.ai_labels = tuple(label.lower() for label in ai_labels)
        if pipeline is None:
            try:
                from transformers import pipeline as make_pipeline
            except ImportError as e:  # pragma: no cover - depends on the optional install
                raise RuntimeError(
                    "the pixel detector needs transformers and torch: "
                    "pip install -r requirements-detector.txt"
                ) from e
            pipeline = make_pipeline("image-classification", model=model_id)
        self._pipe = pipeline

    def score(self, path: Path) -> float:
        with Image.open(path) as img:
            img.draft("RGB", (1024, 1024))
            image = img.convert("RGB")
        outputs = self._pipe(image, top_k=None)
        ai = sum(o["score"] for o in outputs if str(o["label"]).lower() in self.ai_labels)
        return min(1.0, max(0.0, float(ai)))


def file_metadata(path: Path) -> dict:
    """Camera EXIF fields and C2PA markers present in the file."""
    has_make = has_model = has_captured = False
    with Image.open(path) as img:
        exif = img.getexif()
        has_make = bool(exif.get(MAKE))
        has_model = bool(exif.get(MODEL))
        has_captured = bool(exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(306))
    with path.open("rb") as fh:
        head = fh.read(SCAN_BYTES)
    return {
        "camera_make": has_make,
        "camera_model": has_model,
        "capture_time": has_captured,
        "c2pa_marker": any(marker in head for marker in C2PA_MARKERS),
    }


def looks_camera_made(meta: dict) -> bool:
    """A phone photo names its camera and when it was taken. A C2PA marker is reported but
    not held against the photo: some phones sign genuine captures too."""
    return bool(meta["camera_make"] and meta["camera_model"] and meta["capture_time"])
