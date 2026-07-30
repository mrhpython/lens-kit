"""lens-kit review surface — human review of gate artifacts.

Renders three lens-kit artifact types into one self-contained, embeddable
HTML page with a tiny stdlib HTTP server and feedback.json auto-save:

  * eval results       (per-example records: per-lens verdict chips,
                         gold-vs-got, mismatch strings; accuracy/catch/FP
                         summary + variance envelope if present)
  * calibration battery (per-fixture verdict vs acceptable/target)
  * mutation control   (planted-defect marker + caught/missed)

Adapted from Anthropic's skill-creator eval-viewer (Apache-2.0); see
LICENSE.Apache-2.0 and the NOTICE block in this package. The HTTP /
feedback-roundtrip / previous-feedback-diff mechanics are kept; the
skill-creator run-directory model (outputs/, transcript.md, grading.json)
is replaced by lens-kit's flat JSON artifacts.

Public entry: ``run_review`` (called from the ``lens-kit review`` CLI).
"""
from .parser import (
    ArtifactError,
    build_records,
    detect_artifact_type,
    load_artifacts,
)
from .viewer import generate_html, load_previous_feedback
from .server import run_review

__all__ = [
    "ArtifactError",
    "build_records",
    "detect_artifact_type",
    "generate_html",
    "load_artifacts",
    "load_previous_feedback",
    "run_review",
]
