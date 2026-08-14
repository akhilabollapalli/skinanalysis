"""Boundary contracts for the skin-analysis pipeline.

This module is the single source of truth for what crosses a module boundary and,
critically, for what is allowed to reach the user.

The product rule (see CLAUDE.md, Rule 3): the application-facing payload contains only
concern name, ordinal severity, and affected regions. Raw metrics, z-scores,
probabilities, counts, areas, and confidence stay internal — they are logged for
validation and never rendered.

`ScanResult.to_public()` is the enforcement point. If a numeric value would reach the
user, it is a bug in that method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Ordinal severity. The only concern-level value a user may see."""

    NOT_DETECTED = "not_detected"
    MILD = "mild"
    MODERATE = "moderate"
    HIGH = "high"

    #: Region was present but could not be assessed (shadow, blur, occlusion,
    #: insufficient pixel density). Distinct from NOT_DETECTED, which is a real finding.
    UNMEASURABLE = "unmeasurable"

    #: Concern is feature-flagged off pending commercially clear training data.
    DISABLED = "disabled"


class Concern(str, Enum):
    """The seven cosmetic concerns. Hydration is deliberately absent and must stay absent."""

    REDNESS = "redness"
    DARK_SPOTS = "dark_spots"
    TEXTURE = "texture"
    WRINKLES = "wrinkles"
    FINE_LINES = "fine_lines"
    ACNE = "acne"
    PORES = "pores"


class ROI(str, Enum):
    """Anatomical regions. Polygon definitions live in config/rois.yaml."""

    FOREHEAD = "forehead"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    NOSE = "nose"
    LEFT_CROWS_FEET = "left_crows_feet"
    RIGHT_CROWS_FEET = "right_crows_feet"
    LEFT_UNDER_EYE = "left_under_eye"
    RIGHT_UNDER_EYE = "right_under_eye"
    LEFT_NASOLABIAL = "left_nasolabial"
    RIGHT_NASOLABIAL = "right_nasolabial"
    CHIN = "chin"


class QCFailure(str, Enum):
    """Reasons a capture is rejected. The pipeline fails closed: any failure -> RETAKE."""

    BLUR = "blur"
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    FACE_TOO_SMALL = "face_too_small"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    EXTREME_POSE = "extreme_pose"
    SHADOW_ASYMMETRY = "shadow_asymmetry"
    OCCLUSION = "occlusion"
    INSUFFICIENT_ROI_VISIBILITY = "insufficient_roi_visibility"
    SUSPECTED_FILTER_OR_MAKEUP = "suspected_filter_or_makeup"


@dataclass(frozen=True)
class CaptureQC:
    """Result of the capture quality gate.

    If ``passed`` is False the pipeline returns immediately and runs no concern logic.
    Analyzing a bad capture produces a confident-looking wrong answer, which is worse
    than asking for a retake.
    """

    passed: bool
    failures: list[QCFailure] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # internal only

    def to_public(self) -> dict[str, Any]:
        return {"pass": self.passed, "reasons": [f.value for f in self.failures]}


@dataclass(frozen=True)
class FeatureResult:
    """Output of a single concern module.

    ``raw`` and ``confidence_internal`` are validation signals. They are logged and
    used for suppression/re-capture decisions, and they never leave the backend.
    """

    concern: Concern
    severity: Severity
    regions: list[ROI] = field(default_factory=list)
    raw: dict[str, float] = field(default_factory=dict)  # internal only
    confidence_internal: float = 0.0  # internal only
    notes: list[str] = field(default_factory=list)  # internal only

    @property
    def detected(self) -> bool:
        return self.severity in (Severity.MILD, Severity.MODERATE, Severity.HIGH)

    def to_public(self) -> dict[str, Any]:
        """Strip everything numeric. This is the enforcement point for Rule 3."""
        return {
            "status": self.severity.value,
            "regions": [r.value for r in self.regions],
        }


@dataclass(frozen=True)
class ScanResult:
    """Top-level pipeline output."""

    qc: CaptureQC
    concerns: dict[Concern, FeatureResult] = field(default_factory=dict)
    schema_version: str = "1.0.0"
    module_versions: dict[str, str] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        """The only payload the application layer and recommendation engine may consume.

        Contains no z-scores, probabilities, areas, counts, densities, or confidence.
        """
        return {
            "schema_version": self.schema_version,
            "capture_quality": self.qc.to_public(),
            "concerns": {c.value: r.to_public() for c, r in self.concerns.items()},
        }


@dataclass(frozen=True)
class Recommendation:
    """Output of the rules adapter.

    The vision layer contains no product logic; recommendations are derived solely from
    concern + severity + regions.
    """

    concern: Concern
    severity: Severity
    regions: list[ROI]
    message: str
    rule_id: str
