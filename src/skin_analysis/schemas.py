"""Boundary contracts for the skin-analysis pipeline.

This module is the single source of truth for what crosses a module boundary and,
critically, for what is allowed to reach the user.

Three enforcement points live here:

* **Rule 3 / no numbers.** The application-facing payload contains only concern name,
  ordinal severity, and affected regions. Raw metrics, z-scores, probabilities, counts,
  areas, and confidence stay internal -- logged for validation, never rendered.

* **D2 / no output before calibration.** While the active protocol is uncalibrated,
  ``to_public()`` raises :class:`CalibrationRequiredError`. There is deliberately no
  ``Severity.UNCALIBRATED``: any such member is a value UI code could eventually render.
  Uncalibrated is not a finding, it is a state in which the product has no output.

* **D9 / the vision package terminates at PublicScanResult.** That type is where this
  repository stops. It knows nothing about how recommendations are implemented, and no
  internal type has a route across the boundary.

The internal result is always available via ``analyze_scan_internal``. That asymmetry is
the design: measurement is always possible, publication is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class CalibrationRequiredError(RuntimeError):
    """Raised when a public payload is requested before calibration (D2).

    Not recoverable at the call site. The correct response is to run the calibration
    phase, not to catch this and substitute a default severity.
    """


class UnverifiedROIError(RuntimeError):
    """Raised when production code tries to use an ROI whose polygon is unverified.

    ``config/rois.yaml meta.verified`` flips to true only after visual inspection across
    poses, hairstyles and facial hair. Shipping one forgotten polygon silently moves every
    measurement taken inside it, and nothing downstream would look wrong.
    """


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

    @property
    def rank(self) -> int:
        """Position on the ordinal scale, or -1 for the two non-ordinal states.

        Used by the aggregation policy (D6) to take a maximum and to demote one band.
        Comparing UNMEASURABLE or DISABLED against a real finding is meaningless, so they
        sit outside the ordering rather than at one end of it.
        """
        return _ORDINAL_RANK.get(self, -1)

    @classmethod
    def from_rank(cls, rank: int) -> Severity:
        """Inverse of :attr:`rank`, clamped to the ordinal scale."""
        clamped = max(0, min(rank, len(_ORDINAL_SCALE) - 1))
        return _ORDINAL_SCALE[clamped]


_ORDINAL_SCALE: tuple[Severity, ...] = (
    Severity.NOT_DETECTED,
    Severity.MILD,
    Severity.MODERATE,
    Severity.HIGH,
)
_ORDINAL_RANK: dict[Severity, int] = {s: i for i, s in enumerate(_ORDINAL_SCALE)}


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


class ImageCopy(str, Enum):
    """The three preprocessing derivatives (D5).

    They deliberately do not share preprocessing. Contrast enhancement helps structural
    detection but distorts the luminance and chroma relationships redness and pigmentation
    depend on, and it turns GLCM into a measurement of the enhancement itself.
    """

    #: sRGB -> Lab/D65, no gray-world, no illuminant estimation (D4). Redness, pigmentation.
    COLOR = "color"
    #: Luminance with fixed normalization only. NO CLAHE (D5). Texture.
    TEXTURE = "texture"
    #: Optional validated enhancement, Gabor / Hessian input. Wrinkles.
    RIDGE = "ridge"


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
    COLOR_CAST = "color_cast"


class CalibrationState(str, Enum):
    """Whether the active protocol may produce user-facing output (D2)."""

    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"


class RunMode(str, Enum):
    """Production refuses to use unverified ROI polygons; development may."""

    PRODUCTION = "production"
    DEVELOPMENT = "development"


# ---------------------------------------------------------------------------- capture


@dataclass(frozen=True)
class QCVerdict:
    """The QC outcomes a feature module is allowed to act on.

    Deliberately booleans, not metrics. A feature needs to know *whether* the capture
    cleared a check, never by how much -- letting a measurement scale with a QC margin
    would make it depend on the room rather than the skin.

    ``shadow_pass`` and ``color_cast_pass`` matter because V1 corrects neither (D4): a
    concern must not report lighting as a finding.
    """

    passed: bool
    shadow_pass: bool
    color_cast_pass: bool
    exposure_pass: bool
    failures: tuple[QCFailure, ...] = ()


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
    #: Estimated illumination / colour cast. Recorded for validation and for a future
    #: validated colour-constancy step. Under D4 it must NEVER be applied to the pixels.
    illumination_vector: dict[str, float] = field(default_factory=dict)  # internal only
    device_metadata: dict[str, Any] = field(default_factory=dict)  # internal only

    def verdict(self) -> QCVerdict:
        """The sanitized view handed to feature modules via :class:`FeatureContext`."""
        failed = set(self.failures)
        return QCVerdict(
            passed=self.passed,
            shadow_pass=QCFailure.SHADOW_ASYMMETRY not in failed,
            color_cast_pass=QCFailure.COLOR_CAST not in failed,
            exposure_pass=not failed & {
                QCFailure.UNDEREXPOSED,
                QCFailure.OVEREXPOSED,
            },
            failures=tuple(self.failures),
        )

    def to_public(self) -> dict[str, Any]:
        return {"pass": self.passed, "reasons": [f.value for f in self.failures]}


@dataclass(frozen=True)
class FeatureContext:
    """Capture-derived runtime facts a feature module needs but must not look up itself.

    **Contract (locked): capture and runtime facts ONLY.** This must not become a generic
    bag for feature outputs, calibration values, thresholds, or mutable state. Static
    thresholds live in ``config``; reference statistics live in the calibration layer. The
    moment something else is added here, two modules can start disagreeing about it.

    ``anchor_px`` is the canonical facial scale measured from landmark geometry in the
    current image (inter-ocular distance), used to convert scale-relative window and
    kernel sizes into pixels (D1)::

        window_px = scale.to_px(config["local"]["window_frac_of_iod"], context.anchor_px)

    It is a per-capture value, which is exactly why it cannot live in static YAML.
    """

    anchor_px: float
    qc: QCVerdict
    capture_profile: str = "default"
    protocol_version: str = "v1"
    run_mode: RunMode = RunMode.PRODUCTION


# ---------------------------------------------------------------------------- internal


@dataclass(frozen=True)
class ROIResult:
    """One concern measured in one ROI. Never leaves the backend on its own.

    ``raw`` holds this ROI's raw measurements (the ``raw_measurements`` list in
    ``config/severity_thresholds.yaml``); ``z_ref`` holds their population-standardized
    counterparts from ``decision/standardize.py``.
    """

    roi: ROI
    severity: Severity
    raw: dict[str, float] = field(default_factory=dict)
    z_ref: dict[str, float] = field(default_factory=dict)
    unmeasurable_reason: str | None = None


@dataclass(frozen=True)
class FeatureResultInternal:
    """Output of a single concern module.

    Everything except ``severity`` and ``regions`` is a validation signal: logged, used
    for suppression and re-capture decisions, and never rendered.
    """

    concern: Concern
    severity: Severity
    regions: list[ROI] = field(default_factory=list)
    roi_results: list[ROIResult] = field(default_factory=list)  # internal only
    #: ROIs that belong to this concern but could not be assessed (D7). Preserved so
    #: validation can distinguish "no finding" from "never looked".
    unmeasurable_regions: list[ROI] = field(default_factory=list)  # internal only
    raw: dict[str, float] = field(default_factory=dict)  # internal only
    confidence_internal: float = 0.0  # internal only
    notes: list[str] = field(default_factory=list)  # internal only

    @property
    def detected(self) -> bool:
        return self.severity in (Severity.MILD, Severity.MODERATE, Severity.HIGH)

    def to_public(self) -> PublicConcern:
        """Strip everything numeric. This is the enforcement point for Rule 3."""
        return PublicConcern(
            concern=self.concern,
            severity=self.severity,
            regions=tuple(self.regions),
        )


@dataclass(frozen=True)
class ScanResultInternal:
    """Full internal pipeline output. Always produced; not always publishable.

    ``to_public()`` is the only route to a user-facing payload and it refuses to run
    before calibration (D2).
    """

    qc: CaptureQC
    concerns: dict[Concern, FeatureResultInternal] = field(default_factory=dict)
    protocol_version: str = "v1"
    calibration_state: CalibrationState = CalibrationState.UNCALIBRATED
    schema_version: str = "1.0.0"
    module_versions: dict[str, str] = field(default_factory=dict)

    def to_internal_payload(self) -> dict[str, Any]:
        """Everything, for validation logging. Never returned by a request handler."""
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "calibration_state": self.calibration_state.value,
            "capture_quality": {
                "pass": self.qc.passed,
                "reasons": [f.value for f in self.qc.failures],
                "metrics": dict(self.qc.metrics),
                "illumination_vector": dict(self.qc.illumination_vector),
            },
            "concerns": {
                concern.value: {
                    "severity": result.severity.value,
                    "regions": [r.value for r in result.regions],
                    "unmeasurable_regions": [r.value for r in result.unmeasurable_regions],
                    "raw": dict(result.raw),
                    "confidence_internal": result.confidence_internal,
                    "roi_results": [
                        {
                            "roi": rr.roi.value,
                            "severity": rr.severity.value,
                            "raw": dict(rr.raw),
                            "z_ref": dict(rr.z_ref),
                            "unmeasurable_reason": rr.unmeasurable_reason,
                        }
                        for rr in result.roi_results
                    ],
                    "notes": list(result.notes),
                }
                for concern, result in self.concerns.items()
            },
        }

    def to_public(self) -> PublicScanResult:
        """The only payload the application layer may consume.

        Raises:
            CalibrationRequiredError: if the active protocol has not been calibrated, or
                if any concern being published lacks calibrated reference statistics (D2).
        """
        # Imported here rather than at module scope: schemas.py is the leaf contract and
        # must not acquire a load-time dependency on config I/O.
        from .util import calibration

        calibration.assert_public_ready(
            protocol_version=self.protocol_version,
            concerns=[c for c, r in self.concerns.items() if r.severity != Severity.DISABLED],
        )
        return PublicScanResult(
            schema_version=self.schema_version,
            capture_quality=self.qc.to_public(),
            concerns=tuple(r.to_public() for r in self.concerns.values()),
        )


# ------------------------------------------------------------------------------ public


@dataclass(frozen=True)
class PublicConcern:
    """One concern as everything outside the vision package sees it.

    Three fields, deliberately. There is no ``raw``, no ``confidence``, no ``z_ref`` --
    not hidden, absent. The recommendation layer cannot be tuned against a measurement it
    was never handed, because no field exists to reach through.
    """

    concern: Concern
    severity: Severity
    regions: tuple[ROI, ...]


@dataclass(frozen=True)
class PublicScanResult:
    """Where this repository stops (D9).

    Built only by :meth:`ScanResultInternal.to_public`, which enforces the calibration
    gate. Constructing one directly bypasses that gate, so don't.
    """

    schema_version: str
    capture_quality: dict[str, Any]
    concerns: tuple[PublicConcern, ...]

    def by_concern(self, concern: Concern) -> PublicConcern | None:
        return next((c for c in self.concerns if c.concern is concern), None)

    def as_dict(self) -> dict[str, Any]:
        """Serialization for transport. Still contains no numbers."""
        return {
            "schema_version": self.schema_version,
            "capture_quality": self.capture_quality,
            "concerns": {
                c.concern.value: {
                    "status": c.severity.value,
                    "regions": [r.value for r in c.regions],
                }
                for c in self.concerns
            },
        }


# ---------------------------------------------------------------------- recommendations


@dataclass(frozen=True)
class Recommendation:
    """One recommendation, derived solely from concern + severity + regions."""

    concern: Concern
    severity: Severity
    regions: tuple[ROI, ...]
    message: str
    rule_id: str


@dataclass(frozen=True)
class RecommendationResult:
    """What a recommendation engine returns. Opaque to the vision package."""

    recommendations: tuple[Recommendation, ...]


@runtime_checkable
class RecommendationEngine(Protocol):
    """The boundary contract, frozen now; the implementation behind it is TBD (D9).

    A conforming engine may be a local rule engine, a REST service, an existing backend,
    or a database-driven ruleset. This repository knows nothing about which, and must
    stay that way -- the vision layer contains zero product-recommendation logic.

    Note the parameter type: :class:`PublicScanResult`. There is no overload taking
    ``ScanResultInternal``.
    """

    def recommend(self, result: PublicScanResult) -> RecommendationResult:
        ...
