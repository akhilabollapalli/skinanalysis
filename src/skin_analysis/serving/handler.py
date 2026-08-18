"""The scan request handler (D10 -- V1 analysis runs server-side).

Deliberately framework-agnostic: one function, bytes in, a status code and a JSON-ready
body out. No web framework is imported, so none had to be cleared through the licensing
gate (Rule 1), and the same handler can sit behind whatever the platform team already runs.

Four outcomes, and the distinctions between them are the point:

    analysis      the capture passed QC and the product has calibrated output
    retake        the capture failed QC. No concern logic ran (fail closed).
    not_available the product is uncalibrated, so it has NO output for this scan (D2)
    rejected      the request was malformed, unsupported, or oversized

``not_available`` is not an error and not a severity. Under D2 an uncalibrated build has
nothing to say about skin, and the honest response is to say nothing -- never a default
band, never an empty concern list that a client would render as "all clear".

Rule 3 is enforced structurally rather than by review: every success body is built from
:class:`PublicScanResult`, which has no field carrying a number. There is no code path in
this module that can reach a raw metric, because no type it touches has one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..pipeline import analyze_scan_internal
from ..schemas import CalibrationRequiredError, RunMode, UnverifiedROIError
from ..util import config as cfg
from . import decode

#: Response outcomes. Clients switch on this, not on the HTTP status code, so that a
#: retake stays distinguishable from a transport failure that also returned 200 upstream.
OUTCOME_ANALYSIS = "analysis"
OUTCOME_RETAKE = "retake"
OUTCOME_NOT_AVAILABLE = "not_available"
OUTCOME_REJECTED = "rejected"


@dataclass(frozen=True)
class ScanResponse:
    """What the transport layer sends back, plus what only the server may keep.

    ``body`` is the wire payload and contains no numbers. ``log`` is internal: latency,
    QC metrics, the reason a request was rejected. Serialising ``log`` into a response
    would defeat Rule 3, so they are separate fields rather than one dict with a
    convention about which keys are safe.
    """

    status_code: int
    body: dict[str, Any]
    log: dict[str, Any] = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        return str(self.body.get("outcome", ""))


def handle_scan(
    payload: bytes,
    *,
    declared_content_type: str | None = None,
    device_metadata: dict[str, Any] | None = None,
    run_mode: RunMode = RunMode.PRODUCTION,
) -> ScanResponse:
    """Analyse one uploaded image and return a transport-ready response.

    Args:
        payload: raw request body. Treated as hostile: sniffed, bounded and measured from
            its header before any decoder allocates.
        declared_content_type: the client's claim. Logged, then ignored in favour of the
            magic number -- ``config/serving.yaml`` sets ``trust_declared_content_type``
            false and this handler has no branch that would honour it.
        device_metadata: make/model/resolution, recorded for a future capture-profile
            decision (D11). Internal; never echoed back.
        run_mode: production refuses unverified ROI polygons (D15).

    Returns:
        ScanResponse. ``body`` is safe to serialise; ``log`` is not.

    The image is held in memory for the duration of this call and is never written to
    disk. A face image is biometric data, and retaining one turns a stateless analysis
    service into a store of identifiable data with its own consent and deletion duties.
    """
    config = cfg.load("serving")
    started = time.perf_counter()
    log: dict[str, Any] = {
        "declared_content_type": declared_content_type,
        "request_bytes": len(payload),
        "device_metadata": dict(device_metadata or {}),
    }

    max_bytes = int(config["request"]["max_bytes"])
    if len(payload) > max_bytes:
        # Checked before sniffing so an oversized body is dropped without being parsed.
        return _rejected(413, "image_too_large", f"body exceeds {max_bytes} bytes", log)

    try:
        header = decode.sniff(payload)
        decode.check_limits(header, config)
        image = decode.decode(payload)
    except decode.ImageTooLargeError as error:
        return _rejected(413, "image_too_large", str(error), log)
    except decode.ImageTooSmallError as error:
        return _rejected(400, "image_too_small", str(error), log)
    except decode.UnsupportedImageError as error:
        return _rejected(415, "unsupported_image", str(error), log)

    log["format"] = header.fmt
    log["dimensions"] = [header.width, header.height]
    if declared_content_type and header.fmt not in (declared_content_type or ""):
        # Not a rejection: clients get this wrong constantly and the bytes are the
        # authority anyway. Recorded because a systematic mismatch is worth knowing about.
        log["content_type_mismatch"] = True

    try:
        result = analyze_scan_internal(
            image, run_mode=run_mode, device_metadata=device_metadata
        )
    except UnverifiedROIError as error:
        # D15: production must not measure inside polygons nobody has reviewed. This is a
        # deployment fault, not a client fault, so it is a 500 and the detail stays in the
        # log rather than going to the caller.
        log["error"] = str(error)
        return ScanResponse(
            500,
            {"outcome": OUTCOME_REJECTED, "reason": "service_misconfigured"},
            log,
        )

    log["qc"] = {
        "pass": result.qc.passed,
        "reasons": [f.value for f in result.qc.failures],
        "metrics": dict(result.qc.metrics),
        "illumination_vector": dict(result.qc.illumination_vector),
    }
    log["module_versions"] = dict(result.module_versions)
    log["protocol_version"] = result.protocol_version

    if not result.qc.passed:
        # Fail closed. No concern logic ran, so there is nothing to publish and the
        # calibration gate is not involved -- a retake carries no severity at all.
        response = ScanResponse(
            200,
            {
                "outcome": OUTCOME_RETAKE,
                "schema_version": result.schema_version,
                "capture_quality": result.qc.to_public(),
            },
            log,
        )
        return _timed(response, started, config)

    try:
        public = result.to_public()
    except CalibrationRequiredError as error:
        # D2. The ONLY handling of this exception in the codebase, and it substitutes
        # nothing: the body below has no concern field to put a default band into.
        # Catching this to supply a severity is explicitly forbidden (CLAUDE.md §6), and
        # the reason stays internal because it names calibration state the client has no
        # business branching on.
        log["calibration_block"] = str(error)
        response = ScanResponse(
            503,
            {
                "outcome": OUTCOME_NOT_AVAILABLE,
                "schema_version": result.schema_version,
                "capture_quality": result.qc.to_public(),
            },
            log,
        )
        return _timed(response, started, config)

    body = {"outcome": OUTCOME_ANALYSIS, **public.as_dict()}
    return _timed(ScanResponse(200, body, log), started, config)


def _rejected(status: int, reason: str, detail: str, log: dict[str, Any]) -> ScanResponse:
    """A request that never reached the pipeline.

    ``detail`` goes to the log, not to the caller: the specific parse failure describes
    the server's decoder, and echoing it back is free reconnaissance.
    """
    log["reject_detail"] = detail
    return ScanResponse(status, {"outcome": OUTCOME_REJECTED, "reason": reason}, log)


def _timed(response: ScanResponse, started: float, config: dict) -> ScanResponse:
    """Record elapsed time and whether it cleared the D10 budget.

    Recorded, never acted on. A slow correct answer is still correct, and degrading the
    analysis to hit a latency number would trade the thing being sold for the thing being
    measured.
    """
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.log["elapsed_ms"] = elapsed_ms
    response.log["over_p50_budget"] = elapsed_ms > float(config["latency"]["budget_p50_ms"])
    response.log["over_p95_budget"] = elapsed_ms > float(config["latency"]["budget_p95_ms"])
    return response


__all__ = [
    "OUTCOME_ANALYSIS",
    "OUTCOME_NOT_AVAILABLE",
    "OUTCOME_REJECTED",
    "OUTCOME_RETAKE",
    "ScanResponse",
    "handle_scan",
]
