"""Serving-layer tests.

Two things are defended here, and they are the two ways a serving layer breaks a product
whose whole design is about restraint:

* **Rule 3 / D2 at the wire.** No response body may carry a number, and an uncalibrated
  build must return *no output* rather than a default band. The handler is the last place
  where a well-meaning "just show something" change would land, and it is the only place
  in the codebase that catches CalibrationRequiredError.

* **Untrusted input.** Uploads are hostile. Dimensions are read from the header before
  the decoder allocates, and the client's declared Content-Type is evidence of nothing.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from skin_analysis.schemas import CalibrationRequiredError, Severity  # noqa: E402
from skin_analysis.serving import decode, handler  # noqa: E402
from skin_analysis.util import config as cfg  # noqa: E402

CONFIG = cfg.load("serving")


def encode(image: np.ndarray, suffix: str = ".jpg") -> bytes:
    ok, buffer = cv2.imencode(suffix, image)
    assert ok
    return bytes(buffer)


@pytest.fixture
def grey_jpeg() -> bytes:
    return encode(np.full((800, 640, 3), 128, dtype=np.uint8))


# ------------------------------------------------------------------- header sniffing


@pytest.mark.parametrize(
    ("suffix", "fmt"), [(".jpg", "jpeg"), (".png", "png"), (".webp", "webp")]
)
def test_dimensions_are_read_from_the_header_of_every_format(suffix: str, fmt: str) -> None:
    payload = encode(np.full((300, 500, 3), 90, dtype=np.uint8), suffix)
    header = decode.sniff(payload)
    assert (header.fmt, header.width, header.height) == (fmt, 500, 300)


def test_jpeg_dimensions_survive_a_large_exif_segment() -> None:
    """The segment chain is walked rather than scanned.

    0xFFC0 appears freely inside EXIF payloads and entropy-coded data, so a naive search
    for the start-of-frame marker finds image content and reads two bytes of it as a size.
    """
    payload = encode(np.full((300, 500, 3), 90, dtype=np.uint8))
    app1 = b"\xff\xe1" + struct.pack(">H", 1002) + (b"\xff\xc0" * 500)
    spliced = payload[:2] + app1 + payload[2:]
    header = decode.sniff(spliced)
    assert (header.width, header.height) == (500, 300)


def test_unknown_format_is_rejected_not_guessed() -> None:
    with pytest.raises(decode.UnsupportedImageError):
        decode.sniff(b"GIF89a\x00\x01\x00\x01")


def test_truncated_header_is_rejected() -> None:
    with pytest.raises(decode.UnsupportedImageError):
        decode.sniff(b"\xff\xd8\xff")


def test_well_formed_header_with_corrupt_body_is_rejected() -> None:
    payload = encode(np.full((300, 300, 3), 90, dtype=np.uint8))
    with pytest.raises(decode.UnsupportedImageError):
        decode.decode(payload[:60])


# --------------------------------------------------------------------- size limits


def test_declared_dimensions_are_capped_before_decoding() -> None:
    """The decompression-bomb guard.

    A small file declaring an enormous canvas must be rejected from its header. Checking
    after cv2.imdecode returns is checking after the allocation that would have killed
    the process, so this test forges a header rather than encoding a real large image.
    """
    header = decode.ImageHeader("png", 30000, 30000)
    assert header.pixels > int(CONFIG["request"]["max_pixels"])
    with pytest.raises(decode.ImageTooLargeError):
        decode.check_limits(header, CONFIG)


def test_tiny_images_are_rejected_early() -> None:
    with pytest.raises(decode.ImageTooSmallError):
        decode.check_limits(decode.ImageHeader("jpeg", 100, 100), CONFIG)


def test_a_format_outside_the_allow_list_is_refused() -> None:
    with pytest.raises(decode.UnsupportedImageError):
        decode.check_limits(decode.ImageHeader("tiff", 800, 800), CONFIG)


def test_oversized_body_is_dropped_without_parsing() -> None:
    oversized = b"\xff\xd8\xff" + b"\x00" * int(CONFIG["request"]["max_bytes"])
    response = handler.handle_scan(oversized)
    assert response.status_code == 413
    assert response.body == {"outcome": handler.OUTCOME_REJECTED, "reason": "image_too_large"}


# ------------------------------------------------------------------ the wire contract


def test_no_response_body_contains_a_number(grey_jpeg: bytes) -> None:
    """Rule 3, checked structurally rather than by reading the code.

    Walks every value in the body of every reachable outcome. A float or int anywhere is a
    leaked measurement, whatever it was named.
    """
    bodies = [
        handler.handle_scan(grey_jpeg).body,
        handler.handle_scan(b"GIF89a").body,
        handler.handle_scan(encode(np.zeros((300, 300, 3), dtype=np.uint8))).body,
    ]

    def numbers(node, path="") -> list[str]:
        if isinstance(node, bool):
            return []
        if isinstance(node, (int, float)):
            return [path]
        if isinstance(node, dict):
            return [p for k, v in node.items() for p in numbers(v, f"{path}.{k}")]
        if isinstance(node, (list, tuple)):
            return [p for i, v in enumerate(node) for p in numbers(v, f"{path}[{i}]")]
        return []

    for body in bodies:
        assert not numbers(body), f"numeric value in a response body: {numbers(body)}"


def test_internal_metrics_stay_in_the_log_and_out_of_the_body(grey_jpeg: bytes) -> None:
    response = handler.handle_scan(grey_jpeg)
    assert response.log["qc"]["metrics"], "QC metrics must be recorded internally"
    assert "metrics" not in str(response.body)
    assert "illumination_vector" not in str(response.body)


def test_uncalibrated_build_returns_no_output_rather_than_a_default(
    monkeypatch: pytest.MonkeyPatch, grey_jpeg: bytes
) -> None:
    """D2 at the boundary.

    A capture that clears QC on an uncalibrated build must produce ``not_available`` with
    no concern data at all. The failure this guards against is a client rendering an empty
    concern list as "all clear", which is a clinical-sounding claim the product never made.
    """
    from skin_analysis import pipeline

    class _Passing:
        schema_version = "1.0.0"
        protocol_version = "v1"
        module_versions: dict[str, str] = {}

        class qc:  # noqa: N801 - stands in for a CaptureQC
            passed = True
            failures: list = []
            metrics = {"laplacian_var": 1.0}
            illumination_vector = {"r": 1.0, "g": 1.0, "b": 1.0}

            @staticmethod
            def to_public() -> dict:
                return {"pass": True, "reasons": []}

        def to_public(self):
            raise CalibrationRequiredError("meta.calibrated is false")

    monkeypatch.setattr(pipeline, "analyze_scan_internal", lambda *a, **k: _Passing())
    monkeypatch.setattr(handler, "analyze_scan_internal", lambda *a, **k: _Passing())

    response = handler.handle_scan(grey_jpeg)
    assert response.status_code == 503
    assert response.body["outcome"] == handler.OUTCOME_NOT_AVAILABLE
    assert "concerns" not in response.body
    # The reason names calibration state and stays server-side.
    assert "calibrated" in response.log["calibration_block"]
    assert "calibrat" not in str(response.body).lower()


def test_no_severity_value_can_appear_in_a_non_analysis_body(grey_jpeg: bytes) -> None:
    """Retake and not_available must carry no band, not even not_detected."""
    response = handler.handle_scan(grey_jpeg)
    assert response.outcome in (handler.OUTCOME_RETAKE, handler.OUTCOME_NOT_AVAILABLE)
    rendered = str(response.body)
    assert not [s.value for s in Severity if s.value in rendered]


def test_failed_capture_returns_retake_with_reasons(grey_jpeg: bytes) -> None:
    """Fail closed: a flat grey frame has no face, so no concern logic may run."""
    response = handler.handle_scan(grey_jpeg)
    assert response.status_code == 200
    assert response.body["outcome"] == handler.OUTCOME_RETAKE
    assert response.body["capture_quality"]["pass"] is False
    assert response.body["capture_quality"]["reasons"]
    assert "concerns" not in response.body


def test_declared_content_type_does_not_change_the_outcome(grey_jpeg: bytes) -> None:
    """A declared type is a claim; the magic number is evidence."""
    assert CONFIG["request"]["trust_declared_content_type"] is False
    honest = handler.handle_scan(grey_jpeg, declared_content_type="image/jpeg")
    lying = handler.handle_scan(grey_jpeg, declared_content_type="image/png")
    assert honest.body == lying.body
    assert lying.log["content_type_mismatch"] is True


def test_reject_detail_stays_out_of_the_response() -> None:
    """The parse failure describes the server's decoder; echoing it is reconnaissance."""
    response = handler.handle_scan(b"GIF89a\x00\x01")
    assert response.status_code == 415
    assert set(response.body) == {"outcome", "reason"}
    assert response.log["reject_detail"]


def test_latency_is_recorded_against_the_budget(grey_jpeg: bytes) -> None:
    """D10. Recorded, never acted on -- a slow correct answer is still correct."""
    log = handler.handle_scan(grey_jpeg).log
    assert log["elapsed_ms"] >= 0.0
    assert isinstance(log["over_p50_budget"], bool)


def test_device_metadata_is_recorded_and_never_echoed(grey_jpeg: bytes) -> None:
    """D11 wants the evidence; the client does not need it back."""
    meta = {"make": "Google", "model": "Pixel 8"}
    response = handler.handle_scan(grey_jpeg, device_metadata=meta)
    assert response.log["device_metadata"] == meta
    assert "Pixel" not in str(response.body)


def test_privacy_config_forbids_persisting_the_image() -> None:
    """A face image is biometric data. Retaining one gives a stateless service its own
    consent, retention and deletion obligations."""
    assert CONFIG["privacy"]["persist_image"] is False
    assert CONFIG["privacy"]["log_image_bytes"] is False


def test_handler_does_not_import_a_web_framework() -> None:
    """Keeping transport out of this package means no web dependency had to be cleared
    through the licensing gate (Rule 1)."""
    source = (
        __import__("pathlib")
        .Path(handler.__file__)
        .read_text(encoding="utf-8")
    )
    for framework in ("fastapi", "flask", "django", "starlette", "aiohttp"):
        assert framework not in source
