"""Untrusted-upload handling: format sniffing, dimension checks, decoding.

Everything here treats the request body as hostile. Two properties matter:

* **Dimensions are read from the file header, before the decoder allocates.** An upload
  size limit does not bound memory on its own -- a ~200 KB JPEG can declare 30000x30000
  and expand to gigabytes. Checking after ``cv2.imdecode`` returns is checking after the
  allocation that would have killed the process.

* **The client's declared Content-Type is never trusted.** A declared type is a claim; the
  magic number is evidence. A format this module cannot pre-measure is rejected rather
  than passed through, because an unmeasurable format is an unbounded one.

No image byte is ever written to disk or logged (see ``config/serving.yaml`` privacy).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import cv2
import numpy as np

#: Magic-number prefixes. WebP additionally requires "WEBP" at offset 8, checked below.
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_RIFF_MAGIC = b"RIFF"

#: JPEG start-of-frame markers carrying dimensions. 0xC4 (define Huffman table), 0xC8
#: (JPEG extension) and 0xCC (define arithmetic coding) share the range but are not SOF
#: segments, so reading dimensions out of one would return two bytes of a Huffman table.
_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


class UnsupportedImageError(ValueError):
    """The bytes are not a supported image, or their header cannot be measured."""


class ImageTooLargeError(ValueError):
    """The declared dimensions exceed the configured pixel cap (decompression bomb guard)."""


class ImageTooSmallError(ValueError):
    """The image is below the size floor, so it cannot contain a usable face."""


@dataclass(frozen=True)
class ImageHeader:
    """What the file claims about itself, before any of it is decoded."""

    fmt: str
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


def sniff(payload: bytes) -> ImageHeader:
    """Identify the format and read the dimensions from the header.

    Raises:
        UnsupportedImageError: unknown format, or a header too truncated or malformed to
            measure. Both are rejections rather than best-effort decodes: an image whose
            size cannot be established cannot be bounded.
    """
    if payload.startswith(_JPEG_MAGIC):
        return ImageHeader("jpeg", *_jpeg_dimensions(payload))
    if payload.startswith(_PNG_MAGIC):
        return ImageHeader("png", *_png_dimensions(payload))
    if payload[:4] == _RIFF_MAGIC and payload[8:12] == b"WEBP":
        return ImageHeader("webp", *_webp_dimensions(payload))
    raise UnsupportedImageError(
        "unrecognised image format. Only formats whose header can be measured before "
        "decoding are accepted, because an unmeasurable image is an unbounded one."
    )


def check_limits(header: ImageHeader, config: dict) -> None:
    """Enforce the configured pixel cap and size floor.

    Raises:
        UnsupportedImageError: format not in the allow-list.
        ImageTooLargeError / ImageTooSmallError: outside the configured bounds.
    """
    request = config["request"]
    if header.fmt not in set(request["allowed_formats"]):
        raise UnsupportedImageError(f"format {header.fmt!r} is not accepted")

    max_pixels = int(request["max_pixels"])
    if header.pixels > max_pixels:
        raise ImageTooLargeError(
            f"{header.width}x{header.height} = {header.pixels} pixels exceeds the "
            f"{max_pixels} cap"
        )

    min_side = int(request["min_side_px"])
    if min(header.width, header.height) < min_side:
        raise ImageTooSmallError(
            f"shortest side {min(header.width, header.height)}px is below {min_side}px; "
            "no face in it could clear the capture size gate"
        )


def decode(payload: bytes) -> np.ndarray:
    """Decode to BGR uint8 after the header has already been checked.

    ``cv2.imdecode`` applies EXIF orientation, so a portrait capture arrives upright and
    the landmark layer sees the face the way the camera saw it.

    Raises:
        UnsupportedImageError: the bytes did not decode. A header can be well-formed while
            the entropy-coded data behind it is truncated or corrupt.
    """
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise UnsupportedImageError("image data did not decode")
    return np.asarray(image, dtype=np.uint8)


# ------------------------------------------------------------------- header parsing


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    """Walk the JPEG segment chain to the first start-of-frame.

    Segment-by-segment rather than a scan for the marker bytes: 0xFFC0 occurs freely
    inside entropy-coded data, so a naive search finds compressed image content and reads
    it as a size.
    """
    index = 2  # past the SOI marker
    end = len(payload)
    while index + 3 < end:
        if payload[index] != 0xFF:
            raise UnsupportedImageError("malformed JPEG: expected a segment marker")
        marker = payload[index + 1]
        # Standalone markers carry no length field.
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if index + 4 > end:
            break
        length = struct.unpack(">H", payload[index + 2 : index + 4])[0]
        if marker in _SOF_MARKERS:
            if index + 9 > end:
                break
            height, width = struct.unpack(">HH", payload[index + 5 : index + 9])
            return int(width), int(height)
        if length < 2:
            raise UnsupportedImageError("malformed JPEG: non-positive segment length")
        index += 2 + length
    raise UnsupportedImageError("JPEG has no readable start-of-frame segment")


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    """IHDR is required by the spec to be the first chunk, at a fixed offset."""
    if len(payload) < 24 or payload[12:16] != b"IHDR":
        raise UnsupportedImageError("malformed PNG: no IHDR chunk")
    width, height = struct.unpack(">II", payload[16:24])
    return int(width), int(height)


def _webp_dimensions(payload: bytes) -> tuple[int, int]:
    """Three sub-formats, three encodings of the same two numbers.

    VP8 (lossy) stores 14-bit dimensions after a start code; VP8L (lossless) packs
    14-bit-minus-one dimensions into a bit field; VP8X (extended) stores 24-bit
    minus-one dimensions. A file with none of these chunks is not measurable.
    """
    chunk = payload[12:16]

    if chunk == b"VP8X" and len(payload) >= 30:
        width = int.from_bytes(payload[24:27], "little") + 1
        height = int.from_bytes(payload[27:30], "little") + 1
        return width, height

    if chunk == b"VP8 " and len(payload) >= 30:
        if payload[23:26] != b"\x9d\x01\x2a":
            raise UnsupportedImageError("malformed WebP: bad VP8 start code")
        width = struct.unpack("<H", payload[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", payload[28:30])[0] & 0x3FFF
        return int(width), int(height)

    if chunk == b"VP8L" and len(payload) >= 25:
        if payload[20] != 0x2F:
            raise UnsupportedImageError("malformed WebP: bad VP8L signature")
        bits = int.from_bytes(payload[21:25], "little")
        return int(bits & 0x3FFF) + 1, int((bits >> 14) & 0x3FFF) + 1

    raise UnsupportedImageError("WebP has no measurable VP8/VP8L/VP8X chunk")


__all__ = [
    "ImageHeader",
    "ImageTooLargeError",
    "ImageTooSmallError",
    "UnsupportedImageError",
    "check_limits",
    "decode",
    "sniff",
]
