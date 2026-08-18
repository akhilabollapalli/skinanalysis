"""HTTP-shaped entry point for the pipeline, with no HTTP framework in it.

``handler.handle_scan`` takes bytes and returns a status code plus a JSON-ready body, so
the transport can be whatever the platform already runs. Keeping a framework out of this
package has two effects worth stating: no web dependency had to be cleared through the
licensing gate (Rule 1), and the request path stays unit-testable without a server.

This package may depend on the pipeline. Nothing in the pipeline may depend on it.
"""

from __future__ import annotations

from .handler import ScanResponse, handle_scan

__all__ = ["ScanResponse", "handle_scan"]
