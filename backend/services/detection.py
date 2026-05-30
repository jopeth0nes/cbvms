"""Compatibility shim — frame processing lives in core.live_pipeline."""

from core.live_pipeline import (  # noqa: F401
    LiveDetectionWorker,
    detection_to_ws_payload,
)

__all__ = ["LiveDetectionWorker", "detection_to_ws_payload"]
