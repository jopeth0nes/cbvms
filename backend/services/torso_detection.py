"""Compatibility shim — implementation lives in core.torso_detection."""

from core.torso_detection import (  # noqa: F401
    crop_region,
    draw_detections,
    get_face_region,
    get_torso_region,
)

__all__ = [
    "crop_region",
    "draw_detections",
    "get_face_region",
    "get_torso_region",
]
