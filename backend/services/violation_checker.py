"""Compatibility shim — implementation lives in core.violation_checker."""

from core.violation_checker import (  # noqa: F401
    check_face_violations,
    check_torso_violations,
    has_earrings,
    has_prohibited_item,
    is_correct_uniform,
    is_id_badge_visible,
)

__all__ = [
    "check_face_violations",
    "check_torso_violations",
    "has_earrings",
    "has_prohibited_item",
    "is_correct_uniform",
    "is_id_badge_visible",
]
