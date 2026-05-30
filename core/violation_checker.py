"""Zone-specific violation checks (face vs torso crops)."""

from __future__ import annotations

import numpy as np


def has_earrings(_face_crop: np.ndarray) -> bool:
    """Placeholder — replace with trained classifier."""
    return False


def is_correct_uniform(_torso_crop: np.ndarray) -> bool:
    """Placeholder — return True when uniform is compliant."""
    return True


def is_id_badge_visible(_torso_crop: np.ndarray) -> bool:
    """Placeholder — return True when ID badge is visible."""
    return True


def has_prohibited_item(_torso_crop: np.ndarray) -> bool:
    """Placeholder — return True when a prohibited item is detected."""
    return False


def check_face_violations(face_crop: np.ndarray) -> list[str]:
    violations: list[str] = []
    if face_crop is None or face_crop.size == 0:
        return violations
    if has_earrings(face_crop):
        violations.append("earrings_male")
    return violations


def check_torso_violations(torso_crop: np.ndarray) -> list[str]:
    violations: list[str] = []
    if torso_crop is None or torso_crop.size == 0:
        return violations
    if not is_correct_uniform(torso_crop):
        violations.append("dress_code")
    if not is_id_badge_visible(torso_crop):
        violations.append("no_id_badge")
    if has_prohibited_item(torso_crop):
        violations.append("prohibited_items")
    return violations
