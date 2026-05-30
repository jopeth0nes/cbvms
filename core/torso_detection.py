"""Face and torso region helpers for two-zone person detection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def get_face_region(person_box: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    from core.detector import get_face_box

    return get_face_box(tuple(person_box))


def get_torso_region(person_box: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    from core.detector import get_torso_box

    return get_torso_box(tuple(person_box))


def crop_region(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    px1, py1 = max(0, x1), max(0, y1)
    px2, py2 = min(w, x2), min(h, y2)
    if px2 <= px1 or py2 <= py1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[py1:py2, px1:px2]


def draw_detections(frame: np.ndarray, results: list[dict[str, Any]]) -> np.ndarray:
    from core.detector import draw_detections_list

    return draw_detections_list(frame, results)
