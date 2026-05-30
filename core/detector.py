"""YOLO person detection and two-zone box geometry for CBVMS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from core.violation_checker import check_face_violations, check_torso_violations

DETECTION_SIZE = (640, 480)
DETECTION_W, DETECTION_H = DETECTION_SIZE

_global_model = None
_recognizer_instance = None


def get_model() -> YOLO:
    global _global_model
    if _global_model is None:
        root = Path(__file__).resolve().parent.parent
        model_path = root / "models" / "yolov8n.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _global_model = YOLO(str(model_path))
    return _global_model


def recognize_face(face_crop: np.ndarray) -> dict[str, Any]:
    global _recognizer_instance
    if _recognizer_instance is None:
        from database.db_manager import CBVMSDatabase
        from core.recognizer import Recognizer
        try:
            db = CBVMSDatabase()
            db.initialize()
            _recognizer_instance = Recognizer(db)
        except Exception:
            pass

    if _recognizer_instance is None:
        return {"name": "Unknown"}

    from core.recognizer import FACE_RECOGNITION_AVAILABLE
    if not FACE_RECOGNITION_AVAILABLE:
        return {"name": "Unknown"}

    import face_recognition
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    face_location = (0, w, h, 0)
    try:
        encodings = face_recognition.face_encodings(rgb, known_face_locations=[face_location])
    except Exception:
        return {"name": "Unknown"}

    if not encodings:
        return {"name": "Unknown"}

    unknown = encodings[0]
    known = [entry["encoding"] for entry in _recognizer_instance.known_faces]
    if not known:
        return {"name": "Unknown"}

    distances = face_recognition.face_distance(known, unknown)
    if len(distances) == 0:
        return {"name": "Unknown"}

    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    if best_distance > _recognizer_instance.tolerance:
        return {"name": "Unknown"}

    match = _recognizer_instance.known_faces[best_idx]
    return {
        "id": match["student_id"],
        "name": match["name"],
        "course": match["course"],
        "year_level": match["year_level"],
    }


def detect_persons(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    model = get_model()
    results = model(frame, classes=[0], verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def get_face_box(person_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = person_box
    total_height = y2 - y1
    face_y2 = y1 + max(int(total_height * 0.30), 20)
    return (x1, y1, x2, face_y2)


def get_torso_box(person_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = person_box
    total_height = y2 - y1
    torso_y1 = y1 + int(total_height * 0.28)
    torso_y2 = y1 + int(total_height * 0.65)
    if torso_y2 <= torso_y1:
        torso_y2 = torso_y1 + 20
    return (x1, torso_y1, x2, torso_y2)


def scale_box(box: tuple[int, int, int, int], from_size: Any, to_size: Any = None, *args: Any) -> tuple[int, int, int, int]:
    if isinstance(from_size, (int, float)):
        fw = from_size
        fh = to_size
        tw = args[0]
        th = args[1]
    else:
        fw, fh = from_size
        tw, th = to_size
    x1, y1, x2, y2 = box
    return (
        int(x1 * (tw / fw)),
        int(y1 * (th / fh)),
        int(x2 * (tw / fw)),
        int(y2 * (th / fh))
    )


def draw_detections(display_frame: np.ndarray, results: dict[str, dict[str, Any]], detection_size: tuple[int, int] = DETECTION_SIZE) -> np.ndarray:
    display_size = (display_frame.shape[1], display_frame.shape[0])
    for pid, data in results.items():
        face_box  = scale_box(data["face_box"],  detection_size, display_size)
        torso_box = scale_box(data["torso_box"], detection_size, display_size)
        fx1, fy1, fx2, fy2 = face_box
        tx1, ty1, tx2, ty2 = torso_box
        # BLUE — face/head zone
        cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (255, 100, 0), 2)
        cv2.putText(display_frame, data["identity"]["name"],
                    (fx1, fy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        # ORANGE — torso zone
        cv2.rectangle(display_frame, (tx1, ty1), (tx2, ty2), (0, 140, 255), 2)
        cv2.putText(display_frame, "Torso",
                    (tx1, ty1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
        if data.get("torso_violations"):
            cv2.putText(display_frame, ", ".join(data["torso_violations"]),
                        (tx1, ty2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
    return display_frame


def draw_detections_list(
    display_frame: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Adapter: convert list payloads from the live worker into a results dict."""
    if not detections:
        return display_frame

    results: dict[str, dict[str, Any]] = {}
    for i, det in enumerate(detections):
        identity = det.get("identity") or {}
        pid = str(identity.get("id") or det.get("student_id") or f"unknown_{i}")
        results[pid] = det
    return draw_detections(display_frame.copy(), results)


def process_frame(display_frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    small = cv2.resize(display_frame, DETECTION_SIZE)
    person_boxes = detect_persons(small)
    results = {}
    for i, person_box in enumerate(person_boxes):
        face_box  = get_face_box(person_box)
        torso_box = get_torso_box(person_box)
        fx1, fy1, fx2, fy2 = face_box
        tx1, ty1, tx2, ty2 = torso_box
        face_crop  = small[fy1:fy2, fx1:fx2]
        torso_crop = small[ty1:ty2, tx1:tx2]
        if face_crop.size == 0 or torso_crop.size == 0:
            continue
        identity         = recognize_face(face_crop)
        face_violations  = check_face_violations(face_crop)
        torso_violations = check_torso_violations(torso_crop)
        pid = identity.get("id", f"unknown_{i}")
        results[pid] = {
            "identity":         identity,
            "person_box":       person_box,
            "face_box":         face_box,
            "torso_box":        torso_box,
            "face_violations":  face_violations,
            "torso_violations": torso_violations,
            "all_violations":   face_violations + torso_violations,
        }
    return draw_detections(display_frame.copy(), results), results


class Detector:
    """Ultralytics YOLO wrapper — person class only (COCO class 0)."""

    def __init__(self, model_path: str = "models/yolov8n.pt") -> None:
        root = Path(__file__).resolve().parent.parent
        path = Path(model_path)
        if not path.is_absolute():
            path = root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = path
        self.model = YOLO(str(self.model_path))
        global _global_model
        _global_model = self.model
        model_name = getattr(self.model, "model_name", None) or path.name
        print(f"[CBVMS] YOLO model loaded: {model_name}")

    def detect_persons(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        return detect_persons(frame)
