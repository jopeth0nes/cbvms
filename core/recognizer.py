"""Face recognition engine for CBVMS."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from database.db_manager import CBVMSDatabase

try:
    import face_recognition  # type: ignore

    FACE_RECOGNITION_AVAILABLE = True
except Exception:
    face_recognition = None  # type: ignore
    FACE_RECOGNITION_AVAILABLE = False


class Recognizer:
    """Match detected face regions against enrolled student encodings."""

    def __init__(self, db_manager: CBVMSDatabase, tolerance: float = 0.5) -> None:
        self.db_manager = db_manager
        self.tolerance = tolerance
        self.known_faces: list[dict[str, Any]] = []
        self.load_known_faces()

    def load_known_faces(self) -> None:
        """Reload all student encodings from the database."""
        self.known_faces = []
        for row in self.db_manager.get_all_students():
            blob = row["encoding"]
            if blob is None:
                continue
            try:
                encoding = pickle.loads(blob)
            except Exception:
                continue
            self.known_faces.append(
                {
                    "student_id": row["student_id"],
                    "name": row["name"],
                    "course": row["course"] or "",
                    "year_level": row["year_level"] or "",
                    "encoding": np.asarray(encoding),
                }
            )

    def recognize(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> dict[str, Any] | None:
        """
        Compare the face in the bounding box to known encodings.

        Returns a student dict (student_id, name, course, year_level) or None.
        """
        if not FACE_RECOGNITION_AVAILABLE or face_recognition is None:
            return None
        if not self.known_faces:
            return None

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_location = (y1, x2, y2, x1)

        try:
            encodings = face_recognition.face_encodings(
                rgb,
                known_face_locations=[face_location],
            )
        except Exception:
            return None

        if not encodings:
            return None

        unknown = encodings[0]
        known = [entry["encoding"] for entry in self.known_faces]
        distances = face_recognition.face_distance(known, unknown)
        if len(distances) == 0:
            return None

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        if best_distance > self.tolerance:
            return None

        match = self.known_faces[best_idx]
        return {
            "student_id": match["student_id"],
            "name": match["name"],
            "course": match["course"],
            "year_level": match["year_level"],
        }
