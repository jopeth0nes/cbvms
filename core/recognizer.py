"""Face recognition using MTCNN (detection) + InceptionResnetV1 (embedding).

No dlib or CMake required — uses facenet-pytorch which builds on the existing
PyTorch installation already present via ultralytics.
"""

from __future__ import annotations

import pickle
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from database.db_manager import CBVMSDatabase

# Cosine distance threshold: lower = stricter matching.
MATCH_THRESHOLD = 0.6


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a, b))


class FaceRecognizer:
    """Detects and identifies faces using MTCNN + InceptionResnetV1 (VGGFace2)."""

    def __init__(self, db: "CBVMSDatabase") -> None:
        self._db = db
        self._lock = threading.Lock()
        self._mtcnn = None          # lazy — avoid slow import at startup
        self._resnet = None
        self._known: list[tuple[dict, np.ndarray]] = []  # (student_row, embedding)
        self._models_loaded = False
        self.load_known_faces()

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_models(self) -> bool:
        if self._models_loaded:
            return True
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1

            self._mtcnn = MTCNN(
                keep_all=True,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                device="cpu",
            )
            self._resnet = InceptionResnetV1(pretrained="vggface2").eval()
            self._models_loaded = True
            return True
        except Exception as exc:
            print(f"[Recognizer] model load failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_known_faces(self) -> None:
        """Reload all enrolled face embeddings from the database."""
        known: list[tuple[dict, np.ndarray]] = []
        try:
            students = self._db.get_all_students()
            for row in students:
                student = dict(row)          # sqlite3.Row → plain dict
                blob = student.get("encoding")
                if not blob:
                    continue
                try:
                    emb = pickle.loads(blob)
                    if not isinstance(emb, np.ndarray):
                        emb = np.array(emb, dtype=np.float32)
                    known.append((student, emb.astype(np.float32)))
                except Exception:
                    pass
        except Exception as exc:
            print(f"[Recognizer] load_known_faces error: {exc}")
        with self._lock:
            self._known = known

    def encode_face(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, list | None]:
        """Detect the largest face in frame and return its embedding + bounding box.

        Returns (embedding_np, [x1, y1, x2, y2]) or (None, None) if no face found.
        """
        if not self._ensure_models():
            return None, None
        try:
            import torch

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            boxes, probs = self._mtcnn.detect(pil)
            if boxes is None or len(boxes) == 0:
                return None, None

            # Pick the face with highest detection confidence
            best = int(np.argmax(probs))
            box = [int(v) for v in boxes[best]]  # [x1, y1, x2, y2]

            # Get aligned face tensor for the best face
            faces = self._mtcnn(pil)  # (N, 3, 160, 160) or None
            if faces is None:
                return None, None

            face_tensor = faces[best].unsqueeze(0)  # (1, 3, 160, 160)
            with torch.no_grad():
                embedding = self._resnet(face_tensor)  # (1, 512)
            emb_np = embedding.squeeze().numpy().astype(np.float32)
            return emb_np, box

        except Exception as exc:
            print(f"[Recognizer] encode_face error: {exc}")
            return None, None

    def recognize_faces(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect ALL faces in frame and identify each against enrolled students.

        Returns list of dicts:
          {"box": [x1,y1,x2,y2], "name": str, "student_id": str, "matched": bool}
        """
        if not self._ensure_models():
            return []
        try:
            import torch

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            boxes, probs = self._mtcnn.detect(pil)
            if boxes is None or len(boxes) == 0:
                return []

            faces = self._mtcnn(pil)
            if faces is None:
                return []

            with self._lock:
                known_snapshot = list(self._known)

            results = []
            with torch.no_grad():
                embeddings = self._resnet(faces)  # (N, 512)

            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.85:
                    continue
                emb = embeddings[i].numpy().astype(np.float32)
                box_int = [int(v) for v in box]

                name, sid, gender, matched = "Unknown", "", "—", False
                if known_snapshot:
                    distances = [_cosine_distance(emb, k_emb) for _, k_emb in known_snapshot]
                    best_idx = int(np.argmin(distances))
                    if distances[best_idx] < MATCH_THRESHOLD:
                        student = known_snapshot[best_idx][0]
                        name = student.get("name", "Unknown")
                        sid = student.get("student_id", "")
                        gender = student.get("gender", "—") or "—"
                        matched = True

                results.append({
                    "box": box_int,
                    "name": name,
                    "student_id": sid,
                    "gender": gender,
                    "matched": matched,
                })

            return results

        except Exception as exc:
            print(f"[Recognizer] recognize_faces error: {exc}")
            return []
