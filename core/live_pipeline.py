"""Background AI lane for live monitor — decoupled from UI frame display."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from queue import Empty, Full, Queue
from typing import Any

import cv2
import numpy as np

from core.detection_state import DetectionStateTracker, identity_key_from_detection
from core.detector import (
    DETECTION_H,
    DETECTION_SIZE,
    DETECTION_W,
    get_face_box,
    get_torso_box,
    scale_box,
)
from core.violation_checker import check_face_violations, check_torso_violations

PROCESS_EVERY_N_FRAMES = 5
IDENTITY_PROCESS_COOLDOWN_SECONDS = 10.0
DETECT_WIDTH, DETECT_HEIGHT = DETECTION_SIZE
SNAPSHOT_JPEG_QUALITY = 75


def detection_to_ws_payload(det: dict[str, Any]) -> dict[str, Any]:
    """Serialize a detection dict for WebSocket clients."""
    identity = det.get("identity") or {
        "id": det.get("student_id", "unknown"),
        "name": det.get("name", "Unknown"),
    }
    face_box = list(det.get("face_box") or [])
    torso_box = list(det.get("torso_box") or [])
    face_violations = list(det.get("face_violations") or [])
    torso_violations = list(det.get("torso_violations") or [])
    all_violations = list(det.get("all_violations") or det.get("violations") or [])
    return {
        "identity": {
            "id": identity.get("id") or identity.get("student_id") or "unknown",
            "name": identity.get("name") or "Unknown",
        },
        "face_box": face_box,
        "torso_box": torso_box,
        "face_violations": face_violations,
        "torso_violations": torso_violations,
        "all_violations": all_violations,
        "course": det.get("course", ""),
        "year_level": det.get("year_level", ""),
        "status": det.get("status", "unrecognized"),
        "confidence": det.get("confidence"),
    }


class LiveDetectionWorker:
    """
    Consumes frames from a size-1 queue (never backlogs).
    Runs YOLO → face recognition → zone violation checks off the UI thread.
    """

    def __init__(
        self,
        *,
        get_detector: Callable[[], Any | None],
        recognizer: Any,
        violation_engine: Any | None = None,
        on_results: Callable[[list[dict], int], None],
        get_detection_tracker: Callable[[], DetectionStateTracker | None] | None = None,
    ) -> None:
        self._get_detector = get_detector
        self._recognizer = recognizer
        self._violation_engine = violation_engine
        self._on_results = on_results
        self._get_detection_tracker = get_detection_tracker
        self._queue: Queue[np.ndarray] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._identity_last_processed: dict[str, float] = {}
        self._cached_by_identity: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="cbvms-detection", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def reset_caches(self) -> None:
        with self._lock:
            self._identity_last_processed.clear()
            self._cached_by_identity.clear()

    def offer_frame(self, frame: np.ndarray, frame_index: int) -> None:
        if frame_index % PROCESS_EVERY_N_FRAMES != 0:
            return
        if not self._queue.empty():
            return
        try:
            self._queue.put_nowait(frame.copy())
        except Full:
            pass

    def process_frame(self, frame: np.ndarray) -> tuple[list[dict], int]:
        """Public entry for API lane — same logic as worker thread."""
        return self._process_frame(frame)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                enriched, count = self._process_frame(frame)
                self._on_results(enriched, count)
            except Exception:
                continue

    def _process_frame(self, frame: np.ndarray) -> tuple[list[dict], int]:
        detector = self._get_detector()
        if detector is None:
            return [], 0

        small = cv2.resize(frame, DETECTION_SIZE, interpolation=cv2.INTER_LINEAR)
        display_size = (frame.shape[1], frame.shape[0])

        try:
            person_boxes = detector.detect_persons(small)
        except Exception:
            person_boxes = []

        if not person_boxes:
            return [], 0

        now = time.time()
        enriched: list[dict] = []
        unknown_handled = False

        for i, person_box in enumerate(person_boxes):
            face_box = get_face_box(person_box)
            torso_box = get_torso_box(person_box)

            fx1, fy1, fx2, fy2 = face_box
            tx1, ty1, tx2, ty2 = torso_box
            face_crop = small[fy1:fy2, fx1:fx2]
            torso_crop = small[ty1:ty2, tx1:tx2]

            if face_crop.size == 0 or torso_crop.size == 0:
                continue

            face_display = scale_box(face_box, DETECTION_W, DETECTION_H, display_size[0], display_size[1])
            dfx1, dfy1, dfx2, dfy2 = face_display
            match = self._recognizer.recognize(frame, dfx1, dfy1, dfx2, dfy2)

            if match:
                identity = {
                    "id": match["student_id"],
                    "name": match["name"],
                }
                item: dict[str, Any] = {
                    "status": "recognized",
                    "name": match["name"],
                    "student_id": match["student_id"],
                    "course": match.get("course", ""),
                    "year_level": match.get("year_level", ""),
                    "identity": identity,
                }
            else:
                identity = {"id": "unknown", "name": "Unknown"}
                item = {
                    "status": "unrecognized",
                    "name": "Unknown",
                    "student_id": "unknown",
                    "course": "",
                    "year_level": "",
                    "identity": identity,
                }

            px1, py1, px2, py2 = person_box
            item["x1"] = px1
            item["y1"] = py1
            item["x2"] = px2
            item["y2"] = py2
            item["face_box"] = face_box
            item["torso_box"] = torso_box
            item["person_box"] = person_box

            key = identity_key_from_detection(item)
            if key == "unknown" and unknown_handled:
                continue
            if key == "unknown":
                unknown_handled = True

            if self._should_skip_identity(key, now):
                cached = self._get_cached(key)
                if cached is not None:
                    enriched.append(dict(cached))
                    continue

            face_violations = check_face_violations(face_crop)
            torso_violations = check_torso_violations(torso_crop)
            all_violations = face_violations + torso_violations

            item["face_violations"] = face_violations
            item["torso_violations"] = torso_violations
            item["all_violations"] = all_violations
            item["violations"] = all_violations

            snapshot_jpeg = None
            try:
                person_display = scale_box(
                    person_box, DETECTION_W, DETECTION_H, display_size[0], display_size[1]
                )
                dpx1, dpy1, dpx2, dpy2 = person_display
                h, w = frame.shape[:2]
                person_crop = frame[max(0, dpy1) : min(h, dpy2), max(0, dpx1) : min(w, dpx2)]
                ok, buf = cv2.imencode(
                    ".jpg",
                    person_crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), SNAPSHOT_JPEG_QUALITY],
                )
                if ok and buf is not None:
                    snapshot_jpeg = buf.tobytes()
            except Exception:
                snapshot_jpeg = None
            item["snapshot_jpeg"] = snapshot_jpeg

            if all_violations:
                item["status"] = "violation"
                item["violation_type"] = all_violations[0]

            self._store_cached(key, item, now)
            enriched.append(item)

        return enriched, len(person_boxes)

    def _should_skip_identity(self, identity_key: str, now: float) -> bool:
        with self._lock:
            last = self._identity_last_processed.get(identity_key, 0.0)
        if (now - last) < IDENTITY_PROCESS_COOLDOWN_SECONDS:
            return True
        if self._get_detection_tracker is not None:
            tracker = self._get_detection_tracker()
            if tracker is not None:
                state = tracker.detection_state.get(identity_key)
                if (
                    state is not None
                    and state.active
                    and (now - state.last_alert_time) < IDENTITY_PROCESS_COOLDOWN_SECONDS
                ):
                    return True
        return False

    def _get_cached(self, identity_key: str) -> dict | None:
        with self._lock:
            cached = self._cached_by_identity.get(identity_key)
        return dict(cached) if cached is not None else None

    def _store_cached(self, identity_key: str, item: dict, now: float) -> None:
        with self._lock:
            self._cached_by_identity[identity_key] = dict(item)
            self._identity_last_processed[identity_key] = now
