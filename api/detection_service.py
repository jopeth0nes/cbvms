"""API-side detection lane with WebSocket broadcast and annotated MJPEG stream."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from core.detector import Detector, draw_detections_list
from core.live_pipeline import LiveDetectionWorker, detection_to_ws_payload
from core.recognizer import Recognizer
from database.db_manager import CBVMSDatabase

PROCESS_EVERY_N_FRAMES = 5
ALERT_BROADCAST_COOLDOWN_SECONDS = 15.0


def should_emit_alert(identity_key: str, last_emit_times: dict[str, float], now: float) -> bool:
    """One alert broadcast per identity per cooldown window."""
    last = last_emit_times.get(identity_key, 0.0)
    if (now - last) < ALERT_BROADCAST_COOLDOWN_SECONDS:
        return False
    last_emit_times[identity_key] = now
    return True


class DetectionService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: set[Any] = set()
        self._latest: list[dict[str, Any]] = []
        self._latest_raw: list[dict[str, Any]] = []
        self._latest_annotated: np.ndarray | None = None
        self._frame_counter = 0
        self._detector: Detector | None = None
        self._detector_loading = False
        self._last_alert_emit: dict[str, float] = {}
        self._db = CBVMSDatabase()
        self._db.initialize()
        self._recognizer = Recognizer(self._db)
        self._worker = LiveDetectionWorker(
            get_detector=lambda: self._detector,
            recognizer=self._recognizer,
            violation_engine=None,
            on_results=self._on_results,
        )
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(self) -> None:
        self._ensure_detector_async()

    def _ensure_detector_async(self) -> None:
        if self._detector is not None or self._detector_loading:
            return
        self._detector_loading = True

        def _load() -> None:
            try:
                self._detector = Detector()
            except Exception as exc:
                print(f"[CBVMS] Failed to load YOLO model: {exc}")
                self._detector = None
            finally:
                self._detector_loading = False
                self._worker.start()

        threading.Thread(target=_load, name="cbvms-api-detector", daemon=True).start()

    def _on_results(self, enriched: list[dict], _person_count: int) -> None:
        payloads = [detection_to_ws_payload(d) for d in enriched]
        with self._lock:
            self._latest = payloads
            self._latest_raw = [dict(d) for d in enriched]
        self._broadcast_detections(payloads)
        self._broadcast_alerts(enriched)

    def _broadcast_detections(self, payloads: list[dict[str, Any]]) -> None:
        if not self._clients or self._loop is None:
            return
        message = json.dumps({"type": "detections", "detections": payloads})
        for ws in list(self._clients):
            asyncio.run_coroutine_threadsafe(ws.send_text(message), self._loop)

    def _broadcast_alerts(self, enriched: list[dict]) -> None:
        if not self._clients or self._loop is None or not enriched:
            return

        now = time.time()
        with self._lock:
            emit_times = dict(self._last_alert_emit)

        for det in enriched:
            identity = det.get("identity") or {}
            student_id = str(
                identity.get("id") or det.get("student_id") or "unknown"
            )
            name = identity.get("name") or det.get("name") or "Unknown"
            face_v = list(det.get("face_violations") or [])
            torso_v = list(det.get("torso_violations") or [])
            all_v = list(det.get("all_violations") or det.get("violations") or [])

            if not should_emit_alert(student_id, emit_times, now):
                continue

            payload = {
                "type": "detection",
                "name": name,
                "student_id": student_id,
                "grade": det.get("year_level", ""),
                "section": det.get("course", ""),
                "face_violations": face_v,
                "torso_violations": torso_v,
                "all_violations": all_v,
                "time": datetime.now().strftime("%H:%M:%S"),
                "identity": {"id": student_id, "name": name},
            }
            print(
                f"[CBVMS] Broadcasting alert for: {name} | violations: {all_v or 'none'}"
            )
            message = json.dumps(payload)
            for ws in list(self._clients):
                asyncio.run_coroutine_threadsafe(ws.send_text(message), self._loop)

        with self._lock:
            self._last_alert_emit = emit_times

    def register_client(self, ws: Any) -> None:
        with self._lock:
            self._clients.add(ws)
            snapshot = list(self._latest)

        if snapshot and self._loop is not None:
            message = json.dumps({"type": "detections", "detections": snapshot})
            asyncio.run_coroutine_threadsafe(ws.send_text(message), self._loop)

    def unregister_client(self, ws: Any) -> None:
        with self._lock:
            self._clients.discard(ws)

    def offer_frame(self, frame: np.ndarray) -> None:
        self._frame_counter += 1
        self._worker.offer_frame(frame, self._frame_counter)

    def process_stream_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on the stream thread so boxes match the encoded JPEG.
        Falls back to the last annotated frame when the detector is still loading.
        """
        self._frame_counter += 1

        if self._detector is None:
            with self._lock:
                cached = self._latest_annotated
            return cached.copy() if cached is not None else frame

        enriched: list[dict] = []
        if self._frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            try:
                enriched, _ = self._worker._process_frame(frame)  # noqa: SLF001
                payloads = [detection_to_ws_payload(d) for d in enriched]
                with self._lock:
                    self._latest = payloads
                    self._latest_raw = [dict(d) for d in enriched]
                self._broadcast_detections(payloads)
                self._broadcast_alerts(enriched)
            except Exception as exc:
                print(f"[CBVMS] Stream detection error: {exc}")

        with self._lock:
            raw = [dict(d) for d in self._latest_raw]

        if not raw:
            return frame

        annotated = draw_detections_list(frame.copy(), raw)
        with self._lock:
            self._latest_annotated = annotated.copy()
        return annotated

    def get_latest(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._latest)

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.process_stream_frame(frame)

    def reload_faces(self) -> None:
        self._recognizer.load_known_faces()


detection_service = DetectionService()
