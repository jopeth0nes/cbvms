"""Track live detection sessions and decide when to emit alerts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DETECTION_COOLDOWN_SECONDS = 15.0


def identity_key_from_detection(det: dict[str, Any]) -> str:
    student_id = (det.get("student_id") or "").strip()
    status = det.get("status", "unrecognized")
    if status == "unrecognized" or not student_id or student_id.lower() == "unknown":
        return "unknown"
    return student_id


def violations_from_detection(det: dict[str, Any]) -> set[str]:
    raw = det.get("all_violations") or det.get("violations") or []
    if isinstance(raw, (list, tuple, set)):
        return {str(v).strip() for v in raw if str(v).strip()}
    if det.get("violation_type"):
        return {str(det["violation_type"]).strip()}
    return set()


def face_violations_from_detection(det: dict[str, Any]) -> list[str]:
    raw = det.get("face_violations") or []
    return [str(v).strip() for v in raw if str(v).strip()]


def torso_violations_from_detection(det: dict[str, Any]) -> list[str]:
    raw = det.get("torso_violations") or []
    return [str(v).strip() for v in raw if str(v).strip()]


@dataclass
class AlertEmit:
    """Single alert action for the UI layer."""

    kind: str  # "new" | "update"
    identity_key: str
    name: str
    student_id: str
    dot_color: str
    violation_text: str
    violations_to_log: list[str] = field(default_factory=list)
    face_violations: list[str] = field(default_factory=list)
    torso_violations: list[str] = field(default_factory=list)
    snapshot_jpeg: bytes | None = None


@dataclass
class _PersonState:
    last_alert_time: float
    last_seen_time: float
    active: bool
    violations: set[str]
    name: str
    student_id: str


class DetectionStateTracker:
    """
    Tracks who is in frame and when to emit new vs. updated alerts.

    One alert card per person per appearance; DB rows only for new sessions
    or newly discovered violation types.
    """

    def __init__(self) -> None:
        self.detection_state: dict[str, _PersonState] = {}

    def reset(self) -> None:
        self.detection_state.clear()

    def process_frame(self, detections: list[dict[str, Any]], now: float) -> list[AlertEmit]:
        aggregated = self._aggregate_detections(detections)
        keys_present = set(aggregated.keys())

        for key, state in self.detection_state.items():
            if key in keys_present:
                continue
            if now - state.last_seen_time >= DETECTION_COOLDOWN_SECONDS:
                state.active = False

        events: list[AlertEmit] = []
        for identity_key, det in aggregated.items():
            events.extend(self._process_person(identity_key, det, now))
        return events

    def _aggregate_detections(
        self, detections: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        unknown_seen = False

        for det in detections:
            identity_key = identity_key_from_detection(det)
            if identity_key == "unknown":
                if unknown_seen:
                    continue
                unknown_seen = True

            violations = violations_from_detection(det)
            face_v = set(face_violations_from_detection(det))
            torso_v = set(torso_violations_from_detection(det))
            if identity_key not in merged:
                item = dict(det)
                item["violations"] = violations
                item["face_violations"] = sorted(face_v)
                item["torso_violations"] = sorted(torso_v)
                item["all_violations"] = sorted(violations)
                merged[identity_key] = item
                continue

            existing = merged[identity_key]
            existing_violations: set[str] = existing.get("violations") or set()
            existing["violations"] = existing_violations | violations
            existing["all_violations"] = sorted(existing_violations | violations)
            existing_face: set[str] = set(existing.get("face_violations") or [])
            existing_torso: set[str] = set(existing.get("torso_violations") or [])
            existing["face_violations"] = sorted(existing_face | face_v)
            existing["torso_violations"] = sorted(existing_torso | torso_v)
            if det.get("status") == "violation":
                existing["status"] = "violation"
            if det.get("snapshot_jpeg") is not None:
                existing["snapshot_jpeg"] = det["snapshot_jpeg"]
            if existing.get("name") in (None, "", "Unknown") and det.get("name"):
                existing["name"] = det["name"]
            if not existing.get("student_id") or existing.get("student_id") == "unknown":
                existing["student_id"] = det.get("student_id", existing.get("student_id"))

        return merged

    def _process_person(
        self, identity_key: str, det: dict[str, Any], now: float
    ) -> list[AlertEmit]:
        violations = violations_from_detection(det)
        dot_color, name, violation_text = alert_payload_from_detection(det)
        student_id = (det.get("student_id") or "").strip() or "unknown"
        snapshot = det.get("snapshot_jpeg")
        face_v = face_violations_from_detection(det)
        torso_v = torso_violations_from_detection(det)

        state = self.detection_state.get(identity_key)
        # New alert only for first sighting or after leaving frame (inactive).
        needs_new_alert = state is None or not state.active

        if needs_new_alert:
            self.detection_state[identity_key] = _PersonState(
                last_alert_time=now,
                last_seen_time=now,
                active=True,
                violations=set(violations),
                name=name,
                student_id=student_id,
            )
            return [
                AlertEmit(
                    kind="new",
                    identity_key=identity_key,
                    name=name,
                    student_id=student_id,
                    dot_color=dot_color,
                    violation_text=violation_text,
                    violations_to_log=sorted(violations) if violations else [],
                    face_violations=face_v,
                    torso_violations=torso_v,
                    snapshot_jpeg=snapshot,
                )
            ]

        state.last_seen_time = now
        state.active = True
        state.name = name if name != "Unknown" else state.name

        new_violations = violations - state.violations
        if not new_violations:
            return []

        state.violations |= new_violations
        merged_text = ", ".join(sorted(state.violations)) if state.violations else violation_text
        return [
            AlertEmit(
                kind="update",
                identity_key=identity_key,
                name=state.name,
                student_id=student_id,
                dot_color=dot_color,
                violation_text=merged_text,
                violations_to_log=sorted(new_violations),
                face_violations=face_v,
                torso_violations=torso_v,
                snapshot_jpeg=snapshot,
            )
        ]


def alert_payload_from_detection(det: dict[str, Any]) -> tuple[str, str, str]:
    """Return (dot_color, display_name, violation_text) for UI."""
    color_safe = "#10B981"
    color_danger = "#EF4444"
    color_warning = "#F59E0B"

    status = det.get("status", "unrecognized")
    if status == "violation":
        dot_color = color_danger
        name = det.get("name", "Unknown") or "Unknown"
        violations = violations_from_detection(det)
        if violations:
            violation_text = ", ".join(sorted(violations))
        else:
            violation_text = det.get("violation_type", "violation")
    elif status == "recognized":
        dot_color = color_safe
        name = det.get("name", "Unknown") or "Unknown"
        violation_text = "OK"
    else:
        dot_color = color_warning
        name = "Unknown"
        violation_text = "Unknown person"
    return dot_color, name, violation_text
