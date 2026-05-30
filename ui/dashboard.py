"""Main CBVMS dashboard with live camera feed."""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime

import cv2
import customtkinter as ctk
import numpy as np

from core.camera import CameraCapture
from core.recognizer import FaceRecognizer
from database.db_manager import CBVMSDatabase
from ui.camera_feed import CameraFeed
from ui.enrollment import EnrollmentPanel
from ui.settings import SettingsPanel
from ui.violation_log import ViolationLogPanel
from ui.components import (
    COLOR_ACCENT,
    COLOR_ACCENT_HOVER,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_SAFE,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
    CORNER_RADIUS,
    PADDING,
    PADDING_LG,
    SIDEBAR_LEFT_WIDTH,
    SIDEBAR_RIGHT_WIDTH,
    apply_cbvms_theme,
    body_font,
    body_small_font,
    heading_font,
    panel_title_font,
    show_toast,
)

FEED_FPS = 30
MAX_ALERTS = 50
PRESENCE_TIMEOUT_SECS = 10  # seconds absent from frame before next appearance triggers new alert


class CBVMSDashboard(ctk.CTk):
    def __init__(self, username: str = "admin") -> None:
        super().__init__()
        self.username = username
        self._active_nav = "live"
        self._alerts: deque[dict] = deque(maxlen=MAX_ALERTS)
        self._face_presence: dict[str, float] = {}   # identity_key → last_seen_epoch
        self._feed_job: str | None = None
        self._clock_job: str | None = None
        self._stats_job: str | None = None
        self._feed_interval_ms = max(16, 1000 // FEED_FPS)

        self._camera: CameraCapture | None = None
        self._camera_index_setting = 0
        self._camera_source_url: str | None = None
        self._camera_resolution_setting = (1280, 720)
        self._fps_cap_setting = 30
        self._camera_retry_count = 0

        self._database = CBVMSDatabase()
        self._database.initialize()

        # Face recognizer (MTCNN + InceptionResnetV1) — lazy model load inside
        self._recognizer = FaceRecognizer(self._database)

        # Background face detection state
        self._face_detections: list[dict] = []
        self._face_frame_counter: int = 0
        self._face_queue: queue.Queue = queue.Queue(maxsize=1)
        self._face_worker = threading.Thread(target=self._face_worker_loop, daemon=True)
        self._face_worker.start()

        self._enrollment_panel: EnrollmentPanel | None = None
        self._violation_panel: ViolationLogPanel | None = None
        self._settings_panel: SettingsPanel | None = None

        self._stat_today_value: ctk.CTkLabel | None = None
        self._stat_unreviewed_value: ctk.CTkLabel | None = None
        self._stat_students_value: ctk.CTkLabel | None = None
        self._stat_last_value: ctk.CTkLabel | None = None

        apply_cbvms_theme()
        self.title("CBVMS — Dashboard")
        self.geometry("1280x800")
        self.minsize(1280, 800)
        self.configure(fg_color=COLOR_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._load_camera_preference()

        self._build_ui()
        self._build_menubar()
        self._tick_clock()
        self._schedule_feed_update()
        self.after(300, self._deferred_start_camera)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=220)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=280)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_sidebar()
        self._build_center_panel()
        self._build_right_sidebar()

    def _build_left_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(
            self,
            width=SIDEBAR_LEFT_WIDTH,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(PADDING, 0), pady=PADDING)
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar, text="CBVMS", font=heading_font(26), text_color=COLOR_ACCENT,
        ).pack(anchor="w", padx=PADDING, pady=(PADDING_LG, 0))

        ctk.CTkLabel(
            sidebar, text="Vision Monitoring System",
            font=body_small_font(), text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", padx=PADDING, pady=(0, PADDING_LG))

        nav_items = [
            ("live",       "📹  Live Monitor"),
            ("enrollment", "👤  Student Enrollment"),
            ("violations", "⚠  Violation Log"),
            ("settings",   "⚙  Settings"),
        ]
        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, label in nav_items:
            btn = ctk.CTkButton(
                sidebar, text=label, anchor="w", height=40,
                corner_radius=CORNER_RADIUS,
                fg_color=COLOR_ACCENT if key == "live" else "transparent",
                hover_color=COLOR_ACCENT_HOVER if key == "live" else COLOR_BORDER,
                text_color=COLOR_TEXT, font=body_small_font(),
                command=lambda k=key: self._on_nav_select(k),
            )
            btn.pack(fill="x", padx=PADDING, pady=4)
            self._nav_buttons[key] = btn

            def _enter(_e, b=btn, k=key):
                if self._active_nav != k:
                    b.configure(fg_color=COLOR_BORDER)

            def _leave(_e, b=btn, k=key):
                if self._active_nav != k:
                    b.configure(fg_color="transparent")

            btn.bind("<Enter>", _enter)
            btn.bind("<Leave>", _leave)

        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=PADDING, pady=PADDING)

        ctk.CTkLabel(
            footer, text=f"Logged in as {self.username}",
            font=body_small_font(), text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 8))

        ctk.CTkButton(
            footer, text="Logout", height=36, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER, hover_color=COLOR_DANGER, command=self._logout,
        ).pack(fill="x")

    def _build_center_panel(self) -> None:
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=PADDING, pady=PADDING)
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(1, weight=1)

        title_row = ctk.CTkFrame(center, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, PADDING))

        self._center_title = ctk.CTkLabel(
            title_row, text="Live Monitor",
            font=panel_title_font(), text_color=COLOR_TEXT,
        )
        self._center_title.pack(side="left")

        self._datetime_label = ctk.CTkLabel(
            title_row, text="", font=body_small_font(), text_color=COLOR_TEXT_MUTED,
        )
        self._datetime_label.pack(side="right")

        self._content_stack = ctk.CTkFrame(center, fg_color="transparent")
        self._content_stack.grid(row=1, column=0, sticky="nsew")
        self._content_stack.grid_columnconfigure(0, weight=1)
        self._content_stack.grid_rowconfigure(0, weight=1)

        self._view_host = ctk.CTkFrame(self._content_stack, fg_color="transparent")
        self._view_host.grid(row=0, column=0, sticky="nsew")
        self._view_host.grid_columnconfigure(0, weight=1)
        self._view_host.grid_rowconfigure(0, weight=1)

        self._live_frame = ctk.CTkFrame(self._view_host, fg_color="transparent")
        self._live_frame.grid_columnconfigure(0, weight=1)
        self._live_frame.grid_rowconfigure(0, weight=0)  # stat cards
        self._live_frame.grid_rowconfigure(1, weight=1)  # camera feed
        self._live_frame.grid_rowconfigure(2, weight=0)  # status bar

        self._stats_row = self._build_stats_row(self._live_frame)
        self._stats_row.grid(row=0, column=0, sticky="ew", pady=(0, PADDING))

        # Camera feed — no fixed width/height; grid(sticky="nsew") controls size
        self.camera_feed = CameraFeed(self._live_frame, bg_color=COLOR_BG)
        self.camera_feed.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        status_bar = ctk.CTkFrame(
            self._live_frame, fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS, border_width=1, border_color=COLOR_BORDER, height=48,
        )
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.grid_propagate(False)

        self._status_camera = ctk.CTkLabel(
            status_bar, text="Camera: Starting…",
            font=body_small_font(), text_color=COLOR_TEXT_MUTED,
        )
        self._status_camera.pack(side="left", padx=PADDING, pady=12)

        self._camera_spinner = ctk.CTkProgressBar(status_bar, mode="indeterminate", width=140)
        self._camera_spinner.pack(side="right", padx=(0, 10), pady=14)
        self._camera_spinner.stop()
        self._camera_spinner.pack_forget()

        self._status_fps = ctk.CTkLabel(
            status_bar, text="FPS: —",
            font=body_small_font(), text_color=COLOR_ACCENT,
        )
        self._status_fps.pack(side="right", padx=PADDING)

        self._enrollment_panel = EnrollmentPanel(
            self._view_host,
            database=self._database,
            recognizer=self._recognizer,
            get_frame=self._get_camera_frame,
        )
        self._enrollment_panel.grid_remove()

        self._violation_panel = ViolationLogPanel(self._view_host, database=self._database)
        self._violation_panel.grid_remove()

        self._settings_panel = SettingsPanel(
            self._view_host,
            database=self._database,
            recognizer=self._recognizer,
            violation_engine=None,
            username=self.username,
            get_detector_loaded=lambda: False,
            apply_camera_settings=self._apply_camera_settings,
            on_camera_source_connected=self._on_camera_source_connected,
        )
        self._settings_panel.grid_remove()

        self._views: dict[str, ctk.CTkFrame] = {
            "live":       self._live_frame,
            "enrollment": self._enrollment_panel,
            "violations": self._violation_panel,
            "settings":   self._settings_panel,
        }
        self._live_frame.grid(row=0, column=0, sticky="nsew")
        self._schedule_stats_refresh()

    def _build_stats_row(self, master: ctk.CTkFrame) -> ctk.CTkFrame:
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="stats")

        def _card(col: int, *, label: str, accent: str) -> ctk.CTkLabel:
            card = ctk.CTkFrame(
                row, fg_color=COLOR_SURFACE, corner_radius=CORNER_RADIUS,
                border_width=1, border_color=COLOR_BORDER, height=80,
            )
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 10, 0))
            card.grid_propagate(False)
            card.grid_columnconfigure(0, weight=1)
            value = ctk.CTkLabel(card, text="—", font=heading_font(22), text_color=accent)
            value.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 0))
            ctk.CTkLabel(
                card, text=label, font=body_font(12), text_color=COLOR_TEXT_MUTED,
            ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 12))
            return value

        self._stat_today_value    = _card(0, label="Total Violations Today", accent=COLOR_DANGER)
        self._stat_unreviewed_value = _card(1, label="Unreviewed",           accent=COLOR_WARNING)
        self._stat_students_value = _card(2, label="Students Enrolled",       accent=COLOR_ACCENT)
        self._stat_last_value     = _card(3, label="Last Detection",          accent=COLOR_SAFE)
        return row

    def _build_right_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(
            self, width=SIDEBAR_RIGHT_WIDTH, fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS, border_width=1, border_color=COLOR_BORDER,
        )
        sidebar.grid(row=0, column=2, rowspan=4, sticky="nsew", padx=(0, 10), pady=10)
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            sidebar, text="Live Alerts", font=heading_font(18), text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=PADDING, pady=(PADDING, PADDING))

        self._alerts_scroll = ctk.CTkScrollableFrame(
            sidebar, fg_color=COLOR_BG, corner_radius=CORNER_RADIUS,
        )
        self._alerts_scroll.grid(row=1, column=0, sticky="nsew", padx=PADDING, pady=(0, PADDING))

        ctk.CTkButton(
            sidebar, text="Clear Alerts", height=36, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER, hover_color=COLOR_ACCENT_HOVER,
            command=self._clear_alerts,
        ).grid(row=2, column=0, sticky="ew", padx=PADDING, pady=(0, PADDING))

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About CBVMS", command=self._open_about_dialog)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

    def _open_about_dialog(self) -> None:
        from ui.components import APP_COLLEGE_NAME, APP_VERSION

        win = ctk.CTkToplevel(self)
        win.title("About CBVMS")
        win.geometry("460x240")
        win.configure(fg_color=COLOR_BG)
        win.resizable(False, False)

        ctk.CTkLabel(
            win, text="Computer Based Vision Monitoring System (CBVMS)",
            font=panel_title_font(), text_color=COLOR_TEXT,
            wraplength=420, justify="center",
        ).pack(padx=PADDING, pady=(PADDING, 10))

        ctk.CTkLabel(
            win, text=f"Version {APP_VERSION}\n{APP_COLLEGE_NAME}",
            font=body_small_font(), text_color=COLOR_TEXT_MUTED, justify="center",
        ).pack(padx=PADDING, pady=(0, 14))

        ctk.CTkButton(
            win, text="Close", height=34, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER, hover_color=COLOR_ACCENT_HOVER, command=win.destroy,
        ).pack(pady=(0, PADDING))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_nav_select(self, key: str) -> None:
        if key not in self._views:
            return
        previous_nav = self._active_nav
        self._active_nav = key

        for nav_key, btn in self._nav_buttons.items():
            btn.configure(
                fg_color=COLOR_ACCENT if nav_key == key else "transparent",
                hover_color=COLOR_ACCENT_HOVER if nav_key == key else COLOR_BORDER,
            )

        if previous_nav == "enrollment" and self._enrollment_panel is not None:
            self._enrollment_panel.on_hide()

        titles = {
            "live":       "Live Monitor",
            "enrollment": "Student Enrollment",
            "violations": "Violation Log",
            "settings":   "Settings",
        }
        self._center_title.configure(text=titles.get(key, "CBVMS"))

        def _switch_views() -> None:
            for view in self._views.values():
                view.grid_remove()
            self._views[key].grid(row=0, column=0, sticky="nsew")
            self._view_host.tkraise()
            if key == "enrollment" and self._enrollment_panel is not None:
                self._enrollment_panel.on_show()
            if key == "violations" and self._violation_panel is not None:
                self._violation_panel.refresh()

        self._fade_transition(_switch_views)

    def _fade_transition(self, on_midpoint) -> None:
        try:
            original = float(self.attributes("-alpha") or 1.0)
        except Exception:
            on_midpoint()
            return

        steps = [1.0, 0.94, 0.90]

        def _fade_out(i: int = 0) -> None:
            if i >= len(steps):
                on_midpoint()
                self.after(0, _fade_in, len(steps) - 1)
                return
            try:
                self.attributes("-alpha", steps[i])
            except Exception:
                on_midpoint()
                return
            self.after(18, _fade_out, i + 1)

        def _fade_in(i: int) -> None:
            if i < 0:
                try:
                    self.attributes("-alpha", original)
                except Exception:
                    pass
                return
            try:
                self.attributes("-alpha", steps[i])
            except Exception:
                return
            self.after(18, _fade_in, i - 1)

        _fade_out(0)

    # ------------------------------------------------------------------
    # Camera preferences
    # ------------------------------------------------------------------

    def _load_camera_preference(self) -> None:
        try:
            from api.camera_store import get_camera_preference
            pref = get_camera_preference()
        except Exception:
            pref = None
        if not pref:
            return
        cam_type = str(pref.get("type", "")).lower()
        if cam_type == "usb":
            self._camera_index_setting = int(pref.get("index", 0))
            self._camera_source_url = None
        elif cam_type in ("rj45", "ip"):
            url = str(pref.get("url", "")).strip()
            self._camera_source_url = url or None

    def _on_camera_source_connected(self, pref: dict) -> None:
        cam_type = str(pref.get("type", "")).lower()
        if cam_type == "usb":
            self._camera_index_setting = int(pref.get("index", 0))
            self._camera_source_url = None
        elif cam_type in ("rj45", "ip"):
            self._camera_source_url = str(pref.get("url", "")).strip() or None
        self._camera_retry_count = 0
        self._deferred_start_camera()

    def _apply_camera_settings(
        self, camera_index: int, resolution: tuple[int, int], fps_cap: int
    ) -> None:
        if self._camera_source_url is None:
            self._camera_index_setting = int(camera_index)
        self._camera_resolution_setting = (int(resolution[0]), int(resolution[1]))
        self._fps_cap_setting = max(10, min(60, int(fps_cap)))
        self._feed_interval_ms = max(16, 1000 // self._fps_cap_setting)
        self._camera_retry_count = 0
        self._deferred_start_camera()
        show_toast(self, "Restarting camera…", type="info", duration=1800)

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def _deferred_start_camera(self) -> None:
        self.update_idletasks()
        self._stop_camera()
        try:
            self._camera_spinner.pack(side="right", padx=(0, 10), pady=14)
            self._camera_spinner.start()
        except Exception:
            pass
        self._status_camera.configure(text="Camera: Starting…", text_color=COLOR_TEXT_MUTED)

        if self._camera_source_url:
            cap = CameraCapture(
                source_url=self._camera_source_url,
                width=self._camera_resolution_setting[0],
                height=self._camera_resolution_setting[1],
                fps_cap=self._fps_cap_setting,
            )
        else:
            cap = CameraCapture(
                camera_index=self._camera_index_setting,
                width=self._camera_resolution_setting[0],
                height=self._camera_resolution_setting[1],
                fps_cap=self._fps_cap_setting,
            )
        self._camera = cap

        # Open on a background thread so the warmup loop (~1.2 s) doesn't block UI.
        def _open_bg() -> None:
            ok = cap.open()
            if self.winfo_exists():
                self.after(0, lambda: self._on_camera_opened(cap, ok))

        threading.Thread(target=_open_bg, daemon=True).start()

    def _on_camera_opened(self, cap: CameraCapture, ok: bool) -> None:
        try:
            self._camera_spinner.stop()
            self._camera_spinner.pack_forget()
        except Exception:
            pass

        if self._camera is not cap:
            # A newer open attempt superseded this one.
            cap.release()
            return

        if ok:
            self._camera_retry_count = 0
            self._status_camera.configure(text="Camera: Active", text_color=COLOR_SAFE)
        else:
            err = cap.last_error or "Could not open camera"
            self._status_camera.configure(text=f"Camera: {err}", text_color=COLOR_DANGER)
            if self._camera_retry_count == 0:
                show_toast(self, f"Camera error: {err}", type="error", duration=4000)
            if self._camera_retry_count < 8:
                self._camera_retry_count += 1
                self.after(2000, self._deferred_start_camera)

    def _stop_camera(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def _get_camera_frame(self):
        if self._camera and self._camera.is_open:
            frame = self._camera.get_latest_frame()
            return frame if frame is not None else self._camera.read()
        return None

    # ------------------------------------------------------------------
    # Background face detection worker
    # ------------------------------------------------------------------

    def _face_worker_loop(self) -> None:
        """Background thread: picks frames from queue, runs face recognition.
        Uses after(0, ...) to deliver results to the UI thread safely.
        """
        while True:
            try:
                frame = self._face_queue.get(timeout=1.0)
                detections = self._recognizer.recognize_faces(frame)
                if self.winfo_exists():
                    self.after(0, self._on_detections_ready, detections)
            except queue.Empty:
                pass
            except Exception as exc:
                print(f"[CBVMS] face worker error: {exc}")

    def _on_detections_ready(self, detections: list[dict]) -> None:
        """UI-thread callback: update annotation state and fire presence-based alerts.

        One alert fires when a face first appears. While the face stays in frame,
        last_seen is refreshed and no duplicate alert is emitted. After PRESENCE_TIMEOUT_SECS
        without a detection the entry is purged — the next appearance fires a new alert.
        """
        self._face_detections = detections
        now = time.time()

        current_keys: set[str] = set()
        for det in detections:
            key = det["student_id"] if det["matched"] else "unknown"
            current_keys.add(key)

            last_seen = self._face_presence.get(key, 0.0)
            is_new_appearance = (now - last_seen) > PRESENCE_TIMEOUT_SECS

            self._face_presence[key] = now   # always refresh last-seen timestamp

            if is_new_appearance:
                self._push_alert(det)

        # Remove identities that have left the frame long enough
        stale = [
            k for k, t in self._face_presence.items()
            if k not in current_keys and (now - t) > PRESENCE_TIMEOUT_SECS
        ]
        for k in stale:
            del self._face_presence[k]

    def _push_alert(self, det: dict) -> None:
        """Add one alert card for a detected face."""
        entry = {
            "identity_key": det["student_id"] or "unknown",
            "name": det["name"],
            "student_id": det["student_id"] or "—",
            "gender": det.get("gender", "—"),
            "matched": det["matched"],
            "time": datetime.now().strftime("%H:%M:%S"),
            "epoch": time.time(),
        }
        self._alerts.appendleft(entry)
        self._refresh_alerts_ui()

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw face detection boxes + names onto a copy of the frame."""
        if not self._face_detections:
            return frame
        out = frame.copy()
        for det in self._face_detections:
            x1, y1, x2, y2 = det["box"]
            matched = det["matched"]
            color = (16, 185, 129) if matched else (68, 68, 239)  # BGR green / red
            label = det["name"] if matched else "Unknown"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Feed update loop
    # ------------------------------------------------------------------

    def _schedule_feed_update(self) -> None:
        self._feed_job = self.after(self._feed_interval_ms, self._update_feed)

    def _update_feed(self) -> None:
        if not self.winfo_exists():
            return
        try:
            if self._camera and self._camera.is_open:
                frame = self._camera.read()
                if frame is not None:
                    if self._active_nav == "live":
                        # Keep presence timestamps alive for currently-visible identities.
                        # This prevents the inference gap (CPU can take 5-15s per frame)
                        # from falsely resetting a person's presence and re-firing an alert.
                        _now = time.time()
                        for _det in self._face_detections:
                            _key = _det["student_id"] if _det["matched"] else "unknown"
                            if _key in self._face_presence:
                                self._face_presence[_key] = _now

                        # Offer every 5th frame to the face detection worker
                        self._face_frame_counter += 1
                        if self._face_frame_counter % 5 == 0:
                            try:
                                self._face_queue.put_nowait(frame.copy())
                            except queue.Full:
                                pass
                        # Annotate with latest detection results and render
                        annotated = self._annotate_frame(frame)
                        self.camera_feed.render(annotated)
                    if self._active_nav == "enrollment" and self._enrollment_panel is not None:
                        self._enrollment_panel.update_preview(frame)
                else:
                    self.camera_feed.show_placeholder()
            else:
                self.camera_feed.show_placeholder()
        except Exception as exc:
            print(f"[CBVMS] feed error: {exc}")
        self._feed_job = self.after(self._feed_interval_ms, self._update_feed)

    # ------------------------------------------------------------------
    # Clock & stats
    # ------------------------------------------------------------------

    def _tick_clock(self) -> None:
        self._datetime_label.configure(text=datetime.now().strftime("%A, %d %b %Y  %H:%M:%S"))
        self._clock_job = self.after(1000, self._tick_clock)

    def _schedule_stats_refresh(self) -> None:
        if self._stats_job:
            try:
                self.after_cancel(self._stats_job)
            except Exception:
                pass
            self._stats_job = None
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        try:
            with self._database.connect() as conn:
                today      = conn.execute("SELECT COUNT(*) AS c FROM violations WHERE date(timestamp) = date('now')").fetchone()
                unreviewed = conn.execute("SELECT COUNT(*) AS c FROM violations WHERE status = 'unreviewed'").fetchone()
                students   = conn.execute("SELECT COUNT(*) AS c FROM students").fetchone()
                last       = conn.execute("SELECT MAX(timestamp) AS ts FROM violations").fetchone()

            if self._stat_today_value:
                self._stat_today_value.configure(text=str(int(today["c"] if today else 0)))
            if self._stat_unreviewed_value:
                self._stat_unreviewed_value.configure(text=str(int(unreviewed["c"] if unreviewed else 0)))
            if self._stat_students_value:
                self._stat_students_value.configure(text=str(int(students["c"] if students else 0)))
            if self._stat_last_value:
                ts = (last["ts"] if last else None) or ""
                self._stat_last_value.configure(
                    text=str(ts)[11:19] if len(str(ts)) >= 19 else (str(ts) or "—")
                )
        except Exception:
            pass
        self._stats_job = self.after(10_000, self._refresh_stats)

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def _clear_alerts(self) -> None:
        self._alerts.clear()
        self._face_presence.clear()
        self._refresh_alerts_ui()

    def _refresh_alerts_ui(self) -> None:
        for child in self._alerts_scroll.winfo_children():
            child.destroy()

        if not self._alerts:
            ctk.CTkLabel(
                self._alerts_scroll, text="No alerts yet",
                font=body_font(12), text_color=COLOR_TEXT_MUTED,
            ).pack(pady=20)
            return

        for entry in self._alerts:
            matched = entry["matched"]
            dot_color = COLOR_SAFE if matched else COLOR_DANGER
            status_text = "Identified ✓" if matched else "Unidentified ✗"
            status_color = COLOR_SAFE if matched else COLOR_DANGER

            card = ctk.CTkFrame(
                self._alerts_scroll,
                fg_color=COLOR_SURFACE,
                corner_radius=CORNER_RADIUS,
                border_width=1,
                border_color=COLOR_BORDER,
            )
            card.pack(fill="x", pady=(0, 6))

            # Header row: dot + name + time
            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=10, pady=(8, 2))

            ctk.CTkLabel(
                header, text="●", font=body_font(14), text_color=dot_color,
            ).pack(side="left", padx=(0, 6))
            ctk.CTkLabel(
                header, text=entry["name"], font=body_font(13),
                text_color=COLOR_TEXT, anchor="w",
            ).pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(
                header, text=entry["time"], font=body_small_font(),
                text_color=COLOR_TEXT_MUTED,
            ).pack(side="right")

            # Details row: student ID + gender
            details = ctk.CTkFrame(card, fg_color="transparent")
            details.pack(fill="x", padx=10, pady=(0, 4))
            ctk.CTkLabel(
                details,
                text=f"ID: {entry['student_id']}   Gender: {entry['gender']}",
                font=body_small_font(), text_color=COLOR_TEXT_MUTED, anchor="w",
            ).pack(side="left")

            # Status chip
            ctk.CTkLabel(
                card, text=status_text, font=body_small_font(),
                text_color=status_color, anchor="w",
            ).pack(anchor="w", padx=10, pady=(0, 8))

        # Scroll to newest (top)
        try:
            self._alerts_scroll._parent_canvas.yview_moveto(0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _logout(self) -> None:
        self._on_close()
        sys.exit(0)

    def _on_close(self) -> None:
        for job_attr in ("_feed_job", "_clock_job", "_stats_job"):
            job = getattr(self, job_attr, None)
            if job:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, job_attr, None)
        self._stop_camera()
        self.camera_feed.cleanup()
        self.destroy()


def open_dashboard(username: str = "admin") -> None:
    app = CBVMSDashboard(username=username)
    app.mainloop()
