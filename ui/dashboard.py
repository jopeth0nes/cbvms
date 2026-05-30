"""Main CBVMS dashboard with live camera feed."""

from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from core.camera import CameraCapture
from core.detection_state import (
    DETECTION_COOLDOWN_SECONDS,
    AlertEmit,
    DetectionStateTracker,
)
from core.detector import Detector
from core.live_pipeline import LiveDetectionWorker
from core.recognizer import FACE_RECOGNITION_AVAILABLE, Recognizer
from core.torso_detection import draw_detections
from core.violation_engine import ViolationEngine
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

# BGR for OpenCV drawing
BGR_SAFE = (16, 185, 129)
BGR_WARNING = (11, 158, 245)
BGR_DANGER = (68, 68, 239)

MAX_ALERTS = 50
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
FEED_FPS = 15  # Reduced from default to prevent image garbage collection issues


def _make_no_camera_frame(width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), (23, 17, 15), dtype=np.uint8)
    text = "No Camera"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv2.putText(frame, text, (x, y), font, scale, (150, 150, 150), thickness, cv2.LINE_AA)
    return frame


class CBVMSDashboard(ctk.CTk):
    def __init__(self, username: str = "admin") -> None:
        super().__init__()
        self.username = username
        self._active_nav = "live"
        self._alerts: deque[dict] = deque(maxlen=MAX_ALERTS)
        self._detection_tracker = DetectionStateTracker()
        self._feed_job: str | None = None
        self._clock_job: str | None = None
        self._fps_times: deque[float] = deque(maxlen=30)
        self._last_person_count = 0
        self._person_counter = 0
        self._last_detections: list[dict] = []
        self._overlay_detections: list[dict] = []
        self._overlay_lock = threading.Lock()
        self._pipeline_frame_counter = 0
        self._last_person_count = 0
        self._person_counter = 0
        self._last_detections: list[dict] = []
        self._overlay_detections: list[dict] = []
        self._overlay_lock = threading.Lock()
        self._pipeline_frame_counter = 0

        self._database = CBVMSDatabase()
        self._database.initialize()
        self._recognizer = Recognizer(self._database)
        self._violation_engine = ViolationEngine()
        self._live_worker = LiveDetectionWorker(
            get_detector=lambda: self._detector,
            recognizer=self._recognizer,
            violation_engine=self._violation_engine,
            on_results=self._on_pipeline_results,
            get_detection_tracker=lambda: self._detection_tracker,
        )

        self._camera: CameraCapture | None = None
        self._detector: Detector | None = None
        self._detector_loading = False
        self._camera_retry_count = 0
        self._enrollment_panel: EnrollmentPanel | None = None
        self._violation_panel: ViolationLogPanel | None = None

        self._stats_job: str | None = None
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

        # Runtime settings (used by Settings panel).
        self._camera_index_setting = 0
        self._camera_source_url: str | None = None
        self._camera_resolution_setting = (1280, 720)
        self._fps_cap_setting = 30
        self._load_camera_preference()
        self._feed_interval_ms = max(50, int(1000 / FEED_FPS))  # Use slower frame rate for display

        self._build_ui()
        self._build_menubar()
        self._tick_clock()
        self._schedule_feed_update()
        # Defer camera until window is mapped (macOS permissions / AVFoundation).
        self.after(300, self._deferred_start_camera)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=220)   # left sidebar — fixed
        self.grid_columnconfigure(1, weight=1)                 # center — expands
        self.grid_columnconfigure(2, weight=0, minsize=280)   # right sidebar — fixed
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
            sidebar,
            text="CBVMS",
            font=heading_font(26),
            text_color=COLOR_ACCENT,
        ).pack(anchor="w", padx=PADDING, pady=(PADDING_LG, 0))

        ctk.CTkLabel(
            sidebar,
            text="Vision Monitoring System",
            font=body_small_font(),
            text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", padx=PADDING, pady=(0, PADDING_LG))

        self._face_warning_badge = ctk.CTkLabel(
            sidebar,
            text="Face Recognition Unavailable",
            font=body_small_font(),
            text_color=COLOR_TEXT,
            fg_color=COLOR_WARNING,
            corner_radius=999,
            padx=10,
            pady=4,
        )
        if not FACE_RECOGNITION_AVAILABLE:
            self._face_warning_badge.pack(anchor="w", padx=PADDING, pady=(0, 10))

        nav_items = [
            ("live", "📹  Live Monitor"),
            ("enrollment", "👤  Student Enrollment"),
            ("violations", "⚠  Violation Log"),
            ("settings", "⚙  Settings"),
        ]
        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, label in nav_items:
            btn = ctk.CTkButton(
                sidebar,
                text=label,
                anchor="w",
                height=40,
                corner_radius=CORNER_RADIUS,
                fg_color=COLOR_ACCENT if key == "live" else "transparent",
                hover_color=COLOR_ACCENT_HOVER if key == "live" else COLOR_BORDER,
                text_color=COLOR_TEXT,
                font=body_small_font(),
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
            footer,
            text=f"Logged in as {self.username}",
            font=body_small_font(),
            text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 8))

        ctk.CTkButton(
            footer,
            text="Logout",
            height=36,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_DANGER,
            command=self._logout,
        ).pack(fill="x")

    def _build_center_panel(self) -> None:
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=PADDING, pady=PADDING)
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(1, weight=1)

        title_row = ctk.CTkFrame(center, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, PADDING))

        self._center_title = ctk.CTkLabel(
            title_row,
            text="Live Monitor",
            font=panel_title_font(),
            text_color=COLOR_TEXT,
        )
        self._center_title.pack(side="left")

        self._datetime_label = ctk.CTkLabel(
            title_row,
            text="",
            font=body_small_font(),
            text_color=COLOR_TEXT_MUTED,
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
        self._live_frame.grid_rowconfigure(0, weight=0)   # stat cards row — fixed
        self._live_frame.grid_rowconfigure(1, weight=1)   # camera feed row — expands
        self._live_frame.grid_rowconfigure(2, weight=0)   # detection info panel — fixed
        self._live_frame.grid_rowconfigure(3, weight=0)   # status bar — fixed

        self._stats_row = self._build_stats_row(self._live_frame)
        self._stats_row.grid(row=0, column=0, sticky="ew", pady=(0, PADDING))

        # Camera Feed (using new CameraFeed component)
        self.camera_feed = CameraFeed(
            self._live_frame,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            bg_color=COLOR_BG
        )
        self.camera_feed.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Overlay for camera errors (non-blocking).
        self._no_camera_overlay = ctk.CTkFrame(
            self._live_frame,
            fg_color=COLOR_BG,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        ctk.CTkLabel(
            self._no_camera_overlay,
            text="No Camera Detected",
            font=heading_font(18),
            text_color=COLOR_TEXT,
        ).pack(pady=(24, 6))
        self._no_camera_reason = ctk.CTkLabel(
            self._no_camera_overlay,
            text="Check your camera connection and permissions.",
            font=body_small_font(),
            text_color=COLOR_TEXT_MUTED,
        )
        self._no_camera_reason.pack(pady=(0, 14))
        ctk.CTkButton(
            self._no_camera_overlay,
            text="Retry",
            height=34,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._deferred_start_camera,
        ).pack()
        self._no_camera_overlay.grid_remove()

        status_bar = ctk.CTkFrame(
            self._live_frame,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
            height=48,
        )
        self._detection_panel = self._build_detection_info_panel(self._live_frame)
        self._detection_panel.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        status_bar.grid(row=3, column=0, sticky="ew")
        status_bar.grid_propagate(False)

        self._status_camera = ctk.CTkLabel(
            status_bar,
            text="Camera: Starting…",
            font=body_small_font(),
            text_color=COLOR_TEXT_MUTED,
        )
        self._status_camera.pack(side="left", padx=PADDING, pady=12)

        self._status_persons = ctk.CTkLabel(
            status_bar,
            text="Persons: 0",
            font=body_small_font(),
            text_color=COLOR_TEXT,
        )
        self._status_persons.pack(side="left", padx=PADDING)

        self._camera_spinner = ctk.CTkProgressBar(status_bar, mode="indeterminate", width=140)
        self._camera_spinner.pack(side="right", padx=(0, 10), pady=14)
        self._camera_spinner.stop()
        self._camera_spinner.pack_forget()

        self._status_fps = ctk.CTkLabel(
            status_bar,
            text="FPS: —",
            font=body_small_font(),
            text_color=COLOR_ACCENT,
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
            violation_engine=self._violation_engine,
            username=self.username,
            get_detector_loaded=lambda: self._detector is not None,
            apply_camera_settings=self._apply_camera_settings,
            on_camera_source_connected=self._on_camera_source_connected,
        )
        self._settings_panel.grid_remove()

        self._views: dict[str, ctk.CTkFrame] = {
            "live": self._live_frame,
            "enrollment": self._enrollment_panel,
            "violations": self._violation_panel,
            "settings": self._settings_panel,
        }
        self._live_frame.grid(row=0, column=0, sticky="nsew")
        self._schedule_stats_refresh()

    def _build_stats_row(self, master: ctk.CTkFrame) -> ctk.CTkFrame:
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="stats")

        def _card(col: int, *, label: str, accent: str) -> ctk.CTkLabel:
            card = ctk.CTkFrame(
                row,
                fg_color=COLOR_SURFACE,
                corner_radius=CORNER_RADIUS,
                border_width=1,
                border_color=COLOR_BORDER,
                height=80,
            )
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 10, 0))
            card.grid_propagate(False)
            card.grid_columnconfigure(0, weight=1)

            value = ctk.CTkLabel(
                card,
                text="—",
                font=heading_font(22),
                text_color=accent,
            )
            value.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 0))

            ctk.CTkLabel(
                card,
                text=label,
                font=body_font(12),
                text_color=COLOR_TEXT_MUTED,
            ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 12))

            return value

        self._stat_today_value = _card(0, label="Total Violations Today", accent=COLOR_DANGER)
        self._stat_unreviewed_value = _card(1, label="Unreviewed", accent=COLOR_WARNING)
        self._stat_students_value = _card(2, label="Students Enrolled", accent=COLOR_ACCENT)
        self._stat_last_value = _card(3, label="Last Detection", accent=COLOR_SAFE)
        return row

    def _refresh_stats(self) -> None:
        try:
            with self._database.connect() as conn:
                today = conn.execute(
                    "SELECT COUNT(*) AS c FROM violations WHERE date(timestamp) = date('now')"
                ).fetchone()
                unreviewed = conn.execute(
                    "SELECT COUNT(*) AS c FROM violations WHERE status = 'unreviewed'"
                ).fetchone()
                students = conn.execute("SELECT COUNT(*) AS c FROM students").fetchone()
                last = conn.execute("SELECT MAX(timestamp) AS ts FROM violations").fetchone()

            today_c = int(today["c"] if today else 0)
            unreviewed_c = int(unreviewed["c"] if unreviewed else 0)
            students_c = int(students["c"] if students else 0)
            last_ts = (last["ts"] if last else None) or ""

            last_text = "—"
            if last_ts:
                last_text = str(last_ts)[11:19] if len(str(last_ts)) >= 19 else str(last_ts)

            if self._stat_today_value is not None:
                self._stat_today_value.configure(text=str(today_c))
            if self._stat_unreviewed_value is not None:
                self._stat_unreviewed_value.configure(text=str(unreviewed_c))
            if self._stat_students_value is not None:
                self._stat_students_value.configure(text=str(students_c))
            if self._stat_last_value is not None:
                self._stat_last_value.configure(text=last_text)
        except Exception:
            # Keep UI resilient; stats are non-critical.
            pass

        self._stats_job = self.after(10_000, self._refresh_stats)

    def _schedule_stats_refresh(self) -> None:
        if self._stats_job:
            try:
                self.after_cancel(self._stats_job)
            except Exception:
                pass
            self._stats_job = None
        self._refresh_stats()

    def _build_right_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(
            self,
            width=SIDEBAR_RIGHT_WIDTH,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        sidebar.grid(row=0, column=2, rowspan=4, sticky="nsew", padx=(0, 10), pady=10)
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            sidebar,
            text="Live Alerts",
            font=heading_font(18),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=PADDING, pady=(PADDING, PADDING))

        self._alerts_scroll = ctk.CTkScrollableFrame(
            sidebar,
            fg_color=COLOR_BG,
            corner_radius=CORNER_RADIUS,
        )
        self._alerts_scroll.grid(row=1, column=0, sticky="nsew", padx=PADDING, pady=(0, PADDING))

        ctk.CTkButton(
            sidebar,
            text="Clear Alerts",
            height=36,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
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
            win,
            text="Computer Based Vision Monitoring System (CBVMS)",
            font=panel_title_font(),
            text_color=COLOR_TEXT,
            wraplength=420,
            justify="center",
        ).pack(padx=PADDING, pady=(PADDING, 10))

        ctk.CTkLabel(
            win,
            text=f"Version {APP_VERSION}\n{APP_COLLEGE_NAME}",
            font=body_small_font(),
            text_color=COLOR_TEXT_MUTED,
            justify="center",
        ).pack(padx=PADDING, pady=(0, 14))

        ctk.CTkButton(
            win,
            text="Close",
            height=34,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
            command=win.destroy,
        ).pack(pady=(0, PADDING))

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
            "live": "Live Monitor",
            "enrollment": "Student Enrollment",
            "violations": "Violation Log",
            "settings": "Settings",
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
        """Fade out -> switch -> fade in using window alpha steps."""
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

    def _apply_camera_settings(self, camera_index: int, resolution: tuple[int, int], fps_cap: int) -> None:
        if self._camera_source_url is None:
            self._camera_index_setting = int(camera_index)
        self._camera_resolution_setting = (int(resolution[0]), int(resolution[1]))
        self._fps_cap_setting = max(10, min(60, int(fps_cap)))
        self._feed_interval_ms = max(16, int(1000 / float(self._fps_cap_setting)))

        # Restart camera immediately.
        self._camera_retry_count = 0
        self._deferred_start_camera()
        show_toast(self, "Restarting camera…", type="info", duration=1800)

    def _get_camera_frame(self):
        if self._camera and self._camera.is_open:
            frame = self._camera.get_latest_frame()
            if frame is not None:
                return frame
            return self._camera.read()
        return None

    def _pump_camera_frame(self) -> np.ndarray | None:
        """Single reader for the shared camera (required on macOS)."""
        if not self._camera or not self._camera.is_open:
            return None
        frame = self._camera.read()
        if frame is not None:
            return frame
        return self._camera.get_latest_frame()

    def _update_camera_status(self, frame: np.ndarray | None) -> None:
        if not self._camera or not self._camera.is_open:
            try:
                self._no_camera_overlay.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
                self._no_camera_overlay.tkraise()
                # Stop camera feed updates when no camera
                self.camera_feed.stop_updates()
            except Exception:
                pass
            try:
                if self._camera_spinner.winfo_ismapped():
                    self._camera_spinner.stop()
                    self._camera_spinner.pack_forget()
            except Exception:
                pass
            if self._camera_retry_count > 0 and self._camera_retry_count < 8:
                self._status_camera.configure(
                    text="Camera: Starting…",
                    text_color=COLOR_TEXT_MUTED,
                )
            else:
                self._status_camera.configure(text="Camera: No Camera", text_color=COLOR_DANGER)
            return
        else:
            try:
                self._no_camera_overlay.grid_remove()
                # Ensure camera feed is visible and running
                self.camera_feed.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
                if not self.camera_feed._is_running:
                    self.camera_feed.start_updates(self._feed_interval_ms)
            except Exception:
                pass
        if frame is not None:
            self._status_camera.configure(text="Camera: Active", text_color=COLOR_SAFE)
            try:
                if self._camera_spinner.winfo_ismapped():
                    self._camera_spinner.stop()
                    self._camera_spinner.pack_forget()
            except Exception:
                pass
        else:
            self._status_camera.configure(
                text="Camera: Warming up…",
                text_color=COLOR_WARNING,
            )

    def _deferred_start_camera(self) -> None:
        self.update_idletasks()
        self._start_camera()
        self._start_detector_async()

    def _start_camera(self) -> None:
        self._stop_camera()
        try:
            self._camera_spinner.pack(side="right", padx=(0, 10), pady=14)
            self._camera_spinner.start()
        except Exception:
            pass
        
        camera_error = None
        if self._camera_source_url:
            self._camera = CameraCapture(
                source_url=self._camera_source_url,
                width=self._camera_resolution_setting[0],
                height=self._camera_resolution_setting[1],
                fps_cap=self._fps_cap_setting,
            )
        else:
            self._camera = CameraCapture(
                camera_index=self._camera_index_setting,
                width=self._camera_resolution_setting[0],
                height=self._camera_resolution_setting[1],
                fps_cap=self._fps_cap_setting,
            )
        
        if self._camera.open():
            self._pipeline_frame_counter = 0
            self._live_worker.start()
            self._status_camera.configure(text="Camera Active", text_color=COLOR_SAFE)
            return

        camera_error = self._camera.last_error if self._camera else "Unknown error"
        print(f"[CBVMS] Camera initialization failed: {camera_error}")
        self._status_camera.configure(text=f"No Camera: {camera_error}", text_color=COLOR_DANGER)
        
        if self._camera_retry_count == 0:
            show_toast(self, f"Camera error: {camera_error}", type="error", duration=4000)
        
        if self._camera_retry_count < 8:
            self._camera_retry_count += 1
            self.after(1500, self._deferred_start_camera)

    def _start_detector_async(self) -> None:
        if self._detector is not None or self._detector_loading:
            return
        self._detector_loading = True
        self._status_persons.configure(text="Persons: loading model…")

        def _load() -> None:
            try:
                detector = Detector()
                err = None
            except FileNotFoundError as exc:
                detector = None
                err = f"Model not found: {exc}"
            except RuntimeError as exc:
                detector = None
                err = f"Model load error: {exc}"
            except Exception as exc:
                detector = None
                err = f"Unexpected error: {exc}"

            def _apply() -> None:
                self._detector_loading = False
                if detector is not None:
                    self._detector = detector
                    self._status_persons.configure(text="Persons: 0")
                else:
                    self._detector = None
                    self._status_persons.configure(text="Persons: detector offline")
                    if err:
                        print(f"[CBVMS] Detector initialization failed: {err}")
                        show_toast(self, f"Detector error: {err}", type="error", duration=5000)
                    if self._camera and self._camera.is_open:
                        self._status_camera.configure(
                            text=f"Camera OK — Detector offline",
                            text_color=COLOR_WARNING,
                        )

            if self.winfo_exists():
                self.after(0, _apply)

        threading.Thread(target=_load, daemon=True).start()

    def _tick_clock(self) -> None:
        self._datetime_label.configure(
            text=datetime.now().strftime("%A, %d %b %Y  %H:%M:%S")
        )
        self._clock_job = self.after(1000, self._tick_clock)

    def _schedule_feed_update(self) -> None:
        self._feed_job = self.after(self._feed_interval_ms, self._update_feed)

    def _process_detection_pipeline(self, frame: np.ndarray) -> None:
        """Process frame through detection pipeline (separate from display)."""
        try:
            self._pipeline_frame_counter += 1
            
            # Only offer frame to worker if detector is available
            if self._detector is not None:
                self._live_worker.offer_frame(frame, self._pipeline_frame_counter)
            else:
                # Update detection info without detections
                self._refresh_detection_info_panel([])
        except Exception as exc:
            print(f"[CBVMS] Error in detection pipeline: {exc}")

    def _update_feed(self) -> None:
        if not self.winfo_exists():
            return

        try:
            frame = self._pump_camera_frame()
            self._update_camera_status(frame)

            if self._active_nav == "live":
                # Update the camera feed with the new frame
                if frame is not None:
                    self.camera_feed.update_frame(frame)
                    self._process_detection_pipeline(frame)
                else:
                    # Let the camera feed handle the placeholder
                    pass
                    
                # Start camera feed updates if not running
                if not self.camera_feed._is_running:
                    self.camera_feed.start_updates(self._feed_interval_ms)
                    
            elif self._active_nav == "enrollment" and self._enrollment_panel is not None:
                self._enrollment_panel.update_preview(frame)
        except tk.TclError as exc:
            print(f"[CBVMS] TclError in feed update: {exc}")
            self._status_camera.configure(
                text=f"Display error: {exc}",
                text_color=COLOR_DANGER,
            )
        except Exception as exc:
            print(f"[CBVMS] Exception in feed update: {exc}")
            self._status_camera.configure(
                text=f"Feed error: {exc}",
                text_color=COLOR_DANGER,
            )

        self._feed_job = self.after(self._feed_interval_ms, self._update_feed)

    def _on_pipeline_results(self, enriched: list[dict], person_count: int) -> None:
        def _apply() -> None:
            if not self.winfo_exists():
                return
            with self._overlay_lock:
                self._overlay_detections = enriched
                self._last_detections = enriched
            self._last_person_count = person_count
            self._status_persons.configure(text=f"Persons: {person_count}")
            self._refresh_detection_info_panel(enriched)
            self._apply_detection_alerts(enriched)

        if self.winfo_exists():
            self.after(0, _apply)

    def _draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        if not detections:
            return frame
        return draw_detections(frame, detections)

    def _build_detection_info_panel(self, master: ctk.CTkFrame) -> ctk.CTkFrame:
        panel = ctk.CTkFrame(
            master,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        panel.grid_columnconfigure((0, 1), weight=1, uniform="zones")

        self._face_zone_frame = self._make_zone_column(panel, 0, title="FACE DETECTION")
        self._torso_zone_frame = self._make_zone_column(panel, 1, title="TORSO DETECTION")
        self._refresh_detection_info_panel([])
        return panel

    def _make_zone_column(self, master: ctk.CTkFrame, col: int, *, title: str) -> ctk.CTkFrame:
        col_frame = ctk.CTkFrame(master, fg_color="transparent")
        pad = (PADDING, PADDING // 2) if col == 0 else (PADDING // 2, PADDING)
        col_frame.grid(row=0, column=col, sticky="nsew", padx=pad, pady=PADDING)

        ctk.CTkLabel(
            col_frame,
            text=title,
            font=body_font(11),
            text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 6))

        body = ctk.CTkFrame(col_frame, fg_color="transparent")
        body.pack(fill="both", expand=True)
        return body

    @staticmethod
    def _format_violation_label(code: str) -> str:
        labels = {
            "earrings_male": "Earrings (male)",
            "dress_code": "Dress Code",
            "no_id_badge": "No ID Badge",
            "prohibited_items": "Prohibited Items",
        }
        return labels.get(code, code.replace("_", " ").title())

    def _refresh_detection_info_panel(self, detections: list[dict]) -> None:
        if not hasattr(self, "_face_zone_frame"):
            return

        for frame in (self._face_zone_frame, self._torso_zone_frame):
            for child in frame.winfo_children():
                child.destroy()

        primary = detections[0] if detections else None
        if primary is None:
            ctk.CTkLabel(
                self._face_zone_frame,
                text="No person in frame",
                font=body_small_font(),
                text_color=COLOR_TEXT_MUTED,
            ).pack(anchor="w")
            ctk.CTkLabel(
                self._torso_zone_frame,
                text="—",
                font=body_small_font(),
                text_color=COLOR_TEXT_MUTED,
            ).pack(anchor="w")
            return

        identity = primary.get("identity") or {}
        name = identity.get("name") or primary.get("name") or "Unknown"
        student_id = identity.get("id") or primary.get("student_id") or "—"
        year_level = primary.get("year_level") or ""
        course = primary.get("course") or ""
        grade_line = " — ".join(p for p in (year_level, course) if p) or "—"

        ctk.CTkLabel(
            self._face_zone_frame,
            text=f"● {name}",
            font=body_font(13),
            text_color="#3B82F6",
        ).pack(anchor="w")
        ctk.CTkLabel(
            self._face_zone_frame,
            text=grade_line,
            font=body_small_font(),
            text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))
        ctk.CTkLabel(
            self._face_zone_frame,
            text=f"ID: {student_id}",
            font=body_small_font(),
            text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 8))

        ctk.CTkLabel(
            self._face_zone_frame,
            text="Face Violations:",
            font=body_font(11),
            text_color=COLOR_TEXT,
        ).pack(anchor="w", pady=(4, 4))
        face_v = primary.get("face_violations") or []
        self._pack_violation_chips(self._face_zone_frame, face_v)

        ctk.CTkLabel(
            self._torso_zone_frame,
            text="● Upper Body Detected",
            font=body_font(13),
            text_color="#F97316",
        ).pack(anchor="w", pady=(0, 8))
        ctk.CTkLabel(
            self._torso_zone_frame,
            text="Violations:",
            font=body_font(11),
            text_color=COLOR_TEXT,
        ).pack(anchor="w", pady=(4, 4))
        torso_v = primary.get("torso_violations") or []
        self._pack_violation_chips(self._torso_zone_frame, torso_v)

    def _pack_violation_chips(self, parent: ctk.CTkFrame, violations: list[str]) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(anchor="w", fill="x")
        if not violations:
            ctk.CTkLabel(
                row,
                text="✓ No violations",
                font=body_small_font(),
                text_color=COLOR_SAFE,
                fg_color="transparent",
                corner_radius=8,
            ).pack(anchor="w", padx=0, pady=2)
            return
        for code in violations:
            ctk.CTkLabel(
                row,
                text=f"✗ {self._format_violation_label(code)}",
                font=body_small_font(),
                text_color=COLOR_DANGER,
                fg_color="transparent",
                corner_radius=8,
            ).pack(anchor="w", padx=0, pady=2)

    def _apply_detection_alerts(self, detections: list[dict]) -> None:
        now = time.time()
        events = self._detection_tracker.process_frame(detections, now)
        self._sync_inactive_alert_cards()

        ui_dirty = False
        db_logged = False
        for event in events:
            if event.kind == "new":
                if self._emit_new_alert(event, now):
                    ui_dirty = True
            elif event.kind == "update":
                if self._update_alert_card(event, now):
                    ui_dirty = True
            if self._log_violations_for_event(event):
                db_logged = True

        if ui_dirty:
            self._refresh_alerts_ui()
        if db_logged:
            self._schedule_stats_refresh()

    def _sync_inactive_alert_cards(self) -> None:
        for identity_key, state in self._detection_tracker.detection_state.items():
            if state.active:
                continue
            for entry in self._alerts:
                if entry.get("identity_key") == identity_key and entry.get("active"):
                    entry["active"] = False

    def _log_violations_for_event(self, event: AlertEmit) -> bool:
        if not event.violations_to_log:
            return False
        logged = False
        for violation_type in event.violations_to_log:
            if not str(violation_type).strip():
                continue
            try:
                self._database.log_violation(
                    student_id=event.student_id,
                    student_name=event.name,
                    violation_type=violation_type,
                    snapshot_jpeg=event.snapshot_jpeg,
                )
                logged = True
            except Exception:
                pass
        return logged

    def _is_duplicate_alert(self, identity_key: str, now: float) -> bool:
        cutoff = now - DETECTION_COOLDOWN_SECONDS
        for entry in self._alerts:
            if entry.get("identity_key") != identity_key:
                continue
            entry_time = entry.get("epoch_time", 0.0)
            if entry_time >= cutoff:
                return True
        return False

    def _find_active_alert_card(self, identity_key: str) -> dict | None:
        for entry in self._alerts:
            if entry.get("identity_key") == identity_key and entry.get("active"):
                return entry
        return None

    @staticmethod
    def _alert_severity_rank(dot_color: str) -> int:
        if dot_color == COLOR_DANGER:
            return 2
        if dot_color == COLOR_WARNING:
            return 1
        return 0

    def _emit_new_alert(self, event: AlertEmit, now: float) -> bool:
        if self._is_duplicate_alert(event.identity_key, now):
            return False
        self._push_alert(
            event.dot_color,
            event.name,
            event.violation_text,
            identity_key=event.identity_key,
            active=True,
            epoch_time=now,
            face_violations=event.face_violations,
            torso_violations=event.torso_violations,
            refresh=False,
        )
        return True

    def _update_alert_card(self, event: AlertEmit, now: float) -> bool:
        card = self._find_active_alert_card(event.identity_key)
        if card is None:
            return self._emit_new_alert(event, now)

        if self._alert_severity_rank(event.dot_color) > self._alert_severity_rank(
            card["dot_color"]
        ):
            card["dot_color"] = event.dot_color
        card["violation_text"] = event.violation_text
        card["face_violations"] = event.face_violations
        card["torso_violations"] = event.torso_violations
        if event.name != "Unknown":
            card["name"] = event.name
        card["time"] = datetime.now().strftime("%H:%M:%S")
        card["epoch_time"] = now
        return True

    def _push_alert(
        self,
        dot_color: str,
        name: str,
        violation_text: str,
        *,
        identity_key: str,
        active: bool = True,
        epoch_time: float | None = None,
        face_violations: list[str] | None = None,
        torso_violations: list[str] | None = None,
        refresh: bool = True,
    ) -> None:
        now = epoch_time if epoch_time is not None else time.time()
        if self._is_duplicate_alert(identity_key, now):
            return

        entry = {
            "dot_color": dot_color,
            "name": name,
            "violation_text": violation_text,
            "face_violations": list(face_violations or []),
            "torso_violations": list(torso_violations or []),
            "time": datetime.now().strftime("%H:%M:%S"),
            "epoch_time": now,
            "identity_key": identity_key,
            "active": active,
        }
        self._alerts.appendleft(entry)
        if refresh:
            self._refresh_alerts_ui()

    def _refresh_alerts_ui(self) -> None:
        for child in self._alerts_scroll.winfo_children():
            child.destroy()

        if not self._alerts:
            ctk.CTkLabel(
                self._alerts_scroll,
                text="No alerts yet",
                font=body_font(12),
                text_color=COLOR_TEXT_MUTED,
            ).pack(pady=20)
            return

        for entry in self._alerts:
            card = ctk.CTkFrame(
                self._alerts_scroll,
                fg_color=COLOR_SURFACE,
                corner_radius=CORNER_RADIUS,
                border_width=1,
                border_color=COLOR_BORDER,
            )
            card.pack(fill="x", pady=4)

            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=8)

            ctk.CTkLabel(row, text="●", font=body_font(14), text_color=entry["dot_color"]).pack(
                side="left", padx=(0, 8)
            )

            text_col = ctk.CTkFrame(row, fg_color="transparent")
            text_col.pack(side="left", fill="x", expand=True)

            ctk.CTkLabel(
                text_col,
                text=entry["name"],
                font=body_font(13),
                text_color=COLOR_TEXT,
                anchor="w",
            ).pack(fill="x")
            pills_row = ctk.CTkFrame(text_col, fg_color="transparent")
            pills_row.pack(fill="x", pady=(4, 0))
            face_v = entry.get("face_violations") or []
            torso_v = entry.get("torso_violations") or []
            if face_v or torso_v:
                for code in face_v:
                    ctk.CTkLabel(
                        pills_row,
                        text=f"[face] {code}",
                        font=body_font(10),
                        text_color="#3B82F6",
                        fg_color="#3B82F622",
                        corner_radius=6,
                    ).pack(side="left", padx=(0, 4), pady=2)
                for code in torso_v:
                    ctk.CTkLabel(
                        pills_row,
                        text=f"[torso] {code}",
                        font=body_font(10),
                        text_color="#F97316",
                        fg_color="#F9731622",
                        corner_radius=6,
                    ).pack(side="left", padx=(0, 4), pady=2)
            else:
                ctk.CTkLabel(
                    pills_row,
                    text=entry.get("violation_text", ""),
                    font=body_font(11),
                    text_color=COLOR_TEXT_MUTED,
                    anchor="w",
                ).pack(fill="x")

            ctk.CTkLabel(
                row,
                text=entry["time"],
                font=body_font(11),
                text_color=COLOR_TEXT_MUTED,
            ).pack(side="right")

        # Keep the newest-first list scrolled to the top.
        try:
            self._alerts_scroll._parent_canvas.yview_moveto(0)
        except Exception:
            pass

    def _clear_alerts(self) -> None:
        self._alerts.clear()
        self._detection_tracker.reset()
        self._live_worker.reset_caches()
        self._refresh_alerts_ui()

    def _cancel_jobs(self) -> None:
        if self._feed_job:
            self.after_cancel(self._feed_job)
            self._feed_job = None
        if self._clock_job:
            self.after_cancel(self._clock_job)
            self._clock_job = None
        if self._stats_job:
            try:
                self.after_cancel(self._stats_job)
            except Exception:
                pass
            self._stats_job = None

    def _stop_camera(self) -> None:
        self._live_worker.stop()
        with self._overlay_lock:
            self._overlay_detections = []
            self._last_detections = []
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def _logout(self) -> None:
        self._on_close()
        sys.exit(0)

    def _on_close(self) -> None:
        self._cancel_jobs()
        self._stop_camera()
        self._live_worker.stop()
        self.camera_feed.cleanup()
        self.destroy()


def open_dashboard(username: str = "admin") -> None:
    app = CBVMSDashboard(username=username)
    app.mainloop()
