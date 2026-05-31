"""Settings panel — Computer Based Vision Monitoring System (CBVMS)."""

from __future__ import annotations

import colorsys
from datetime import datetime
from pathlib import Path
from typing import Callable

from typing import TYPE_CHECKING

import customtkinter as ctk

from core.trainer import MODULES

if TYPE_CHECKING:
    from core.recognizer import Recognizer
    from core.trainer import ViolationTrainer
    from core.violation_engine import LiveViolationChecker, ViolationEngine

from database.db_manager import CBVMSDatabase
from ui.camera_configuration import CameraConfigurationSection
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
    ROW_STRIPE_EVEN,
    body_small_font,
    panel_title_font,
    section_title_font,
    show_toast,
)


def _hex_from_hsv(h: int, s: int, v: int) -> str:
    # OpenCV HSV uses H:0-179, S/V:0-255. Convert to 0..1 for colorsys.
    hf = max(0.0, min(1.0, h / 179.0 if 179 else 0.0))
    sf = max(0.0, min(1.0, s / 255.0 if 255 else 0.0))
    vf = max(0.0, min(1.0, v / 255.0 if 255 else 0.0))
    r, g, b = colorsys.hsv_to_rgb(hf, sf, vf)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


class SettingsPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        *,
        database: CBVMSDatabase,
        recognizer: "Recognizer | None",
        violation_engine: "ViolationEngine | None",
        username: str,
        get_detector_loaded: Callable[[], bool],
        apply_camera_settings: Callable[[int, tuple[int, int], int], None],
        on_camera_source_connected: Callable[[dict], None] | None = None,
        trainer: "ViolationTrainer | None" = None,
        checker: "LiveViolationChecker | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color=COLOR_BG, **kwargs)
        self.database = database
        self.recognizer = recognizer
        self.violation_engine = violation_engine
        self.trainer = trainer
        self.checker = checker
        self.username = username
        self.get_detector_loaded = get_detector_loaded
        self.apply_camera_settings = apply_camera_settings
        self.on_camera_source_connected = on_camera_source_connected or (lambda _p: None)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build()
        self._refresh_faces_loaded()
        self._refresh_model_status()

    def _build(self) -> None:
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            scroll,
            text="Settings",
            font=panel_title_font(),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=PADDING, pady=(PADDING, 10))

        row = 1
        row = self._camera_configuration_section(scroll, row=row)
        row = self._recognition_section(scroll, row=row)
        row = self._violation_section(scroll, row=row)
        row = self._model_status_section(scroll, row=row)
        row = self._admin_section(scroll, row=row)

    def _section_card(self, master, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            master,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        ctk.CTkLabel(card, text=title, font=section_title_font(), text_color=COLOR_TEXT).pack(
            anchor="w", padx=PADDING, pady=(PADDING, 8)
        )
        return card

    def _camera_configuration_section(self, master, *, row: int) -> int:
        section = CameraConfigurationSection(
            master,
            on_camera_connected=self.on_camera_source_connected,
            apply_stream_settings=self.apply_camera_settings,
        )
        section.grid(row=row, column=0, sticky="ew", padx=PADDING, pady=(0, 12))
        return row + 1

    def _recognition_section(self, master, *, row: int) -> int:
        card = self._section_card(master, "Recognition Settings")
        card.grid(row=row, column=0, sticky="ew", padx=PADDING, pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)

        self._tol_var = ctk.DoubleVar(value=float(getattr(self.recognizer, "tolerance", 0.5)))

        tol_row = ctk.CTkFrame(card, fg_color="transparent")
        tol_row.pack(fill="x", padx=PADDING, pady=(0, 10))
        tol_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(tol_row, text="Tolerance", font=body_small_font(), text_color=COLOR_TEXT_MUTED).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        self._tol_value = ctk.CTkLabel(
            tol_row, text=f"{self._tol_var.get():.2f}", font=body_small_font(), text_color=COLOR_TEXT
        )
        self._tol_value.grid(row=0, column=2, sticky="e")

        def _on_tol(v: float) -> None:
            stepped = round(float(v) / 0.05) * 0.05
            stepped = max(0.30, min(0.70, stepped))
            self._tol_var.set(stepped)
            self._tol_value.configure(text=f"{stepped:.2f}")
            try:
                self.recognizer.tolerance = float(stepped)
            except Exception:
                pass

        tol_slider = ctk.CTkSlider(
            tol_row,
            from_=0.30,
            to=0.70,
            number_of_steps=8,
            command=_on_tol,
        )
        tol_slider.set(float(self._tol_var.get() or 0.5))
        tol_slider.grid(row=0, column=1, sticky="ew")

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=PADDING, pady=(0, 6))

        ctk.CTkButton(
            btn_row,
            text="Reload Known Faces",
            height=34,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._reload_faces,
        ).pack(side="left")

        self._faces_loaded_label = ctk.CTkLabel(
            btn_row, text="0 faces loaded", font=body_small_font(), text_color=COLOR_TEXT_MUTED
        )
        self._faces_loaded_label.pack(side="right")

        return row + 1

    def _reload_faces(self) -> None:
        try:
            self.recognizer.load_known_faces()
        except Exception as exc:
            show_toast(self, f"Failed to reload faces: {exc}", type="error")
            return
        self._refresh_faces_loaded()
        show_toast(self, "Known faces reloaded.", type="success")

    def _refresh_faces_loaded(self) -> None:
        try:
            count = len(getattr(self.recognizer, "known_faces", []) or [])
        except Exception:
            count = 0
        if hasattr(self, "_faces_loaded_label"):
            self._faces_loaded_label.configure(text=f"{count} faces loaded")

    def _violation_section(self, master, *, row: int) -> int:
        card = self._section_card(master, "Violation Settings")
        card.grid(row=row, column=0, sticky="ew", padx=PADDING, pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)

        # HSV sliders (lower + upper)
        lower = getattr(self.violation_engine, "ALLOWED_HSV_LOWER", (0, 0, 0))
        upper = getattr(self.violation_engine, "ALLOWED_HSV_UPPER", (180, 140, 120))
        self._h_low = ctk.IntVar(value=int(lower[0]))
        self._s_low = ctk.IntVar(value=int(lower[1]))
        self._v_low = ctk.IntVar(value=int(lower[2]))
        self._h_up = ctk.IntVar(value=int(upper[0]))
        self._s_up = ctk.IntVar(value=int(upper[1]))
        self._v_up = ctk.IntVar(value=int(upper[2]))

        hair = ctk.CTkFrame(card, fg_color=ROW_STRIPE_EVEN, corner_radius=CORNER_RADIUS)
        hair.pack(fill="x", padx=PADDING, pady=(0, 10))
        hair.grid_columnconfigure(1, weight=1)

        top = ctk.CTkFrame(hair, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 6))
        top.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(top, text="Hair Color Rule (HSV)", font=body_small_font(), text_color=COLOR_TEXT).grid(
            row=0, column=0, sticky="w"
        )
        self._hair_swatch = ctk.CTkFrame(top, width=42, height=18, corner_radius=6, fg_color=COLOR_BORDER)
        self._hair_swatch.grid(row=0, column=2, sticky="e")

        def _slider_row(r: int, label: str, var: ctk.IntVar, frm: int, to: int) -> None:
            rowf = ctk.CTkFrame(hair, fg_color="transparent")
            rowf.grid(row=r, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
            rowf.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(rowf, text=label, font=body_small_font(), text_color=COLOR_TEXT_MUTED).grid(
                row=0, column=0, sticky="w", padx=(0, 10)
            )
            val = ctk.CTkLabel(rowf, text=str(var.get()), font=body_small_font(), text_color=COLOR_TEXT)
            val.grid(row=0, column=2, sticky="e")

            def _on(v: float) -> None:
                vv = int(round(float(v)))
                var.set(vv)
                val.configure(text=str(vv))
                self._apply_hair_hsv()

            s = ctk.CTkSlider(rowf, from_=frm, to=to, number_of_steps=int(to - frm), command=_on)
            s.set(var.get())
            s.grid(row=0, column=1, sticky="ew")

        _slider_row(1, "Lower H (0–179)", self._h_low, 0, 179)
        _slider_row(2, "Lower S (0–255)", self._s_low, 0, 255)
        _slider_row(3, "Upper S (0–255)", self._s_up, 0, 255)

        # Two sliders only requested; we map them to S/V bounds (dark hair heuristic).
        # Keep H and V fixed unless you want to expand later.
        # (We still expose H lower as an extra for fine tuning in the UI.)

        # Toggle switches
        switches = ctk.CTkFrame(card, fg_color="transparent")
        switches.pack(fill="x", padx=PADDING, pady=(0, PADDING))
        switches.grid_columnconfigure(0, weight=1)

        self._enable_hair = ctk.BooleanVar(value=True)
        self._enable_id = ctk.BooleanVar(value=True)
        self._enable_uniform = ctk.BooleanVar(value=True)
        self._enable_earring = ctk.BooleanVar(value=True)

        items = [
            ("Enable Hair Color Check", self._enable_hair, "hair"),
            ("Enable ID Badge Check", self._enable_id, "id_badge"),
            ("Enable Uniform Check", self._enable_uniform, "uniform"),
            ("Enable Earring Check", self._enable_earring, "earring"),
        ]
        for i, (label, var, key) in enumerate(items):
            sw = ctk.CTkSwitch(
                switches,
                text=label,
                variable=var,
                onvalue=True,
                offvalue=False,
                command=lambda k=key: self._apply_violation_toggles(k),
            )
            sw.grid(row=i, column=0, sticky="w", pady=4)

        self._apply_hair_hsv()
        self._apply_violation_toggles(None)
        return row + 1

    def _apply_hair_hsv(self) -> None:
        # Keep within sane bounds
        h_low = int(self._h_low.get())
        s_low = int(self._s_low.get())
        s_up = int(self._s_up.get())
        h_low = max(0, min(179, h_low))
        s_low = max(0, min(255, s_low))
        s_up = max(0, min(255, s_up))
        if s_low > s_up:
            s_low, s_up = s_up, s_low
            self._s_low.set(s_low)
            self._s_up.set(s_up)

        # ViolationEngine uses upper S/V thresholds; we map these sliders accordingly.
        try:
            self.violation_engine.ALLOWED_HSV_LOWER = (h_low, s_low, 0)
            self.violation_engine.ALLOWED_HSV_UPPER = (180, s_up, 120)
        except Exception:
            pass

        # Preview: show the "upper bound" color as a swatch.
        try:
            preview = _hex_from_hsv(h_low, s_up, 200)
            self._hair_swatch.configure(fg_color=preview)
        except Exception:
            pass

    def _apply_violation_toggles(self, _changed: str | None) -> None:
        # Uniform / earring map to the live checker (YOLOv8 classifiers).
        # Hair / ID badge have no live backend yet — left as UI-only switches.
        for attr, var in [
            ("check_uniform", self._enable_uniform),
            ("check_earring", self._enable_earring),
        ]:
            try:
                setattr(self.checker, attr, bool(var.get()))
            except Exception:
                pass

    _CLASSIFIER_TITLES = {"uniform": "Uniform Check", "earring": "Earring Check"}

    def _model_status_section(self, master, *, row: int) -> int:
        card = self._section_card(master, "Model Status")
        card.grid(row=row, column=0, sticky="ew", padx=PADDING, pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)

        # Person detector (YOLOv8) — single-line status
        det_row = ctk.CTkFrame(card, fg_color="transparent")
        det_row.pack(fill="x", padx=PADDING, pady=4)
        ctk.CTkLabel(
            det_row, text="Person Detector  ·  yolov8n.pt",
            font=body_small_font(), text_color=COLOR_TEXT,
        ).pack(side="left")
        self._detector_status_label = ctk.CTkLabel(
            det_row, text="—", font=body_small_font(), text_color=COLOR_TEXT_MUTED,
        )
        self._detector_status_label.pack(side="right")

        # Trainable classifiers (uniform / earring)
        self._cls_status: dict[str, dict] = {}
        for module in ("uniform", "earring"):
            title = self._CLASSIFIER_TITLES[module]
            model_file = Path(MODULES[module]["model_out"]).name

            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=PADDING, pady=(8, 0))
            ctk.CTkLabel(
                top, text=f"{title}  ·  {model_file}",
                font=body_small_font(), text_color=COLOR_TEXT,
            ).pack(side="left")
            badge = ctk.CTkLabel(
                top, text="—", font=body_small_font(), text_color=COLOR_TEXT_MUTED,
            )
            badge.pack(side="right")

            detail = ctk.CTkLabel(
                card, text="", font=body_small_font(), text_color=COLOR_TEXT_MUTED,
                anchor="w", justify="left",
            )
            detail.pack(fill="x", padx=PADDING, pady=(0, 2))

            self._cls_status[module] = {"badge": badge, "detail": detail}

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(fill="x", padx=PADDING, pady=(10, PADDING))
        ctk.CTkButton(
            btns,
            text="Training Guide",
            height=34,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._open_training_guide,
        ).pack(side="left")

        ctk.CTkButton(
            btns,
            text="Refresh Status",
            height=34,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._refresh_model_status,
        ).pack(side="right")

        return row + 1

    def _refresh_model_status(self) -> None:
        # Person detector
        yolov8 = Path(__file__).resolve().parents[1] / "models" / "yolov8n.pt"
        loaded = self.get_detector_loaded()
        self._detector_status_label.configure(
            text="Loaded" if loaded else ("Not found" if not yolov8.exists() else "Not loaded"),
            text_color=COLOR_SAFE if loaded else (COLOR_DANGER if not yolov8.exists() else COLOR_WARNING),
        )

        # Trainable classifiers
        for module, refs in getattr(self, "_cls_status", {}).items():
            if self.trainer is None:
                refs["badge"].configure(text="Unavailable", text_color=COLOR_TEXT_MUTED)
                refs["detail"].configure(text="")
                continue

            trained = self.trainer.is_trained(module)
            counts = self.trainer.get_sample_counts(module)
            refs["badge"].configure(
                text="Trained ✓" if trained else "Not trained",
                text_color=COLOR_SAFE if trained else COLOR_WARNING,
            )

            parts = "  ·  ".join(
                f"{label.replace('_', ' ')}: {count}" for label, count in counts.items()
            )
            if trained:
                mtime = self.trainer.model_mtime(module)
                if mtime:
                    parts += f"  ·  last trained {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M}"
            refs["detail"].configure(text=parts)

    def _open_training_guide(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("CBVMS — Model Training Guide")
        win.geometry("620x520")
        win.configure(fg_color=COLOR_BG)

        ctk.CTkLabel(
            win,
            text="CBVMS — Model Training Guide",
            font=panel_title_font(),
            text_color=COLOR_TEXT,
        ).pack(anchor="w", padx=PADDING, pady=(PADDING, 8))

        text = (
            "Train classifiers directly inside CBVMS — no scripts or manual files needed.\n\n"
            "1) Open the Training tab\n"
            "   - Click 🎓 Training in the left sidebar.\n"
            "   - Choose a tab: Uniform Check or Earring Check.\n\n"
            "2) Build the dataset (per class)\n"
            "   - UNIFORM: correct_uniform vs wrong_uniform.\n"
            "   - EARRING: no_earring vs with_earring.\n"
            "   - Use 'Upload Photos' or 'Capture from Camera' for each class.\n"
            "   - Minimum 10 photos per class (more = better accuracy).\n\n"
            "3) Train\n"
            "   - Click 'Train Now'. A YOLOv8 classifier trains on your photos.\n"
            "   - The model is saved automatically to the models/ folder\n"
            "     (uniform_cls.pt / earring_cls.pt).\n\n"
            "4) Check status\n"
            "   - Return here and click 'Refresh Status'.\n"
            "   - A trained model shows 'Trained ✓' with its sample counts\n"
            "     and last-trained time.\n"
        )
        box = ctk.CTkTextbox(
            win,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
            font=body_small_font(),
            text_color=COLOR_TEXT,
        )
        box.pack(fill="both", expand=True, padx=PADDING, pady=(0, PADDING))
        box.insert("1.0", text)
        box.configure(state="disabled")

    def _admin_section(self, master, *, row: int) -> int:
        card = self._section_card(master, "Admin")
        card.grid(row=row, column=0, sticky="ew", padx=PADDING, pady=(0, PADDING))

        form = ctk.CTkFrame(card, fg_color="transparent")
        form.pack(fill="x", padx=PADDING, pady=(0, 8))
        form.grid_columnconfigure(1, weight=1)

        self._pw_current = ctk.CTkEntry(form, show="•")
        self._pw_new = ctk.CTkEntry(form, show="•")
        self._pw_confirm = ctk.CTkEntry(form, show="•")

        for r, (label, entry) in enumerate(
            [
                ("Current Password", self._pw_current),
                ("New Password", self._pw_new),
                ("Confirm New Password", self._pw_confirm),
            ]
        ):
            ctk.CTkLabel(form, text=label, font=body_small_font(), text_color=COLOR_TEXT_MUTED).grid(
                row=r, column=0, sticky="w", padx=(0, 12), pady=6
            )
            entry.grid(row=r, column=1, sticky="ew", pady=6)

        self._pw_msg = ctk.CTkLabel(card, text="", font=body_small_font(), text_color=COLOR_TEXT_MUTED)
        self._pw_msg.pack(anchor="w", padx=PADDING, pady=(0, 8))

        ctk.CTkButton(
            card,
            text="Save Password",
            height=36,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._save_password,
        ).pack(anchor="e", padx=PADDING, pady=(0, PADDING))

        return row + 1

    def _save_password(self) -> None:
        current = self._pw_current.get().strip()
        new = self._pw_new.get().strip()
        confirm = self._pw_confirm.get().strip()

        if not current or not new or not confirm:
            self._pw_msg.configure(text="Please fill in all password fields.", text_color=COLOR_WARNING)
            return
        if len(new) < 6:
            self._pw_msg.configure(text="New password must be at least 6 characters.", text_color=COLOR_WARNING)
            return
        if new != confirm:
            self._pw_msg.configure(text="New password and confirmation do not match.", text_color=COLOR_WARNING)
            return
        if current == new:
            self._pw_msg.configure(text="New password must be different from current.", text_color=COLOR_WARNING)
            return

        try:
            ok = self.database.verify_user(self.username, current)
        except Exception as exc:
            self._pw_msg.configure(text=f"Database error: {exc}", text_color=COLOR_DANGER)
            return
        if not ok:
            self._pw_msg.configure(text="Current password is incorrect.", text_color=COLOR_DANGER)
            return

        try:
            # Update password hash directly (kept here to avoid API drift).
            from database.db_manager import hash_password

            with self.database.connect() as conn:
                conn.execute(
                    "UPDATE users SET password_hash = ? WHERE username = ?",
                    (hash_password(new), self.username),
                )
                conn.commit()
        except Exception as exc:
            self._pw_msg.configure(text=f"Failed to save password: {exc}", text_color=COLOR_DANGER)
            return

        self._pw_current.delete(0, "end")
        self._pw_new.delete(0, "end")
        self._pw_confirm.delete(0, "end")
        self._pw_msg.configure(text="Password updated successfully.", text_color=COLOR_SAFE)
        show_toast(self, "Password updated.", type="success")

