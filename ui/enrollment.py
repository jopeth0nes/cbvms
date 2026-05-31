"""Student enrollment panel for CBVMS."""

from __future__ import annotations

import pickle
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Callable

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

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
    CORNER_RADIUS,
    PADDING,
    body_font,
    heading_font,
)

if TYPE_CHECKING:
    from core.recognizer import Recognizer
    from database.db_manager import CBVMSDatabase

PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240


class EnrollmentPanel(ctk.CTkFrame):
    """Student list + selected-photo preview. Enrolling happens in a modal.

    All photo rendering uses ImageTk.PhotoImage (CTkImage does not display on
    this macOS/CustomTkinter build).
    """

    def __init__(
        self,
        master,
        database: CBVMSDatabase,
        recognizer: Recognizer,
        get_frame: Callable[[], np.ndarray | None],
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color=COLOR_BG, **kwargs)
        self.database = database
        self.recognizer = recognizer
        self.get_frame = get_frame

        self._students: list[dict] = []
        self._selected_pk: int | None = None

        # Modal-scoped widgets (created when a modal opens)
        self._entries: dict[str, ctk.CTkEntry] = {}
        self._gender_var: ctk.StringVar | None = None
        self._enroll_status_label: ctk.CTkLabel | None = None
        self._enroll_close: Callable[[], None] | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_preview_panel()
        self._reload_students()
        self.grid_remove()

    # ------------------------------------------------------------------
    # Left panel — student list
    # ------------------------------------------------------------------

    def _build_left_panel(self) -> None:
        left = ctk.CTkFrame(
            self,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, PADDING // 2))
        left.grid_rowconfigure(2, weight=1)
        left.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(left, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=PADDING, pady=(PADDING, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header, text="Enrolled Students", font=heading_font(16), text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            header,
            text="+ Enroll New Student",
            height=32,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._open_enroll_modal,
        ).grid(row=0, column=1, sticky="e")

        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._apply_filter())
        search = ctk.CTkEntry(
            left,
            placeholder_text="Search by name or student ID…",
            textvariable=self._search_var,
        )
        search.grid(row=1, column=0, sticky="ew", padx=PADDING, pady=(0, 8))

        tree_wrap = ctk.CTkFrame(left, fg_color=COLOR_BG, corner_radius=CORNER_RADIUS)
        tree_wrap.grid(row=2, column=0, sticky="nsew", padx=PADDING, pady=(0, 8))
        tree_wrap.grid_rowconfigure(0, weight=1)
        tree_wrap.grid_columnconfigure(0, weight=1)

        self._configure_tree_style()
        columns = ("name", "student_id", "course", "year_and_section", "gender", "enrolled_at")
        self._tree = ttk.Treeview(
            tree_wrap,
            columns=columns,
            show="headings",
            style="CBVMS.Treeview",
            selectmode="browse",
        )
        headings = {
            "name": "Name",
            "student_id": "Student ID",
            "course": "Course",
            "year_and_section": "Year & Section",
            "gender": "Gender",
            "enrolled_at": "Date Enrolled",
        }
        widths = {"name": 130, "student_id": 90, "course": 90,
                  "year_and_section": 100, "gender": 65, "enrolled_at": 100}
        for col in columns:
            self._tree.heading(col, text=headings[col])
            self._tree.column(col, width=widths[col], anchor="w")

        scroll_y = ttk.Scrollbar(tree_wrap, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll_y.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)

        footer = ctk.CTkFrame(left, fg_color="transparent")
        footer.grid(row=4, column=0, sticky="ew", padx=PADDING, pady=(0, PADDING))
        footer.grid_columnconfigure(0, weight=1)

        self._count_label = ctk.CTkLabel(
            footer, text="0 students enrolled", font=body_font(12), text_color=COLOR_TEXT_MUTED,
        )
        self._count_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        btn_row = ctk.CTkFrame(footer, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew")
        btn_row.grid_columnconfigure((0, 1, 2), weight=1, uniform="enroll_btns")

        ctk.CTkButton(
            btn_row, text="Reload", height=32, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER, hover_color=COLOR_ACCENT_HOVER, command=self._reload_students,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._update_btn = ctk.CTkButton(
            btn_row, text="Update Photo", height=32, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            command=self._update_selected_photo, state="disabled",
        )
        self._update_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="Delete Selected", height=32, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_DANGER, hover_color="#DC2626", command=self._delete_selected,
        ).grid(row=0, column=2, sticky="ew")

    # ------------------------------------------------------------------
    # Right panel — selected student photo preview
    # ------------------------------------------------------------------

    def _build_preview_panel(self) -> None:
        right = ctk.CTkFrame(
            self, fg_color=COLOR_SURFACE, corner_radius=CORNER_RADIUS,
            border_width=1, border_color=COLOR_BORDER,
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(PADDING // 2, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            right, text="Student Photo  ·  click to enlarge",
            font=heading_font(16), text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=PADDING, pady=(PADDING, 8))

        photo_wrap = ctk.CTkFrame(right, fg_color=COLOR_BG, corner_radius=CORNER_RADIUS)
        photo_wrap.grid(row=1, column=0, sticky="nsew", padx=PADDING, pady=(0, 8))
        photo_wrap.grid_rowconfigure(0, weight=1)
        photo_wrap.grid_columnconfigure(0, weight=1)

        # tk.Label (not CTkLabel): CTkLabel.configure(image=None) fails to clear a
        # raw ImageTk image, leaving a deleted student's photo on screen. tk.Label
        # clears reliably with image="".
        self._selected_photo_label = tk.Label(
            photo_wrap, text="Select a student to view photo",
            bg=COLOR_BG, fg=COLOR_TEXT_MUTED, cursor="hand2",
            font=("Helvetica", 13), bd=0,
        )
        self._selected_photo_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self._selected_photo_label.bind("<Button-1>", lambda _e: self._view_selected_photo())

        self._preview_caption = ctk.CTkLabel(
            right, text="", font=body_font(12), text_color=COLOR_TEXT_MUTED,
        )
        self._preview_caption.grid(row=2, column=0, sticky="w", padx=PADDING, pady=(0, 4))

        self._status_label = ctk.CTkLabel(
            right, text="", font=body_font(12), text_color=COLOR_TEXT_MUTED, wraplength=360,
        )
        self._status_label.grid(row=3, column=0, sticky="w", padx=PADDING, pady=(0, PADDING))

    def _configure_tree_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "CBVMS.Treeview",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            fieldbackground=COLOR_BG,
            bordercolor=COLOR_BORDER,
            rowheight=28,
        )
        style.configure(
            "CBVMS.Treeview.Heading",
            background=COLOR_SURFACE,
            foreground=COLOR_TEXT,
            relief="flat",
        )
        style.map(
            "CBVMS.Treeview",
            background=[("selected", COLOR_ACCENT)],
            foreground=[("selected", COLOR_TEXT)],
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str, *, success: bool = False, error: bool = False) -> None:
        color = COLOR_DANGER if error else COLOR_SAFE if success else COLOR_TEXT_MUTED
        self._status_label.configure(text=message, text_color=color)

    def _set_enroll_status(self, message: str, *, success: bool = False, error: bool = False) -> None:
        label = self._enroll_status_label
        if label is None or not label.winfo_exists():
            self._set_status(message, success=success, error=error)
            return
        color = COLOR_DANGER if error else COLOR_SAFE if success else COLOR_TEXT_MUTED
        label.configure(text=message, text_color=color)

    # ------------------------------------------------------------------
    # Data + list
    # ------------------------------------------------------------------

    def _reload_students(self) -> None:
        rows = self.database.get_all_students()
        self._students = [dict(row) for row in rows]
        self._apply_filter()
        self._count_label.configure(text=f"{len(self._students)} students enrolled")

    def _apply_filter(self) -> None:
        query = self._search_var.get().strip().lower()
        for item in self._tree.get_children():
            self._tree.delete(item)

        for student in self._students:
            name = (student.get("name") or "").lower()
            sid = (student.get("student_id") or "").lower()
            if query and query not in name and query not in sid:
                continue
            enrolled = student.get("enrolled_at") or ""
            if enrolled and "T" not in enrolled:
                enrolled = enrolled.replace(" ", " ")[:16]
            self._tree.insert(
                "", "end", iid=str(student["id"]),
                values=(
                    student.get("name", ""),
                    student.get("student_id", ""),
                    student.get("course", "") or "—",
                    student.get("year_and_section", "") or "—",
                    student.get("gender", "") or "—",
                    enrolled or "—",
                ),
            )

    def _on_row_select(self, _event: tk.Event | None = None) -> None:
        selection = self._tree.selection()
        if not selection:
            self._selected_pk = None
            self._update_btn.configure(state="disabled")
            self._clear_photo_label("Select a student to view photo")
            self._preview_caption.configure(text="")
            return

        self._selected_pk = int(selection[0])
        self._update_btn.configure(state="normal")
        student = self._resolve_selected_student()

        if student and student.get("photo"):
            self._show_photo_bytes(student["photo"], self._selected_photo_label)
        else:
            self._clear_photo_label("No photo on file")

        if student:
            self._preview_caption.configure(
                text=f"{student.get('name', '')}  ·  {student.get('student_id', '')}"
            )
        else:
            self._preview_caption.configure(text="")

    # ------------------------------------------------------------------
    # Multi-frame capture (async — camera cache refreshes between after() ticks)
    # ------------------------------------------------------------------

    def _collect_frames(self, on_done, *, count: int = 10, interval_ms: int = 100) -> None:
        """Collect `count` fresh camera frames `interval_ms` apart without blocking
        the UI loop, then call on_done(frames). A blocking sleep loop would freeze
        the event loop and return identical cached frames, so we chain after()."""
        frames: list = []

        def _grab(i: int = 0) -> None:
            if not self.winfo_exists() or i >= count:
                on_done(frames)
                return
            f = self.get_frame()
            if f is not None:
                frames.append(f.copy())
            self.after(interval_ms, _grab, i + 1)

        _grab(0)

    # ------------------------------------------------------------------
    # Image rendering (ImageTk — the render path that works here)
    # ------------------------------------------------------------------

    def _frame_to_photo(self, frame_bgr, max_w: int, max_h: int):
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image=pil, master=self)
        except Exception:
            return None

    def _photo_bytes_to_photo(self, photo_blob: bytes, max_w: int, max_h: int):
        try:
            arr = np.frombuffer(photo_blob, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            return self._frame_to_photo(frame, max_w, max_h)
        except Exception:
            return None

    def _show_photo_bytes(self, photo_blob: bytes, label, max_w: int = 440, max_h: int = 440) -> None:
        photo = self._photo_bytes_to_photo(photo_blob, max_w, max_h)
        if photo is None:
            self._clear_photo_label("Could not load photo")
            return
        label.configure(image=photo, text="")
        label._cbvms_photo = photo  # prevent GC

    def _clear_photo_label(self, text: str) -> None:
        """Reliably clear the preview photo (tk.Label clears with image='')."""
        self._selected_photo_label.configure(image="", text=text)
        self._selected_photo_label._cbvms_photo = None

    def _resolve_selected_student(self) -> dict | None:
        if self._selected_pk is None:
            return None
        student = next((s for s in self._students if s["id"] == self._selected_pk), None)
        if student is None:
            row = self.database.get_student(self._selected_pk)
            student = dict(row) if row is not None else None
        return student

    @staticmethod
    def _safe_grab(modal) -> None:
        try:
            if modal.winfo_exists():
                modal.grab_set()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Dashboard hooks (camera preview now lives in modals — no-ops)
    # ------------------------------------------------------------------

    def on_show(self) -> None:
        return

    def on_hide(self) -> None:
        return

    def update_preview(self, frame: np.ndarray | None) -> None:
        return

    # ------------------------------------------------------------------
    # Enroll modal
    # ------------------------------------------------------------------

    def _open_enroll_modal(self) -> None:
        if self.recognizer is None:
            self._set_status(
                "Face recognition not ready. Please wait for the model to load.", error=True,
            )
            return

        modal = ctk.CTkToplevel(self)
        modal.title("Enroll New Student")
        modal.configure(fg_color=COLOR_BG)
        modal.geometry("860x560")
        modal.resizable(False, False)
        modal.transient(self.winfo_toplevel())
        modal.after(120, modal.lift)
        modal.after(200, lambda: self._safe_grab(modal))

        modal.grid_columnconfigure((0, 1), weight=1)
        modal.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            modal, text="Enroll New Student", font=heading_font(16), text_color=COLOR_TEXT,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=PADDING, pady=(PADDING, 8))

        # LEFT — form
        form_card = ctk.CTkFrame(modal, fg_color=COLOR_SURFACE, corner_radius=CORNER_RADIUS,
                                 border_width=1, border_color=COLOR_BORDER)
        form_card.grid(row=1, column=0, sticky="nsew", padx=(PADDING, 8), pady=(0, 8))
        form_card.grid_columnconfigure(0, weight=1)

        form = ctk.CTkFrame(form_card, fg_color="transparent")
        form.pack(fill="x", padx=PADDING, pady=PADDING)
        form.grid_columnconfigure(1, weight=1)

        fields = [
            ("Full Name", "name"),
            ("Student ID", "student_id"),
            ("Course", "course"),
            ("Year and Section", "year_and_section"),
        ]
        self._entries = {}
        for r, (label, key) in enumerate(fields):
            ctk.CTkLabel(form, text=label, font=body_font(12), text_color=COLOR_TEXT_MUTED).grid(
                row=r, column=0, sticky="w", pady=6, padx=(0, 12)
            )
            entry = ctk.CTkEntry(form)
            entry.grid(row=r, column=1, sticky="ew", pady=6)
            self._entries[key] = entry

        ctk.CTkLabel(form, text="Gender", font=body_font(12), text_color=COLOR_TEXT_MUTED).grid(
            row=4, column=0, sticky="w", pady=6, padx=(0, 12)
        )
        self._gender_var = ctk.StringVar(value="Male")
        ctk.CTkSegmentedButton(
            form, values=["Male", "Female"], variable=self._gender_var,
        ).grid(row=4, column=1, sticky="ew", pady=6)

        self._enroll_status_label = ctk.CTkLabel(
            form_card, text="", font=body_font(12), text_color=COLOR_TEXT_MUTED, wraplength=360,
        )
        self._enroll_status_label.pack(anchor="w", padx=PADDING, pady=(0, PADDING))

        # RIGHT — live camera preview
        cam_card = ctk.CTkFrame(modal, fg_color=COLOR_SURFACE, corner_radius=CORNER_RADIUS,
                                border_width=1, border_color=COLOR_BORDER)
        cam_card.grid(row=1, column=1, sticky="nsew", padx=(8, PADDING), pady=(0, 8))
        ctk.CTkLabel(cam_card, text="Camera Preview", font=body_font(12),
                     text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=12, pady=(12, 6))
        canvas = tk.Canvas(cam_card, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT,
                           bg=COLOR_BG, highlightthickness=0, borderwidth=0)
        canvas.pack(padx=12, pady=(0, 12))

        state: dict = {"job": None, "img": None, "item": None}

        def _tick() -> None:
            if not modal.winfo_exists():
                return
            frame = self.get_frame()
            if frame is not None:
                disp = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                rgb = np.ascontiguousarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
                pil = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(image=pil, master=canvas)
                state["img"] = photo
                if state["item"] is None:
                    state["item"] = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                else:
                    canvas.itemconfig(state["item"], image=photo)
            state["job"] = modal.after(50, _tick)

        def _close() -> None:
            if state["job"] is not None:
                try:
                    modal.after_cancel(state["job"])
                except Exception:
                    pass
                state["job"] = None
            try:
                modal.grab_release()
            except Exception:
                pass
            self._enroll_status_label = None
            self._enroll_close = None
            modal.destroy()

        self._enroll_close = _close

        btns = ctk.CTkFrame(modal, fg_color="transparent")
        btns.grid(row=2, column=0, columnspan=2, sticky="ew", padx=PADDING, pady=(0, PADDING))
        btns.grid_columnconfigure(0, weight=1)

        self._enroll_capture_btn = ctk.CTkButton(
            btns, text="Capture & Enroll", height=40, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, command=self._capture_and_enroll,
        )
        self._enroll_capture_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            btns, text="Cancel", width=130, height=40, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER, hover_color="#DC2626", command=_close,
        ).grid(row=0, column=1)

        modal.protocol("WM_DELETE_WINDOW", _close)
        _tick()

    def _clear_form(self) -> None:
        for entry in self._entries.values():
            try:
                entry.delete(0, "end")
            except Exception:
                pass

    def _capture_and_enroll(self) -> None:
        if not self._entries or self._gender_var is None:
            return
        name = self._entries["name"].get().strip()
        student_id = self._entries["student_id"].get().strip()
        course = self._entries["course"].get().strip()
        year_and_section = self._entries["year_and_section"].get().strip()
        gender = self._gender_var.get()

        if not all([name, student_id, course, year_and_section]):
            self._set_enroll_status("Please fill in all fields.", error=True)
            return
        if self.database.student_id_exists(student_id):
            self._set_enroll_status(f"Student ID '{student_id}' is already enrolled.", error=True)
            return
        if self.recognizer is None:
            self._set_enroll_status("Face recognition not ready.", error=True)
            return
        if self.get_frame() is None:
            self._set_enroll_status("Camera unavailable.", error=True)
            return

        # Collect 10 frames over ~1s, then average the embeddings.
        btn = getattr(self, "_enroll_capture_btn", None)
        if btn is not None:
            btn.configure(state="disabled")
        self._set_enroll_status("Capturing frames… hold still")
        self._collect_frames(
            lambda frames: self._finish_enroll(
                frames, name, student_id, course, year_and_section, gender
            )
        )

    def _finish_enroll(self, frames, name, student_id, course, year_and_section, gender) -> None:
        btn = getattr(self, "_enroll_capture_btn", None)

        def _fail(msg: str) -> None:
            self._set_enroll_status(msg, error=True)
            if btn is not None and btn.winfo_exists():
                btn.configure(state="normal")

        if self.recognizer is None:
            _fail("Face recognition not ready.")
            return

        self._set_enroll_status("Processing…")
        embedding, box = self.recognizer.encode_face_multi(frames, min_valid=3)
        if embedding is None or box is None:
            _fail("Could not detect a stable face. Ensure good lighting and hold still.")
            return

        last = next((f for f in reversed(frames) if f is not None), None)
        if last is None:
            _fail("Camera unavailable.")
            return

        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        face_crop = last[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = last
        ok, buf = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            _fail("Failed to encode photo.")
            return

        try:
            self.database.insert_student(
                student_id=student_id,
                name=name,
                course=course,
                year_and_section=year_and_section,
                gender=gender,
                encoding=pickle.dumps(embedding),
                photo=buf.tobytes(),
            )
        except Exception as exc:
            _fail(f"Enrollment failed: {exc}")
            return

        self._clear_form()
        self._reload_students()
        self.recognizer.load_known_faces()
        self._set_status("Student enrolled successfully.", success=True)
        if self._enroll_close is not None:
            self._enroll_close()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _delete_selected(self) -> None:
        if self._selected_pk is None:
            self._set_status("Select a student to delete.", error=True)
            return

        student = next((s for s in self._students if s["id"] == self._selected_pk), None)
        name = student["name"] if student else "this student"
        if not messagebox.askyesno(
            "Confirm Delete",
            f"Delete {name}? This cannot be undone.",
            parent=self.winfo_toplevel(),
        ):
            return

        if self.database.delete_student(self._selected_pk):
            self._selected_pk = None
            self._update_btn.configure(state="disabled")
            self._clear_photo_label("Select a student to view photo")
            self._preview_caption.configure(text="")
            self._set_status("Student deleted.", success=True)
            self._reload_students()
            if self.recognizer is not None:
                self.recognizer.load_known_faces()
        else:
            self._set_status("Delete failed.", error=True)

    # ------------------------------------------------------------------
    # View photo (enlarge) modal
    # ------------------------------------------------------------------

    def _view_selected_photo(self) -> None:
        student = self._resolve_selected_student()
        if student is None:
            self._set_status("Select a student to view.", error=True)
            return

        modal = ctk.CTkToplevel(self)
        modal.title(f"Photo — {student.get('name', '')}")
        modal.configure(fg_color=COLOR_BG)
        modal.resizable(False, False)
        modal.transient(self.winfo_toplevel())
        modal.after(120, modal.lift)
        modal.after(200, lambda: self._safe_grab(modal))

        ctk.CTkLabel(
            modal, text=f"{student.get('name', '')}  ·  {student.get('student_id', '')}",
            font=heading_font(15), text_color=COLOR_TEXT,
        ).pack(padx=PADDING, pady=(PADDING, 8))

        photo = student.get("photo")
        img = self._photo_bytes_to_photo(photo, 560, 560) if photo else None
        if img is None:
            ctk.CTkLabel(
                modal, text="No photo on file", font=body_font(13),
                text_color=COLOR_TEXT_MUTED, width=400, height=300,
            ).pack(padx=PADDING, pady=PADDING)
        else:
            lbl = ctk.CTkLabel(modal, text="", image=img)
            lbl._cbvms_photo = img
            lbl.pack(padx=PADDING, pady=(0, PADDING))

        ctk.CTkButton(
            modal, text="Close", width=120, height=34, corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER, hover_color=COLOR_ACCENT_HOVER, command=modal.destroy,
        ).pack(pady=(0, PADDING))

    # ------------------------------------------------------------------
    # Update photo modal (explicit capture)
    # ------------------------------------------------------------------

    def _update_selected_photo(self) -> None:
        if self._selected_pk is None:
            self._set_status("Select a student to update.", error=True)
            return
        if self.recognizer is None:
            self._set_status(
                "Face recognition not ready. Please wait for the model to load.", error=True,
            )
            return
        student = self._resolve_selected_student()
        if student is None:
            self._set_status("Selected student not found.", error=True)
            return
        self._open_update_modal(student)

    def _open_update_modal(self, student: dict) -> None:
        PV_W, PV_H = 320, 240
        target_pk = int(student["id"])
        state: dict = {"job": None, "img": None, "preview_on": True, "capturing": False}

        modal = ctk.CTkToplevel(self)
        modal.title(f"Update Photo — {student.get('name', '')}")
        modal.configure(fg_color=COLOR_BG)
        modal.resizable(False, False)
        modal.transient(self.winfo_toplevel())
        modal.update_idletasks()
        sw, sh = modal.winfo_screenwidth(), modal.winfo_screenheight()
        modal.geometry(f"480x360+{(sw - 480) // 2}+{(sh - 360) // 2}")
        modal.after(120, modal.lift)
        modal.after(200, lambda: self._safe_grab(modal))

        # Rounded content card
        card = ctk.CTkFrame(modal, fg_color=COLOR_SURFACE, corner_radius=16,
                            border_width=1, border_color=COLOR_BORDER)
        card.pack(fill="both", expand=True, padx=PADDING, pady=PADDING)

        ctk.CTkLabel(
            card, text=student.get("name", ""), font=heading_font(16), text_color=COLOR_TEXT,
        ).pack(pady=(PADDING, 6))

        # Live preview — tk.Label (crash-safe with ImageTk reassignment)
        preview_wrap = ctk.CTkFrame(card, fg_color=COLOR_BG, corner_radius=CORNER_RADIUS)
        preview_wrap.pack(padx=PADDING, pady=(0, 8))
        preview_label = tk.Label(
            preview_wrap, text="Starting camera…", bd=0,
            bg=COLOR_BG, fg=COLOR_TEXT_MUTED, width=44, height=12,
        )
        preview_label.pack(padx=6, pady=6)

        status_lbl = ctk.CTkLabel(
            card, text="Look at the camera, then Capture & Update.",
            font=body_font(12), text_color=COLOR_TEXT_MUTED,
        )

        def _preview() -> None:
            if not modal.winfo_exists() or not state["preview_on"]:
                return
            frame = self.get_frame()
            if frame is not None:
                img = self._frame_to_photo(frame, PV_W, PV_H)
                if img is not None:
                    state["img"] = img
                    preview_label.configure(image=img, text="", width=PV_W, height=PV_H)
            else:
                preview_label.configure(image="", text="Camera unavailable")
            state["job"] = modal.after(50, _preview)

        def _close() -> None:
            state["preview_on"] = False
            if state["job"] is not None:
                try:
                    modal.after_cancel(state["job"])
                except Exception:
                    pass
                state["job"] = None
            try:
                modal.grab_release()
            except Exception:
                pass
            modal.destroy()

        def _finish(frames) -> None:
            def _retry(msg: str) -> None:
                status_lbl.configure(text=msg, text_color=COLOR_DANGER)
                if capture_btn.winfo_exists():
                    capture_btn.configure(state="normal")
                state["capturing"] = False

            status_lbl.configure(text="Processing…", text_color=COLOR_TEXT_MUTED)
            modal.update_idletasks()
            embedding, box = self.recognizer.encode_face_multi(frames, min_valid=3)
            if embedding is None or box is None:
                _retry("Could not detect a stable face. Ensure good lighting and hold still.")
                return
            last = next((f for f in reversed(frames) if f is not None), None)
            if last is None:
                _retry("Camera unavailable.")
                return
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            crop = last[y1:y2, x1:x2]
            if crop.size == 0:
                crop = last
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                _retry("Failed to encode photo.")
                return
            if self.database.update_student_encoding(target_pk, pickle.dumps(embedding), buf.tobytes()):
                self.recognizer.load_known_faces()
                self._reload_students()
                try:
                    self._show_photo_bytes(buf.tobytes(), self._selected_photo_label)
                except Exception:
                    pass
                self._set_status("Photo updated successfully.", success=True)
                _close()
            else:
                _retry("Update failed.")

        def _capture() -> None:
            if state["capturing"]:
                return
            state["capturing"] = True
            capture_btn.configure(state="disabled")
            status_lbl.configure(text="Capturing… hold still", text_color=COLOR_TEXT_MUTED)
            self._collect_frames(_finish)

        capture_btn = ctk.CTkButton(
            card, text="📷  Capture & Update", height=44, corner_radius=12,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, font=body_font(14),
            command=_capture,
        )
        capture_btn.pack(fill="x", padx=PADDING, pady=(2, 6))

        status_lbl.pack(pady=(0, 6))

        ctk.CTkButton(
            card, text="Cancel", height=30, corner_radius=CORNER_RADIUS,
            fg_color="transparent", hover_color=COLOR_BORDER,
            text_color=COLOR_TEXT_MUTED, command=_close,
        ).pack(pady=(0, PADDING))

        modal.protocol("WM_DELETE_WINDOW", _close)
        _preview()
