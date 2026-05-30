"""Student enrollment panel for CBVMS."""

from __future__ import annotations

import pickle
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Callable

import cv2
import customtkinter as ctk
import numpy as np

try:
    import face_recognition  # type: ignore

    FACE_RECOGNITION_AVAILABLE = True
except Exception:
    face_recognition = None  # type: ignore
    FACE_RECOGNITION_AVAILABLE = False
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
    """Student list and enrollment form — shares the dashboard camera."""

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
        self.preview_active = False
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._preview_canvas_image_id: int | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()
        self._reload_students()
        self.grid_remove()

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

        ctk.CTkLabel(
            left,
            text="Enrolled Students",
            font=heading_font(16),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=PADDING, pady=(PADDING, 8))

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
        columns = ("name", "student_id", "course", "year_level", "enrolled_at")
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
            "year_level": "Year Level",
            "enrolled_at": "Date Enrolled",
        }
        widths = {"name": 140, "student_id": 90, "course": 100, "year_level": 80, "enrolled_at": 110}
        for col in columns:
            self._tree.heading(col, text=headings[col])
            self._tree.column(col, width=widths[col], anchor="w")

        scroll_y = ttk.Scrollbar(tree_wrap, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll_y.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)

        photo_card = ctk.CTkFrame(left, fg_color=COLOR_BG, corner_radius=CORNER_RADIUS)
        photo_card.grid(row=3, column=0, sticky="ew", padx=PADDING, pady=(0, 8))
        ctk.CTkLabel(
            photo_card,
            text="Selected Student Photo",
            font=body_font(11),
            text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", padx=12, pady=(8, 4))
        self._selected_photo_label = ctk.CTkLabel(
            photo_card,
            text="Select a student to view photo",
            font=body_font(11),
            text_color=COLOR_TEXT_MUTED,
            width=PREVIEW_WIDTH,
            height=100,
        )
        self._selected_photo_label.pack(padx=12, pady=(0, 12))

        footer = ctk.CTkFrame(left, fg_color="transparent")
        footer.grid(row=4, column=0, sticky="ew", padx=PADDING, pady=(0, PADDING))

        self._count_label = ctk.CTkLabel(
            footer,
            text="0 students enrolled",
            font=body_font(12),
            text_color=COLOR_TEXT_MUTED,
        )
        self._count_label.pack(side="left")

        btn_row = ctk.CTkFrame(footer, fg_color="transparent")
        btn_row.pack(side="right")

        ctk.CTkButton(
            btn_row,
            text="Reload",
            width=80,
            height=32,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._reload_students,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row,
            text="Delete Selected",
            width=120,
            height=32,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_DANGER,
            hover_color="#DC2626",
            command=self._delete_selected,
        ).pack(side="left")

    def _build_right_panel(self) -> None:
        right = ctk.CTkFrame(
            self,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(PADDING // 2, 0))
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right,
            text="Enroll New Student",
            font=heading_font(16),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=PADDING, pady=(PADDING, 8))

        form = ctk.CTkFrame(right, fg_color="transparent")
        form.grid(row=1, column=0, sticky="ew", padx=PADDING)
        form.grid_columnconfigure(1, weight=1)

        fields = [
            ("Full Name", "name"),
            ("Student ID", "student_id"),
            ("Course", "course"),
            ("Year Level", "year_level"),
        ]
        self._entries: dict[str, ctk.CTkEntry] = {}
        for row, (label, key) in enumerate(fields):
            ctk.CTkLabel(form, text=label, font=body_font(12), text_color=COLOR_TEXT_MUTED).grid(
                row=row, column=0, sticky="w", pady=6, padx=(0, 12)
            )
            entry = ctk.CTkEntry(form)
            entry.grid(row=row, column=1, sticky="ew", pady=6)
            self._entries[key] = entry

        preview_card = ctk.CTkFrame(
            right,
            fg_color=COLOR_BG,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        preview_card.grid(row=2, column=0, sticky="ew", padx=PADDING, pady=PADDING)
        preview_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            preview_card,
            text="Camera Preview",
            font=body_font(12),
            text_color=COLOR_TEXT_MUTED,
        ).pack(anchor="w", padx=12, pady=(12, 4))

        preview_wrap = ctk.CTkFrame(preview_card, fg_color=COLOR_BG)
        preview_wrap.pack(padx=12, pady=(0, 12))

        self._preview_canvas = tk.Canvas(
            preview_wrap,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            bg=COLOR_BG,
            highlightthickness=0,
            borderwidth=0,
        )
        self._preview_canvas.pack()
        self._preview_placeholder = ctk.CTkLabel(
            self._preview_canvas,
            text="Waiting for camera…",
            font=body_font(11),
            text_color=COLOR_TEXT_MUTED,
            fg_color=COLOR_BG,
        )
        self._preview_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.grid(row=3, column=0, sticky="ew", padx=PADDING)

        self._preview_btn = ctk.CTkButton(
            btn_row,
            text="Start Preview",
            height=36,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._toggle_preview,
        )
        self._preview_btn.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row,
            text="Capture & Enroll",
            height=36,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._capture_and_enroll,
        ).pack(side="left")

        self._status_label = ctk.CTkLabel(
            right,
            text="",
            font=body_font(12),
            text_color=COLOR_TEXT_MUTED,
            wraplength=360,
        )
        self._status_label.grid(row=4, column=0, sticky="w", padx=PADDING, pady=(8, PADDING))

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

    def _set_status(self, message: str, *, success: bool = False, error: bool = False) -> None:
        if error:
            color = COLOR_DANGER
        elif success:
            color = COLOR_SAFE
        else:
            color = COLOR_TEXT_MUTED
        self._status_label.configure(text=message, text_color=color)

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
                "",
                "end",
                iid=str(student["id"]),
                values=(
                    student.get("name", ""),
                    student.get("student_id", ""),
                    student.get("course", "") or "—",
                    student.get("year_level", "") or "—",
                    enrolled or "—",
                ),
            )

    def _on_row_select(self, _event: tk.Event | None = None) -> None:
        selection = self._tree.selection()
        if not selection:
            self._selected_pk = None
            return
        self._selected_pk = int(selection[0])
        student = next((s for s in self._students if s["id"] == self._selected_pk), None)
        if student is None:
            student = self.database.get_student(self._selected_pk)
            if student is not None:
                student = dict(student)
        if student and student.get("photo"):
            self._show_photo_bytes(student["photo"], self._selected_photo_label, 120)
        else:
            self._selected_photo_label.configure(
                image=None,
                text="No photo on file",
            )

    def _show_photo_bytes(self, photo_blob: bytes, label: ctk.CTkLabel, height: int) -> None:
        arr = np.frombuffer(photo_blob, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            label.configure(image=None, text="Could not load photo")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((PREVIEW_WIDTH, height), Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        label.configure(image=photo, text="")
        label._cbvms_photo = photo  # prevent GC

    def on_show(self) -> None:
        """Start preview when the enrollment view is opened."""
        if not self.preview_active:
            self._start_preview()

    def update_preview(self, frame: np.ndarray | None) -> None:
        """Paint a frame from the dashboard camera pump (main thread)."""
        if not self.preview_active or not self.winfo_exists():
            return
        if frame is None:
            return
        if not self._preview_canvas.winfo_ismapped():
            return
        display = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        rgb = np.ascontiguousarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        pil = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=pil, master=self._preview_canvas)
        self._preview_photo = photo
        self._preview_placeholder.place_forget()
        if self._preview_canvas_image_id is None:
            self._preview_canvas_image_id = self._preview_canvas.create_image(
                0,
                0,
                anchor=tk.NW,
                image=photo,
            )
        else:
            self._preview_canvas.itemconfig(self._preview_canvas_image_id, image=photo)

    def _toggle_preview(self) -> None:
        if self.preview_active:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self) -> None:
        self.preview_active = True
        self._preview_btn.configure(text="Stop Preview")
        self._set_status("Camera preview active", success=False)

    def _stop_preview(self) -> None:
        self.preview_active = False
        self._preview_btn.configure(text="Start Preview")
        self._preview_canvas_image_id = None
        self._preview_photo = None
        self._preview_canvas.delete("all")
        self._preview_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _clear_form(self) -> None:
        for entry in self._entries.values():
            entry.delete(0, "end")

    def _capture_and_enroll(self) -> None:
        name = self._entries["name"].get().strip()
        student_id = self._entries["student_id"].get().strip()
        course = self._entries["course"].get().strip()
        year_level = self._entries["year_level"].get().strip()

        if not all([name, student_id, course, year_level]):
            self._set_status("Please fill in all fields.", error=True)
            return

        if self.database.student_id_exists(student_id):
            self._set_status(f"Student ID '{student_id}' is already enrolled.", error=True)
            return

        if not FACE_RECOGNITION_AVAILABLE or face_recognition is None:
            self._set_status(
                "Face recognition is not installed. On Windows install CMake and "
                "Visual Studio C++ Build Tools, then: pip install -r requirements-face.txt",
                error=True,
            )
            return

        frame = self.get_frame()
        if frame is None:
            self._set_status("Camera unavailable. Start preview when the camera is active.", error=True)
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            self._set_status(
                "No face detected. Adjust position and try again.",
                error=True,
            )
            return

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            self._set_status(
                "No face detected. Adjust position and try again.",
                error=True,
            )
            return

        encoding_blob = pickle.dumps(encodings[0])
        top, right, bottom, left = locations[0]
        face_crop = frame[top:bottom, left:right]
        if face_crop.size == 0:
            face_crop = frame
        ok, buf = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            self._set_status("Failed to encode photo.", error=True)
            return

        try:
            self.database.insert_student(
                student_id=student_id,
                name=name,
                course=course,
                year_level=year_level,
                encoding=encoding_blob,
                photo=buf.tobytes(),
            )
        except Exception as exc:
            self._set_status(f"Enrollment failed: {exc}", error=True)
            return

        self._clear_form()
        self._set_status("Student enrolled successfully", success=True)
        self._reload_students()
        self.recognizer.load_known_faces()

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
            self._selected_photo_label.configure(image=None, text="Select a student to view photo")
            self._set_status("Student deleted.", success=True)
            self._reload_students()
            self.recognizer.load_known_faces()
        else:
            self._set_status("Delete failed.", error=True)

    def on_hide(self) -> None:
        """Pause enrollment preview when leaving this view."""
        if self.preview_active:
            self._stop_preview()
