"""Violation history panel — Computer Based Vision Monitoring System (CBVMS)."""

from __future__ import annotations

import csv
import io
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
from PIL import Image, ImageTk

from database.db_manager import CBVMSDatabase
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
    CBVMSCard,
    body_font,
    heading_font,
)


PAGE_SIZE = 20


def _parse_yyyy_mm_dd(value: str) -> str | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return value


def _human_violation_type(value: str) -> str:
    v = (value or "").strip()
    mapping = {
        "wrong_hair_color": "Wrong Hair Color",
        "no_id_badge": "No ID Badge",
        "wrong_uniform": "Wrong Uniform",
        "earring_violation": "Earring Violation",
        "with_earring": "Earring Violation",
    }
    return mapping.get(v, v.replace("_", " ").title() if v else "—")


def _violation_badge_color(value: str) -> str:
    v = (value or "").strip()
    if v in ("no_id_badge",):
        return COLOR_DANGER
    if v in ("wrong_uniform",):
        return COLOR_WARNING
    if v in ("wrong_hair_color", "earring_violation", "with_earring"):
        return COLOR_ACCENT
    return COLOR_BORDER


class ViolationLogPanel(CBVMSCard):
    def __init__(self, master, *, database: CBVMSDatabase, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.database = database

        self._selected_violation_id: int | None = None
        self._selected_violation_row: dict | None = None
        self._snapshot_photo: ImageTk.PhotoImage | None = None

        self._page = 1
        self._page_count = 1
        self._total_records = 0
        self._rows: list[dict] = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_top_bar()
        self._build_main_area()
        self._configure_tree_style()

        self.refresh()

    def _build_top_bar(self) -> None:
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=PADDING, pady=(PADDING, 10))
        top.grid_columnconfigure(0, weight=1)

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew")
        row.grid_columnconfigure(0, weight=1)

        self._search_var = tk.StringVar()
        self._type_var = tk.StringVar(value="All")
        self._status_var = tk.StringVar(value="All")
        self._from_var = tk.StringVar()
        self._to_var = tk.StringVar()

        self._search_entry = ctk.CTkEntry(
            row,
            textvariable=self._search_var,
            placeholder_text="Search by student name or ID…",
            height=34,
            fg_color=COLOR_BG,
            border_color=COLOR_BORDER,
            text_color=COLOR_TEXT,
        )
        self._search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self._type_combo = ctk.CTkComboBox(
            row,
            values=["All", "wrong_hair_color", "no_id_badge", "wrong_uniform", "earring_violation"],
            variable=self._type_var,
            height=34,
            fg_color=COLOR_BG,
            border_color=COLOR_BORDER,
            button_color=COLOR_BORDER,
            button_hover_color=COLOR_ACCENT_HOVER,
            dropdown_fg_color=COLOR_SURFACE,
            dropdown_hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
        )
        self._type_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self._status_combo = ctk.CTkComboBox(
            row,
            values=["All", "Unreviewed", "Reviewed"],
            variable=self._status_var,
            height=34,
            fg_color=COLOR_BG,
            border_color=COLOR_BORDER,
            button_color=COLOR_BORDER,
            button_hover_color=COLOR_ACCENT_HOVER,
            dropdown_fg_color=COLOR_SURFACE,
            dropdown_hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
        )
        self._status_combo.grid(row=0, column=2, sticky="ew", padx=(0, 8))

        self._from_entry = ctk.CTkEntry(
            row,
            textvariable=self._from_var,
            placeholder_text="From (YYYY-MM-DD)",
            width=140,
            height=34,
            fg_color=COLOR_BG,
            border_color=COLOR_BORDER,
            text_color=COLOR_TEXT,
        )
        self._from_entry.grid(row=0, column=3, sticky="ew", padx=(0, 8))

        self._to_entry = ctk.CTkEntry(
            row,
            textvariable=self._to_var,
            placeholder_text="To (YYYY-MM-DD)",
            width=140,
            height=34,
            fg_color=COLOR_BG,
            border_color=COLOR_BORDER,
            text_color=COLOR_TEXT,
        )
        self._to_entry.grid(row=0, column=4, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            row,
            text="Search",
            height=34,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._on_search,
        ).grid(row=0, column=5, padx=(0, 8))

        ctk.CTkButton(
            row,
            text="Clear Filters",
            height=34,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._clear_filters,
        ).grid(row=0, column=6, padx=(0, 8))

        export = ctk.CTkButton(
            row,
            text="Export CSV",
            height=34,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_SAFE,
            hover_color="#0EA371",
            text_color=COLOR_TEXT,
            command=self._export_csv,
        )
        export.grid(row=0, column=7, sticky="e")

        # Reasonable widths for fixed columns
        row.grid_columnconfigure(1, weight=0, minsize=190)
        row.grid_columnconfigure(2, weight=0, minsize=140)
        row.grid_columnconfigure(3, weight=0, minsize=140)
        row.grid_columnconfigure(4, weight=0, minsize=140)

    def _build_main_area(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=PADDING, pady=(0, PADDING))
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # LEFT: Table + pagination
        left = ctk.CTkFrame(
            main,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)

        table_host = ctk.CTkFrame(left, fg_color="transparent")
        table_host.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        table_host.grid_columnconfigure(0, weight=1)
        table_host.grid_rowconfigure(0, weight=1)

        columns = ("idx", "student_name", "student_id", "violation_type", "timestamp", "status")
        self._tree = ttk.Treeview(
            table_host,
            columns=columns,
            show="headings",
            style="CBVMS.Treeview",
            selectmode="browse",
        )
        self._tree.heading("idx", text="#")
        self._tree.heading("student_name", text="Student Name")
        self._tree.heading("student_id", text="Student ID")
        self._tree.heading("violation_type", text="Violation Type")
        self._tree.heading("timestamp", text="Date & Time")
        self._tree.heading("status", text="Status")

        self._tree.column("idx", width=50, anchor="center", stretch=False)
        self._tree.column("student_name", width=220, anchor="w")
        self._tree.column("student_id", width=120, anchor="w", stretch=False)
        self._tree.column("violation_type", width=170, anchor="w")
        self._tree.column("timestamp", width=160, anchor="w", stretch=False)
        self._tree.column("status", width=110, anchor="w", stretch=False)

        vsb = ttk.Scrollbar(table_host, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._tree.tag_configure("unreviewed", foreground=COLOR_WARNING)
        self._tree.tag_configure("reviewed", foreground=COLOR_TEXT_MUTED)

        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)

        pager = ctk.CTkFrame(left, fg_color="transparent")
        pager.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        self._prev_btn = ctk.CTkButton(
            pager,
            text="Prev",
            height=32,
            width=80,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._prev_page,
        )
        self._prev_btn.pack(side="left")

        self._page_label = ctk.CTkLabel(
            pager,
            text="Page 1 of 1",
            font=body_font(12),
            text_color=COLOR_TEXT_MUTED,
        )
        self._page_label.pack(side="left", padx=12)

        self._next_btn = ctk.CTkButton(
            pager,
            text="Next",
            height=32,
            width=80,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_BORDER,
            hover_color=COLOR_ACCENT_HOVER,
            command=self._next_page,
        )
        self._next_btn.pack(side="left")

        self._count_label = ctk.CTkLabel(
            pager,
            text="0 records",
            font=body_font(12),
            text_color=COLOR_TEXT_MUTED,
        )
        self._count_label.pack(side="right")

        # RIGHT: Detail card
        right = ctk.CTkFrame(
            main,
            fg_color=COLOR_SURFACE,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right,
            text="Violation Details",
            font=heading_font(18),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=PADDING, pady=(PADDING, 8))

        self._snapshot_label = ctk.CTkLabel(
            right,
            text="No snapshot",
            font=body_font(12),
            text_color=COLOR_TEXT_MUTED,
            width=300,
            height=200,
            fg_color=COLOR_BG,
            corner_radius=CORNER_RADIUS,
        )
        self._snapshot_label.grid(row=1, column=0, sticky="ew", padx=PADDING, pady=(0, 12))

        self._detail_name = ctk.CTkLabel(right, text="Student: —", font=body_font(13), text_color=COLOR_TEXT)
        self._detail_name.grid(row=2, column=0, sticky="w", padx=PADDING)

        self._detail_id = ctk.CTkLabel(right, text="Student ID: —", font=body_font(13), text_color=COLOR_TEXT)
        self._detail_id.grid(row=3, column=0, sticky="w", padx=PADDING, pady=(2, 0))

        self._detail_course = ctk.CTkLabel(
            right, text="Course: —", font=body_font(13), text_color=COLOR_TEXT
        )
        self._detail_course.grid(row=4, column=0, sticky="w", padx=PADDING, pady=(2, 0))

        self._detail_year = ctk.CTkLabel(
            right, text="Year Level: —", font=body_font(13), text_color=COLOR_TEXT
        )
        self._detail_year.grid(row=5, column=0, sticky="w", padx=PADDING, pady=(2, 0))

        badge_row = ctk.CTkFrame(right, fg_color="transparent")
        badge_row.grid(row=6, column=0, sticky="ew", padx=PADDING, pady=(12, 0))

        self._violation_badge = ctk.CTkLabel(
            badge_row,
            text="—",
            font=body_font(12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_BORDER,
            corner_radius=999,
            padx=10,
            pady=4,
        )
        self._violation_badge.pack(side="left")

        self._detail_timestamp = ctk.CTkLabel(
            right, text="Timestamp: —", font=body_font(12), text_color=COLOR_TEXT_MUTED
        )
        self._detail_timestamp.grid(row=7, column=0, sticky="w", padx=PADDING, pady=(10, 0))

        self._detail_status = ctk.CTkLabel(
            right, text="Status: —", font=body_font(12), text_color=COLOR_TEXT_MUTED
        )
        self._detail_status.grid(row=8, column=0, sticky="w", padx=PADDING, pady=(2, 0))

        self._toggle_btn = ctk.CTkButton(
            right,
            text="Mark as Reviewed",
            height=36,
            corner_radius=CORNER_RADIUS,
            fg_color=COLOR_SAFE,
            hover_color="#0EA371",
            command=self._toggle_status,
            state="disabled",
        )
        self._toggle_btn.grid(row=9, column=0, sticky="ew", padx=PADDING, pady=(14, PADDING))

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

    def _on_search(self) -> None:
        self._page = 1
        self.refresh()

    def _clear_filters(self) -> None:
        self._search_var.set("")
        self._type_var.set("All")
        self._status_var.set("All")
        self._from_var.set("")
        self._to_var.set("")
        self._page = 1
        self.refresh()

    def _filters_to_where(self) -> tuple[str, list]:
        params: list = []
        clauses: list[str] = []

        q = (self._search_var.get() or "").strip()
        if q:
            clauses.append("(v.student_name LIKE ? OR v.student_id LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like])

        vtype = (self._type_var.get() or "All").strip()
        if vtype and vtype != "All":
            if vtype == "earring_violation":
                clauses.append("(v.violation_type = ? OR v.violation_type = ?)")
                params.extend(["earring_violation", "with_earring"])
            else:
                clauses.append("v.violation_type = ?")
                params.append(vtype)

        status = (self._status_var.get() or "All").strip()
        if status == "Unreviewed":
            clauses.append("v.status = 'unreviewed'")
        elif status == "Reviewed":
            clauses.append("v.status = 'reviewed'")

        from_date = _parse_yyyy_mm_dd(self._from_var.get())
        to_date = _parse_yyyy_mm_dd(self._to_var.get())
        if (self._from_var.get().strip() and not from_date) or (self._to_var.get().strip() and not to_date):
            raise ValueError("Date range must be in YYYY-MM-DD format.")
        if from_date:
            clauses.append("date(v.timestamp) >= date(?)")
            params.append(from_date)
        if to_date:
            clauses.append("date(v.timestamp) <= date(?)")
            params.append(to_date)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def _fetch_page(self) -> None:
        where, params = self._filters_to_where()
        offset = max(0, (self._page - 1) * PAGE_SIZE)

        with self.database.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS c FROM violations v {where}", params).fetchone()
            self._total_records = int(total["c"] if total else 0)
            self._page_count = max(1, (self._total_records + PAGE_SIZE - 1) // PAGE_SIZE)
            self._page = min(max(1, self._page), self._page_count)
            offset = (self._page - 1) * PAGE_SIZE

            rows = conn.execute(
                f"""
                SELECT v.id, v.student_id, v.student_name, v.violation_type, v.timestamp, v.status
                FROM violations v
                {where}
                ORDER BY datetime(v.timestamp) DESC, v.id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, PAGE_SIZE, offset],
            ).fetchall()

        self._rows = [dict(r) for r in rows]

    def refresh(self) -> None:
        try:
            self._fetch_page()
        except ValueError as exc:
            messagebox.showerror("CBVMS", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("CBVMS", f"Failed to load violations.\n\n{exc}")
            return

        # Rebuild table
        for item in self._tree.get_children():
            self._tree.delete(item)

        start_idx = (self._page - 1) * PAGE_SIZE
        for i, row in enumerate(self._rows, start=1):
            status = (row.get("status") or "unreviewed").strip().lower()
            tag = "unreviewed" if status == "unreviewed" else "reviewed"
            self._tree.insert(
                "",
                "end",
                iid=str(row["id"]),
                values=(
                    start_idx + i,
                    row.get("student_name") or "—",
                    row.get("student_id") or "—",
                    _human_violation_type(row.get("violation_type") or ""),
                    (row.get("timestamp") or "—")[:19],
                    "Unreviewed" if status == "unreviewed" else "Reviewed",
                ),
                tags=(tag,),
            )

        self._page_label.configure(text=f"Page {self._page} of {self._page_count}")
        self._count_label.configure(text=f"{self._total_records} records")

        self._prev_btn.configure(state="normal" if self._page > 1 else "disabled")
        self._next_btn.configure(state="normal" if self._page < self._page_count else "disabled")

        # Preserve selection if possible
        if self._selected_violation_id is not None:
            iid = str(self._selected_violation_id)
            if self._tree.exists(iid):
                self._tree.selection_set(iid)
                self._tree.see(iid)
                self._load_violation_details(self._selected_violation_id)
                return

        self._clear_details()

    def _prev_page(self) -> None:
        if self._page <= 1:
            return
        self._page -= 1
        self.refresh()

    def _next_page(self) -> None:
        if self._page >= self._page_count:
            return
        self._page += 1
        self.refresh()

    def _on_row_select(self, _event: tk.Event | None = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        try:
            vid = int(sel[0])
        except Exception:
            return
        self._selected_violation_id = vid
        self._load_violation_details(vid)

    def _clear_details(self) -> None:
        self._selected_violation_row = None
        self._snapshot_photo = None
        self._snapshot_label.configure(image=None, text="No snapshot")
        self._detail_name.configure(text="Student: —")
        self._detail_id.configure(text="Student ID: —")
        self._detail_course.configure(text="Course: —")
        self._detail_year.configure(text="Year Level: —")
        self._violation_badge.configure(text="—", fg_color=COLOR_BORDER)
        self._detail_timestamp.configure(text="Timestamp: —")
        self._detail_status.configure(text="Status: —")
        self._toggle_btn.configure(state="disabled", text="Mark as Reviewed", fg_color=COLOR_SAFE)

    def _load_violation_details(self, violation_id: int) -> None:
        with self.database.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    v.id, v.student_id, v.student_name, v.violation_type, v.timestamp, v.snapshot, v.status,
                    s.course AS course, s.year_level AS year_level
                FROM violations v
                LEFT JOIN students s ON s.student_id = v.student_id
                WHERE v.id = ?
                """,
                (violation_id,),
            ).fetchone()

        if row is None:
            self._clear_details()
            return

        data = dict(row)
        self._selected_violation_row = data

        self._detail_name.configure(text=f"Student: {data.get('student_name') or '—'}")
        self._detail_id.configure(text=f"Student ID: {data.get('student_id') or '—'}")
        self._detail_course.configure(text=f"Course: {data.get('course') or '—'}")
        self._detail_year.configure(text=f"Year Level: {data.get('year_level') or '—'}")

        vtype_raw = data.get("violation_type") or ""
        self._violation_badge.configure(
            text=_human_violation_type(vtype_raw),
            fg_color=_violation_badge_color(vtype_raw),
        )

        ts = (data.get("timestamp") or "—")[:19]
        self._detail_timestamp.configure(text=f"Timestamp: {ts}")

        status = (data.get("status") or "unreviewed").strip().lower()
        self._detail_status.configure(text=f"Status: {'Unreviewed' if status == 'unreviewed' else 'Reviewed'}")
        if status == "unreviewed":
            self._toggle_btn.configure(
                state="normal",
                text="Mark as Reviewed",
                fg_color=COLOR_SAFE,
                hover_color="#0EA371",
            )
        else:
            self._toggle_btn.configure(
                state="normal",
                text="Mark as Unreviewed",
                fg_color=COLOR_WARNING,
                hover_color="#D97706",
            )

        snap = data.get("snapshot")
        if not snap:
            self._snapshot_photo = None
            self._snapshot_label.configure(image=None, text="No snapshot")
            return

        try:
            img = Image.open(io.BytesIO(snap))
            img = img.convert("RGB")
            max_w = 300
            if img.width > max_w:
                ratio = max_w / float(img.width)
                img = img.resize((max_w, max(1, int(img.height * ratio))))
            photo = ImageTk.PhotoImage(img, master=self)
            self._snapshot_photo = photo
            self._snapshot_label.configure(image=photo, text="")
        except Exception:
            self._snapshot_photo = None
            self._snapshot_label.configure(image=None, text="Snapshot unavailable")

    def _toggle_status(self) -> None:
        if self._selected_violation_row is None:
            return
        vid = int(self._selected_violation_row["id"])
        current = (self._selected_violation_row.get("status") or "unreviewed").strip().lower()
        new_status = "reviewed" if current == "unreviewed" else "unreviewed"

        try:
            with self.database.connect() as conn:
                conn.execute("UPDATE violations SET status = ? WHERE id = ?", (new_status, vid))
                conn.commit()
        except Exception as exc:
            messagebox.showerror("CBVMS", f"Failed to update status.\n\n{exc}")
            return

        self._selected_violation_row["status"] = new_status
        self.refresh()

    def _export_csv(self) -> None:
        try:
            where, params = self._filters_to_where()
        except ValueError as exc:
            messagebox.showerror("CBVMS", str(exc))
            return

        filename = filedialog.asksaveasfilename(
            title="Export Violations CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not filename:
            return

        try:
            with self.database.connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT v.student_name, v.student_id, v.violation_type, v.timestamp, v.status
                    FROM violations v
                    {where}
                    ORDER BY datetime(v.timestamp) DESC, v.id DESC
                    """,
                    params,
                ).fetchall()

            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Student Name", "Student ID", "Violation Type", "Date & Time", "Status"])
                for r in rows:
                    status = (r["status"] or "unreviewed").strip().lower()
                    writer.writerow(
                        [
                            r["student_name"] or "—",
                            r["student_id"] or "—",
                            _human_violation_type(r["violation_type"] or ""),
                            (r["timestamp"] or "—")[:19],
                            "Unreviewed" if status == "unreviewed" else "Reviewed",
                        ]
                    )
        except Exception as exc:
            messagebox.showerror("CBVMS", f"Export failed.\n\n{exc}")
            return

        messagebox.showinfo("CBVMS", f"Exported {len(rows)} records to {filename}")
