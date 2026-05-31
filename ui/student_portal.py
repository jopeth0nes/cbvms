"""SECURE — standalone student portal (light theme).

Entirely self-contained: does NOT import ui/components.py (different theme). A student
sees only data belonging to their own student_id. Launched from auth/login.py.
"""

from __future__ import annotations

import io
import tkinter as tk
from datetime import date, datetime

import customtkinter as ctk
from PIL import Image, ImageTk

from database.db_manager import CBVMSDatabase

# --- Light-theme palette (all colors live here; nothing hardcoded below) ---
SP_BG = "#F4F6F9"          # page background
SP_SIDEBAR = "#1E2A3A"     # dark navy sidebar
SP_SURFACE = "#FFFFFF"     # card/panel background
SP_ACCENT = "#1A56DB"      # blue accent
SP_TEXT = "#111827"        # primary text
SP_MUTED = "#6B7280"       # secondary/muted text
SP_BORDER = "#E5E7EB"      # card borders
SP_SAFE = "#059669"        # green (compliant)
SP_DANGER = "#DC2626"      # red (violation)
SP_WARNING = "#D97706"     # orange (unreviewed)
# Local supporting tints (kept SP_*-prefixed; no equivalents above)
SP_ACCENT_HOVER = "#1648B0"
SP_WHITE = "#FFFFFF"
SP_SIDEBAR_ACTIVE = "#2B3B52"   # active/hover nav highlight (lighter navy)
SP_SIDEBAR_HOVER = "#26344A"
SP_SIDEBAR_MUTED = "#9AA7B8"
SP_PILL_WARN_BG = "#FEF3C7"     # unreviewed pill bg
SP_PILL_OK_BG = "#D1FAE5"       # reviewed pill bg
SP_PLACEHOLDER_BG = "#EDEFF3"
SP_HOVER_LIGHT = "#EEF2FB"

_SIDEBAR_FULL = 280
_SIDEBAR_COMPACT = 200

_NAV_ITEMS = [
    ("dashboard", "🏠  Dashboard"),
    ("violations", "⚠️  My Violations"),
    ("profile", "👤  My Profile"),
    ("settings", "⚙️  User Settings"),
    ("report", "📋  System Report"),
]
_REPORT_CATEGORIES = ["System Bug", "Account Issue", "Violation Dispute", "Other"]


def _f(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(size=size, weight=weight)


def _parse_ts(ts: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(str(ts), fmt)
        except (ValueError, TypeError):
            continue
    return None


class StudentPortal(ctk.CTk):
    """Light-theme single-window portal scoped to one student_id."""

    def __init__(self, *, student_id: str, display_name: str) -> None:
        super().__init__()
        ctk.set_appearance_mode("light")

        self.student_id = student_id
        self.display_name = display_name
        self.logged_out = False
        self._prefs: dict = {"email_notifications": True}
        self._active = "dashboard"
        self._compact = False
        self._violation_filter = "All"
        self._image_refs: list = []  # keep ImageTk refs alive

        self.db = CBVMSDatabase()
        self._student = self.db.get_student_by_student_id(student_id) or {}
        self._violations = self.db.get_violations_for_student(student_id)

        self._activity_log: list[dict] = [
            {"action": "Logged in", "detail": "Student signed in to SECURE", "ts": datetime.now()},
        ]

        self.title("SECURE — Student Portal")
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.configure(fg_color=SP_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._content = ctk.CTkFrame(self, fg_color=SP_BG)
        self._content.grid(row=0, column=1, sticky="nsew")
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)

        self._show("dashboard")

    # ------------------------------------------------------------------
    # Activity log + toast
    # ------------------------------------------------------------------

    def _log_activity(self, action: str, detail: str) -> None:
        self._activity_log.append({"action": action, "detail": detail, "ts": datetime.now()})

    def _toast(self, message: str, kind: str = "success") -> None:
        accent = {"success": SP_SAFE, "error": SP_DANGER, "info": SP_ACCENT}.get(kind, SP_ACCENT)
        toast = ctk.CTkFrame(self, fg_color=SP_SURFACE, corner_radius=10,
                             border_width=1, border_color=SP_BORDER)
        toast.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-20)
        ctk.CTkFrame(toast, fg_color=accent, width=5, corner_radius=10).pack(side="left", fill="y")
        ctk.CTkLabel(toast, text=message, font=_f(13), text_color=SP_TEXT,
                     wraplength=300, justify="left").pack(side="left", padx=14, pady=12)
        toast.after(3000, lambda: toast.winfo_exists() and toast.destroy())

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _build_sidebar(self) -> None:
        self._sidebar = ctk.CTkFrame(self, width=_SIDEBAR_FULL, fg_color=SP_SIDEBAR, corner_radius=0)
        self._sidebar.grid(row=0, column=0, sticky="nsw")
        self._sidebar.grid_propagate(False)
        self._sidebar.grid_rowconfigure(2, weight=1)

        # Branding
        brand = ctk.CTkFrame(self._sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=20, pady=(22, 10))
        top = ctk.CTkFrame(brand, fg_color="transparent")
        top.pack(anchor="w")
        ctk.CTkLabel(top, text="🛡️", font=_f(28)).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(top, text="SECURE", font=_f(22, "bold"), text_color=SP_WHITE).pack(side="left")
        self._subtitle = ctk.CTkLabel(
            brand,
            text="Student Entrance Camera-based Uniform, Grooming, Accessory "
                 "Recognition and Evaluation",
            font=_f(10), text_color=SP_SIDEBAR_MUTED, justify="left", wraplength=230,
        )
        self._subtitle.pack(anchor="w", pady=(6, 0))

        # User
        user = ctk.CTkFrame(self._sidebar, fg_color="transparent")
        user.grid(row=1, column=0, sticky="ew", padx=20, pady=(8, 6))
        ctk.CTkLabel(user, text=self.display_name, font=_f(14, "bold"),
                     text_color=SP_WHITE, anchor="w").pack(anchor="w")
        ctk.CTkLabel(user, text="STUDENT", font=_f(11, "bold"),
                     text_color=SP_ACCENT, anchor="w").pack(anchor="w")
        ctk.CTkFrame(self._sidebar, height=1, fg_color=SP_SIDEBAR_ACTIVE).grid(
            row=1, column=0, sticky="ew", padx=20, pady=(0, 0))

        # Nav
        nav = ctk.CTkFrame(self._sidebar, fg_color="transparent")
        nav.grid(row=2, column=0, sticky="new", padx=14, pady=(14, 0))
        nav.grid_columnconfigure(0, weight=1)
        self._nav_btns: dict[str, ctk.CTkButton] = {}
        for i, (key, label) in enumerate(_NAV_ITEMS):
            btn = ctk.CTkButton(
                nav, text=label, anchor="w", height=42, corner_radius=8,
                fg_color="transparent", hover_color=SP_SIDEBAR_HOVER,
                text_color=SP_WHITE, font=_f(13),
                command=lambda k=key: self._show(k),
            )
            btn.grid(row=i, column=0, sticky="ew", pady=3)
            self._nav_btns[key] = btn

        # Logout pinned bottom
        logout = ctk.CTkButton(
            self._sidebar, text="🚪  Logout", anchor="w", height=42, corner_radius=8,
            fg_color="transparent", hover_color=SP_DANGER, text_color=SP_WHITE, font=_f(13),
            command=self._logout,
        )
        logout.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 18))

    def _set_active_nav(self) -> None:
        for k, btn in self._nav_btns.items():
            btn.configure(fg_color=SP_SIDEBAR_ACTIVE if k == self._active else "transparent")

    # ------------------------------------------------------------------
    # Panel dispatch
    # ------------------------------------------------------------------

    def _show(self, key: str) -> None:
        self._active = key
        self._set_active_nav()
        for w in self._content.winfo_children():
            w.destroy()
        self._image_refs.clear()
        {
            "dashboard": self._panel_dashboard,
            "violations": self._panel_violations,
            "profile": self._panel_profile,
            "settings": self._panel_settings,
            "report": self._panel_report,
        }[key]()

    def _scroll_host(self, title: str, subtitle: str = "") -> ctk.CTkScrollableFrame:
        scroll = ctk.CTkScrollableFrame(self._content, fg_color=SP_BG)
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(scroll, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=30, pady=(26, 4))
        ctk.CTkLabel(head, text=title, font=_f(24, "bold"), text_color=SP_TEXT).pack(anchor="w")
        if subtitle:
            ctk.CTkLabel(head, text=subtitle, font=_f(12), text_color=SP_MUTED,
                         justify="left", wraplength=760).pack(anchor="w", pady=(2, 0))
        return scroll

    @staticmethod
    def _card(parent) -> ctk.CTkFrame:
        return ctk.CTkFrame(parent, fg_color=SP_SURFACE, corner_radius=12,
                            border_width=1, border_color=SP_BORDER)

    def _status_pill(self, parent, status: str) -> ctk.CTkLabel:
        reviewed = status == "reviewed"
        return ctk.CTkLabel(
            parent, text=status, font=_f(11, "bold"),
            text_color=SP_SAFE if reviewed else SP_WARNING,
            fg_color=SP_PILL_OK_BG if reviewed else SP_PILL_WARN_BG,
            corner_radius=999, padx=10, pady=2,
        )

    # ------------------------------------------------------------------
    # Image helpers (ImageTk — CTkImage does not render on this build)
    # ------------------------------------------------------------------

    def _photo_from_blob(self, blob, max_w: int, max_h: int):
        if not blob:
            return None
        try:
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            img.thumbnail((max_w, max_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._image_refs.append(photo)
            return photo
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Dashboard panel
    # ------------------------------------------------------------------

    def _panel_dashboard(self) -> None:
        v = self._violations
        total = len(v)
        unreviewed = sum(1 for x in v if x.get("status") == "unreviewed")
        last_dt = _parse_ts(v[0]["timestamp"]) if v else None
        last_str = last_dt.strftime("%b %d, %Y") if last_dt else "None"

        if last_dt:
            streak = max(0, (date.today() - last_dt.date()).days)
        else:
            enr = _parse_ts(self._student.get("enrolled_at", ""))
            streak = max(0, (date.today() - enr.date()).days) if enr else 0

        scroll = self._scroll_host("Dashboard", f"Welcome back, {self.display_name}.")

        cards = ctk.CTkFrame(scroll, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="ew", padx=30, pady=(14, 8))
        for c in range(4):
            cards.grid_columnconfigure(c, weight=1, uniform="stat")
        specs = [
            ("⚠️", "Total Violations", str(total), SP_DANGER if total > 0 else SP_SAFE),
            ("📋", "Unreviewed", str(unreviewed), SP_WARNING),
            ("🕒", "Last Violation", last_str, SP_TEXT),
            ("✅", "Compliance Streak", f"{streak} days", SP_SAFE),
        ]
        for col, (icon, label, value, color) in enumerate(specs):
            card = self._card(cards)
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 12, 0))
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(card, text=icon, font=_f(22)).grid(row=0, column=0, rowspan=2, padx=(16, 8), pady=14)
            ctk.CTkLabel(card, text=value, font=_f(22, "bold"), text_color=color,
                         anchor="w").grid(row=0, column=1, sticky="w", pady=(16, 0), padx=(0, 14))
            ctk.CTkLabel(card, text=label, font=_f(11), text_color=SP_MUTED,
                         anchor="w").grid(row=1, column=1, sticky="w", pady=(0, 16), padx=(0, 14))

        body = ctk.CTkFrame(scroll, fg_color="transparent")
        body.grid(row=2, column=0, sticky="ew", padx=30, pady=(8, 26))
        body.grid_columnconfigure(0, weight=3, uniform="b")
        body.grid_columnconfigure(1, weight=2, uniform="b")

        self._dash_recent(body)
        self._dash_breakdown(body)

    def _dash_recent(self, parent) -> None:
        card = self._card(parent)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        card.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="Recent Violations", font=_f(15, "bold"),
                     text_color=SP_TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(head, text="View All →", width=80, height=26, corner_radius=8,
                      fg_color="transparent", hover_color=SP_HOVER_LIGHT, text_color=SP_ACCENT,
                      font=_f(12), command=lambda: self._show("violations")).grid(row=0, column=1, sticky="e")

        if not self._violations:
            self._empty_state(card, row=1)
            return
        for i, viol in enumerate(self._violations[:5], start=1):
            self._recent_row(card, i, viol)

    def _recent_row(self, parent, row: int, viol: dict) -> None:
        rf = ctk.CTkFrame(parent, fg_color=SP_BG, corner_radius=8)
        rf.grid(row=row, column=0, sticky="ew", padx=16, pady=(0, 8))
        rf.grid_columnconfigure(1, weight=1)
        ctk.CTkFrame(rf, width=4, fg_color=SP_DANGER, corner_radius=8).grid(
            row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 10), pady=2)
        ctk.CTkLabel(rf, text=viol.get("violation_type", "—"), font=_f(13, "bold"),
                     text_color=SP_TEXT, anchor="w").grid(row=0, column=1, sticky="w", pady=(8, 0))
        dt = _parse_ts(viol.get("timestamp", ""))
        ts = dt.strftime("%b %d, %Y %H:%M") if dt else str(viol.get("timestamp", ""))
        ctk.CTkLabel(rf, text=ts, font=_f(11), text_color=SP_MUTED, anchor="w").grid(
            row=1, column=1, sticky="w", pady=(0, 8))
        self._status_pill(rf, viol.get("status", "unreviewed")).grid(
            row=0, column=2, rowspan=2, padx=(8, 12))

    def _dash_breakdown(self, parent) -> None:
        card = self._card(parent)
        card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text="Violation Types", font=_f(15, "bold"),
                     text_color=SP_TEXT).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 10))

        counts: dict[str, int] = {}
        for v in self._violations:
            t = v.get("violation_type", "—")
            counts[t] = counts.get(t, 0) + 1
        if not counts:
            ctk.CTkLabel(card, text="No data yet", font=_f(12), text_color=SP_MUTED).grid(
                row=1, column=0, sticky="w", padx=16, pady=(0, 16))
            return
        maxc = max(counts.values())
        for i, (vtype, cnt) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1]), start=1):
            rowf = ctk.CTkFrame(card, fg_color="transparent")
            rowf.grid(row=i, column=0, sticky="ew", padx=16, pady=4)
            rowf.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(rowf, text=vtype, font=_f(12), text_color=SP_TEXT, anchor="w").grid(
                row=0, column=0, sticky="w")
            ctk.CTkLabel(rowf, text=str(cnt), font=_f(12, "bold"), text_color=SP_ACCENT).grid(
                row=0, column=1, sticky="e", padx=(8, 0))
            track = ctk.CTkFrame(rowf, fg_color=SP_BORDER, height=10, corner_radius=6)
            track.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
            track.grid_propagate(False)
            width = max(8, int(200 * cnt / maxc))
            bar = ctk.CTkFrame(track, fg_color=SP_ACCENT, height=10, width=width, corner_radius=6)
            bar.place(x=0, y=0, relheight=1.0)

    def _empty_state(self, parent, row: int) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", padx=16, pady=30)
        wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(wrap, text="✅", font=_f(40)).grid(row=0, column=0)
        ctk.CTkLabel(wrap, text="No violations recorded", font=_f(15, "bold"),
                     text_color=SP_TEXT).grid(row=1, column=0, pady=(6, 2))
        ctk.CTkLabel(wrap, text="You are compliant.", font=_f(12),
                     text_color=SP_MUTED).grid(row=2, column=0)

    # ------------------------------------------------------------------
    # My Violations panel
    # ------------------------------------------------------------------

    def _panel_violations(self) -> None:
        self._log_activity("Violations viewed", "Opened the My Violations page")
        scroll = self._scroll_host("My Violations")

        head = ctk.CTkFrame(scroll, fg_color="transparent")
        head.grid(row=1, column=0, sticky="ew", padx=30, pady=(8, 10))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text=f"{len(self._violations)} total", font=_f(12, "bold"),
                     text_color=SP_WHITE, fg_color=SP_ACCENT, corner_radius=999,
                     padx=12, pady=3).grid(row=0, column=0, sticky="w")
        seg = ctk.CTkSegmentedButton(
            head, values=["All", "Unreviewed", "Reviewed"], command=self._on_violation_filter,
            fg_color=SP_BORDER, selected_color=SP_ACCENT, selected_hover_color=SP_ACCENT_HOVER,
            unselected_color=SP_SURFACE, unselected_hover_color=SP_HOVER_LIGHT,
            text_color=SP_TEXT,
        )
        seg.set(self._violation_filter)
        seg.grid(row=0, column=1, sticky="e")

        self._viol_list = ctk.CTkFrame(scroll, fg_color="transparent")
        self._viol_list.grid(row=2, column=0, sticky="ew", padx=30, pady=(0, 26))
        self._viol_list.grid_columnconfigure(0, weight=1)
        self._render_violation_list()

    def _on_violation_filter(self, value: str) -> None:
        self._violation_filter = value
        self._render_violation_list()

    def _render_violation_list(self) -> None:
        for w in self._viol_list.winfo_children():
            w.destroy()
        items = self._violations
        if self._violation_filter == "Unreviewed":
            items = [v for v in items if v.get("status") == "unreviewed"]
        elif self._violation_filter == "Reviewed":
            items = [v for v in items if v.get("status") == "reviewed"]

        if not items:
            self._empty_state(self._viol_list, row=0)
            return
        for i, viol in enumerate(items):
            self._violation_card(i, viol)

    def _violation_card(self, row: int, viol: dict) -> None:
        reviewed = viol.get("status") == "reviewed"
        card = self._card(self._viol_list)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkFrame(card, width=4, fg_color=SP_SAFE if reviewed else SP_DANGER,
                     corner_radius=8).grid(row=0, column=0, rowspan=3, sticky="nsw", padx=(0, 12), pady=2)
        ctk.CTkLabel(card, text=viol.get("violation_type", "—"), font=_f(14, "bold"),
                     text_color=SP_TEXT, anchor="w").grid(row=0, column=1, sticky="w", pady=(12, 0))
        dt = _parse_ts(viol.get("timestamp", ""))
        ts = dt.strftime("%b %d, %Y · %H:%M") if dt else str(viol.get("timestamp", ""))
        ctk.CTkLabel(card, text=ts, font=_f(11), text_color=SP_MUTED, anchor="e").grid(
            row=0, column=2, sticky="e", padx=(0, 16), pady=(12, 0))
        self._status_pill(card, viol.get("status", "unreviewed")).grid(
            row=1, column=1, sticky="w", pady=(4, 0))
        if viol.get("snapshot"):
            ctk.CTkButton(card, text="📷  View Photo", width=120, height=30, corner_radius=8,
                          fg_color=SP_ACCENT, hover_color=SP_ACCENT_HOVER, text_color=SP_WHITE,
                          font=_f(12), command=lambda x=viol: self._open_snapshot(x)).grid(
                row=2, column=1, sticky="w", pady=(8, 12))
        else:
            ctk.CTkFrame(card, fg_color="transparent", height=10).grid(row=2, column=1, pady=(0, 6))

    def _open_snapshot(self, viol: dict) -> None:
        modal = ctk.CTkToplevel(self)
        modal.title(f"Violation Snapshot — {viol.get('violation_type', '')}")
        modal.configure(fg_color=SP_BG)
        modal.geometry("520x420")
        modal.resizable(False, False)
        modal.transient(self)
        modal.after(120, modal.lift)
        modal.after(200, lambda: modal.winfo_exists() and modal.grab_set())

        holder = tk.Label(modal, bg=SP_BG, bd=0)
        holder.pack(padx=20, pady=(20, 8))
        photo = self._photo_from_blob(viol.get("snapshot"), 480, 360)
        if photo is not None:
            holder.configure(image=photo)
            holder._img_ref = photo  # extra ref on the widget
        else:
            holder.configure(text="Could not load image", fg=SP_MUTED, font=("Helvetica", 13))

        dt = _parse_ts(viol.get("timestamp", ""))
        ts = dt.strftime("%b %d, %Y · %H:%M") if dt else str(viol.get("timestamp", ""))
        ctk.CTkLabel(modal, text=viol.get("violation_type", "—"), font=_f(14, "bold"),
                     text_color=SP_TEXT).pack()
        ctk.CTkLabel(modal, text=ts, font=_f(11), text_color=SP_MUTED).pack(pady=(0, 6))
        ctk.CTkButton(modal, text="Close", width=120, height=34, corner_radius=8,
                      fg_color=SP_ACCENT, hover_color=SP_ACCENT_HOVER, text_color=SP_WHITE,
                      command=modal.destroy).pack(pady=(0, 12))

    # ------------------------------------------------------------------
    # My Profile panel
    # ------------------------------------------------------------------

    def _panel_profile(self) -> None:
        scroll = self._scroll_host("User Profile", "View and update your account information.")

        card = self._card(scroll)
        card.grid(row=1, column=0, sticky="ew", padx=30, pady=(14, 12))
        card.grid_columnconfigure(1, weight=1)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(18, 8))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="Profile Information", font=_f(17, "bold"),
                     text_color=SP_TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(head, text="Update Profile", width=130, height=34, corner_radius=8,
                      fg_color=SP_ACCENT, hover_color=SP_ACCENT_HOVER, text_color=SP_WHITE,
                      font=_f(13), command=self._open_update_profile).grid(row=0, column=1, sticky="e")

        # Photo
        photo_box = ctk.CTkFrame(card, fg_color=SP_PLACEHOLDER_BG, corner_radius=12,
                                 width=200, height=200)
        photo_box.grid(row=1, column=0, sticky="nw", padx=20, pady=(0, 16))
        photo_box.grid_propagate(False)
        photo = self._photo_from_blob(self._student.get("photo"), 196, 196)
        if photo is not None:
            lbl = tk.Label(photo_box, image=photo, bg=SP_PLACEHOLDER_BG, bd=0)
            lbl._img_ref = photo
            lbl.place(relx=0.5, rely=0.5, anchor="center")
        else:
            ph = ctk.CTkFrame(photo_box, fg_color="transparent")
            ph.place(relx=0.5, rely=0.5, anchor="center")
            ctk.CTkLabel(ph, text="👤", font=_f(40), text_color=SP_MUTED).pack()
            ctk.CTkLabel(ph, text="No photo uploaded", font=_f(11), text_color=SP_MUTED).pack()

        # Fields
        fields = ctk.CTkFrame(card, fg_color="transparent")
        fields.grid(row=1, column=1, sticky="new", padx=(10, 20), pady=(0, 16))
        fields.grid_columnconfigure(0, weight=1)
        s = self._student
        enr = _parse_ts(s.get("enrolled_at", ""))
        rows = [
            ("STUDENT ID", s.get("student_id", self.student_id)),
            ("FULL NAME", s.get("name", "—")),
            ("COURSE", s.get("course") or "—"),
            ("YEAR & SECTION", s.get("year_and_section") or "—"),
            ("GENDER", s.get("gender") or "—"),
            ("ENROLLED ON", enr.strftime("%b %d, %Y") if enr else (s.get("enrolled_at") or "—")),
        ]
        for i, (label, value) in enumerate(rows):
            ctk.CTkLabel(fields, text=label, font=_f(10, "bold"), text_color=SP_MUTED,
                         anchor="w").grid(row=i * 2, column=0, sticky="w", pady=(8 if i else 0, 0))
            ctk.CTkLabel(fields, text=str(value), font=_f(14, "bold"), text_color=SP_TEXT,
                         anchor="w").grid(row=i * 2 + 1, column=0, sticky="w")

        # Password section
        pwd = ctk.CTkFrame(card, fg_color="transparent")
        pwd.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 20))
        ctk.CTkFrame(pwd, height=1, fg_color=SP_BORDER).pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(pwd, text="Password", font=_f(15, "bold"), text_color=SP_TEXT).pack(anchor="w")
        ctk.CTkLabel(pwd, text="To change your password, contact your system administrator.",
                     font=_f(12), text_color=SP_MUTED).pack(anchor="w", pady=(4, 0))

        self._activity_log_card(scroll, row=2)

    def _open_update_profile(self) -> None:
        modal = ctk.CTkToplevel(self)
        modal.title("Update Profile")
        modal.configure(fg_color=SP_BG)
        modal.geometry("420x320")
        modal.resizable(False, False)
        modal.transient(self)
        modal.after(120, modal.lift)
        modal.after(200, lambda: modal.winfo_exists() and modal.grab_set())

        ctk.CTkLabel(modal, text="Update Profile", font=_f(18, "bold"),
                     text_color=SP_TEXT).pack(anchor="w", padx=20, pady=(18, 10))
        body = ctk.CTkFrame(modal, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20)

        ctk.CTkLabel(body, text="FULL NAME", font=_f(10, "bold"), text_color=SP_MUTED).pack(anchor="w")
        name_entry = ctk.CTkEntry(body, height=38, corner_radius=8, fg_color=SP_SURFACE,
                                  border_color=SP_BORDER, text_color=SP_TEXT)
        name_entry.insert(0, self._student.get("name", ""))
        name_entry.pack(fill="x", pady=(4, 12))

        for label, value in (("STUDENT ID", self.student_id),
                             ("COURSE", self._student.get("course") or "—"),
                             ("YEAR & SECTION", self._student.get("year_and_section") or "—")):
            ctk.CTkLabel(body, text=f"{label}: {value}  (read-only)", font=_f(11),
                         text_color=SP_MUTED).pack(anchor="w", pady=1)

        err = ctk.CTkLabel(body, text="", font=_f(11), text_color=SP_DANGER)
        err.pack(anchor="w", pady=(6, 0))

        def _save() -> None:
            new = name_entry.get().strip()
            if not new:
                err.configure(text="Full name cannot be empty.")
                return
            if self.db.update_student_name(self.student_id, new):
                self._student["name"] = new
                self._log_activity("Profile updated", "Updated full name")
                modal.destroy()
                self._toast("Profile updated successfully.")
                self._show("profile")
            else:
                err.configure(text="Could not update profile.")

        btns = ctk.CTkFrame(modal, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(8, 16))
        ctk.CTkButton(btns, text="Save", height=38, corner_radius=8, fg_color=SP_ACCENT,
                      hover_color=SP_ACCENT_HOVER, text_color=SP_WHITE, command=_save).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(btns, text="Cancel", width=110, height=38, corner_radius=8,
                      fg_color=SP_SURFACE, hover_color=SP_HOVER_LIGHT, text_color=SP_TEXT,
                      border_width=1, border_color=SP_BORDER, command=modal.destroy).pack(side="right")

    def _activity_log_card(self, parent, row: int) -> None:
        card = self._card(parent)
        card.grid(row=row, column=0, sticky="ew", padx=30, pady=(0, 26))
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text="System Activity Log", font=_f(17, "bold"),
                     text_color=SP_TEXT).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 2))
        ctk.CTkLabel(card, text="Your sign-ins, profile changes, report submissions, and other "
                                "actions on SECURE (view only).",
                     font=_f(12), text_color=SP_MUTED, justify="left", wraplength=720).grid(
            row=1, column=0, sticky="w", padx=20, pady=(0, 10))
        for i, evt in enumerate(reversed(self._activity_log), start=2):
            rf = ctk.CTkFrame(card, fg_color=SP_BG, corner_radius=8)
            rf.grid(row=i, column=0, sticky="ew", padx=20, pady=(0, 8))
            rf.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(rf, text=evt["action"], font=_f(13, "bold"), text_color=SP_TEXT,
                         anchor="w").grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))
            ts = evt["ts"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(evt["ts"], datetime) else str(evt["ts"])
            ctk.CTkLabel(rf, text=ts, font=_f(11), text_color=SP_MUTED, anchor="e").grid(
                row=0, column=1, sticky="e", padx=12, pady=(8, 0))
            ctk.CTkLabel(rf, text=evt["detail"], font=_f(11), text_color=SP_MUTED,
                         anchor="w").grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 8))
        ctk.CTkFrame(card, fg_color="transparent", height=4).grid(row=len(self._activity_log) + 2, column=0)

    # ------------------------------------------------------------------
    # User Settings panel
    # ------------------------------------------------------------------

    def _panel_settings(self) -> None:
        scroll = self._scroll_host("User Settings",
                                   "Customize how the SECURE portal looks and behaves for your account.")

        appearance = self._card(scroll)
        appearance.grid(row=1, column=0, sticky="ew", padx=30, pady=(14, 12))
        appearance.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(appearance, text="Appearance", font=_f(17, "bold"),
                     text_color=SP_TEXT).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 8))
        self._setting_toggle(appearance, 1, "Dark mode",
                             "Use a darker color scheme across the portal", self._toggle_dark, False)
        self._setting_toggle(appearance, 2, "Compact sidebar",
                             "Reduce sidebar spacing for more workspace", self._toggle_compact, self._compact)

        notif = self._card(scroll)
        notif.grid(row=2, column=0, sticky="ew", padx=30, pady=(0, 26))
        notif.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(notif, text="Notifications", font=_f(17, "bold"),
                     text_color=SP_TEXT).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 8))
        self._setting_toggle(notif, 1, "Email notifications",
                             "Receive email alerts for violations and system updates",
                             self._toggle_email, self._prefs.get("email_notifications", True))

    def _setting_toggle(self, parent, row, title, subtitle, command, initial) -> None:
        rowf = ctk.CTkFrame(parent, fg_color=SP_BG, corner_radius=8)
        rowf.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 10))
        rowf.grid_columnconfigure(0, weight=1)
        text = ctk.CTkFrame(rowf, fg_color="transparent")
        text.grid(row=0, column=0, sticky="w", padx=14, pady=12)
        ctk.CTkLabel(text, text=title, font=_f(13, "bold"), text_color=SP_TEXT, anchor="w").pack(anchor="w")
        ctk.CTkLabel(text, text=subtitle, font=_f(11), text_color=SP_MUTED, anchor="w").pack(anchor="w")
        var = ctk.BooleanVar(value=bool(initial))
        chk = ctk.CTkCheckBox(rowf, text="", variable=var, width=24, onvalue=True, offvalue=False,
                              fg_color=SP_ACCENT, hover_color=SP_ACCENT_HOVER,
                              command=lambda: command(var.get()))
        chk.grid(row=0, column=1, sticky="e", padx=14)

    def _toggle_dark(self, on: bool) -> None:
        ctk.set_appearance_mode("dark" if on else "light")

    def _toggle_compact(self, on: bool) -> None:
        self._compact = on
        self._sidebar.configure(width=_SIDEBAR_COMPACT if on else _SIDEBAR_FULL)
        if on:
            self._subtitle.pack_forget()
        else:
            self._subtitle.pack(anchor="w", pady=(6, 0))

    def _toggle_email(self, on: bool) -> None:
        self._prefs["email_notifications"] = on

    # ------------------------------------------------------------------
    # System Report panel
    # ------------------------------------------------------------------

    def _panel_report(self) -> None:
        scroll = self._scroll_host(
            "System Report",
            "Submit an issue report about the SECURE system to the administrator. "
            "Use this for bugs, account problems, security concerns, or other platform issues.",
        )
        card = self._card(scroll)
        card.grid(row=1, column=0, sticky="ew", padx=30, pady=(14, 26))
        card.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=20, pady=18)
        inner.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            inner, text=f"Reporting as: {self.display_name} ({self.student_id})",
            font=_f(12), text_color=SP_MUTED,
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        ctk.CTkLabel(inner, text="Report title", font=_f(13, "bold"),
                     text_color=SP_TEXT).grid(row=1, column=0, sticky="w")
        title_entry = ctk.CTkEntry(inner, height=40, corner_radius=8, fg_color=SP_SURFACE,
                                   border_color=SP_BORDER, text_color=SP_TEXT,
                                   placeholder_text="Brief summary of the issue")
        title_entry.grid(row=2, column=0, sticky="ew", pady=(4, 14))

        ctk.CTkLabel(inner, text="Category", font=_f(13, "bold"),
                     text_color=SP_TEXT).grid(row=3, column=0, sticky="w")
        category = ctk.CTkOptionMenu(inner, values=_REPORT_CATEGORIES, height=40, corner_radius=8,
                                     fg_color=SP_SURFACE, button_color=SP_BORDER,
                                     button_hover_color=SP_HOVER_LIGHT, text_color=SP_TEXT,
                                     dropdown_fg_color=SP_SURFACE, dropdown_text_color=SP_TEXT,
                                     dropdown_hover_color=SP_HOVER_LIGHT)
        category.set(_REPORT_CATEGORIES[0])
        category.grid(row=4, column=0, sticky="ew", pady=(4, 14))

        ctk.CTkLabel(inner, text="Description", font=_f(13, "bold"),
                     text_color=SP_TEXT).grid(row=5, column=0, sticky="w")
        desc = ctk.CTkTextbox(inner, height=120, corner_radius=8, fg_color=SP_SURFACE,
                              border_color=SP_BORDER, border_width=1, text_color=SP_TEXT)
        desc.grid(row=6, column=0, sticky="ew", pady=(4, 6))

        err = ctk.CTkLabel(inner, text="", font=_f(11), text_color=SP_DANGER)
        err.grid(row=7, column=0, sticky="w", pady=(0, 6))

        def _submit() -> None:
            title = title_entry.get().strip()
            description = desc.get("1.0", "end").strip()
            if not title or not description:
                err.configure(text="Please provide both a title and a description.")
                return
            ok = self.db.insert_system_report(
                self.student_id, self.display_name, category.get(), title, description)
            if ok:
                self._log_activity("System report submitted", f"Report: {title}")
                title_entry.delete(0, "end")
                desc.delete("1.0", "end")
                err.configure(text="")
                self._toast("Report submitted successfully.")
            else:
                err.configure(text="Could not submit the report. Try again.")

        ctk.CTkButton(inner, text="Send Report to Admin", height=42, corner_radius=8,
                      fg_color=SP_SIDEBAR, hover_color=SP_ACCENT, text_color=SP_WHITE,
                      font=_f(13, "bold"), command=_submit).grid(row=8, column=0, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def _logout(self) -> None:
        self._log_activity("Logged out", "Signed out of SECURE")
        self.logged_out = True
        self.destroy()

    def _on_close(self) -> None:
        self.logged_out = False
        self.destroy()
