"""SQLite operations for CBVMS."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from database.models import ALL_TABLES

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"

# Defined here (not models.py) so the student portal's report feature is self-contained.
SYSTEM_REPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS system_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reporter_id TEXT,
    reporter_name TEXT,
    category TEXT,
    title TEXT,
    description TEXT,
    submitted_at TEXT DEFAULT (datetime('now')),
    status TEXT DEFAULT 'open'
);
"""


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


class CBVMSDatabase:
    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            root = Path(__file__).resolve().parent.parent
            db_path = root / "data" / "cbvms.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            for ddl in ALL_TABLES:
                conn.execute(ddl)
            conn.execute(SYSTEM_REPORTS_TABLE)
            # Migrations for existing databases
            cols = [row[1] for row in conn.execute("PRAGMA table_info(students)").fetchall()]
            if "gender" not in cols:
                conn.execute("ALTER TABLE students ADD COLUMN gender TEXT DEFAULT 'Unknown'")
            if "year_level" in cols and "year_and_section" not in cols:
                conn.execute("ALTER TABLE students RENAME COLUMN year_level TO year_and_section")
            conn.commit()
        self._seed_default_admin()

    def _seed_default_admin(self) -> None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE username = ?",
                (DEFAULT_ADMIN_USERNAME,),
            ).fetchone()
            if row is not None:
                return
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD)),
            )
            conn.commit()

    def verify_user(self, username: str, password: str) -> bool:
        password_hash = hash_password(password)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE username = ? AND password_hash = ?",
                (username.strip(), password_hash),
            ).fetchone()
        return row is not None

    def get_all_students(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, student_id, name, course, year_and_section, gender, encoding, photo, enrolled_at
                FROM students
                ORDER BY name COLLATE NOCASE
                """
            ).fetchall()
        return list(rows)

    def get_student(self, student_pk: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT id, student_id, name, course, year_and_section, gender, encoding, photo, enrolled_at
                FROM students WHERE id = ?
                """,
                (student_pk,),
            ).fetchone()

    def student_id_exists(self, student_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM students WHERE student_id = ?",
                (student_id.strip(),),
            ).fetchone()
        return row is not None

    def insert_student(
        self,
        student_id: str,
        name: str,
        course: str,
        year_and_section: str,
        encoding: bytes,
        photo: bytes,
        gender: str = "Unknown",
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO students (student_id, name, course, year_and_section, gender, encoding, photo)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id.strip(),
                    name.strip(),
                    course.strip(),
                    year_and_section.strip(),
                    gender.strip() or "Unknown",
                    encoding,
                    photo,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def update_student_encoding(self, student_pk: int, encoding: bytes, photo: bytes) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE students SET encoding = ?, photo = ? WHERE id = ?",
                (encoding, photo, student_pk),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_student(self, student_pk: int) -> bool:

        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM students WHERE id = ?", (student_pk,))
            conn.commit()
            return cursor.rowcount > 0

    def log_violation(
        self,
        student_id: str,
        student_name: str,
        violation_type: str,
        snapshot_jpeg: bytes | None = None,
        status: str = "unreviewed",
    ) -> int:
        """
        Persist a detected violation into the `violations` table.

        Returns the inserted row id.
        """

        safe_student_id = (student_id or "").strip() or "unknown"
        safe_student_name = (student_name or "").strip() or "Unknown"
        safe_violation_type = (violation_type or "").strip() or "unknown_violation"
        safe_status = (status or "").strip() or "unreviewed"

        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO violations (student_id, student_name, violation_type, snapshot, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (safe_student_id, safe_student_name, safe_violation_type, snapshot_jpeg, safe_status),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def delete_violation(self, violation_id: int) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_violations(self, violation_ids: list[int]) -> int:
        if not violation_ids:
            return 0
        placeholders = ",".join("?" for _ in violation_ids)
        with self.connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM violations WHERE id IN ({placeholders})", violation_ids
            )
            conn.commit()
            return cursor.rowcount

    def delete_all_violations(self, where: str = "", params: list | None = None) -> int:
        sql = f"DELETE FROM violations {where}"
        with self.connect() as conn:
            cursor = conn.execute(sql, params or [])
            conn.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Student-portal helpers
    # ------------------------------------------------------------------

    def get_student_by_student_id(self, student_id: str) -> dict | None:
        """Fetch a student row by their student_id string (not PK). Dict or None."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM students WHERE student_id = ?",
                ((student_id or "").strip(),),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_violations_for_student(self, student_id: str) -> list[dict]:
        """All violations for a student, newest first (snapshot column included)."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM violations WHERE student_id = ? ORDER BY timestamp DESC",
                ((student_id or "").strip(),),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_student_name(self, student_id: str, name: str) -> bool:
        """Update a student's display name. Returns True if a row was changed."""
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE students SET name = ? WHERE student_id = ?",
                ((name or "").strip(), (student_id or "").strip()),
            )
            conn.commit()
            return cursor.rowcount > 0

    def insert_system_report(
        self,
        reporter_id: str,
        reporter_name: str,
        category: str,
        title: str,
        description: str,
    ) -> bool:
        """Persist a student-submitted system report. Returns True on success."""
        try:
            with self.connect() as conn:
                conn.execute(SYSTEM_REPORTS_TABLE)  # defensive: ensure table exists
                conn.execute(
                    """
                    INSERT INTO system_reports
                        (reporter_id, reporter_name, category, title, description)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        (reporter_id or "").strip(),
                        (reporter_name or "").strip(),
                        (category or "").strip(),
                        (title or "").strip(),
                        (description or "").strip(),
                    ),
                )
                conn.commit()
            return True
        except Exception as exc:
            print(f"[DB] insert_system_report error: {exc}")
            return False
