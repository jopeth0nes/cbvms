"""CBVMS application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auth.auth_manager import AuthManager
from auth.login import run_login
from database.db_manager import CBVMSDatabase
from ui.dashboard import open_dashboard


def main() -> None:
    database = CBVMSDatabase()
    database.initialize()

    auth = AuthManager(database)

    def on_login_success(username: str) -> None:
        open_dashboard(username=username)

    run_login(auth, on_login_success)


if __name__ == "__main__":
    main()
