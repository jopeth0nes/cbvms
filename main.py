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

    # Loop: login -> dashboard -> (logout) -> login -> ...
    # Exits only when the login window is closed or the dashboard is closed
    # directly (not via Logout).
    while True:
        username = run_login(auth)
        if not username:
            break  # login window closed
        logged_out = open_dashboard(username=username)
        if not logged_out:
            break  # dashboard closed/exited directly


if __name__ == "__main__":
    main()
