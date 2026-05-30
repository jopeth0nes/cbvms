"""Credential verification for CBVMS."""

from database.db_manager import CBVMSDatabase


class AuthManager:
    def __init__(self, database: CBVMSDatabase) -> None:
        self._db = database

    def verify_login(self, username: str, password: str) -> bool:
        if not username or not password:
            return False
        return self._db.verify_user(username, password)
