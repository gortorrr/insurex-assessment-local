import tempfile
import unittest
from pathlib import Path

from insurex.auth import (
    create_login_session,
    register,
    restore_login_session,
    revoke_login_session,
)
from insurex.db import connect_local


class LoginSessionTests(unittest.TestCase):
    def test_login_token_restores_account_and_database_stores_only_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_local(Path(temporary) / "auth.sqlite")
            try:
                owner_id = register(connection, "alice", "long-password-a")
                token = create_login_session(connection, owner_id, ttl_seconds=1800)

                self.assertEqual(
                    restore_login_session(connection, token),
                    {"owner_id": owner_id, "username": "alice", "account_role": "customer"},
                )
                stored = connection.execute(
                    "SELECT token_hash FROM login_sessions WHERE owner_id=?", (owner_id,)
                ).fetchone()["token_hash"]
                self.assertNotEqual(stored, token)
                self.assertNotIn(token, str(connection.execute("SELECT * FROM login_sessions").fetchall()))
            finally:
                connection.close()

    def test_expired_and_revoked_tokens_cannot_restore_account(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_local(Path(temporary) / "auth.sqlite")
            try:
                owner_id = register(connection, "alice", "long-password-a")
                expired = create_login_session(connection, owner_id, ttl_seconds=1800)
                connection.execute("UPDATE login_sessions SET expires_at=0")
                connection.commit()
                self.assertIsNone(restore_login_session(connection, expired))

                revoked = create_login_session(connection, owner_id, ttl_seconds=1800)
                revoke_login_session(connection, revoked)
                self.assertIsNone(restore_login_session(connection, revoked))
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
