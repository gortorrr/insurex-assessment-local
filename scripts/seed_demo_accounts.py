"""Create the two synthetic accounts used by the local multi-user demo.

The passwords are intentionally local-demo credentials.  Only their scrypt
hashes are stored in SQLite; this script prints the credentials so a reviewer
can sign in without inspecting database internals.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from insurex.auth import register, setup
from insurex.db import connect_assistant_local


DEFAULTS = (
    ("demo-agent-a", "Demo-Agent-A-2026!", "customer"),
    ("demo-agent-b", "Demo-Agent-B-2026!", "customer"),
    ("demo-staff-a", "Demo-Staff-A-2026!", "staff"),
    ("demo-staff-b", "Demo-Staff-B-2026!", "staff"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic local demo accounts")
    parser.add_argument("--database", default=None, help="SQLite path; defaults to ASSISTANT_DB_PATH")
    args = parser.parse_args()
    database = Path(args.database or os.getenv("ASSISTANT_DB_PATH", os.getenv("BUSINESS_DB_PATH", "rag/db/assistant.sqlite")))
    connection = connect_assistant_local(database)
    try:
        setup(connection)
        for username, password, role in DEFAULTS:
            row = connection.execute("SELECT owner_id FROM user_accounts WHERE username=?", (username,)).fetchone()
            if row is None:
                register(connection, username, password, account_role=role)
                status = "created"
            else:
                connection.execute("UPDATE user_accounts SET account_role=? WHERE username=?", (role, username))
                connection.commit()
                status = "already-exists"
            print(f"{status}: role={role} username={username} password={password}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
