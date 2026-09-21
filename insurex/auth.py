"""Local account authentication; owner IDs come only from verified accounts."""
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
import uuid


LOGIN_TTL_SECONDS = 30 * 60
ACCOUNT_ROLES = {"customer", "staff"}


def setup(connection):
    try:
        connection.execute("SELECT account_role FROM user_accounts LIMIT 0")
        connection.execute("SELECT token_hash FROM login_sessions LIMIT 0")
        return
    except Exception:
        connection.rollback()
    connection.execute("""CREATE TABLE IF NOT EXISTS user_accounts (
        owner_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        salt TEXT NOT NULL, password_hash TEXT NOT NULL,
        failures INTEGER NOT NULL DEFAULT 0, locked_until REAL NOT NULL DEFAULT 0,
        account_role TEXT NOT NULL DEFAULT 'customer')""")
    try:
        connection.execute("SELECT account_role FROM user_accounts LIMIT 0")
    except Exception:
        connection.rollback()
        connection.execute(
            "ALTER TABLE user_accounts ADD COLUMN account_role TEXT NOT NULL DEFAULT 'customer'"
        )
    connection.execute("""CREATE TABLE IF NOT EXISTS login_sessions (
        token_hash TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL REFERENCES user_accounts(owner_id),
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        revoked_at REAL)""")
    connection.execute("CREATE INDEX IF NOT EXISTS login_sessions_owner_idx ON login_sessions(owner_id)")
    connection.commit()


def _hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


def register(connection, username, password, *, account_role="customer"):
    username = username.strip().casefold()
    if not re.fullmatch(r"[a-z0-9_.-]{3,40}", username):
        raise ValueError("ชื่อบัญชีใช้ a-z, 0-9, จุด ขีดกลาง หรือขีดล่าง 3–40 ตัว")
    if not 10 <= len(password) <= 128:
        raise ValueError("รหัสผ่านต้องยาว 10–128 ตัวอักษร")
    if account_role not in ACCOUNT_ROLES:
        raise ValueError("account_role must be customer or staff")
    setup(connection)
    owner_id, salt = str(uuid.uuid4()), secrets.token_hex(16)
    try:
        connection.execute("INSERT INTO user_accounts(owner_id,username,salt,password_hash,account_role) VALUES (?,?,?,?,?)",
                           (owner_id, username, salt, _hash(password, salt), account_role))
        connection.commit()
    except sqlite3.IntegrityError:
        connection.rollback()
        raise ValueError("ชื่อบัญชีนี้ไม่สามารถใช้งานได้") from None
    return owner_id


def authenticate(connection, username, password):
    setup(connection)
    row = connection.execute("SELECT * FROM user_accounts WHERE username=?", (username.strip().casefold(),)).fetchone()
    salt = row['salt'] if row else '00' * 16
    candidate = _hash(password[:129], salt)
    if row is None:
        raise PermissionError("ชื่อบัญชีหรือรหัสผ่านไม่ถูกต้อง")
    if row['locked_until'] > time.time():
        raise PermissionError("บัญชีถูกพักชั่วคราว กรุณาลองใหม่ภายหลัง")
    if not hmac.compare_digest(candidate, row['password_hash']):
        failures = row['failures'] + 1
        connection.execute("UPDATE user_accounts SET failures=?,locked_until=? WHERE owner_id=?",
                           (failures, time.time() + 60 if failures >= 5 else 0, row['owner_id']))
        connection.commit()
        raise PermissionError("ชื่อบัญชีหรือรหัสผ่านไม่ถูกต้อง")
    connection.execute("UPDATE user_accounts SET failures=0,locked_until=0 WHERE owner_id=?", (row['owner_id'],))
    connection.commit()
    return {"owner_id": row['owner_id'], "username": row['username'], "account_role": row['account_role']}


def _login_token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_login_session(connection, owner_id, *, ttl_seconds=LOGIN_TTL_SECONDS):
    """Create an opaque browser-login token; only its hash is stored."""

    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    setup(connection)
    account = connection.execute(
        "SELECT owner_id FROM user_accounts WHERE owner_id=?", (owner_id,)
    ).fetchone()
    if account is None:
        raise PermissionError("account does not exist")
    token = secrets.token_urlsafe(32)
    now = time.time()
    connection.execute(
        "INSERT INTO login_sessions(token_hash,owner_id,created_at,expires_at,revoked_at) VALUES (?,?,?,?,NULL)",
        (_login_token_hash(token), owner_id, now, now + ttl_seconds),
    )
    connection.commit()
    return token


def restore_login_session(connection, token):
    """Resolve a valid, unexpired login token to its account."""

    if not token:
        return None
    setup(connection)
    now = time.time()
    row = connection.execute(
        """
        SELECT a.owner_id, a.username, a.account_role
        FROM login_sessions AS s
        JOIN user_accounts AS a ON a.owner_id=s.owner_id
        WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>?
        """,
        (_login_token_hash(token), now),
    ).fetchone()
    if row is None:
        return None
    return {"owner_id": row["owner_id"], "username": row["username"], "account_role": row["account_role"]}


def account_role(connection, owner_id):
    """Return the persisted role for a trusted account identifier."""

    setup(connection)
    row = connection.execute(
        "SELECT account_role FROM user_accounts WHERE owner_id=?", (owner_id,)
    ).fetchone()
    if row is None:
        raise PermissionError("account does not exist")
    return row["account_role"]


def revoke_login_session(connection, token):
    """Revoke one browser-login token without affecting other devices."""

    if not token:
        return
    setup(connection)
    connection.execute(
        "UPDATE login_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
        (time.time(), _login_token_hash(token)),
    )
    connection.commit()
