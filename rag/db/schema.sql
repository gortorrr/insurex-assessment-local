PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS user_accounts (
    owner_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    failures INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0,
    account_role TEXT NOT NULL DEFAULT 'customer' CHECK (account_role IN ('customer', 'staff'))
);

CREATE TABLE IF NOT EXISTS login_sessions (
    token_hash TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES user_accounts(owner_id),
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    revoked_at REAL
);

CREATE INDEX IF NOT EXISTS login_sessions_owner_idx ON login_sessions(owner_id);

CREATE TABLE IF NOT EXISTS leads (
    lead_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    product_id TEXT NOT NULL,
    name TEXT NOT NULL,
    occupation TEXT NOT NULL,
    income_thb NUMERIC NOT NULL CHECK (income_thb >= 0),
    income_period TEXT NOT NULL CHECK (income_period = 'monthly_thb'),
    phone_normalized TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    payload_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    profile_verified_at TEXT,
    refresh_requested_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    thread_id TEXT NOT NULL UNIQUE,
    owner_token_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'closed')),
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lead_drafts (
    owner_id TEXT PRIMARY KEY,
    last_session_id TEXT NOT NULL REFERENCES sessions(session_id),
    product_id TEXT,
    name TEXT,
    occupation TEXT,
    income_value TEXT,
    income_min TEXT,
    income_max TEXT,
    income_currency TEXT,
    income_period TEXT,
    phone_raw TEXT,
    phone_normalized TEXT,
    status TEXT NOT NULL CHECK (status IN ('unconfirmed', 'confirmed', 'cancelled')),
    draft_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lead_request_log (
    request_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    lead_id INTEGER NOT NULL REFERENCES leads(lead_id),
    payload_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    request_id TEXT,
    purpose TEXT NOT NULL DEFAULT 'private',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    sender_type TEXT NOT NULL DEFAULT 'ai' CHECK (sender_type IN ('customer', 'ai', 'staff', 'system')),
    sender_account_id TEXT,
    sender_label TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS message_request_unique
ON messages(session_id, request_id)
WHERE request_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS handoff_cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'resolved')),
    trigger_reason TEXT NOT NULL,
    trigger_message_id INTEGER REFERENCES messages(message_id),
    assigned_staff_id TEXT REFERENCES user_accounts(owner_id),
    created_at TEXT NOT NULL,
    accepted_at TEXT,
    resolved_at TEXT,
    resolution_note TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS one_open_handoff_per_session
ON handoff_cases(session_id)
WHERE status IN ('pending', 'active');
