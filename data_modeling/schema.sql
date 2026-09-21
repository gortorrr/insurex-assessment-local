PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    joined_on TEXT NOT NULL,
    inactive_on TEXT,
    initial_contract_type TEXT NOT NULL CHECK (initial_contract_type IN ('salary', 'commission')),
    CHECK (date(joined_on) IS NOT NULL),
    CHECK (inactive_on IS NULL OR date(inactive_on) IS NOT NULL),
    CHECK (inactive_on IS NULL OR inactive_on >= joined_on)
);

CREATE TABLE IF NOT EXISTS kpi_rules (
    rule_id TEXT PRIMARY KEY,
    valid_from_month TEXT NOT NULL,
    valid_to_month TEXT,
    premium_threshold_satang INTEGER NOT NULL CHECK (premium_threshold_satang >= 0),
    policy_threshold INTEGER NOT NULL CHECK (policy_threshold >= 0),
    required_streak INTEGER NOT NULL CHECK (required_streak > 0),
    comparison_operator TEXT NOT NULL CHECK (comparison_operator = '>'),
    CHECK (date(valid_from_month) IS NOT NULL AND valid_from_month = date(valid_from_month, 'start of month')),
    CHECK (valid_to_month IS NULL OR (date(valid_to_month) IS NOT NULL AND valid_to_month = date(valid_to_month, 'start of month'))),
    CHECK (valid_to_month IS NULL OR valid_to_month >= valid_from_month),
    UNIQUE (valid_from_month, valid_to_month)
);

CREATE TABLE IF NOT EXISTS monthly_agent_performance (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id),
    month_start TEXT NOT NULL,
    total_premium_satang INTEGER CHECK (total_premium_satang IS NULL OR total_premium_satang >= 0),
    new_policy_count INTEGER CHECK (new_policy_count IS NULL OR new_policy_count >= 0),
    data_status TEXT NOT NULL CHECK (data_status IN ('COMPLETE', 'PENDING')),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('synthetic', 'external')),
    month_closed INTEGER NOT NULL DEFAULT 1 CHECK (month_closed IN (0, 1)),
    input_revision INTEGER NOT NULL DEFAULT 1 CHECK (input_revision > 0),
    input_hash TEXT NOT NULL,
    CHECK (date(month_start) IS NOT NULL AND month_start = date(month_start, 'start of month')),
    CHECK (data_status = 'PENDING' OR (total_premium_satang IS NOT NULL AND new_policy_count IS NOT NULL)),
    UNIQUE (agent_id, month_start)
);

CREATE TABLE IF NOT EXISTS monthly_kpi_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    performance_id INTEGER NOT NULL REFERENCES monthly_agent_performance(performance_id),
    rule_id TEXT NOT NULL REFERENCES kpi_rules(rule_id),
    evaluation_revision INTEGER NOT NULL CHECK (evaluation_revision > 0),
    result TEXT NOT NULL CHECK (result IN ('PASS', 'FAIL', 'PENDING')),
    pass_streak INTEGER NOT NULL CHECK (pass_streak >= 0),
    fail_streak INTEGER NOT NULL CHECK (fail_streak >= 0),
    evaluated_at TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    performance_input_revision INTEGER NOT NULL,
    snapshot_total_premium_satang INTEGER CHECK (snapshot_total_premium_satang IS NULL OR snapshot_total_premium_satang >= 0),
    snapshot_new_policy_count INTEGER CHECK (snapshot_new_policy_count IS NULL OR snapshot_new_policy_count >= 0),
    snapshot_data_status TEXT NOT NULL CHECK (snapshot_data_status IN ('COMPLETE', 'PENDING')),
    snapshot_rule_premium_threshold_satang INTEGER NOT NULL,
    snapshot_rule_policy_threshold INTEGER NOT NULL,
    snapshot_rule_required_streak INTEGER NOT NULL,
    snapshot_rule_comparison_operator TEXT NOT NULL CHECK (snapshot_rule_comparison_operator = '>'),
    is_current INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    CHECK (
        (result = 'PASS' AND pass_streak > 0 AND fail_streak = 0) OR
        (result = 'FAIL' AND fail_streak > 0 AND pass_streak = 0) OR
        (result = 'PENDING' AND pass_streak = 0 AND fail_streak = 0)
    ),
    CHECK (snapshot_data_status = 'PENDING' OR (snapshot_total_premium_satang IS NOT NULL AND snapshot_new_policy_count IS NOT NULL)),
    UNIQUE (performance_id, evaluation_revision)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_current_evaluation_performance
ON monthly_kpi_evaluations(performance_id)
WHERE is_current = 1;

CREATE TABLE IF NOT EXISTS contract_history (
    contract_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id),
    from_type TEXT CHECK (from_type IS NULL OR from_type IN ('salary', 'commission')),
    to_type TEXT NOT NULL CHECK (to_type IN ('salary', 'commission')),
    effective_from TEXT NOT NULL,
    trigger_evaluation_id INTEGER REFERENCES monthly_kpi_evaluations(evaluation_id),
    reason TEXT NOT NULL,
    superseded_at TEXT,
    CHECK (date(effective_from) IS NOT NULL AND effective_from = date(effective_from, 'start of month')),
    CHECK (from_type IS NULL OR from_type <> to_type),
    UNIQUE (agent_id, effective_from, to_type)
);
