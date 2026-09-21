SELECT
    p.agent_id,
    a.agent_name,
    p.month_start,
    p.total_premium_satang / 100.0 AS total_premium_thb,
    p.new_policy_count,
    e.result AS validation_result,
    e.pass_streak,
    e.fail_streak,
    r.premium_threshold_satang / 100.0 AS premium_threshold_thb,
    r.policy_threshold,
    r.comparison_operator,
    c.from_type AS contract_from,
    c.to_type AS contract_to,
    c.effective_from AS contract_effective_from
FROM monthly_agent_performance AS p
JOIN agents AS a
  ON a.agent_id = p.agent_id
JOIN monthly_kpi_evaluations AS e
  ON e.performance_id = p.performance_id
 AND e.is_current = 1
JOIN kpi_rules AS r
  ON r.rule_id = e.rule_id
LEFT JOIN contract_history AS c
  ON c.trigger_evaluation_id = e.evaluation_id
 AND c.superseded_at IS NULL
ORDER BY p.agent_id, p.month_start;
